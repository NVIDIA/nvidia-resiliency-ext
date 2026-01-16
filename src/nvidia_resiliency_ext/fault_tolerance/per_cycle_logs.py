#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom LogsSpecs implementation for consolidated per-cycle logging.

This module provides PipeBasedLogsSpecs:
   - Uses pipes (like srun --output) to prevent buffer loss
   - Parent ft_launcher reads from pipes and writes to file
   - Line-buffered pipes ensure data isn't lost on worker termination (SIGTERM)
   - Rank prefixes added in parent (works for C++ logs too)
   - Solves the stack trace loss problem

Architecture:
   Worker → pipe (line-buffered by kernel) → parent reader thread → file

Why not use PyTorch's built-in redirect?
--------------------------------------
PyTorch's torch.distributed.elastic.multiprocessing.redirects.redirect() opens files with
mode="w+b" which truncates the file. This is fine for per-rank files, but causes data loss
when multiple ranks write to the same file.
"""

import errno
import logging
import os
import select
import subprocess
import tempfile
import threading
from typing import Any, Optional

from torch.distributed.elastic.multiprocessing import LogsDest, LogsSpecs, Std
from torch.distributed.elastic.multiprocessing.subprocess_handler import SubprocessHandler

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

# Special marker string to signal pipe-based logging
# We can't use subprocess.PIPE directly because PyTorch expects strings
_PIPE_MARKER = "__SUBPROCESS_PIPE__"

# Module-level flag to track if we've patched get_subprocess_handler
_SUBPROCESS_HANDLER_PATCHED = False


def _patch_subprocess_handler_once():
    """
    Monkey-patch get_subprocess_handler to use PipeSubprocessHandler.

    We need to patch it in the api module where it's actually used, not just in
    the handlers module where it's defined, because api.py has its own import reference.

    This is a module-level operation that happens once. After patching,
    all subsequent subprocess handler creation will use PipeSubprocessHandler.

    This is safe because:
    1. ft_launcher only uses one LogsSpecs type at a time
    2. Once using PipeBasedLogsSpecs, it stays that way for the process lifetime
    3. The patch should persist - no need to unpatch
    """
    global _SUBPROCESS_HANDLER_PATCHED

    if _SUBPROCESS_HANDLER_PATCHED:
        return  # Already patched

    # Import the api module where get_subprocess_handler is actually called
    from torch.distributed.elastic.multiprocessing import api

    # Create patched version that returns our custom handler
    def patched_get_subprocess_handler(**kwargs):
        return PipeSubprocessHandler(**kwargs)

    # Patch it in the api module where SubprocessContext uses it
    api.get_subprocess_handler = patched_get_subprocess_handler
    _SUBPROCESS_HANDLER_PATCHED = True


class PipeSubprocessHandler(SubprocessHandler):
    """
    Custom SubprocessHandler that supports pipe-based logging via a special marker string.

    Extends PyTorch's SubprocessHandler to detect our _PIPE_MARKER string and convert it
    to subprocess.PIPE. We can't pass subprocess.PIPE directly because PyTorch's LogsDest
    type signature expects strings (file paths), not integers.

    Merges stderr into stdout pipe (like default srun --output= behavior where both
    streams go to the same file).
    """

    def __init__(
        self,
        entrypoint: str,
        args: tuple,
        env: dict[str, str],
        stdout: Optional[str],  # Can be path, _PIPE_MARKER, or None
        stderr: Optional[str],  # Can be path, _PIPE_MARKER, or None
        local_rank_id: int,
    ):
        # Override parent to handle our pipe marker
        # Don't call super().__init__() because it would try to open() the marker

        # Handle stdout
        if stdout == _PIPE_MARKER:
            self._stdout = subprocess.PIPE
        elif stdout:
            self._stdout = open(stdout, "w")
        else:
            self._stdout = None

        # Handle stderr - merge into stdout if both are pipes (like srun default)
        if stderr == _PIPE_MARKER and stdout == _PIPE_MARKER:
            # Merge stderr into stdout pipe (like srun --output= default behavior)
            self._stderr = subprocess.STDOUT
        elif stderr == _PIPE_MARKER:
            self._stderr = subprocess.PIPE
        elif stderr:
            self._stderr = open(stderr, "w")
        else:
            self._stderr = None

        # Inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        self.local_rank_id = local_rank_id
        self.proc: subprocess.Popen = self._popen(args_str, env_vars)

    def close(self, death_sig: Optional[Any] = None) -> None:
        """Override close to handle subprocess.PIPE properly."""
        # Import here to avoid circular dependency
        import signal

        if not death_sig:
            death_sig = signal.SIGTERM

        # Send signal to process group
        try:
            os.killpg(self.proc.pid, death_sig)
        except (OSError, AttributeError):
            # Fallback if killpg fails or on Windows
            try:
                self.proc.send_signal(death_sig)
            except OSError:
                pass

        # Close file handles for non-pipe cases (when we opened actual files)
        if self._stdout and self._stdout not in (subprocess.PIPE, subprocess.STDOUT):
            try:
                self._stdout.close()
            except Exception:
                pass

        if self._stderr and self._stderr not in (subprocess.PIPE, subprocess.STDOUT):
            try:
                self._stderr.close()
            except Exception:
                pass


class MultiplexingReaderThread(threading.Thread):
    """
    Reader thread that multiplexes multiple worker pipes into a single log file.

    Uses select.poll() to efficiently monitor multiple pipes without busy-waiting.
    Automatically terminates when all worker pipes close (workers exit).

    This mimics how 'srun --output' works: one reader thread per node aggregates
    all local worker output and writes to a consolidated log file.

    Line Buffering Strategy:
        To prevent line merging when multiple ranks write simultaneously, we buffer
        incomplete lines (those not ending with '\\n') until the next read completes them.

        Tradeoff: If a worker crashes mid-line, the incomplete line is still written
        when the pipe closes (with an added '\\n'), ensuring no data loss for debugging
        crash scenarios.
    """

    def __init__(
        self,
        pipes: dict[int, int],
        log_file_path: str,
        world_size: Optional[int] = None,
        local_to_global_rank: Optional[dict[int, int]] = None,
    ):
        """
        Initialize reader thread.

        Args:
            pipes: {local_rank: pipe_fd} mapping for all local workers
                   Each pipe carries both stdout and stderr (merged via subprocess.STDOUT)
            log_file_path: Path to consolidated log file to write to
            world_size: Total number of ranks across all nodes (for rank padding like srun -l)
            local_to_global_rank: Optional mapping of local_rank -> global_rank for prefixes.
                                 If provided, uses global ranks in prefixes (like srun -l).
                                 If None, uses local ranks.
        """
        super().__init__(daemon=True)
        self.pipes = pipes.copy()
        self.log_file_path = log_file_path
        self.world_size = world_size
        self.local_to_global_rank = local_to_global_rank or {}
        self.poller = select.poll()
        self.fd_to_local_rank = {}  # Maps file descriptor to local rank
        self.logger = logging.getLogger(LogConfig.name)

        # Calculate rank prefix padding width (like srun -l)
        # Determine padding width based on world_size (max rank is world_size - 1)
        self._rank_width = (
            len(str(world_size - 1)) if (world_size is not None and world_size > 1) else 0
        )

        # Buffer for incomplete lines (lines without trailing '\n')
        # Maps rank -> accumulated incomplete line text
        # This prevents line merging when os.read() returns partial data
        self._line_buffers = {}

        # Register all pipes with poller
        for local_rank, pipe_fd in pipes.items():
            self.poller.register(pipe_fd, select.POLLIN | select.POLLHUP)
            self.fd_to_local_rank[pipe_fd] = local_rank

    def _safe_write(self, log_file, text: str, rank: int) -> None:
        """
        Safely write text to log file, handling null bytes if present.

        Args:
            log_file: Open file object to write to
            text: Text to write (may contain null bytes)
            rank: Rank number for logging purposes
        """
        try:
            log_file.write(text)
        except OSError as e:
            # errno 22 = Invalid argument, typically caused by null bytes in text mode
            # This can happen during worker crashes when binary data ends up in stdout/stderr
            if e.errno == errno.EINVAL:
                # Replace null bytes with visible marker and retry
                text_cleaned = text.replace('\x00', '<NUL>')
                try:
                    log_file.write(text_cleaned)
                except OSError:
                    # If it still fails, just skip this write to avoid crashing the thread
                    self.logger.debug(
                        f"Failed to write output from rank {rank} even after cleaning"
                    )
            else:
                # Re-raise other OSErrors
                raise

    def _write_lines(self, log_file, local_rank: int, lines: list[str]) -> None:
        """
        Write complete lines to log file with rank prefix.

        Args:
            log_file: Open file object to write to
            local_rank: Local rank number (will be mapped to global rank if mapping provided)
            lines: List of lines to write (should all end with '\\n')
        """
        # Use global rank for prefix if mapping is available, otherwise use local rank
        rank = self.local_to_global_rank.get(local_rank, local_rank)
        prefix = f'{rank:>{self._rank_width}}: ' if self._rank_width > 0 else f'{rank}: '
        for line in lines:
            self._safe_write(log_file, prefix + line, rank)

    def _flush_incomplete_line(self, log_file, local_rank: int) -> None:
        """
        Flush any buffered incomplete line for a rank when its pipe closes.

        This ensures we don't lose partial output when a worker crashes.
        We add a newline to prevent merging with subsequent output.

        Args:
            log_file: Open file object to write to
            local_rank: Local rank whose incomplete line to flush
        """
        if local_rank in self._line_buffers:
            incomplete = self._line_buffers[local_rank]
            if incomplete:  # Only write if non-empty
                # Use global rank for prefix if mapping is available, otherwise use local rank
                rank = self.local_to_global_rank.get(local_rank, local_rank)
                prefix = f'{rank:>{self._rank_width}}: ' if self._rank_width > 0 else f'{rank}: '
                # Add newline to prevent merging with next rank's output
                # This changes the original output slightly, but prevents corruption
                self._safe_write(log_file, prefix + incomplete + '\n', rank)
                self.logger.debug(
                    f"Flushed incomplete line for rank {rank} on pipe close: {incomplete[:50]}..."
                )
            del self._line_buffers[local_rank]

    def run(self):
        """
        Main loop: poll pipes, read data, add rank prefixes, write to file.

        Exits when all pipes close (all workers terminated).

        Implementation notes:
            - Buffers incomplete lines (no trailing '\\n') to prevent line merging
            - Flushes incomplete lines when pipe closes (worker exits)
            - Handles partial reads from os.read() correctly
        """
        try:
            with open(self.log_file_path, 'a', buffering=1) as log_file:  # Line buffered
                while self.pipes:
                    try:
                        # Wait for any pipe to have data (timeout 1s for responsiveness)
                        events = self.poller.poll(1000)  # timeout is positional arg
                    except OSError as e:
                        # Handle EINTR (interrupted system call) - can happen if process receives signal
                        if e.errno == errno.EINTR:
                            continue
                        else:
                            # Other OSError - log and re-raise
                            self.logger.error(f"poll() failed with OSError: {e}")
                            raise

                    for fd, event in events:
                        local_rank = self.fd_to_local_rank.get(fd)
                        if local_rank is None:
                            continue

                        if event & (select.POLLIN | select.POLLHUP):
                            try:
                                # Read available data
                                data = os.read(fd, 65536)
                            except OSError as e:
                                # EINTR can also happen on read()
                                if e.errno == errno.EINTR:
                                    continue
                                # Use global rank for logging if available
                                rank_for_log = self.local_to_global_rank.get(local_rank, local_rank)
                                self.logger.warning(
                                    f"Error reading from rank {rank_for_log} pipe: {e}"
                                )
                                data = None

                            if not data:
                                # Pipe closed - worker exited
                                # First, flush any buffered incomplete line to avoid data loss
                                self._flush_incomplete_line(log_file, local_rank)

                                # Then clean up pipe resources
                                self.poller.unregister(fd)
                                try:
                                    os.close(fd)
                                except OSError:
                                    pass
                                del self.pipes[local_rank]
                                del self.fd_to_local_rank[fd]
                            else:
                                # Process data from this rank's pipe
                                # Decode with error replacement to handle any encoding issues
                                text = data.decode('utf-8', errors='replace')

                                # Split into lines, keeping line endings
                                # Important: splitlines(keepends=True) on "abc\ndef" returns ['abc\n', 'def']
                                # The last element 'def' has NO newline - it's an incomplete line!
                                lines = text.splitlines(keepends=True)

                                if not lines:
                                    # Empty read (shouldn't happen, but be defensive)
                                    continue

                                # First, complete any previously buffered incomplete line
                                if local_rank in self._line_buffers:
                                    # Prepend buffered text to first line of this read
                                    lines[0] = self._line_buffers[local_rank] + lines[0]
                                    del self._line_buffers[local_rank]

                                # Check if last line is incomplete and buffer it
                                if not lines[-1].endswith('\n'):
                                    # Last line incomplete - buffer it for next read
                                    self._line_buffers[local_rank] = lines.pop()

                                # Write any remaining complete lines
                                if lines:
                                    self._write_lines(log_file, local_rank, lines)

                                # Flush to ensure data is written immediately
                                log_file.flush()

        except Exception as e:
            # Catch-all for unexpected errors
            # This prevents the thread from dying silently
            self.logger.error(f"MultiplexingReaderThread fatal error: {e}", exc_info=True)


class PipeBasedLogsSpecs(LogsSpecs):
    """
    LogsSpecs using pipes + reader thread (like srun --output) for consolidated per-cycle logging.

    This implementation solves buffer loss problems by using a pipe-based architecture where
    workers write to pipes and a parent thread reads from all pipes and writes to the consolidated
    log file.

    Architecture:
        1. Each worker's stdout/stderr is redirected to a pipe (via subprocess.PIPE)
        2. Parent ft_launcher runs a MultiplexingReaderThread that reads from all pipes
        3. Reader thread adds rank prefixes and writes to consolidated log file
        4. When workers exit, pipes close and reader thread terminates automatically

    Benefits:
        - Line-buffered pipes prevent log loss on worker termination (SIGTERM)
        - Rank prefixes work for C++ logs (added in parent, not child)
        - Reduced Lustre lock contention (one writer per node vs N workers)
        - No need for sitecustomize.py injection
        - Simpler worker startup (no PYTHONPATH manipulation)

    How it works (mimics srun --output):
        Worker → pipe (line-buffered by kernel) → parent reader thread → file

        Even if worker dies (SIGTERM), data in pipe is not lost because parent
        continues reading until pipe is fully drained.

    Implementation note:
        Uses monkey-patching to inject PipeSubprocessHandler into PyTorch's
        get_subprocess_handler() function so that subprocess.PIPE markers
        are handled correctly.

    Args:
        base_log_file: Base path for log files (e.g., "/lustre/logs/job.log").
                      Creates: job_cycle0.log, job_cycle1.log, etc.
    """

    logger = logging.getLogger(LogConfig.name)

    def __init__(self, base_log_file: str) -> None:
        if not base_log_file:
            raise ValueError("base_log_file is required for PipeBasedLogsSpecs")

        # Convert to absolute path for multi-node safety
        self._base_log_file = os.path.abspath(base_log_file)

        # Extract directory and ensure it exists
        log_dir = os.path.dirname(self._base_log_file) or "."
        os.makedirs(log_dir, exist_ok=True)

        super().__init__(
            log_dir=log_dir,
            redirects=Std.ALL,
            tee=Std.NONE,
            local_ranks_filter=None,
        )

        self._error_dir = None
        self._reader_thread = None
        self._current_cycle_log = None
        self._world_size = None  # Will be set in reify()

    @property
    def root_log_dir(self) -> str:
        return str(self._root_log_dir)

    def reify(
        self,
        envs: dict[int, dict[str, str]],
    ) -> LogsDest:
        """
        Returns LogsDest with _PIPE_MARKER strings for stdout/stderr.

        We use a special marker string instead of subprocess.PIPE because PyTorch's
        LogsDest type signature expects strings (file paths). Our custom
        PipeSubprocessHandler detects this marker and converts to subprocess.PIPE.

        The actual pipe creation and reading happens in start_reader() after workers spawn.

        Also installs monkey-patch to use PipeSubprocessHandler (once, module-level).

        Creates:
        - /tmp/torchelastic_errors_<run_id>/cycle_<N>/<rank>/error.json (per rank)
        """
        # Install monkey-patch before workers are created (idempotent, module-level)
        _patch_subprocess_handler_once()

        nprocs = len(envs)

        # For standby nodes (hot spares) with 0 workers, return early
        # _current_cycle_log will remain None until the node becomes active and reify() is called again
        if nprocs == 0:
            self.logger.info(
                "No workers to spawn (likely a standby/hot spare node). "
                "Log paths will be set up when node becomes active."
            )
            return LogsDest({}, {}, {}, {}, {})

        # Get restart count from environment
        global_env = envs[0]
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        # Get world_size for rank prefix padding (like srun -l)
        world_size_str = global_env.get("WORLD_SIZE")
        world_size = int(world_size_str) if world_size_str else None

        # Build local_rank -> global_rank mapping for log prefixing
        # The 'envs' dict is keyed by local_rank and contains RANK (global_rank) env var
        local_to_global_rank = {
            local_rank: int(rank_env["RANK"]) for local_rank, rank_env in envs.items()
        }

        # Store info needed for start_reader()
        self._restart_count = restart_count
        self._nprocs = nprocs
        self._world_size = world_size
        self._local_to_global_rank = local_to_global_rank

        # Calculate cycle log file path (will be created in start_reader())
        base_without_ext = os.path.splitext(self._base_log_file)[0]
        ext = os.path.splitext(self._base_log_file)[1] or ".log"
        self._current_cycle_log = f"{base_without_ext}_cycle{restart_count}{ext}"

        # Return pipe marker strings to signal pipe-based redirection
        # We use a special string marker instead of subprocess.PIPE because
        # PyTorch's LogsDest expects strings, not integers
        # Our PipeSubprocessHandler will detect this marker and convert to subprocess.PIPE
        stdouts = {rank: _PIPE_MARKER for rank in range(nprocs)}
        stderrs = {rank: _PIPE_MARKER for rank in range(nprocs)}

        # Create error directory
        if not self._error_dir:
            self._error_dir = tempfile.mkdtemp(prefix=f"torchelastic_errors_{run_id}_")

        error_base_dir = os.path.join(self._error_dir, f"cycle_{restart_count}")
        os.makedirs(error_base_dir, exist_ok=True)

        # No tee
        tee_stdouts: dict[int, str] = {}
        tee_stderrs: dict[int, str] = {}

        # Create error files (per rank, required by PyTorch)
        error_files = {}
        for local_rank in range(nprocs):
            error_dir = os.path.join(error_base_dir, str(local_rank))
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, "error.json")
            error_files[local_rank] = error_file
            envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

        return LogsDest(stdouts, stderrs, tee_stdouts, tee_stderrs, error_files)

    def start_reader(self, subprocess_handlers: dict) -> MultiplexingReaderThread:
        """
        Start the multiplexing reader thread for current cycle's workers.

        Called from launcher.py after workers are spawned. Creates one thread that
        reads from all local worker pipes and writes to the consolidated log file.

        Args:
            subprocess_handlers: {rank: SubprocessHandler} for all local workers

        Returns:
            The reader thread (already started)
        """
        # Check if previous reader thread is still alive
        #
        # WHY THIS HAPPENS:
        # The reader thread exits when all pipes close. However, pipes don't close immediately
        # when workers die because:
        # 1. PyTorch's SubprocessContext._close() only calls handler.close() if proc.poll() is None
        # 2. By the time _close() runs, workers have often already died (proc.poll() != None)
        # 3. So handler.close() is never called, and Popen.stdout pipes remain open
        # 4. Pipes only close when Python's GC collects the old Popen objects
        #
        # WHEN THIS IS OK:
        # - With hour-long restart intervals: plenty of time for GC to run, FD reuse unlikely
        # - Old thread will exit within minutes when GC closes pipes
        # - New thread uses different FDs, so no conflict
        #
        # WHEN THIS COULD BE PROBLEMATIC:
        # - Fast restart scenarios (seconds apart): old FDs might be reused before cleanup
        # - Could cause log corruption if same FD monitored by multiple threads
        #
        # WHAT HAPPENS:
        # - We proceed immediately without waiting (old thread will exit via GC)
        # - Old thread monitors old FDs, new thread monitors new FDs
        # - No conflict as long as FDs are different (typical with hour-long intervals)
        if self._reader_thread and self._reader_thread.is_alive():
            self.logger.debug(
                "Previous reader thread still alive (expected, will exit when GC closes old pipes). "
                "Proceeding with new cycle."
            )

        # Check if log file path is set up
        if self._current_cycle_log is None:
            self.logger.debug(
                "start_reader() called but _current_cycle_log is None "
                "(normal for hot spare nodes - will be set up when node becomes active)"
            )
            return None

        if not os.path.exists(self._current_cycle_log):
            open(self._current_cycle_log, 'a').close()
            self.logger.info("Created consolidated log file: %s", self._current_cycle_log)

        # Collect pipes from all local workers
        # stdout carries both stdout and stderr (merged via subprocess.STDOUT)
        # This matches srun --output= default behavior
        pipes = {}
        for rank, handler in subprocess_handlers.items():
            # Only collect stdout pipe since stderr is redirected to it
            if handler.proc.stdout:
                pipes[rank] = handler.proc.stdout.fileno()

        if not pipes:
            # No pipes available - this is expected for hot spare nodes with no workers
            # Only log at debug level since it's normal behavior
            self.logger.debug(
                "No pipes available from subprocess handlers (likely hot spare node with no workers)"
            )
            return None

        # Create and start multiplexing reader thread
        # Pass local_to_global_rank mapping so reader can use global ranks for prefixes
        self._reader_thread = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=self._current_cycle_log,
            world_size=self._world_size,
            local_to_global_rank=self._local_to_global_rank,
        )
        self._reader_thread.start()

        self.logger.info(
            f"Started multiplexing reader thread for {len(pipes)} workers, "
            f"writing to {self._current_cycle_log}"
        )

        return self._reader_thread

    def __repr__(self) -> str:
        return f"PipeBasedLogsSpecs(base_log_file={self._base_log_file})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipeBasedLogsSpecs):
            return False
        return self._base_log_file == other._base_log_file
