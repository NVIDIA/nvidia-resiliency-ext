# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nvidia_resiliency_ext.fault_tolerance.per_cycle_logs module."""

import errno
import os
import tempfile
import threading
import time
from unittest.mock import patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import (
    MultiplexingReaderThread,
    PipeSubprocessHandler,
    _should_filter_line,
)


class TestLogFiltering:
    """Tests for log line filtering functionality."""

    def test_filter_character_devices_header(self):
        """Test filtering of 'Character devices:' header."""
        assert _should_filter_line("Character devices:\n") is True
        assert _should_filter_line("  Character devices:\n") is True

    def test_filter_block_devices_header(self):
        """Test filtering of 'Block devices:' header."""
        assert _should_filter_line("Block devices:\n") is True
        assert _should_filter_line("  Block devices:\n") is True

    def test_filter_device_entries(self):
        """Test filtering of device number entries."""
        # Examples from nvidia driver dumps
        assert _should_filter_line("252 device-mapper\n") is True
        assert _should_filter_line("253 virtblk\n") is True
        assert _should_filter_line("195 nvidia\n") is True
        assert _should_filter_line("195 nvidia-modeset\n") is True
        assert _should_filter_line("195 nvidiactl\n") is True
        assert _should_filter_line("  1 mem\n") is True
        assert _should_filter_line("  4 /dev/vc/0\n") is True
        assert _should_filter_line("497 gdrdrv\n") is True
        assert _should_filter_line("500 nvidia-caps\n") is True

    def test_keep_useful_log_lines(self):
        """Test that useful log lines are NOT filtered."""
        # Regular log messages should not be filtered
        assert (
            _should_filter_line("Rank 1117/12296: Raising exception to signal error...\n") is False
        )
        assert (
            _should_filter_line("Rank 1117/12296: ========== EXCEPTION CAUGHT ==========\n")
            is False
        )
        assert (
            _should_filter_line("RuntimeError: Simulated collective failure (NCCL timeout)\n")
            is False
        )
        assert _should_filter_line("Worker exiting with failure code 123\n") is False

        # Lines with numbers that aren't device entries
        assert _should_filter_line("Error code: 123 in function test\n") is False
        assert _should_filter_line("Processing 195 items\n") is False

        # Empty lines should be kept
        assert _should_filter_line("\n") is False
        assert _should_filter_line("") is False

    def test_filter_without_newline(self):
        """Test filtering works with or without trailing newline."""
        assert _should_filter_line("252 device-mapper") is True
        assert _should_filter_line("Character devices:") is True
        assert _should_filter_line("Rank 1117: Error") is False


class TestMultiplexingReaderThread:
    """Tests for MultiplexingReaderThread class."""

    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file."""
        fd, path = tempfile.mkstemp(suffix='.log')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def _create_pipe_pair(self):
        """Create a pipe and return (read_fd, write_fd)."""
        return os.pipe()

    def _write_to_pipe(self, write_fd, data: bytes):
        """Write data to pipe."""
        os.write(write_fd, data)

    def _close_pipe(self, fd):
        """Close a pipe file descriptor."""
        try:
            os.close(fd)
        except OSError:
            pass

    def _read_log_file(self, path: str) -> str:
        """Read the contents of the log file."""
        with open(path, 'r') as f:
            return f.read()

    def test_single_rank_basic_output(self, temp_log_file):
        """Test basic output from a single rank."""
        # Create pipe
        read_fd, write_fd = self._create_pipe_pair()

        # Create reader thread
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write some data
        self._write_to_pipe(write_fd, b'Hello from rank 0\n')
        self._write_to_pipe(write_fd, b'Second line\n')

        # Close pipe to signal end
        self._close_pipe(write_fd)

        # Wait for thread to finish
        reader.shutdown()
        reader.join(timeout=2.0)

        # Check output
        content = self._read_log_file(temp_log_file)
        assert '0: Hello from rank 0\n' in content
        assert '0: Second line\n' in content

    def test_nvidia_driver_dump_filtering(self, temp_log_file):
        """Test that nvidia driver dumps are filtered out from logs."""
        # Create pipe
        read_fd, write_fd = self._create_pipe_pair()

        # Create reader thread
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write useful log messages mixed with nvidia driver dumps
        self._write_to_pipe(write_fd, b'Rank 0/100: Starting worker\n')
        self._write_to_pipe(write_fd, b'Rank 0/100: Worker exiting with failure code 123\n')
        self._write_to_pipe(write_fd, b'\n')
        # Nvidia driver dump that should be filtered
        self._write_to_pipe(write_fd, b'252 device-mapper\n')
        self._write_to_pipe(write_fd, b'253 virtblk\n')
        self._write_to_pipe(write_fd, b'254 mdp\n')
        self._write_to_pipe(write_fd, b'259 blkext\n')
        self._write_to_pipe(write_fd, b'\n')
        self._write_to_pipe(write_fd, b'Character devices:\n')
        self._write_to_pipe(write_fd, b'  1 mem\n')
        self._write_to_pipe(write_fd, b'  4 /dev/vc/0\n')
        self._write_to_pipe(write_fd, b'195 nvidia\n')
        self._write_to_pipe(write_fd, b'195 nvidia-modeset\n')
        self._write_to_pipe(write_fd, b'195 nvidiactl\n')
        self._write_to_pipe(write_fd, b'497 gdrdrv\n')
        self._write_to_pipe(write_fd, b'\n')
        self._write_to_pipe(write_fd, b'Block devices:\n')
        self._write_to_pipe(write_fd, b'  7 loop\n')
        self._write_to_pipe(write_fd, b'  8 sd\n')
        # More useful logs after the dump
        self._write_to_pipe(write_fd, b'Rank 0/100: Cleanup complete\n')

        # Close pipe to signal end
        self._close_pipe(write_fd)

        # Wait for thread to finish
        reader.shutdown()
        reader.join(timeout=2.0)

        # Check output
        content = self._read_log_file(temp_log_file)

        # Useful messages should be present
        assert '0: Rank 0/100: Starting worker\n' in content
        assert '0: Rank 0/100: Worker exiting with failure code 123\n' in content
        assert '0: Rank 0/100: Cleanup complete\n' in content

        # Nvidia driver dump lines should be filtered out
        assert 'device-mapper' not in content
        assert 'virtblk' not in content
        assert 'Character devices:' not in content
        assert 'Block devices:' not in content
        assert 'nvidia-modeset' not in content
        assert 'nvidiactl' not in content
        assert 'gdrdrv' not in content

        # Empty lines should still be present (they're intentional formatting)
        assert '0: \n' in content

    def test_multiple_ranks(self, temp_log_file):
        """Test output from multiple ranks."""
        # Create pipes for 3 ranks
        pipes = {}
        write_fds = {}
        for rank in range(3):
            read_fd, write_fd = self._create_pipe_pair()
            pipes[rank] = read_fd
            write_fds[rank] = write_fd

        # Create reader thread with world_size for proper padding
        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=3,
        )
        reader.start()

        # Write data from each rank
        self._write_to_pipe(write_fds[0], b'Rank 0 output\n')
        self._write_to_pipe(write_fds[1], b'Rank 1 output\n')
        self._write_to_pipe(write_fds[2], b'Rank 2 output\n')

        # Close all pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        # Wait for thread to finish
        reader.shutdown()
        reader.join(timeout=2.0)

        # Check output
        content = self._read_log_file(temp_log_file)
        assert '0: Rank 0 output\n' in content
        assert '1: Rank 1 output\n' in content
        assert '2: Rank 2 output\n' in content

    def test_global_rank_mapping(self, temp_log_file):
        """Test local to global rank mapping."""
        # Create pipes for 2 local ranks
        pipes = {}
        write_fds = {}
        for local_rank in [0, 1]:
            read_fd, write_fd = self._create_pipe_pair()
            pipes[local_rank] = read_fd
            write_fds[local_rank] = write_fd

        # Map local ranks to global ranks: local 0 -> global 4, local 1 -> global 5
        local_to_global = {0: 4, 1: 5}

        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=100,  # Large world size for padding test
            local_to_global_rank=local_to_global,
        )
        reader.start()

        # Write data
        self._write_to_pipe(write_fds[0], b'Local 0\n')
        self._write_to_pipe(write_fds[1], b'Local 1\n')

        # Close pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # Check output uses global ranks with proper padding (world_size=100, so 2 digits)
        content = self._read_log_file(temp_log_file)
        assert ' 4: Local 0\n' in content
        assert ' 5: Local 1\n' in content

    def test_null_byte_handling(self, temp_log_file):
        """Test that NULL bytes are replaced with <NUL> marker."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write data with NULL bytes
        self._write_to_pipe(write_fd, b'Before NULL\x00After NULL\n')
        self._write_to_pipe(write_fd, b'Multiple \x00 nulls \x00 here\n')

        self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # Check NULL bytes are replaced
        content = self._read_log_file(temp_log_file)
        assert '<NUL>' in content
        assert '0: Before NULL<NUL>After NULL\n' in content
        assert '0: Multiple <NUL> nulls <NUL> here\n' in content

        # Verify file is still detected as text, not binary
        # Use Python's built-in text detection instead of 'file' command for portability
        try:
            with open(temp_log_file, 'r', encoding='utf-8') as f:
                f.read()
            is_text = True
        except (UnicodeDecodeError, ValueError):
            is_text = False
        assert is_text, f"Log file {temp_log_file} is not valid UTF-8 text"

    def test_invalid_utf8_handling(self, temp_log_file):
        """Test handling of invalid UTF-8 sequences."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write invalid UTF-8 sequences
        # 0xFF and 0xFE are invalid UTF-8 start bytes
        self._write_to_pipe(write_fd, b'Invalid: \xff\xfe bytes here\n')
        # Incomplete UTF-8 sequence (start of 3-byte sequence without continuation)
        self._write_to_pipe(write_fd, b'Incomplete: \xe2\x28\xa1\n')

        self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # Check that invalid bytes are replaced with Unicode replacement character
        content = self._read_log_file(temp_log_file)
        assert '0: Invalid:' in content
        assert '0: Incomplete:' in content
        # The replacement character should be present (U+FFFD)
        assert '\ufffd' in content or '?' in content  # Different systems may render differently

    def test_incomplete_line_buffering(self, temp_log_file):
        """Test that incomplete lines (without \\n) are buffered correctly."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write incomplete line (no \n)
        self._write_to_pipe(write_fd, b'Incomplete')
        time.sleep(0.1)  # Give thread time to process

        # Complete the line
        self._write_to_pipe(write_fd, b' line completed\n')
        time.sleep(0.1)

        self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # Check that incomplete line was buffered and completed
        content = self._read_log_file(temp_log_file)
        assert '0: Incomplete line completed\n' in content

    def test_incomplete_line_on_pipe_close(self, temp_log_file):
        """Test that incomplete line is flushed when pipe closes."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write incomplete line and close pipe immediately
        self._write_to_pipe(write_fd, b'Incomplete line without newline')
        self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # Check that incomplete line was flushed with added newline
        content = self._read_log_file(temp_log_file)
        assert '0: Incomplete line without newline\n' in content

    def test_mixed_data_multiple_ranks(self, temp_log_file):
        """Test complex scenario with multiple ranks, NULL bytes, and invalid UTF-8."""
        # Create pipes for 2 ranks
        pipes = {}
        write_fds = {}
        for rank in range(2):
            read_fd, write_fd = self._create_pipe_pair()
            pipes[rank] = read_fd
            write_fds[rank] = write_fd

        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=10,
        )
        reader.start()

        # Rank 0: Normal output
        self._write_to_pipe(write_fds[0], b'Rank 0: Normal line\n')

        # Rank 1: Output with NULL bytes
        self._write_to_pipe(write_fds[1], b'Rank 1: Has \x00 null\n')

        # Rank 0: Invalid UTF-8
        self._write_to_pipe(write_fds[0], b'Rank 0: Invalid \xff\xfe\n')

        # Rank 1: Incomplete line
        self._write_to_pipe(write_fds[1], b'Rank 1: Incomplete')
        time.sleep(0.1)
        self._write_to_pipe(write_fds[1], b' completed\n')

        # Close all pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # Check all output
        content = self._read_log_file(temp_log_file)
        assert '0: Rank 0: Normal line\n' in content
        assert '1: Rank 1: Has <NUL> null\n' in content
        assert '0: Rank 0: Invalid' in content
        assert '1: Rank 1: Incomplete completed\n' in content

    def test_large_data_throughput(self, temp_log_file):
        """Test handling of large amounts of data."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write many lines
        num_lines = 1000
        for i in range(num_lines):
            self._write_to_pipe(write_fd, f'Line {i}\n'.encode('utf-8'))

        self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=5.0)

        # Check all lines are present
        content = self._read_log_file(temp_log_file)
        lines = content.strip().split('\n')
        assert len(lines) == num_lines
        assert '0: Line 0' in content
        assert f'0: Line {num_lines-1}' in content

    def test_rank_prefix_padding(self, temp_log_file):
        """Test that rank prefixes are properly padded based on world_size."""
        # Test with world_size=1 (no padding needed)
        read_fd1, write_fd1 = self._create_pipe_pair()
        reader1 = MultiplexingReaderThread(
            pipes={0: read_fd1},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader1.start()
        self._write_to_pipe(write_fd1, b'Test\n')
        self._close_pipe(write_fd1)
        reader1.join(timeout=2.0)

        content1 = self._read_log_file(temp_log_file)
        # With world_size=1, no padding (rank_width=0)
        assert '0: Test\n' in content1

        # Clear file
        with open(temp_log_file, 'w') as f:
            f.write('')

        # Test with world_size=100 (2-digit padding for ranks 0-99)
        read_fd2, write_fd2 = self._create_pipe_pair()
        reader2 = MultiplexingReaderThread(
            pipes={5: read_fd2},
            log_file_path=temp_log_file,
            world_size=100,
        )
        reader2.start()
        self._write_to_pipe(write_fd2, b'Test\n')
        self._close_pipe(write_fd2)
        reader2.join(timeout=2.0)

        content2 = self._read_log_file(temp_log_file)
        # Should be right-aligned with width 2 (max rank is 99, which is 2 digits)
        assert ' 5: Test\n' in content2

    def test_concurrent_writes(self, temp_log_file):
        """Test that concurrent writes from multiple ranks don't merge lines."""
        pipes = {}
        write_fds = {}
        for rank in range(4):
            read_fd, write_fd = self._create_pipe_pair()
            pipes[rank] = read_fd
            write_fds[rank] = write_fd

        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=4,
        )
        reader.start()

        # Write from all ranks concurrently
        def write_from_rank(rank, write_fd):
            for i in range(10):
                self._write_to_pipe(write_fd, f'Rank {rank} line {i}\n'.encode('utf-8'))
                time.sleep(0.001)  # Small delay to interleave writes

        threads = []
        for rank, write_fd in write_fds.items():
            t = threading.Thread(target=write_from_rank, args=(rank, write_fd))
            t.start()
            threads.append(t)

        # Wait for all writers to finish
        for t in threads:
            t.join()

        # Close all pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=5.0)

        # Check that all lines are present and properly formatted
        content = self._read_log_file(temp_log_file)
        lines = content.strip().split('\n')

        # Should have 40 lines total (4 ranks * 10 lines each)
        assert len(lines) == 40

        # Check each line has proper format
        for line in lines:
            assert ': Rank ' in line
            assert ' line ' in line

    def test_append_mode(self, temp_log_file):
        """Test that reader appends to existing file."""
        # Write some initial content
        with open(temp_log_file, 'w') as f:
            f.write('Existing content\n')

        read_fd, write_fd = self._create_pipe_pair()
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        self._write_to_pipe(write_fd, b'New content\n')
        self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # Check both old and new content are present
        content = self._read_log_file(temp_log_file)
        assert 'Existing content\n' in content
        assert '0: New content\n' in content

    def test_empty_pipe(self, temp_log_file):
        """Test handling of pipe that closes without any data."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Close pipe immediately without writing
        self._close_pipe(write_fd)

        reader.shutdown()
        reader.join(timeout=2.0)

        # File should be empty or contain only pre-existing content
        content = self._read_log_file(temp_log_file)
        # No crash, thread terminates cleanly
        assert reader is not None


class TestPipeSubprocessHandler:
    """Tests for PipeSubprocessHandler class."""

    def test_close_releases_pipe_fds(self):
        """
        Test that PipeSubprocessHandler.close() properly closes parent's pipe FDs.

        This is a regression test for the FD reuse bug where:
        1. Workers from cycle N are shut down
        2. Pipes remain open because close() didn't close proc.stdout/stderr
        3. FDs get reused when cycle N+1 workers spawn quickly
        4. Old reader thread tries to read from reused FDs → EBADF/EINVAL errors

        The fix ensures that close() explicitly closes self.proc.stdout/stderr
        to release the parent's read-end of the pipes.
        """
        # Create a worker process with pipes
        handler = PipeSubprocessHandler(
            entrypoint="bash",
            args=("-c", "sleep 0.5"),
            env={},
            stdout="__SUBPROCESS_PIPE__",
            stderr="__SUBPROCESS_PIPE__",
            local_rank_id=0,
        )

        # Get the parent's read-end FD
        parent_fd_stdout = handler.proc.stdout.fileno()

        # Verify FD is valid before close
        os.fstat(parent_fd_stdout)  # Should not raise

        # Close the handler (should close parent's pipe FDs)
        handler.close()

        # Verify FD is now closed (should raise EBADF)
        with pytest.raises(OSError) as exc_info:
            os.fstat(parent_fd_stdout)

        assert exc_info.value.errno == 9  # EBADF - Bad file descriptor

        # Clean up
        handler.proc.wait(timeout=2.0)

    def test_close_with_non_pipe_stdout(self):
        """
        Test that close() works correctly when stdout is a file (not a pipe).

        In this case, proc.stdout should be None, so closing it should be a no-op.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name

        try:
            handler = PipeSubprocessHandler(
                entrypoint="bash",
                args=("-c", "echo test"),
                env={},
                stdout=temp_path,
                stderr=temp_path,
                local_rank_id=0,
            )

            # For non-pipe case, proc.stdout should be None
            assert handler.proc.stdout is None

            # close() should work without errors
            handler.close()
            handler.proc.wait(timeout=2.0)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_fd_closed_immediately_when_process_alive(self):
        """
        Test that FDs are closed immediately when process is still alive (normal case).

        This tests PyTorch's conditional logic:
            if handler.proc.poll() is None:  # Process still alive
                handler.close(death_sig=death_sig)  # CALLED!

        When proc.poll() is None (process alive), handler.close() IS called.
        Our fix closes proc.stdout/stderr immediately, releasing FDs.
        """
        # Create a worker with a long-running process
        handler = PipeSubprocessHandler(
            entrypoint="bash",
            args=("-c", "sleep 10"),  # Long enough that it's still alive
            env={},
            stdout="__SUBPROCESS_PIPE__",
            stderr="__SUBPROCESS_PIPE__",
            local_rank_id=0,
        )

        parent_fd = handler.proc.stdout.fileno()

        # Verify process is still alive (normal shutdown scenario)
        assert handler.proc.poll() is None, "Process should still be alive"

        # Verify FD is open before cleanup
        os.fstat(parent_fd)  # Should not raise

        # Simulate PyTorch's SubprocessContext._close() conditional logic:
        # It calls handler.close() if proc.poll() is None
        if handler.proc.poll() is None:
            handler.close()  # This IS called (scenario 1)

        # Our fix (lines 180-189 in per_cycle_logs.py) should have closed the FD
        # immediately inside handler.close()
        with pytest.raises(OSError) as exc_info:
            os.fstat(parent_fd)
        assert exc_info.value.errno == 9  # EBADF - FD properly closed!

        # Clean up
        handler.proc.wait(timeout=2.0)

        # Summary: Normal case where our fix works. When handler.close()
        # is called, FDs are released immediately, preventing FD reuse issues.

    def test_fd_not_closed_when_process_already_dead(self):
        """
        Test that FDs remain open when process dies before close (edge case).

        This tests PyTorch's conditional logic:
            if handler.proc.poll() is None:  # Only if process still running
                handler.close(death_sig=death_sig)

        When proc.poll() != None (process dead), handler.close() is SKIPPED.
        Pipes remain open until GC collects the Popen object.

        NOTE: This edge case is NOW FIXED in launcher.py!
        The _stop_workers() method explicitly closes all pipe FDs before
        spawning new workers, ensuring deterministic cleanup regardless
        of whether PyTorch called handler.close() or not.

        This test documents the PyTorch behavior, but the actual FD cleanup
        happens in _stop_workers() which is tested in integration tests.
        """

        # Create a worker with a short-lived process
        handler = PipeSubprocessHandler(
            entrypoint="bash",
            args=("-c", "exit 0"),  # Dies immediately
            env={},
            stdout="__SUBPROCESS_PIPE__",
            stderr="__SUBPROCESS_PIPE__",
            local_rank_id=0,
        )

        parent_fd = handler.proc.stdout.fileno()

        # Wait for process to die
        handler.proc.wait(timeout=2.0)

        # Verify process is dead (simulates fast worker failure)
        assert handler.proc.poll() is not None, "Process should be dead"

        # Verify FD is still open before any cleanup
        os.fstat(parent_fd)  # Should NOT raise - FD still open

        # Simulate PyTorch's SubprocessContext._close() conditional logic:
        # It only calls handler.close() if proc.poll() is None
        if handler.proc.poll() is None:
            handler.close()  # Would be called
        # else: handler.close() is SKIPPED (our case!)

        # Verify FD is still open because handler.close() was skipped
        # This demonstrates that PyTorch's conditional logic leaves FDs open
        # when processes die before _close() is called
        os.fstat(parent_fd)  # Should still NOT raise - FD still open

        # Document the behavior: FD only closes when we manually close it
        # or when GC collects the Popen object
        handler.proc.stdout.close()

        # Now it should be closed
        with pytest.raises(OSError) as exc_info:
            os.fstat(parent_fd)
        assert exc_info.value.errno == 9  # EBADF

        # Summary: This test documents PyTorch's conditional close behavior.
        # The actual fix for FD reuse issues is in _stop_workers() (launcher.py)
        # which explicitly closes all pipe FDs before spawning new workers.

    def test_shutdown_mechanism_prevents_fd_reuse_catastrophe(self):
        """
        Test that the shutdown mechanism allows graceful thread termination.

        The thread reuse pattern prevents the old FD reuse catastrophe by:
        1. Using ONE long-lived thread across all cycles (no lingering threads)
        2. Thread updates its pipes via update_pipes() for new cycles
        3. Reader thread ONLY unregisters FDs, doesn't close them (ownership with file objects)
        4. _stop_workers() explicitly closes FDs before spawning new workers

        This test verifies the shutdown() mechanism works correctly for process exit:
        - shutdown() sets _shutdown_requested flag
        - Thread breaks main loop and exits cleanly
        - finally block calls _cleanup_resources() to flush logs
        """
        # Create two pipes for the old reader
        r1, w1 = os.pipe()
        r2, w2 = os.pipe()

        # Create old reader thread
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name

        try:
            from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import (
                MultiplexingReaderThread,
            )

            old_reader = MultiplexingReaderThread(
                pipes={0: r1, 1: r2},
                log_file_path=log_file,
            )
            old_reader.start()

            # Give it a moment to start polling
            time.sleep(0.1)

            # Request shutdown (simulates process exit cleanup)
            old_reader.shutdown()

            # Thread should exit quickly
            old_reader.join(timeout=2.0)
            assert not old_reader.is_alive(), "Reader thread should exit after shutdown()"

            # Thread exited cleanly - shutdown mechanism works!

        finally:
            # Cleanup
            for fd in [w1, w2]:  # Only close write ends, read ends handled by thread
                try:
                    os.close(fd)
                except OSError:
                    pass
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_thread_reuse_pattern3_atomic_swap(self):
        """
        Test that thread reuse uses Pattern 3 (Atomic Pointer Swap).

        Pattern 3 advantages over Pattern 4 (queues):
        ✓ Simpler - no queue module needed
        ✓ Atomic - GIL guarantees reference assignment is atomic
        ✓ Direct - just compare object identity
        ✓ Perfect for our use case (one update per cycle)

        This test verifies the API exists. Full integration testing
        is done in end-to-end tests with real workers.
        """
        from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import (
            MultiplexingReaderThread,
            ReaderConfig,
        )

        # Just verify the API and structure - don't actually run the thread
        # (Full functional testing requires real worker lifecycle)
        # Verify ReaderConfig dataclass exists
        config = ReaderConfig(
            pipes={0: 123},
            log_file_path="/tmp/test.log",
            world_size=None,
            local_to_global_rank={},
            launcher_pipe_fd=None,  # New optional field
            launcher_log_file_path=None,  # New optional field
        )
        assert config.pipes == {0: 123}
        assert config.log_file_path == "/tmp/test.log"
        assert config.launcher_pipe_fd is None
        assert config.launcher_log_file_path is None

        # Verify MultiplexingReaderThread has the Pattern 3 structure
        # (We check the class, not an instance, to avoid pipe registration issues)
        assert hasattr(
            MultiplexingReaderThread, 'update_pipes'
        ), "Should have update_pipes method for thread reuse"
        assert hasattr(
            MultiplexingReaderThread, '_apply_config'
        ), "Should have _apply_config for atomic state switching"


class TestReaderThreadErrorHandling:
    """Tests for MultiplexingReaderThread error handling during pipe operations.

    These are regression tests for a race condition bug where:
    - Launcher closes pipe FDs in _stop_workers() during worker termination
    - OS immediately reuses those FD numbers for other operations
    - Reader thread's poller still has the old FD registered
    - Reader thread tries to read from the reused FD
    - Gets EBADF (errno 9) or EISDIR (errno 21) errors

    The fix treats these errors as expected pipe closures (DEBUG log) instead of WARNING,
    since they're normal during worker restart scenarios.
    """

    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file."""
        fd, path = tempfile.mkstemp(suffix='.log')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_ebadf_error_handling(self, temp_log_file):
        """Test that EBADF (Bad file descriptor) is handled gracefully.

        This is a regression test for a race condition where:
        1. Launcher closes pipe FDs in _stop_workers()
        2. OS reuses FD for another operation
        3. Reader thread tries to read from the reused FD
        4. Gets EBADF error

        The fix treats EBADF as expected pipe closure (DEBUG log) instead of WARNING.
        """
        # Create a pipe
        read_fd, write_fd = os.pipe()

        # Create reader thread
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
            local_to_global_rank={0: 0},
        )
        reader.start()

        # Write some data
        os.write(write_fd, b'Before close\n')
        time.sleep(0.1)

        # Close both ends to simulate external closure
        os.close(write_fd)
        os.close(read_fd)

        # Give thread time to detect closure and handle EBADF
        time.sleep(0.3)

        # Thread should still be running (not crashed)
        assert reader.is_alive()

        # Shutdown cleanly
        reader.shutdown()
        reader.join(timeout=2.0)

        # Verify data was written before the error
        with open(temp_log_file, 'r') as f:
            content = f.read()
        assert 'Before close\n' in content

    def test_eisdir_error_handling(self, temp_log_file):
        """Test that EISDIR (Is a directory) is handled gracefully.

        This tests the case where an FD gets reused for a directory operation
        after being closed externally.
        """
        # Create a pipe
        read_fd, write_fd = os.pipe()

        # Create reader thread
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
            local_to_global_rank={0: 0},
        )

        # Mock os.read to simulate EISDIR error
        original_read = os.read
        call_count = [0]

        def mock_read(fd, size):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call simulates EISDIR
                raise OSError(errno.EISDIR, "Is a directory")
            return original_read(fd, size)

        with patch('os.read', side_effect=mock_read):
            reader.start()

            # Write some data (first read succeeds)
            os.write(write_fd, b'Before EISDIR\n')
            time.sleep(0.2)

            # Second write will trigger EISDIR in mock
            os.write(write_fd, b'Trigger EISDIR\n')
            time.sleep(0.2)

        # Close pipes
        os.close(write_fd)

        # Thread should handle EISDIR gracefully
        reader.shutdown()
        reader.join(timeout=2.0)

        # Verify first data was written
        with open(temp_log_file, 'r') as f:
            content = f.read()
        assert 'Before EISDIR\n' in content

    def test_any_oserror_handled_gracefully(self, temp_log_file):
        """Test that any OSError (not just EBADF/EISDIR) is handled gracefully.

        When an FD is closed and reused, it can be reused for any type of file descriptor
        (socket, regular file, device, etc.), resulting in various errno values (ESPIPE,
        ENOTSOCK, EINVAL, etc.). All should be treated as expected pipe closures.
        """
        # Create a pipe
        read_fd, write_fd = os.pipe()

        # Create reader thread
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
            local_to_global_rank={0: 0},
        )

        # Mock os.read to simulate a different OSError (ESPIPE - illegal seek)
        original_read = os.read
        call_count = [0]

        def mock_read(fd, size):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call raises ESPIPE
                raise OSError(errno.ESPIPE, "Illegal seek")
            return original_read(fd, size)

        with patch('os.read', side_effect=mock_read):
            reader.start()

            # Write some data (first read succeeds)
            os.write(write_fd, b'Before ESPIPE\n')
            time.sleep(0.2)

            # Second write will trigger ESPIPE in mock
            os.write(write_fd, b'Trigger ESPIPE\n')
            time.sleep(0.2)

        # Close pipes
        os.close(write_fd)

        # Thread should handle ESPIPE gracefully (no crash)
        reader.shutdown()
        reader.join(timeout=2.0)

        # Verify first data was written
        with open(temp_log_file, 'r') as f:
            content = f.read()
        assert 'Before ESPIPE\n' in content
