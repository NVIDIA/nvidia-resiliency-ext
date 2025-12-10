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
Custom LogsSpecs implementation that consolidates all ranks' logs into a single file per cycle.

This module provides PerCycleLogsSpecs which consolidates logs using O_APPEND for safe concurrent writes:
1. Parent creates the log file path and passes it to children via environment variable
2. Each child opens the log file with O_APPEND flag at Python startup (in sitecustomize.py)
3. Child redirects stdout/stderr to the log file BEFORE any user/PyTorch code runs
4. PyTorch's redirect mechanism is bypassed (we pass None for log paths)

The O_APPEND flag ensures atomic appends, making concurrent writes from multiple ranks safe.
This approach uses standard OS primitives and requires no monkey-patching.

Why not use PyTorch's built-in redirect?
--------------------------------------
PyTorch's torch.distributed.elastic.multiprocessing.redirects.redirect() opens files with
mode="w+b" which truncates the file. This is fine for per-rank files, but causes data loss
when multiple ranks write to the same file. Additionally, PyTorch's redirect happens inside
the worker process after it starts, making it difficult to intercept without monkey-patching.

Our approach redirects stdout/stderr at Python startup (before PyTorch loads) using
sitecustomize.py, and opens the file with O_APPEND for safe concurrent writes from all ranks.
"""

import logging
import os
import tempfile
from pathlib import Path

from torch.distributed.elastic.multiprocessing import LogsDest, LogsSpecs, Std

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig


class PerCycleLogsSpecs(LogsSpecs):
    """
    LogsSpecs that consolidates all ranks' logs into a single file per cycle using O_APPEND.

    Designed for SLURM environments where you want:
    - One log file per restart cycle (e.g., job_cycle0.log)
    - All ranks' output in the same file (truly consolidated)
    - Both stdout and stderr combined (simplified debugging)
    - Flat file structure (no nested directories)
    - Works with both 'fork' and 'spawn' start methods

    Log file structure:
        /path/to/job_cycle0.log  # All ranks, stdout+stderr, cycle 0
        /path/to/job_cycle1.log  # All ranks, stdout+stderr, cycle 1

        /tmp/torchelastic_errors_<run_id>/
            cycle_0/
                0/error.json
                1/error.json

    Args:
        base_log_file: Base path for log files (e.g., "/lustre/logs/job.log").
                      Creates: job_cycle0.log, job_cycle1.log, etc.

    Implementation:
        Consolidated logging approach:
        1. Parent creates log file and passes path via TORCHELASTIC_CONSOLIDATED_LOG env var
        2. Each child opens the file with O_APPEND in sitecustomize.py (runs at startup)
        3. Child redirects stdout/stderr to the log file before any other code runs
        4. PyTorch's redirect mechanism is bypassed (we pass None for log paths)

        Note: With Python's 'spawn' start method, file descriptors are NOT inherited,
        so each child must open the file. O_APPEND flag ensures safe concurrent writes.

        Rank prefixing is automatically enabled: each line is prefixed with [rank]:
        (using global rank, like 'srun -l') for easy identification of which rank
        produced each line.
    """

    logger = logging.getLogger(LogConfig.name)

    def __init__(self, base_log_file: str) -> None:
        if not base_log_file:
            raise ValueError("base_log_file is required for PerCycleLogsSpecs")

        # Convert to absolute path for multi-node safety (all nodes must access same path)
        self._base_log_file = os.path.abspath(base_log_file)

        # Extract directory and ensure it exists
        log_dir = os.path.dirname(self._base_log_file) or "."
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        super().__init__(
            log_dir=log_dir,
            redirects=Std.ALL,
            tee=Std.NONE,
            local_ranks_filter=None,
        )

        self._error_dir = None

    @property
    def root_log_dir(self) -> str:
        return str(self._root_log_dir)

    def reify(
        self,
        envs: dict[int, dict[str, str]],
    ) -> LogsDest:
        """
        Creates log destination with a single consolidated log file opened with O_APPEND.

        Creates:
        - /path/to/base_cycle<N>.log (all ranks, stdout+stderr combined)
        - /tmp/torchelastic_errors_<run_id>/cycle_<N>/<rank>/error.json (per rank)
        - /tmp/nvrx_ft_startup_<rand>/sitecustomize.py (for early redirection)

        Each child process opens the log file with O_APPEND at startup (via sitecustomize.py)
        and redirects stdout/stderr to it before any user or PyTorch code runs.
        """
        nprocs = len(envs)
        if nprocs == 0:
            self.logger.debug("Empty envs map provided when defining logging destinations.")
            return LogsDest({}, {}, {}, {}, {})

        # Get restart count from environment
        global_env = envs[0]
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        # Create per-cycle log file
        base_without_ext = os.path.splitext(self._base_log_file)[0]
        ext = os.path.splitext(self._base_log_file)[1] or ".log"
        cycle_log_file = f"{base_without_ext}_cycle{restart_count}{ext}"

        # Create the consolidated log file if it doesn't exist
        # This serves two purposes:
        # 1. Early detection of permission/path issues before launching workers
        # 2. Ensures file exists for child processes (though O_CREAT would create it anyway)
        if not os.path.exists(cycle_log_file):
            open(cycle_log_file, 'a').close()
            self.logger.info("Created consolidated log file: %s", cycle_log_file)

        # Note: Unlike binary processes, Python multiprocessing with 'spawn'
        # doesn't inherit file descriptors. So we pass the FILE PATH instead of fd,
        # and each child opens it with O_APPEND in sitecustomize.py

        # Create a sitecustomize.py that redirects stdout/stderr to our log file
        if not hasattr(self, '_sitecustomize_dir'):
            self._sitecustomize_dir = tempfile.mkdtemp(prefix="nvrx_ft_startup_")
            sitecustomize_path = os.path.join(self._sitecustomize_dir, "sitecustomize.py")

            # Read the sitecustomize template and write it to the temp directory
            template_path = Path(__file__).parent / "sitecustomize_template.py"
            sitecustomize_contents = template_path.read_text()

            with open(sitecustomize_path, 'w') as f:
                f.write(sitecustomize_contents)

            self.logger.debug(
                "Created sitecustomize.py for early log redirection in: %s",
                self._sitecustomize_dir,
            )

        # Configure environment for each rank
        for rank_env in envs.values():
            # Pass the log file path (each child will open it with O_APPEND)
            rank_env["TORCHELASTIC_CONSOLIDATED_LOG"] = cycle_log_file

            # Add sitecustomize dir to PYTHONPATH so it auto-imports at startup
            current_pythonpath = rank_env.get("PYTHONPATH", "")
            if current_pythonpath:
                rank_env["PYTHONPATH"] = f"{self._sitecustomize_dir}:{current_pythonpath}"
            else:
                rank_env["PYTHONPATH"] = self._sitecustomize_dir

        # Bypass PyTorch's redirect mechanism by passing None
        #
        # Why we don't use PyTorch's redirect:
        # - PyTorch's redirect (torch.distributed.elastic.multiprocessing.redirects.redirect)
        #   opens files with mode="w+b" which TRUNCATES the file
        # - If multiple ranks write to the same file, they would overwrite each other
        # - PyTorch's redirect happens INSIDE the worker process (after it starts)
        #
        # Our approach instead:
        # - We redirect stdout/stderr at Python startup (via sitecustomize.py)
        # - BEFORE PyTorch or any user code runs
        # - Each rank opens the file with O_APPEND flag for safe concurrent writes
        #
        # By passing None here, PyTorch's get_std_cm() returns nullcontext() and skips
        # its redirect entirely, leaving our early redirection in place.
        stdouts = {rank: None for rank in range(nprocs)}
        stderrs = {rank: None for rank in range(nprocs)}

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

        self.logger.debug(
            "Cycle %s log: %s (consolidated via O_APPEND, %d ranks)",
            restart_count,
            cycle_log_file,
            nprocs,
        )

        return LogsDest(stdouts, stderrs, tee_stdouts, tee_stderrs, error_files)

    def __repr__(self) -> str:
        return f"PerCycleLogsSpecs(base_log_file={self._base_log_file})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PerCycleLogsSpecs):
            return False
        return self._base_log_file == other._base_log_file
