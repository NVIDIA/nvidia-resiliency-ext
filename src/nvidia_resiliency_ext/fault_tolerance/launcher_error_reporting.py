# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local fallback for fatal launcher errors before/after log forwarding."""

import contextlib
import os
import socket
import traceback
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .per_cycle_logs import PipeBasedLogsSpecs


class LauncherErrorReporter:
    def __init__(self):
        self._stderr_fd: Optional[int] = None
        self.logs_specs: Optional["PipeBasedLogsSpecs"] = None

    def save_stderr(self):
        """Preserve Slurm/console stderr before the launcher redirects it."""
        self.close()
        # Non-inheritable across exec; forked helpers close their own copy.
        self._stderr_fd = os.dup(2)

    def report_exception(self, exc: Exception):
        """Use original stderr unless launcher log forwarding is available."""
        # Reporting must not mask the original exception or prevent cleanup.
        with contextlib.suppress(Exception):
            if self._stderr_fd is None:
                return
            if self.logs_specs is not None and self.logs_specs.launcher_forwarding_available:
                return
            with os.fdopen(
                self._stderr_fd, "w", encoding="utf-8", errors="backslashreplace", closefd=False
            ) as stream:
                print(
                    f"ft_launcher failed on {socket.gethostname()} (pid={os.getpid()}):",
                    file=stream,
                )
                traceback.print_exception(exc, file=stream)

    def close(self):
        """Release this process's copy, including in forked rank monitors."""
        if self._stderr_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._stderr_fd)
            self._stderr_fd = None
        self.logs_specs = None


launcher_error_reporter = LauncherErrorReporter()
