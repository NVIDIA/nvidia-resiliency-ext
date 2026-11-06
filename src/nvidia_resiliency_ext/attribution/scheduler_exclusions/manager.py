# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process lifecycle for a scheduler publisher, independent of its deployment owner."""

from __future__ import annotations

import logging
import os
import signal
import subprocess  # nosec B404
import sys
from collections.abc import Mapping

from .publisher import PublisherConfig

logger = logging.getLogger(__name__)
_TERMINATE_TIMEOUT = 2.0
_KILL_TIMEOUT = 0.5


class SegmentHealthPublisherManager:
    """Own an isolated publisher without readiness waits or restart supervision.

    The caller selects the process owner and supplies explicit job configuration.
    This manager does not select a launcher or discover jobs to monitor.
    """

    def __init__(
        self,
        config: PublisherConfig,
        *,
        env: Mapping[str, str] | None = None,
    ):
        self.config = config
        self._env = dict(os.environ if env is None else env)
        self._env.update(
            {
                "SLURM_ARRAY_JOB_ID": config.job_id,
                "SLURM_JOB_PARTITION": config.partition,
                "NVRX_SEGMENT_HEALTH_CHECK_DIR": str(config.directory),
                "NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS": str(config.stale_seconds),
                "NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT": str(config.command_timeout),
            }
        )
        self.process: subprocess.Popen | None = None

    def start(self) -> bool:
        if self.process is not None:
            return True
        command = [
            sys.executable,
            "-m",
            "nvidia_resiliency_ext.services.scheduler_exclusions",
            "--parent-pid",
            str(os.getpid()),
        ]
        # The child alone touches Slurm and shared artifacts. It is not a
        # training worker and has its own process group for bounded cleanup.
        self.process = subprocess.Popen(  # nosec B603
            command,
            env=self._env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
            start_new_session=True,
            close_fds=True,
        )
        logger.info(
            "Segment health publisher started PID=%s array=%s",
            self.process.pid,
            self.config.job_id,
        )
        return True

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            self._signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=_TERMINATE_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Segment health publisher did not stop; sending SIGKILL")
        # Clean up command descendants too, even if their publisher already exited.
        self._signal_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=_KILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.error("Segment health publisher could not be reaped")
        else:
            self.process = None

    @staticmethod
    def _signal_group(process: subprocess.Popen, sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            logger.exception("Could not signal segment health publisher group PID=%s", process.pid)
