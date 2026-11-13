# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass configuration to supervisor-owned children and exec fresh interpreters.

An inherited, unlinked file preserves parsed arguments without shell quoting.
Child lifecycle is managed exclusively through process exit status and signals.
"""

import logging
import os
import pickle  # nosec B403 - private supervisor-created configuration, never external data
import signal
import subprocess
import sys
import tempfile

from ..shared_utils.log_manager import LogConfig

CONFIG_FD_ENV = "_NVRX_LAUNCH_CONFIG_FD"
KILL_WAIT_TIMEOUT = 2.0
logger = logging.getLogger(LogConfig.name)


def wait_or_kill_child(
    proc: subprocess.Popen,
    timeout: float,
    *,
    kill_timeout: float = KILL_WAIT_TIMEOUT,
    process_group: bool = False,
) -> bool:
    """Wait after SIGTERM, escalate to SIGKILL, then wait for a bounded time.

    The caller sends SIGTERM so multiple children can begin shutdown together.
    Return False if the child survives the final wait; cleanup must continue.
    """
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Child PID=%s did not exit within %.1fs; sending SIGKILL", proc.pid, timeout)
    try:
        if process_group:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except OSError:
        logger.exception("Failed to send SIGKILL to child PID=%s", proc.pid)
    try:
        proc.wait(timeout=kill_timeout)
        return True
    except subprocess.TimeoutExpired:
        logger.error(
            "Child PID=%s still alive %.1fs after SIGKILL; continuing cleanup",
            proc.pid,
            kill_timeout,
        )
        return False


def read_child_args():
    fd = os.environ.pop(CONFIG_FD_ENV, None)
    if fd is None:
        return None
    with os.fdopen(int(fd), "rb") as stream:
        # Only read the private, unlinked file inherited from the supervisor.
        return pickle.load(stream)  # nosec B301


def spawn_child(module, args, **popen_kwargs):
    """Exec a fresh interpreter without shell quoting or reparsing training argv.

    Telemetry configuration, including OTEL_RESOURCE_ATTRIBUTES, is inherited.
    Children that emit telemetry initialize and shut down their own providers.
    """
    env = os.environ.copy()
    env.pop(CONFIG_FD_ENV, None)
    env["PYTHONUNBUFFERED"] = "1"
    with tempfile.TemporaryFile() as config:
        pickle.dump(args, config, protocol=pickle.HIGHEST_PROTOCOL)
        config.seek(0)
        env[CONFIG_FD_ENV] = str(config.fileno())
        return subprocess.Popen(  # nosec B603
            [sys.executable, "-u", "-m", module],
            env=env,
            pass_fds=(config.fileno(),),
            **popen_kwargs,
        )
