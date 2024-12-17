#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: BSD-3-Clause
# Modifications made by NVIDIA
# Added shell=False to Popen to mitigate security thread
# Added suppression for subprocess low serverity issue

import os
import signal
# Issue: [B404:blacklist] Consider possible security implications associated with the subprocess module.
# Severity: Low   Confidence: High
# CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
# More Info: https://bandit.readthedocs.io/en/1.7.9/blacklists/blacklist_imports.html#b404-import-subprocess
import subprocess  # nosec
import sys

from typing import Any, Dict, Optional, Tuple

__all__ = ["SubprocessHandler"]

IS_WINDOWS = sys.platform == "win32"


def _get_default_signal() -> signal.Signals:
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGTERM


class SubprocessHandler:
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """

    def __init__(
        self,
        entrypoint: str,
        args: Tuple,
        env: Dict[str, str],
        stdout: str,
        stderr: str,
        local_rank_id: int,
    ):
        self._stdout = open(stdout, "w") if stdout else None
        self._stderr = open(stderr, "w") if stderr else None
        # inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        self.local_rank_id = local_rank_id
        self.proc: subprocess.Popen = self._popen(args_str, env_vars)

    def _popen(self, args: Tuple, env: Dict[str, str]) -> subprocess.Popen:
        kwargs: Dict[str, Any] = {}
        if not IS_WINDOWS:
            kwargs["start_new_session"] = True
        # Issue: [B603:subprocess_without_shell_equals_true] subprocess call - check for execution of untrusted input.
        # Severity: Low   Confidence: High
        # CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)
        # More Info: https://bandit.readthedocs.io/en/1.7.9/plugins/b603_subprocess_without_shell_equals_true.html
        return subprocess.Popen(
            # pyre-fixme[6]: Expected `Union[typing.Sequence[Union[_PathLike[bytes],
            #  _PathLike[str], bytes, str]], bytes, str]` for 1st param but got
            #  `Tuple[str, *Tuple[Any, ...]]`.
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
            **kwargs,
            shell=False,
        )  # nosec

    def close(self, death_sig: Optional[signal.Signals] = None) -> None:
        if not death_sig:
            death_sig = _get_default_signal()
        if IS_WINDOWS:
            self.proc.send_signal(death_sig)
        else:
            os.killpg(self.proc.pid, death_sig)
        if self._stdout:
            self._stdout.close()
        if self._stderr:
            self._stderr.close()
