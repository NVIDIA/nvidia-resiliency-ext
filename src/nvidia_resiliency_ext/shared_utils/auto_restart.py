# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import os
import sys
import time
from datetime import datetime


def log_message(message: str) -> None:
    """
    Log a message to stderr with timestamp.

    Args:
        message: The message to log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write(f"[{timestamp}] {message}\n")


def fork_and_monitor(restart_delay: float = 5.0) -> None:
    """
    Initiates monitor and restart by forking the process.

    ⚠️ CRITICAL: This function MUST be called at the very beginning of your program,
    before ANY other imports or initialization. This is similar to how gevent.monkey_patch()
    must be called before any other imports.

    The function uses os.fork(), which is NOT CUDA-safe. When a process with an active
    CUDA context is forked, both parent and child processes inherit the same CUDA context,
    which can cause crashes, memory corruption, or unpredictable behavior. By calling
    this function before any CUDA initialization, only the child process will create
    a CUDA context, avoiding these issues.

    Control is returned to the caller from the child process. If the child exits
    normally, with a status code of zero, the parent process will also exit.
    Under all other circumstances in which the child terminates (i.e., with a
    signal or non-zero status code), the parent will attempt to fork a new child
    after a short delay.

    Args:
        restart_delay: Delay in seconds between restart attempts

    Returns:
        None.

    Example:
        # ✅ CORRECT - Call at the very beginning
        from nvidia_resiliency_ext.shared_utils.auto_restart import fork_and_monitor
        fork_and_monitor()

        # Now import other modules
        import torch
        import numpy as np
        # ... rest of your code

        if __name__ == "__main__":
            main()

    Example:
        # ❌ WRONG - Don't import other modules first
        import torch
        import numpy as np
        from nvidia_resiliency_ext.shared_utils.auto_restart import fork_and_monitor
        fork_and_monitor()   # Too late! torch and numpy already imported
        # ... rest of code
    """
    log_message("Starting auto-restart mode")

    while True:
        try:
            pid = os.fork()
            if pid == 0:
                return
        except OSError as e:
            log_message(f"fork: {e}")
            os._exit(1)

        log_message(f"Launched child {pid}")
        while True:
            try:
                pid, ret = os.waitpid(pid, 0)
                break
            except OSError as e:
                if e.errno == errno.EINTR:
                    continue
                log_message(f"waitpid: {e}")
                os._exit(1)

        if os.WIFEXITED(ret):
            status = os.WEXITSTATUS(ret)
            log_message(f"Child {pid} exited with status {status}")
            if status == 0:
                os._exit(0)
        elif os.WIFSIGNALED(ret):
            sig = os.WTERMSIG(ret)
            log_message(f"Child {pid} exited with signal {sig}")
        else:
            log_message(f"Child {pid} exited abnormally ({ret:x})")

        time.sleep(restart_delay)
