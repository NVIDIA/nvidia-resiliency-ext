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


def monitor_and_restart(restart_delay: float = 5.0) -> None:
    """
    Initiates monitor and restart by forking the process.

    Control is returned to the caller from the child process. If the child exits
    normally, with a status code of zero, the parent process will also exit.
    Under all other circumstances in which the child terminates (i.e., with a
    signal or non-zero status code), the parent will attempt to fork a new child
    after a short delay.

    Args:
        restart_delay: Delay in seconds between restart attempts

    Returns:
        None.
    """
    sys.stderr.write("Starting self-monitoring mode\n")

    while True:
        try:
            pid = os.fork()
            if pid == 0:
                return
        except OSError as e:
            sys.stderr.write("fork: %s\n" % (e,))
            os._exit(1)

        sys.stderr.write("Launched child %u\n" % (pid,))
        while True:
            try:
                pid, ret = os.waitpid(pid, 0)
                break
            except OSError as e:
                if e.errno == errno.EINTR:
                    continue
                sys.stderr.write("waitpid: %s\n" % (e,))
                os._exit(1)

        if os.WIFEXITED(ret):
            status = os.WEXITSTATUS(ret)
            sys.stderr.write("Child %u exited with status %d\n" % (pid, status))
            if status == 0:
                os._exit(0)
        elif os.WIFSIGNALED(ret):
            sig = os.WTERMSIG(ret)
            sys.stderr.write("Child %u exited with signal %u\n" % (pid, sig))
        else:
            sys.stderr.write("Child %u exited abnormally (%x)\n" % (pid, ret))

        time.sleep(restart_delay)
