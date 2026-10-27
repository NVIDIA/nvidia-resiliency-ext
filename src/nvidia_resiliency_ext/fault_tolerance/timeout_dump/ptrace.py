# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ptrace authorization for the rank monitor."""

import ctypes
import os
import socket
import struct


def authorize_monitor_ptrace(sock):
    """Allow this socket's monitor, and its descendants, under Linux Yama mode 1."""
    pid, uid, _ = struct.unpack(
        "iII", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("iII"))
    )
    if pid <= 0 or uid != os.geteuid():
        raise PermissionError("rank monitor has an unexpected PID or UID")
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PTRACER changes this worker's exception, not the node's ptrace policy.
    if libc.prctl(0x59616D61, ctypes.c_ulong(pid), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
