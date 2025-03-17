# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import contextlib
import ctypes
import multiprocessing
import os
import socket
import struct
import sys
import time

import psutil
import torch

_IPC_PICKLER = multiprocessing.reduction.ForkingPickler(open(os.devnull, mode='wb'))


def is_process_alive(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and not process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wait_until_process_terminated(pid, timeout=0):
    sleep_time = 0.1
    remaining_time = timeout
    wait_forever = timeout == 0
    while (remaining_time > 0 or wait_forever) and is_process_alive(pid):
        remaining_time -= sleep_time
        time.sleep(sleep_time)
    if is_process_alive(pid):
        raise Exception(f"wait_until_process_terminated: {pid} is alive after {timeout} seconds")


def wait_for_mp_events(events, timeout=60):
    """
    Wait until all events are set
    """
    remaining_time = timeout
    for ev in events:
        started = time.monotonic()
        is_ev_set = ev.wait(remaining_time)
        elapsed = time.monotonic() - started
        remaining_time = max(0, remaining_time - elapsed)
        if not is_ev_set:
            raise RuntimeError(f"Not all events ready after {timeout} seconds")


def terminate_mp_processes(allowed_ppids, allowed_pgids):
    """
    Terminate auxiliary processes spawned by the `multiprocessing` package.

    If an worker is terminated with a signal (e.g. TERM) and *it has no sig handler installed*,
    there are some leftover processes spawned by `multiprocessing`:
      - `python -c from multiprocessing.spawn import spawn_main; spawn_main (...)`
      - `python -c from multiprocessing.resource_tracker (...)`

    Such leftover processes can block the tests, e.g., .commuminicate() in `subprocess` hangs
    if there are some subprocesses alive after the main process is terminated.

    We don't want to terminate the processes that might be used elsewhere in the system,
    so we use `allowed_ppids` and `allowed_pgids` to find processes that are related to the fault tolerance.

    Args:
        allowed_ppids: what parent PIDs that are allowed
        allowed_pgids: what process group IDs that are allowed

    Swallow any exceptions, as this is not critical functionality.
    NOTE: this is workaround, it should be removed after we eliminate  `Manager()` from
    `ParametersUpdateManager`
    """

    if 1 in allowed_pgids:
        # workaround for sim-multinode tests, where rank monitors have PGID=1 and gets terminated.
        return

    try:
        all_processes = psutil.process_iter(attrs=['pid', 'ppid', 'name', 'cmdline'])
        for process in all_processes:
            try:
                ppid_match = process.ppid() in allowed_ppids
                pgid_match = os.getpgid(process.pid) in allowed_pgids
                if ppid_match and pgid_match:
                    cmd_line = " ".join(process.cmdline())
                    patt1 = "from multiprocessing.resource_tracker import main"
                    patt2 = "from multiprocessing.spawn import spawn_main"
                    if patt1 in cmd_line or patt2 in cmd_line:
                        process.terminate()
            except Exception:
                pass
    except Exception:
        pass


def set_ipc_socket_timeouts(fileno, timeout):
    # NOTE: this assumes that platform is 64bit
    sock = socket.fromfd(fileno, socket.AF_UNIX, socket.SOCK_STREAM)
    num_seconds = int(timeout)
    num_useconds = int((timeout - int(timeout)) * 1e6)
    timeval = struct.pack("ll", num_seconds, num_useconds)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, timeval)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeval)


async def read_obj_from_ipc_stream(stream: asyncio.StreamReader):
    """
    Helper for reading pickled objects from stream
    Returns unpickled obect or None if an error happened
    """
    try:
        obj_size_as_bytes = await stream.readexactly(4)
        obj_pickled_size = int.from_bytes(obj_size_as_bytes, byteorder="big")
        obj_pickled = await stream.readexactly(obj_pickled_size)
        return _IPC_PICKLER.loads(obj_pickled)
    except (asyncio.IncompleteReadError, Exception):
        # print(f"Error in _read_obj_from_stream: {e}", file=sys.stderr)
        return None


async def write_obj_to_ipc_stream(obj, stream: asyncio.StreamWriter):
    """
    Helper for writing pickled objects to stream
    """
    try:
        obj_pickled = _IPC_PICKLER.dumps(obj)
        obj_size_as_bytes = len(obj_pickled).to_bytes(length=4, byteorder="big")
        stream.write(obj_size_as_bytes)
        stream.write(obj_pickled)
        await stream.drain()
    except Exception:
        # print(f"Error in _write_obj_to_stream: {e}", file=sys.stderr)
        raise  # Might need to do something about it


def recv_all(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket connection broken while receiving")
        data += packet
    return data


def read_obj_from_ipc_socket(sock, raise_exc=False):
    try:
        obj_size_as_bytes = recv_all(sock, 4)
        obj_pickled_size = int.from_bytes(obj_size_as_bytes, byteorder="big")
        obj_pickled = recv_all(sock, obj_pickled_size)
        return _IPC_PICKLER.loads(obj_pickled)
    except Exception as e:
        if raise_exc:
            raise
        else:
            print(f"Exception while read_obj_from_ipc_socket: {e}", file=sys.stderr)
            return None


def write_object_to_ipc_socket(obj, sock):
    obj_pickled = _IPC_PICKLER.dumps(obj)
    obj_size_as_bytes = len(obj_pickled).to_bytes(length=4, byteorder="big")
    sock.sendall(obj_size_as_bytes)
    sock.sendall(obj_pickled)


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return os.getenv("RANK")


def reduce_cuda_ctx_size():
    # This needs to be called before CUDA context is initialized
    # It will reduce CUDA per-thread stack size to bare minimum
    # should not be a problem, if we do not run any kernels
    try:
        cuda = ctypes.CDLL('libcudart.so')
        cuda.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
        cuda.cudaDeviceSetLimit.restype = ctypes.c_int
        # 0x00 is cudaLimitStackSize, try to set it to smallest reasonable value (4)
        cuda.cudaDeviceSetLimit(0x00, 4)
    except Exception as _:
        # ignore exception, can continue if this fails
        pass


@contextlib.contextmanager
def patched_method(obj, method_name, new_method):
    """
    Temporarily patch `method_name` on `obj` with `new_method`.
    Restores the original method upon exiting the context.
    """
    # Save the original method
    original_method = getattr(obj, method_name)

    # Patch the method
    setattr(obj, method_name, new_method)

    try:
        yield
    finally:
        # Restore the original method
        setattr(obj, method_name, original_method)
