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
import logging
import multiprocessing
import os
import signal
import socket
import struct
import sys
import time

import psutil
import torch

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

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
    Terminate ALL processes in the worker process groups.

    This is called after pcontext.close() has already attempted killpg on worker processes.
    Any surviving processes are stragglers (e.g., orphaned children, multiprocessing helpers,
    checkpoint writers, data loaders) that need aggressive cleanup.

    Since worker PGIDs are specifically created for workers and their descendants, it's safe
    to terminate everything in these process groups. By the time this runs, the main worker
    processes should already be dead from pcontext.close().

    Args:
        allowed_ppids: Legacy parameter, kept for compatibility
        allowed_pgids: Process group IDs of worker processes to clean up

    The function attempts two strategies:
    1. killpg on each PGID with SIGKILL (most efficient, kills entire group at once)
    2. Enumerate survivors with psutil and kill individually (catches edge cases)
    """

    if 1 in allowed_pgids:
        # Workaround for sim-multinode tests, where rank monitors have PGID=1 and gets terminated.
        logger.debug("Skipping terminate_mp_processes: PGID 1 in allowed_pgids")
        return

    killed_count = 0
    survivor_count = 0

    # Strategy 1: Try killpg on each process group (most efficient)
    for pgid in allowed_pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
            logger.debug(f"Sent SIGKILL to entire PGID {pgid}")
            killed_count += 1
        except ProcessLookupError:
            # Expected: No processes in this group (already cleaned up)
            logger.debug(f"PGID {pgid} already empty")
        except PermissionError as e:
            # Unexpected: Should not happen for same-UID processes
            # Log as warning to investigate potential container/security module issues
            logger.warning(f"Permission denied killing PGID {pgid}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error killing PGID {pgid}: {e}")

    # Strategy 2: Enumerate any survivors and kill individually
    # This catches processes that might have changed PGIDs or other edge cases
    try:
        all_processes = psutil.process_iter(attrs=['pid', 'ppid', 'name', 'cmdline'])
        for process in all_processes:
            try:
                proc_pgid = os.getpgid(process.info['pid'])
                if proc_pgid in allowed_pgids:
                    survivor_count += 1
                    # Get cmdline for logging (may be None for orphaned processes)
                    cmdline = process.info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline) if cmdline else 'N/A'

                    logger.debug(
                        f"Force killing survivor process: PID={process.info['pid']}, "
                        f"PPID={process.info['ppid']}, PGID={proc_pgid}, "
                        f"name={process.info['name']}, cmdline={cmdline_str[:100]}"
                    )
                    process.kill()  # SIGKILL, not terminate (can't be caught)
            except psutil.NoSuchProcess:
                # Expected: Process died between enumeration and kill (race condition)
                pass
            except (psutil.AccessDenied, OSError) as e:
                # Unexpected but non-fatal: Log for visibility
                # Note: AccessDenied shouldn't happen for same-UID processes but psutil
                # can raise it for various reasons (kernel restrictions, zombies, etc.)
                logger.debug(f"Could not kill PID={process.info.get('pid')}: {e}")
            except Exception as e:
                # Catch-all for other unexpected errors
                logger.debug(f"Unexpected error checking/killing process: {e}")

    except Exception as e:
        logger.error(f"Error in terminate_mp_processes survivor enumeration: {e}")

    # Log results: killed_count=0 and survivor_count=0 is the ideal case
    # (means pcontext.close() already cleaned up everything)
    if killed_count > 0 or survivor_count > 0:
        logger.debug(
            f"terminate_mp_processes: killed {killed_count} process groups, "
            f"found and terminated {survivor_count} individual survivors"
        )


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
        raise


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


def get_processes_by_pgids(pgids, exclude_launcher=True):
    """
    Find all processes belonging to specific process groups.
    This catches workers and ALL their descendants (including checkpoint writers, etc.).

    Args:
        pgids: Set or list of process group IDs to find
        exclude_launcher: If True, exclude the launcher process itself

    Returns:
        List of dicts containing process info (pid, ppid, pgid, name, cmdline, status)
    """
    try:
        launcher_pid = os.getpid()
        processes = []

        for proc in psutil.process_iter(attrs=['pid', 'ppid', 'name', 'cmdline', 'status']):
            try:
                pid = proc.info['pid']

                # Skip launcher itself
                if exclude_launcher and pid == launcher_pid:
                    continue

                # Skip zombie processes (normal during cleanup)
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    continue

                # Get process group ID
                proc_pgid = os.getpgid(pid)

                if proc_pgid in pgids:
                    proc_info = proc.info.copy()
                    proc_info['pgid'] = proc_pgid
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

        return processes
    except Exception:
        return []


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
