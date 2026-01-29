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
import socket
import struct
import sys
import time
import traceback

import psutil
import torch

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

_IPC_PICKLER = multiprocessing.reduction.ForkingPickler(open(os.devnull, mode='wb'))


def get_infrastructure_rank(skip_nodename_logic: bool = False) -> int:
    """Get infrastructure rank from environment variables with SLURM validation.

    Returns infrastructure rank with the following precedence:
    1. NVRX_INFRA_RANK_FROM_NODENAME (if set and not skipped) - calculate rank by extracting all digits from SLURMD_NODENAME
       - Example: "nvl72134-T01" -> rank 7213401
    2. CROSS_SLURM_PROCID (for multi-job coordination)
    3. SLURM_TOPOLOGY_ADDR with block awareness (if SLURM_TOPOLOGY_ADDR_PATTERN is "block.node" and not skipped)
       - Parses format "blockX.nodeY" and calculates rank as X * multiplier + Y
       - Default multiplier is 10^10 (10 billion), reserving 10 digits for node numbers
       - This keeps block index in MSB for proper ordering with 64-bit integers
       - Can be overridden with SLURM_TOPOLOGY_NODES_PER_BLOCK env var
       - Raises ValueError if node number >= 10^10 (exceeds 10 digits)
       - Examples with default multiplier=10^10:
         * "block5.node3"   -> rank 5*10^10 + 3 = 50000000003
         * "block5.node9"   -> rank 5*10^10 + 9 = 50000000009
         * "block5.node10"  -> rank 5*10^10 + 10 = 50000000010
         * "block6.node2"   -> rank 6*10^10 + 2 = 60000000002
    4. SLURM_PROCID (set by SLURM), with job array support
    5. GROUP_RANK (fallback, set by launcher)

    For SLURM job arrays with one task per node, the infrastructure rank is calculated as:
        array_task_id * nnodes_per_array_task + slurm_procid
    This ensures unique ranks across all nodes in all array tasks.

    If none are set, returns -1 to indicate it should be assigned deterministically.

    Args:
        skip_nodename_logic: If True, skip the NVRX_INFRA_RANK_FROM_NODENAME and SLURM_TOPOLOGY_ADDR logic
                           and fall through to SLURM array task ID calculation. Default is False.

    Returns:
        int: Infrastructure rank (>=0) or -1 if not set

    Raises:
        RuntimeError: If SLURM_JOB_ID is set but neither CROSS_SLURM_PROCID nor SLURM_PROCID is defined
        ValueError: If NVRX_INFRA_RANK_FROM_NODENAME is set (and not skipped) but SLURMD_NODENAME
                   is not set or contains no digits
        ValueError: If SLURM_TOPOLOGY_ADDR_PATTERN is "block.node" (and not skipped) but SLURM_TOPOLOGY_ADDR
                   does not match expected format or parts contain no digits
        ValueError: If node number in SLURM_TOPOLOGY_ADDR exceeds 10 digits (>= 10^10)
    """
    # Check NVRX_INFRA_RANK_FROM_NODENAME first (for nodename-based rank calculation)
    if not skip_nodename_logic and os.getenv('NVRX_INFRA_RANK_FROM_NODENAME') is not None:
        nodename = os.getenv('SLURMD_NODENAME')
        if nodename is None:
            raise ValueError(
                "NVRX_INFRA_RANK_FROM_NODENAME is set but SLURMD_NODENAME environment variable is not set"
            )
        # Extract all digits from nodename
        digits = ''.join(c for c in nodename if c.isdigit())
        if not digits:
            raise ValueError(
                f"NVRX_INFRA_RANK_FROM_NODENAME is set but SLURMD_NODENAME '{nodename}' contains no digits"
            )
        infra_rank = int(digits)
        logger.debug(f"Using infrastructure rank {infra_rank} from SLURMD_NODENAME '{nodename}'")
        return infra_rank

    # Check CROSS_SLURM_PROCID second (for multi-job scenarios)
    cross_slurm_procid = os.getenv('CROSS_SLURM_PROCID')
    if cross_slurm_procid is not None:
        infra_rank = int(cross_slurm_procid)
        logger.debug(f"Using infrastructure rank {infra_rank} from CROSS_SLURM_PROCID")
        return infra_rank

    # Check SLURM_TOPOLOGY_ADDR with block awareness third
    if not skip_nodename_logic:
        topology_addr = os.getenv('SLURM_TOPOLOGY_ADDR')
        topology_pattern = os.getenv('SLURM_TOPOLOGY_ADDR_PATTERN')

        if (
            topology_addr is not None
            and topology_pattern is not None
            and topology_pattern.lower() == 'block.node'
        ):
            # Parse block.node format to extract block and node numbers separately
            # Format: blockX.nodeY -> rank = X * multiplier + Y
            # The multiplier ensures block index stays in MSB for proper ordering
            parts = topology_addr.split('.')
            if len(parts) != 2:
                raise ValueError(
                    f"SLURM_TOPOLOGY_ADDR_PATTERN is 'block.node' but SLURM_TOPOLOGY_ADDR '{topology_addr}' "
                    f"does not match expected format (expected 2 dot-separated parts, got {len(parts)})"
                )

            # Extract digits from block part
            block_digits = ''.join(c for c in parts[0] if c.isdigit())
            if not block_digits:
                raise ValueError(
                    f"SLURM_TOPOLOGY_ADDR_PATTERN is 'block.node' but block part '{parts[0]}' contains no digits"
                )
            block_num = int(block_digits)

            # Extract digits from node part
            node_digits = ''.join(c for c in parts[1] if c.isdigit())
            if not node_digits:
                raise ValueError(
                    f"SLURM_TOPOLOGY_ADDR_PATTERN is 'block.node' but node part '{parts[1]}' contains no digits"
                )
            node_num = int(node_digits)

            # Calculate multiplier to ensure block index stays in MSB
            # Default to 10^10 (10 billion) to reserve 10 digits for node numbers
            # This works with 64-bit integers and ensures proper ordering
            multiplier_env = os.getenv('SLURM_TOPOLOGY_NODES_PER_BLOCK')
            if multiplier_env is not None:
                multiplier = int(multiplier_env)
            else:
                multiplier = 10_000_000_000  # 10^10: reserves 10 digits for node numbers

            # Validate node number doesn't exceed 10 digits
            if node_num >= multiplier:
                raise ValueError(
                    f"Node number {node_num} exceeds maximum supported value {multiplier - 1}. "
                    f"Node numbers must fit within 10 digits when using default multiplier."
                )

            # Calculate rank: block * multiplier + node
            # This keeps block index in MSB for proper ordering
            infra_rank = block_num * multiplier + node_num
            logger.debug(
                f"Using infrastructure rank {infra_rank} from SLURM_TOPOLOGY_ADDR '{topology_addr}' "
                f"(block={block_num}, node={node_num}, multiplier={multiplier}) with pattern '{topology_pattern}'"
            )
            return infra_rank

    # Get SLURM_PROCID once and reuse it
    slurm_procid = os.getenv('SLURM_PROCID')

    # Check if we're running in a SLURM job array
    slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')

    if slurm_array_task_id is not None and slurm_procid is not None:
        # In a SLURM job array with one task per node, calculate global rank across all array tasks
        # based on node count rather than process count
        array_task_id = int(slurm_array_task_id)
        proc_id = int(slurm_procid)

        # Get the number of nodes per array task
        # SLURM_NNODES is the number of nodes allocated to the current job step
        nnodes_per_array = os.getenv('SLURM_NNODES', os.getenv('SLURM_JOB_NUM_NODES'))
        if nnodes_per_array is None:
            # If SLURM_NNODES is not set, we can't compute the offset
            # This should not happen in a properly configured SLURM array job
            raise RuntimeError(
                "SLURM_ARRAY_TASK_ID is set but SLURM_NNODES/SLURM_JOB_NUM_NODES is not defined. "
                "Cannot calculate infrastructure rank for job array. "
                "Ensure the job array is properly configured."
            )

        nnodes = int(nnodes_per_array)

        # For one launcher per node, SLURM_PROCID should match SLURM_NODEID (local node ID within job)
        # Calculate global infrastructure rank: array_task_id * nodes_per_task + local_node_id
        infra_rank = array_task_id * nnodes + proc_id
        logger.debug(
            f"Using infrastructure rank {infra_rank} from SLURM job array "
            f"(array_task_id={array_task_id}, nnodes={nnodes}, procid={proc_id})"
        )
        return infra_rank

    # Try SLURM_PROCID (already retrieved), then fall back to GROUP_RANK (set by launcher)
    infra_rank_str = slurm_procid if slurm_procid is not None else os.getenv('GROUP_RANK')

    if infra_rank_str is not None:
        infra_rank = int(infra_rank_str)
        logger.debug(f"Using infrastructure rank {infra_rank} from environment")
        return infra_rank

    # Check if we're running under SLURM - if so, SLURM_PROCID should be defined
    # (unless CROSS_SLURM_PROCID was set, which we already handled)
    if os.getenv('SLURM_JOB_ID') is not None:
        raise RuntimeError(
            "SLURM_JOB_ID is set but neither CROSS_SLURM_PROCID nor SLURM_PROCID is defined. "
            "This indicates a SLURM deployment error. "
            "SLURM_PROCID should be automatically set by SLURM for each task."
        )

    # Neither env var is set - will be assigned deterministically later
    logger.debug(
        "Neither SLURM_PROCID nor GROUP_RANK is set. Infrastructure rank will be assigned deterministically."
    )
    return -1


def is_slurm_job_array() -> bool:
    """Check if the current job is running in a SLURM job array.

    Returns:
        bool: True if running in a SLURM job array (SLURM_ARRAY_TASK_ID is set), False otherwise
    """
    return os.getenv('SLURM_ARRAY_TASK_ID') is not None


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


def install_exception_handler():
    """
    Install a custom exception handler to capture uncaught exceptions in training worker processes.

    When an uncaught exception occurs:
    1. Formats and logs the complete traceback
    2. Uses os._exit() to reliably terminate the process

    This ensures that exceptions are properly captured and logged, and the process exits
    reliably without running cleanup handlers that might hang or interfere with fault tolerance.
    """

    def exception_handler(exc_type, exc_value, exc_traceback):
        """
        Custom exception handler that logs the exception and exits reliably.

        Args:
            exc_type: The type of the exception
            exc_value: The exception instance
            exc_traceback: The traceback object
        """
        # Don't log KeyboardInterrupt - these are intentional
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Format the complete traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)

        # Get rank information for better debugging
        rank = get_rank()
        rank_str = f"Rank {rank}" if rank is not None else "Unknown rank"

        # Log the exception with full traceback
        error_msg = (
            f"\n{rank_str}: UNCAUGHT EXCEPTION in training worker process\n"
            f"{tb_text}"
            f"{rank_str}: Process will exit with code 1\n"
        )

        # Get logger from application context (not module-level logger)
        # Use root logger to ensure we capture in application's logging context
        app_logger = logging.getLogger()
        app_logger.error(error_msg)

        # Also print to stderr to ensure visibility
        print(error_msg, file=sys.stderr, flush=True)

        # Use os._exit() to terminate immediately without cleanup
        # This is more reliable than sys.exit() which raises SystemExit
        # and can be caught by exception handlers
        os._exit(1)

    # Install the custom exception handler
    sys.excepthook = exception_handler
