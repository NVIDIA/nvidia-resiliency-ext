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

import contextlib
import gc
import os
import socket
import sys
import tempfile

from nvidia_resiliency_ext.common.device_utils import get_distributed_init_method
import torch


def assert_fn(arg):
    assert arg


def is_port_open(host, port):
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


def find_free_port(host="127.0.0.1"):
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        _, port = sock.getsockname()
        return port


def setup_distributed_worker_tcp_store(
    rank,
    world_size,
    queue,
    barrier,
    backend,
):
    host = "127.0.0.1"
    port = queue.get()

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = str(host)
    os.environ["MASTER_PORT"] = str(port)

    barrier.wait()
    port_open = is_port_open(host, port)
    assert not port_open, (host, port)
    barrier.wait()

    torch.distributed.init_process_group(backend, 
                                         init_method=get_distributed_init_method())


def setup_distributed_worker_file_store(
    rank,
    world_size,
    queue,
    barrier,
    backend,
):
    store_file = queue.get()

    # webdataset needs these to properly split shards between ranks
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # FIXME needed by RankMonitorServer

    barrier.wait()
    torch.distributed.init_process_group(
        backend,
        init_method=f"file://{store_file}",
        world_size=world_size,
        rank=rank,
    )


def distributed_worker(
    rank,
    world_size,
    dist_store_type,
    queue,
    barrier,
    ready_flag,
    worker_fn,
    backend,
    **kwargs,
):
    store_types = ["tcp", "file"]
    assert dist_store_type in store_types

    if dist_store_type == "tcp":
        setup_distributed_worker_tcp_store(rank, world_size, queue, barrier, backend)
    elif dist_store_type == "file":
        setup_distributed_worker_file_store(rank, world_size, queue, barrier, backend)

    rank = torch.distributed.get_rank()

    if backend == "nccl":
        torch.cuda.set_device(rank)

    ready_flag.set()

    worker_fn(**kwargs)

    # `destroy_process_group` hangs were observed in CI
    # use GC collect and barrier to mitigate the issue
    gc.collect()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    sys.exit(0)


def multiprocessing_execute_start(
    worker_fn,
    mp_ctx,
    world_size=2,
    dist_store_type="file",
    backend="nccl",
    **kwargs,
):
    processes = []
    queue = mp_ctx.Queue()
    barrier = mp_ctx.Barrier(world_size)
    ready_flag = mp_ctx.Event()
    for rank in range(world_size):
        p = mp_ctx.Process(
            target=distributed_worker,
            args=(
                rank,
                world_size,
                dist_store_type,
                queue,
                barrier,
                ready_flag,
                worker_fn,
                backend,
            ),
            kwargs=kwargs,
        )
        processes.append(p)
        p.start()

    store_types = ["tcp", "file"]
    assert dist_store_type in store_types

    if dist_store_type == "tcp":
        port = find_free_port()
        for rank in range(world_size):
            queue.put(port)
    elif dist_store_type == "file":
        # torch.distributed will delete the file, we just need an unique name
        with tempfile.NamedTemporaryFile(delete=True) as tmpf:
            dist_store_file = tmpf.name
        for rank in range(world_size):
            queue.put(dist_store_file)

    ready_flag.wait(timeout=120)
    assert ready_flag.is_set()

    return processes


def multiprocessing_execute_join(processes, timeout):
    for p in processes:
        p.join(timeout=timeout)
    for p in processes:
        if p.is_alive():
            raise Exception(f"Process {p} is still alive. Waited {timeout}")
    exit_codes = [p.exitcode for p in processes]
    return exit_codes
