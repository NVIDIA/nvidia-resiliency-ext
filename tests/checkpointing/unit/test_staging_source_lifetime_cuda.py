# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA IPC regression: release source storage while filesystem writes are blocked."""

import gc
import os
import weakref
from functools import partial

import pytest
import torch
from torch.distributed.checkpoint import DefaultSavePlanner, FileSystemReader, load

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync

from .test_staging_source_lifetime import _BlockingOpen, _finalize_checkpoint, _wait_for_file

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not torch.cuda.is_available(),
    reason="requires Linux and CUDA IPC; CPU tests are in test_staging_source_lifetime.py",
)


def _schedule_cuda_checkpoint(queue, checkpoint_dir, started, release):
    # All application-side tensor, state_dict, planner and request references leave
    # scope on return. Only weak references to the writer's actual source survive.
    source = torch.full((8 * 1024 * 1024,), 3.5, device="cuda", dtype=torch.float32)
    planner = DefaultSavePlanner()
    planner.set_up_planner({"value": source}, is_coordinator=True)
    writer = FileSystemWriterAsync(
        checkpoint_dir,
        open_file=_BlockingOpen(started, release),
        use_cached_data_structure=False,
        use_cpu_shm_for_gpu_tensors=False,
    )
    writer.set_up_storage_writer(True)
    local_plan = writer.prepare_local_plan(planner.create_local_plan())
    plans, metadata = planner.create_global_plan([local_plan])
    plans = writer.prepare_global_plan(plans)
    writer.prepare_write_data(planner.finish_plan(plans[0]), planner)

    # prepare_write_data() detaches CUDA tensors. A weakref to the original wrapper
    # alone would miss retained storage aliases, so observe the detached wrapper.
    source_ref = weakref.ref(writer.cached_tensor_data[1][0])
    allocated_with_source = torch.cuda.memory_allocated()
    write_fn, preload_fn, args = writer.get_save_function_and_args()
    request = AsyncRequest(
        write_fn, args, [partial(_finalize_checkpoint, writer, metadata)], preload_fn=preload_fn
    )
    call_idx = queue.schedule_async_request(request)
    return source_ref, allocated_with_source, call_idx


@pytest.mark.parametrize("is_daemon", [True, False])
def test_cuda_source_storage_released_before_filesystem_write(tmp_path, monkeypatch, is_daemon):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    queue = AsyncCallsQueue(is_daemon=is_daemon, cpu_priority=0)
    started, release = tmp_path / "write_started", tmp_path / "allow_write"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    finalized = False
    try:
        # Warm the worker before measuring the source allocation.
        caller = queue._get_async_caller()
        caller._start_worker(rank)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        baseline = torch.cuda.memory_allocated()

        source_ref, allocated_with_source, call_idx = _schedule_cuda_checkpoint(
            queue, checkpoint_dir, started, release
        )
        _wait_for_file(started)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        allocated_after_staging = torch.cuda.memory_allocated()

        # Both staging and ownership release must precede persistence completion.
        assert not release.exists()
        assert queue.get_num_unfinalized_calls() == 1
        assert queue.maybe_finalize_async_calls(no_dist=True) == []
        assert source_ref() is None
        assert allocated_with_source - baseline >= 32 * 1024 * 1024
        # Allow small context/IPC bookkeeping allocations; a retained 32 MiB source
        # must fail this check. Do not test memory_reserved() or nvidia-smi values.
        assert allocated_after_staging <= baseline + 4 * 1024 * 1024

        release.touch()
        assert queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [call_idx]
        finalized = True
        restored = {"value": torch.empty(8 * 1024 * 1024, dtype=torch.float32)}
        load(restored, storage_reader=FileSystemReader(checkpoint_dir), no_dist=True)
        assert torch.all(restored["value"] == 3.5)
    finally:
        release.touch()
        # Abort on failure so a failed IPC/preload operation cannot hang teardown.
        queue.close(abort=not finalized)
        queue.async_calls.clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
