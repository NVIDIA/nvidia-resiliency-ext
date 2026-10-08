# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source ownership ends at staging; persistence and finalization still work."""

import gc
import weakref
from collections import namedtuple
from dataclasses import replace
from functools import partial
from pathlib import Path
from queue import Queue
from time import monotonic, sleep
from unittest.mock import Mock

import pytest
import torch
from torch.distributed.checkpoint import DefaultSavePlanner, FileSystemReader, load
from torch.distributed.checkpoint.filesystem import _StoragePrefix
from torch.distributed.checkpoint.utils import _DistWrapper

from nvidia_resiliency_ext.checkpointing.async_ckpt import core, filesystem_async
from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync


class _Payload:
    def __init__(self, value=42):
        self.value = value


def _preload_payload(payload):
    return payload.value


def _noop(*args, **kwargs):
    pass


class _PendingCaller:
    """A caller that accepts execution without retaining the trainer's request."""

    def __init__(self):
        self.done = False
        self.indices = []

    def schedule_async_call(self, request):
        assert request.finalize_fns == []
        self.indices.append(request.call_idx)

    def is_current_async_call_done(self, blocking, no_dist):
        return self.done

    def close(self, abort=False):
        pass


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize(
    "execution_field", ["async_fn", "async_fn_args", "async_fn_kwargs", "preload_fn"]
)
def test_active_queue_releases_execution_ownership(monkeypatch, persistent, execution_field):
    queue = AsyncCallsQueue(persistent=persistent)
    caller = _PendingCaller()
    monkeypatch.setattr(queue, "_get_async_caller", lambda: caller)
    finalized = []
    try:
        for expected_idx in range(2):
            payload = _Payload()
            payload_ref = weakref.ref(payload)
            execution = {
                "async_fn": partial(_noop, payload),
                "async_fn_args": (payload,),
                "async_fn_kwargs": {"source": payload},
                "preload_fn": partial(_preload_payload, payload),
            }[execution_field]
            request = AsyncRequest(_noop, (), [partial(finalized.append, expected_idx)])
            request = request._replace(**{execution_field: execution})
            assert queue.schedule_async_request(request) == expected_idx
            del request, execution, payload
            gc.collect()

            assert payload_ref() is None
            active = queue.async_calls[-1].async_request
            assert active.is_frozen
            assert active.call_idx == expected_idx
            assert active.async_fn is None
            assert active.async_fn_args == ()
            assert active.async_fn_kwargs is None
            assert active.preload_fn is None

        assert queue.maybe_finalize_async_calls(no_dist=True) == []
        assert finalized == []
        caller.done = True
        assert queue.maybe_finalize_async_calls(no_dist=True) == [0, 1]
        assert finalized == [0, 1]
        assert caller.indices == [0, 1]
    finally:
        queue.close(abort=True)
        queue.async_calls.clear()


def test_old_request_and_noop_keep_finalizers(monkeypatch):
    old_request_type = namedtuple("OldAsyncRequest", "async_fn async_fn_args finalize_fns")
    finalized = []
    queue = AsyncCallsQueue()
    caller = _PendingCaller()
    monkeypatch.setattr(queue, "_get_async_caller", lambda: caller)
    try:
        request = old_request_type(None, (), [partial(finalized.append, "noop")])
        queue.schedule_async_request(request)
        caller.done = True
        assert queue.maybe_finalize_async_calls(no_dist=True) == [0]
        assert finalized == ["noop"]
    finally:
        queue.close(abort=True)


def test_temporal_caller_keeps_staged_data_until_child_finishes(monkeypatch):
    class ChildProcess:
        def __init__(self, target, args, kwargs):
            self.args = args

        def start(self):
            # multiprocessing.Process.start() drops its parent-side args too.
            del self.args

        def is_alive(self):
            return True

        def join(self):
            pass

    context = Mock()
    context.Process = ChildProcess
    monkeypatch.setattr(core.mp, "get_context", lambda method: context)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    queue = AsyncCallsQueue(persistent=False)
    source = _Payload()
    source_ref = weakref.ref(source)

    def stage(payload):
        return torch.tensor(payload.value)

    try:
        request = AsyncRequest(_noop, (0, None), [], preload_fn=partial(stage, source))
        queue.schedule_async_request(request)
        caller = queue.async_calls[0].async_caller
        staged_ref = weakref.ref(caller.preloaded_holder)
        del request, source
        gc.collect()
        assert source_ref() is None
        assert staged_ref() is not None
        assert queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [0]
        gc.collect()
        assert staged_ref() is None
    finally:
        queue.maybe_finalize_async_calls(blocking=True, no_dist=True)
        queue.close(abort=True)
        queue.async_calls.clear()


@pytest.mark.parametrize("execution_field", ["async_fn", "async_fn_args", "async_fn_kwargs"])
@pytest.mark.parametrize("abort", [False, True])
def test_temporal_caller_without_preload_keeps_execution_alive(monkeypatch, execution_field, abort):
    class ChildProcess:
        def __init__(self, target, args, kwargs):
            self.execution = (target, args, kwargs)

        def start(self):
            # The parent Process object does not own these after fork().
            del self.execution

        def is_alive(self):
            return True

        def join(self):
            assert source_ref() is not None

        def kill(self):
            assert source_ref() is not None

    context = Mock()
    context.Process = ChildProcess
    monkeypatch.setattr(core.mp, "get_context", lambda method: context)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    queue = AsyncCallsQueue(persistent=False)
    source = torch.tensor(42)
    source_ref = weakref.ref(source)
    execution = {
        "async_fn": partial(_noop, source),
        "async_fn_args": (source,),
        "async_fn_kwargs": {"source": source},
    }[execution_field]
    request = AsyncRequest(_noop, (), [])._replace(**{execution_field: execution})
    try:
        queue.schedule_async_request(request)
        del source, execution, request
        gc.collect()
        # Reclaiming parent-owned pinned host tensors before the child finishes
        # can invalidate the child's COW pages (the same lifetime addressed by #291).
        assert source_ref() is not None
        assert queue.maybe_finalize_async_calls(no_dist=True) == []
        if abort:
            queue.async_calls[0].async_caller.close(abort=True)
        assert queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [0]
        gc.collect()
        assert source_ref() is None
    finally:
        queue.maybe_finalize_async_calls(blocking=True, no_dist=True)
        queue.close(abort=True)
        queue.async_calls.clear()


def test_schedule_failure_does_not_enqueue_finalization(monkeypatch):
    queue = AsyncCallsQueue()
    caller = _PendingCaller()
    monkeypatch.setattr(queue, "_get_async_caller", lambda: caller)
    monkeypatch.setattr(
        caller, "schedule_async_call", Mock(side_effect=RuntimeError("schedule failed"))
    )
    try:
        with pytest.raises(RuntimeError, match="schedule failed"):
            queue.schedule_async_request(AsyncRequest(_noop, (), []))
        assert queue.get_num_unfinalized_calls() == 0
    finally:
        queue.close(abort=True)


def _prepare_cpu_writer(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    # Keep these unit tests independent of multiprocessing manager startup.
    monkeypatch.setattr(filesystem_async, "get_write_results_queue", Queue)
    planner = DefaultSavePlanner()
    planner.set_up_planner({"value": torch.arange(8), "step": 3}, is_coordinator=True)
    plan = replace(planner.create_local_plan(), storage_data=_StoragePrefix("__0_"))
    writer = FileSystemWriterAsync(tmp_path, **kwargs)
    writer.prepare_write_data(plan, planner)
    return writer, plan, planner


@pytest.mark.parametrize("use_cpu_shm", [False, True])
def test_writer_transfers_payload_and_still_writes(tmp_path, monkeypatch, use_cpu_shm):
    writer, plan, planner = _prepare_cpu_writer(
        tmp_path, monkeypatch, use_cpu_shm_for_gpu_tensors=use_cpu_shm
    )
    group = writer.cached_tensor_data or writer.uncached_tensor_data
    snapshot_ref = weakref.ref(group[1][0])
    byte_io_ref = weakref.ref(writer.byte_io_data[1][0])
    del group
    results_queue = writer.results_queue

    write_fn, preload_fn, args = writer.get_save_function_and_args()
    assert writer.cached_tensor_data is None
    assert writer.uncached_tensor_data is None
    assert writer.byte_io_data is None
    assert writer.results_queue is results_queue
    assert writer.has_data_to_write
    assert snapshot_ref() is not None
    assert byte_io_ref() is not None

    buckets = preload_fn()
    write_fn(args[0], buckets, args[2])
    results = writer.retrieve_write_results()
    assert len(results) == len(plan.items)
    assert list(tmp_path.glob("*.distcp"))

    del buckets, preload_fn
    gc.collect()
    assert snapshot_ref() is None
    assert byte_io_ref() is None
    # The writer retained for finalization can be prepared for another checkpoint.
    writer.prepare_write_data(plan, planner)
    _, next_preload, _ = writer.get_save_function_and_args()
    assert next_preload()


def test_writer_rejects_double_payload_transfer(tmp_path, monkeypatch):
    writer, _, _ = _prepare_cpu_writer(tmp_path, monkeypatch)
    writer.get_save_function_and_args()
    with pytest.raises(RuntimeError, match="already been transferred"):
        writer.get_save_function_and_args()


def test_empty_writer_is_a_noop(tmp_path):
    writer = FileSystemWriterAsync(tmp_path)
    assert writer.get_save_function_and_args() == (None, None, [])
    assert writer.retrieve_write_results() == []


class _SourceFenceQueue(Queue):
    def __init__(self, source_ref):
        super().__init__()
        self.source_ref = source_ref

    def task_done(self):
        # Verify release BEFORE the preload-completion signal, not after writing.
        assert self.source_ref() is None
        super().task_done()


def _run_worker(monkeypatch, requests, preload_queue, completion_queue):
    monkeypatch.setattr(core, "_set_process_qos", lambda **kwargs: None)
    monkeypatch.setattr(core.signal, "signal", lambda *args: None)
    monkeypatch.setattr(core.telemetry, "setup_telemetry", lambda *args: None)
    monkeypatch.setattr(core.telemetry, "shutdown", lambda *args: None)
    core.PersistentAsyncCaller.async_process_target(
        0, requests, preload_queue, completion_queue, cpu_shm_mode=True
    )


def test_worker_drops_request_before_signaling_preload(monkeypatch):
    source = _Payload()
    source_ref = weakref.ref(source)
    preload_queue = _SourceFenceQueue(source_ref)
    preload_queue.put(7)
    requests, completions = Queue(), Queue()
    written = []

    def write(rank, snapshot, *, label):
        assert preload_queue.unfinished_tasks == 0
        assert source_ref() is None
        assert completions.empty()
        written.append((rank, snapshot, label))

    # Include the old-style source argument to ensure deleting the whole request
    # also releases aliases held outside preload_fn.
    requests.put(
        AsyncRequest(
            write,
            (0, source),
            [],
            {"label": "snapshot"},
            partial(_preload_payload, source),
            call_idx=7,
        )
    )
    requests.put("DONE")
    del source
    _run_worker(monkeypatch, requests, preload_queue, completions)
    assert written == [(0, 42, "snapshot")]
    assert completions.get_nowait() == 7
    assert requests.unfinished_tasks == 0


def test_worker_without_preload_preserves_execution_arguments(monkeypatch):
    requests, preloads, completions = Queue(), Queue(), Queue()
    written = []
    requests.put(AsyncRequest(partial(written.append), (42,), [], call_idx=3))
    requests.put(AsyncRequest(None, (), [], call_idx=4))
    requests.put("DONE")
    _run_worker(monkeypatch, requests, preloads, completions)
    assert written == [42]
    assert [completions.get_nowait(), completions.get_nowait()] == [3, 4]
    assert preloads.unfinished_tasks == 0
    assert requests.unfinished_tasks == 0


def test_preload_failure_does_not_signal_success(monkeypatch):
    def fail_preload():
        raise RuntimeError("preload failed")

    requests, preloads, completions = Queue(), Queue(), Queue()
    write = Mock()
    requests.put(AsyncRequest(write, (0, None), [], preload_fn=fail_preload))
    preloads.put(0)
    with pytest.raises(RuntimeError, match="preload failed"):
        _run_worker(monkeypatch, requests, preloads, completions)
    write.assert_not_called()
    assert preloads.unfinished_tasks == 1
    assert completions.empty()


def test_write_failure_does_not_signal_completion(monkeypatch):
    def fail_write(*args):
        raise RuntimeError("write failed")

    requests, preloads, completions = Queue(), Queue(), Queue()
    requests.put(AsyncRequest(fail_write, (), []))
    with pytest.raises(RuntimeError, match="write failed"):
        _run_worker(monkeypatch, requests, preloads, completions)
    assert requests.unfinished_tasks == 1
    assert completions.empty()


def test_intentional_worker_cache_survives_payload_transfer(tmp_path, monkeypatch):
    writer, _, _ = _prepare_cpu_writer(tmp_path, monkeypatch)
    identifier = filesystem_async.ConsistentDataIdentifier("source-lifetime-test")
    writer.consistent_data_identifier = identifier
    writer.cached_tensor_data = writer.uncached_tensor_data
    writer.uncached_tensor_data = None
    source_ref = weakref.ref(writer.cached_tensor_data[1][0])
    try:
        _, preload, _ = writer.get_save_function_and_args()
        buckets = preload()
        del buckets, preload
        gc.collect()
        assert source_ref() is not None
        assert identifier.key in core.PersistentAsyncCaller._worker_data_cache
    finally:
        core.PersistentAsyncCaller._worker_data_cache.pop(identifier.key, None)
    gc.collect()
    assert source_ref() is None


def test_cpu_checkpoint_roundtrip_after_payload_transfer(tmp_path, monkeypatch):
    """Exercise real DCP planning, writing and metadata after dropping source locals."""
    writer, _, planner = _prepare_cpu_writer(tmp_path, monkeypatch)
    local_plan = planner.create_local_plan()
    plans, metadata = planner.create_global_plan([local_plan])
    plans = writer.prepare_global_plan(plans)
    plan = planner.finish_plan(plans[0])
    writer.prepare_write_data(plan, planner)
    write, preload, args = writer.get_save_function_and_args()
    del plan, plans, local_plan, planner
    buckets = preload()
    del preload
    write(args[0], buckets, args[2])
    del buckets
    gc.collect()

    # Single-rank finalization without a CUDA-dependent distributed broadcast.
    dist_wrapper = _DistWrapper(None, False, 0)
    dist_wrapper.all_reduce(
        "write", writer.retrieve_write_results, partial(writer.finish, metadata)
    )
    restored = {"value": torch.empty(8, dtype=torch.int64), "step": 0}
    load(restored, storage_reader=FileSystemReader(tmp_path), no_dist=True)
    assert torch.equal(restored["value"], torch.arange(8))
    assert restored["step"] == 3


class _BlockingOpen:
    """Picklable gate that holds real filesystem I/O until the parent releases it."""

    def __init__(self, started, release):
        self.started = str(started)
        self.release = str(release)

    def __call__(self, path, mode="rb", **kwargs):
        Path(self.started).touch()
        _wait_for_file(Path(self.release))
        return open(path, mode, **kwargs)


def _wait_for_file(path, timeout=60):
    deadline = monotonic() + timeout
    while not path.exists():
        if monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for filesystem gate: {path}")
        sleep(0.01)


def _finalize_checkpoint(writer, metadata):
    # Real rank-local metadata/result handling without a CUDA-dependent broadcast.
    dist_wrapper = _DistWrapper(None, False, 0)
    dist_wrapper.all_reduce(
        "write", writer.retrieve_write_results, partial(writer.finish, metadata)
    )


def _schedule_cpu_checkpoint(queue, checkpoint_dir, started, release):
    planner = DefaultSavePlanner()
    planner.set_up_planner({"value": torch.arange(8), "step": 3}, is_coordinator=True)
    writer = FileSystemWriterAsync(checkpoint_dir, open_file=_BlockingOpen(started, release))
    writer.set_up_storage_writer(True)
    local_plan = writer.prepare_local_plan(planner.create_local_plan())
    plans, metadata = planner.create_global_plan([local_plan])
    plans = writer.prepare_global_plan(plans)
    writer.prepare_write_data(planner.finish_plan(plans[0]), planner)
    snapshot_ref = weakref.ref(writer.uncached_tensor_data[1][0])
    write_fn, preload_fn, args = writer.get_save_function_and_args()
    request = AsyncRequest(
        write_fn, args, [partial(_finalize_checkpoint, writer, metadata)], preload_fn=preload_fn
    )
    queue.schedule_async_request(request)
    return snapshot_ref


@pytest.mark.parametrize("is_daemon", [True, False])
def test_real_cpu_worker_writes_after_trainer_releases_staging(tmp_path, monkeypatch, is_daemon):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    # cpu_shm_mode only avoids CUDA context initialization in this CPU worker test.
    # None disables Linux QoS syscalls, allowing the spawn test to run on Windows.
    queue = AsyncCallsQueue(is_daemon=is_daemon, cpu_shm_mode=True, cpu_priority=None)
    started, release = tmp_path / "write_started", tmp_path / "allow_write"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    finalized = False
    try:
        snapshot_ref = _schedule_cpu_checkpoint(queue, checkpoint_dir, started, release)
        _wait_for_file(started)
        gc.collect()
        assert snapshot_ref() is None
        assert not release.exists()
        assert queue.get_num_unfinalized_calls() == 1
        assert queue.maybe_finalize_async_calls(no_dist=True) == []
        release.touch()
        assert queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [0]
        finalized = True
        restored = {"value": torch.empty(8, dtype=torch.int64), "step": 0}
        load(restored, storage_reader=FileSystemReader(checkpoint_dir), no_dist=True)
        assert torch.equal(restored["value"], torch.arange(8))
        assert restored["step"] == 3
    finally:
        release.touch()
        queue.close(abort=not finalized)
        queue.async_calls.clear()
