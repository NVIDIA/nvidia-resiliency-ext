# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import multiprocessing
import os
import pickle
import queue
import sys
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import Metadata

from nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader import (
    CachedMetadataFileSystemReader,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.filesystem import (
    ArrowFileSystem,
    CheckpointFileSystem,
    LocalFileSystem,
    MultiStorageFileSystem,
)


class MemoryFileSystem(CheckpointFileSystem):
    def __init__(self, failure=None):
        self.files = {}
        self.directories = set()
        self.calls = []
        self.failure = failure

    def init_path(self, path):
        return os.fspath(path)

    def concat_path(self, path, suffix):
        return str(path).rstrip("/") + "/" + suffix

    @contextmanager
    def create_stream(self, path, mode):
        self.calls.append((mode, path))
        stream = io.BytesIO(self.files[path] if mode == "rb" else b"")
        try:
            yield stream
            if mode == "wb":
                if self.failure == "close":
                    raise OSError("commit failed")
                self.files[path] = stream.getvalue()
        finally:
            stream.close()

    def mkdir(self, path):
        self.directories.add(path)

    def exists(self, path):
        return self.stat(path) != "missing"

    def stat(self, path):
        if path in self.directories:
            return "directory"
        return "file" if path in self.files else "missing"

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        return [key[len(prefix) :] for key in self.files if key.startswith(prefix)]

    def rename(self, path, new_path):
        self.calls.append(("rename", new_path))
        self.files[new_path] = self.files.pop(path)

    def rm_file(self, path):
        del self.files[path]

    def sync(self, stream):
        self.calls.append(("sync", None))
        if self.failure == "sync":
            raise OSError("sync failed")
        stream.flush()

    @classmethod
    def validate_checkpoint_id(cls, path):
        return True


def arrow_local():
    from pyarrow.fs import LocalFileSystem

    return LocalFileSystem()


@pytest.fixture
def cpu_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def write_checkpoint(
    path, filesystem, *, sync_files=True, multiproc=False, use_msc=False, serialize=False
):
    state = {"tensor": torch.arange(12).reshape(3, 4), "other": torch.ones(5), "step": 7}
    writer = FileSystemWriterAsync(
        path,
        filesystem=filesystem,
        use_msc=use_msc,
        sync_files=sync_files,
        thread_count=2,
        is_multiproc_io=multiproc,
    )
    writer.set_up_storage_writer(True)
    planner = DefaultSavePlanner()
    planner.set_up_planner(state, is_coordinator=True)
    local = writer.prepare_local_plan(planner.create_local_plan())
    plans, metadata = planner.create_global_plan([local])
    plan = planner.finish_plan(writer.prepare_global_plan(plans)[0])
    writer.prepare_write_data(plan, planner)
    save, preload, args = writer.get_save_function_and_args()
    request = AsyncRequest(save, args, [], preload_fn=preload)
    if serialize:
        request = pickle.loads(pickle.dumps(request))
    request.execute_sync()
    results = writer.retrieve_write_results()
    assert isinstance(results, list), results
    writer.finish(metadata, [results])
    return state


@pytest.mark.parametrize("multiproc", [False, True])
@pytest.mark.parametrize("backend", ["local", "arrow"])
def test_async_payload_roundtrip(cpu_group, tmp_path, multiproc, backend):
    if backend == "arrow":
        pytest.importorskip("pyarrow.fs")
        filesystem = ArrowFileSystem(arrow_local, uri_scheme="file", max_write_bytes=31)
        path = (tmp_path / "checkpoint").as_uri()
    else:
        filesystem = LocalFileSystem()
        path = tmp_path / "checkpoint"
    state = write_checkpoint(path, filesystem, multiproc=multiproc, serialize=True)
    restored = {
        "tensor": torch.zeros_like(state["tensor"]),
        "other": torch.zeros_like(state["other"]),
        "step": 0,
    }
    reader = CachedMetadataFileSystemReader(path, filesystem=filesystem)
    dcp.load(restored, storage_reader=reader)
    assert torch.equal(restored["tensor"], state["tensor"])
    assert torch.equal(restored["other"], state["other"])
    assert restored["step"] == state["step"]
    assert sorted(filesystem.listdir(path)) == [".metadata", "__0_0.distcp", "__0_1.distcp"]


@pytest.mark.parametrize("sync_files", [False, True])
def test_no_fileno_and_legacy_msc_precedence(cpu_group, sync_files):
    filesystem = MemoryFileSystem()
    writer = FileSystemWriterAsync("memory://root", filesystem=filesystem, use_msc=True)
    assert writer.fs is filesystem
    write_checkpoint("memory://root", filesystem, sync_files=sync_files, use_msc=True)
    assert "memory://root/.metadata" in filesystem.files
    assert all(path is None or path.startswith("memory://root/") for _, path in filesystem.calls)
    assert any(operation == "sync" for operation, _ in filesystem.calls) is sync_files


@pytest.mark.parametrize("fast", ["0", "1"])
@pytest.mark.parametrize("failure", ["sync", "close"])
def test_metadata_failure_does_not_publish(monkeypatch, fast, failure):
    monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", fast)
    filesystem = MemoryFileSystem(failure)
    writer = FileSystemWriterAsync("memory://root", filesystem=filesystem)
    with pytest.raises(OSError, match="failed"):
        writer.finish(Metadata(state_dict_metadata={}), [])
    assert "memory://root/.metadata" not in filesystem.files
    assert not any(operation == "rename" for operation, _ in filesystem.calls)


@pytest.mark.parametrize("failure", ["sync", "close"])
def test_payload_failure_reaches_result_queue(failure):
    filesystem = MemoryFileSystem(failure)
    writer = FileSystemWriterAsync("memory://root", filesystem=filesystem)
    results = queue.Queue()
    transforms = [writer.transforms] if hasattr(writer, "transforms") else []
    writer.write_preloaded_data_multithread(
        transforms, filesystem, True, 0, [("memory://root/data", "data", ([], []))], results
    )
    assert isinstance(results.get_nowait(), Exception)
    assert "memory://root/data" not in filesystem.files


def test_cache_separates_backend_identity():
    path = "memory://same/checkpoint"
    first, second = MemoryFileSystem(), MemoryFileSystem()
    for filesystem, key in [(first, "first"), (second, "second")]:
        filesystem.files[path + "/.metadata"] = pickle.dumps(Metadata(state_dict_metadata={key: 1}))
    CachedMetadataFileSystemReader.clear_metadata_cache()
    one = CachedMetadataFileSystemReader(path, filesystem=first).read_metadata()
    two = CachedMetadataFileSystemReader(path, filesystem=second).read_metadata()
    assert set(one.state_dict_metadata) == {"first"}
    assert set(two.state_dict_metadata) == {"second"}
    assert CachedMetadataFileSystemReader(path, filesystem=first).read_metadata() is one
    assert len(first.calls) == 1
    CachedMetadataFileSystemReader.clear_metadata_cache(path, filesystem=first)
    assert CachedMetadataFileSystemReader(path, filesystem=first).read_metadata() is not one
    assert CachedMetadataFileSystemReader(path, filesystem=second).read_metadata() is two
    CachedMetadataFileSystemReader.clear_metadata_cache()


def test_reader_reset_changes_cache_path():
    filesystem = MemoryFileSystem()
    for path in ("memory://first", "memory://second"):
        filesystem.files[path + "/.metadata"] = pickle.dumps(
            Metadata(state_dict_metadata={path: 1})
        )
    reader = CachedMetadataFileSystemReader("memory://first", filesystem=filesystem)
    assert set(reader.read_metadata().state_dict_metadata) == {"memory://first"}
    reader.reset("memory://second")
    assert set(reader.read_metadata().state_dict_metadata) == {"memory://second"}
    CachedMetadataFileSystemReader.clear_metadata_cache()


@pytest.mark.parametrize("suffix", ["/escape", "../escape", "a/../escape", "s3://else/key"])
@pytest.mark.parametrize("backend", ["local", "msc", "arrow"])
def test_join_rejects_escape(suffix, backend):
    filesystem = {
        "local": LocalFileSystem,
        "msc": MultiStorageFileSystem,
        "arrow": lambda: ArrowFileSystem(arrow_local),
    }[backend]()
    with pytest.raises(ValueError, match="inside"):
        filesystem.concat_path("root", suffix)


def test_local_stat_propagates_permission_error(monkeypatch):
    def denied(path):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "stat", denied)
    with pytest.raises(PermissionError):
        LocalFileSystem().stat("root")


def test_msc_native_local_roundtrip(tmp_path):
    pytest.importorskip("multistorageclient")
    filesystem = MultiStorageFileSystem()
    root = str(tmp_path / "checkpoint")
    filesystem.mkdir(root)
    path = filesystem.concat_path(root, "part")
    with filesystem.create_stream(path, "wb") as stream:
        stream.write(b"payload")
        filesystem.sync(stream)
    assert filesystem.stat(root) == "directory"
    assert filesystem.stat(path) == "file"
    renamed = filesystem.concat_path(root, "renamed")
    filesystem.rename(path, renamed)
    with filesystem.create_stream(renamed, "rb") as stream:
        assert stream.read() == b"payload"
    assert list(filesystem.listdir(root)) == ["renamed"]
    filesystem.rm_file(renamed)
    assert filesystem.stat(renamed) == "missing"


class ProcessNative:
    def __init__(self):
        self.pid = os.getpid()

    def create_dir(self, path, recursive):
        assert self.pid == os.getpid(), "native client crossed a process boundary"

    def __reduce__(self):
        raise TypeError("native client must not be serialized")


def process_native():
    return ProcessNative()


def create_in_child(filesystem, results):
    try:
        filesystem.mkdir("root")
        results.put(None)
    except AssertionError as exc:
        results.put(str(exc))


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_arrow_reconstructs_clients_in_workers(method):
    filesystem = ArrowFileSystem(process_native)
    filesystem.mkdir("root")
    restored = pickle.loads(pickle.dumps(filesystem))
    restored.mkdir("root")
    context = multiprocessing.get_context(method)
    results = context.Queue()
    process = context.Process(target=create_in_child, args=(filesystem, results))
    process.start()
    try:
        assert results.get(timeout=30) is None
        process.join(timeout=30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


class RecordingNative:
    def __init__(self):
        self.events = []
        self.closed = False

    def __call__(self):
        return self

    def open_output_stream(self, path, **kwargs):
        assert kwargs == {"compression": None, "buffer_size": 0}
        self.events.append(("open", path))
        return self

    def write(self, data):
        self.events.append(("write", len(data)))
        return len(data)

    def flush(self):
        self.events.append(("flush", None))

    def close(self):
        self.closed = True


def test_arrow_chunks_writes_and_flushes_before_return():
    native = RecordingNative()
    filesystem = ArrowFileSystem(native, uri_scheme="example", max_write_bytes=3)
    with filesystem.create_stream("example://bucket/key", "wb") as stream:
        assert stream.write(b"abcdefghij") == 10
        assert stream.tell() == 10
        with pytest.raises(io.UnsupportedOperation):
            stream.fileno()
    assert native.events == [("open", "bucket/key")] + [
        event for size in [3, 3, 3, 1] for event in [("write", size), ("flush", None)]
    ]
    assert native.closed


def test_arrow_metadata_supports_pickle_readline(tmp_path):
    pytest.importorskip("pyarrow.fs")
    filesystem = ArrowFileSystem(arrow_local)
    path = str(tmp_path / "metadata.gz")
    value = {"format": "uncompressed", "step": 7}
    with filesystem.create_stream(path, "wb") as stream:
        pickle.dump(value, stream, protocol=0)
    assert (tmp_path / "metadata.gz").read_bytes() == pickle.dumps(value, protocol=0)
    with filesystem.create_stream(path, "rb") as stream:
        assert stream.seekable()
        assert pickle.load(stream) == value
        stream.seek(0)
        assert stream.readline() == b"(dp0\n"


def test_optional_dependencies_are_not_imported():
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import nvidia_resiliency_ext.checkpointing.filesystem; "
                "assert 'pyarrow' not in sys.modules; assert 'multistorageclient' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_obsolete_opener_has_actionable_error(tmp_path):
    with pytest.raises(TypeError, match="filesystem=.*create_stream"):
        FileSystemWriterAsync(tmp_path, open_file=open)
