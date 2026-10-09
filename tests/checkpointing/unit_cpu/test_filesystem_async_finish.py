# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""FileSystemWriterAsync.finish against torch's FileSystemWriter.finish, on CPU.

The full save path, on GPUs, is tested in tests/checkpointing/unit/test_async_writer.py.
"""

import inspect
import os
import pickle
from dataclasses import fields

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.storage import WriteResult

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import table, writer
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync

# FileSystemWriter.set_up_storage_writer takes rank and use_collectives from PyTorch 2.9.
HAS_PER_RANK_METADATA = (
    "kwargs" in inspect.signature(FileSystemWriter.set_up_storage_writer).parameters
)
ON_TESTED_TORCH = writer.fast_metadata_enabled()
writer.fast_metadata_enabled.cache_clear()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A small checkpoint saved by torch: its Metadata, and the write results it was built from."""
    path = tmp_path_factory.mktemp("saved")
    state_dict = {f"t{i}": torch.randn(3, 5 + i) for i in range(20)}
    state_dict["step"] = 7
    dcp.save(state_dict, storage_writer=FileSystemWriter(path))  # no process group
    md = FileSystemReader(path).read_metadata()
    results = [
        [
            WriteResult(index=index, size_in_bytes=info.length, storage_data=info)
            for index, info in md.storage_data.items()
        ]
    ]
    return md, results


def finish(writer_cls, path, saved, sync_files=True, as_rows=False, **setup_kwargs):
    """Run writer_cls.finish into path, given the write results or (as_rows) their table; return
    the files it wrote and their loaded contents."""
    md, results = saved
    md = pickle.loads(pickle.dumps(md))
    path.mkdir(exist_ok=True)  # as prepare_local_plan does before a save
    md.storage_data = None
    writer = writer_cls(path, sync_files=sync_files)
    writer.set_up_storage_writer(True, **setup_kwargs)
    if as_rows:
        rows = np.frombuffer(table.encode(results[0]), dtype=np.uint8)[None, :]
        writer._finish(md, [], storage_rows=rows)
    else:
        writer.finish(md, results)
    files = sorted(os.listdir(path))
    return files, (
        [FileSystemReader(path).read_metadata()]
        if ".metadata" in files
        else [pickle.load(open(os.path.join(path, f), "rb")) for f in files]
    )


def assert_same_as_torch(tmp_path, saved, **kwargs):
    nvrx_files, nvrx_mds = finish(FileSystemWriterAsync, tmp_path / "nvrx", saved, **kwargs)
    torch_files, torch_mds = finish(FileSystemWriter, tmp_path / "torch", saved, **kwargs)
    assert nvrx_files == torch_files
    for nvrx_md, torch_md in zip(nvrx_mds, torch_mds):
        for field in fields(torch_md):
            if field.name != "storage_meta":  # holds the checkpoint path
                assert getattr(nvrx_md, field.name) == getattr(torch_md, field.name), field.name
    return nvrx_mds


@pytest.mark.parametrize("sync_files", [True, False])
def test_finish_writes_what_torch_writes(tmp_path, saved, sync_files):
    (md,) = assert_same_as_torch(tmp_path, saved, sync_files=sync_files)
    assert md.storage_data == saved[0].storage_data


@pytest.mark.skipif(not HAS_PER_RANK_METADATA, reason="per-rank metadata needs PyTorch 2.9+")
def test_finish_writes_per_rank_metadata(tmp_path, saved):
    assert_same_as_torch(tmp_path, saved, rank=3, use_collectives=False)


@pytest.mark.parametrize(
    ("mode", "version", "defers"),
    [
        pytest.param(
            None,
            None,
            False,
            marks=pytest.mark.skipif(not ON_TESTED_TORCH, reason="torch outside the tested range"),
        ),
        ("0", None, True),
        ("force", "2.3.1", True),  # in INCOMPATIBLE_TORCH_VERSIONS
    ],
)
@pytest.mark.parametrize("as_rows", [False, True], ids=["results", "rows"])
def test_finish_defers_to_torch(monkeypatch, tmp_path, saved, mode, version, defers, as_rows):
    """finish writes with torch's writer if nvrx's is off, also given tables, which it converts."""
    if mode is not None:
        monkeypatch.setenv("NVRX_FAST_METADATA_PICKLE", mode)
    if version is not None:
        monkeypatch.setattr(torch, "__version__", version)
    calls = []
    torch_finish = FileSystemWriter.finish
    monkeypatch.setattr(
        FileSystemWriter, "finish", lambda self, *args: calls.append(1) or torch_finish(self, *args)
    )
    files, (md,) = finish(FileSystemWriterAsync, tmp_path, saved, as_rows=as_rows)
    assert bool(calls) is defers
    assert md.storage_data == saved[0].storage_data
