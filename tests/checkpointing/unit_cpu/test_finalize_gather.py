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

"""How finalize sends write results to the coordinator, on CPU: a save without a process group,
and the coordinator telling tables from pickles. The collectives are tested on GPUs in
tests/checkpointing/unit/test_async_writer.py."""

import pickle

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper, _is_wrapped_exception

from nvidia_resiliency_ext.checkpointing.async_ckpt import state_dict_saver
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import table, writer
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_template import MetadataTemplate
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
    FileSystemWriterAsync,
    _wrap_exception_for_gather,
)


def results(rank: int, n: int = 3):
    """n write results of a rank."""
    return [
        WriteResult(
            index=MetadataIndex(f"r{rank}.w", torch.Size([i, 0]), i),
            size_in_bytes=10,
            storage_data=_StorageInfo(f"__{rank}_0.distcp", 10 * i, 10),
        )
        for i in range(n)
    ]


def test_without_process_group(tmp_path):
    """Without a process group, the rank's own payload is the one row, padded to 8 bytes."""
    payload = state_dict_saver._encode_write_results(results(0), FileSystemWriterAsync(tmp_path))
    rows, _ = state_dict_saver._gather_payloads(payload, _DistWrapper(None, False, 0))
    assert rows.shape[0] == 1 and rows.shape[1] % 8 == 0 and rows.shape[1] >= len(payload)
    assert rows[0, : len(payload)].tobytes() == payload
    is_table, pickled = state_dict_saver._decode_payloads(rows)
    if writer.writes_tables():
        assert is_table.tolist() == [True] and pickled == {}
        assert table.to_write_results(table.decode(rows[0])) == results(0)
    else:
        assert is_table.tolist() == [False] and pickled == {0: results(0)}


def test_coordinator_tells_tables_from_pickles():
    """Rows of tables and pickles (write results, an exception, a pickle shorter than 8 bytes):
    the coordinator unpickles only the pickles, and gets every rank's write results back."""
    failure = _wrap_exception_for_gather(RuntimeError("worker failed"))
    payloads = [
        table.encode(results(0)),
        pickle.dumps(results(1)),
        pickle.dumps(failure),
        pickle.dumps([]),
    ]
    assert len(payloads[3]) < 8
    width = -(-max(map(len, payloads)) // 8) * 8
    rows = np.stack([np.frombuffer(p.ljust(width, b"\0"), dtype=np.uint8) for p in payloads])
    is_table, pickled = state_dict_saver._decode_payloads(rows)
    assert is_table.tolist() == [True, False, False, False]
    assert sorted(pickled) == [1, 2, 3]
    assert pickled[1] == results(1) and pickled[3] == []
    assert _is_wrapped_exception(pickled[2])
    assert table.to_write_results(table.decode(rows[0])) == results(0)


def _rows(payloads):
    """Return the payloads as the coordinator receives them: one zero-padded row each."""
    width = -(-max(map(len, payloads)) // 8) * 8
    return np.stack([np.frombuffer(p.ljust(width, b"\0"), dtype=np.uint8) for p in payloads])


def test_metadata_with_an_fqn_that_isnt_a_str_is_written(tmp_path):
    """The rank writing that fqn sends its write results pickled, the other one a table if tables
    are on; the coordinator writes both."""
    int_result = WriteResult(MetadataIndex(1), 4, _StorageInfo("__1_0.distcp", 0, 4))
    metadata = Metadata(
        state_dict_metadata={"r0.w": BytesStorageMetadata(), 1: BytesStorageMetadata()}
    )
    storage_writer = FileSystemWriterAsync(tmp_path)
    rows = _rows(
        [
            state_dict_saver._encode_write_results(rank_results, storage_writer)
            for rank_results in (results(0), [int_result])
        ]
    )
    is_table, pickled = state_dict_saver._decode_payloads(rows)
    assert is_table.tolist() == [writer.writes_tables(), False]
    template = MetadataTemplate.from_metadata(metadata)
    state_dict_saver._write_metadata(storage_writer, template, rows, is_table, pickled)
    written = FileSystemReader(tmp_path).read_metadata()
    assert written.storage_data == {r.index: r.storage_data for r in results(0) + [int_result]}


class _OwnFinish(FileSystemWriterAsync):
    """A storage writer whose subclass has its own finish."""

    def finish(self, metadata, results):
        super().finish(metadata, results)


def test_overridden_finish_gets_the_write_results(tmp_path):
    """A rank whose storage writer overrides finish sends its write results pickled, as they are,
    so that finish gets them unchanged; one that doesn't sends a table if tables are on."""
    duplicate = WriteResult(
        index=MetadataIndex("r0.w", torch.Size([0, 0]), 7),
        size_in_bytes=4,
        storage_data=_StorageInfo("__0_1.distcp", 0, 4),
    )
    rank_results = results(0) + [duplicate]
    payload = state_dict_saver._encode_write_results(rank_results, _OwnFinish(tmp_path))
    sent = pickle.loads(payload)
    assert [(vars(r.index), r.size_in_bytes, r.storage_data) for r in sent] == [
        (vars(r.index), r.size_in_bytes, r.storage_data) for r in rank_results
    ]
    payload = state_dict_saver._encode_write_results(rank_results, FileSystemWriterAsync(tmp_path))
    assert (payload[:8] == table.MAGIC.to_bytes(8, "little")) == writer.writes_tables()


@pytest.mark.skipif(not writer.writes_tables(), reason="needs the fast .metadata writer")
def test_write_results_name_fqns_by_entry_in_the_fqn_list(tmp_path):
    """With an fqn list holding every fqn the rank writes, it sends an index table."""
    storage_writer = FileSystemWriterAsync(tmp_path)
    fqn_list = table.FqnList.of(["r0.w", "other"])
    payload = state_dict_saver._encode_write_results(results(0), storage_writer, fqn_list)
    sent = table.decode(np.frombuffer(payload, dtype=np.uint8))
    assert sent.fqn_mode == table.FQN_INDICES
    assert table.to_write_results(sent, fqn_list) == results(0)


@pytest.mark.skipif(not writer.writes_tables(), reason="needs the fast .metadata writer")
def test_an_fqn_missing_from_the_list_sends_fqn_strings(tmp_path):
    storage_writer = FileSystemWriterAsync(tmp_path)
    fqn_list = table.FqnList.of(["other"])
    payload = state_dict_saver._encode_write_results(results(0), storage_writer, fqn_list)
    sent = table.decode(np.frombuffer(payload, dtype=np.uint8))
    assert sent.fqn_mode == table.FQN_STRINGS
    assert table.to_write_results(sent) == results(0)


@pytest.mark.skipif(not writer.writes_tables(), reason="needs the fast .metadata writer")
def test_write_results_no_table_holds_are_pickled_after_one_try(tmp_path, monkeypatch):
    """Only an fqn the list lacks makes a rank try a string table after the index table; write
    results that no table can hold are sent pickled after the first try."""
    unencodable = [
        WriteResult(
            index=MetadataIndex("r0.w", torch.Size([0]), 0),
            size_in_bytes=10,
            storage_data=_StorageInfo("__0_0.distcp", 2**70, 10),  # outside int64
        )
    ]
    tries = []
    encode = table.encode

    def counting(*args):
        tries.append(args)
        return encode(*args)

    monkeypatch.setattr(table, "encode", counting)
    storage_writer = FileSystemWriterAsync(tmp_path)
    fqn_list = table.FqnList.of(["r0.w"])
    payload = state_dict_saver._encode_write_results(unencodable, storage_writer, fqn_list)
    assert pickle.loads(payload) == unencodable and len(tries) == 1


@pytest.mark.skipif(not writer.writes_tables(), reason="needs the fast .metadata writer")
@given(
    written=st.lists(st.text(min_size=1), min_size=1, max_size=5, unique=True),
    listed=st.lists(st.text(min_size=1), max_size=8, unique=True),
)
def test_write_results_name_fqns_by_entry_iff_the_list_holds_them(
    tmp_path_factory, written, listed
):
    """Index table iff the fqn list holds every fqn written; either way they decode back."""
    sent_results = [
        WriteResult(
            index=MetadataIndex(fqn, torch.Size([i]), i),
            size_in_bytes=10,
            storage_data=_StorageInfo("__0_0.distcp", 10 * i, 10),
        )
        for i, fqn in enumerate(written)
    ]
    fqn_list = table.FqnList.of(listed)
    storage_writer = FileSystemWriterAsync(tmp_path_factory.mktemp("writer"))
    payload = state_dict_saver._encode_write_results(sent_results, storage_writer, fqn_list)
    sent = table.decode(np.frombuffer(payload, dtype=np.uint8))
    holds_all = set(written) <= set(listed)
    assert sent.fqn_mode == (table.FQN_INDICES if holds_all else table.FQN_STRINGS)
    assert table.to_write_results(sent, fqn_list) == sent_results
