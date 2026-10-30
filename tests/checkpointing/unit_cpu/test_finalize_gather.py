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
import torch
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper, _is_wrapped_exception

from nvidia_resiliency_ext.checkpointing.async_ckpt import state_dict_saver
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import table, writer
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
    rows = state_dict_saver._gather_payloads(payload, _DistWrapper(None, False, 0))
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
