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

"""Tests for the prepared-metadata reuse *layout guard*.

Background. ``save_state_dict_async_finalize(..., reuse_metadata_obj=...)`` skips
the write-result gather and publishes a prepared ``.metadata`` verbatim, while each
rank still rewrites its data with offsets recomputed from the bytes actually
written. That is only safe if every item lands at the same
``(relative_path, offset, length)`` the prepared metadata records. Tensors are
raw-stored so their length is fixed by ``(shape, dtype)``, but a non-tensor
(BYTE_IO) item is ``torch.save``-pickled and its length can drift within a run --
most concretely the distributed optimizer's ``param_groups[*].step`` (a Python int)
whose pickled size steps up as ``step`` crosses 256 / 65536 / 2**31. When that
happens the item, and every item after it in the same file, shifts, and a reused
``.metadata`` would address the wrong bytes and load corrupt.

These tests cover the guard added for exactly that failure:

* ``_find_layout_mismatch`` (pure, no distributed), both directions: a length
  drift, a downstream offset shift, an item missing from the prepared metadata, OR
  an item the prepared metadata expects in this rank's file that was not written
  this save (removed / resharded) -> a ``(fqn, detail)`` report; a matching layout
  and another rank's entries -> ``None``. The numbers mirror the real measurement
  (the optimizer BYTE_IO item is 1645 B at a low step and grows to 1709 B once
  ``step`` crosses 65536).
* End-to-end through ``save_state_dict_async_finalize``: a matching layout writes
  ``.metadata``; a stale one raises ``CheckpointException`` and writes NO
  ``.metadata`` (fail-fast, no silent corruption).
"""

import os
import pickle

import pytest
import torch
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper

from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
    FileSystemWriterAsync,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    _find_layout_mismatch,
    save_state_dict_async_finalize,
)

from . import TempNamedDir
from .test_utilities import Utils

# Real distributed-optimizer item + a tensor written after it in the same file.
# Byte sizes mirror the measured production-shaped optimizer item: 1645 B while the
# optimizer `step` is small, growing to 1709 B once `step` crosses 65536.
_OPT_FQN = "chained_0.optimizer.distributed.dp_group_idx_0.optimizer/shard_0_1"
_WGT_FQN = "module.decoder.weight/shard_0_1"
_OPT_LEN, _WGT_LEN = 1645, 4096
_OPT_LEN_DRIFTED = 1709  # +64: one torch.save 64B alignment quantum


def _storage_info(path, offset, length):
    return _StorageInfo(relative_path=path, offset=offset, length=length)


def _write_result(fqn, path, offset, length):
    """A WriteResult exactly as filesystem `_write_item` produces it (index + the
    per-item `_StorageInfo` locator)."""
    return WriteResult(
        index=MetadataIndex(fqn=fqn),
        size_in_bytes=length,
        storage_data=_storage_info(path, offset, length),
    )


def _prepared_metadata(entries):
    """Build a complete Metadata from ``entries`` = list of (fqn, path, offset, length).

    Only ``storage_data`` matters to the guard (it is the byte map the loader would
    trust); ``state_dict_metadata`` is left empty since the guard never consults it.
    """
    storage_data = {
        MetadataIndex(fqn=fqn): _storage_info(path, offset, length)
        for (fqn, path, offset, length) in entries
    }
    return Metadata(state_dict_metadata={}, storage_data=storage_data)


_SINGLE_FILE_LAYOUT = [
    (_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN),
    (_WGT_FQN, "__0_0.distcp", _OPT_LEN, _WGT_LEN),
]


# --------------------------------------------------------------------------- #
# Pure-logic tests of _find_layout_mismatch (local_layout_matches_prepared)   #
# --------------------------------------------------------------------------- #
def test_matching_layout_reports_no_mismatch():
    """Identical fresh layout -> the guard reports a match (None)."""
    prepared = _prepared_metadata(_SINGLE_FILE_LAYOUT)
    write_results = [
        _write_result(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN),
        _write_result(_WGT_FQN, "__0_0.distcp", _OPT_LEN, _WGT_LEN),
    ]
    assert _find_layout_mismatch(write_results, prepared) is None


def test_optimizer_step_length_drift_is_detected():
    """The real hazard: the optimizer BYTE_IO item grows 1645 -> 1709 B when `step`
    crosses 65536, so its length no longer matches the prepared metadata."""
    prepared = _prepared_metadata(_SINGLE_FILE_LAYOUT)  # captured at a low step
    # step crossed 65536: the optimizer item grew by one 64B quantum and the tensor
    # written after it in the same file shifted by the same amount.
    write_results = [
        _write_result(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN_DRIFTED),
        _write_result(_WGT_FQN, "__0_0.distcp", _OPT_LEN_DRIFTED, _WGT_LEN),
    ]
    mismatch = _find_layout_mismatch(write_results, prepared)
    assert mismatch is not None
    fqn, detail = mismatch
    assert fqn == _OPT_FQN  # the first (offending) item is reported
    assert str(_OPT_LEN) in detail and str(_OPT_LEN_DRIFTED) in detail


def test_downstream_offset_shift_is_detected():
    """Even if an item's own length is unchanged, an upstream drift shifts its
    offset -- which the guard also catches."""
    prepared = _prepared_metadata(_SINGLE_FILE_LAYOUT)
    write_results = [
        _write_result(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN),  # unchanged
        _write_result(_WGT_FQN, "__0_0.distcp", _OPT_LEN_DRIFTED, _WGT_LEN),  # shifted
    ]
    mismatch = _find_layout_mismatch(write_results, prepared)
    assert mismatch is not None and mismatch[0] == _WGT_FQN


def test_item_absent_from_prepared_is_detected():
    """An item written this save but not present in the prepared metadata (a
    structure change) is reported as a mismatch."""
    prepared = _prepared_metadata(_SINGLE_FILE_LAYOUT)
    new_fqn = "module.new_layer.weight/shard_0_1"
    mismatch = _find_layout_mismatch(
        [_write_result(new_fqn, "__0_0.distcp", 0, 128)], prepared
    )
    assert mismatch is not None
    assert mismatch[0] == new_fqn and "absent" in mismatch[1]


def test_removed_last_item_is_detected():
    """Reverse direction: an item the prepared metadata places at the END of this
    rank's file but which this save did not write (a removal) shifts nothing and
    would slip past a forward-only check -- the reverse check must catch it."""
    prepared = _prepared_metadata(_SINGLE_FILE_LAYOUT)  # [_OPT_FQN@0, _WGT_FQN@_OPT_LEN]
    write_results = [_write_result(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN)]  # _WGT_FQN removed
    mismatch = _find_layout_mismatch(write_results, prepared)
    assert mismatch is not None
    assert mismatch[0] == _WGT_FQN and "not written" in mismatch[1].lower()


def test_resharded_item_is_detected():
    """Reverse direction: an item the prepared metadata keeps in a file this rank
    still writes, but which moved to another rank this save (resharding), leaves a
    dangling entry -- must be caught."""
    moved = "moved.item/shard_0_1"
    prepared = _prepared_metadata(
        [(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN), (moved, "__0_0.distcp", _OPT_LEN, 256)]
    )
    write_results = [_write_result(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN)]  # `moved` elsewhere now
    mismatch = _find_layout_mismatch(write_results, prepared)
    assert mismatch is not None and mismatch[0] == moved


def test_entry_in_another_ranks_file_is_not_a_false_positive():
    """The reverse check is scoped to files THIS rank wrote: a prepared entry in a
    file this rank did not write (owned by another rank) must NOT be flagged."""
    prepared = _prepared_metadata(
        [(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN), ("peer.item/shard_0_1", "__1_0.distcp", 0, 512)]
    )
    write_results = [_write_result(_OPT_FQN, "__0_0.distcp", 0, _OPT_LEN)]  # only wrote __0_0.distcp
    assert _find_layout_mismatch(write_results, prepared) is None


# --------------------------------------------------------------------------- #
# End-to-end tests through save_state_dict_async_finalize (reuse path)         #
# --------------------------------------------------------------------------- #
def _read_metadata(ckpt_dir):
    with open(os.path.join(str(ckpt_dir), ".metadata"), "rb") as f:
        return pickle.load(f)


def _per_rank_entries(world_size):
    """Prepared layout covering every rank: rank r writes an optimizer BYTE_IO item
    then a weight tensor into its own ``__r_0.distcp`` (unique fqns per rank, so no
    dedup and each rank owns a self-contained slice to validate)."""
    entries = []
    for r in range(world_size):
        f = f"__{r}_0.distcp"
        entries.append((f"opt.rank_{r}", f, 0, _OPT_LEN))
        entries.append((f"w.rank_{r}", f, _OPT_LEN, _WGT_LEN))
    return entries


def _finalize(writer, prepared):
    """Drive the reuse finalize with a coordinator-rank-0 dist wrapper over the
    default process group (group=None). All ranks participate (collective)."""
    dist_wrapper = _DistWrapper(group=None, use_dist=True, coordinator_rank=0)
    return save_state_dict_async_finalize(
        writer, global_metadata=None, dist_wrapper=dist_wrapper, reuse_metadata_obj=prepared
    )


class TestReuseFinalizeLayoutGuard:
    def test_matching_layout_publishes_metadata(self, tmp_path_dist_ckpt):
        """When every rank's fresh layout matches the prepared metadata, the reuse
        finalize publishes it verbatim as ``.metadata``."""
        Utils.initialize_distributed()
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        with TempNamedDir(tmp_path_dist_ckpt / "reuse_match") as ckpt_dir:
            prepared = _prepared_metadata(_per_rank_entries(world))
            writer = FileSystemWriterAsync(ckpt_dir, single_file_per_rank=True)
            f = f"__{rank}_0.distcp"
            # stub this rank's freshly-written results to MATCH the prepared layout
            writer.retrieve_write_results = lambda: [
                _write_result(f"opt.rank_{rank}", f, 0, _OPT_LEN),
                _write_result(f"w.rank_{rank}", f, _OPT_LEN, _WGT_LEN),
            ]
            _finalize(writer, prepared)
            torch.distributed.barrier()  # ensure coordinator's write is visible
            if rank == 0:
                published = _read_metadata(ckpt_dir)
                assert published.storage_data == prepared.storage_data

    def test_stale_layout_raises_and_writes_no_metadata(self, tmp_path_dist_ckpt):
        """When a rank's fresh layout drifts from the prepared metadata (optimizer
        `step` crossed 65536 on rank 0), the reuse finalize must RAISE and must NOT
        publish a mislocating ``.metadata``."""
        Utils.initialize_distributed()
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        with TempNamedDir(tmp_path_dist_ckpt / "reuse_stale") as ckpt_dir:
            prepared = _prepared_metadata(_per_rank_entries(world))  # low-step layout
            writer = FileSystemWriterAsync(ckpt_dir, single_file_per_rank=True)
            f = f"__{rank}_0.distcp"
            if rank == 0:
                # optimizer item grew 1645 -> 1709 and the weight shifted by +64
                stubbed = [
                    _write_result("opt.rank_0", f, 0, _OPT_LEN_DRIFTED),
                    _write_result("w.rank_0", f, _OPT_LEN_DRIFTED, _WGT_LEN),
                ]
            else:
                stubbed = [
                    _write_result(f"opt.rank_{rank}", f, 0, _OPT_LEN),
                    _write_result(f"w.rank_{rank}", f, _OPT_LEN, _WGT_LEN),
                ]
            writer.retrieve_write_results = lambda: stubbed
            # The guard reduces a 3-valued status with MIN, so ALL ranks observe the
            # stale status and raise (not just rank 0).
            with pytest.raises(CheckpointException):
                _finalize(writer, prepared)
            torch.distributed.barrier()
            assert not os.path.exists(os.path.join(str(ckpt_dir), ".metadata"))
