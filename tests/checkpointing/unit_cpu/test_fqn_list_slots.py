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

"""Which fqn list each rank's write results refer to, across loads, builds, reuses and finalizes
that run after the next save was planned. A coordinator's and another rank's caches are driven
through random sequences of these, with the plan's and the finalize's own helpers; only the
collectives that carry the list are left out. They are tested on GPUs in
tests/checkpointing/unit/test_async_writer.py."""

import tempfile
import types
from collections import deque

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata

from nvidia_resiliency_ext.checkpointing.async_ckpt import state_dict_saver
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler import writer
from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.write_result_table import (
    FqnList,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import CheckpointMetadataCache

# Global metadata of different state dicts, as fqns; one has an fqn that isn't a str, and so no
# fqn list.
STATE_DICTS = [("a", "b"), ("b", "a"), ("a", "b", "c"), ("x",), ("a", 1)]


class OwnFinish(FileSystemWriterAsync):
    """A storage writer whose finish a subclass overrides: it gets write results, not tables."""

    def finish(self, metadata, results):
        super().finish(metadata, results)


# What the plan and the finalize read of a storage writer and a distributed wrapper.
STORAGE_WRITERS = {True: FileSystemWriterAsync(tempfile.gettempdir()), False: OwnFinish(".")}
COORDINATOR = types.SimpleNamespace(is_coordinator=True)
OTHER_RANK = types.SimpleNamespace(is_coordinator=False)


def metadata_of(fqns) -> Metadata:
    """Return global metadata of a state dict of bytes entries named fqns."""
    return Metadata(state_dict_metadata={fqn: BytesStorageMetadata() for fqn in fqns})


class FqnListDelivery(RuleBasedStateMachine):
    """A coordinator's and another rank's caches, and the saves planned but not finalized."""

    def __init__(self):
        super().__init__()
        self.coordinator = CheckpointMetadataCache()
        self.other = CheckpointMetadataCache()
        # Per pending save: both ranks' PendingSave, and the fqn list each must encode against.
        self.pending = deque()

    @rule(state_dict=st.sampled_from(STATE_DICTS))
    def load(self, state_dict):
        for cache in (self.coordinator, self.other):
            cache.set_cached_global_metadata(metadata_of(state_dict))

    @rule(state_dict=st.sampled_from(STATE_DICTS), tables=st.booleans())
    def build(self, state_dict, tables):
        self.plan(metadata_of(state_dict), reused=False, tables=tables)

    @precondition(lambda self: self.coordinator._metadata_template is not None)
    @rule(tables=st.booleans())
    def reuse(self, tables):
        self.plan(None, reused=True, tables=tables)

    def plan(self, built, reused, tables):
        """Plan a save on both ranks, as save_state_dict_async_plan ends it, with a storage writer
        that writes from tables or one that gets write results."""
        storage_writer = STORAGE_WRITERS[tables]
        coordinator = state_dict_saver._pending_save(
            storage_writer, COORDINATOR, self.coordinator, None, None, built, reused
        )
        other = state_dict_saver._pending_save(
            storage_writer, OTHER_RANK, self.other, None, None, None, reused
        )
        assert coordinator._delivers_fqn_list == other._delivers_fqn_list == (tables and not reused)
        expected = FqnList.of(coordinator._metadata_template.to_metadata().state_dict_metadata)
        self.pending.append((coordinator, other, expected))

    @precondition(lambda self: self.pending)
    @rule()
    def finalize(self):
        """Finalize the oldest save: what each rank encodes against, and the delivery."""
        coordinator, other, expected = self.pending.popleft()
        assert coordinator._fqn_list_slot.fqn_list == expected
        assert other._fqn_list_slot.fqn_list in (None, expected)
        delivered = state_dict_saver._fqn_list_to_deliver(coordinator)
        # Only a build with tables delivers, and only metadata that has a list.
        assert bool(delivered) == (coordinator._delivers_fqn_list and expected is not None)
        if delivered:
            state_dict_saver._receive_fqn_list(other, delivered)
            assert other._fqn_list_slot.fqn_list == expected

    @invariant()
    def the_ranks_agree_once_every_save_is_finalized(self):
        if not self.pending:
            assert self.other._fqn_list_slot.fqn_list in (
                None,
                self.coordinator._fqn_list_slot.fqn_list,
            )


TestFqnListDelivery = pytest.mark.skipif(
    not writer.writes_tables(), reason="the fqn list is delivered only with write-result tables"
)(FqnListDelivery.TestCase)
