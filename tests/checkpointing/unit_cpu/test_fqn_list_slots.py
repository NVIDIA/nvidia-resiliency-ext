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
through random sequences of these; the finalize's delivery of the list is simulated as the
broadcast does it. The collectives are tested on GPUs in
tests/checkpointing/unit/test_async_writer.py."""

from collections import deque

from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata

from nvidia_resiliency_ext.checkpointing.async_ckpt._metadata_pickler.table import FqnList
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import CheckpointMetadataCache

# Global metadata of different state dicts, as fqns.
STATE_DICTS = [("a", "b"), ("b", "a"), ("a", "b", "c"), ("x",)]


def metadata_of(fqns) -> Metadata:
    """Return global metadata of a state dict of bytes entries named fqns."""
    return Metadata(state_dict_metadata={fqn: BytesStorageMetadata() for fqn in fqns})


class FqnListDelivery(RuleBasedStateMachine):
    """A coordinator's and another rank's caches, and the saves planned but not finalized."""

    def __init__(self):
        super().__init__()
        self.coordinator = CheckpointMetadataCache()
        self.other = CheckpointMetadataCache()
        # Per pending save: the slots of both ranks, whether it delivers, and the list to expect.
        self.pending = deque()

    @rule(state_dict=st.sampled_from(STATE_DICTS))
    def load(self, state_dict):
        for cache in (self.coordinator, self.other):
            cache.set_cached_global_metadata(metadata_of(state_dict))

    @rule(state_dict=st.sampled_from(STATE_DICTS))
    def build(self, state_dict):
        self.plan(metadata_of(state_dict), reused=False)

    @precondition(lambda self: self.coordinator._metadata_template is not None)
    @rule()
    def reuse(self):
        self.plan(None, reused=True)

    def plan(self, built, reused):
        """Record a planned save in both caches, as save_state_dict_async_plan does."""
        template = self.coordinator._update(None, None, built, reused, is_coordinator=True)
        self.other._update(None, None, None, reused, is_coordinator=False)
        self.pending.append(
            (
                self.coordinator._fqn_list_slot,
                self.other._fqn_list_slot,
                not reused,
                FqnList.of(template.to_metadata().state_dict_metadata),
            )
        )

    @precondition(lambda self: self.pending)
    @rule()
    def finalize(self):
        coordinator_slot, other_slot, delivers, expected = self.pending.popleft()
        # What each rank encodes its write results against: the list of this save's metadata.
        assert coordinator_slot.fqn_list == expected
        assert other_slot.fqn_list in (None, expected)
        if delivers:
            other_slot.fqn_list = FqnList.decode(coordinator_slot.fqn_list.encode())
        assert other_slot.fqn_list == expected

    @invariant()
    def the_ranks_agree_once_every_save_is_finalized(self):
        if not self.pending:
            assert self.other._fqn_list_slot.fqn_list in (
                None,
                self.coordinator._fqn_list_slot.fqn_list,
            )


TestFqnListDelivery = FqnListDelivery.TestCase
