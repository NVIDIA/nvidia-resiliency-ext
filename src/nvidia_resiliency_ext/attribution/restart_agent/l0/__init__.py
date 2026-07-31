# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L0 deterministic log interpretation and evidence projection."""

from .assembly import build_cascades_for_primary, build_l0_bundle
from .codec import read_l0_bundle, write_l0_bundle
from .decision import (
    build_decision_evidence,
    canonical_identity_anchor_line,
    distributed_incident_for_line,
)
from .projection import build_l0_model_facing_view
from .streaming import (
    FinalizedL0A,
    ProgressiveL0Accumulator,
    ProgressiveL0State,
    ProgressiveSourceUnavailable,
    canonical_l0a_hash,
    canonical_l0a_payload,
    finalize_log_snapshot,
)

__all__ = [
    "build_cascades_for_primary",
    "build_decision_evidence",
    "build_l0_bundle",
    "build_l0_model_facing_view",
    "canonical_identity_anchor_line",
    "distributed_incident_for_line",
    "FinalizedL0A",
    "ProgressiveL0Accumulator",
    "ProgressiveL0State",
    "ProgressiveSourceUnavailable",
    "canonical_l0a_hash",
    "canonical_l0a_payload",
    "finalize_log_snapshot",
    "read_l0_bundle",
    "write_l0_bundle",
]
