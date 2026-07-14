# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem-backed corpus cases shared by evaluation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from restart_agent_eval import evaluate


def write_evaluation_case(root: Path, *, with_log: bool = True) -> tuple[Path, Path]:
    """Create one mirrored log/gold case and return the two corpus roots."""
    log_root = root / "logs"
    gold_root = root / "gold"
    relative = Path("cases/case-a.log")
    gold_dir = gold_root / relative
    gold_dir.mkdir(parents=True)
    label = {
        "schema_version": evaluate.SCHEMA_VERSION,
        "label_version": 1,
        "case_id": "case-a",
        "review_status": "human_approved",
        "source_sha256": hashlib.sha256(b"root\n").hexdigest(),
        "decision": "RESTART",
        "recovery_assessment_expectation": {
            "failure_domain": ["infrastructure", "unknown"],
            "failure_domain_status": ["supported_but_unconfirmed", "unknown"],
            "retry_outlook_without_workload_change": ["may_recover"],
            "retry_outlook_status": ["supported_but_unconfirmed"],
        },
        "retry_policy_expectation": {
            "accepted_rules": ["bounded_retry"],
            "allowed_retries": 1,
            "retry_budget_exhausted": False,
        },
        "action_expectation": {"accepted": ["RESTART"]},
        "l0b_expectation": {"required_evidence_lines": [12]},
        "primary_anchor_expectation": {
            "accepted_lines": [12],
            "rejected_downstream_lines": [20],
            "tolerance_lines": 0,
        },
    }
    (gold_dir / "gold.json").write_text(json.dumps(label), encoding="utf-8")
    if with_log:
        log_path = log_root / relative
        log_path.parent.mkdir(parents=True)
        log_path.write_text("root\n", encoding="utf-8")
    return log_root, gold_root
