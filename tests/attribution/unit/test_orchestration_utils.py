# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia_resiliency_ext.attribution.orchestration.utils import (
    selected_log_analyzer_cycle_payload,
)


def _item(action: str, raw_text: str) -> dict:
    return {
        "raw_text": raw_text,
        "auto_resume": raw_text,
        "auto_resume_explanation": "details",
        "attribution_text": "details",
        "checkpoint_saved_flag": 0,
        "action": action,
        "primary_issues": [],
        "secondary_issues": [],
    }


def test_selected_cycle_payload_has_scalar_recommendation_for_selected_cycle():
    continue_item = _item("CONTINUE", "CONTINUE")
    stop_item = _item("STOP", "STOP - DONT RESTART IMMEDIATE")
    log_result = {
        "module": "log_analyzer",
        "result": [continue_item, stop_item],
        "recommendation": {"action": "STOP", "source": "log_analyzer"},
        "result_id": "rid",
        "resource_uri": "attribution://log_analyzer/rid",
    }

    payload = selected_log_analyzer_cycle_payload(log_result, continue_item)

    assert payload["result"] == [continue_item]
    assert payload["recommendation"] == {
        "action": "CONTINUE",
        "source": "log_analyzer",
    }
    assert payload["result_id"] == "rid"
    assert payload["resource_uri"] == "attribution://log_analyzer/rid"


def test_selected_cycle_payload_preserves_recommendation_source():
    stop_item = _item("STOP", "STOP - DONT RESTART IMMEDIATE")
    log_result = {
        "module": "log_fr_analyzer",
        "result": [stop_item],
        "recommendation": {"action": "STOP", "source": "log_analyzer"},
    }

    payload = selected_log_analyzer_cycle_payload(log_result, stop_item)

    assert payload["module"] == "log_fr_analyzer"
    assert payload["recommendation"] == {
        "action": "STOP",
        "source": "log_analyzer",
    }
