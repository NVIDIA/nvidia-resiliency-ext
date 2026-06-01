import pytest

from nvidia_resiliency_ext.attribution.orchestration.llm_output import (
    fr_only_no_log_payload,
    logsage_recommendation,
    logsage_recommendation_from_payload,
    logsage_timeout_payload,
    recommendation_payload,
)


def _item(raw_text, action):
    return {
        "raw_text": raw_text,
        "auto_resume": raw_text.split("\n", 1)[0],
        "auto_resume_explanation": "",
        "attribution_text": "",
        "checkpoint_saved_flag": 0,
        "action": action,
        "primary_issues": [],
        "secondary_issues": [],
    }


def test_logsage_recommendation_derives_restart_from_structured_item():
    recommendation = logsage_recommendation(
        [_item("RESTART IMMEDIATE\ntransient timeout", "RESTART")],
        source="log_analyzer",
    )

    assert recommendation.action == "RESTART"
    assert recommendation.reason == "RESTART IMMEDIATE\ntransient timeout"
    assert recommendation.source == "log_analyzer"


def test_logsage_recommendation_uses_highest_priority_cycle_action():
    recommendation = logsage_recommendation(
        [
            _item("ERRORS NOT FOUND", "CONTINUE"),
            _item("STOP - DONT RESTART IMMEDIATE\nuser error", "STOP"),
        ],
        source="log_analyzer",
    )

    assert recommendation.action == "STOP"
    assert recommendation.reason == "STOP - DONT RESTART IMMEDIATE\nuser error"
    assert recommendation.source == "log_analyzer"


def test_logsage_recommendation_from_payload_maps_timeout():
    payload = logsage_timeout_payload("LLM analysis timed out")
    recommendation = logsage_recommendation_from_payload(
        payload,
    )

    assert payload == {
        "module": "log_analyzer",
        "state": "timeout",
        "result": [],
        "error": "LLM analysis timed out",
        "recommendation": {
            "action": "TIMEOUT",
            "source": "log_analyzer",
        },
    }
    assert recommendation.action == "TIMEOUT"
    assert recommendation.reason == "LLM analysis timed out"
    assert recommendation.source == "log_analyzer"


def test_logsage_recommendation_from_payload_maps_fr_only_no_log():
    payload = fr_only_no_log_payload()
    recommendation = logsage_recommendation_from_payload(payload)

    assert payload == {
        "module": "fr_only",
        "state": "no_log",
        "result": [],
        "recommendation": {
            "action": "UNKNOWN",
            "source": "fr_only",
        },
    }
    assert recommendation.action == "UNKNOWN"
    assert recommendation.reason == "no_log"
    assert recommendation.source == "fr_only"


def test_logsage_recommendation_from_payload_uses_module_as_source_fallback():
    recommendation = logsage_recommendation_from_payload(
        {
            "module": "log_analyzer",
            "recommendation": {
                "action": "STOP",
                "reason": "terminal issue",
            },
            "result": [],
        }
    )

    assert recommendation.action == "STOP"
    assert recommendation.reason == "terminal issue"
    assert recommendation.source == "log_analyzer"


def test_logsage_recommendation_from_payload_does_not_rederive_from_result_items():
    recommendation = logsage_recommendation_from_payload(
        {
            "module": "log_analyzer",
            "result": [_item("STOP - DONT RESTART IMMEDIATE\nuser error", "STOP")],
        }
    )

    assert recommendation.action == "UNKNOWN"
    assert recommendation.reason == ""
    assert recommendation.source == "log_analyzer"


def test_recommendation_payload_keeps_public_contract_action_source_only():
    recommendation = logsage_recommendation(
        [_item("STOP - DONT RESTART IMMEDIATE\nuser error", "STOP")],
        source="log_analyzer",
    )

    assert recommendation_payload(recommendation) == {
        "action": "STOP",
        "source": "log_analyzer",
    }


def test_logsage_recommendation_rejects_unstructured_items():
    with pytest.raises(TypeError):
        logsage_recommendation(["STOP - DONT RESTART"], source="log_analyzer")
