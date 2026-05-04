import logging

from nvidia_resiliency_ext.attribution.orchestration.llm_output import attribution_recommendation


def test_attribution_recommendation_uses_state_stop():
    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "STOP",
            "result": ["STOP - DONT RESTART"],
        }
    )

    assert recommendation.action == "STOP"
    assert recommendation.reason == "STOP - DONT RESTART"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_maps_timeout():
    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "timeout",
            "error": "LLM analysis timed out",
            "result": [],
        }
    )

    assert recommendation.action == "TIMEOUT"
    assert recommendation.reason == "LLM analysis timed out"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_maps_continue_state_with_restart_text_to_restart():
    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "CONTINUE",
            "result": ["RESTART IMMEDIATE"],
        }
    )

    assert recommendation.action == "RESTART"
    assert recommendation.reason == "RESTART IMMEDIATE"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_unwraps_nested_restart_result_reason():
    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "CONTINUE",
            "result": [["RESTART IMMEDIATE\ntransient timeout", "AttributionState.CONTINUE"]],
        }
    )

    assert recommendation.action == "RESTART"
    assert recommendation.reason == "RESTART IMMEDIATE\ntransient timeout"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_unwraps_nested_stop_result_reason():
    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "STOP",
            "result": [["STOP - DONT RESTART IMMEDIATE\nuser error", "AttributionState.STOP"]],
        }
    )

    assert recommendation.action == "STOP"
    assert recommendation.reason == "STOP - DONT RESTART IMMEDIATE\nuser error"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_maps_bare_stop_text_to_stop():
    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "CONTINUE",
            "result": ["STOP"],
        }
    )

    assert recommendation.action == "STOP"
    assert recommendation.reason == "STOP"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_keeps_continue_state_when_no_restart_text():
    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "CONTINUE",
            "result": ["ERRORS NOT FOUND"],
        }
    )

    assert recommendation.action == "CONTINUE"
    assert recommendation.reason == "ERRORS NOT FOUND"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_does_not_warn_on_known_state(caplog):
    caplog.set_level(
        logging.WARNING,
        logger="nvidia_resiliency_ext.attribution.orchestration.llm_output",
    )

    recommendation = attribution_recommendation(
        {
            "module": "log_analyzer",
            "state": "CONTINUE",
            "result": [],
        }
    )

    assert recommendation.action == "CONTINUE"
    assert recommendation.reason == "state=CONTINUE"
    assert recommendation.source == "log_analyzer"
    assert "falling through to string matching" not in caplog.text


def test_attribution_recommendation_does_not_warn_on_fr_only_no_log(caplog):
    caplog.set_level(
        logging.WARNING,
        logger="nvidia_resiliency_ext.attribution.orchestration.llm_output",
    )

    recommendation = attribution_recommendation(
        {
            "module": "fr_only",
            "state": "no_log",
            "result": [],
        }
    )

    assert recommendation.action == "UNKNOWN"
    assert recommendation.reason == "no_log"
    assert recommendation.source == "fr_only"
    assert "falling through to string matching" not in caplog.text


def test_attribution_recommendation_ignores_blank_outer_source():
    recommendation = attribution_recommendation(
        {
            "module": " ",
            "result": {
                "module": "log_analyzer",
                "state": "CONTINUE",
                "result": ["ERRORS NOT FOUND"],
            },
        }
    )

    assert recommendation.action == "CONTINUE"
    assert recommendation.reason == "ERRORS NOT FOUND"
    assert recommendation.source == "log_analyzer"


def test_attribution_recommendation_warns_on_full_dict_fallback(caplog):
    caplog.set_level(
        logging.WARNING,
        logger="nvidia_resiliency_ext.attribution.orchestration.llm_output",
    )

    recommendation = attribution_recommendation({"unexpected": "STOP - DONT RESTART"})

    assert recommendation.action == "STOP"
    assert recommendation.reason == ""
    assert "falling through to string matching" in caplog.text
