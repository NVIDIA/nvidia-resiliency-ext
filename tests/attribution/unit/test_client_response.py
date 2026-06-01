from nvidia_resiliency_ext.attribution import AttrSvcResult, parse_attrsvc_response


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


def test_parse_attrsvc_response_uses_serialized_recommendation():
    parsed = parse_attrsvc_response(
        {
            "status": "completed",
            "recommendation": {
                "action": "STOP",
                "reason": "standard stop",
                "source": "log_analyzer",
            },
            "result": {
                "module": "log_analyzer",
                "result": [_item("raw backend payload", "UNKNOWN")],
            },
        }
    )

    assert parsed.status == "completed"
    assert isinstance(parsed, AttrSvcResult)
    assert "state" not in parsed.result
    assert parsed.recommendation.action == "STOP"
    assert parsed.recommendation.reason == "standard stop"
    assert parsed.recommendation.source == "log_analyzer"
    assert parsed.should_stop is True


def test_parse_attrsvc_response_missing_recommendation_is_unknown():
    parsed = parse_attrsvc_response(
        {
            "result": {
                "module": "log_analyzer",
                "result": [_item("RESTART IMMEDIATE", "RESTART")],
            },
        }
    )

    assert parsed.status == "completed"
    assert parsed.recommendation.action == "UNKNOWN"
    assert parsed.recommendation.reason == ""
    assert parsed.should_stop is False


def test_parse_attrsvc_response_uses_explicit_recommendation_over_inner_result():
    parsed = parse_attrsvc_response(
        {
            "status": "completed",
            "recommendation": {
                "action": "CONTINUE",
                "reason": "top-level decision",
                "source": "log_analyzer",
            },
            "result": {
                "module": "log_analyzer",
                "state": "STOP",
                "result": [_item("STOP - DONT RESTART", "STOP")],
            },
        }
    )

    assert parsed.recommendation.action == "CONTINUE"
    assert parsed.recommendation.reason == "top-level decision"
    assert parsed.should_stop is False


def test_parse_attrsvc_response_keeps_client_log_path():
    parsed = parse_attrsvc_response(
        {
            "status": "completed",
            "recommendation": {
                "action": "CONTINUE",
                "reason": "no terminal issue",
                "source": "log_analyzer",
            },
            "result": {"module": "log_analyzer"},
        },
        log_path="/tmp/train.log",
    )

    assert parsed.log_path == "/tmp/train.log"
    assert parsed.recommendation_reason == "no terminal issue"


def test_attrsvc_result_formats_smon_summary():
    parsed = parse_attrsvc_response(
        {
            "status": "completed",
            "mode": "splitlog",
            "log_file": "/tmp/cycle-0.log",
            "wl_restart": 0,
            "sched_restarts": 2,
            "recommendation": {
                "action": "RESTART",
                "reason": "safe to restart",
                "source": "log_analyzer",
            },
            "result": {
                "module": "log_analyzer",
                "result_id": "abcdef0123456789",
                "result": [
                    _item("RESTART IMMEDIATE", "RESTART"),
                    _item("details", "UNKNOWN"),
                ],
            },
        },
        log_path="/tmp/slurm.out",
    )

    summary = parsed.format_summary(prefix="[123] ")

    assert "[123] Attribution result:" in summary
    assert "Mode: splitlog (wl_restart 0/2)" in summary
    assert "Slurm output: /tmp/slurm.out" in summary
    assert "Analyzed log: /tmp/cycle-0.log" in summary
    assert "Result ID: abcdef0123456789" in summary
    assert "Recommendation: RESTART" in summary
    assert "Reason: safe to restart" in summary
    assert "Attribution: RESTART IMMEDIATE | details" in summary


def test_attrsvc_result_formats_launcher_log_message():
    parsed = parse_attrsvc_response(
        {
            "status": "completed",
            "recommendation": {
                "action": "CONTINUE",
                "reason": "training cycle still running",
                "source": "log_analyzer",
            },
            "result": {
                "module": "log_analyzer",
                "result": [_item("ERRORS NOT FOUND", "CONTINUE")],
            },
        },
        log_path="/tmp/train.log",
    )

    message = parsed.format_log_message()

    assert message == (
        "AttrSvcResult for /tmp/train.log: status=completed "
        "recommendation=CONTINUE reason=training cycle still running "
        "should_stop=False result preview: ERRORS NOT FOUND"
    )
