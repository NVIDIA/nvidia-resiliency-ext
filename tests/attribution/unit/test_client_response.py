from nvidia_resiliency_ext.attribution import AttrSvcResult, parse_attrsvc_response


def test_parse_attrsvc_response_uses_standard_recommendation():
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
                "state": "CONTINUE",
                "result": ["raw backend payload"],
            },
        }
    )

    assert parsed.status == "completed"
    assert isinstance(parsed, AttrSvcResult)
    assert parsed.result["state"] == "CONTINUE"
    assert parsed.recommendation.action == "STOP"
    assert parsed.recommendation.reason == "standard stop"
    assert parsed.recommendation.source == "log_analyzer"
    assert parsed.should_stop is True


def test_parse_attrsvc_response_falls_back_to_raw_result():
    parsed = parse_attrsvc_response(
        {
            "result": {
                "module": "log_analyzer",
                "state": "CONTINUE",
                "result": ["RESTART IMMEDIATE"],
            },
        }
    )

    assert parsed.status == "completed"
    assert parsed.recommendation.action == "RESTART"
    assert parsed.recommendation.reason == "RESTART IMMEDIATE"
    assert parsed.should_stop is False


def test_parse_attrsvc_response_fallback_uses_inner_result_only():
    parsed = parse_attrsvc_response(
        {
            "status": "completed",
            "state": "STOP",
            "result": {
                "module": "log_analyzer",
                "state": "CONTINUE",
                "result": ["ERRORS NOT FOUND"],
            },
        }
    )

    assert parsed.recommendation.action == "CONTINUE"
    assert parsed.recommendation.reason == "ERRORS NOT FOUND"
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
                "result": ["RESTART IMMEDIATE", "details"],
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
                "result": ["ERRORS NOT FOUND"],
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
