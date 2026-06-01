# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from nvidia_resiliency_ext.attribution.orchestration.progressive import ANALYSIS_INTENT_TERMINAL
from nvidia_resiliency_ext.services.smonsvc.job_handlers import (
    fetch_results,
    log_attribution_result,
)
from nvidia_resiliency_ext.services.smonsvc.models import MonitorState


def _job():
    return SimpleNamespace(job_id="123")


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _AttrsvcClient:
    def __init__(self, post_error: str | None = None):
        self.calls = []
        self._post_error = post_error

    def request_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["method"] == "POST":
            if self._post_error:
                kwargs["on_client_error"](self._post_error)
                return
            kwargs["on_success"](_Response({"submitted": True, "mode": "single"}))
            return
        kwargs["on_success"](
            _Response(
                {
                    "recommendation": {
                        "action": "RESTART",
                        "reason": "safe to restart",
                        "source": "log_analyzer",
                    },
                    "result": {
                        "module": "log_analyzer",
                        "result": [],
                    },
                }
            )
        )


def _item(raw_text, reason_code):
    return {
        "raw_text": raw_text,
        "auto_resume": raw_text.split("\n", 1)[0],
        "auto_resume_explanation": "",
        "attribution_text": "",
        "checkpoint_saved_flag": 0,
        "primary_issues": [],
        "secondary_issues": [],
    }


def test_log_attribution_result_uses_recommendation_over_raw_state(capsys):
    response = {
        "recommendation": {
            "action": "RESTART",
            "reason": "safe to restart failed run",
            "source": "log_analyzer",
        },
        "result": {
            "module": "log_analyzer",
            "result_id": "abcdef0123456789",
            "result": [_item("raw backend text", "UNKNOWN")],
        },
    }

    log_attribution_result(_job(), "/tmp/job.log", response)

    output = capsys.readouterr().out
    assert "Recommendation: RESTART" in output
    assert "Reason: safe to restart failed run" in output
    assert "State:" not in output


def test_log_attribution_result_uses_recommendation_for_timeout(capsys):
    response = {
        "recommendation": {
            "action": "TIMEOUT",
            "reason": "LLM analysis timed out",
            "source": "log_analyzer",
        },
        "result": {
            "module": "log_analyzer",
            "state": "CONTINUE",
            "result": [],
        },
    }

    log_attribution_result(_job(), "/tmp/job.log", response)

    output = capsys.readouterr().out
    assert "Attribution timeout" in output
    assert "Recommendation: TIMEOUT" in output
    assert "Reason: LLM analysis timed out" in output
    assert "State:" not in output


def test_fetch_results_signals_terminal_analysis_before_get():
    job = SimpleNamespace(
        job_id="123",
        user="alice",
        result_fetched=False,
        terminal_signaled=False,
    )
    state = MonitorState()
    attrsvc_client = _AttrsvcClient()

    fetch_results(job, "/tmp/job.log", state, attrsvc_client)

    assert [call["method"] for call in attrsvc_client.calls] == ["POST", "GET"]
    assert attrsvc_client.calls[0]["analysis_intent"] == ANALYSIS_INTENT_TERMINAL
    assert attrsvc_client.calls[0]["user"] == "alice"
    assert job.terminal_signaled is True
    assert job.result_fetched is True
    assert state.results_fetched == 1


def test_fetch_results_skips_terminal_signal_once_sent():
    job = SimpleNamespace(
        job_id="123",
        user="alice",
        result_fetched=False,
        terminal_signaled=True,
    )
    state = MonitorState()
    attrsvc_client = _AttrsvcClient()

    fetch_results(job, "/tmp/job.log", state, attrsvc_client)

    assert [call["method"] for call in attrsvc_client.calls] == ["GET"]


def test_fetch_results_terminal_signal_failure_does_not_count_path_error():
    job = SimpleNamespace(
        job_id="123",
        user="alice",
        result_fetched=False,
        terminal_signaled=False,
    )
    state = MonitorState()
    attrsvc_client = _AttrsvcClient(post_error="permission denied")

    fetch_results(job, "/tmp/job.log", state, attrsvc_client)

    assert [call["method"] for call in attrsvc_client.calls] == ["POST", "GET"]
    assert job.terminal_signaled is True
    assert job.result_fetched is True
    assert state.results_fetched == 1
    assert state.path_errors_permission == 0
    assert state.path_errors_not_found == 0
    assert state.path_errors_empty == 0
    assert state.path_errors_other == 0


def test_fetch_results_terminal_signal_failure_is_not_retried():
    job = SimpleNamespace(
        job_id="123",
        user="alice",
        result_fetched=False,
        terminal_signaled=False,
    )
    state = MonitorState()
    attrsvc_client = _AttrsvcClient(post_error="max retries exceeded")

    fetch_results(job, "/tmp/job.log", state, attrsvc_client)
    fetch_results(job, "/tmp/job.log", state, attrsvc_client)

    assert [call["method"] for call in attrsvc_client.calls] == ["POST", "GET", "GET"]
