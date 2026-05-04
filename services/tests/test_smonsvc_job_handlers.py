# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from nvidia_resiliency_ext.services.smonsvc.job_handlers import log_attribution_result


def _job():
    return SimpleNamespace(job_id="123")


def test_log_attribution_result_uses_recommendation_over_raw_state(capsys):
    response = {
        "recommendation": {
            "action": "RESTART",
            "reason": "safe to restart failed run",
            "source": "log_analyzer",
        },
        "result": {
            "module": "log_analyzer",
            "state": "STOP",
            "result_id": "abcdef0123456789",
            "result": ["raw backend text"],
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
