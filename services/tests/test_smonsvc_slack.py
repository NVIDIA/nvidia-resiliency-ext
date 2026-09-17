# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.attribution.orchestration.client_response import parse_attrsvc_response
from nvidia_resiliency_ext.services.smonsvc import slack as slack_mod
from nvidia_resiliency_ext.services.smonsvc.job_handlers import log_attribution_result
from nvidia_resiliency_ext.services.smonsvc.slack import (
    DEFAULT_NOTIFY_ACTIONS,
    SlackConfig,
    SlackNotifier,
    build_slack_record,
    latest_result_item,
    parse_notify_actions,
)


def _job(job_id="123", name="nemotron_pretrain", user="alice"):
    return SimpleNamespace(job_id=job_id, name=name, user=user)


def _item(primary_issues, explanation="checkpoint corrupted"):
    return {
        "raw_text": "raw backend text",
        "auto_resume": "STOP",
        "auto_resume_explanation": explanation,
        "attribution_text": "",
        "checkpoint_saved_flag": 0,
        "action": "STOP",
        "primary_issues": primary_issues,
        "secondary_issues": ["nccl"],
    }


def _response(action="STOP", items=None, reason="terminal failure"):
    return {
        "recommendation": {"action": action, "reason": reason, "source": "log_analyzer"},
        "result": {
            "module": "log_analyzer",
            "result": items if items is not None else [_item(["hardware"])],
        },
    }


def _parsed(action="STOP", items=None, log_path="/lustre/logs/job.log"):
    return parse_attrsvc_response(_response(action=action, items=items), log_path=log_path)


class _StubClient:
    def __init__(self, error=None):
        self.messages = []
        self._error = error

    def chat_postMessage(self, channel, text):
        if self._error is not None:
            raise self._error
        self.messages.append({"channel": channel, "text": text})


def _notifier(monkeypatch, client=None, **config_kwargs):
    """Build an enabled notifier backed by a stub Slack client."""
    monkeypatch.setattr(slack_mod, "HAS_SLACK", True)
    monkeypatch.setattr(slack_mod, "get_slack_user_id", lambda user, token: None)
    config = SlackConfig(token="xoxb-test", channel="#trng-alerts", **config_kwargs)
    notifier = SlackNotifier(config)
    notifier._client = client if client is not None else _StubClient()
    return notifier


# ─── configuration ───


def test_parse_notify_actions_defaults_to_stop():
    assert parse_notify_actions(None) == frozenset(DEFAULT_NOTIFY_ACTIONS)
    assert parse_notify_actions("   ") == frozenset(DEFAULT_NOTIFY_ACTIONS)


@pytest.mark.parametrize("raw", ["STOP,RESTART", "stop restart", " stop , restart "])
def test_parse_notify_actions_accepts_separators_and_case(raw):
    assert parse_notify_actions(raw) == frozenset({"STOP", "RESTART"})


def test_parse_notify_actions_drops_unknown_entries_instead_of_normalizing():
    # normalize_recommendation_action() would map "bogus" to UNKNOWN, which would
    # page on every unattributed job. Unknown entries must be dropped instead.
    assert parse_notify_actions("STOP,bogus") == frozenset({"STOP"})
    assert parse_notify_actions("bogus") == frozenset(DEFAULT_NOTIFY_ACTIONS)


def test_config_from_env_reads_token_channel_and_actions(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
    monkeypatch.setenv("SLACK_CHANNEL", " #trng-alerts ")
    monkeypatch.setenv("NVRX_SMONSVC_SLACK_NOTIFY_ACTIONS", "STOP,RESTART")

    config = SlackConfig.from_env()

    assert config.token == "xoxb-from-env"
    assert config.channel == "#trng-alerts"
    assert config.notify_actions == frozenset({"STOP", "RESTART"})
    assert config.configured is True


def test_config_is_not_configured_without_channel(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
    monkeypatch.delenv("SLACK_CHANNEL", raising=False)

    assert SlackConfig.from_env().configured is False


def test_notifier_disabled_without_slack_sdk(monkeypatch):
    monkeypatch.setattr(slack_mod, "HAS_SLACK", False)
    notifier = SlackNotifier(SlackConfig(token="xoxb-test", channel="#alerts"))

    assert notifier.enabled is False
    assert "slack-sdk" in notifier.describe()
    assert notifier.notify(_job(), _parsed()) is False


def test_notifier_describe_reports_channel_and_actions(monkeypatch):
    notifier = _notifier(monkeypatch)
    assert "#trng-alerts" in notifier.describe()
    assert "STOP" in notifier.describe()


# ─── record and message shape ───


def test_latest_result_item_returns_last_parsable_cycle():
    parsed = _parsed(items=[_item(["software"]), _item(["hardware"])])
    item = latest_result_item(parsed)

    assert item is not None
    assert item.primary_issues == ["hardware"]


def test_latest_result_item_returns_none_without_items():
    assert latest_result_item(_parsed(items=[])) is None


def test_build_slack_record_uses_canonical_dataflow_keys():
    record = build_slack_record(_job(), _parsed())

    assert record["s_job_id"] == "123 (nemotron_pretrain)"
    assert record["s_user"] == "alice"
    assert record["s_log_path"] == "/lustre/logs/job.log"
    assert record["s_recommendation_action"] == "STOP"
    assert record["s_recommendation_source"] == "log_analyzer"
    assert record["s_primary_issues"] == ["hardware"]
    assert record["s_auto_resume_explanation"] == "checkpoint corrupted"
    assert json.loads(record["s_attribution_result_json"])["secondary_issues"] == ["nccl"]


def test_build_slack_record_without_job_name_uses_bare_job_id():
    record = build_slack_record(_job(name=""), _parsed())
    assert record["s_job_id"] == "123"


def test_notify_posts_message_with_action_reason_and_details(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    assert notifier.notify(_job(), _parsed()) is True

    assert len(client.messages) == 1
    message = client.messages[0]
    assert message["channel"] == "#trng-alerts"
    assert "*NVRx attribution:* `STOP`" in message["text"]
    assert "terminal failure" in message["text"]
    assert "123 (nemotron_pretrain)" in message["text"]
    assert "/lustre/logs/job.log" in message["text"]
    assert "checkpoint corrupted" in message["text"]


def test_notify_mentions_the_job_owner_when_resolvable(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    monkeypatch.setattr(slack_mod, "get_slack_user_id", lambda user, token: "U123")

    notifier.notify(_job(), _parsed())

    assert client.messages[0]["text"].endswith("<@U123>")


# ─── notify gating and failure handling ───


def test_notify_skips_actions_outside_the_configured_set(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    assert notifier.notify(_job(), _parsed(action="RESTART")) is False
    assert client.messages == []
    assert notifier.stats.skipped_action == 1
    assert notifier.stats.attempts == 0


def test_notify_sends_for_configured_non_default_action(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client, notify_actions=frozenset({"RESTART"}))

    assert notifier.notify(_job(), _parsed(action="RESTART")) is True
    assert notifier.stats.sent == 1


def test_notify_counts_failures_without_raising(monkeypatch):
    monkeypatch.setattr(slack_mod, "SlackApiError", RuntimeError)
    notifier = _notifier(monkeypatch, client=_StubClient(error=RuntimeError("channel_not_found")))

    assert notifier.notify(_job(), _parsed()) is False
    assert notifier.stats.attempts == 1
    assert notifier.stats.failed == 1
    assert notifier.stats.sent == 0


def test_stats_as_dict_exposes_all_counters(monkeypatch):
    notifier = _notifier(monkeypatch)
    notifier.notify(_job(), _parsed())
    notifier.notify(_job(), _parsed(action="CONTINUE"))

    assert notifier.stats.as_dict() == {
        "attempts": 1,
        "sent": 1,
        "failed": 0,
        "skipped_action": 1,
    }


# ─── integration with the result handler ───


def test_log_attribution_result_notifies_slack(monkeypatch, capsys):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    log_attribution_result(_job(), "/lustre/logs/job.log", _response(), slack_notifier=notifier)

    assert "Recommendation: STOP" in capsys.readouterr().out
    assert len(client.messages) == 1


def test_log_attribution_result_without_notifier_is_unchanged(capsys):
    log_attribution_result(_job(), "/lustre/logs/job.log", _response())

    assert "Recommendation: STOP" in capsys.readouterr().out


def test_log_attribution_result_survives_notifier_errors(monkeypatch, capsys):
    class _Exploding:
        def notify(self, job, parsed):
            raise RuntimeError("slack is down")

    log_attribution_result(_job(), "/lustre/logs/job.log", _response(), slack_notifier=_Exploding())

    # The summary is still printed: alerting is best effort and must not break
    # the monitor's result-processing loop.
    assert "Recommendation: STOP" in capsys.readouterr().out


def test_log_attribution_result_notifies_on_timeout_when_configured(monkeypatch, capsys):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client, notify_actions=frozenset({"TIMEOUT"}))

    log_attribution_result(
        _job(),
        "/lustre/logs/job.log",
        _response(action="TIMEOUT", items=[]),
        slack_notifier=notifier,
    )

    assert "Attribution timeout" in capsys.readouterr().out
    assert len(client.messages) == 1


# ─── restart-agent responses carry no "module" key ───


def _restart_agent_response(action="STOP"):
    """Shape produced by the direct Restart Agent backend: no module, no item list."""
    return {
        "result": {
            "decision": action,
            "decision_basis": "concrete_confirmation_retry_exhausted",
            "justification": "Line 28693 matched failure class observed_exception.",
            "schema_version": "restart_agent_response.v1",
        },
        "status": "completed",
        "recommendation": {
            "action": action,
            "reason": "Line 28693 matched failure class observed_exception.",
            "source": "deterministic",
        },
    }


def test_log_attribution_result_accepts_restart_agent_response(capsys):
    # The legacy guard required inner["module"], which no Restart Agent result
    # sets, so every direct-backend result was dropped before being reported.
    log_attribution_result(_job(), "/lustre/logs/job.log", _restart_agent_response())

    output = capsys.readouterr().out
    assert "Recommendation: STOP" in output
    assert "unrecognized" not in output


def test_restart_agent_response_reaches_slack(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    log_attribution_result(
        _job(), "/lustre/logs/job.log", _restart_agent_response(), slack_notifier=notifier
    )

    assert len(client.messages) == 1
    assert "*NVRx attribution:* `STOP`" in client.messages[0]["text"]


def test_log_attribution_result_still_rejects_unusable_responses(capsys, caplog):
    log_attribution_result(_job(), "/lustre/logs/job.log", {"result": {}})

    assert capsys.readouterr().out == ""
    assert "empty or unrecognized" in caplog.text


# ─── restart-agent payloads have no LogSage item list ───


def _restart_agent_payload():
    return {
        "result": {
            "decision": "STOP",
            "decision_basis": "concrete_confirmation_retry_exhausted",
            "justification": "Line 28693 matched failure class observed_exception.",
            "primary_failure": {
                "failure_class": "cuda_oom",
                "signature": "CUDA out of memory:",
                "line": 28693,
                "rank": "0",
            },
            "secondary_failures": [
                {"failure_class": "observed_exception", "signature": "RuntimeError:"},
                {"failure_class": "observed_exception", "signature": "RuntimeError:"},
            ],
            "schema_version": "restart_agent_response.v1",
        },
        "status": "completed",
        "recommendation": {"action": "STOP", "reason": "cuda oom", "source": "deterministic"},
    }


def test_restart_agent_fields_extracts_failures_and_justification():
    parsed = parse_attrsvc_response(_restart_agent_payload(), log_path="/x.log")
    fields = slack_mod.attribution_fields(parsed)

    assert fields.primary_issues == ["cuda_oom: CUDA out of memory"]
    # Duplicate secondary failures collapse to one label.
    assert fields.secondary_issues == ["observed_exception: RuntimeError"]
    assert "Line 28693" in fields.explanation
    assert "concrete_confirmation_retry_exhausted" in fields.explanation


def test_restart_agent_fields_ignores_other_payload_shapes():
    assert slack_mod.restart_agent_fields({"module": "log_analyzer"}) is None
    assert slack_mod.restart_agent_fields(None) is None


def test_build_slack_record_populates_body_from_restart_agent_payload():
    parsed = parse_attrsvc_response(_restart_agent_payload(), log_path="/x.log")
    record = build_slack_record(_job(), parsed)

    assert record["s_primary_issues"] == ["cuda_oom: CUDA out of memory"]
    assert "Line 28693" in record["s_auto_resume_explanation"]
    assert json.loads(record["s_attribution_result_json"])["secondary_issues"] == [
        "observed_exception: RuntimeError"
    ]


def test_restart_agent_message_has_no_placeholder_text(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    log_attribution_result(_job(), "/x.log", _restart_agent_payload(), slack_notifier=notifier)

    text = client.messages[0]["text"]
    assert "No attribution available" not in text
    assert "No explanation available" not in text
    assert "cuda_oom: CUDA out of memory" in text


def test_logsage_item_still_takes_precedence():
    parsed = _parsed()
    fields = slack_mod.attribution_fields(parsed)

    assert fields.primary_issues == ["hardware"]
    assert fields.explanation == "checkpoint corrupted"
