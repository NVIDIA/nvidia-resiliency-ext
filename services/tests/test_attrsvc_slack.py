# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.attribution.orchestration.client_response import parse_attrsvc_response
from nvidia_resiliency_ext.services.attrsvc import slack as slack_mod
from nvidia_resiliency_ext.services.attrsvc.slack import (
    DEFAULT_NOTIFY_ACTIONS,
    AnalysisIdentity,
    SlackConfig,
    SlackNotifier,
    format_notification,
    latest_result_item,
    parse_notify_actions,
    run_name_from_log_path,
)


def _job(job_id="123", name="nemotron_pretrain", user="alice"):
    return AnalysisIdentity(job_id=job_id, run_name=name, user=user)


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
    monkeypatch.setenv("NVRX_ATTRSVC_SLACK_NOTIFY_ACTIONS", "STOP,RESTART")

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


def test_logsage_result_keeps_its_issue_wording():
    text = format_notification(_job(), _parsed())

    assert "Primary issues: [hardware]" in text
    assert "Secondary issues: [nccl]" in text
    assert "checkpoint corrupted" in text


def test_notification_carries_job_label_and_log_path():
    text = format_notification(_job(), _parsed())

    assert "`123 (nemotron_pretrain)`" in text
    assert "/lustre/logs/job.log" in text


def test_notification_without_job_name_uses_bare_job_id():
    text = format_notification(_job(name=''), _parsed())
    assert "`123`" in text


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


def test_restart_agent_response_reaches_slack(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_response(), log_path='/x.log'))

    assert len(client.messages) == 1
    assert "*NVRx attribution:* `STOP`" in client.messages[0]["text"]


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
            "l1_assessment": {
                "root_cause_assessment": {
                    "status": "supported_but_unconfirmed",
                    "summary": "Rank 3 exhausted device memory during the optimizer step.",
                    "plausible_causes": ["Activation memory grew after the batch-size change."],
                    "missing_evidence": ["No allocator snapshot around the failing step."],
                }
            },
            "schema_version": "restart_agent_response.v1",
        },
        "status": "completed",
        "recommendation": {
            "action": "STOP",
            # The Restart Agent reports its justification as the reason.
            "reason": "Line 28693 matched failure class observed_exception.",
            "source": "deterministic",
        },
    }


def test_restart_agent_fields_prefer_the_narrative_root_cause():
    parsed = parse_attrsvc_response(_restart_agent_payload(), log_path="/x.log")
    fields = slack_mod.attribution_fields(parsed)

    # The narrative summary is the headline; the typed record becomes evidence.
    assert fields.headline.startswith("Rank 3 exhausted device memory")
    assert "cuda_oom" in fields.evidence
    assert "line 28693" in fields.evidence
    assert "Line 28693" in fields.explanation


def test_restart_agent_fields_ignores_other_payload_shapes():
    assert slack_mod.restart_agent_fields({"module": "log_analyzer"}) is None
    assert slack_mod.restart_agent_fields(None) is None


def test_restart_agent_falls_back_to_typed_record_without_a_narrative():
    # Deterministic-only results carry no l1_assessment.
    payload = _restart_agent_payload()
    payload["result"].pop("l1_assessment")
    fields = slack_mod.attribution_fields(parse_attrsvc_response(payload, log_path="/x.log"))

    assert fields.headline == "cuda_oom: CUDA out of memory"
    assert fields.show_alternatives is False


def test_restart_agent_message_has_no_placeholder_text(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))

    text = client.messages[0]["text"]
    assert "No attribution available" not in text
    assert "No explanation available" not in text
    assert "Rank 3 exhausted device memory during the optimizer step." in text


def test_logsage_item_still_takes_precedence():
    fields = slack_mod.attribution_fields(_parsed())

    assert "Primary issues: [hardware]" in fields.headline
    assert fields.explanation == "checkpoint corrupted"
    assert fields.evidence == ""  # no typed record on the legacy path


def test_notification_does_not_repeat_reason_as_terminal_issue(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))

    # The Restart Agent justification is both the reason and the terminal issue.
    text = client.messages[0]["text"]
    assert text.count("Line 28693 matched failure class observed_exception.") == 1
    assert "*Reason:*" not in text


def test_notification_keeps_reason_when_it_differs_from_explanation(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), _parsed())

    text = client.messages[0]["text"]
    assert "*Reason:* terminal failure" in text
    assert "checkpoint corrupted" in text


# ─── narrative cause, evidence line, and gated hypotheses ───


def test_message_leads_with_the_narrative_cause_not_the_typed_label(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))
    text = client.messages[0]["text"]

    assert "Rank 3 exhausted device memory during the optimizer step." in text
    # The raw signature is no longer the headline.
    assert "Primary issues:" not in text


def test_message_carries_an_evidence_line(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))
    text = client.messages[0]["text"]

    assert "*Evidence:*" in text
    assert "`cuda_oom`" in text
    assert "line 28693" in text
    assert "rank 0" in text


def test_unconfirmed_results_show_causes_and_missing_evidence(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))
    text = client.messages[0]["text"]

    assert "*Plausible causes*" in text
    assert "supported_but_unconfirmed" in text
    assert "Activation memory grew after the batch-size change." in text
    assert "*Missing evidence:*" in text
    assert "No allocator snapshot around the failing step." in text


def test_confirmed_results_omit_causes_and_missing_evidence(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    payload = _restart_agent_payload()
    payload["result"]["l1_assessment"]["root_cause_assessment"][
        "status"
    ] = "established_by_current_log"

    notifier.notify(_job(), parse_attrsvc_response(payload, log_path='/x.log'))
    text = client.messages[0]["text"]

    # The log established the cause; alternatives would be noise.
    assert "*Plausible causes*" not in text
    assert "*Missing evidence:*" not in text
    assert "Rank 3 exhausted device memory" in text


def test_evidence_line_omits_absent_location_fields():
    fields = slack_mod.restart_agent_fields(
        {
            "schema_version": "restart_agent_response.v1",
            "primary_failure": {"failure_class": "cuda_oom", "causal_role": "unknown"},
        }
    )
    assert fields.evidence == "`cuda_oom`"  # no line/rank/phase, and 'unknown' role dropped


# ─── decision rationale: why L4 chose this action ───


def _policy_payload(**policy):
    base = {
        "base_rule": "general_retry",
        "applied_policy_context": None,
        "retry_budget_exhausted": False,
        "effective_policy": {"rule": "general_retry", "allowed_retries": 2},
        "failure_domain": "unknown",
        "failure_domain_confidence": 1,
        "retry_outlook_without_workload_change": "may_recover",
        "retry_outlook_status": "supported_but_unconfirmed",
        "retry_outlook_confidence": 67,
    }
    base.update(policy)
    return {
        "schema_version": "restart_agent_response.v1",
        "retry_policy": base,
        "l1_assessment": {"category_selection": {"category_id": 13, "category_confidence": 72}},
    }


def test_rationale_reports_rule_budget_and_category():
    r = slack_mod.decision_rationale(_policy_payload())
    lines = r.as_lines()

    assert "rule `general_retry`" in lines[0]
    assert "budget 2, not exhausted" in lines[0]
    assert "category 13" in lines[1]
    assert "NCCL remote process exited" in lines[1]
    assert "→ RESTART" in lines[1]
    assert "confidence 72" in lines[1]


def test_rationale_drops_unknown_claims():
    # failure_domain=unknown with confidence 1 is noise, not information.
    lines = " ".join(slack_mod.decision_rationale(_policy_payload()).as_lines())
    assert "failure domain" not in lines
    assert "retry outlook: may_recover" in lines


def test_rationale_names_the_policy_context_that_overrode_the_base_rule():
    payload = _policy_payload(
        applied_policy_context={"policy_context_id": "l1_category_confirmed_stop"},
        effective_policy={"rule": "workload_unrecoverable", "allowed_retries": 0},
        retry_budget_exhausted=True,
    )
    first = slack_mod.decision_rationale(payload).as_lines()[0]

    assert "`l1_category_confirmed_stop`" in first
    assert "overrides `general_retry`" in first
    assert "budget 0, exhausted" in first


def test_rationale_handles_a_bare_string_policy_context():
    payload = _policy_payload(applied_policy_context="cuda_oom_no_retry")
    assert "`cuda_oom_no_retry`" in slack_mod.decision_rationale(payload).as_lines()[0]


def test_rationale_without_category_still_reports_the_rule():
    payload = _policy_payload()
    payload.pop("l1_assessment")  # deterministic-only result
    lines = slack_mod.decision_rationale(payload).as_lines()

    assert any("general_retry" in line for line in lines)
    assert not any("category" in line for line in lines)


def test_rationale_is_empty_for_non_restart_agent_payloads():
    assert slack_mod.decision_rationale({"module": "log_analyzer"}).empty is True
    assert slack_mod.decision_rationale(None).empty is True


def test_message_includes_the_why_section(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    payload = _restart_agent_payload()
    payload["result"]["retry_policy"] = _policy_payload()["retry_policy"]

    notifier.notify(_job(), parse_attrsvc_response(payload, log_path='/x.log'))
    text = client.messages[0]["text"]

    assert "*Why STOP:*" in text
    assert "rule `general_retry`" in text


def test_message_omits_the_why_section_without_policy_data(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), _parsed())  # LogSage result, no retry_policy

    assert "*Why " not in client.messages[0]["text"]


# ─── identity recovered from the analysis, not from a SLURM job ───


@pytest.mark.parametrize(
    "path,job_id,expected",
    [
        (
            "/x/logs/nemotron4_derisking_nano_21t_phase1_3901259_date_26-09-21_time_10-45-42_cycle0.log",
            "3901259_26",
            "nemotron4_derisking_nano_21t_phase1",
        ),
        # The parent job ID is what the filename embeds, so array tasks agree.
        ("/x/logs/run_777_date_x.log", "777_3", "run"),
        ("/x/logs/run_777_date_x.log", "777+1", "run"),
    ],
)
def test_run_name_recovered_from_log_filename(path, job_id, expected):
    assert run_name_from_log_path(path, job_id) == expected


@pytest.mark.parametrize(
    "path,job_id",
    [
        ("/x/slurm_out/slurm-813606.out", "813606"),  # wrapper, no run prefix
        ("/x/logs/anything.log", ""),  # no job id to anchor on
        ("", "777"),
    ],
)
def test_run_name_returns_empty_when_unrecoverable(path, job_id):
    assert run_name_from_log_path(path, job_id) == ""


def test_identity_label_falls_back_to_bare_job_id():
    assert AnalysisIdentity(job_id="777").label == "777"
    assert AnalysisIdentity(job_id="777", run_name="my_run").label == "777 (my_run)"
    assert AnalysisIdentity().label == "unknown"


def test_config_from_settings_uses_attrsvc_settings(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("NVRX_ATTRSVC_SLACK_NOTIFY_ACTIONS", raising=False)
    settings = SimpleNamespace(SLACK_BOT_TOKEN="xoxb-from-settings", SLACK_CHANNEL=" C123 ")

    config = SlackConfig.from_settings(settings)

    assert config.token == "xoxb-from-settings"
    assert config.channel == "C123"
    assert config.configured is True


def test_config_from_settings_defers_to_the_key_file_when_token_is_empty(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
    settings = SimpleNamespace(SLACK_BOT_TOKEN="", SLACK_CHANNEL="C123")

    assert SlackConfig.from_settings(settings).token == "xoxb-from-env"


# ─── the backend notifies, so both deployment modes are covered ───


def test_backend_notifies_on_terminal_completion(monkeypatch):
    """Both smonsvc and inline NVRx reach Slack through this one call site."""
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        RestartAgentServiceBackend,
    )

    sent = []

    class _Recorder:
        enabled = True

        def notify(self, identity, result):
            sent.append((identity, result))
            return True

    entry = SimpleNamespace(
        job_id="3901259_26",
        user="dnarayanan",
        log_path="/x/logs/nemotron4_derisking_nano_21t_phase1_3901259_date_x_cycle0.log",
    )
    public = SimpleNamespace(
        result={"schema_version": "restart_agent_response.v1", "decision": "STOP"},
        status="completed",
        recommendation={"action": "STOP", "reason": "terminal", "source": "deterministic"},
    )
    backend = SimpleNamespace(
        _slack_notifier=_Recorder(),
        _lock=threading.RLock(),
        _entries={"k": entry},
        _public_result=lambda e: public,
    )

    RestartAgentServiceBackend._notify_slack(backend, "k")

    assert len(sent) == 1
    identity, result = sent[0]
    assert identity.job_id == "3901259_26"
    assert identity.user == "dnarayanan"
    # attrsvc never sees the SLURM job name; it is recovered from the log path.
    assert identity.run_name == "nemotron4_derisking_nano_21t_phase1"
    assert result.recommendation.action == "STOP"


def test_backend_notification_failure_does_not_break_analysis():
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        RestartAgentServiceBackend,
    )

    class _Exploding:
        enabled = True

        def notify(self, identity, result):
            raise RuntimeError("slack is down")

    backend = SimpleNamespace(
        _slack_notifier=_Exploding(),
        _lock=threading.RLock(),
        _entries={"k": SimpleNamespace(job_id="1", user="u", log_path="/x.log")},
        _public_result=lambda e: SimpleNamespace(result={}, status="completed", recommendation={}),
    )

    # Must return rather than propagate into the analysis path.
    RestartAgentServiceBackend._notify_slack(backend, "k")


def test_backend_skips_notification_when_slack_is_disabled():
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        RestartAgentServiceBackend,
    )

    called = []
    backend = SimpleNamespace(
        _slack_notifier=SimpleNamespace(enabled=False, notify=lambda *a: called.append(a)),
        _lock=threading.RLock(),
        _entries={},
        _public_result=lambda e: None,
    )

    RestartAgentServiceBackend._notify_slack(backend, "k")
    assert called == []
