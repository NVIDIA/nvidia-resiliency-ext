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
    format_details,
    format_summary,
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

    def chat_postMessage(self, channel, text, thread_ts=None):
        if self._error is not None:
            raise self._error
        self.messages.append({"channel": channel, "text": text, "thread_ts": thread_ts})
        return {"ok": True, "ts": f"171000000.{len(self.messages):06d}"}

    @property
    def alert_text(self):
        """Summary and its thread reply, as one blob for content assertions."""
        return "\n".join(m["text"] for m in self.messages)

    @property
    def summaries(self):
        return [m for m in self.messages if m["thread_ts"] is None]

    @property
    def replies(self):
        return [m for m in self.messages if m["thread_ts"] is not None]


def summary_ts(client):
    """The ts the stub returned for the summary, which replies must thread onto."""
    return "171000000.000001"


def _notifier(monkeypatch, client=None, **config_kwargs):
    """Build an enabled notifier backed by a stub Slack client."""
    monkeypatch.setattr(slack_mod, "HAS_SLACK", True)
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
    text = format_summary(_job(), _parsed()) + format_details(_job(), _parsed())

    assert "Primary issues: [hardware]" in text
    assert "Secondary issues: [nccl]" in text
    assert "checkpoint corrupted" in text


def test_notification_carries_job_label_and_log_path():
    text = format_summary(_job(), _parsed()) + format_details(_job(), _parsed())

    assert "*Job ID:* `123`" in text
    assert "*Job Name:* `nemotron_pretrain`" in text
    assert "/lustre/logs/job.log" in text


def test_notification_without_job_name_uses_bare_job_id():
    text = format_summary(_job(name=''), _parsed())
    assert "`123`" in text


def test_notify_posts_message_with_action_reason_and_details(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    assert notifier.notify(_job(), _parsed()) is True

    assert len(client.summaries) == 1
    assert len(client.replies) == 1

    summary = client.summaries[0]
    assert summary["channel"] == "#trng-alerts"
    # The channel line stays scannable: decision, job, log path, nothing else.
    assert "*NVRx attribution:* `STOP`" in summary["text"]
    assert "*Job ID:* `123`" in summary["text"]
    assert "*Job Name:* `nemotron_pretrain`" in summary["text"]
    assert "/lustre/logs/job.log" in summary["text"]
    assert "checkpoint corrupted" not in summary["text"]

    reply = client.replies[0]
    assert reply["thread_ts"] == summary_ts(client)
    assert "terminal failure" in reply["text"]
    assert "checkpoint corrupted" in reply["text"]


class _LookupClient(_StubClient):
    """Stub that also answers the email lookup used for @ mentions."""

    def __init__(self, user_id="U123", email_error=None):
        super().__init__()
        self._user_id = user_id
        self._email_error = email_error
        self.looked_up = []

    def users_lookupByEmail(self, email):
        self.looked_up.append(email)
        if self._email_error is not None:
            raise self._email_error
        return {"user": {"id": self._user_id}} if self._user_id else {}


def test_notify_mentions_the_job_owner_when_a_domain_is_configured(monkeypatch):
    client = _LookupClient()
    notifier = _notifier(monkeypatch, client=client, email_domain="example.com")

    notifier.notify(_job(), _parsed())

    assert client.looked_up == ["alice@example.com"]
    assert client.summaries[0]["text"].endswith("<@U123>")


def test_no_mention_is_attempted_without_a_configured_domain(monkeypatch):
    # This library runs outside NVIDIA; an owner name is not an email address.
    client = _LookupClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), _parsed())

    assert client.looked_up == []
    assert "<@" not in client.alert_text
    assert len(client.summaries) == 1


def test_failed_user_lookup_still_sends_the_alert(monkeypatch):
    monkeypatch.setattr(slack_mod, "SlackApiError", RuntimeError)
    client = _LookupClient(email_error=RuntimeError("users_not_found"))
    notifier = _notifier(monkeypatch, client=client, email_domain="example.com")

    assert notifier.notify(_job(), _parsed()) is True
    assert "<@" not in client.alert_text


def test_email_domain_strips_a_leading_at(monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_SLACK_EMAIL_DOMAIN", "@example.com")
    assert SlackConfig.from_env().email_domain == "example.com"


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

    assert len(client.summaries) == 1
    assert "*NVRx attribution:* `STOP`" in client.alert_text


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

    text = client.alert_text
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
    text = client.alert_text
    assert text.count("Line 28693 matched failure class observed_exception.") == 1
    assert "*Reason:*" not in text


def test_notification_keeps_reason_when_it_differs_from_explanation(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), _parsed())

    text = client.alert_text
    assert "*Reason:* terminal failure" in text
    assert "checkpoint corrupted" in text


# ─── narrative cause, evidence line, and gated hypotheses ───


def test_message_leads_with_the_narrative_cause_not_the_typed_label(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))
    text = client.alert_text

    assert "Rank 3 exhausted device memory during the optimizer step." in text
    # The raw signature is no longer the headline.
    assert "Primary issues:" not in text


def test_message_carries_an_evidence_line(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))
    text = client.alert_text

    assert "*Evidence:*" in text
    assert "`cuda_oom`" in text
    assert "line 28693" in text
    assert "rank 0" in text


def test_unconfirmed_results_show_causes_and_missing_evidence(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), parse_attrsvc_response(_restart_agent_payload(), log_path='/x.log'))
    text = client.alert_text

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
    text = client.alert_text

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


def _ledgers(**overrides):
    """The four L4 ledgers, defaulting to an unexhausted same-root ceiling."""
    base = {
        "exhausted_by": [],
        "general_root_ceiling": {
            "ledger_id": "general_root_ceiling",
            "applicable": True,
            "allowed_retries": 2,
            "matching_prior_attempts": 0,
            "exhausted": False,
        },
        "selected_policy_ledger": None,
        "job_no_progress_guard": {
            "ledger_id": "job_no_progress_guard",
            "applicable": True,
            "allowed_retries": 2,
            "matching_prior_attempts": 0,
            "exhausted": False,
        },
        "job_unknown_progress_guard": {
            "ledger_id": "job_unknown_progress_guard",
            "applicable": True,
            "allowed_retries": 3,
            "matching_prior_attempts": 1,
            "exhausted": False,
        },
    }
    base.update(overrides)
    return base


def test_rationale_names_the_guard_that_exhausted_not_the_rule_budget():
    # Job 4125659: the same-root ceiling had budget to spare; what stopped the
    # job was the unverifiable-progress guard, which ignores the root cause.
    payload = _policy_payload(
        retry_budget_exhausted=True,
        **_ledgers(
            exhausted_by=["job_unknown_progress_guard"],
            job_unknown_progress_guard={
                "ledger_id": "job_unknown_progress_guard",
                "applicable": True,
                "allowed_retries": 3,
                "matching_prior_attempts": 4,
                "exhausted": True,
            },
        ),
    )
    lines = slack_mod.decision_rationale(payload).as_lines()

    # The rule's own budget did not run out and must not claim it did.
    assert "budget 2, not exhausted" in lines[0]
    assert "`job_unknown_progress_guard`" in lines[1]
    assert "4 attempts with unverifiable progress for this job" in lines[1]
    assert "budget 3" in lines[1]
    assert "regardless of root cause" in lines[1]


def test_rationale_does_not_repeat_the_rules_own_exhausted_ledger():
    payload = _policy_payload(
        retry_budget_exhausted=True,
        **_ledgers(
            exhausted_by=["general_root_ceiling"],
            general_root_ceiling={
                "ledger_id": "general_root_ceiling",
                "applicable": True,
                "allowed_retries": 2,
                "matching_prior_attempts": 2,
                "exhausted": True,
            },
        ),
    )
    lines = slack_mod.decision_rationale(payload).as_lines()

    assert "budget 2, exhausted" in lines[0]
    assert not any("budget stop" in line for line in lines)


def test_rationale_reports_a_guard_alongside_an_exhausted_rule_budget():
    payload = _policy_payload(
        retry_budget_exhausted=True,
        **_ledgers(
            exhausted_by=["general_root_ceiling", "job_no_progress_guard"],
            general_root_ceiling={
                "ledger_id": "general_root_ceiling",
                "applicable": True,
                "allowed_retries": 2,
                "matching_prior_attempts": 2,
                "exhausted": True,
            },
            job_no_progress_guard={
                "ledger_id": "job_no_progress_guard",
                "applicable": True,
                "allowed_retries": 2,
                "matching_prior_attempts": 3,
                "exhausted": True,
            },
        ),
    )
    lines = slack_mod.decision_rationale(payload).as_lines()

    assert "budget 2, exhausted" in lines[0]
    assert "`job_no_progress_guard`" in lines[1]
    assert "3 attempts with no progress for this job" in lines[1]


def test_rationale_credits_an_applicable_selected_ledger_as_the_rules_own():
    # With a policy context in force the selected ledger, not the general
    # ceiling, is the budget the rule line is quoting.
    payload = _policy_payload(
        applied_policy_context={"policy_context_id": "l1_category_confirmed_stop"},
        effective_policy={"rule": "workload_unrecoverable", "allowed_retries": 0},
        retry_budget_exhausted=True,
        **_ledgers(
            exhausted_by=["selected_policy_ledger"],
            selected_policy_ledger={
                "ledger_id": "selected_policy_ledger",
                "applicable": True,
                "allowed_retries": 0,
                "matching_prior_attempts": 0,
                "exhausted": True,
            },
        ),
    )
    lines = slack_mod.decision_rationale(payload).as_lines()

    assert "budget 0, exhausted" in lines[0]
    assert not any("budget stop" in line for line in lines)


def test_rationale_falls_back_to_the_aggregate_flag_without_ledgers():
    # Payloads predating exhausted_by still have to render something sane.
    payload = _policy_payload(retry_budget_exhausted=True)
    lines = slack_mod.decision_rationale(payload).as_lines()

    assert "budget 2, exhausted" in lines[0]
    assert not any("budget stop" in line for line in lines)


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
    text = client.alert_text

    assert "*Why STOP:*" in text
    assert "rule `general_retry`" in text


def test_message_omits_the_why_section_without_policy_data(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), _parsed())  # LogSage result, no retry_policy

    assert "*Why " not in client.alert_text


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
        cycle_id=0,
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
        _entries={"k": SimpleNamespace(job_id="1", user="u", log_path="/x.log", cycle_id=None)},
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


# ─── cluster identity ───


def test_message_names_the_cluster(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client, cluster="oci-aga-slurm-1")

    notifier.notify(_job(), _parsed())

    assert "on *oci-aga-slurm-1*" in client.alert_text


def test_message_omits_the_cluster_when_unset(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), _parsed())

    assert " on *" not in client.alert_text


def test_config_from_settings_reads_the_cluster_name(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    settings = SimpleNamespace(
        SLACK_BOT_TOKEN="xoxb-x", SLACK_CHANNEL="C1", CLUSTER_NAME=" aws-cmh-slurm-1 "
    )
    assert SlackConfig.from_settings(settings).cluster == "aws-cmh-slurm-1"


# ─── "none of the categories matched" is an answer, not an absence ───


def test_category_zero_is_reported_rather_than_dropped():
    payload = _policy_payload()
    payload["l1_assessment"]["category_selection"] = {"category_id": 0, "category_confidence": 0}
    lines = " ".join(slack_mod.decision_rationale(payload).as_lines())

    # L1 considered the taxonomy and said none applies; silence would read as
    # "the model never categorised", which is a different fact.
    assert "none of the listed categories matched" in lines


def test_absent_category_selection_says_nothing():
    payload = _policy_payload()
    payload.pop("l1_assessment")
    lines = " ".join(slack_mod.decision_rationale(payload).as_lines())

    assert "category" not in lines


# ─── the lines at the reported failure ───


def _log_with(tmp_path, total=40, marker_line=30, marker="RuntimeError: CUDA error"):
    p = tmp_path / "run_1_date_x.log"
    with p.open("w") as fh:
        for i in range(1, total + 1):
            fh.write(f"{marker}\n" if i == marker_line else f"ordinary output {i}\n")
    return p


def test_log_excerpt_starts_at_the_reported_line(tmp_path):
    log = _log_with(tmp_path)
    lines = slack_mod.log_excerpt(str(log), 30)

    assert len(lines) == 3
    assert lines[0].startswith("30: RuntimeError: CUDA error")
    assert lines[1].startswith("31: ")
    assert lines[2].startswith("32: ")


def test_log_excerpt_truncates_a_very_long_line(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("A" * 5000 + "\n")
    line = slack_mod.log_excerpt(str(p), 1)[0]

    assert line.endswith("…")
    assert len(line) < 300


def test_log_excerpt_stops_at_end_of_file(tmp_path):
    log = _log_with(tmp_path, total=31)
    assert len(slack_mod.log_excerpt(str(log), 30)) == 2


@pytest.mark.parametrize("bad", [None, 0, -5, "abc"])
def test_log_excerpt_rejects_unusable_line_numbers(tmp_path, bad):
    assert slack_mod.log_excerpt(str(_log_with(tmp_path)), bad) == []


def test_log_excerpt_on_a_missing_file_is_empty():
    assert slack_mod.log_excerpt("/no/such/file.log", 1) == []


def test_message_quotes_the_failure_lines(monkeypatch, tmp_path):
    log = _log_with(tmp_path)
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    payload = _restart_agent_payload()
    payload["result"]["primary_failure"]["line"] = 30

    notifier.notify(_job(), parse_attrsvc_response(payload, log_path=str(log)))
    text = client.alert_text

    assert "*Log at the failure:*" in text
    assert "30: RuntimeError: CUDA error" in text


def test_message_omits_the_excerpt_when_the_log_is_unreadable(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    payload = _restart_agent_payload()
    payload["result"]["primary_failure"]["line"] = 30

    notifier.notify(_job(), parse_attrsvc_response(payload, log_path="/no/such.log"))

    assert "*Log at the failure:*" not in client.alert_text


# ─── channel stays scannable; detail goes to the thread ───


def test_summary_is_four_fields(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client, cluster="aws-cmh-slurm-1")

    notifier.notify(_job(), _parsed())
    lines = client.summaries[0]["text"].splitlines()

    # No retry policy in this result, so there is no reason to state: header,
    # job, then the path as a fenced block.
    assert lines[0].startswith("*NVRx attribution:*")
    assert lines[1].startswith("*Job ID:*")
    assert lines[2].startswith("*Log path:*")
    assert lines[3].startswith("```")
    assert len(lines) == 4


def test_summary_states_the_reason_when_there_is_one(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    parsed = _parsed()
    parsed.result["retry_policy"] = _policy_payload()["retry_policy"]
    parsed.result["l1_assessment"] = _policy_payload()["l1_assessment"]

    notifier.notify(_job(), parsed)
    lines = client.summaries[0]["text"].splitlines()

    assert len(lines) == 5
    assert lines[2].startswith("*Why STOP:*")
    assert "category 13" in lines[2]


def test_detail_sections_are_not_in_the_channel_line(monkeypatch, tmp_path):
    log = _log_with(tmp_path)
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    payload = _restart_agent_payload()
    payload["result"]["primary_failure"]["line"] = 30
    payload["result"]["retry_policy"] = _policy_payload()["retry_policy"]

    notifier.notify(_job(), parse_attrsvc_response(payload, log_path=str(log)))
    summary, reply = client.summaries[0]["text"], client.replies[0]["text"]

    for section in (
        "*Failed due to:*",
        "*Terminal issue:*",
        "*Evidence:*",
        "*Log at the failure:*",
        "*Plausible causes*",
        "*Missing evidence:*",
    ):
        assert section not in summary, section
        assert section in reply, section

    # The reason is summarised in one line in the channel and itemised in the
    # thread; the bullet list must not leak out of the thread.
    assert "*Why " in summary
    assert "*Why " in reply
    assert "\n  •" not in summary
    assert "•" in reply


def test_reply_threads_onto_the_summary(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)

    notifier.notify(_job(), _parsed())

    assert client.summaries[0]["thread_ts"] is None
    assert client.replies[0]["thread_ts"] == summary_ts(client)


def test_mention_rides_on_the_summary(monkeypatch):
    client = _LookupClient()
    notifier = _notifier(monkeypatch, client=client, email_domain="example.com")

    notifier.notify(_job(), _parsed())

    # The owner should be pinged by the visible line, not buried in a reply.
    assert "<@U123>" in client.summaries[0]["text"]
    assert "<@U123>" not in client.replies[0]["text"]


def test_a_failed_reply_does_not_fail_the_alert(monkeypatch):
    class _ReplyFails(_StubClient):
        def chat_postMessage(self, channel, text, thread_ts=None):
            if thread_ts is not None:
                raise RuntimeError("thread_not_found")
            return super().chat_postMessage(channel, text)

    monkeypatch.setattr(slack_mod, "SlackApiError", RuntimeError)
    client = _ReplyFails()
    notifier = _notifier(monkeypatch, client=client)

    # The alert is already delivered; losing the detail is not a failed send.
    assert notifier.notify(_job(), _parsed()) is True
    assert notifier.stats.sent == 1
    assert notifier.stats.failed == 0
    assert len(client.summaries) == 1


def test_no_reply_when_there_is_no_detail(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    bare = parse_attrsvc_response(
        {"result": {}, "recommendation": {"action": "STOP", "reason": "", "source": ""}},
        log_path="/x.log",
    )

    notifier.notify(_job(), bare)

    assert len(client.summaries) == 1
    assert client.replies == []


# ─── cycle number on the job line ───


def test_job_line_shows_the_cycle_when_there_is_one():
    ident = AnalysisIdentity(job_id="4103814_0", run_name="ultra_60t", cycle_id=3)
    assert ident.label == "4103814_0 (ultra_60t) cycle 3"


def test_job_line_omits_the_cycle_when_there_is_none():
    # Logs without _cycle<N> have no attempt ordering, so there is nothing to show.
    ident = AnalysisIdentity(job_id="4103814_0", run_name="ultra_60t")
    assert ident.label == "4103814_0 (ultra_60t)"


def test_cycle_zero_is_shown_not_treated_as_absent():
    assert AnalysisIdentity(job_id="1", cycle_id=0).label == "1 cycle 0"


def test_cycle_shows_without_a_run_name():
    assert AnalysisIdentity(job_id="1", cycle_id=2).label == "1 cycle 2"


def test_summary_carries_the_cycle(monkeypatch):
    client = _StubClient()
    notifier = _notifier(monkeypatch, client=client)
    ident = AnalysisIdentity(job_id="4103814_0", user="alice", run_name="ultra_60t", cycle_id=4)

    notifier.notify(ident, _parsed())

    assert "*Cycle:* 4" in client.summaries[0]["text"]


def test_backend_passes_the_cycle_to_the_alert():
    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        RestartAgentServiceBackend,
    )

    sent = []
    backend = SimpleNamespace(
        _slack_notifier=SimpleNamespace(enabled=True, notify=lambda i, r: sent.append(i) or True),
        _lock=threading.RLock(),
        _entries={
            "k": SimpleNamespace(
                job_id="4103814_0",
                user="u",
                log_path="/x/logs/run_4103814_date_x_cycle2.log",
                cycle_id=2,
            )
        },
        _public_result=lambda e: SimpleNamespace(
            result={},
            status="completed",
            recommendation={"action": "STOP", "reason": "", "source": ""},
        ),
    )

    RestartAgentServiceBackend._notify_slack(backend, "k")

    assert sent[0].cycle_id == 2
    assert "cycle 2" in sent[0].label


# ─── the one-line reason in the channel ───


def test_headline_names_the_guard_over_the_category():
    # Live case 4234179: L1 picked a category that said RESTART and a job guard
    # forced STOP anyway. The channel line has to show what actually decided.
    payload = _policy_payload(
        retry_budget_exhausted=True,
        **_ledgers(
            exhausted_by=["job_no_progress_guard"],
            job_no_progress_guard={
                "ledger_id": "job_no_progress_guard",
                "applicable": True,
                "allowed_retries": 3,
                "matching_prior_attempts": 3,
                "exhausted": True,
            },
        ),
    )
    headline = slack_mod.decision_rationale(payload).headline()

    assert headline == (
        "`job_no_progress_guard` exhausted — 3 of 3 attempts with no progress for this job"
    )


def test_headline_falls_back_to_the_category():
    assert "category 13" in slack_mod.decision_rationale(_policy_payload()).headline()


def test_headline_reports_the_rules_own_exhausted_budget():
    payload = _policy_payload(
        retry_budget_exhausted=True,
        effective_policy={"rule": "workload_unrecoverable", "allowed_retries": 0},
    )
    assert slack_mod.decision_rationale(payload).headline() == (
        "`workload_unrecoverable` budget exhausted (0)"
    )


def test_headline_falls_back_to_the_policy_without_a_category():
    payload = _policy_payload()
    payload.pop("l1_assessment")
    assert slack_mod.decision_rationale(payload).headline() == "policy `general_retry`"


def test_headline_is_empty_without_a_decision():
    assert slack_mod.decision_rationale({}).headline() == ""
