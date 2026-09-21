# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slack notifications for completed attribution analyses.

Notifying here rather than in a client covers both deployment shapes with one
implementation: as a service (``nvrx-smonsvc`` + ``nvrx-attrsvc``) and inline
(NVRx + ``nvrx-attrsvc``), which has no monitor to hook. Both submit through the
same ``POST /logs`` and both carry the job identity, so the alert is identical.

It also makes one analysis produce exactly one alert regardless of how many
clients fetch the result — in service mode every array task of a job fetches the
same completed analysis.

Credentials use unprefixed environment variables, matching the existing attrsvc
settings:

``SLACK_BOT_TOKEN`` / ``SLACK_BOT_TOKEN_FILE``
    Bot token, resolved by
    :func:`~nvidia_resiliency_ext.attribution.api_keys.load_slack_bot_token`.
``SLACK_CHANNEL``
    Target channel ID or name, e.g. ``#trng-alerts``.
``NVRX_ATTRSVC_SLACK_NOTIFY_ACTIONS``
    Comma- or space-separated recommendation actions that trigger a message.
    Defaults to ``STOP`` so only terminal failures page.

Requires ``slack-sdk`` (``pip install 'nvidia-resiliency-ext[attribution]'``).
Without it, or without a token and channel, the notifier reports itself
disabled and attribution behaves exactly as before.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from nvidia_resiliency_ext.attribution.api_keys import load_slack_bot_token
from nvidia_resiliency_ext.attribution.orchestration.client_response import AttrSvcResult
from nvidia_resiliency_ext.attribution.orchestration.posting_markdown import (
    format_attribution_markdown,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    RECOMMENDATION_ACTIONS,
    RECOMMENDATION_STOP,
    RawAnalysisResultItem,
    normalize_recommendation_action,
)
from nvidia_resiliency_ext.attribution.postprocessing.slack import (
    HAS_SLACK,
    SlackApiError,
    WebClient,
    get_slack_user_id,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import category_by_id

logger = logging.getLogger(__name__)

#: Only terminal failures page by default; restarts are routine and noisy.
DEFAULT_NOTIFY_ACTIONS = (RECOMMENDATION_STOP,)

NOTIFY_ACTIONS_ENV = "NVRX_ATTRSVC_SLACK_NOTIFY_ACTIONS"
CHANNEL_ENV = "SLACK_CHANNEL"

# Key of the attribution result list inside the inner backend result payload.
_RESULT_ITEMS_KEY = "result"


@dataclass
class SlackStats:
    """Cumulative notification counters, surfaced on the monitor ``/stats`` endpoint."""

    attempts: int = 0
    sent: int = 0
    failed: int = 0
    skipped_action: int = 0

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "sent": self.sent,
            "failed": self.failed,
            "skipped_action": self.skipped_action,
        }


@dataclass(frozen=True)
class SlackConfig:
    """Resolved Slack settings for the monitor."""

    token: str = ""
    channel: str = ""
    notify_actions: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_NOTIFY_ACTIONS)
    )

    @property
    def configured(self) -> bool:
        """Whether a token and a channel are both available."""
        return bool(self.token and self.channel)

    @classmethod
    def from_env(cls) -> "SlackConfig":
        """Build from environment variables, ignoring unusable values with a warning."""
        return cls(
            token=load_slack_bot_token(),
            channel=(os.environ.get(CHANNEL_ENV) or "").strip(),
            notify_actions=parse_notify_actions(os.environ.get(NOTIFY_ACTIONS_ENV)),
        )

    @classmethod
    def from_settings(cls, settings: Any) -> "SlackConfig":
        """Build from attrsvc ``Settings``, falling back to the key-file lookup.

        ``SLACK_BOT_TOKEN`` and ``SLACK_CHANNEL`` are existing attrsvc settings;
        an empty token defers to ``SLACK_BOT_TOKEN_FILE`` and the default paths.
        """
        token = str(getattr(settings, "SLACK_BOT_TOKEN", "") or "").strip()
        return cls(
            token=token or load_slack_bot_token(),
            channel=str(getattr(settings, "SLACK_CHANNEL", "") or "").strip(),
            notify_actions=parse_notify_actions(os.environ.get(NOTIFY_ACTIONS_ENV)),
        )


def parse_notify_actions(raw: Optional[str]) -> frozenset[str]:
    """Parse the configured action list, falling back to the default on empty input.

    Unrecognized entries are dropped with a warning rather than normalized to
    ``UNKNOWN``, which would otherwise silently page on every unattributed job.
    """
    if not raw or not raw.strip():
        return frozenset(DEFAULT_NOTIFY_ACTIONS)

    actions: set[str] = set()
    for token in raw.replace(",", " ").split():
        candidate = token.strip().upper().replace(" ", "_")
        if candidate in RECOMMENDATION_ACTIONS:
            actions.add(candidate)
        else:
            logger.warning(
                f"Ignoring unrecognized {NOTIFY_ACTIONS_ENV} entry {token!r}; "
                f"valid actions: {', '.join(RECOMMENDATION_ACTIONS)}"
            )

    if not actions:
        logger.warning(
            f"{NOTIFY_ACTIONS_ENV} contained no valid actions; "
            f"defaulting to {', '.join(DEFAULT_NOTIFY_ACTIONS)}"
        )
        return frozenset(DEFAULT_NOTIFY_ACTIONS)
    return frozenset(actions)


def latest_result_item(result: AttrSvcResult) -> Optional[RawAnalysisResultItem]:
    """Return the most recent parsable attribution item, or ``None``.

    Single-file responses carry one item per workload cycle; the last one is the
    cycle that produced the terminal recommendation.
    """
    payload = result.result
    items = payload.get(_RESULT_ITEMS_KEY) if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    for value in reversed(items):
        try:
            return RawAnalysisResultItem.from_payload(value)
        except (TypeError, ValueError):
            continue
    return None


#: Restart Agent results are identified by their response schema version.
_RESTART_AGENT_SCHEMA_PREFIX = "restart_agent_response."


#: Only this status means the log itself established the cause; the others leave
#: the agent's alternatives worth showing.
CONFIRMED_STATUS = "established_by_current_log"


@dataclass(frozen=True)
class AttributionFields:
    """Display fields shared by both backends' result shapes."""

    headline: str = ""
    explanation: str = ""
    evidence: str = ""
    status: str = ""
    plausible_causes: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()

    @property
    def show_alternatives(self) -> bool:
        """Whether the agent's hypotheses add information beyond the headline."""
        return bool(self.status) and self.status != CONFIRMED_STATUS


def _failure_label(failure: Any) -> str:
    """Render one Restart Agent failure record as ``class: signature``."""
    if not isinstance(failure, Mapping):
        return ""
    failure_class = str(failure.get("failure_class") or "").strip()
    signature = str(failure.get("signature") or "").strip().rstrip(":").strip()
    if failure_class and signature:
        return f"{failure_class}: {signature}"
    return failure_class or signature


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(v).strip() for v in value if str(v).strip())


def _evidence_line(failure: Any) -> str:
    """Locate the failure: class, line, rank, phase, and its role in the cascade.

    This is what makes an alert actionable - it says where to look in the log and
    whether that line initiated the failure or merely followed one.
    """
    if not isinstance(failure, Mapping):
        return ""
    failure_class = str(failure.get("failure_class") or "").strip()
    if not failure_class:
        return ""

    where = []
    if failure.get("line") is not None:
        where.append(f"line {failure['line']}")
    for key, label in (("rank", "rank"), ("node", "node"), ("gpu", "gpu")):
        if failure.get(key) is not None:
            where.append(f"{label} {failure[key]}")
    if failure.get("phase"):
        where.append(f"phase {failure['phase']}")

    qualifiers = [
        str(failure[key]).strip()
        for key in ("fault_outcome", "causal_role")
        if str(failure.get(key) or "").strip() and str(failure.get(key)).strip() != "unknown"
    ]

    text = f"`{failure_class}`"
    if where:
        text += " at " + ", ".join(where)
    if qualifiers:
        text += f" ({', '.join(qualifiers)})"
    return text


#: Values that carry no information and are better omitted than rendered.
_EMPTY_CLAIMS = ("", "unknown", "none")


@dataclass(frozen=True)
class DecisionRationale:
    """Why L4 landed on this action, in the order a reader should weigh it."""

    rule: str = ""
    base_rule: str = ""
    allowed_retries: Optional[int] = None
    budget_exhausted: bool = False
    category_id: Optional[int] = None
    category_name: str = ""
    category_decision: str = ""
    category_confidence: Optional[int] = None
    retry_outlook: str = ""
    retry_outlook_status: str = ""
    retry_outlook_confidence: Optional[int] = None
    failure_domain: str = ""
    failure_domain_confidence: Optional[int] = None

    @property
    def empty(self) -> bool:
        return not (self.rule or self.category_id or self.retry_outlook)

    def as_lines(self) -> list[str]:
        """Bullet lines, most decision-relevant first, omitting empty claims."""
        lines = []
        if self.rule:
            rule = f"rule `{self.rule}`"
            # A policy context outranks the base rule; name what it displaced.
            if self.base_rule and self.base_rule != self.rule:
                rule += f" (overrides `{self.base_rule}`)"
            if self.allowed_retries is not None:
                rule += f" — budget {self.allowed_retries}"
                rule += ", exhausted" if self.budget_exhausted else ", not exhausted"
            lines.append(rule)
        if self.category_id:
            cat = f"category {self.category_id}"
            if self.category_name:
                cat += f" _{self.category_name}_"
            if self.category_decision:
                cat += f" → {self.category_decision}"
            if self.category_confidence is not None:
                cat += f" (confidence {self.category_confidence})"
            lines.append(cat)
        if self.retry_outlook:
            outlook = f"retry outlook: {self.retry_outlook}"
            qual = [q for q in (self.retry_outlook_status,) if q and q not in _EMPTY_CLAIMS]
            if self.retry_outlook_confidence is not None:
                qual.append(f"confidence {self.retry_outlook_confidence}")
            if qual:
                outlook += f" ({', '.join(qual)})"
            lines.append(outlook)
        if self.failure_domain:
            domain = f"failure domain: {self.failure_domain}"
            if self.failure_domain_confidence is not None:
                domain += f" (confidence {self.failure_domain_confidence})"
            lines.append(domain)
        return lines


def _claim(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _EMPTY_CLAIMS else text


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def decision_rationale(payload: Any) -> DecisionRationale:
    """Extract the decision audit trail from a ``restart_agent_response.v1`` payload.

    ``failure_domain`` and ``retry_outlook`` are the model's abstract policy
    claims and are materially less accurate than the category pick, so they are
    reported with their confidence and dropped when unknown.
    """
    if not isinstance(payload, Mapping):
        return DecisionRationale()
    policy = payload.get("retry_policy")
    policy = policy if isinstance(policy, Mapping) else {}

    effective = policy.get("effective_policy")
    effective = effective if isinstance(effective, Mapping) else {}
    context = policy.get("applied_policy_context")
    context_id = ""
    if isinstance(context, Mapping):
        context_id = _claim(context.get("policy_context_id"))
    elif isinstance(context, str):
        context_id = _claim(context)

    base_rule = _claim(policy.get("base_rule"))
    rule = context_id or _claim(effective.get("rule")) or base_rule

    category_id, category_name, category_decision = None, "", ""
    confidence = None
    assessment = payload.get("l1_assessment")
    if isinstance(assessment, Mapping):
        selection = assessment.get("category_selection")
        if isinstance(selection, Mapping):
            category_id = _int_or_none(selection.get("category_id")) or None
            confidence = _int_or_none(selection.get("category_confidence"))
            if category_id:
                definition = category_by_id(category_id)
                if definition is not None:
                    category_name = definition.name
                    category_decision = definition.decision

    return DecisionRationale(
        rule=rule,
        base_rule=base_rule,
        allowed_retries=_int_or_none(effective.get("allowed_retries")),
        budget_exhausted=bool(policy.get("retry_budget_exhausted")),
        category_id=category_id,
        category_name=category_name,
        category_decision=category_decision,
        category_confidence=confidence,
        retry_outlook=_claim(policy.get("retry_outlook_without_workload_change")),
        retry_outlook_status=_claim(policy.get("retry_outlook_status")),
        retry_outlook_confidence=_int_or_none(policy.get("retry_outlook_confidence")),
        failure_domain=_claim(policy.get("failure_domain")),
        failure_domain_confidence=_int_or_none(policy.get("failure_domain_confidence")),
    )


def restart_agent_fields(payload: Any) -> Optional[AttributionFields]:
    """Extract display fields from a ``restart_agent_response.v1`` payload.

    The headline is the model's narrative root cause rather than the typed
    failure record: ``failure_class`` plus a raw log snippet says what matched,
    not what went wrong. The typed record is kept as the evidence line, which is
    where its line/rank/phase actually help. Returns ``None`` for any other
    payload shape.
    """
    if not isinstance(payload, Mapping):
        return None
    schema = str(payload.get("schema_version") or "")
    if not schema.startswith(_RESTART_AGENT_SCHEMA_PREFIX):
        return None

    primary = payload.get("primary_failure")
    assessment = payload.get("l1_assessment")
    root_cause = {}
    if isinstance(assessment, Mapping):
        candidate = assessment.get("root_cause_assessment")
        if isinstance(candidate, Mapping):
            root_cause = candidate

    # Fall back to the typed record when the model produced no narrative, which
    # happens on deterministic-only results.
    headline = str(root_cause.get("summary") or "").strip() or _failure_label(primary)

    explanation = str(payload.get("justification") or "").strip()
    basis = str(payload.get("decision_basis") or "").strip()
    # The justification usually already names the basis; only append when it does not.
    if basis and basis not in explanation:
        explanation = f"{explanation} (decision basis: {basis})" if explanation else basis

    return AttributionFields(
        headline=headline,
        explanation=explanation,
        evidence=_evidence_line(primary),
        status=str(root_cause.get("status") or "").strip(),
        plausible_causes=_string_list(root_cause.get("plausible_causes")),
        missing_evidence=_string_list(root_cause.get("missing_evidence")),
    )


def _format_issues(primary: Sequence[str], secondary: Sequence[str]) -> str:
    """LogSage's issue wording, kept so legacy alerts read as they always did."""
    return f"Primary issues: [{', '.join(primary)}], Secondary issues: [{', '.join(secondary)}]"


def attribution_fields(result: AttrSvcResult) -> Optional[AttributionFields]:
    """Display fields for either backend, preferring a LogSage item when present."""
    item = latest_result_item(result)
    if item is not None:
        # LogSage wrote its issue lists as prose; keep that wording verbatim.
        issues = _format_issues(item.primary_issues, item.secondary_issues)
        return AttributionFields(
            headline=issues,
            explanation=item.auto_resume_explanation,
        )
    return restart_agent_fields(result.result)


@dataclass(frozen=True)
class AnalysisIdentity:
    """Who the analysis belongs to, as supplied on ``POST /logs``."""

    job_id: str = ""
    user: str = ""
    run_name: str = ""

    @property
    def label(self) -> str:
        job_id = self.job_id or "unknown"
        return f"{job_id} ({self.run_name})" if self.run_name else job_id


def run_name_from_log_path(log_path: str, job_id: str = "") -> str:
    """Recover the run name from an application log filename.

    Logs are named ``<run>_<jobid>_date_...``, so the run name is the prefix
    before the job ID. attrsvc never sees the SLURM job name, and this keeps the
    alert self-describing in both deployment modes. Returns ``""`` when the
    filename does not follow that convention.
    """
    if not log_path:
        return ""
    stem = os.path.basename(log_path)
    base = str(job_id).split("_", 1)[0].split("+", 1)[0].strip()
    if not base:
        return ""
    marker = f"_{base}_"
    index = stem.find(marker)
    return stem[:index] if index > 0 else ""


def format_notification(identity: AnalysisIdentity, result: AttrSvcResult) -> str:
    """Render the alert: decision, narrative cause, policy rationale, evidence."""
    job_label = identity.label

    header = f"*NVRx attribution:* `{result.recommendation.action}`"
    if result.recommendation.source:
        header += f" _(source: {result.recommendation.source})_"

    fields = attribution_fields(result) or AttributionFields()

    # A Restart Agent justification is both the recommendation reason and the
    # terminal-issue text; print it once rather than twice. Concatenating the two
    # is wrong when they differ, so surface the reason as its own line instead.
    reason = result.recommendation.reason.strip()
    if reason and reason not in fields.explanation:
        header += f"\n*Reason:* {reason}"
    explanation = fields.explanation or reason

    body = format_attribution_markdown(
        job_id=job_label,
        attribution_text=fields.headline,
        auto_resume_explanation=explanation,
        log_path=result.log_path or "",
    )

    extras = []
    rationale = decision_rationale(result.result)
    if not rationale.empty:
        bullets = "\n".join(f"  • {line}" for line in rationale.as_lines())
        extras.append(f"*Why {result.recommendation.action}:*\n{bullets}")
    if fields.evidence:
        extras.append(f"*Evidence:* {fields.evidence}")
    if fields.show_alternatives:
        if fields.plausible_causes:
            causes = "\n".join(f"  • {c}" for c in fields.plausible_causes)
            extras.append(f"*Plausible causes* _({fields.status})_:\n{causes}")
        if fields.missing_evidence:
            missing = "\n".join(f"  • {m}" for m in fields.missing_evidence)
            extras.append(f"*Missing evidence:*\n{missing}")

    text = f"{header}\n{body}"
    if extras:
        text += "\n" + "\n".join(extras)
    return text


class SlackNotifier:
    """Posts attribution results to a Slack channel.

    A notifier is always constructed; it simply reports ``enabled == False``
    when ``slack-sdk`` is missing or no token/channel is configured, so callers
    never need to branch on whether Slack is set up.
    """

    def __init__(self, config: Optional[SlackConfig] = None):
        self.config = SlackConfig.from_env() if config is None else config
        self.stats = SlackStats()
        self._client: Any = None

    @property
    def enabled(self) -> bool:
        """Whether notifications can actually be delivered."""
        return HAS_SLACK and self.config.configured

    def describe(self) -> str:
        """One-line status suitable for a startup log."""
        if not HAS_SLACK:
            return "disabled (slack-sdk not installed)"
        if not self.config.token:
            return "disabled (no bot token; set SLACK_BOT_TOKEN or SLACK_BOT_TOKEN_FILE)"
        if not self.config.channel:
            return f"disabled (no channel; set {CHANNEL_ENV})"
        actions = ", ".join(sorted(self.config.notify_actions))
        return f"enabled (channel: {self.config.channel}, actions: {actions})"

    def should_notify(self, action: str) -> bool:
        """Whether a recommendation action is in the configured notify set."""
        return normalize_recommendation_action(action) in self.config.notify_actions

    def notify(self, identity: AnalysisIdentity, result: AttrSvcResult) -> bool:
        """Send a notification for ``result`` if it is configured to page.

        Returns ``True`` only when a message was delivered.
        """
        if not self.enabled:
            return False
        if not self.should_notify(result.recommendation.action):
            self.stats.skipped_action += 1
            return False

        text = format_notification(identity, result)

        user = identity.user
        if user:
            slack_user_id = get_slack_user_id(user, self.config.token)
            if slack_user_id:
                text += f"\n<@{slack_user_id}>"
            else:
                logger.warning(f"[{identity.job_id}] Slack user not found for {user}")

        self.stats.attempts += 1
        try:
            self._web_client().chat_postMessage(channel=self.config.channel, text=text)
        except SlackApiError as e:
            self.stats.failed += 1
            logger.error(f"[{identity.job_id}] Slack notification failed: {e}")
            return False
        self.stats.sent += 1
        logger.info(f"[{identity.job_id}] Slack notification sent to {self.config.channel}")
        return True

    def _web_client(self) -> Any:
        if self._client is None:
            self._client = WebClient(token=self.config.token)
        return self._client
