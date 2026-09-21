# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slack notifications for attribution results observed by the monitor.

The monitor already parses a normalized :class:`AttrSvcResult` for every
terminal job it fetches. This module turns the results worth paging on into a
Slack message, reusing the shared attribution markdown body so a notification
looks the same regardless of which analysis path produced it.

Credentials use unprefixed environment variables, matching the equivalent
attrsvc settings:

``SLACK_BOT_TOKEN`` / ``SLACK_BOT_TOKEN_FILE``
    Bot token, resolved by
    :func:`~nvidia_resiliency_ext.attribution.api_keys.load_slack_bot_token`.
``SLACK_CHANNEL``
    Target channel ID or name, e.g. ``#trng-alerts``.
``NVRX_SMONSVC_SLACK_NOTIFY_ACTIONS``
    Comma- or space-separated recommendation actions that trigger a message.
    Defaults to ``STOP`` so only terminal failures page.

Requires ``slack-sdk`` (``pip install 'nvidia-resiliency-ext[attribution]'``).
Without it, or without a token and channel, the notifier reports itself
disabled and the monitor behaves exactly as before.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

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

if TYPE_CHECKING:
    from .models import SlurmJob

logger = logging.getLogger(__name__)

#: Only terminal failures page by default; restarts are routine and noisy.
DEFAULT_NOTIFY_ACTIONS = (RECOMMENDATION_STOP,)

NOTIFY_ACTIONS_ENV = "NVRX_SMONSVC_SLACK_NOTIFY_ACTIONS"
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


def format_notification(job: "SlurmJob", result: AttrSvcResult) -> str:
    """Render the alert: decision, narrative cause, policy rationale, evidence."""
    job_name = getattr(job, "name", "")
    job_label = f"{job.job_id} ({job_name})" if job_name else str(job.job_id)

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
    if fields.evidence:
        extras.append(f"*Evidence:* {fields.evidence}")
    if fields.show_alternatives:
        if fields.plausible_causes:
            causes = "\n".join(f"  • {c}" for c in fields.plausible_causes)
            extras.append(f"*Plausible causes* _({fields.status})_:\n{causes}")
        if fields.missing_evidence:
            missing = "\n".join(f"  • {m}" for m in fields.missing_evidence)
            extras.append(f"*Missing evidence:*\n{missing}")

    user = getattr(job, "user", "") or ""
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

    def notify(self, job: "SlurmJob", result: AttrSvcResult) -> bool:
        """Send a notification for ``result`` if it is configured to page.

        Returns ``True`` only when a message was delivered.
        """
        if not self.enabled:
            return False
        if not self.should_notify(result.recommendation.action):
            self.stats.skipped_action += 1
            return False

        text = format_notification(job, result)

        user = getattr(job, "user", "") or ""
        if user:
            slack_user_id = get_slack_user_id(user, self.config.token)
            if slack_user_id:
                text += f"\n<@{slack_user_id}>"
            else:
                logger.warning(f"[{job.job_id}] Slack user not found for {user}")

        self.stats.attempts += 1
        try:
            self._web_client().chat_postMessage(channel=self.config.channel, text=text)
        except SlackApiError as e:
            self.stats.failed += 1
            logger.error(f"[{job.job_id}] Slack notification failed: {e}")
            return False
        self.stats.sent += 1
        logger.info(f"[{job.job_id}] Slack notification sent to {self.config.channel}")
        return True

    def _web_client(self) -> Any:
        if self._client is None:
            self._client = WebClient(token=self.config.token)
        return self._client
