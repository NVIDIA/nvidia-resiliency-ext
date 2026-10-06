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
``NVRX_ATTRSVC_SLACK_EMAIL_DOMAIN``
    Domain used to turn a job owner into an email address for an ``@`` mention,
    e.g. ``example.com``. Unset means no mention is attempted; the owner is
    still named in the message.

``slack-sdk`` is a regular dependency. Without a token and channel the notifier
reports itself disabled and attribution behaves exactly as before; the import is
still guarded so a stripped install degrades rather than fails.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from nvidia_resiliency_ext.attribution.api_keys import load_slack_bot_token
from nvidia_resiliency_ext.attribution.orchestration.client_response import AttrSvcResult
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
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import category_by_id

logger = logging.getLogger(__name__)

#: Only terminal failures page by default; restarts are routine and noisy.
DEFAULT_NOTIFY_ACTIONS = (RECOMMENDATION_STOP,)

NOTIFY_ACTIONS_ENV = "NVRX_ATTRSVC_SLACK_NOTIFY_ACTIONS"
EMAIL_DOMAIN_ENV = "NVRX_ATTRSVC_SLACK_EMAIL_DOMAIN"
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
    #: Empty means do not guess an address for the job owner.
    email_domain: str = ""
    #: Which cluster produced the analysis; shown so alerts from several are distinguishable.
    cluster: str = ""

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
            email_domain=(os.environ.get(EMAIL_DOMAIN_ENV) or "").strip().lstrip("@"),
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
            cluster=str(getattr(settings, "CLUSTER_NAME", "") or "").strip(),
            email_domain=(os.environ.get(EMAIL_DOMAIN_ENV) or "").strip().lstrip("@"),
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


#: What each L4 retry ledger actually counts. Without this an exhausted budget
#: reads as same-root exhaustion even when a job-history guard was what fired.
_LEDGER_COUNTS = {
    "general_root_ceiling": "attempts sharing this root cause",
    "selected_policy_ledger": "attempts matching this rule's scope",
    "job_no_progress_guard": "attempts with no progress for this job",
    "job_unknown_progress_guard": "attempts with unverifiable progress for this job",
}
#: Guards that fire on job history alone, with no root-cause match required.
_ROOT_INDEPENDENT_LEDGERS = frozenset({"job_no_progress_guard", "job_unknown_progress_guard"})

_GENERAL_ROOT_CEILING = "general_root_ceiling"
_SELECTED_POLICY_LEDGER = "selected_policy_ledger"


@dataclass(frozen=True)
class ExhaustedLedger:
    """A retry budget that ran out, and what it was counting when it did."""

    ledger_id: str
    attempts: Optional[int] = None
    allowed_retries: Optional[int] = None

    def as_line(self) -> str:
        line = f"budget stop: `{self.ledger_id}`"
        counted = _LEDGER_COUNTS.get(self.ledger_id)
        if self.attempts is not None and counted:
            line += f" — {self.attempts} {counted}"
            if self.allowed_retries is not None:
                line += f", budget {self.allowed_retries}"
        elif self.allowed_retries is not None:
            line += f" — budget {self.allowed_retries}"
        if self.ledger_id in _ROOT_INDEPENDENT_LEDGERS:
            line += " (counted regardless of root cause)"
        return line


@dataclass(frozen=True)
class DecisionRationale:
    """Why L4 landed on this action, in the order a reader should weigh it."""

    rule: str = ""
    base_rule: str = ""
    allowed_retries: Optional[int] = None
    #: Whether the rule's *own* ledger ran out, not whether anything did.
    budget_exhausted: bool = False
    #: Budgets that ran out other than the rule's own, named so a job-history
    #: guard is not misread as same-root exhaustion.
    other_exhausted: tuple[ExhaustedLedger, ...] = ()
    #: 0 is L1's sanctioned "no listed category matches"; None means it did not run.
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
        return not (self.rule or self.category_id is not None or self.retry_outlook)

    def headline(self) -> str:
        """One line naming what actually drove the decision.

        Ordered by what overrode what: a budget that ran out beats the category,
        because L4 stops on an exhausted ledger whatever L1 concluded - the live
        case being a guard forcing STOP over a category that said RESTART.
        """
        if self.other_exhausted:
            ledger = self.other_exhausted[0]
            counted = _LEDGER_COUNTS.get(ledger.ledger_id, "matching attempts")
            if ledger.attempts is not None and ledger.allowed_retries is not None:
                return (
                    f"`{ledger.ledger_id}` exhausted — "
                    f"{ledger.attempts} of {ledger.allowed_retries} {counted}"
                )
            return f"`{ledger.ledger_id}` exhausted"
        if self.budget_exhausted and self.rule:
            budget = "" if self.allowed_retries is None else f" ({self.allowed_retries})"
            return f"`{self.rule}` budget exhausted{budget}"
        if self.category_id:
            name = f" — {self.category_name}" if self.category_name else ""
            return f"category {self.category_id}{name}"
        if self.category_id == 0:
            return "no listed category matched"
        if self.rule:
            return f"policy `{self.rule}`"
        return ""

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
        lines.extend(ledger.as_line() for ledger in self.other_exhausted)
        if self.category_id == 0:
            # Distinct from an absent selection: the model considered the
            # taxonomy and reported that none of it applies.
            lines.append("category: none of the listed categories matched")
        elif self.category_id:
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


def _exhaustion(policy: Mapping[str, Any]) -> tuple[bool, tuple[ExhaustedLedger, ...]]:
    """Split L4's exhausted budgets into the rule's own and everything else.

    ``retry_budget_exhausted`` is true when *any* ledger runs out, so reporting
    it beside the rule's budget claims same-root exhaustion that may not have
    happened. ``exhausted_by`` names the ledgers that actually fired; fall back
    to the aggregate flag only for payloads that predate it.
    """
    exhausted_by = policy.get("exhausted_by")
    if not isinstance(exhausted_by, (list, tuple)):
        return bool(policy.get("retry_budget_exhausted")), ()

    selected = policy.get(_SELECTED_POLICY_LEDGER)
    rule_ledger_id = (
        _SELECTED_POLICY_LEDGER
        if isinstance(selected, Mapping) and selected.get("applicable")
        else _GENERAL_ROOT_CEILING
    )

    rule_exhausted = False
    others = []
    for ledger_id in (str(entry) for entry in exhausted_by):
        if ledger_id == rule_ledger_id:
            rule_exhausted = True
            continue
        detail = policy.get(ledger_id)
        detail = detail if isinstance(detail, Mapping) else {}
        others.append(
            ExhaustedLedger(
                ledger_id=ledger_id,
                attempts=_int_or_none(detail.get("matching_prior_attempts")),
                allowed_retries=_int_or_none(detail.get("allowed_retries")),
            )
        )
    return rule_exhausted, tuple(others)


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
    rule_exhausted, other_exhausted = _exhaustion(policy)

    category_id, category_name, category_decision = None, "", ""
    confidence = None
    assessment = payload.get("l1_assessment")
    if isinstance(assessment, Mapping):
        selection = assessment.get("category_selection")
        if isinstance(selection, Mapping):
            category_id = _int_or_none(selection.get("category_id"))
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
        budget_exhausted=rule_exhausted,
        other_exhausted=other_exhausted,
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
    #: Restart attempt within the job. None when the logs carry no cycle numbers.
    cycle_id: Optional[int] = None

    @property
    def label(self) -> str:
        job_id = self.job_id or "unknown"
        label = f"{job_id} ({self.run_name})" if self.run_name else job_id
        return f"{label} cycle {self.cycle_id}" if self.cycle_id is not None else label

    def as_line(self) -> str:
        """The job line: id, name and cycle as separate labelled fields.

        Cycle is omitted rather than shown empty: logs without cycle numbers
        come from runs that never restarted in place, where "cycle" is noise.
        """
        parts = [f"*Job ID:* `{self.job_id or 'unknown'}`"]
        if self.run_name:
            parts.append(f"*Job Name:* `{self.run_name}`")
        if self.cycle_id is not None:
            parts.append(f"*Cycle:* {self.cycle_id}")
        return " | ".join(parts)


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


#: How much of the log to quote around the reported failure line.
EXCERPT_LINES = 3
#: Long lines are usually a serialized payload; keep the alert readable.
EXCERPT_MAX_CHARS = 220


def log_excerpt(log_path: str, line_number: Any, count: int = EXCERPT_LINES) -> list:
    """Return ``count`` lines starting at ``line_number`` (1-based), or ``[]``.

    The decision cites a line number but not the text at it, which is the first
    thing a reader wants. Streams rather than reads the file: these logs run to
    tens of megabytes.
    """
    start = _int_or_none(line_number)
    if not log_path or start is None or start < 1:
        return []
    picked = []
    try:
        with open(log_path, "r", errors="replace") as handle:
            for number, text in enumerate(handle, start=1):
                if number < start:
                    continue
                if number >= start + count:
                    break
                text = text.rstrip("\n")
                if len(text) > EXCERPT_MAX_CHARS:
                    text = text[:EXCERPT_MAX_CHARS] + " …"
                picked.append(f"{number}: {text}")
    except OSError:
        return []
    return picked


def primary_failure_line(result: AttrSvcResult) -> Any:
    """Line number of the primary failure, when the payload reports one."""
    payload = result.result
    if not isinstance(payload, Mapping):
        return None
    primary = payload.get("primary_failure")
    return primary.get("line") if isinstance(primary, Mapping) else None


def format_summary(
    identity: AnalysisIdentity,
    result: AttrSvcResult,
    cluster: str = "",
) -> str:
    """The channel-level line: decision, job, log path.

    Deliberately short. A channel of full attributions is unreadable, so the
    detail goes to a thread reply and the channel keeps one scannable line per
    job.
    """
    header = f"*NVRx attribution:* `{result.recommendation.action}`"
    if cluster:
        header += f" on *{cluster}*"
    if result.recommendation.source:
        header += f" _(source: {result.recommendation.source})_"

    lines = [header, identity.as_line()]
    headline = decision_rationale(result.result).headline()
    if headline:
        lines.append(f"*Why {result.recommendation.action}:* {headline}")
    lines.append(f"*Log path:*\n```{result.log_path or ''}```")
    return "\n".join(lines)


def format_details(identity: AnalysisIdentity, result: AttrSvcResult) -> str:
    """The thread reply: why this decision, and the evidence behind it."""
    fields = attribution_fields(result) or AttributionFields()

    # A Restart Agent justification is both the recommendation reason and the
    # terminal-issue text; print it once rather than twice. Concatenating the two
    # is wrong when they differ, so surface the reason as its own line instead.
    reason = result.recommendation.reason.strip()
    explanation = fields.explanation or reason

    sections = []
    if reason and reason not in fields.explanation:
        sections.append(f"*Reason:* {reason}")
    if fields.headline:
        sections.append(f"*Failed due to:*\n```{fields.headline}```")
    if explanation:
        sections.append(f"*Terminal issue:*\n```{explanation}```")

    rationale = decision_rationale(result.result)
    if not rationale.empty:
        bullets = "\n".join(f"  • {line}" for line in rationale.as_lines())
        sections.append(f"*Why {result.recommendation.action}:*\n{bullets}")
    if fields.evidence:
        sections.append(f"*Evidence:* {fields.evidence}")
    excerpt = log_excerpt(result.log_path or "", primary_failure_line(result))
    if excerpt:
        quoted = "\n".join(excerpt)
        sections.append(f"*Log at the failure:*\n```{quoted}```")
    if fields.show_alternatives:
        if fields.plausible_causes:
            causes = "\n".join(f"  • {c}" for c in fields.plausible_causes)
            sections.append(f"*Plausible causes* _({fields.status})_:\n{causes}")
        if fields.missing_evidence:
            missing = "\n".join(f"  • {m}" for m in fields.missing_evidence)
            sections.append(f"*Missing evidence:*\n{missing}")
    return "\n".join(sections)


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

    def _mention(self, user: str) -> str:
        """Resolve ``user`` to an ``@`` mention, or "" when that is not possible.

        A job owner is a local account name, not an address. Turning one into an
        email needs a site-specific domain, so no mention is attempted unless one
        is configured — this library runs outside NVIDIA too.
        """
        if not user or not self.config.email_domain:
            return ""
        email = f"{user}@{self.config.email_domain}"
        try:
            result = self._web_client().users_lookupByEmail(email=email)
        except SlackApiError as e:
            logger.debug("Slack user lookup failed for %s: %s", email, e)
            return ""
        user_id = (result.get("user") or {}).get("id") if hasattr(result, "get") else None
        return f"\n<@{user_id}>" if user_id else ""

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

        summary = format_summary(identity, result, self.config.cluster)
        summary += self._mention(identity.user)

        self.stats.attempts += 1
        try:
            posted = self._web_client().chat_postMessage(channel=self.config.channel, text=summary)
        except SlackApiError as e:
            self.stats.failed += 1
            logger.error(f"[{identity.job_id}] Slack notification failed: {e}")
            return False
        self.stats.sent += 1
        logger.info(f"[{identity.job_id}] Slack notification sent to {self.config.channel}")

        details = format_details(identity, result)
        thread_ts = posted.get("ts") if hasattr(posted, "get") else None
        if details and thread_ts:
            # The alert is already delivered; a failed thread reply costs detail,
            # not the notification, so it must not be reported as a failure.
            try:
                self._web_client().chat_postMessage(
                    channel=self.config.channel, thread_ts=thread_ts, text=details
                )
            except SlackApiError as e:
                logger.warning(f"[{identity.job_id}] Slack detail reply failed: {e}")
        return True

    def _web_client(self) -> Any:
        if self._client is None:
            self._client = WebClient(token=self.config.token)
        return self._client
