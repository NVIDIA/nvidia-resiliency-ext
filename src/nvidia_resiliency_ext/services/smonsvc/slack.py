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

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from nvidia_resiliency_ext.attribution.api_keys import load_slack_bot_token
from nvidia_resiliency_ext.attribution.orchestration.client_response import AttrSvcResult
from nvidia_resiliency_ext.attribution.orchestration.posting_markdown import (
    format_posting_markdown_body,
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


def build_slack_record(job: "SlurmJob", result: AttrSvcResult) -> dict:
    """Build the posting record consumed by :func:`format_posting_markdown_body`.

    Uses the canonical ``s_``-prefixed dataflow keys so the message body matches
    the one produced by the attribution posting pipeline.
    """
    item = latest_result_item(result)
    job_name = getattr(job, "name", "")
    job_label = f"{job.job_id} ({job_name})" if job_name else str(job.job_id)

    record: dict[str, Any] = {
        "s_job_id": job_label,
        "s_user": getattr(job, "user", "") or "",
        "s_log_path": result.log_path or "",
        "s_recommendation_action": result.recommendation.action,
        "s_recommendation_source": result.recommendation.source,
    }
    if item is not None:
        record["s_primary_issues"] = item.primary_issues
        record["s_auto_resume_explanation"] = item.auto_resume_explanation
        record["s_attribution_result_json"] = json.dumps(item.to_payload())
    return record


def format_notification(record: dict, result: AttrSvcResult) -> str:
    """Prefix the shared attribution body with the monitor's decision header."""
    header = f"*NVRx attribution:* `{result.recommendation.action}`"
    if result.recommendation.source:
        header += f" _(source: {result.recommendation.source})_"
    if result.recommendation.reason:
        header += f"\n*Reason:* {result.recommendation.reason}"
    return f"{header}\n{format_posting_markdown_body(record)}"


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

        record = build_slack_record(job, result)
        text = format_notification(record, result)

        user = record.get("s_user", "")
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
