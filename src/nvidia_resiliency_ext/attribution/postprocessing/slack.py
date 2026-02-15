# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slack integration for attribution alerts.

API (send_slack_notification) and maybe_send_slack_notification used by post_results.
Set config.slack_bot_token and config.slack_channel at startup.

Usage:
    config.slack_bot_token = token; config.slack_channel = channel  # at startup
    # One-off: send_slack_notification(data, token, channel) when should_notify_slack(auto_resume)

Requires slack-sdk (optional). When not installed, HAS_SLACK is False and send no-ops.
"""

import logging
from dataclasses import dataclass

from .config import config

logger = logging.getLogger(__name__)

# Value of auto_resume that indicates terminal failure (should notify Slack)
AUTO_RESUME_TERMINAL = "STOP - DONT RESTART IMMEDIATE"


@dataclass
class SlackStats:
    """Statistics for Slack notification operations."""

    total_attempts: int = 0
    total_successful: int = 0
    total_failed: int = 0
    user_lookups: int = 0
    user_not_found: int = 0


# Global stats instance
_slack_stats = SlackStats()

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore


def get_slack_stats() -> SlackStats:
    """Get current Slack statistics."""
    return _slack_stats


def get_slack_user_id(user_id: str, token: str) -> str | None:
    """Look up Slack user ID from NVIDIA email.

    Args:
        user_id: NVIDIA username (will be converted to {user_id}@nvidia.com)
        token: Slack bot token

    Returns:
        Slack user ID if found, None otherwise
    """
    if not HAS_SLACK:
        logger.warning("slack-sdk not installed, cannot look up user")
        return None

    _slack_stats.user_lookups += 1
    client = WebClient(token=token)

    try:
        result = client.users_lookupByEmail(email=f"{user_id}@nvidia.com")
        return result.get("user", {}).get("id")
    except SlackApiError as e:
        _slack_stats.user_not_found += 1
        logger.error(f"Error fetching Slack user for {user_id}: {e.response['error']}")
        return None


def send_slack_notification(
    data: dict,
    slack_bot_token: str,
    slack_channel: str,
) -> bool:
    """Send attribution result to Slack channel.

    Args:
        data: Attribution result dict with keys:
            - s_job_id: Job ID
            - s_user: Username
            - s_attribution: Attribution text
            - s_auto_resume_explanation: Explanation of why job shouldn't restart
        slack_bot_token: Slack bot OAuth token
        slack_channel: Slack channel name or ID

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not HAS_SLACK:
        logger.warning("slack-sdk not installed, cannot send notification")
        return False

    if not slack_bot_token:
        logger.debug("Slack notification skipped: no bot token configured")
        return False

    if not slack_channel:
        logger.warning("Slack notification skipped: no channel configured")
        return False

    client = WebClient(token=slack_bot_token)

    # Try to mention the user
    slack_user_id = get_slack_user_id(data.get("s_user", ""), slack_bot_token)
    mention = f"\n<@{slack_user_id}>" if slack_user_id else ""
    if not slack_user_id and data.get("s_user"):
        logger.warning(f"User {data.get('s_user')} not found in Slack")

    text = (
        f"*Job ID:* `{data.get('s_job_id', 'unknown')}`\n"
        "*Failed due to:*\n"
        f"```{data.get('s_attribution', 'No attribution available')}```\n"
        "*Terminal issue:*\n"
        f"```{data.get('s_auto_resume_explanation', 'No explanation available')}```"
        f"{mention}"
    )

    _slack_stats.total_attempts += 1
    try:
        client.chat_postMessage(
            channel=slack_channel,
            text=text,
        )
        _slack_stats.total_successful += 1
        logger.info(f"Slack notification sent for job {data.get('s_job_id')}")
        return True
    except SlackApiError as e:
        _slack_stats.total_failed += 1
        logger.error(f"Error posting Slack message: {e.response['error']}")
        return False


def should_notify_slack(auto_resume: str) -> bool:
    """Check if this attribution result should trigger a Slack notification.

    Args:
        auto_resume: The auto_resume field from attribution result

    Returns:
        True if should notify (terminal failure), False otherwise
    """
    return auto_resume == AUTO_RESUME_TERMINAL


def maybe_send_slack_notification(data: dict) -> None:
    """If Slack is configured and this result is terminal, send notification.

    Called from post_results after the custom post_fn. No-op if token/channel
    unset or not a terminal result.
    """
    if (
        config.slack_bot_token
        and config.slack_channel
        and should_notify_slack(data.get("s_auto_resume", ""))
    ):
        send_slack_notification(data, config.slack_bot_token, config.slack_channel)
