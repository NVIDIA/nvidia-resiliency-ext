# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for attribution postprocessing (poster, index, Slack)."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from nvidia_resiliency_ext.attribution.api_keys import load_slack_bot_token

logger = logging.getLogger(__name__)


def load_slack_from_env() -> Tuple[str, str]:
    """Read Slack token and channel.

    Token: :func:`~nvidia_resiliency_ext.attribution.api_keys.load_slack_bot_token`
    (``SLACK_BOT_TOKEN``, ``SLACK_BOT_TOKEN_FILE``, or default file paths).
    Channel: ``SLACK_CHANNEL`` environment variable.
    Returns ``(token, channel)``; empty strings if unset.
    """
    token = load_slack_bot_token()
    channel = (os.getenv("SLACK_CHANNEL") or "").strip()
    return (token, channel)


@dataclass
class PostprocessingConfig:
    """Mutable singleton; set fields at process startup (or use :func:`configure`)."""

    default_poster: Any = None  # ResultPoster; Any avoids circular import
    cluster_name: str = ""
    dataflow_index: str = ""
    slack_bot_token: str = ""
    slack_channel: str = ""


config = PostprocessingConfig()


def configure(
    *,
    default_poster: Any = None,
    cluster_name: str = "",
    dataflow_index: str = "",
    slack_bot_token: str = "",
    slack_channel: str = "",
) -> None:
    """Assign postprocessing settings. ``default_poster=None`` leaves the current poster unchanged."""
    if default_poster is not None:
        config.default_poster = default_poster
    config.cluster_name = cluster_name
    config.dataflow_index = dataflow_index
    config.slack_bot_token = slack_bot_token
    config.slack_channel = slack_channel

    if dataflow_index and not cluster_name:
        logger.warning(
            "postprocessing: dataflow_index is set but cluster_name is empty; "
            "records may have an empty cluster identifier"
        )
    if slack_channel and not slack_bot_token:
        logger.warning(
            "postprocessing: slack_channel is set but slack_bot_token is empty; "
            "Slack notifications will not be sent"
        )


def configure_from_env(
    *,
    default_poster: Any = None,
    cluster_name: str = "",
    dataflow_index: str = "",
    slack_token: Optional[str] = None,
    slack_channel: Optional[str] = None,
    cluster_name_env: Optional[str] = "SLURM_CLUSTER_NAME",
    autoconfigure_poster: bool = False,
) -> None:
    """Like :func:`configure`, but fills Slack from environment **per field** where the argument is
    ``None``. Pass ``""`` explicitly to force an empty token or channel without loading that field
    from env.

    Args:
        default_poster: Explicit :class:`~.pipeline.ResultPoster`, or ``None``.
        cluster_name: Override; if empty and ``cluster_name_env`` is set, reads that env var.
        dataflow_index: Target index/project name passed to the poster.
        slack_token: ``None`` = load token from env / token files (see
            :func:`~nvidia_resiliency_ext.attribution.api_keys.load_slack_bot_token`). Any other value,
            including ``""``, is used as-is (after ``strip``) and does not pull from env.
        slack_channel: ``None`` = load from ``SLACK_CHANNEL``. Any other value, including ``""``,
            is used as-is (after ``strip``). Token and channel are resolved **independently** so an
            explicit empty token does not cause the channel to be replaced by env (and vice versa).
        cluster_name_env: Env var used when ``cluster_name`` is empty.
        autoconfigure_poster: If ``True``, ``dataflow_index`` is set, and ``default_poster`` is
            ``None``, builds a poster from :func:`~.post_backend.get_retrying_post_fn` when a
            backend exists.
    """
    # Resolve token and channel independently: None = "not provided, use env", else explicit
    # (including "") so we never overwrite one field when filling the other from env.
    if slack_token is None or slack_channel is None:
        env_tok, env_ch = load_slack_from_env()
    else:
        env_tok, env_ch = "", ""

    slack_token = env_tok if slack_token is None else slack_token.strip()
    slack_channel = env_ch if slack_channel is None else slack_channel.strip()

    if not cluster_name and cluster_name_env:
        cluster_name = os.environ.get(cluster_name_env, "")

    poster = default_poster
    if poster is None and autoconfigure_poster and dataflow_index:
        from . import post_backend
        from .pipeline import ResultPoster

        post_fn = post_backend.get_retrying_post_fn()
        if post_fn:
            poster = ResultPoster(post_fn=post_fn)

    configure(
        default_poster=poster,
        cluster_name=cluster_name,
        dataflow_index=dataflow_index,
        slack_bot_token=slack_token,
        slack_channel=slack_channel,
    )

    if config.slack_bot_token:
        logger.info(
            "Slack notifications enabled for channel: %s",
            config.slack_channel or "(none)",
        )
    if dataflow_index:
        if config.default_poster is not None:
            logger.info(
                "Attribution posting enabled (index=%s)",
                dataflow_index,
            )
        else:
            logger.warning(
                "dataflow_index set but no post backend "
                "(install nvdataflow, post_backend.set_post_override, or pass default_poster)"
            )
