# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config for postprocessing (poster, dataflow, Slack). Assign attributes at startup."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def load_slack_from_env() -> Tuple[str, str]:
    """Load Slack token and channel from environment (mirrors load_nvidia_api_key).

    Token checks in order:
    1. SLACK_BOT_TOKEN environment variable
    2. SLACK_BOT_TOKEN_FILE environment variable (path to token file)

    Channel: SLACK_CHANNEL environment variable.

    Returns:
        (token, channel) tuple; empty strings if not found.
    """
    token = ""
    token_val = os.getenv("SLACK_BOT_TOKEN")
    if token_val:
        token = token_val.strip()
    if not token:
        key_file = os.getenv("SLACK_BOT_TOKEN_FILE")
        if key_file and os.path.isfile(key_file):
            try:
                with open(key_file) as f:
                    token = f.read().strip()
            except OSError:
                pass
    channel = (os.getenv("SLACK_CHANNEL") or "").strip()
    return (token, channel)


@dataclass
class PostprocessingConfig:
    """Single place for postprocessing state. Callers set attributes directly (e.g. config.slack_bot_token = ...)."""

    default_poster: Any = None  # ResultPoster; Any to avoid circular import
    cluster_name: str = ""
    dataflow_index: str = ""
    slack_bot_token: str = ""
    slack_channel: str = ""


# Singleton used by base/slack and by service at startup
config = PostprocessingConfig()


def configure(
    *,
    default_poster: Any = None,
    cluster_name: str = "",
    dataflow_index: str = "",
    slack_bot_token: str = "",
    slack_channel: str = "",
) -> None:
    """Set postprocessing config. Pass only what you want to set; default_poster=None leaves it unchanged."""
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


def configure_postprocessing_resolved(
    *,
    default_poster: Any = None,
    cluster_name: str = "",
    dataflow_index: str = "",
    slack_token: Optional[str] = None,
    slack_channel: Optional[str] = None,
    cluster_name_env: Optional[str] = "SLURM_CLUSTER_NAME",
    create_dataflow_poster_if_needed: bool = False,
) -> None:
    """Configure postprocessing singleton. Resolves from env when params are None/empty.

    Centralizes logic used by nvrx_attrsvc and FT lib/mcp. Call this instead of
    manually resolving slack + calling configure().

    Args:
        default_poster: ResultPoster to use (or None).
        cluster_name: Override; if empty and cluster_name_env set, uses that env var.
        dataflow_index: Elasticsearch index for dataflow posting.
        slack_token: Override; None = resolve from SLACK_BOT_TOKEN/SLACK_BOT_TOKEN_FILE.
        slack_channel: Override; None = resolve from SLACK_CHANNEL.
        cluster_name_env: Env var for cluster_name when cluster_name empty (e.g. SLURM_CLUSTER_NAME).
        create_dataflow_poster_if_needed: If True, dataflow_index set, and default_poster None,
            creates ResultPoster from nvdataflow when available.
    """
    if not (slack_token or "").strip() or not (slack_channel or "").strip():
        env_tok, env_ch = load_slack_from_env()
        slack_token = (slack_token or env_tok).strip() if slack_token is not None else env_tok
        slack_channel = (slack_channel or env_ch).strip() if slack_channel is not None else env_ch
    else:
        slack_token = (slack_token or "").strip()
        slack_channel = (slack_channel or "").strip()

    if not cluster_name and cluster_name_env:
        cluster_name = os.environ.get(cluster_name_env, "")

    poster = default_poster
    if poster is None and create_dataflow_poster_if_needed and dataflow_index:
        from .base import ResultPoster
        from .dataflow import get_nvdataflow_post_fn

        post_fn = get_nvdataflow_post_fn()
        if post_fn:
            poster = ResultPoster(post_fn=post_fn)

    configure(
        default_poster=poster,
        cluster_name=cluster_name,
        dataflow_index=dataflow_index,
        slack_bot_token=slack_token,
        slack_channel=slack_channel,
    )

    # Log status of optional integrations
    if config.slack_bot_token:
        logger.info(
            "Slack notifications enabled for channel: %s",
            config.slack_channel or "(none)",
        )
    if dataflow_index:
        if config.default_poster is not None:
            logger.info(
                "Dataflow posting enabled for attribution (index=%s)",
                dataflow_index,
            )
        else:
            logger.warning(
                "dataflow_index set but nvdataflow not installed; dataflow posting disabled"
            )
