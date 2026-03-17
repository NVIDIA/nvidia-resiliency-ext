# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config for postprocessing (poster, dataflow, Slack). Assign attributes at startup."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


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
