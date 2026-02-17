# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postprocessing for attribution results.

- config: Singleton PostprocessingConfig; set config.default_poster, config.cluster_name, config.dataflow_index, config.slack_bot_token, config.slack_channel at startup.
- base: ResultPoster, post_results, get_default_poster.
- slack: API and maybe_send_slack_notification (used by post_results).

Example:
    from nvidia_resiliency_ext.attribution.postprocessing import configure, ResultPoster, post_results
    configure(
        default_poster=ResultPoster(post_fn=my_dataflow_post),
        cluster_name="my-cluster",
        dataflow_index="my-index",
        slack_bot_token=token,
        slack_channel=channel,
    )
"""

from .base import (
    DataflowStats,
    PostFunction,
    ResultPoster,
    get_dataflow_stats,
    get_default_poster,
    post_results,
)
from .config import PostprocessingConfig, config, configure
from .slack import (
    HAS_SLACK,
    SlackStats,
    get_slack_stats,
    get_slack_user_id,
    maybe_send_slack_notification,
    send_slack_notification,
    should_notify_slack,
)

__all__ = [
    # Config (assign at startup or use configure())
    "PostprocessingConfig",
    "config",
    "configure",
    # Base
    "DataflowStats",
    "PostFunction",
    "ResultPoster",
    "get_dataflow_stats",
    "get_default_poster",
    "post_results",
    # Slack
    "HAS_SLACK",
    "SlackStats",
    "get_slack_stats",
    "get_slack_user_id",
    "maybe_send_slack_notification",
    "send_slack_notification",
    "should_notify_slack",
]
