# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution postprocessing: configure poster, post results, optional Slack.

- :data:`config` — cluster, Slack, default :class:`ResultPoster`.
- :func:`post_results` — build record, log, post, maybe Slack.
- :func:`post_analysis_items` — post each cycle item (LogSage + optional FR).
- :func:`build_dataflow_record` — build the dataflow dict (LogSage + optional FR fields).
- :mod:`nvidia_resiliency_ext.attribution.postprocessing.post_backend` — retrying post
  (custom override or direct HTTP).

Example:

    from nvidia_resiliency_ext.attribution.postprocessing import (
        ResultPoster,
        configure,
        post_results,
        post_backend,
    )

    configure(
        default_poster=ResultPoster(post_fn=post_backend.post),
        cluster_name="my-cluster",
    )
"""

from . import post_backend
from .config import PostprocessingConfig, config, configure, configure_from_env, load_slack_from_env
from .pipeline import (
    PostFunction,
    PostingStats,
    ResultPoster,
    build_dataflow_record,
    get_default_poster,
    get_posting_stats,
    post_analysis_items,
    post_results,
)
from .slack import (
    HAS_SLACK,
    SlackStats,
    get_slack_stats,
    get_slack_user_id,
    maybe_send_slack_notification,
    send_slack_notification,
    should_notify_slack,
)

get_retrying_post_fn = post_backend.get_retrying_post_fn
make_dataflow_http_post_fn = post_backend.make_dataflow_http_post_fn
set_post_override = post_backend.set_post_override

__all__ = [
    "PostprocessingConfig",
    "config",
    "configure",
    "configure_from_env",
    "load_slack_from_env",
    "post_backend",
    "get_retrying_post_fn",
    "make_dataflow_http_post_fn",
    "set_post_override",
    "PostingStats",
    "PostFunction",
    "ResultPoster",
    "get_posting_stats",
    "get_default_poster",
    "post_results",
    "post_analysis_items",
    "build_dataflow_record",
    "SlackStats",
    "get_slack_stats",
    "get_slack_user_id",
    "maybe_send_slack_notification",
    "send_slack_notification",
    "should_notify_slack",
    "HAS_SLACK",
]
