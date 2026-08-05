# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reporting. Deliberately separate from observation: if the scheduler is unreachable,
the alert path must still work, so blindness itself pages."""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Protocol

from .types import Finding

logger = logging.getLogger("nvrx_watch")

PAGERDUTY_URL = "https://events.pagerduty.com/v2/enqueue"
HTTP_TIMEOUT = 20.0


class Sink(Protocol):
    """A place a finding is emitted to. Mirrors the Platform protocol: one declared
    contract, several interchangeable implementations, added without touching runner."""

    name: str

    def emit(self, finding: Finding) -> bool: ...


def _post_json(url: str, payload: dict) -> bool:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("POST to %s failed: %s", url.split("?")[0], exc)
        return False


class LogSink:
    """Always on. The record of what the watcher decided, whether or not paging works."""

    name = "log"

    def emit(self, finding: Finding) -> bool:
        level = {"critical": logging.ERROR, "warning": logging.WARNING}.get(
            finding.severity, logging.INFO
        )
        logger.log(level, "[%s] %s %s", finding.severity.upper(), finding.key, finding.summary)
        if finding.detail:
            logger.log(level, "    %s", finding.detail)
        return True


class PagerDutySink:
    """PagerDuty Events v2. dedup_key is deterministic, so every login node running this
    watcher collapses into one incident."""

    name = "pagerduty"

    def __init__(self, routing_key: str, source: str | None = None) -> None:
        self._routing_key = routing_key
        self._source = source or socket.gethostname()

    def emit(self, finding: Finding) -> bool:
        return _post_json(
            PAGERDUTY_URL,
            {
                "routing_key": self._routing_key,
                "event_action": "trigger",
                "dedup_key": finding.key,
                "payload": {
                    "summary": finding.summary,
                    "source": self._source,
                    "severity": finding.severity,
                    "component": "nvrx-watch",
                    "custom_details": {
                        "detector": finding.detector,
                        "detail": finding.detail,
                    },
                },
            },
        )


class WebhookSink:
    """Generic JSON POST, for Slack-compatible endpoints or anything home-grown."""

    name = "webhook"

    def __init__(self, url: str) -> None:
        self._url = url

    def emit(self, finding: Finding) -> bool:
        return _post_json(
            self._url,
            {
                "key": finding.key,
                "detector": finding.detector,
                "severity": finding.severity,
                "summary": finding.summary,
                "detail": finding.detail,
                "text": f"[{finding.severity.upper()}] {finding.summary}",
            },
        )


def heartbeat(url: str) -> bool:
    """Ping a dead-man timer. Called only after a pass that could actually observe."""
    if not url:
        return True
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("heartbeat GET failed: %s", exc)
        return False


def build(config) -> list[Sink]:
    sinks: list[Sink] = [LogSink()]
    if config.pd_routing_key:
        sinks.append(PagerDutySink(config.pd_routing_key))
    if config.webhook_url:
        sinks.append(WebhookSink(config.webhook_url))
    return sinks
