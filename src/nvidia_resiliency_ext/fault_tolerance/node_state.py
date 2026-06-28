# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for the scheduler node-state service used by FT rendezvous."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

DEFAULT_NODE_STATE_DECISION_TIMEOUT_SECONDS = 60.0
DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS = 5.0
DEFAULT_NODE_STATE_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_NODE_STATE_TIMEOUT_SECONDS = DEFAULT_NODE_STATE_DECISION_TIMEOUT_SECONDS


@dataclass(frozen=True)
class NodeStateCycleStatus:
    """Scheduler node-state result for one NVRx cycle."""

    job_id: str
    cycle_id: str
    available: bool
    bad_nodes: tuple[str, ...] = ()
    unknown_nodes: tuple[str, ...] = ()
    nodes: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    error: Optional[str] = None
    retryable: bool = False

    @property
    def scheduler_avoid_nodes(self) -> tuple[str, ...]:
        """Nodes that scheduler state says must not be active in the next cycle."""
        return tuple(dict.fromkeys((*self.bad_nodes, *self.unknown_nodes)))


class NodeStateServiceClient:
    """Small fail-open HTTP client for nvrx-nodestatesvc."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = DEFAULT_NODE_STATE_DECISION_TIMEOUT_SECONDS,
        request_timeout: float = DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS,
        poll_interval: float = DEFAULT_NODE_STATE_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.decision_timeout = float(timeout)
        if self.decision_timeout <= 0:
            raise ValueError("node-state service timeout must be positive")
        self.request_timeout = float(request_timeout)
        if self.request_timeout <= 0:
            raise ValueError("node-state service request timeout must be positive")
        self.poll_interval = float(poll_interval)
        if self.poll_interval < 0:
            raise ValueError("node-state service poll interval must be non-negative")

    def post_cycle_start(self, job_id: str, cycle_id: str, nodes: Sequence[str]) -> bool:
        payload = {"job_id": str(job_id), "cycle_id": str(cycle_id), "nodes": list(nodes)}
        return self._post("/v1/cycles/start", payload, action="cycle start")

    def post_cycle_end(self, job_id: str, cycle_id: str) -> bool:
        payload = {"job_id": str(job_id), "cycle_id": str(cycle_id)}
        return self._post("/v1/cycles/end", payload, action="cycle end")

    def get_cycle_node_states(
        self,
        job_id: str,
        cycle_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> NodeStateCycleStatus:
        clean_job_id = str(job_id)
        clean_cycle_id = str(cycle_id)
        request_timeout = self.request_timeout if timeout is None else timeout
        try:
            with httpx.Client(base_url=self.endpoint, timeout=request_timeout) as client:
                response = client.get(
                    f"/v1/cycles/{clean_cycle_id}/node-states",
                    params={"job_id": clean_job_id},
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Node-state service GET failed for job %s cycle %s: %s",
                clean_job_id,
                clean_cycle_id,
                message,
            )
            return NodeStateCycleStatus(
                job_id=clean_job_id,
                cycle_id=clean_cycle_id,
                available=False,
                error=message,
                retryable=True,
            )

        if response.status_code != 200:
            message = _response_error_message(response)
            retryable = response.status_code == 409 and "cycle_status_not_ready" in message
            logger.warning(
                "Node-state service GET for job %s cycle %s returned %s: %s",
                clean_job_id,
                clean_cycle_id,
                response.status_code,
                message,
            )
            return NodeStateCycleStatus(
                job_id=clean_job_id,
                cycle_id=clean_cycle_id,
                available=False,
                error=message,
                retryable=retryable,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            message = f"invalid JSON response: {exc}"
            logger.warning(
                "Node-state service GET for job %s cycle %s returned %s",
                clean_job_id,
                clean_cycle_id,
                message,
            )
            return NodeStateCycleStatus(
                job_id=clean_job_id,
                cycle_id=clean_cycle_id,
                available=False,
                error=message,
            )

        return NodeStateCycleStatus(
            job_id=str(payload.get("job_id") or clean_job_id),
            cycle_id=str(payload.get("cycle_id") or clean_cycle_id),
            available=True,
            bad_nodes=tuple(str(node) for node in payload.get("bad_nodes", [])),
            unknown_nodes=tuple(str(node) for node in payload.get("unknown_nodes", [])),
            nodes=tuple(payload.get("nodes", [])),
        )

    def end_cycle_and_get_node_states(self, job_id: str, cycle_id: str) -> NodeStateCycleStatus:
        deadline = time.monotonic() + self.decision_timeout
        if not self.post_cycle_end(job_id, cycle_id):
            return NodeStateCycleStatus(
                job_id=str(job_id),
                cycle_id=str(cycle_id),
                available=False,
                error="node-state cycle-end notification failed",
            )
        last_status: Optional[NodeStateCycleStatus] = None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = f" last error: {last_status.error}" if last_status else ""
                return NodeStateCycleStatus(
                    job_id=str(job_id),
                    cycle_id=str(cycle_id),
                    available=False,
                    error=(
                        "node-state decision budget exhausted after "
                        f"{self.decision_timeout:.1f}s.{last_error}"
                    ),
                )

            status = self.get_cycle_node_states(
                job_id,
                cycle_id,
                timeout=min(self.request_timeout, remaining),
            )
            if status.available or not status.retryable:
                return status

            last_status = status
            sleep_seconds = min(self.poll_interval, max(0.0, deadline - time.monotonic()))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    def _post(self, path: str, payload: dict[str, Any], *, action: str) -> bool:
        try:
            with httpx.Client(base_url=self.endpoint, timeout=self.request_timeout) as client:
                response = client.post(path, json=payload)
        except Exception as exc:
            logger.warning(
                "Node-state service POST %s failed for job %s cycle %s: %s: %s",
                action,
                payload.get("job_id"),
                payload.get("cycle_id"),
                type(exc).__name__,
                exc,
            )
            return False

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "Node-state service POST %s for job %s cycle %s returned %s: %s",
                action,
                payload.get("job_id"),
                payload.get("cycle_id"),
                response.status_code,
                response.text.strip(),
            )
            return False

        return True


def _response_error_message(response: httpx.Response) -> str:
    text = response.text.strip()
    try:
        payload = response.json() if text else {}
    except ValueError:
        payload = {}

    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return text or f"HTTP {response.status_code}"


class NodeStateCycleReporter:
    """Reports NVRx cycle lifecycle events to nvrx-nodestatesvc."""

    def __init__(
        self,
        endpoint: Optional[str],
        *,
        job_id: str,
        is_enabled: bool,
    ) -> None:
        self.job_id = str(job_id)
        self.client: Optional[NodeStateServiceClient] = (
            NodeStateServiceClient(endpoint) if endpoint and is_enabled else None
        )
        self._cycle_start_reported: set[str] = set()
        self._cycle_end_requested: set[str] = set()
        self._cycle_status_by_id: dict[str, NodeStateCycleStatus] = {}
        self._cycle_deadline_by_id: dict[str, float] = {}
        self._last_pending_status_by_id: dict[str, NodeStateCycleStatus] = {}
        self._next_probe_time_by_id: dict[str, float] = {}

    def report_cycle_start(self, cycle_id: str, nodes: Sequence[str]) -> None:
        if self.client is None:
            return

        clean_cycle_id = str(cycle_id)
        active_nodes = [str(node) for node in nodes if str(node)]
        if not active_nodes:
            logger.debug(
                "Node-state cycle start skipped for cycle %s: no active nodes", clean_cycle_id
            )
            return

        self._cycle_start_reported.add(clean_cycle_id)
        if self.client.post_cycle_start(self.job_id, clean_cycle_id, active_nodes):
            logger.info(
                "Node-state service registered cycle %s with %s active nodes",
                clean_cycle_id,
                len(active_nodes),
            )

    def request_cycle_end(self, cycle_id: str) -> bool:
        if self.client is None:
            return False

        clean_cycle_id = str(cycle_id)
        if clean_cycle_id in self._cycle_end_requested:
            return True

        if clean_cycle_id not in self._cycle_start_reported:
            status = NodeStateCycleStatus(
                job_id=self.job_id,
                cycle_id=clean_cycle_id,
                available=False,
                error="node-state cycle-end skipped because cycle start was not reported",
            )
            self._cycle_end_requested.add(clean_cycle_id)
            self._cycle_status_by_id[clean_cycle_id] = status
            logger.debug(
                "Node-state service cycle-end skipped for cycle %s: cycle start was not reported",
                clean_cycle_id,
            )
            return False

        if not self.client.post_cycle_end(self.job_id, clean_cycle_id):
            status = NodeStateCycleStatus(
                job_id=self.job_id,
                cycle_id=clean_cycle_id,
                available=False,
                error="node-state cycle-end notification failed",
            )
            self._cycle_end_requested.add(clean_cycle_id)
            self._cycle_status_by_id[clean_cycle_id] = status
            self._log_status(status)
            return False

        self._cycle_end_requested.add(clean_cycle_id)
        self._cycle_deadline_by_id[clean_cycle_id] = time.monotonic() + self.client.decision_timeout
        logger.info("Node-state service accepted cycle-end request for cycle %s", clean_cycle_id)
        return True

    def get_cycle_status(self, cycle_id: str) -> Optional[NodeStateCycleStatus]:
        """Return a resolved cycle status, or ``None`` while the service is still working."""
        if self.client is None:
            return None

        clean_cycle_id = str(cycle_id)
        cached_status = self._cycle_status_by_id.get(clean_cycle_id)
        if cached_status is not None:
            return cached_status

        if clean_cycle_id not in self._cycle_end_requested:
            if not self.request_cycle_end(clean_cycle_id):
                return self._cycle_status_by_id.get(clean_cycle_id)

        deadline = self._cycle_deadline_by_id.get(clean_cycle_id)
        if deadline is None:
            deadline = time.monotonic() + self.client.decision_timeout
            self._cycle_deadline_by_id[clean_cycle_id] = deadline

        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            last_status = self._last_pending_status_by_id.get(clean_cycle_id)
            last_error = f" last error: {last_status.error}" if last_status else ""
            status = NodeStateCycleStatus(
                job_id=self.job_id,
                cycle_id=clean_cycle_id,
                available=False,
                error=(
                    "node-state decision budget exhausted after "
                    f"{self.client.decision_timeout:.1f}s.{last_error}"
                ),
            )
            self._cycle_status_by_id[clean_cycle_id] = status
            self._log_status(status)
            return status

        next_probe_time = self._next_probe_time_by_id.get(clean_cycle_id)
        if next_probe_time is not None and now < next_probe_time:
            return None

        status = self.client.get_cycle_node_states(
            self.job_id,
            clean_cycle_id,
            timeout=min(self.client.request_timeout, remaining),
        )
        if status.available or not status.retryable:
            self._cycle_status_by_id[clean_cycle_id] = status
            self._log_status(status)
            return status

        self._last_pending_status_by_id[clean_cycle_id] = status
        self._next_probe_time_by_id[clean_cycle_id] = time.monotonic() + self.client.poll_interval
        return None

    def report_cycle_end(self, cycle_id: str) -> Optional[NodeStateCycleStatus]:
        if self.client is None:
            return None

        clean_cycle_id = str(cycle_id)
        if clean_cycle_id in self._cycle_status_by_id:
            return None

        if not self.request_cycle_end(clean_cycle_id):
            return self._cycle_status_by_id.get(clean_cycle_id)

        while True:
            status = self.get_cycle_status(clean_cycle_id)
            if status is not None:
                return status

            deadline = self._cycle_deadline_by_id.get(clean_cycle_id, time.monotonic())
            sleep_seconds = min(
                self.client.poll_interval,
                max(0.0, deadline - time.monotonic()),
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    def _log_status(self, status: NodeStateCycleStatus) -> None:
        if not status.available:
            logger.warning(
                "Node-state service unavailable for cycle %s; continuing without "
                "scheduler-state avoids: %s",
                status.cycle_id,
                status.error or "unknown error",
            )
            return

        if status.bad_nodes or status.unknown_nodes:
            logger.warning(
                "Node-state service reported scheduler-bad nodes for cycle %s: "
                "bad_nodes=%s unknown_nodes=%s details=%s",
                status.cycle_id,
                list(status.bad_nodes),
                list(status.unknown_nodes),
                list(status.nodes),
            )
            return

        logger.info(
            "Node-state service reported no scheduler-bad nodes for cycle %s", status.cycle_id
        )
