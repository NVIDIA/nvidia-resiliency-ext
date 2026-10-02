# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP interface for the NVRx Scheduler Exclusion Service."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .monitor import SchedulerExclusionMonitor

logger = logging.getLogger(__name__)


class SchedulerExclusionHttpServer(ThreadingHTTPServer):
    """Threaded cache-only HTTP server."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        monitor: SchedulerExclusionMonitor,
    ) -> None:
        self.monitor = monitor
        super().__init__(server_address, SchedulerExclusionRequestHandler)


class SchedulerExclusionRequestHandler(BaseHTTPRequestHandler):
    """Serve health, cache reads, and asynchronous refresh hints."""

    server: SchedulerExclusionHttpServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/healthz":
            snapshot = self.server.monitor.snapshot()
            self._send_json(
                HTTPStatus.OK,
                {
                    "service": "nvrx-scheduler-exclusion-service",
                    "status": "ok",
                    "job_id": snapshot["job_id"],
                },
            )
            return
        if path == "/stats":
            snapshot = self.server.monitor.snapshot()
            self._send_json(
                HTTPStatus.OK,
                {
                    "job_id": snapshot["job_id"],
                    "last_complete_poll": snapshot["last_complete_poll"],
                    "last_poll_attempt": snapshot["last_poll_attempt"],
                    "last_decision_write": snapshot["last_decision_write"],
                    "last_error": snapshot["last_error"],
                    "last_decision_error": snapshot["last_decision_error"],
                    **snapshot["stats"],
                },
            )
            return
        if path == "/scheduler-exclusions":
            decision = self.server.monitor.scheduler_exclusions()
            if decision is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "scheduler_exclusions_unavailable"},
                )
                return
            self._send_json(HTTPStatus.OK, decision)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._discard_request_body():
            return
        path = urlsplit(self.path).path
        if path != "/refresh":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        accepted = self.server.monitor.request_refresh()
        self._send_json(HTTPStatus.ACCEPTED, {"accepted": accepted})

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _discard_request_body(self) -> bool:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return True
        try:
            length = int(raw_length)
        except ValueError:
            self.close_connection = True
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_content_length"},
            )
            return False
        if length < 0:
            self.close_connection = True
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_content_length"},
            )
            return False
        if length:
            self.rfile.read(length)
        return True

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
