#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""HTTP status server for nvrx_smonsvc."""

import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


def create_status_handler(
    get_stats: Callable[[], dict],
    get_jobs: Callable[[], list],
    get_health: Callable[[], tuple],
) -> type:
    """
    Create a request handler class with access to monitor data.

    Args:
        get_stats: Callback to get monitor statistics dict
        get_jobs: Callback to get list of job dicts
        get_health: Callback returning (is_healthy: bool, details: dict)

    Returns:
        A BaseHTTPRequestHandler subclass
    """

    class StatusHandler(BaseHTTPRequestHandler):
        """HTTP handler for status endpoints."""

        def log_message(self, format: str, *args) -> None:
            """Suppress default logging to avoid noise."""
            pass

        def _parse_path(self) -> tuple[str, dict]:
            """Parse path and query parameters."""
            parsed = urlparse(self.path)
            query_params = parse_qs(parsed.query)
            return parsed.path, query_params

        def _is_pretty(self, query_params: dict) -> bool:
            """Check if pretty=true is requested."""
            pretty_values = query_params.get("pretty", [])
            return any(v.lower() in ("true", "1", "yes") for v in pretty_values)

        def _send_json(self, data: dict, status: int = 200, pretty: bool = True) -> None:
            """Send JSON response."""
            indent = 2 if pretty else None
            body = json.dumps(data, indent=indent, default=str).encode("utf-8")
            self._send_raw(body, status)

        def _send_raw(self, body: bytes, status: int = 200) -> None:
            """Send raw response body."""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            """Handle GET requests."""
            path, query_params = self._parse_path()
            pretty = self._is_pretty(query_params)

            if path in ("/stats", "/"):
                self._handle_stats(pretty)
            elif path == "/jobs":
                self._handle_jobs(pretty)
            elif path == "/healthz":
                self._handle_health(pretty)
            else:
                self._send_json({"error": "Not found"}, 404, pretty)

        def _handle_health(self, pretty: bool) -> None:
            """Health check endpoint."""
            is_healthy, details = get_health()
            status_code = 200 if is_healthy else 503
            response = {"status": "ok" if is_healthy else "degraded", **details}
            self._send_json(response, status_code, pretty)

        def _handle_stats(self, pretty: bool) -> None:
            """Return monitor statistics."""
            self._send_json(get_stats(), pretty=pretty)

        def _handle_jobs(self, pretty: bool) -> None:
            """Return all jobs with their state."""
            jobs_list = get_jobs()
            self._send_json({"jobs": jobs_list, "count": len(jobs_list)}, pretty=pretty)

    return StatusHandler


class StatusServer:
    """
    HTTP status server for monitoring endpoints.

    Runs in a background thread and provides /stats, /jobs, /healthz endpoints.
    """

    def __init__(
        self,
        port: int,
        get_stats: Callable[[], dict],
        get_jobs: Callable[[], list],
        get_health: Callable[[], tuple],
    ):
        """
        Initialize the status server.

        Args:
            port: Port to bind to
            get_stats: Callback to get monitor statistics dict
            get_jobs: Callback to get list of job dicts
            get_health: Callback returning (is_healthy: bool, details: dict)
        """
        self._port = port
        self._get_stats = get_stats
        self._get_jobs = get_jobs
        self._get_health = get_health
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        handler_class = create_status_handler(
            self._get_stats,
            self._get_jobs,
            self._get_health,
        )
        try:
            self._server = HTTPServer(("0.0.0.0", self._port), handler_class)
        except OSError as e:
            logger.error(f"Failed to bind status server to port {self._port}: {e}")
            raise SystemExit(1) from e

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="status-server",
        )
        self._thread.start()
        logger.info(f"  Status server: http://0.0.0.0:{self._port}")

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "StatusServer":
        """Context manager entry - starts the server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops the server."""
        self.stop()
