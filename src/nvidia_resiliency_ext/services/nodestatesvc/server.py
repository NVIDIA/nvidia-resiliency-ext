# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP API for the NVRx node-state service."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol
from urllib.parse import parse_qs, urlparse

from .slurm import NodeStateRecord, SlurmNodeStateClient, SlurmQueryError

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"  # nosec B104 - service container bind address
DEFAULT_PORT = 8000
MAX_REQUEST_BYTES = 1024 * 1024
DEFAULT_MAX_CYCLES = 1024


class NodeStateClient(Protocol):
    """Protocol implemented by Slurm-backed and test node-state clients."""

    def check_available(self) -> tuple[bool, str | None]:
        """Return whether the node-state backend is available."""

    def get_node_states(self, nodes: Sequence[str]) -> dict[str, NodeStateRecord]:
        """Return node-state records keyed by node name."""


class NodeStateService:
    """Thin service layer around the node-state backend."""

    def __init__(
        self,
        client: NodeStateClient | None = None,
        *,
        max_cycles: int = DEFAULT_MAX_CYCLES,
    ):
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        self._client = client if client is not None else SlurmNodeStateClient()
        self._cycles: dict[tuple[str, str], CycleRegistration] = {}
        self._max_cycles = max_cycles
        self._lock = threading.Lock()

    def health(self) -> dict:
        return {"status": "ok", "service": "nvrx-nodestatesvc"}

    def readiness(self) -> tuple[int, dict]:
        available, error = self._client.check_available()
        if available:
            return 200, {"status": "ok", "backend": "slurm"}
        return 503, {"status": "degraded", "backend": "slurm", "error": error}

    def query_node_states(self, nodes: Sequence[str]) -> tuple[int, dict]:
        clean_nodes = _dedupe_nodes(nodes)
        if not clean_nodes:
            return 400, {"error": "nodes must contain at least one node name"}

        status, body = self._query_node_states(clean_nodes)
        if status != 200:
            return status, body
        return status, body

    def register_cycle_start(
        self, job_id: str, cycle_id: str, nodes: Sequence[str]
    ) -> tuple[int, dict]:
        clean_job_id = str(job_id).strip()
        if not clean_job_id:
            return 400, {"error": "job_id must be non-empty"}

        clean_cycle_id = str(cycle_id).strip()
        if not clean_cycle_id:
            return 400, {"error": "cycle_id must be non-empty"}

        clean_nodes = _dedupe_nodes(nodes)
        if not clean_nodes:
            return 400, {"error": "nodes must contain at least one node name"}

        with self._lock:
            self._cycles[(clean_job_id, clean_cycle_id)] = CycleRegistration(
                job_id=clean_job_id,
                cycle_id=clean_cycle_id,
                nodes=clean_nodes,
                started_at=time.time(),
            )
            self._evict_old_cycles_locked()
        return (
            200,
            {
                "job_id": clean_job_id,
                "cycle_id": clean_cycle_id,
                "registered_nodes": len(clean_nodes),
            },
        )

    def register_cycle_end(self, job_id: str, cycle_id: str) -> tuple[int, dict]:
        if not str(job_id).strip():
            return 400, {"error": "job_id must be non-empty"}
        if not str(cycle_id).strip():
            return 400, {"error": "cycle_id must be non-empty"}

        with self._lock:
            registration = self._cycle_registration_locked(job_id, cycle_id)
            if registration is None:
                return (
                    404,
                    {
                        "error": "cycle_not_found",
                        "job_id": str(job_id),
                        "cycle_id": str(cycle_id),
                    },
                )

            if registration.status_code is not None and registration.status_body is not None:
                return registration.status_code, registration.status_body

            if registration.materializing:
                return (
                    202,
                    {
                        "job_id": registration.job_id,
                        "cycle_id": registration.cycle_id,
                        "accepted": True,
                        "materializing": True,
                        "registered_nodes": len(registration.nodes),
                    },
                )

            registration.materializing = True
            registration.ended_at = time.time()
            registration_nodes = list(registration.nodes)
            clean_job_id = registration.job_id
            clean_cycle_id = registration.cycle_id

        worker = threading.Thread(
            target=self._materialize_cycle_node_states,
            args=(registration, registration_nodes),
            name=f"nvrx-nodestate-cycle-{clean_job_id}-{clean_cycle_id}",
            daemon=True,
        )
        worker.start()
        return (
            202,
            {
                "job_id": clean_job_id,
                "cycle_id": clean_cycle_id,
                "accepted": True,
                "materializing": True,
                "registered_nodes": len(registration_nodes),
            },
        )

    def _materialize_cycle_node_states(
        self,
        registration: "CycleRegistration",
        registration_nodes: Sequence[str],
    ) -> None:
        registered_nodes = len(registration_nodes)
        clean_job_id = registration.job_id
        clean_cycle_id = registration.cycle_id

        try:
            query_status, query_body = self._query_node_states(registration_nodes)
        except Exception as exc:
            logger.exception(
                "Unexpected node-state materialization failure for job %s cycle %s",
                clean_job_id,
                clean_cycle_id,
            )
            query_status, query_body = (
                503,
                {
                    "error": "node_state_materialization_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            )
        if query_status != 200:
            status_body = {
                **query_body,
                "job_id": clean_job_id,
                "cycle_id": clean_cycle_id,
                "registered_nodes": registered_nodes,
                "ended": True,
            }
            self._store_cycle_status(registration, query_status, status_body)
            return

        problem_records = _problem_records(query_body["nodes"])
        status_body = {
            "job_id": clean_job_id,
            "cycle_id": clean_cycle_id,
            "requested_nodes": query_body["requested_nodes"],
            "bad_nodes": query_body["bad_nodes"],
            "unknown_nodes": query_body["unknown_nodes"],
            "nodes": problem_records,
            "registered_nodes": registered_nodes,
            "ended": True,
            "queried_at": time.time(),
        }
        self._store_cycle_status(registration, 200, status_body)

    def query_cycle_node_states(self, job_id: str, cycle_id: str) -> tuple[int, dict]:
        if not str(job_id).strip():
            return 400, {"error": "job_id must be non-empty"}
        if not str(cycle_id).strip():
            return 400, {"error": "cycle_id must be non-empty"}

        with self._lock:
            registration = self._cycle_registration_locked(job_id, cycle_id)
            if registration is None:
                return (
                    404,
                    {
                        "error": "cycle_not_found",
                        "job_id": str(job_id),
                        "cycle_id": str(cycle_id),
                    },
                )

            if registration.status_code is None or registration.status_body is None:
                return (
                    409,
                    {
                        "error": "cycle_status_not_ready",
                        "job_id": registration.job_id,
                        "cycle_id": registration.cycle_id,
                    },
                )

            return registration.status_code, dict(registration.status_body)

    def _cycle_registration_locked(self, job_id: str, cycle_id: str) -> "CycleRegistration | None":
        return self._cycles.get((str(job_id).strip(), str(cycle_id).strip()))

    def _evict_old_cycles_locked(self) -> None:
        while len(self._cycles) > self._max_cycles:
            oldest_key = min(self._cycles, key=lambda key: self._cycles[key].started_at)
            del self._cycles[oldest_key]

    def _store_cycle_status(
        self,
        registration: "CycleRegistration",
        status_code: int,
        status_body: dict,
    ) -> None:
        with self._lock:
            registration.status_code = status_code
            registration.status_body = status_body
            registration.materializing = False
            registration.nodes = []

    def _query_node_states(self, clean_nodes: Sequence[str]) -> tuple[int, dict]:
        try:
            records = self._client.get_node_states(clean_nodes)
        except SlurmQueryError as exc:
            return 503, {"error": "slurm_query_failed", "message": str(exc)}

        node_records = [records[node].to_dict() for node in clean_nodes if node in records]
        bad_nodes = [record["node"] for record in node_records if record["bad"]]
        unknown_nodes = [record["node"] for record in node_records if record["state"] == "UNKNOWN"]
        return (
            200,
            {
                "source": "slurm:sinfo",
                "requested_nodes": len(clean_nodes),
                "returned_nodes": len(node_records),
                "bad_nodes": bad_nodes,
                "unknown_nodes": unknown_nodes,
                "nodes": node_records,
            },
        )


@dataclass
class CycleRegistration:
    job_id: str
    cycle_id: str
    nodes: list[str]
    started_at: float
    ended_at: float | None = None
    status_code: int | None = None
    status_body: dict | None = None
    materializing: bool = False


def _dedupe_nodes(nodes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for node in nodes:
        clean = str(node).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _problem_records(records: Sequence[dict]) -> list[dict]:
    return [record for record in records if record.get("bad") or record.get("state") == "UNKNOWN"]


def _json_bytes(data: dict, *, pretty: bool = False) -> bytes:
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent, sort_keys=True).encode("utf-8")


def _parse_nodes_from_query(query_params: dict[str, list[str]]) -> list[str]:
    nodes: list[str] = []
    for value in query_params.get("nodes", []):
        nodes.extend(part.strip() for part in value.split(","))
    return nodes


def _cycle_node_state_id(path: str) -> str | None:
    parts = path.strip("/").split("/")
    if len(parts) == 4 and parts[0] == "v1" and parts[1] == "cycles" and parts[3] == "node-states":
        return parts[2]
    return None


def create_handler(service: NodeStateService) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler bound to a NodeStateService instance."""

    class NodeStateHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

        def _send_json(self, status: int, body: dict, *, pretty: bool = False) -> None:
            payload = _json_bytes(body, pretty=pretty)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _parse_path(self) -> tuple[str, dict[str, list[str]]]:
            parsed = urlparse(self.path)
            return parsed.path, parse_qs(parsed.query)

        def do_GET(self) -> None:
            path, query_params = self._parse_path()
            pretty = any(v.lower() in ("1", "true", "yes") for v in query_params.get("pretty", []))

            if path in ("/", "/healthz"):
                self._send_json(200, service.health(), pretty=pretty)
                return
            if path == "/readyz":
                status, body = service.readiness()
                self._send_json(status, body, pretty=pretty)
                return
            if path == "/v1/node-states":
                status, body = service.query_node_states(_parse_nodes_from_query(query_params))
                self._send_json(status, body, pretty=pretty)
                return
            cycle_id = _cycle_node_state_id(path)
            if cycle_id is not None:
                job_id = query_params.get("job_id", [""])[0]
                status, body = service.query_cycle_node_states(job_id, cycle_id)
                self._send_json(status, body, pretty=pretty)
                return
            self._send_json(404, {"error": "not_found"}, pretty=pretty)

        def do_POST(self) -> None:
            path, query_params = self._parse_path()
            pretty = any(v.lower() in ("1", "true", "yes") for v in query_params.get("pretty", []))
            if path not in ("/v1/node-states", "/v1/cycles/start", "/v1/cycles/end"):
                self._send_json(404, {"error": "not_found"}, pretty=pretty)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                self._send_json(400, {"error": "invalid_content_length"}, pretty=pretty)
                return
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(400, {"error": "invalid_request_size"}, pretty=pretty)
                return

            try:
                request = json.loads(self.rfile.read(content_length))
            except ValueError:
                self._send_json(400, {"error": "invalid_json"}, pretty=pretty)
                return

            if path == "/v1/cycles/start":
                nodes = request.get("nodes", [])
                if not isinstance(nodes, list):
                    self._send_json(400, {"error": "nodes must be a list"}, pretty=pretty)
                    return
                status, body = service.register_cycle_start(
                    request.get("job_id", ""),
                    request.get("cycle_id", ""),
                    nodes,
                )
                self._send_json(status, body, pretty=pretty)
                return

            if path == "/v1/cycles/end":
                status, body = service.register_cycle_end(
                    request.get("job_id", ""),
                    request.get("cycle_id", ""),
                )
                self._send_json(status, body, pretty=pretty)
                return

            nodes = request.get("nodes", [])
            if not isinstance(nodes, list):
                self._send_json(400, {"error": "nodes must be a list"}, pretty=pretty)
                return

            status, body = service.query_node_states(nodes)
            self._send_json(status, body, pretty=pretty)

    return NodeStateHandler


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    service: NodeStateService | None = None,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer for nvrx-nodestatesvc."""
    bound_service = service if service is not None else NodeStateService()
    server = ThreadingHTTPServer((host, port), create_handler(bound_service))
    logger.info("nvrx-nodestatesvc listening on http://%s:%s", host, port)
    return server
