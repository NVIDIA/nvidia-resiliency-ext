# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic, inspectable lifecycle artifacts for local collect-all execution."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..execution import L0Artifacts
from ..runtime import SYSTEM_CLOCK, Clock
from .artifact_io import write_json_atomic

LIVE_STATUS_SCHEMA_VERSION = "restart_agent_live_status.v1"
LIVE_EVENT_SCHEMA_VERSION = "restart_agent_live_event.v1"
DETERMINISTIC_FALLBACK_SCHEMA_VERSION = "restart_agent_deterministic_fallback.v1"


class LiveArtifactWriter:
    """Publish complete snapshots as a collect-all analysis advances."""

    def __init__(self, root: str | Path, *, clock: Clock = SYSTEM_CLOCK) -> None:
        self.root = Path(root)
        self.events_path = self.root / "events.jsonl"
        self.status_path = self.root / "run_status.json"
        self._lock = threading.Lock()
        self._clock = clock
        self._started_monotonic = clock.monotonic()
        self._sequence = 0
        self._status: dict[str, Any] = {}

    def start(
        self,
        *,
        routes: Sequence[Mapping[str, Any]],
        config_metadata: Mapping[str, Any],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        route_status: dict[str, Any] = {}
        for route in routes:
            route_id = str(route.get("route_id") or "unknown")
            route_status[route_id] = {
                "status": "pending",
                "model": route.get("model"),
                "endpoint": route.get("endpoint"),
                "credential_ref": route.get("credential_ref"),
            }
        now = self._utc_now()
        self._status = {
            "schema_version": LIVE_STATUS_SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": now,
            "updated_at_utc": now,
            "elapsed_s": 0.0,
            "l0": {"status": "pending"},
            "fallback": {"status": "pending"},
            "routes": route_status,
            "completed_routes": 0,
            "total_routes": len(route_status),
            "config": _jsonable(config_metadata),
        }
        self.events_path.write_text("", encoding="utf-8")
        write_json_atomic(self.status_path, self._status)
        self._append_event("run_started", route_count=len(route_status))

    def publish_l0_artifacts(
        self,
        artifacts: L0Artifacts,
        artifact_paths: Mapping[str, str],
    ) -> None:
        """Record that canonical shared L0 artifacts are durable and inspectable."""

        with self._lock:
            display_paths = {
                name: _display_path(path, relative_to=self.root.parent)
                for name, path in artifact_paths.items()
            }
            self._status["l0"] = {
                "status": "ready",
                "artifacts": display_paths,
                "l0_wall_clock_s": artifacts.l0_wall_clock_s,
                "l0a_wall_clock_s": artifacts.l0a_wall_clock_s,
                "decision_evidence_wall_clock_s": (artifacts.decision_evidence_wall_clock_s),
                "l0b_wall_clock_s": artifacts.l0b_wall_clock_s,
                "l0_reused": artifacts.l0_reused,
            }
            self._update_status()
            self._append_event(
                "l0_artifacts_ready",
                artifacts=display_paths,
                l0_wall_clock_s=artifacts.l0_wall_clock_s,
                l0a_wall_clock_s=artifacts.l0a_wall_clock_s,
                decision_evidence_wall_clock_s=(artifacts.decision_evidence_wall_clock_s),
                l0b_wall_clock_s=artifacts.l0b_wall_clock_s,
                l0_reused=artifacts.l0_reused,
            )

    def publish_fallback(
        self,
        candidate: Mapping[str, Any],
        *,
        artifact_path: str | None,
    ) -> None:
        with self._lock:
            result = _mapping(candidate.get("result"))
            display_path = (
                _display_path(artifact_path, relative_to=self.root.parent)
                if artifact_path
                else None
            )
            self._status["fallback"] = {
                "status": "ready",
                "decision": result.get("decision"),
                "decision_basis": result.get("decision_basis"),
                "ready_wall_clock_s": candidate.get("ready_wall_clock_s"),
                "artifact": display_path,
            }
            self._update_status()
            self._append_event(
                "deterministic_fallback_ready",
                decision=result.get("decision"),
                decision_basis=result.get("decision_basis"),
                ready_wall_clock_s=candidate.get("ready_wall_clock_s"),
                artifact=display_path,
            )

    def publish_route(
        self,
        route_result: Mapping[str, Any],
        *,
        artifact_paths: Mapping[str, str],
    ) -> None:
        with self._lock:
            route_id = str(route_result.get("route_id") or "unknown")
            display_paths = {
                name: _display_path(path, relative_to=self.root.parent)
                for name, path in artifact_paths.items()
            }
            analysis_result = _mapping(route_result.get("analysis_result"))
            route_state = dict(self._status.get("routes", {}).get(route_id) or {})
            route_state.update(
                {
                    "status": route_result.get("execution_status"),
                    "l1_usable": route_result.get("l1_usable"),
                    "decision": analysis_result.get("decision"),
                    "error": route_result.get("error"),
                    "completed_at_utc": self._utc_now(),
                    "result_artifact": display_paths.get("result_json"),
                    "trace_artifact": display_paths.get("trace_json"),
                }
            )
            self._status.setdefault("routes", {})[route_id] = route_state
            self._status["completed_routes"] = sum(
                route.get("status") != "pending"
                for route in self._status.get("routes", {}).values()
            )
            self._update_status()
            self._append_event(
                "route_completed",
                route_id=route_id,
                model=route_result.get("model"),
                execution_status=route_result.get("execution_status"),
                l1_usable=route_result.get("l1_usable"),
                decision=analysis_result.get("decision"),
                result_artifact=display_paths.get("result_json"),
                trace_artifact=display_paths.get("trace_json"),
            )

    def complete(
        self,
        *,
        final_artifacts: Mapping[str, str | None],
    ) -> None:
        with self._lock:
            if self._status.get("fallback", {}).get("status") == "pending":
                self._status["fallback"] = {"status": "not_published"}
            if self._status.get("l0", {}).get("status") == "pending":
                self._status["l0"] = {"status": "not_published"}
            self._status.update(
                {
                    "status": "completed",
                    "completed_at_utc": self._utc_now(),
                    "final_result_artifact": final_artifacts.get("result_json"),
                    "final_artifacts": _jsonable(final_artifacts),
                }
            )
            self._update_status()
            self._append_event(
                "run_completed",
                completed_routes=self._status.get("completed_routes"),
                total_routes=self._status.get("total_routes"),
                artifact=final_artifacts.get("result_json"),
            )

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            error = f"{type(exc).__name__}: {exc}"
            self._status.update(
                {
                    "status": "failed",
                    "failed_at_utc": self._utc_now(),
                    "error": error,
                }
            )
            self._update_status()
            self._append_event("run_failed", error=error)

    def _update_status(self) -> None:
        self._status["updated_at_utc"] = self._utc_now()
        self._status["elapsed_s"] = round(
            self._clock.monotonic() - self._started_monotonic,
            3,
        )
        write_json_atomic(self.status_path, self._status)

    def _append_event(self, event: str, **fields: Any) -> None:
        self._sequence += 1
        payload = {
            "schema_version": LIVE_EVENT_SCHEMA_VERSION,
            "sequence": self._sequence,
            "event": event,
            "timestamp_utc": self._utc_now(),
            "elapsed_s": round(
                self._clock.monotonic() - self._started_monotonic,
                3,
            ),
            **_jsonable(fields),
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(
            self.events_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _utc_now(self) -> str:
        return self._clock.now_utc().isoformat()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _display_path(path: str, *, relative_to: Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(relative_to))
    except ValueError:
        return str(candidate)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
