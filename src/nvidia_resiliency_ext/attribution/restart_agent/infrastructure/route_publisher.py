# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-directed persistence for independently completed model routes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from ..execution import L0Artifacts
from ..models import ModelAnalysisResult
from ..observability.schemas import CLI_TRACE_SCHEMA_VERSION
from .artifact_io import write_json_atomic

ROUTE_ARTIFACT_MANIFEST_SCHEMA_VERSION = "restart_agent_route_artifacts.v1"


@dataclass(frozen=True)
class RouteArtifactPaths:
    result_json: Path
    trace_json: Path


def load_route_artifact_manifest(
    path: str | Path,
    *,
    expected_route_ids: Sequence[str],
) -> Mapping[str, RouteArtifactPaths]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("route artifact manifest must be an object")
    if payload.get("schema_version") != ROUTE_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("route artifact manifest schema_version is invalid")
    routes = payload.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("route artifact manifest routes must be an object")
    expected = set(expected_route_ids)
    observed = {str(route_id) for route_id in routes}
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ValueError(
            "route artifact manifest route IDs do not match configuration: "
            f"missing={missing}, unknown={unknown}"
        )
    result: dict[str, RouteArtifactPaths] = {}
    claimed_paths: dict[Path, str] = {}
    for route_id, value in routes.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"route artifact paths for {route_id!r} must be an object")
        route_paths = RouteArtifactPaths(
            result_json=_resolve_output_path(
                value.get("result_json"),
                manifest_path=manifest_path,
                field=f"routes.{route_id}.result_json",
            ),
            trace_json=_resolve_output_path(
                value.get("trace_json"),
                manifest_path=manifest_path,
                field=f"routes.{route_id}.trace_json",
            ),
        )
        for field, output_path in (
            ("result_json", route_paths.result_json),
            ("trace_json", route_paths.trace_json),
        ):
            owner = f"routes.{route_id}.{field}"
            previous_owner = claimed_paths.get(output_path)
            if previous_owner is not None:
                raise ValueError(
                    f"route artifact path {output_path} is shared by "
                    f"{previous_owner} and {owner}"
                )
            claimed_paths[output_path] = owner
        result[str(route_id)] = route_paths
    return result


class RouteArtifactPublisher:
    """Write canonical route result/trace files before the batch completes."""

    def __init__(
        self,
        paths: Mapping[str, RouteArtifactPaths],
        *,
        request: Mapping[str, Any],
        batch_trace_path: str | Path | None,
    ) -> None:
        self._paths = dict(paths)
        self._request = dict(request)
        self._batch_trace_path = str(batch_trace_path) if batch_trace_path else None
        self._l0_artifacts: L0Artifacts | None = None
        self._published: set[str] = set()
        self._lock = Lock()

    def set_l0_artifacts(self, artifacts: L0Artifacts) -> None:
        with self._lock:
            if self._l0_artifacts is not None:
                raise RuntimeError("route publisher L0 artifacts were already set")
            self._l0_artifacts = artifacts

    def publish(
        self,
        route_result: ModelAnalysisResult,
        analyzer_trace: Mapping[str, Any],
    ) -> Mapping[str, str]:
        with self._lock:
            paths = self._paths.get(route_result.route_id)
            if paths is None:
                raise ValueError(
                    f"no artifact paths configured for route {route_result.route_id!r}"
                )
            if route_result.route_id in self._published:
                raise RuntimeError(f"route {route_result.route_id!r} was already published")
            l0_artifacts = self._l0_artifacts
            analysis_result = route_result.analysis_result.to_payload()
            trace_payload = {
                "schema_version": CLI_TRACE_SCHEMA_VERSION,
                "request": self._request,
                "analysis_result": analysis_result,
                "analyzer_trace": dict(analyzer_trace),
                "l0_bundle": (asdict(l0_artifacts.bundle) if l0_artifacts is not None else None),
                "collect_all_context": {
                    "route_id": route_result.route_id,
                    "execution_status": route_result.execution_status,
                    "publication": "route_complete",
                    "batch_trace": self._batch_trace_path,
                },
            }
            # Result is the commit marker: when it appears, the trace is ready too.
            write_json_atomic(paths.trace_json, trace_payload)
            write_json_atomic(paths.result_json, analysis_result)
            self._published.add(route_result.route_id)
            return {
                "result_json": str(paths.result_json),
                "trace_json": str(paths.trace_json),
            }


def _resolve_output_path(
    value: Any,
    *,
    manifest_path: Path,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()
