"""Versioned product-CLI payloads owned by the harness adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

RESTART_AGENT_CONFIG_SCHEMA = "restart_agent_config.v1"
ROUTE_ARTIFACT_SCHEMA = "restart_agent_route_artifacts.v1"


def collect_all_config(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": RESTART_AGENT_CONFIG_SCHEMA,
        "config_id": "restart-agent-eval-panel",
        "config_version": 1,
        "routing": {"mode": "collect_all", "max_parallel_models": len(routes)},
        "model_routes": [dict(route) for route in routes],
    }


def route_artifact_manifest(routes: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": ROUTE_ARTIFACT_SCHEMA,
        "routes": {route_id: dict(paths) for route_id, paths in routes.items()},
    }
