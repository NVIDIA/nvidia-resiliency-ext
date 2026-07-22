# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned restart-agent configuration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .attempt_records import DEFAULT_MAX_ATTEMPTS_PER_JOB, DEFAULT_MAX_TOTAL_RECORDS
from .l1 import (
    DEFAULT_ADVERTISED_TOOLS,
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    EvidenceExtractor,
    LlmConfig,
    ModelRoute,
)
from .models import normalize_restart_environment_context, normalize_retry_policy

RESTART_AGENT_CONFIG_SCHEMA_VERSION = "restart_agent_config.v1"
RESTART_AGENT_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "config_id",
        "config_version",
        "routing",
        "runtime",
        "restart_environment_context",
        "retry_policy",
        "model_defaults",
        "model_routes",
    }
)
TOOL_ADVERTISEMENT_ORDER = (*DEFAULT_ADVERTISED_TOOLS, "get_evidence_objects")


@dataclass(frozen=True)
class ModelRouteSpec:
    """Validated model-route data without a live provider implementation."""

    route_id: str
    llm_config: LlmConfig
    model: str
    endpoint: str
    credential_ref: str


EvidenceExtractorFactory = Callable[[LlmConfig], EvidenceExtractor]


@dataclass(frozen=True)
class HistoryConfig:
    """Bounded current-process attempt-record configuration."""

    enabled: bool = True
    max_attempts_per_job: int = DEFAULT_MAX_ATTEMPTS_PER_JOB
    max_total_records: int = DEFAULT_MAX_TOTAL_RECORDS


@dataclass(frozen=True)
class RestartAgentConfig:
    """Resolved, credential-free restart-agent configuration."""

    config_id: str
    config_version: int
    routing_mode: str
    max_parallel_models: int
    timeout_seconds: float
    history: HistoryConfig
    restart_environment_context: Mapping[str, bool]
    retry_policy: Mapping[str, Any]
    model_route_specs: tuple[ModelRouteSpec, ...]
    effective_config: Mapping[str, Any]
    config_fingerprint: str

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": RESTART_AGENT_CONFIG_SCHEMA_VERSION,
            "config_id": self.config_id,
            "config_version": self.config_version,
            "config_fingerprint": self.config_fingerprint,
            "routing_mode": self.routing_mode,
            "max_parallel_models": self.max_parallel_models,
            "timeout_seconds": self.timeout_seconds,
            "history": {
                "enabled": self.history.enabled,
                "max_attempts_per_job": self.history.max_attempts_per_job,
                "max_total_records": self.history.max_total_records,
            },
            "effective_config": self.effective_config,
        }


def load_restart_agent_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> RestartAgentConfig:
    """Read and parse restart-agent configuration from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_restart_agent_config(payload, environ=environ)


def parse_restart_agent_config(
    payload: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> RestartAgentConfig:
    """Validate already-loaded configuration without performing file I/O."""

    if not isinstance(payload, Mapping):
        raise TypeError("restart-agent config must be an object")
    if payload.get("schema_version") != RESTART_AGENT_CONFIG_SCHEMA_VERSION:
        raise ValueError("restart-agent config schema_version is invalid")
    unknown_fields = set(payload).difference(RESTART_AGENT_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError(
            "restart-agent config has unsupported fields: " + ", ".join(sorted(unknown_fields))
        )

    config_id = _required_string(payload, "config_id", "restart-agent config")
    config_version = payload.get("config_version")
    if not isinstance(config_version, int) or isinstance(config_version, bool):
        raise TypeError("restart-agent config config_version must be an integer")
    if config_version < 1:
        raise ValueError("restart-agent config config_version must be at least 1")

    routing = _optional_mapping(payload, "routing")
    routing_mode = str(routing.get("mode") or "collect_all")
    if routing_mode != "collect_all":
        raise ValueError("restart-agent config routing.mode must be 'collect_all'")
    restart_environment_context = normalize_restart_environment_context(
        _optional_mapping(payload, "restart_environment_context")
    )
    retry_policy = normalize_retry_policy(_optional_mapping(payload, "retry_policy"))
    history = _parse_history_config(_optional_mapping(payload, "runtime"))

    model_defaults = _optional_mapping(payload, "model_defaults")
    raw_routes = payload.get("model_routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise ValueError("restart-agent config model_routes must be a non-empty array")

    environment = os.environ if environ is None else environ
    model_route_specs: list[ModelRouteSpec] = []
    effective_routes: list[dict[str, Any]] = []
    route_ids: set[str] = set()
    for raw_route in raw_routes:
        if not isinstance(raw_route, Mapping):
            raise TypeError("each model route must be an object")
        route_spec, effective_route = _resolve_route(model_defaults, raw_route, environment)
        if route_spec.route_id in route_ids:
            raise ValueError(f"duplicate model route_id: {route_spec.route_id!r}")
        route_ids.add(route_spec.route_id)
        model_route_specs.append(route_spec)
        effective_routes.append(effective_route)

    max_parallel_models = routing.get("max_parallel_models", len(model_route_specs))
    if not isinstance(max_parallel_models, int) or isinstance(max_parallel_models, bool):
        raise TypeError("routing.max_parallel_models must be an integer")
    if max_parallel_models < 1:
        raise ValueError("routing.max_parallel_models must be at least 1")
    max_parallel_models = min(max_parallel_models, len(model_route_specs))
    timeout_seconds = routing.get("timeout_seconds", DEFAULT_ANALYSIS_TIMEOUT_SECONDS)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("routing.timeout_seconds must be a number")
    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0:
        raise ValueError("routing.timeout_seconds must be greater than zero")

    effective_config = {
        "schema_version": RESTART_AGENT_CONFIG_SCHEMA_VERSION,
        "routing": {
            "mode": routing_mode,
            "max_parallel_models": max_parallel_models,
            "timeout_seconds": timeout_seconds,
        },
        "runtime": {
            "history": {
                "enabled": history.enabled,
                "max_attempts_per_job": history.max_attempts_per_job,
                "max_total_records": history.max_total_records,
            }
        },
        "restart_environment_context": dict(restart_environment_context),
        "retry_policy": dict(retry_policy),
        "model_routes": effective_routes,
    }
    return RestartAgentConfig(
        config_id=config_id,
        config_version=config_version,
        routing_mode=routing_mode,
        max_parallel_models=max_parallel_models,
        timeout_seconds=timeout_seconds,
        history=history,
        restart_environment_context=restart_environment_context,
        retry_policy=retry_policy,
        model_route_specs=tuple(model_route_specs),
        effective_config=effective_config,
        config_fingerprint=_stable_hash(effective_config),
    )


def _parse_history_config(runtime: Mapping[str, Any]) -> HistoryConfig:
    unknown_runtime = sorted(set(runtime).difference({"history"}))
    if unknown_runtime:
        raise ValueError("runtime has unsupported fields: " + ", ".join(unknown_runtime))
    history = _optional_mapping(runtime, "history")
    unknown_history = sorted(
        set(history).difference({"enabled", "max_attempts_per_job", "max_total_records"})
    )
    if unknown_history:
        raise ValueError("runtime.history has unsupported fields: " + ", ".join(unknown_history))
    enabled = history.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("runtime.history.enabled must be boolean")
    max_attempts_per_job = history.get("max_attempts_per_job", DEFAULT_MAX_ATTEMPTS_PER_JOB)
    max_total_records = history.get("max_total_records", DEFAULT_MAX_TOTAL_RECORDS)
    for field_name, value in (
        ("max_attempts_per_job", max_attempts_per_job),
        ("max_total_records", max_total_records),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"runtime.history.{field_name} must be an integer")
        if value < 1:
            raise ValueError(f"runtime.history.{field_name} must be greater than zero")
    return HistoryConfig(
        enabled=enabled,
        max_attempts_per_job=max_attempts_per_job,
        max_total_records=max_total_records,
    )


def _resolve_route(
    defaults: Mapping[str, Any],
    route: Mapping[str, Any],
    environment: Mapping[str, str],
) -> tuple[ModelRouteSpec, dict[str, Any]]:
    route_id = _required_string(route, "route_id", "model route")
    model = _setting(route, defaults, "model")
    base_url = _setting(route, defaults, "base_url")
    credential_ref = str(_setting(route, defaults, "credential_ref", default="LLM_API_KEY_FILE"))
    api_key_file = environment.get(credential_ref)
    if not api_key_file:
        raise ValueError(f"model route {route_id!r} credential_ref {credential_ref!r} is not set")

    request = _merged_group(defaults, route, "request")
    tools = _merged_group(defaults, route, "tools")
    reasoning = _merged_group(defaults, route, "reasoning")
    reliability = _merged_group(defaults, route, "reliability")

    tools_enabled = tools.get("enabled")
    if tools_enabled is not None and not isinstance(tools_enabled, bool):
        raise TypeError(f"model route {route_id!r} tools.enabled must be boolean")
    advertised_tools = _advertised_tools(route_id, tools.get("advertisement"))

    base_config = LlmConfig(api_key=None, api_key_file=api_key_file)
    overrides = {
        "base_url": _optional_string(base_url, f"model route {route_id!r} base_url"),
        "model": _optional_string(model, f"model route {route_id!r} model"),
        "timeout_seconds": _optional_number(request, "timeout_seconds"),
        "max_output_tokens": _optional_int(request, "max_output_tokens"),
        "context_window_tokens": _optional_int(request, "context_window_tokens"),
        "context_safety_tokens": _optional_int(request, "context_safety_tokens"),
        "temperature": _optional_number(request, "temperature"),
        "top_p": _optional_number(request, "top_p"),
        "tools_enabled": tools_enabled,
        "advertised_tools": (tuple(advertised_tools) if advertised_tools is not None else None),
        "max_tool_rounds": _optional_int(tools, "max_rounds"),
        "thinking_mode": _optional_group_string(reasoning, "thinking_mode"),
        "reasoning_effort": _optional_group_string(reasoning, "reasoning_effort"),
        "max_retries": _optional_int(reliability, "max_retries"),
        "retry_backoff_seconds": _optional_number(reliability, "retry_backoff_seconds"),
    }
    config = replace(
        base_config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    _validate_config(route_id, config)
    model_route = ModelRouteSpec(
        route_id=route_id,
        llm_config=config,
        model=config.model,
        endpoint=config.base_url,
        credential_ref=credential_ref,
    )
    effective_route = {
        "route_id": route_id,
        "model": config.model,
        "base_url": config.base_url,
        "credential_ref": credential_ref,
        "request": {
            "timeout_seconds": config.timeout_seconds,
            "max_output_tokens": config.max_output_tokens,
            "configured_context_window_tokens": config.context_window_tokens,
            "effective_context_window_tokens": config.resolved_context_window_tokens(),
            "context_safety_tokens": config.context_safety_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "temperature_sent": config.request_temperature() is not None,
            "top_p_sent": config.request_top_p() is not None,
        },
        "tools": {
            "enabled": config.tools_enabled,
            "advertisement": {
                name: name in config.advertised_tools for name in TOOL_ADVERTISEMENT_ORDER
            },
            "effective_advertised": (list(config.advertised_tools) if config.tools_enabled else []),
            "max_rounds": config.max_tool_rounds,
        },
        "reasoning": {
            "thinking_mode": config.thinking_mode,
            "thinking_disabled": config.disable_thinking(),
            "reasoning_effort": config.reasoning_effort,
        },
        "reliability": {
            "max_retries": config.max_retries,
            "retry_backoff_seconds": config.retry_backoff_seconds,
        },
    }
    return model_route, effective_route


def build_model_routes(
    config: RestartAgentConfig,
    evidence_extractor_factory: EvidenceExtractorFactory,
) -> tuple[ModelRoute, ...]:
    """Compose validated route data with a caller-selected L1 implementation."""

    return tuple(
        ModelRoute(
            route_id=spec.route_id,
            evidence_extractor=evidence_extractor_factory(spec.llm_config),
            model=spec.model,
            endpoint=spec.endpoint,
            credential_ref=spec.credential_ref,
        )
        for spec in config.model_route_specs
    )


def _advertised_tools(
    route_id: str,
    advertisement: Any,
) -> tuple[str, ...] | None:
    if advertisement is None:
        return None
    if not isinstance(advertisement, Mapping):
        raise TypeError(f"model route {route_id!r} tools.advertisement must be an object")
    unknown = set(advertisement).difference(TOOL_ADVERTISEMENT_ORDER)
    if unknown:
        raise ValueError(
            f"model route {route_id!r} tools.advertisement has unknown tools: "
            + ", ".join(sorted(unknown))
        )
    invalid = [name for name, enabled in advertisement.items() if not isinstance(enabled, bool)]
    if invalid:
        raise TypeError(
            f"model route {route_id!r} tools.advertisement values must be boolean: "
            + ", ".join(sorted(invalid))
        )
    resolved = {name: name in DEFAULT_ADVERTISED_TOOLS for name in TOOL_ADVERTISEMENT_ORDER}
    resolved.update(advertisement)
    return tuple(name for name in TOOL_ADVERTISEMENT_ORDER if resolved[name])


def _validate_config(route_id: str, config: LlmConfig) -> None:
    owner = f"model route {route_id!r}"
    if config.timeout_seconds <= 0:
        raise ValueError(f"{owner} request.timeout_seconds must be positive")
    if config.max_output_tokens <= 0:
        raise ValueError(f"{owner} request.max_output_tokens must be positive")
    if config.context_window_tokens is not None and config.context_window_tokens <= 0:
        raise ValueError(f"{owner} request.context_window_tokens must be positive")
    if config.context_safety_tokens < 0:
        raise ValueError(f"{owner} request.context_safety_tokens must not be negative")
    if config.temperature is not None and not 0 <= config.temperature <= 2:
        raise ValueError(f"{owner} request.temperature must be between 0 and 2")
    if config.top_p is not None and not 0 <= config.top_p <= 1:
        raise ValueError(f"{owner} request.top_p must be between 0 and 1")
    if config.max_tool_rounds < 0:
        raise ValueError(f"{owner} tools.max_rounds must not be negative")
    if config.max_retries < 0:
        raise ValueError(f"{owner} reliability.max_retries must not be negative")
    if config.retry_backoff_seconds < 0:
        raise ValueError(f"{owner} reliability.retry_backoff_seconds must not be negative")


def _setting(
    route: Mapping[str, Any],
    defaults: Mapping[str, Any],
    field: str,
    *,
    default: Any = None,
) -> Any:
    if field in route:
        return route[field]
    return defaults.get(field, default)


def _merged_group(
    defaults: Mapping[str, Any], route: Mapping[str, Any], field: str
) -> dict[str, Any]:
    merged = dict(_optional_mapping(defaults, field))
    merged.update(_optional_mapping(route, field))
    return merged


def _optional_mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    item = value.get(field)
    if item is None:
        return {}
    if not isinstance(item, Mapping):
        raise TypeError(f"{field} must be an object")
    return item


def _required_string(value: Mapping[str, Any], field: str, owner: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{owner} requires non-empty {field}")
    return item.strip()


def _optional_string(value: Any, owner: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{owner} must be a non-empty string")
    return value.strip()


def _optional_group_string(value: Mapping[str, Any], field: str) -> str | None:
    if field not in value:
        return None
    return _optional_string(value[field], field)


def _optional_int(value: Mapping[str, Any], field: str) -> int | None:
    item = value.get(field)
    if item is None:
        return None
    if not isinstance(item, int) or isinstance(item, bool):
        raise TypeError(f"{field} must be an integer")
    return item


def _optional_number(value: Mapping[str, Any], field: str) -> float | None:
    item = value.get(field)
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise TypeError(f"{field} must be a number")
    return float(item)


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
