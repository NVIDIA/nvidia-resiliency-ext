"""Versioned adapter for restart-agent product trace envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SINGLE_TRACE_SCHEMA = "restart_agent_cli_trace.v1"
COLLECT_ALL_TRACE_SCHEMA = "restart_agent_cli_collect_all_trace.v1"
SUPPORTED_TRACE_SCHEMAS = frozenset({SINGLE_TRACE_SCHEMA, COLLECT_ALL_TRACE_SCHEMA})


@dataclass(frozen=True)
class ProductTrace:
    schema_version: str
    request: Mapping[str, Any]
    analyzer_trace: Mapping[str, Any]
    l0_bundle: Mapping[str, Any] | None
    analysis_result: Mapping[str, Any] | None = None
    collect_all_result: Mapping[str, Any] | None = None
    collect_all_context: Mapping[str, Any] | None = None

    @property
    def is_collect_all(self) -> bool:
        return self.schema_version == COLLECT_ALL_TRACE_SCHEMA

    @classmethod
    def from_payload(cls, value: Any) -> "ProductTrace":
        if not isinstance(value, Mapping):
            raise TypeError("restart-agent trace must be a JSON object")
        schema_version = str(value.get("schema_version") or "")
        if schema_version not in SUPPORTED_TRACE_SCHEMAS:
            raise ValueError(
                f"unsupported restart-agent trace schema: {schema_version or '<missing>'}"
            )
        request = _mapping(value.get("request"), "request")
        analyzer_trace = _mapping(value.get("analyzer_trace"), "analyzer_trace")
        l0_bundle = _optional_mapping(value.get("l0_bundle"))
        analysis_result = _optional_mapping(value.get("analysis_result"))
        collect_all_result = _optional_mapping(value.get("collect_all_result"))
        collect_all_context = _optional_mapping(value.get("collect_all_context"))
        if schema_version == SINGLE_TRACE_SCHEMA and analysis_result is None:
            raise ValueError("single-model trace is missing analysis_result")
        if schema_version == COLLECT_ALL_TRACE_SCHEMA and collect_all_result is None:
            raise ValueError("collect-all trace is missing collect_all_result")
        return cls(
            schema_version=schema_version,
            request=request,
            analyzer_trace=analyzer_trace,
            l0_bundle=l0_bundle,
            analysis_result=analysis_result,
            collect_all_result=collect_all_result,
            collect_all_context=collect_all_context,
        )

    @classmethod
    def read(cls, path: Path) -> "ProductTrace":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_payload(json.load(handle))


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"restart-agent trace field {field_name} must be an object")
    return dict(value)


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("restart-agent trace result fields must be objects")
    return dict(value)


def decision_candidate_result(candidate: Any) -> dict[str, Any]:
    """Return the result payload from a typed decision-candidate envelope."""

    if not isinstance(candidate, Mapping):
        return {}
    result = candidate.get("result")
    return dict(result) if isinstance(result, Mapping) else {}
