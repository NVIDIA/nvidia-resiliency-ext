# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-source contracts for L1 read-only evidence tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from .contracts import EvidenceTools

TOOL_RESULT_SCHEMA_VERSION = "restart_agent_tool_result.v1"
OVERVIEW_HEAD_LINES = 40
OVERVIEW_TAIL_LINES = 80
OVERVIEW_MAX_CHARS = 12_000
GREP_PATTERN_MAX_CHARS = 4_096
GREP_MAX_MATCHES = 50
GREP_MAX_MATCHES_HARD_LIMIT = 200
GREP_RESULT_MODES = ("compact", "raw")
READ_WINDOW_DEFAULT_BEFORE = 20
READ_WINDOW_DEFAULT_AFTER = 80
READ_WINDOW_SIDE_MAX_LINES = 120
READ_WINDOW_MAX_LINES = 241
READ_WINDOW_MAX_CHARS = 50_000
TOOL_LINE_MAX_CHARS = 2_000
EVIDENCE_OBJECTS_SCHEMA_VERSION = "restart_agent_evidence_objects.v1"
EVIDENCE_OBJECTS_MAX_REFS = 8
EVIDENCE_OBJECTS_MAX_CHARS = 50_000
EVIDENCE_OBJECT_REF_MAX_CHARS = 128
EVIDENCE_OBJECTS_METADATA_RESERVE_CHARS = 2_048


class L1ToolErrorCode(str, Enum):
    """Closed failure codes returned by every L1 evidence tool."""

    MALFORMED_ARGUMENTS_JSON = "malformed_arguments_json"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_REGEX = "invalid_regex"
    TOOL_NOT_ADVERTISED = "tool_not_advertised"
    TOOL_NOT_IMPLEMENTED = "tool_not_implemented"
    SOURCE_UNAVAILABLE = "source_unavailable"
    LINE_OUT_OF_RANGE = "line_out_of_range"
    INTERNAL_TOOL_ERROR = "internal_tool_error"


class L1ToolContractError(ValueError):
    """Typed validation or execution failure safe to return to the model."""

    def __init__(
        self,
        code: L1ToolErrorCode,
        message: str,
        *,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


ArgumentValidator = Callable[[Mapping[str, Any], EvidenceTools], dict[str, Any]]
ToolExecutor = Callable[[EvidenceTools, Mapping[str, Any]], dict[str, Any]]
LimitsBuilder = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class L1ToolContract:
    """One advertised schema and its matching executable validation contract."""

    name: str
    description: str
    argument_schema: Mapping[str, Any]
    result_required_fields: tuple[str, ...]
    validate_arguments: ArgumentValidator
    execute: ToolExecutor
    limits: LimitsBuilder

    def advertised_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.argument_schema),
            },
        }


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Parse one model tool argument object without coercing malformed input."""

    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "Tool arguments must be a JSON object.",
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise L1ToolContractError(
            L1ToolErrorCode.MALFORMED_ARGUMENTS_JSON,
            "Tool arguments are not valid JSON.",
        ) from exc
    if not isinstance(parsed, dict):
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "Tool arguments must decode to a JSON object.",
        )
    return parsed


def execute_tool_request(
    tools: EvidenceTools,
    *,
    name: str,
    raw_arguments: Any,
    advertised_tools: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Validate and execute one request using the canonical tool registry."""

    contract = L1_TOOL_CONTRACTS.get(name)
    parsed_arguments: dict[str, Any] = {}
    if name not in advertised_tools:
        return (
            tool_error_result(
                name,
                L1ToolErrorCode.TOOL_NOT_ADVERTISED,
                "The requested tool was not advertised for this route.",
            ),
            parsed_arguments,
            True,
        )
    if contract is None:
        return (
            tool_error_result(
                name,
                L1ToolErrorCode.TOOL_NOT_IMPLEMENTED,
                "The requested tool is not implemented.",
            ),
            parsed_arguments,
            True,
        )

    try:
        parsed_arguments = parse_tool_arguments(raw_arguments)
        normalized = contract.validate_arguments(parsed_arguments, tools)
        data = contract.execute(tools, normalized)
        _validate_success_result(contract, data)
        return tool_success_result(contract, data, normalized), normalized, False
    except L1ToolContractError as exc:
        return (
            tool_error_result(name, exc.code, str(exc), field=exc.field),
            parsed_arguments,
            False,
        )
    except OSError:
        return (
            tool_error_result(
                name,
                L1ToolErrorCode.SOURCE_UNAVAILABLE,
                "The immutable source snapshot is unavailable.",
            ),
            parsed_arguments,
            False,
        )
    except Exception:
        return (
            tool_error_result(
                name,
                L1ToolErrorCode.INTERNAL_TOOL_ERROR,
                "The tool failed while reading the immutable evidence source.",
            ),
            parsed_arguments,
            False,
        )


def tool_success_result(
    contract: L1ToolContract,
    data: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": contract.name,
        "status": "ok",
        "data": dict(data),
        "error": None,
        "truncated": bool(data.get("samples_truncated", data.get("truncated"))),
        "limits": contract.limits(arguments),
    }


def tool_error_result(
    name: str,
    code: L1ToolErrorCode,
    message: str,
    *,
    field: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": name,
        "status": "error",
        "data": None,
        "error": {
            "code": code.value,
            "field": field,
            "message": message,
        },
        "truncated": False,
        "limits": {},
    }


def advertised_tool_schemas(names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [L1_TOOL_CONTRACTS[name].advertised_schema() for name in names]


def _validate_success_result(contract: L1ToolContract, data: Any) -> None:
    if not isinstance(data, Mapping):
        raise RuntimeError("tool implementation returned a non-object result")
    missing = [field for field in contract.result_required_fields if field not in data]
    if missing:
        raise RuntimeError("tool implementation omitted required result fields")


def _reject_extra_arguments(arguments: Mapping[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(arguments).difference(allowed))
    if extra:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "Tool arguments contain unsupported fields.",
            field=extra[0],
        )


def _required_string(
    arguments: Mapping[str, Any],
    field: str,
    *,
    max_chars: int,
) -> str:
    value = arguments.get(field)
    if not isinstance(value, str) or not value:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            f"{field} must be a non-empty string.",
            field=field,
        )
    if len(value) > max_chars:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            f"{field} exceeds the supported length.",
            field=field,
        )
    return value


def _optional_bool(arguments: Mapping[str, Any], field: str, default: bool) -> bool:
    value = arguments.get(field, default)
    if not isinstance(value, bool):
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            f"{field} must be a boolean.",
            field=field,
        )
    return value


def _integer(
    arguments: Mapping[str, Any],
    field: str,
    *,
    default: int | None = None,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = arguments.get(field, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            f"{field} must be an integer.",
            field=field,
        )
    if value < minimum or (maximum is not None and value > maximum):
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            f"{field} is outside the supported range.",
            field=field,
        )
    return value


def _validate_overview(
    arguments: Mapping[str, Any],
    _tools: EvidenceTools,
) -> dict[str, Any]:
    _reject_extra_arguments(arguments, set())
    return {}


def _validate_grep_log(
    arguments: Mapping[str, Any],
    _tools: EvidenceTools,
) -> dict[str, Any]:
    _reject_extra_arguments(
        arguments,
        {"pattern", "ignore_case", "max_matches", "result_mode"},
    )
    pattern = _required_string(arguments, "pattern", max_chars=GREP_PATTERN_MAX_CHARS)
    ignore_case = _optional_bool(arguments, "ignore_case", True)
    max_matches = _integer(
        arguments,
        "max_matches",
        default=GREP_MAX_MATCHES,
        minimum=0,
        maximum=GREP_MAX_MATCHES_HARD_LIMIT,
    )
    result_mode = arguments.get("result_mode", "compact")
    if not isinstance(result_mode, str) or result_mode not in GREP_RESULT_MODES:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "result_mode must be compact or raw.",
            field="result_mode",
        )
    try:
        re.compile(pattern, re.I if ignore_case else 0)
    except re.error as exc:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_REGEX,
            "pattern is not a valid Python regular expression.",
            field="pattern",
        ) from exc
    return {
        "pattern": pattern,
        "ignore_case": ignore_case,
        "max_matches": max_matches,
        "result_mode": result_mode,
    }


def _validate_read_window(
    arguments: Mapping[str, Any],
    tools: EvidenceTools,
) -> dict[str, Any]:
    _reject_extra_arguments(arguments, {"center_line", "before", "after"})
    center_line = _integer(arguments, "center_line", minimum=1)
    before = _integer(
        arguments,
        "before",
        default=READ_WINDOW_DEFAULT_BEFORE,
        minimum=0,
        maximum=READ_WINDOW_SIDE_MAX_LINES,
    )
    after = _integer(
        arguments,
        "after",
        default=READ_WINDOW_DEFAULT_AFTER,
        minimum=0,
        maximum=READ_WINDOW_SIDE_MAX_LINES,
    )
    if before + after + 1 > READ_WINDOW_MAX_LINES:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "before and after request more than the supported total lines.",
        )
    if center_line > tools.line_count:
        raise L1ToolContractError(
            L1ToolErrorCode.LINE_OUT_OF_RANGE,
            "center_line is outside the immutable source snapshot.",
            field="center_line",
        )
    return {"center_line": center_line, "before": before, "after": after}


def _validate_evidence_refs(
    arguments: Mapping[str, Any],
    _tools: EvidenceTools,
) -> dict[str, Any]:
    _reject_extra_arguments(arguments, {"refs"})
    refs = arguments.get("refs")
    if not isinstance(refs, list) or not refs:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "refs must be a non-empty array of strings.",
            field="refs",
        )
    if len(refs) > EVIDENCE_OBJECTS_MAX_REFS:
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "refs contains more than the supported number of items.",
            field="refs",
        )
    if not all(
        isinstance(ref, str) and 0 < len(ref) <= EVIDENCE_OBJECT_REF_MAX_CHARS for ref in refs
    ):
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "Each refs item must be a bounded non-empty string.",
            field="refs",
        )
    if len(set(refs)) != len(refs):
        raise L1ToolContractError(
            L1ToolErrorCode.INVALID_ARGUMENTS,
            "refs must not contain duplicate items.",
            field="refs",
        )
    return {"refs": list(refs)}


def _overview(tools: EvidenceTools, _arguments: Mapping[str, Any]) -> dict[str, Any]:
    return tools.overview()


def _grep_log(tools: EvidenceTools, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return tools.grep_log(
        arguments["pattern"],
        ignore_case=arguments["ignore_case"],
        max_matches=arguments["max_matches"],
        result_mode=arguments["result_mode"],
    )


def _read_window(tools: EvidenceTools, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return tools.read_window(
        arguments["center_line"],
        before=arguments["before"],
        after=arguments["after"],
    )


def _get_evidence_objects(
    tools: EvidenceTools,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    return tools.get_evidence_objects(arguments["refs"])


def _fixed_limits(**values: Any) -> LimitsBuilder:
    return lambda _arguments: dict(values)


L1_TOOL_CONTRACTS: dict[str, L1ToolContract] = {
    "overview": L1ToolContract(
        name="overview",
        description="Return a compact summary, head/tail preview, and L0 findings.",
        argument_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        result_required_fields=(
            "line_count",
            "byte_size",
            "head",
            "tail",
            "deterministic_summary",
            "truncated",
        ),
        validate_arguments=_validate_overview,
        execute=_overview,
        limits=_fixed_limits(
            head_lines=OVERVIEW_HEAD_LINES,
            tail_lines=OVERVIEW_TAIL_LINES,
            max_chars=OVERVIEW_MAX_CHARS,
            line_max_chars=TOOL_LINE_MAX_CHARS,
        ),
    ),
    "grep_log": L1ToolContract(
        name="grep_log",
        description=(
            "Search the source log with a Python regular expression. By default, "
            "matches reuse L0 normalized occurrence groups and distributed-incident "
            "boundaries, with occurrence and rank counts while distinct evidence is preserved. "
            "Use raw mode when individual matching lines are required."
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": GREP_PATTERN_MAX_CHARS,
                },
                "ignore_case": {"type": "boolean", "default": True},
                "max_matches": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": GREP_MAX_MATCHES_HARD_LIMIT,
                    "default": GREP_MAX_MATCHES,
                },
                "result_mode": {
                    "type": "string",
                    "enum": list(GREP_RESULT_MODES),
                    "default": "compact",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        result_required_fields=(
            "pattern",
            "result_mode",
            "matches",
            "total_raw_matches",
            "total_match_groups",
            "collapsed_matches",
            "scan_complete",
            "samples_truncated",
            "initial_view_overlap",
        ),
        validate_arguments=_validate_grep_log,
        execute=_grep_log,
        limits=lambda arguments: {
            "max_matches": arguments["max_matches"],
            "max_matches_hard_limit": GREP_MAX_MATCHES_HARD_LIMIT,
            "line_max_chars": TOOL_LINE_MAX_CHARS,
        },
    ),
    "read_window": L1ToolContract(
        name="read_window",
        description="Read original log lines around one center line.",
        argument_schema={
            "type": "object",
            "properties": {
                "center_line": {"type": "integer", "minimum": 1},
                "before": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": READ_WINDOW_SIDE_MAX_LINES,
                    "default": READ_WINDOW_DEFAULT_BEFORE,
                },
                "after": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": READ_WINDOW_SIDE_MAX_LINES,
                    "default": READ_WINDOW_DEFAULT_AFTER,
                },
            },
            "required": ["center_line"],
            "additionalProperties": False,
        },
        result_required_fields=("start_line", "end_line", "lines", "truncated"),
        validate_arguments=_validate_read_window,
        execute=_read_window,
        limits=lambda arguments: {
            "before": arguments["before"],
            "after": arguments["after"],
            "max_lines": READ_WINDOW_MAX_LINES,
            "max_chars": READ_WINDOW_MAX_CHARS,
            "line_max_chars": TOOL_LINE_MAX_CHARS,
        },
    ),
    "get_evidence_objects": L1ToolContract(
        name="get_evidence_objects",
        description=(
            "Expand object IDs listed in "
            "decision_evidence_view.selected_evidence_references into their structured "
            "current-log payloads. Use this before grep_log when the needed evidence may "
            "already be represented by those IDs. Pass only IDs from the *_ids fields; "
            "do not convert source_lines into line-* IDs. This tool does not read "
            "external files or search arbitrary log text."
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "refs": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": EVIDENCE_OBJECT_REF_MAX_CHARS,
                    },
                    "minItems": 1,
                    "maxItems": EVIDENCE_OBJECTS_MAX_REFS,
                    "uniqueItems": True,
                }
            },
            "required": ["refs"],
            "additionalProperties": False,
        },
        result_required_fields=(
            "schema_version",
            "requested_refs",
            "objects",
            "missing_refs",
            "invalid_refs",
            "omitted_refs",
            "limits",
            "truncated",
        ),
        validate_arguments=_validate_evidence_refs,
        execute=_get_evidence_objects,
        limits=_fixed_limits(
            max_refs=EVIDENCE_OBJECTS_MAX_REFS,
            max_chars=EVIDENCE_OBJECTS_MAX_CHARS,
        ),
    ),
}

DEFAULT_ADVERTISED_TOOLS = ("grep_log", "read_window", "get_evidence_objects")
IMPLEMENTED_TOOL_NAMES = frozenset(L1_TOOL_CONTRACTS)
