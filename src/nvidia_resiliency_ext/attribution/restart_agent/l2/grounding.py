# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers for determining which source evidence was visible to L1."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..l1.contracts import L1EvidenceResult
from ..models import L0ModelFacingView


def model_visible_line_numbers(
    model_view: L0ModelFacingView,
    result: L1EvidenceResult,
) -> set[int]:
    """Return log-line references actually exposed to the model."""

    visible: set[int] = set()
    saw_model_payload = False
    for event in result.transcript_events:
        event_type = event.get("event_type")
        if event_type == "bundle_snapshot":
            model_payload = event.get("model_visible_payload")
            if isinstance(model_payload, Mapping):
                saw_model_payload = True
                visible.update(_line_references(model_payload))
        elif event_type == "tool_result":
            tool_data = _successful_tool_data(event)
            if tool_data is not None:
                visible.update(_line_references(tool_data))
    if not saw_model_payload:
        visible.update(_line_references(model_view.prompt_payload()))
    return visible


def model_visible_line_texts(
    model_view: L0ModelFacingView,
    result: L1EvidenceResult,
) -> dict[int, set[str]]:
    """Return log-line text exactly as rendered to the model."""

    visible: dict[int, set[str]] = {}
    saw_model_payload = False
    for event in result.transcript_events:
        event_type = event.get("event_type")
        if event_type == "bundle_snapshot":
            model_payload = event.get("model_visible_payload")
            if isinstance(model_payload, Mapping):
                saw_model_payload = True
                _collect_line_texts(model_payload, visible)
        elif event_type == "tool_result":
            tool_data = _successful_tool_data(event)
            if tool_data is not None:
                _collect_line_texts(tool_data, visible)
    if not saw_model_payload:
        _collect_line_texts(model_view.prompt_payload(), visible)
    return visible


def model_visible_value_line_numbers(
    model_view: L0ModelFacingView,
    result: L1EvidenceResult,
    value: str,
) -> set[int]:
    """Return source lines where an exact model-visible value was rendered."""

    visible: set[int] = set()
    saw_model_payload = False
    for event in result.transcript_events:
        event_type = event.get("event_type")
        if event_type == "bundle_snapshot":
            model_payload = event.get("model_visible_payload")
            if isinstance(model_payload, Mapping):
                saw_model_payload = True
                _collect_value_line_numbers(model_payload, value, visible)
        elif event_type == "tool_result":
            tool_data = _successful_tool_data(event)
            if tool_data is not None:
                _collect_value_line_numbers(tool_data, value, visible)
    if not saw_model_payload:
        _collect_value_line_numbers(model_view.prompt_payload(), value, visible)
    return visible


def text_contains_exact_value(text: str, value: str) -> bool:
    """Return whether a rendered value appears as a complete token."""

    return (
        re.search(
            rf"(?<![A-Za-z0-9._/-]){re.escape(value)}(?![A-Za-z0-9._/-])",
            text,
        )
        is not None
    )


def _successful_tool_data(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    result = event.get("result")
    if not isinstance(result, Mapping) or result.get("status") != "ok":
        return None
    data = result.get("data")
    return data if isinstance(data, Mapping) else None


def _collect_line_texts(value: Any, result: dict[int, set[str]]) -> None:
    if isinstance(value, Mapping):
        for line_field, line in value.items():
            if not _is_line_field(line_field, line):
                continue
            for text_field in _paired_text_fields(str(line_field)):
                text = value.get(text_field)
                if isinstance(text, str) and text:
                    result.setdefault(line, set()).add(text)
        for item in value.values():
            _collect_line_texts(item, result)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_line_texts(item, result)


def _collect_value_line_numbers(value: Any, candidate: str, result: set[int]) -> None:
    if isinstance(value, Mapping):
        local_lines = {int(item) for field, item in value.items() if _is_line_field(field, item)}
        if local_lines and any(
            isinstance(item, str)
            and (item == candidate or text_contains_exact_value(item, candidate))
            for item in value.values()
        ):
            result.update(local_lines)
        for item in value.values():
            _collect_value_line_numbers(item, candidate, result)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_value_line_numbers(item, candidate, result)


def _is_line_field(field: Any, value: Any) -> bool:
    return (
        isinstance(field, str)
        and (field == "line" or field.endswith("_line"))
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _paired_text_fields(line_field: str) -> tuple[str, ...]:
    if line_field == "line":
        return ("text", "quote")
    stem = line_field[: -len("_line")]
    fields = (f"{stem}_text", f"{stem}_quote")
    if line_field == "first_line":
        return (*fields, "representative_quote")
    return fields


def _line_references(value: Any, *, field_name: str | None = None) -> set[int]:
    result: set[int] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.update(_line_references(item, field_name=str(key)))
        return result
    if isinstance(value, (list, tuple)):
        if field_name and field_name.endswith("lines"):
            result.update(
                int(item)
                for item in value
                if isinstance(item, int) and not isinstance(item, bool) and item > 0
            )
        for item in value:
            result.update(_line_references(item))
        return result
    if (
        field_name
        and (field_name == "line" or field_name.endswith("_line"))
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    ):
        result.add(value)
    return result
