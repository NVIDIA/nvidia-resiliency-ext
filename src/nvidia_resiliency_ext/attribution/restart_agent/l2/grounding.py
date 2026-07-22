# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers for determining which source evidence was visible to L1."""

from __future__ import annotations

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
            visible.update(_line_references(event.get("result")))
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
            _collect_line_texts(event.get("result"), visible)
    if not saw_model_payload:
        _collect_line_texts(model_view.prompt_payload(), visible)
    return visible


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
