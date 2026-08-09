# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize provider output to the advertised L1 semantic contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .response_contract import model_response_schema


def normalize_model_evidence_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return contract-defined fields and the paths removed from provider output."""

    removed: list[str] = []
    normalized = _project_value(payload, model_response_schema(), path="", removed=removed)
    if not isinstance(normalized, dict):  # The root schema is always an object.
        raise TypeError("normalized L1 model evidence must be an object")
    return normalized, tuple(sorted(removed))


def _project_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    removed: list[str],
) -> Any:
    schema = _select_schema_branch(value, schema)
    properties = schema.get("properties")
    if isinstance(value, Mapping) and isinstance(properties, Mapping):
        projected: dict[str, Any] = {}
        for key, item in value.items():
            field_path = f"{path}.{key}" if path else key
            field_schema = properties.get(key)
            if not isinstance(field_schema, Mapping):
                if schema.get("additionalProperties") is False:
                    removed.append(field_path)
                    continue
                projected[key] = deepcopy(item)
                continue
            projected[key] = _project_value(
                item,
                field_schema,
                path=field_path,
                removed=removed,
            )
        return projected

    item_schema = schema.get("items")
    if isinstance(value, list) and isinstance(item_schema, Mapping):
        return [
            _project_value(
                item,
                item_schema,
                path=f"{path}[{index}]",
                removed=removed,
            )
            for index, item in enumerate(value)
        ]
    return deepcopy(value)


def _select_schema_branch(value: Any, schema: Mapping[str, Any]) -> Mapping[str, Any]:
    alternatives = schema.get("oneOf")
    if not isinstance(alternatives, list):
        return schema
    for alternative in alternatives:
        if isinstance(alternative, Mapping) and _type_matches(value, alternative.get("type")):
            return alternative
    return schema


def _type_matches(value: Any, expected: Any) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    return any(
        (item == "object" and isinstance(value, Mapping))
        or (item == "array" and isinstance(value, list))
        or (item == "null" and value is None)
        or (item == "string" and isinstance(value, str))
        or (item == "integer" and isinstance(value, int) and not isinstance(value, bool))
        for item in expected_types
    )
