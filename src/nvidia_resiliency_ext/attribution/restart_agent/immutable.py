# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable, JSON-compatible containers for shared pipeline payloads."""

from __future__ import annotations

from typing import Any, Mapping


class FrozenDict(dict):
    """JSON-compatible dictionary that rejects mutation after construction."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen JSON payload cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class FrozenList(list):
    """JSON-compatible list that rejects mutation after construction."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen JSON payload cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def freeze_json_value(value: Any) -> Any:
    """Recursively freeze JSON-like stage payloads without breaking serialization."""

    if isinstance(value, Mapping):
        return FrozenDict({key: freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return FrozenList(freeze_json_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_json_value(item) for item in value)
    return value
