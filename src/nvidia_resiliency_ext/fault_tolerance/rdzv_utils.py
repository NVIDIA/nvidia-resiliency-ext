# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for interpreting raw rendezvous configuration dictionaries."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def rdzv_config_get_as_bool(
    configs: Mapping[str, Any], key: str, default: Optional[bool] = None
) -> Optional[bool]:
    """Return a raw rendezvous config value as a bool.

    This mirrors ``torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_bool``
    for call sites that only have the parsed rendezvous config dictionary.
    """
    value = configs.get(key, default)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    elif isinstance(value, str):
        if value.lower() in ["1", "true", "t", "yes", "y"]:
            return True
        if value.lower() in ["0", "false", "f", "no", "n"]:
            return False
    raise ValueError(
        f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
    )


def rdzv_config_get_as_int(
    configs: Mapping[str, Any], key: str, default: Optional[int] = None
) -> Optional[int]:
    """Return a raw rendezvous config value as an int."""
    value = configs.get(key, default)
    if value is None:
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid integer value."
        ) from exc


def rdzv_config_get_as_float(
    configs: Mapping[str, Any], key: str, default: Optional[float] = None
) -> Optional[float]:
    """Return a raw rendezvous config value as a float."""
    value = configs.get(key, default)
    if value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid float value."
        ) from exc
