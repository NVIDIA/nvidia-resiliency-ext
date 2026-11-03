# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the NVRx Scheduler Exclusion Service."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from .monitor import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    DEFAULT_REFRESH_INTERVAL_SECONDS,
    SchedulerExclusionConfig,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
ENV_PREFIX = "NVRX_SCHEDULER_EXCLUSION_"
_T = TypeVar("_T")


@dataclass(frozen=True)
class SchedulerExclusionServiceSettings:
    """Process and monitor settings loaded from CLI or environment."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    slurm_bin_dir: str = ""
    slurm_conf: str = ""
    scheduler_exclusion_dir: str = ""
    refresh_interval_seconds: float = DEFAULT_REFRESH_INTERVAL_SECONDS
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    query_timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        for name in ("slurm_bin_dir", "slurm_conf", "scheduler_exclusion_dir"):
            value = getattr(self, name)
            if value and not os.path.isabs(value):
                raise ValueError(f"{name} must be an absolute path")
        for name in (
            "refresh_interval_seconds",
            "cache_ttl_seconds",
            "query_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "SchedulerExclusionServiceSettings":
        """Load settings from the ``NVRX_SCHEDULER_EXCLUSION_`` environment."""
        values = os.environ if env is None else env

        def get(name: str, default: object) -> str:
            return str(values.get(f"{ENV_PREFIX}{name}", default)).strip()

        def convert(name: str, default: object, parser: Callable[[str], _T]) -> _T:
            raw = get(name, default)
            try:
                return parser(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid value for {ENV_PREFIX}{name}: {raw!r}") from exc

        return cls(
            host=get("HOST", DEFAULT_HOST),
            port=convert("PORT", DEFAULT_PORT, int),
            slurm_bin_dir=get("SLURM_BIN_DIR", ""),
            slurm_conf=get("SLURM_CONF", ""),
            scheduler_exclusion_dir=get("DIR", ""),
            refresh_interval_seconds=convert(
                "REFRESH_INTERVAL_SECONDS",
                DEFAULT_REFRESH_INTERVAL_SECONDS,
                float,
            ),
            cache_ttl_seconds=convert(
                "CACHE_TTL_SECONDS",
                DEFAULT_CACHE_TTL_SECONDS,
                float,
            ),
            query_timeout_seconds=convert(
                "QUERY_TIMEOUT_SECONDS",
                DEFAULT_QUERY_TIMEOUT_SECONDS,
                float,
            ),
        )

    def monitor_config(self) -> SchedulerExclusionConfig:
        """Build the scheduler-query configuration."""
        return SchedulerExclusionConfig(
            slurm_bin_dir=self.slurm_bin_dir,
            slurm_conf=self.slurm_conf,
            scheduler_exclusion_dir=self.scheduler_exclusion_dir,
            refresh_interval_seconds=self.refresh_interval_seconds,
            cache_ttl_seconds=self.cache_ttl_seconds,
            query_timeout_seconds=self.query_timeout_seconds,
        )
