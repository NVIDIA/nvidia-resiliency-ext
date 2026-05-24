# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class FactHistoryRecord:
    cluster: str
    node: str
    episode_id: str
    event_time: datetime


@dataclass(frozen=True)
class HotCacheEpisode:
    cluster: str
    node: str
    episode_id: str
    event_time: datetime


@dataclass(frozen=True)
class AvoidCandidate:
    node: str
    repeat_count: int
    prior_last_seen: datetime | None = None


@dataclass(frozen=True)
class AvoidDecision:
    cycle_id: int
    status: str
    avoid_nodes: list[str] = field(default_factory=list)
    candidates: list[AvoidCandidate] = field(default_factory=list)
