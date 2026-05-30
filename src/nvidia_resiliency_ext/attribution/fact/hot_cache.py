# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Iterable

from nvidia_resiliency_ext.attribution.fact.models import FactHistoryRecord, HotCacheEpisode


class FactHotCache:
    """In-memory current-process FACT episode overlay."""

    def __init__(self, max_episodes: int = 4096) -> None:
        self.max_episodes = max(1, int(max_episodes))
        self._episodes: "OrderedDict[tuple[str, str, str], HotCacheEpisode]" = OrderedDict()

    def add_episode(self, episode: HotCacheEpisode) -> None:
        key = (episode.cluster, episode.node, episode.episode_id)
        self._episodes.pop(key, None)
        self._episodes[key] = episode
        while len(self._episodes) > self.max_episodes:
            self._episodes.popitem(last=False)

    def add_current_cycle(
        self,
        *,
        cluster: str,
        nodes: Iterable[str],
        job_id: str,
        cycle_id: int,
        event_time: datetime,
    ) -> None:
        episode_id = f"{job_id}_{cycle_id}"
        for node in sorted({str(node) for node in nodes if str(node)}):
            self.add_episode(
                HotCacheEpisode(
                    cluster=cluster,
                    node=node,
                    episode_id=episode_id,
                    event_time=event_time,
                )
            )

    def records_for(
        self,
        *,
        cluster: str,
        nodes: Iterable[str],
        before: datetime,
    ) -> list[FactHistoryRecord]:
        node_set = {str(node) for node in nodes if str(node)}
        records = []
        for episode in self._episodes.values():
            if episode.cluster != cluster or episode.node not in node_set:
                continue
            if episode.event_time >= before:
                continue
            records.append(
                FactHistoryRecord(
                    cluster=episode.cluster,
                    node=episode.node,
                    episode_id=episode.episode_id,
                    event_time=episode.event_time,
                )
            )
        return records
