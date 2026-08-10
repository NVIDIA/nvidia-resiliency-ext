# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-pass state: the little that cannot be recovered from disk.

Cycle-info files hold the restart history, so state keeps only *when a value was first
seen* (which turns a snapshot into a stall timer) and which alerts have already been
sent. Losing this file is not an error: stall timers restart from the current pass,
delaying a detection by at most one threshold. That is the right trade for never having
the watcher itself be the thing that needs recovering.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

from .parsing import parse_iso
from .types import CheckpointProgress, CycleRecord, PriorState, utcnow


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def load(path: str) -> tuple[PriorState, dict[str, datetime]]:
    """Return (prior observation, alert key -> last sent time)."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return PriorState(), {}
    if not isinstance(data, dict):
        return PriorState(), {}

    checkpoint = data.get("checkpoint_iteration") or {}
    cycle = data.get("latest_cycle") or {}
    prior = PriorState(
        checkpoint_value=checkpoint.get("value"),
        checkpoint_first_seen=parse_iso(checkpoint.get("first_seen")),
        latest_cycle_key=cycle.get("key"),
        latest_cycle_first_seen=parse_iso(cycle.get("first_seen")),
        last_pass=parse_iso(data.get("last_pass")),
    )
    alerts = {
        key: sent
        for key, sent in ((key, parse_iso(raw)) for key, raw in (data.get("alerts") or {}).items())
        if sent is not None
    }
    return prior, alerts


def advance(
    prior: PriorState,
    checkpoint: CheckpointProgress,
    latest_cycle: CycleRecord | None,
    now: datetime | None = None,
) -> PriorState:
    """Carry first-seen timestamps forward, resetting them when the value changes."""
    now = now or utcnow()

    if checkpoint.value is None:
        checkpoint_first_seen = None
    elif checkpoint.value == prior.checkpoint_value:
        checkpoint_first_seen = prior.checkpoint_first_seen or checkpoint.mtime or now
    else:
        checkpoint_first_seen = checkpoint.mtime or now

    cycle_key = latest_cycle.key if latest_cycle else None
    if cycle_key is not None and cycle_key == prior.latest_cycle_key:
        cycle_first_seen = prior.latest_cycle_first_seen or now
    else:
        cycle_first_seen = now if cycle_key is not None else None

    return PriorState(
        checkpoint_value=checkpoint.value,
        checkpoint_first_seen=checkpoint_first_seen,
        latest_cycle_key=cycle_key,
        latest_cycle_first_seen=cycle_first_seen,
        last_pass=now,
    )


def save(path: str, prior: PriorState, alerts: dict[str, datetime]) -> None:
    """Write state atomically; a torn state file would be read as 'no history'."""
    payload = {
        "last_pass": _iso(prior.last_pass),
        "checkpoint_iteration": {
            "value": prior.checkpoint_value,
            "first_seen": _iso(prior.checkpoint_first_seen),
        },
        "latest_cycle": {
            "key": prior.latest_cycle_key,
            "first_seen": _iso(prior.latest_cycle_first_seen),
        },
        "alerts": {key: _iso(sent) for key, sent in alerts.items()},
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".state-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def prune_alerts(
    alerts: dict[str, datetime], keep_seconds: float, now: datetime | None = None
) -> dict[str, datetime]:
    """Drop dedup entries older than the cooldown so the file cannot grow without bound."""
    now = now or utcnow()
    return {
        key: sent for key, sent in alerts.items() if (now - sent).total_seconds() < keep_seconds
    }
