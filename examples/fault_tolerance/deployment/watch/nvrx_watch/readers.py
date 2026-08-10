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

"""Filesystem readers: the run's on-disk artifacts, turned into model records.

Two artifacts, both under the run's work dir: NVRx cycle-info files (JSON
serializations of the NVRxCycleInfo protobuf written by
``fault_tolerance/cycle_info_writer.py``, one per cycle, named
``cycle_info.<job_id>.<attempt>.<cycle>``) and the launcher's checkpoint iteration
file. Everything here reads defensively -- a half-written file must degrade to "skip
this record", never to a crashed pass.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

from .parsing import parse_iso
from .types import CheckpointProgress, CycleRecord


def parse_cycle_file(path: str) -> CycleRecord | None:
    """Parse one cycle-info file. Returns None if it is unreadable or not a cycle info."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "job_id" not in data:
        return None
    try:
        attempt = int(data.get("attempt_index", 0) or 0)
        cycle_number = int(data.get("cycle_number", 0) or 0)
    except (TypeError, ValueError):
        return None
    return CycleRecord(
        job_id=str(data.get("job_id", "")),
        attempt_index=attempt,
        cycle_number=cycle_number,
        start_time=parse_iso(data.get("cycle_start_time")),
        end_time=parse_iso(data.get("cycle_end_time")),
        active_nodes=str(data.get("active_nodes", "") or ""),
        standby_nodes=str(data.get("standby_nodes", "") or ""),
        log_file=str(data.get("cycle_log_file", "") or ""),
        path=path,
    )


def read_cycles(pattern: str) -> tuple[CycleRecord, ...]:
    """Read every cycle-info file matching ``pattern``, skipping the .current symlinks.

    The symlink duplicates a file already in the glob; counting it would inflate every
    cycle-count threshold by one.
    """
    if not pattern:
        return ()
    records: dict[str, CycleRecord] = {}
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".current") or os.path.islink(path):
            continue
        record = parse_cycle_file(path)
        if record is not None:
            records[record.key] = record
    return tuple(records.values())


def read_checkpoint_progress(path: str) -> CheckpointProgress:
    """Read the launcher's --ft-checkpoint-iteration-file: a single integer, plus mtime."""
    if not path:
        return CheckpointProgress()
    try:
        stat = os.stat(path)
        with open(path) as fh:
            raw = fh.read().strip()
    except OSError:
        return CheckpointProgress()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    try:
        # Megatron writes 'release' for a released checkpoint; treat as "no iteration".
        return CheckpointProgress(value=int(raw), mtime=mtime)
    except ValueError:
        return CheckpointProgress(value=None, mtime=mtime)
