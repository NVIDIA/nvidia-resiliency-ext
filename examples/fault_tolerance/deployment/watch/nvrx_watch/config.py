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

"""Configuration: defaults, then a JSON config file, then env, then CLI flags."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any

ENV_PREFIX = "NVRX_WATCH_"


@dataclass
class Config:
    # --- what to watch -------------------------------------------------------------
    job_id: str = ""  # a Slurm job id of the chain; job_name/user/work_dir derive from it
    job_name: str = "nvrx_singleton"  # overridden by what job_id resolves to
    work_dir: str = ""  # the sbatch's NVRX_WORK_DIR; cycle infos and checkpoints below it
    cycle_info_glob: str = ""  # overrides the work-dir layout when set
    checkpoint_iteration_file: str = ""  # overrides the work-dir layout when set
    platform: str = "slurm"  # "slurm" or "none" (cycle-info-only mode)
    max_restarts: int | None = None  # ft_launcher --max-restarts, for restart_budget_low

    # --- behaviour -----------------------------------------------------------------
    dry_run: bool = False
    # Detect and page but take no corrective action -- for an SRE who monitors jobs they
    # do not own (and cannot scancel). Detection and notification are all reads; only the
    # one action (orphaned_generation's cancel) is suppressed and reported instead. Default
    # on: the common operator is the SRE. The job owner opts into actions with --act.
    observe_only: bool = True
    user: str = ""  # scope squeue/sacct to this owner (-u); empty = all users of the name
    state_dir: str = ""  # default ~/.nvrx_watch
    expect_file: str = ""  # default ~/.nvrx_watch_expect_chain
    command_timeout: float = 30.0
    alert_cooldown: float = 3600.0

    # --- chain reconciliation ------------------------------------------------------
    grace: float = 120.0  # after task 0 ends, before treating a generation as orphaned
    churn_window: float = 6 * 3600.0
    max_generations_per_window: int = 3
    no_restart_exit_code: int = 93

    # --- restart anomalies ---------------------------------------------------------
    storm_cycles: int = 5
    storm_window: float = 30 * 60.0
    stall_cycles: int = 3  # completed cycles with no checkpoint movement
    stall_seconds: float = 3600.0  # open cycle with nothing moving
    short_cycle_seconds: float = 600.0
    suspect_cycles: int = 3
    budget_fraction: float = 0.8

    # --- reporting -----------------------------------------------------------------
    heartbeat_url: str = ""
    pd_routing_key: str = ""
    webhook_url: str = ""
    log_file: str = ""  # default <state_dir>/watch.log

    # --- detector selection --------------------------------------------------------
    disable: tuple[str, ...] = ()  # detector names to skip

    def __post_init__(self) -> None:
        home = os.path.expanduser("~")
        self.state_dir = self.state_dir or os.path.join(home, ".nvrx_watch")
        self.expect_file = self.expect_file or os.path.join(home, ".nvrx_watch_expect_chain")
        self.log_file = self.log_file or os.path.join(self.state_dir, "watch.log")
        if isinstance(self.disable, str):
            self.disable = tuple(d.strip() for d in self.disable.split(",") if d.strip())
        else:
            self.disable = tuple(self.disable)

    # --- derived paths -------------------------------------------------------------
    @property
    def resolved_cycle_info_glob(self) -> str:
        """Where cycle-info files live.

        The sbatch writes them to <work_dir>/nvrx/<array_job_id>/cycle_infos/, one
        directory per generation, so the default glob spans every generation of the run.
        """
        if self.cycle_info_glob:
            return self.cycle_info_glob
        if not self.work_dir:
            return ""
        return os.path.join(self.work_dir, "nvrx", "*", "cycle_infos", "cycle_info.*")

    @property
    def resolved_checkpoint_file(self) -> str:
        if self.checkpoint_iteration_file:
            return self.checkpoint_iteration_file
        if not self.work_dir:
            return ""
        return os.path.join(self.work_dir, "checkpoints", "latest_checkpointed_iteration.txt")

    @property
    def state_file(self) -> str:
        return os.path.join(self.state_dir, "state.json")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(name: str, raw: str, current: Any) -> Any:
    if name == "disable":
        return tuple(d.strip() for d in raw.split(",") if d.strip())
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, int) or name in ("max_restarts",):
        return int(raw)
    return raw


def load(
    config_file: str | None = None,
    env: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Build a Config from defaults, an optional JSON file, env, then explicit overrides."""
    env = os.environ if env is None else env
    values: dict[str, Any] = {}

    if config_file:
        with open(config_file) as fh:
            loaded = json.load(fh)
        unknown = set(loaded) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"unknown keys in {config_file}: {', '.join(sorted(unknown))}")
        values.update(loaded)

    defaults = Config()

    # Ergonomic bridge: the sbatch already exports NVRX_WORK_DIR and NVRX_JOB_NAME to
    # submit the chain, so let the watcher inherit them -- one env block serves both, and
    # a bare `python3 -m nvrx_watch` just works. The watcher's own NVRX_WATCH_* vars
    # (loaded next) still take precedence.
    for env_name, field_name in (("NVRX_WORK_DIR", "work_dir"), ("NVRX_JOB_NAME", "job_name")):
        raw = env.get(env_name)
        if raw:
            values[field_name] = raw

    for f in fields(Config):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is not None and raw != "":
            values[f.name] = _coerce(f.name, raw, getattr(defaults, f.name))

    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value

    return Config(**values)
