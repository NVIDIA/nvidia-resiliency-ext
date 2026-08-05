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

"""Tests for the nvrx-watch example tool.

Detectors are pure functions over a Snapshot, so these build snapshots directly: no
cluster, no subprocess. The source layer is exercised against cycle-info JSON in the
format CycleInfoWriter produces.
"""

import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
WATCH_DIR = os.path.join(REPO_ROOT, "examples", "fault_tolerance", "deployment", "watch")
sys.path.insert(0, os.path.abspath(WATCH_DIR))

from nvrx_watch import detectors, parsing, persistence, readers, runner, types  # noqa: E402
from nvrx_watch.config import Config  # noqa: E402
from nvrx_watch.platform import NullPlatform, PlatformError, SlurmPlatform  # noqa: E402

NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


def ago(**kwargs):
    return NOW - timedelta(**kwargs)


def cycle(
    number,
    *,
    job_id="job1",
    start_min_ago=10,
    duration_min=None,
    active="node[001-002]",
    standby="node003",
):
    start = ago(minutes=start_min_ago)
    end = start + timedelta(minutes=duration_min) if duration_min is not None else None
    return types.CycleRecord(
        job_id=job_id,
        attempt_index=0,
        cycle_number=number,
        start_time=start,
        end_time=end,
        active_nodes=active,
        standby_nodes=standby,
    )


def make_snapshot(**kwargs):
    caps = kwargs.pop("capabilities", (types.CAP_CYCLES, types.CAP_CHECKPOINT, types.CAP_PLATFORM))
    return types.Snapshot(observed_at=NOW, job_name="chain", capabilities=frozenset(caps), **kwargs)


@pytest.fixture
def config(tmp_path):
    return Config(job_name="chain", work_dir=str(tmp_path), state_dir=str(tmp_path / "state"))


# ---------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------
class TestNodelist:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("node001", ["node001"]),
            ("node[001-003]", ["node001", "node002", "node003"]),
            ("node[001-002,007]", ["node001", "node002", "node007"]),
            ("a[1-2],b3", ["a1", "a2", "b3"]),
            ("", []),
            ("node[01-02]suffix", ["node01suffix", "node02suffix"]),
        ],
    )
    def test_expand(self, raw, expected):
        assert parsing.expand_nodelist(raw) == expected

    def test_split_keeps_ranges_intact(self):
        assert parsing.split_nodelist("n[1,2],m[3,4]") == ["n[1,2]", "m[3,4]"]


class TestFtLauncherPathResolution:
    PROD = (
        "#!/bin/bash\n"
        "CELL_ID=injob_512n\n"
        'LOG_ROOT_BASE="/scratch/team/restart_matrix"\n'
        'LOG_ROOT="${LOG_ROOT_BASE}/${CELL_ID}"\n'
        'CHECKPOINT_DIR="${LOG_ROOT}/checkpoints"\n'
        'NVRX_DIR="${LOG_ROOT}/nvrx"\n'
        'NVRX_JOB_DIR="${NVRX_DIR}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"\n'
        "LAUNCHER_ARGS=\" \\\n"
        "    --ft-cycle-info-dir=${NVRX_JOB_DIR}/cycle_infos \\\n"
        "    --ft-checkpoint-iteration-file=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt \\\n"
        '"\n'
    )

    def test_production_style_resolves_with_generation_glob(self):
        cyc, ckpt = parsing.resolve_ft_launcher_paths(self.PROD)
        # the array id becomes '*' so the glob spans every generation
        assert cyc == "/scratch/team/restart_matrix/injob_512n/nvrx/*/cycle_infos/cycle_info.*"
        assert (
            ckpt == "/scratch/team/restart_matrix/injob_512n/checkpoints/"
            "latest_checkpointed_iteration.txt"
        )

    def test_exported_env_var_stays_unresolved(self):
        # run dir comes from an exported NVRX_WORK_DIR (not defined in-script) -> unresolved
        script = (
            'NVRX_WORK_DIR="${NVRX_WORK_DIR:-${PWD}/nvrx-run}"\n'
            'NVRX_DIR="${NVRX_WORK_DIR}/nvrx"\n'
            "run=\" --ft-cycle-info-dir=${NVRX_DIR}/${SLURM_ARRAY_JOB_ID}/cycle_infos \"\n"
        )
        cyc, ckpt = parsing.resolve_ft_launcher_paths(script)
        assert cyc is None and ckpt is None

    def test_missing_args_return_none(self):
        assert parsing.resolve_ft_launcher_paths("#!/bin/bash\necho hi\n") == (None, None)

    def test_space_separated_arg_form(self):
        script = 'D=/abs/run\nargs="--ft-cycle-info-dir ${D}/ci"\n'
        cyc, _ = parsing.resolve_ft_launcher_paths(script)
        assert cyc == "/abs/run/ci/cycle_info.*"

    def test_sourced_libraries_are_followed(self):
        """The real production shape: the cell sources common libs that build the ft args
        and hold the path vars, and roots COMMON_DIR at its own path via cd/dirname/pwd."""
        cell_path = "/lustre/u/hexinw/exp/slurm/hsg/restart_matrix/cells/injob_v3.sh"
        common = "/lustre/u/hexinw/exp/slurm/restart_matrix_common"
        cell = (
            "#!/bin/bash\n"
            "CELL_ID=injob_v3\n"
            'SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | awk -F= \'/Command=/{print $2}\')"\n'
            'COMMON_DIR="$(cd "$(dirname "${SCRIPT_PATH}")/../../../restart_matrix_common" && pwd)"\n'
            'source "${COMMON_DIR}/env.common.sh"\n'
            'source "${COMMON_DIR}/lib/paths.sh"\n'
            'source "${COMMON_DIR}/lib/ft_launcher.sh"\n'
        )
        files = {
            f"{common}/env.common.sh": 'MY_RUN_DIR="/lustre/u/hexinw/run"\n'
            'LOG_ROOT_BASE="${MY_RUN_DIR}/restart_matrix"\n',
            f"{common}/lib/paths.sh": 'LOG_ROOT="${LOG_ROOT_BASE}/${CELL_ID}"\n'
            'NVRX_DIR="${LOG_ROOT}/nvrx"\n'
            'CHECKPOINT_DIR="${LOG_ROOT}/checkpoints"\n'
            'NVRX_JOB_DIR="${NVRX_DIR}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"\n',
            f"{common}/lib/ft_launcher.sh": "ARGS=\" \\\n"
            "  --ft-cycle-info-dir=${NVRX_JOB_DIR}/cycle_infos \\\n"
            "  --ft-checkpoint-iteration-file=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt \\\n"
            '"\n',
        }
        cyc, ckpt = parsing.resolve_ft_launcher_paths(
            cell, script_path=cell_path, read_file=files.get
        )
        base = "/lustre/u/hexinw/run/restart_matrix/injob_v3"
        assert cyc == f"{base}/nvrx/*/cycle_infos/cycle_info.*"
        assert ckpt == f"{base}/checkpoints/latest_checkpointed_iteration.txt"

    def test_unreadable_source_leaves_paths_unresolved(self):
        cell = 'source "${COMMON_DIR}/lib/ft_launcher.sh"\nCOMMON_DIR=/x\n'
        # COMMON_DIR is set after the source line (as in a real cd/pwd root that failed),
        # and the sourced file cannot be read -> nothing resolves, caller falls back.
        cyc, ckpt = parsing.resolve_ft_launcher_paths(cell, read_file=lambda _p: None)
        assert cyc is None and ckpt is None


class TestCycleInfoReader:
    def write(self, directory, job_id, attempt, number, **overrides):
        payload = {
            "job_id": job_id,
            "attempt_index": attempt,
            "cycle_number": number,
            "cycle_start_time": "2026-07-30T11:00:00.000000Z",
            "cycle_end_time": "",
            "cycle_log_file": "/logs/c.log",
            "active_nodes": "node[001-002]",
            "standby_nodes": "node003",
            "generation": 0,
            "active_ranks": "0-1",
        }
        payload.update(overrides)
        path = os.path.join(directory, f"cycle_info.{job_id}.{attempt}.{number}")
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path

    def test_parses_writer_format(self, tmp_path):
        path = self.write(str(tmp_path), "job1", 0, 3)
        record = readers.parse_cycle_file(path)
        assert (record.job_id, record.cycle_number) == ("job1", 3)
        assert record.is_open  # empty cycle_end_time means the cycle is still running
        assert record.active_nodes == "node[001-002]"

    def test_end_time_closes_cycle(self, tmp_path):
        path = self.write(str(tmp_path), "job1", 0, 1, cycle_end_time="2026-07-30T11:30:00.000000Z")
        record = readers.parse_cycle_file(path)
        assert not record.is_open
        assert record.duration_seconds() == 1800

    def test_current_symlink_is_not_double_counted(self, tmp_path):
        self.write(str(tmp_path), "job1", 0, 0)
        link = tmp_path / "cycle_info.job1.current"
        os.symlink(tmp_path / "cycle_info.job1.0.0", link)
        found = readers.read_cycles(str(tmp_path / "cycle_info.*"))
        assert len(found) == 1

    def test_malformed_file_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "cycle_info.bad.0.0").write_text("{not json")
        self.write(str(tmp_path), "job1", 0, 0)
        assert len(readers.read_cycles(str(tmp_path / "cycle_info.*"))) == 1

    def test_checkpoint_progress(self, tmp_path):
        target = tmp_path / "latest_checkpointed_iteration.txt"
        target.write_text("4200\n")
        progress = readers.read_checkpoint_progress(str(target))
        assert progress.value == 4200 and progress.mtime is not None

    def test_checkpoint_release_marker(self, tmp_path):
        target = tmp_path / "iter.txt"
        target.write_text("release")
        progress = readers.read_checkpoint_progress(str(target))
        assert progress.value is None and progress.mtime is not None

    def test_missing_checkpoint_file(self):
        assert readers.read_checkpoint_progress("/nonexistent/iter.txt").mtime is None


# ---------------------------------------------------------------------------------
# Chain reconciliation
# ---------------------------------------------------------------------------------
class TestOrphanedGeneration:
    def generation(self, task0_state="RUNNING", pending=2):
        tasks = [types.TaskInfo(task=0, state=task0_state)] if task0_state else []
        tasks += [types.TaskInfo(task=i + 1, state="PENDING") for i in range(pending)]
        return types.ChainGeneration(gen_id="5615858", tasks=tuple(tasks))

    def test_healthy_generation_is_quiet(self, config):
        snap = make_snapshot(generations=(self.generation(),))
        assert detectors.orphaned_generation(snap, config) == []

    def test_task0_gone_with_spares_queued_fires_and_cancels(self, config):
        snap = make_snapshot(
            generations=(self.generation(task0_state=None),),
            terminal_info={
                ("5615858", 0): types.TaskInfo(
                    task=0, state="FAILED", end_time=ago(minutes=30), exit_code=1
                )
            },
        )
        findings = detectors.orphaned_generation(snap, config)
        assert len(findings) == 1
        assert findings[0].action.kind == "cancel_pending"
        assert findings[0].action.target == "5615858"

    def test_grace_period_lets_a_mid_flight_trap_finish(self, config):
        snap = make_snapshot(
            generations=(self.generation(task0_state=None),),
            terminal_info={
                ("5615858", 0): types.TaskInfo(task=0, state="FAILED", end_time=ago(seconds=30))
            },
        )
        assert detectors.orphaned_generation(snap, config) == []

    def test_sacct_silence_is_unknown_never_gone(self, config):
        """The most dangerous false positive: cancelling a live generation because
        accounting lagged."""
        snap = make_snapshot(generations=(self.generation(task0_state=None),), terminal_info={})
        assert detectors.orphaned_generation(snap, config) == []

    def test_no_pending_spares_is_a_normal_drain(self, config):
        snap = make_snapshot(
            generations=(self.generation(task0_state=None, pending=0),),
            terminal_info={
                ("5615858", 0): types.TaskInfo(task=0, state="FAILED", end_time=ago(minutes=30))
            },
        )
        assert detectors.orphaned_generation(snap, config) == []

    def test_task0_still_running_per_sacct_is_quiet(self, config):
        snap = make_snapshot(
            generations=(self.generation(task0_state=None),),
            terminal_info={("5615858", 0): types.TaskInfo(task=0, state="RUNNING")},
        )
        assert detectors.orphaned_generation(snap, config) == []


class TestChainExhausted:
    def test_fires_only_when_a_chain_is_expected(self, config):
        assert detectors.chain_exhausted(make_snapshot(chain_expected=False), config) == []
        findings = detectors.chain_exhausted(make_snapshot(chain_expected=True), config)
        assert len(findings) == 1 and findings[0].severity == types.CRITICAL

    def test_quiet_while_a_generation_exists(self, config):
        snap = make_snapshot(
            chain_expected=True,
            generations=(types.ChainGeneration("1", (types.TaskInfo(task=0, state="RUNNING"),)),),
        )
        assert detectors.chain_exhausted(snap, config) == []


class TestChainNotCancelled:
    def test_no_restart_verdict_with_queued_successor_is_critical(self, config):
        snap = make_snapshot(
            generations=(types.ChainGeneration("2", (types.TaskInfo(task=0, state="PENDING"),)),),
            recent_endings=(
                (
                    "1",
                    types.TaskInfo(task=0, state="FAILED", exit_code=93, end_time=ago(hours=1)),
                ),
            ),
        )
        findings = detectors.chain_not_cancelled(snap, config)
        assert len(findings) == 1 and findings[0].severity == types.CRITICAL

    def test_ordinary_failure_does_not_fire(self, config):
        snap = make_snapshot(
            generations=(types.ChainGeneration("2", (types.TaskInfo(task=0, state="PENDING"),)),),
            recent_endings=(
                (
                    "1",
                    types.TaskInfo(task=0, state="FAILED", exit_code=1, end_time=ago(hours=1)),
                ),
            ),
        )
        assert detectors.chain_not_cancelled(snap, config) == []

    def test_draining_predecessor_does_not_mask_its_successors(self, config):
        """cancel_chain scancels by job name, so one call covers both the successors and
        the ended generation's own cold spares: a failure leaves generation 1 queued
        alongside them. The predecessor draining must not suppress the verdict, and the
        count must exclude it."""
        spares = tuple(types.TaskInfo(task=t, state="PENDING") for t in (2, 3, 4))
        snap = make_snapshot(
            generations=(
                types.ChainGeneration("1", spares),
                types.ChainGeneration("2", (types.TaskInfo(task=0, state="PENDING"),)),
                types.ChainGeneration("3", (types.TaskInfo(task=0, state="PENDING"),)),
            ),
            recent_endings=(
                (
                    "1",
                    types.TaskInfo(task=0, state="FAILED", exit_code=93, end_time=ago(hours=1)),
                ),
            ),
        )
        findings = detectors.chain_not_cancelled(snap, config)
        assert len(findings) == 1 and findings[0].severity == types.CRITICAL
        assert "2 'chain' generation(s)" in findings[0].summary

    def test_lone_draining_generation_is_not_a_successor(self, config):
        """The other half of the same test: when the ended generation is the only thing
        left in the queue there is no successor to stop, so this stays silent."""
        snap = make_snapshot(
            generations=(types.ChainGeneration("1", (types.TaskInfo(task=2, state="RUNNING"),)),),
            recent_endings=(
                (
                    "1",
                    types.TaskInfo(task=0, state="FAILED", exit_code=93, end_time=ago(hours=1)),
                ),
            ),
        )
        assert detectors.chain_not_cancelled(snap, config) == []

    def test_empty_queue_after_verdict_is_correct_behaviour(self, config):
        snap = make_snapshot(
            recent_endings=(
                (
                    "1",
                    types.TaskInfo(task=0, state="FAILED", exit_code=93, end_time=ago(hours=1)),
                ),
            ),
        )
        assert detectors.chain_not_cancelled(snap, config) == []


class TestGenerationChurn:
    def test_fires_above_threshold(self, config):
        endings = tuple(
            (str(i), types.TaskInfo(task=0, state="FAILED", end_time=ago(hours=i)))
            for i in range(1, 6)
        )
        findings = detectors.generation_churn(make_snapshot(recent_endings=endings), config)
        assert len(findings) == 1

    def test_ignores_endings_outside_the_window(self, config):
        endings = tuple(
            (str(i), types.TaskInfo(task=0, state="FAILED", end_time=ago(hours=10 + i)))
            for i in range(1, 6)
        )
        assert detectors.generation_churn(make_snapshot(recent_endings=endings), config) == []


# ---------------------------------------------------------------------------------
# Restart anomalies
# ---------------------------------------------------------------------------------
class TestRestartStorm:
    def test_fires_on_rate(self, config):
        records = tuple(cycle(i, start_min_ago=25 - i, duration_min=1) for i in range(6))
        findings = detectors.restart_storm(make_snapshot(cycles=records), config)
        assert len(findings) == 1

    def test_slow_restarts_are_quiet(self, config):
        records = tuple(cycle(i, start_min_ago=600 - i * 100, duration_min=90) for i in range(6))
        assert detectors.restart_storm(make_snapshot(cycles=records), config) == []


class TestStalledProgress:
    def live_owner(self, job_id="job1"):
        # cycle() defaults job_id="job1", so this is the live generation that owns them.
        return types.ChainGeneration(job_id, (types.TaskInfo(0, "RUNNING"),))

    def test_cycles_without_checkpoint_movement_are_critical(self, config):
        records = tuple(cycle(i, start_min_ago=40 - i * 10, duration_min=5) for i in range(4))
        snap = make_snapshot(
            cycles=records,
            generations=(self.live_owner(),),
            checkpoint=types.CheckpointProgress(value=4200, mtime=ago(hours=3)),
            prior=types.PriorState(checkpoint_value=4200, checkpoint_first_seen=ago(hours=3)),
        )
        findings = detectors.stalled_progress(snap, config)
        assert len(findings) == 1 and findings[0].severity == types.CRITICAL

    def test_progress_since_last_pass_clears_it(self, config):
        records = tuple(cycle(i, start_min_ago=40 - i * 10, duration_min=5) for i in range(4))
        snap = make_snapshot(
            cycles=records,
            generations=(self.live_owner(),),
            checkpoint=types.CheckpointProgress(value=4300, mtime=ago(minutes=5)),
            prior=types.PriorState(checkpoint_value=4300, checkpoint_first_seen=ago(minutes=5)),
        )
        assert detectors.stalled_progress(snap, config) == []

    def test_never_checkpointed_after_several_cycles(self, config):
        records = tuple(cycle(i, start_min_ago=40 - i * 10, duration_min=5) for i in range(3))
        snap = make_snapshot(
            cycles=records,
            generations=(self.live_owner(),),
            checkpoint=types.CheckpointProgress(),
        )
        findings = detectors.stalled_progress(snap, config)
        assert len(findings) == 1
        assert "no checkpoint iteration" in findings[0].summary

    def test_one_cycle_without_a_checkpoint_is_not_yet_a_problem(self, config):
        snap = make_snapshot(
            cycles=(cycle(0, duration_min=5),),
            generations=(self.live_owner(),),
            checkpoint=types.CheckpointProgress(),
        )
        assert detectors.stalled_progress(snap, config) == []

    def test_stale_dead_generation_is_gated_out(self, config):
        # The cycle-info glob spans generations; when nothing is running (the live gen is a
        # PENDING successor, or gone entirely) the newest cycles belong to a dead generation
        # whose files persist on disk. No progress/checkpoint alert for that -- it never ran
        # in this pass's live set.
        records = tuple(cycle(i, start_min_ago=40 - i * 10, duration_min=5) for i in range(3))
        snap = make_snapshot(
            cycles=records,  # job_id "job1"
            generations=(types.ChainGeneration("job2", (types.TaskInfo(0, "PENDING"),)),),
            checkpoint=types.CheckpointProgress(),
        )
        assert detectors.stalled_progress(snap, config) == []

    def test_cycle_only_mode_still_fires_without_platform(self, config):
        # --platform none: no generations to check liveness against, so the gate does not
        # apply and cycle-info evidence alone drives the finding.
        records = tuple(cycle(i, start_min_ago=40 - i * 10, duration_min=5) for i in range(3))
        snap = make_snapshot(
            cycles=records,
            capabilities=(types.CAP_CYCLES, types.CAP_CHECKPOINT),  # no CAP_PLATFORM
            checkpoint=types.CheckpointProgress(),
        )
        findings = detectors.stalled_progress(snap, config)
        assert len(findings) == 1 and "no checkpoint iteration" in findings[0].summary


class TestCycleStalled:
    def live_owner(self, job_id="job1"):
        return types.ChainGeneration(job_id, (types.TaskInfo(0, "RUNNING"),))

    def test_open_cycle_with_no_activity_and_live_generation_fires(self, config):
        snap = make_snapshot(
            cycles=(cycle(0, start_min_ago=180),),
            generations=(self.live_owner(),),
            checkpoint=types.CheckpointProgress(value=10, mtime=ago(hours=4)),
        )
        findings = detectors.cycle_stalled(snap, config)
        assert len(findings) == 1 and findings[0].severity == types.CRITICAL

    def test_recent_checkpoint_means_the_job_is_alive(self, config):
        snap = make_snapshot(
            cycles=(cycle(0, start_min_ago=180),),
            generations=(self.live_owner(),),
            checkpoint=types.CheckpointProgress(value=10, mtime=ago(minutes=2)),
        )
        assert detectors.cycle_stalled(snap, config) == []

    def test_closed_cycle_is_not_a_stall(self, config):
        snap = make_snapshot(cycles=(cycle(0, start_min_ago=180, duration_min=5),))
        assert detectors.cycle_stalled(snap, config) == []

    def test_unwritten_end_time_on_a_dead_generation_is_not_a_stall(self, config):
        """The cycle_end_time of the last cycle is not guaranteed to be written. A stale
        open cycle from a generation the platform can see is gone must not read as a hung
        cycle -- that is orphaned_generation / chain_exhausted, not cycle_stalled."""
        snap = make_snapshot(
            cycles=(cycle(0, start_min_ago=180),),  # is_open forever: no end_time
            generations=(),  # squeue shows nothing running for this run
            checkpoint=types.CheckpointProgress(value=10, mtime=ago(hours=4)),
        )
        assert detectors.cycle_stalled(snap, config) == []

    def test_platform_none_falls_back_to_best_effort(self, config):
        """Without a platform the generation's liveness is unknowable, so the detector
        still fires -- the operator opted into cycle-info-only mode."""
        snap = make_snapshot(
            capabilities=(types.CAP_CYCLES, types.CAP_CHECKPOINT),
            cycles=(cycle(0, start_min_ago=180),),
            checkpoint=types.CheckpointProgress(value=10, mtime=ago(hours=4)),
        )
        findings = detectors.cycle_stalled(snap, config)
        assert len(findings) == 1 and findings[0].severity == types.CRITICAL


class TestRestartBudget:
    def test_fires_near_the_limit(self, tmp_path):
        config = Config(max_restarts=7, state_dir=str(tmp_path))
        snap = make_snapshot(cycles=(cycle(6, duration_min=None),))
        assert len(detectors.restart_budget_low(snap, config)) == 1

    def test_quiet_early(self, tmp_path):
        config = Config(max_restarts=7, state_dir=str(tmp_path))
        assert detectors.restart_budget_low(make_snapshot(cycles=(cycle(1),)), config) == []

    def test_disabled_without_max_restarts(self, config):
        assert detectors.restart_budget_low(make_snapshot(cycles=(cycle(99),)), config) == []


class TestSparesExhausted:
    def test_no_standby_and_no_queued_spare(self, config):
        snap = make_snapshot(
            cycles=(cycle(0, standby=""),),
            generations=(types.ChainGeneration("job1", (types.TaskInfo(0, "RUNNING"),)),),
        )
        findings = detectors.spares_exhausted(snap, config)
        assert len(findings) == 1 and findings[0].severity == types.INFO

    def test_standby_present_is_quiet(self, config):
        assert detectors.spares_exhausted(make_snapshot(cycles=(cycle(0),)), config) == []

    def test_queued_cold_spare_is_quiet(self, config):
        snap = make_snapshot(
            cycles=(cycle(0, standby=""),),
            generations=(
                types.ChainGeneration(
                    "job1", (types.TaskInfo(0, "RUNNING"), types.TaskInfo(1, "PENDING"))
                ),
            ),
        )
        assert detectors.spares_exhausted(snap, config) == []


class TestSuspectNode:
    def test_node_common_to_consecutive_short_cycles(self, config):
        records = (
            cycle(0, start_min_ago=40, duration_min=2, active="node[001-002]"),
            cycle(1, start_min_ago=30, duration_min=2, active="node[002-003]"),
            cycle(2, start_min_ago=20, duration_min=2, active="node[002,004]"),
        )
        findings = detectors.suspect_node(make_snapshot(cycles=records), config)
        assert len(findings) == 1 and "node002" in findings[0].summary

    def test_no_shared_node_is_quiet(self, config):
        records = (
            cycle(0, start_min_ago=40, duration_min=2, active="node001"),
            cycle(1, start_min_ago=30, duration_min=2, active="node002"),
            cycle(2, start_min_ago=20, duration_min=2, active="node003"),
        )
        assert detectors.suspect_node(make_snapshot(cycles=records), config) == []

    def test_long_cycles_are_not_suspicious(self, config):
        records = tuple(
            cycle(i, start_min_ago=300 - i * 60, duration_min=55, active="node[001-002]")
            for i in range(3)
        )
        assert detectors.suspect_node(make_snapshot(cycles=records), config) == []


# ---------------------------------------------------------------------------------
# State, platform parsing, runner
# ---------------------------------------------------------------------------------
class TestState:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "persistence.json")
        prior = types.PriorState(
            checkpoint_value=42, checkpoint_first_seen=ago(hours=1), last_pass=NOW
        )
        persistence.save(path, prior, {"key1": NOW})
        loaded, alerts = persistence.load(path)
        assert loaded.checkpoint_value == 42
        assert loaded.checkpoint_first_seen == ago(hours=1)
        assert alerts["key1"] == NOW

    def test_missing_state_is_not_an_error(self, tmp_path):
        prior, alerts = persistence.load(str(tmp_path / "absent.json"))
        assert prior.checkpoint_value is None and alerts == {}

    def test_first_seen_holds_while_the_value_is_unchanged(self):
        first = ago(hours=2)
        prior = types.PriorState(checkpoint_value=100, checkpoint_first_seen=first)
        advanced = persistence.advance(
            prior, types.CheckpointProgress(value=100, mtime=first), None, NOW
        )
        assert advanced.checkpoint_first_seen == first

    def test_first_seen_resets_when_the_value_moves(self):
        prior = types.PriorState(checkpoint_value=100, checkpoint_first_seen=ago(hours=2))
        advanced = persistence.advance(
            prior, types.CheckpointProgress(value=200, mtime=ago(minutes=1)), None, NOW
        )
        assert advanced.checkpoint_first_seen == ago(minutes=1)

    def test_prune_drops_stale_alerts(self):
        alerts = {"old": ago(days=2), "fresh": ago(minutes=5)}
        assert set(persistence.prune_alerts(alerts, 3600.0, NOW)) == {"fresh"}


class TestSlurmParsing:
    def test_squeue_grouping_uses_the_array_job_id(self, monkeypatch):
        platform = SlurmPlatform()
        monkeypatch.setattr(
            platform, "_run", lambda argv: "100|0|RUNNING\n100|1|PENDING\n200|0|PENDING\n"
        )
        generations = {g.gen_id: g for g in platform.list_generations("chain")}
        assert set(generations) == {"100", "200"}
        assert len(generations["100"].pending) == 1
        assert generations["100"].task0.state == "RUNNING"

    def test_sacct_exit_code_is_parsed(self, monkeypatch):
        platform = SlurmPlatform()
        monkeypatch.setattr(platform, "_run", lambda argv: "FAILED|2026-07-30T11:00:00|93:0\n")
        info = platform.terminal_info("100", 0)
        assert info.exit_code == 93 and info.state == "FAILED"

    def test_describe_job_parses_name_and_owner(self, monkeypatch):
        platform = SlurmPlatform()
        scontrol_out = (
            "JobId=5728163 ArrayJobId=5728163 JobName=team_pretrain UserId=alice(1234) "
            "GroupId=alice(1234) Priority=1 JobState=RUNNING WorkDir=/home/alice/submit\n"
        )
        monkeypatch.setattr(platform, "_run", lambda argv: scontrol_out)
        desc = platform.describe_job("5728163")
        assert desc.job_name == "team_pretrain"
        assert desc.user == "alice"  # UserId=alice(1234) -> alice

    def test_describe_job_falls_back_to_sacct_when_scontrol_ages_out(self, monkeypatch):
        platform = SlurmPlatform()

        def fake_run(argv):
            if argv[0] == "scontrol":
                raise PlatformError("Invalid job id specified")
            return "team_run|alice\n"  # sacct JobName|User

        monkeypatch.setattr(platform, "_run", fake_run)
        desc = platform.describe_job("999")
        assert desc.job_name == "team_run" and desc.user == "alice"

    def test_describe_job_returns_none_when_both_fail(self, monkeypatch):
        platform = SlurmPlatform()

        def boom(argv):
            raise PlatformError("Invalid job id specified")

        monkeypatch.setattr(platform, "_run", boom)
        assert platform.describe_job("999") is None

    def test_batch_script_reads_the_command_path(self, tmp_path, monkeypatch):
        # No `scontrol write batch_script` (privileged); get Command= and read the file.
        script = tmp_path / "job.sbatch"
        script.write_text("#!/bin/bash\n--ft-cycle-info-dir=/x/ci\n")
        platform = SlurmPlatform()
        monkeypatch.setattr(
            platform, "_run", lambda argv: f"JobId=1 Command={script} JobState=RUNNING\n"
        )
        assert platform.batch_script("1").startswith("#!/bin/bash")

        # path present but unreadable -> None
        monkeypatch.setattr(platform, "_run", lambda argv: "JobId=1 Command=/no/such/file\n")
        assert platform.batch_script("1") is None

        # scontrol itself fails -> None
        def boom(argv):
            raise PlatformError("x")

        monkeypatch.setattr(platform, "_run", boom)
        assert platform.batch_script("1") is None  # non-owner / unknown id

    def test_user_scopes_squeue_and_sacct(self, monkeypatch):
        captured = {}

        def fake_run(argv):
            captured["argv"] = argv
            return ""

        platform = SlurmPlatform(user="alice")
        monkeypatch.setattr(platform, "_run", fake_run)
        platform.list_generations("chain")
        assert "-u" in captured["argv"] and "alice" in captured["argv"]

    def test_sacct_silence_returns_none(self, monkeypatch):
        platform = SlurmPlatform()
        monkeypatch.setattr(platform, "_run", lambda argv: "\n")
        assert platform.terminal_info("100", 0) is None

    def test_recent_endings_counts_each_generation_once(self, monkeypatch):
        # Real sacct output: JobID is "<arrayjob>_<task>" (there is no ArrayJobID
        # column); a still-pending array collapses to "<arrayjob>_[0-3%3]".
        captured = {}

        def fake_run(argv):
            captured["argv"] = argv
            return (
                "100_0|FAILED|2026-07-30T11:00:00|93:0\n"
                "100_1|CANCELLED|2026-07-30T11:00:05|0:0\n"  # non-zero task, ignored
                "200_0|RUNNING|Unknown|0:0\n"  # still running, skipped
                "300_[0-3%3]|PENDING|Unknown|0:0\n"  # pending array, skipped
                "400|COMPLETED|2026-07-30T10:00:00|0:0\n"  # non-array job, task ""
            )

        platform = SlurmPlatform()
        monkeypatch.setattr(platform, "_run", fake_run)
        endings = platform.recent_endings("chain", 3600)
        # 100 (task-0 ended) and 400 (non-array), each once; 200/300 excluded.
        assert [gen for gen, _ in endings] == ["400", "100"]
        assert dict(endings)["100"].exit_code == 93
        # Guard the field name that a real cluster rejects: JobID, never ArrayJobID.
        assert "-o" in captured["argv"]
        fields = captured["argv"][captured["argv"].index("-o") + 1]
        assert fields.startswith("JobID") and "ArrayJobID" not in fields


class TestRunner:
    class BlindPlatform(NullPlatform):
        name = "slurm"

        def list_generations(self, job_name):
            raise PlatformError("squeue timed out")

    def test_blind_pass_is_degraded_and_sends_no_heartbeat(self, tmp_path, monkeypatch):
        beats = []
        monkeypatch.setattr(runner.sinks, "heartbeat", lambda url: beats.append(url))
        config = Config(
            job_name="chain",
            work_dir=str(tmp_path),
            state_dir=str(tmp_path / "state"),
            heartbeat_url="https://example.invalid/beat",
        )
        result = runner.run_once(config, self.BlindPlatform())
        assert result.degraded and result.exit_code == 1
        assert beats == []  # a blind watcher must not look healthy

    def test_healthy_pass_heartbeats(self, tmp_path, monkeypatch):
        beats = []
        monkeypatch.setattr(runner.sinks, "heartbeat", lambda url: beats.append(url))
        config = Config(
            job_name="chain",
            work_dir=str(tmp_path),
            state_dir=str(tmp_path / "state"),
            heartbeat_url="https://example.invalid/beat",
        )
        result = runner.run_once(config, NullPlatform())
        assert not result.degraded and result.exit_code == 0
        assert beats == ["https://example.invalid/beat"]

    def test_dry_run_takes_no_action(self, tmp_path):
        config = Config(job_name="chain", state_dir=str(tmp_path), dry_run=True)
        cancelled = []

        class RecordingPlatform(NullPlatform):
            name = "slurm"

            def list_generations(self, job_name):
                return [
                    types.ChainGeneration(
                        "100",
                        (types.TaskInfo(1, "PENDING"), types.TaskInfo(2, "PENDING")),
                    )
                ]

            def terminal_info(self, gen_id, task):
                return types.TaskInfo(task=0, state="FAILED", end_time=ago(hours=1))

            def cancel_pending(self, gen_id):
                cancelled.append(gen_id)
                return True

        result = runner.run_once(config, RecordingPlatform())
        assert any(f.detector == "orphaned_generation" for f in result.findings)
        assert cancelled == []
        assert result.actions_taken and result.actions_taken[0].startswith("[dry-run]")

    def test_action_is_applied_in_owner_mode(self, tmp_path):
        config = Config(job_name="chain", state_dir=str(tmp_path), observe_only=False)
        cancelled = []

        class RecordingPlatform(NullPlatform):
            name = "slurm"

            def list_generations(self, job_name):
                return [types.ChainGeneration("100", (types.TaskInfo(1, "PENDING"),))]

            def terminal_info(self, gen_id, task):
                return types.TaskInfo(task=0, state="NODE_FAIL", end_time=ago(hours=1))

            def cancel_pending(self, gen_id):
                cancelled.append(gen_id)
                return True

        runner.run_once(config, RecordingPlatform())
        assert cancelled == ["100"]

    def test_observe_only_reports_but_does_not_act(self, tmp_path):
        """SRE mode: the orphaned-generation finding still fires (and would page), but the
        watcher never touches jobs it may not own."""
        cancelled = []

        class RecordingPlatform(NullPlatform):
            name = "slurm"

            def list_generations(self, job_name):
                return [types.ChainGeneration("100", (types.TaskInfo(1, "PENDING"),))]

            def terminal_info(self, gen_id, task):
                return types.TaskInfo(task=0, state="NODE_FAIL", end_time=ago(hours=1))

            def cancel_pending(self, gen_id):
                cancelled.append(gen_id)  # must never be reached
                return True

        # observe-only still PAGES (that is the point -- notify the owner); dry-run would not.
        paged = []

        class RecordingSink:
            name = "pagerduty"

            def emit(self, finding):
                paged.append(finding.key)
                return True

        config = Config(job_name="chain", state_dir=str(tmp_path), observe_only=True)
        sinks = [runner.sinks.LogSink(), RecordingSink()]
        result = runner.run_once(config, RecordingPlatform(), sink_list=sinks)
        assert cancelled == []  # no action taken
        assert any(f.detector == "orphaned_generation" for f in result.findings)  # still reported
        assert any("orphan" in k for k in paged)  # and the owner was paged
        assert result.actions_taken and result.actions_taken[0].startswith("[observe-only]")
        assert not result.degraded and result.exit_code == 0

    def test_detectors_skip_when_their_source_is_missing(self, config):
        """A failed squeue must never read as 'no generations are queued'."""
        snap = make_snapshot(chain_expected=True, capabilities=(types.CAP_CYCLES,))
        assert [f.detector for f in detectors.run(snap, config)] == []

    def test_sacct_failure_degrades_without_crashing_or_heartbeating(self, tmp_path, monkeypatch):
        """squeue works but sacct fails mid-pass: the pass must not crash, must be
        degraded (no heartbeat), and must still report the blindness."""
        beats = []
        monkeypatch.setattr(runner.sinks, "heartbeat", lambda url: beats.append(url))

        class SacctBlindPlatform(NullPlatform):
            name = "slurm"

            def list_generations(self, job_name):
                # task 0 gone + a pending spare -> the code path that queries terminal_info
                return [types.ChainGeneration("100", (types.TaskInfo(1, "PENDING"),))]

            def terminal_info(self, gen_id, task):
                raise PlatformError("sacct timed out")

        config = Config(
            job_name="chain",
            work_dir=str(tmp_path),
            state_dir=str(tmp_path / "state"),
            heartbeat_url="https://example.invalid/beat",
        )
        result = runner.run_once(config, SacctBlindPlatform())  # must not raise
        assert result.degraded and beats == []
        assert any(f.detector == "observer" for f in result.findings)

    def test_one_sacct_query_failing_does_not_suppress_the_others(self, tmp_path, monkeypatch):
        """A terminal_info failure must not skip recent_endings: an exit-93 verdict with a
        queued successor still has to fire chain_not_cancelled, alongside the degraded flag."""
        monkeypatch.setattr(runner.sinks, "heartbeat", lambda url: None)

        class PartialBlindPlatform(NullPlatform):
            name = "slurm"

            def list_generations(self, job_name):
                # gen A: task 0 gone + pending spare -> terminal_info(A) is queried (fails);
                # gen B: still queued -> a successor for chain_not_cancelled.
                return [
                    types.ChainGeneration("A", (types.TaskInfo(1, "PENDING"),)),
                    types.ChainGeneration("B", (types.TaskInfo(0, "PENDING"),)),
                ]

            def terminal_info(self, gen_id, task):
                raise PlatformError("sacct -j timed out")

            def recent_endings(self, job_name, since_seconds):
                # generation C ended 93 while B is still queued
                return [("C", types.TaskInfo(0, "FAILED", exit_code=93, end_time=ago(hours=1)))]

        config = Config(job_name="chain", work_dir=str(tmp_path), state_dir=str(tmp_path / "s"))
        result = runner.run_once(config, PartialBlindPlatform())
        detectors_fired = {f.detector for f in result.findings}
        assert "chain_not_cancelled" in detectors_fired  # recent_endings not suppressed
        assert "observer" in detectors_fired and result.degraded  # still flagged blind

    def test_failed_pager_is_retried_not_cooled_down(self, tmp_path):
        """A due finding whose pager rejects it must not enter the cooldown -- the next
        pass has to retry, or a real condition is suppressed for the whole window."""

        class FailingSink:
            name = "pagerduty"

            def emit(self, finding):
                return False  # pager rejected / timed out

        class OneCriticalPlatform(NullPlatform):
            name = "slurm"

            def list_generations(self, job_name):
                return []  # empty + chain marker -> chain_exhausted (critical)

        config = Config(
            job_name="chain",
            work_dir=str(tmp_path),
            state_dir=str(tmp_path / "s"),
            expect_file=str(tmp_path / "expect_chain"),  # never the real ~/ marker
        )
        with open(config.expect_file, "w") as fh:  # arm chain_exhausted
            fh.write("chain\n")
        sinks = [runner.sinks.LogSink(), FailingSink()]
        runner.run_once(config, OneCriticalPlatform(), sink_list=sinks)
        _, alerts = persistence.load(config.state_file)
        # per-(finding, sink) key stays unset for the sink that rejected -> retried next pass
        assert "nvrx-chain-exhausted-chain\x00pagerduty" not in alerts

    def test_cooldown_is_per_sink_not_shared(self, tmp_path):
        """One sink accepting must not force a re-page to it while another still fails: the
        accepting sink enters its own cooldown; the failing one keeps retrying."""

        class GoodSink:
            name = "pagerduty"

            def __init__(self):
                self.calls = 0

            def emit(self, finding):
                self.calls += 1
                return True

        class BadSink:
            name = "webhook"

            def emit(self, finding):
                return False  # never accepts

        class OneCriticalPlatform(NullPlatform):
            name = "slurm"

            def list_generations(self, job_name):
                return []  # empty + chain marker -> chain_exhausted (critical)

        config = Config(
            job_name="chain",
            work_dir=str(tmp_path),
            state_dir=str(tmp_path / "s"),
            expect_file=str(tmp_path / "expect_chain"),
        )
        with open(config.expect_file, "w") as fh:
            fh.write("chain\n")
        good = GoodSink()
        sinks = [good, BadSink()]
        runner.run_once(config, OneCriticalPlatform(), sink_list=sinks)
        runner.run_once(config, OneCriticalPlatform(), sink_list=sinks)
        _, alerts = persistence.load(config.state_file)
        assert good.calls == 1  # accepted once, then cooled down -> no duplicate page
        assert "nvrx-chain-exhausted-chain\x00pagerduty" in alerts  # good sink cooled down
        assert "nvrx-chain-exhausted-chain\x00webhook" not in alerts  # bad sink still retries


class TestCliJobIdResolution:
    def _wire(self, monkeypatch, tmp_path, desc, run_once_capture):
        from nvrx_watch import __main__ as cli
        from nvrx_watch import platform as plat_mod
        from nvrx_watch import runner as run_mod
        from nvrx_watch.platform import NullPlatform

        class FakePlatform(NullPlatform):
            name = "slurm"

            def describe_job(self, job_id):
                return desc

        class Result:
            exit_code = 0

        monkeypatch.setattr(plat_mod, "create", lambda *a, **k: FakePlatform())
        monkeypatch.setattr(run_mod, "run_once", run_once_capture(Result()))
        return cli

    def test_job_id_fills_name_and_owner(self, monkeypatch, tmp_path):
        from nvrx_watch.platform import JobDescription

        captured = {}
        cli = self._wire(
            monkeypatch,
            tmp_path,
            JobDescription(job_name="team_run", user="alice"),
            lambda result: (
                lambda config, plat, sink_list=None: captured.update(c=config) or result
            ),
        )
        assert cli.main(["12345", "--state-dir", str(tmp_path / "st")]) == 0
        c = captured["c"]
        assert c.job_id == "12345" and c.job_name == "team_run" and c.user == "alice"

    def test_explicit_flags_win_over_resolution(self, monkeypatch, tmp_path):
        from nvrx_watch.platform import JobDescription

        captured = {}
        cli = self._wire(
            monkeypatch,
            tmp_path,
            JobDescription(job_name="team_run", user="alice"),
            lambda result: (
                lambda config, plat, sink_list=None: captured.update(c=config) or result
            ),
        )
        cli.main(["12345", "--job-name", "override", "--state-dir", str(tmp_path / "st")])
        assert captured["c"].job_name == "override"  # CLI wins over the resolved name

    def test_ft_paths_resolved_from_batch_script(self, monkeypatch, tmp_path):
        """The cycle-info glob and checkpoint file come from the job's batch script (any
        InJob sbatch), not from a work-dir convention."""
        from nvrx_watch import __main__ as cli
        from nvrx_watch import platform as plat_mod
        from nvrx_watch import runner as run_mod
        from nvrx_watch.platform import JobDescription, NullPlatform

        script = (
            'LOG_ROOT="/scratch/x"\n'
            'NVRX_DIR="${LOG_ROOT}/nvrx"\n'
            "args=\" --ft-cycle-info-dir=${NVRX_DIR}/${SLURM_ARRAY_JOB_ID}/cycle_infos "
            "--ft-checkpoint-iteration-file=${LOG_ROOT}/checkpoints/latest_checkpointed_iteration.txt \"\n"
        )

        class FakePlatform(NullPlatform):
            name = "slurm"

            def describe_job(self, job_id):
                return JobDescription(job_name="team_run", user="alice")

            def list_generations(self, job_name):
                return [types.ChainGeneration("G", (types.TaskInfo(0, "RUNNING"),))]

            def batch_script(self, job_id):
                return script

        class Result:
            exit_code = 0

        captured = {}
        monkeypatch.setattr(plat_mod, "create", lambda *a, **k: FakePlatform())
        monkeypatch.setattr(
            run_mod,
            "run_once",
            lambda config, plat, sink_list=None: captured.update(c=config) or Result(),
        )
        cli.main(["12345", "--state-dir", str(tmp_path / "st")])
        c = captured["c"]
        assert c.resolved_cycle_info_glob == "/scratch/x/nvrx/*/cycle_infos/cycle_info.*"
        assert (
            c.resolved_checkpoint_file == "/scratch/x/checkpoints/latest_checkpointed_iteration.txt"
        )

    def test_ft_paths_recovered_from_a_live_generation(self, monkeypatch, tmp_path):
        """The bootstrap id has aged out (no batch script), so ft paths are read from
        whatever generation is live now -- the script is identical across the chain."""
        from nvrx_watch import __main__ as cli
        from nvrx_watch import platform as plat_mod
        from nvrx_watch import runner as run_mod
        from nvrx_watch.platform import JobDescription, NullPlatform

        script = (
            'LOG_ROOT="/scratch/x"\n'
            " --ft-cycle-info-dir=${LOG_ROOT}/nvrx/${SLURM_ARRAY_JOB_ID}/cycle_infos \n"
        )

        class FakePlatform(NullPlatform):
            name = "slurm"

            def describe_job(self, job_id):
                return JobDescription(job_name="team_run", user="alice")

            def list_generations(self, job_name):
                return [types.ChainGeneration("LIVE_GEN", (types.TaskInfo(0, "RUNNING"),))]

            def batch_script(self, job_id):
                return script if job_id == "LIVE_GEN" else None  # OLD aged out

        class Result:
            exit_code = 0

        captured = {}
        monkeypatch.setattr(plat_mod, "create", lambda *a, **k: FakePlatform())
        monkeypatch.setattr(
            run_mod,
            "run_once",
            lambda config, plat, sink_list=None: captured.update(c=config) or Result(),
        )
        cli.main(["OLD", "--state-dir", str(tmp_path / "st")])
        assert (
            captured["c"].cycle_info_glob == "/scratch/x/nvrx/*/cycle_infos/cycle_info.*"
        )  # recovered from LIVE_GEN's script


class TestPackagingConstraints:
    def test_stdlib_only(self):
        """nvrx-watch runs on a login node, outside the training container, where
        nvidia_resiliency_ext need not be installed. Any third-party import -- including
        NVRx itself -- breaks that, so the constraint is checked rather than trusted."""
        import ast

        imported = set()
        for path in sorted(glob.glob(os.path.join(WATCH_DIR, "nvrx_watch", "*.py"))):
            with open(path) as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])

        allowed = sys.stdlib_module_names | {"__future__"}
        assert not (imported - allowed), (
            "nvrx_watch must import stdlib only; found "
            f"{sorted(imported - allowed)}. Copying the directory to a login node must "
            "stay sufficient to run it."
        )


class TestConfig:
    def test_env_overrides_defaults(self):
        from nvrx_watch import config as config_module

        loaded = config_module.load(
            env={
                "NVRX_WATCH_JOB_NAME": "burn",
                "NVRX_WATCH_GRACE": "300",
                "NVRX_WATCH_DRY_RUN": "1",
                "NVRX_WATCH_DISABLE": "suspect_node,cycle_stalled",
            }
        )
        assert loaded.job_name == "burn" and loaded.grace == 300.0 and loaded.dry_run
        assert loaded.disable == ("suspect_node", "cycle_stalled")

    def test_work_dir_layout_matches_the_sbatch(self):
        loaded = Config(work_dir="/scratch/run")
        assert loaded.resolved_cycle_info_glob == ("/scratch/run/nvrx/*/cycle_infos/cycle_info.*")
        assert loaded.resolved_checkpoint_file == (
            "/scratch/run/checkpoints/latest_checkpointed_iteration.txt"
        )

    def test_inherits_sbatch_work_dir_and_job_name(self):
        from nvrx_watch import config as config_module

        loaded = config_module.load(env={"NVRX_WORK_DIR": "/scratch/run", "NVRX_JOB_NAME": "burn"})
        assert loaded.work_dir == "/scratch/run" and loaded.job_name == "burn"

    def test_watch_prefixed_env_wins_over_sbatch_vars(self):
        from nvrx_watch import config as config_module

        loaded = config_module.load(env={"NVRX_JOB_NAME": "sbatch", "NVRX_WATCH_JOB_NAME": "watch"})
        assert loaded.job_name == "watch"

    def test_disable_skips_detectors(self):
        loaded = Config(disable=("restart_storm",))
        assert "restart_storm" not in [d.name for d in detectors.enabled(loaded)]
