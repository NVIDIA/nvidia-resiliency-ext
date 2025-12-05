# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

import pytest

from nvidia_resiliency_ext import fault_tolerance
from nvidia_resiliency_ext.fault_tolerance.progress_tracker import TrainingProgressTracker

WORLD_SIZE = 4
DEFAULT_TIMEOUT = 90


@pytest.fixture
def tmp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def _get_func_name():
    return sys._getframe(1).f_code.co_name


def _run_launcher(cmd_to_run, timeout):
    try:
        proc = subprocess.Popen(
            cmd_to_run,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        return proc.returncode, stdout
    except subprocess.TimeoutExpired:
        with contextlib.suppress():
            proc.kill()
            proc.wait()
        assert False, f"ft_launcher was still running after {timeout} seconds"


def _save_ft_cfg(cfg, dirpath):
    cfg_path = os.path.join(dirpath, "_tmp_ft_cfg.yaml")
    cfg.to_yaml_file(cfg_path)
    return cfg_path


def _get_util_script_path():
    return os.path.join(os.path.dirname(__file__), "_launcher_test_util.py")


def test_rank_not_send_initial_hb(tmp_dir):
    # If one rank does not send initial heartbeat,
    # FT should terminate the rank, and launcher should kill all other ranks
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg.domain_id_from_node_name = False
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)
    cmd_to_run = f"{_get_util_script_path()} --scenario={_get_func_name()} --which_rank=1"
    launcher_cmd = (
        "ft_launcher --monitor-interval=1"
        f" --ft-cfg-path={ft_cfg_path} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert "ALL RANKS STARTED" in output
    assert "RANK IS SKIPPING INITIAL HB" in output
    assert ret_code == 1


def test_rank_failed(tmp_dir):
    # If one rank failed (returns non-zero exit code),
    # launcher should kill other ranks
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)
    cmd_to_run = f"{_get_util_script_path()} --scenario={_get_func_name()} --which_rank=1"
    launcher_cmd = (
        "ft_launcher --monitor-interval=1"
        f" --ft-cfg-path={ft_cfg_path} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert "ALL RANKS STARTED" in output
    assert "RANK FAILED" in output
    assert ret_code == 1


def test_ranks_exit_gracefully(tmp_dir):
    # All ranks exit gracefully, there should be no error
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)
    cmd_to_run = f"{_get_util_script_path()} --scenario={_get_func_name()}"
    launcher_cmd = (
        "ft_launcher --monitor-interval=1"
        f" --ft-cfg-path={ft_cfg_path} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert "ALL RANKS STARTED" in output
    assert "RANK EXITS GRACEFULLY" in output
    assert ret_code == 0


def test_launcher_sigterm_graceful_exit(tmp_dir):
    # Simulated preemption:
    # Launcher get SIGTERM, ranks exit gracefully with code 0.
    # No error should be returned by the launcher.
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)
    cmd_to_run = f"{_get_util_script_path()} --scenario={_get_func_name()} --term_handler=return0"
    launcher_cmd = (
        "ft_launcher --monitor-interval=1"
        f" --ft-cfg-path={ft_cfg_path} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert "ALL RANKS STARTED" in output
    assert "SIGTERM SENT TO LAUNCHER" in output
    assert "RANK GOT SIGTERM: RETURN0" in output
    assert ret_code == 0


def test_launcher_sigterm_ignored(tmp_dir):
    # Simulated preemption:
    # Launcher get SIGTERM, ranks do not exit
    # FT launcher should forcefuly kill all ranks after `--term-timeout`
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)
    cmd_to_run = f"{_get_util_script_path()} --scenario={_get_func_name()} --term_handler=ignore"
    launcher_cmd = (
        "ft_launcher --term-timeout=5 --monitor-interval=1"
        f" --ft-cfg-path={ft_cfg_path} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert "ALL RANKS STARTED" in output
    assert "SIGTERM SENT TO LAUNCHER" in output
    assert "RANK GOT SIGTERM: IGNORED" in output
    assert ret_code == 1


def test_ranks_restart(tmp_dir):
    # Run 0 is `test_rank_not_send_initial_hb`
    # Run 1 is `test_rank_failed`
    # Run 2 is `test_ranks_exit_gracefully`
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg.workload_check_interval = 1.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)
    cmd_to_run = f"{_get_util_script_path()} --scenario={_get_func_name()} --tmp_dir={tmp_dir}"
    launcher_cmd = (
        "ft_launcher --max-restarts=2 --monitor-interval=1"
        f" --ft-cfg-path={ft_cfg_path} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert "ALL RANKS STARTED" in output
    assert "RESTART #0" in output
    assert "RANK IS SKIPPING INITIAL HB" in output
    assert "RESTART #1" in output
    assert "RANK FAILED" in output
    assert "RESTART #2" in output
    assert "RANK EXITS GRACEFULLY" in output
    assert ret_code == 0


def test_missing_cfg(tmp_dir):
    # Empty config file, cant be parsed
    empty_ft_cfg_path = os.path.join(tmp_dir, "_empty_ft_cfg.yaml")
    with open(empty_ft_cfg_path, 'a'):
        pass  # touch file
    # Empty config file again, But this time there are FT args in CLI, so should be fine
    cmd_to_run = f"{_get_util_script_path()} --scenario=test_ranks_exit_gracefully"
    launcher_cmd = (
        "ft_launcher --monitor-interval=1"
        f" --ft-cfg-path={empty_ft_cfg_path} --nproc-per-node={WORLD_SIZE} --ft-rank-heartbeat-timeout=1.0"
        f" {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert ret_code == 0
    # Invalid config file path - should fail despite FT args specified via CLI
    cmd_to_run = f"{_get_util_script_path()} --scenario=test_ranks_exit_gracefully"
    launcher_cmd = (
        "ft_launcher --monitor-interval=1"
        " --ft-cfg-path=/not/there.yaml"
        " --ft-rank-heartbeat-timeout=1.0"
        f" --nproc-per-node={WORLD_SIZE}"
        f" {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert ret_code != 0


def test_config_provided_via_cli(tmp_dir):
    # Check if FT args passed via CLI were propagated to the FT monitor process
    ft_params_str = (
        "--ft-workload-check-interval=321.0"
        " --ft-initial-rank-heartbeat-timeout=1.0"
        " --ft-rank-heartbeat-timeout=2.0"
        " --ft-rank-termination-signal=SIGUSR2"
        " --ft-log-level=WARNING"
    )
    cmd_to_run = f"{_get_util_script_path()} --scenario=dump_cfg --tmp_dir={tmp_dir}"
    launcher_cmd = "ft_launcher" f" {ft_params_str} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert ret_code == 0

    dumped_ft_cfg_path = os.path.join(tmp_dir, "cfg_dump.yaml")
    assert os.path.exists(dumped_ft_cfg_path)

    restored_ft_conf = fault_tolerance.FaultToleranceConfig.from_yaml_file(dumped_ft_cfg_path)
    assert restored_ft_conf.workload_check_interval == 321.0
    assert restored_ft_conf.initial_rank_heartbeat_timeout == 1.0
    assert restored_ft_conf.rank_heartbeat_timeout == 2.0
    assert restored_ft_conf.rank_termination_signal == signal.SIGUSR2
    assert restored_ft_conf.log_level == logging.WARNING


def test_config_provided_via_cli_overwrites_yaml(tmp_dir):
    # Check if FT args passed via CLI were propagated to the FT monitor process
    # Args provided via CLI should overwrite the ones from the config file
    base_cfg = fault_tolerance.FaultToleranceConfig(
        workload_check_interval=321.0,
        initial_rank_heartbeat_timeout=111.0,
        rank_heartbeat_timeout=222.0,
        rank_termination_signal=signal.SIGTSTP,
        log_level=logging.INFO,
    )
    ft_cfg_path = os.path.join(tmp_dir, "ft_cfg.yaml")
    base_cfg.to_yaml_file(ft_cfg_path)

    ft_params_str = (
        "--ft-rank-heartbeat-timeout=123.0"
        " --ft-safety-factor=7.7"
        " --ft-rank-termination-signal=SIGUSR1"
        " --ft-log-level=CRITICAL"
    )
    cmd_to_run = f"{_get_util_script_path()} --scenario=dump_cfg --tmp_dir={tmp_dir}"
    launcher_cmd = (
        "ft_launcher"
        f" {ft_params_str} --ft-cfg-path={ft_cfg_path} --nproc-per-node={WORLD_SIZE} {cmd_to_run}"
    )
    ret_code, output = _run_launcher(launcher_cmd, DEFAULT_TIMEOUT)
    assert ret_code == 0

    dumped_ft_cfg_path = os.path.join(tmp_dir, "cfg_dump.yaml")
    assert os.path.exists(dumped_ft_cfg_path)

    restored_ft_conf = fault_tolerance.FaultToleranceConfig.from_yaml_file(dumped_ft_cfg_path)
    assert restored_ft_conf.workload_check_interval == 321.0
    assert restored_ft_conf.initial_rank_heartbeat_timeout == 111.0
    assert restored_ft_conf.rank_heartbeat_timeout == 123.0
    assert restored_ft_conf.safety_factor == 7.7
    assert restored_ft_conf.rank_termination_signal == signal.SIGUSR1
    assert restored_ft_conf.log_level == logging.CRITICAL


# ==============================================================================
# Unit tests for launcher iteration aggregation logic
# ==============================================================================


class MockLauncher:
    """Mock launcher with iteration aggregation logic for testing."""

    def __init__(self, progress_tracker):
        self._rank_iterations = {}
        self._progress_tracker = progress_tracker

    def _update_progress_iteration(self, local_rank: int, iteration: int):
        """Update iteration for a specific rank and aggregate using MIN strategy.

        This is extracted from the actual launcher.py for testing.
        """
        # Update this rank's max iteration
        self._rank_iterations[local_rank] = max(self._rank_iterations.get(local_rank, 0), iteration)

        # Use minimum across all ranks (most conservative - slowest rank determines progress)
        min_iteration = min(self._rank_iterations.values()) if self._rank_iterations else 0
        self._progress_tracker.update_iteration(min_iteration)


class TestLauncherIterationAggregation(unittest.TestCase):
    """Unit tests for launcher's MIN aggregation strategy across ranks."""

    def setUp(self):
        """Set up test fixtures."""
        self.progress_tracker = TrainingProgressTracker(
            min_progress_iterations=200,
            max_no_progress_restarts=3,
        )
        self.launcher = MockLauncher(self.progress_tracker)

    def test_single_rank_iteration_update(self):
        """Test iteration update with single rank."""
        self.launcher._update_progress_iteration(0, 100)

        # Progress tracker should have the iteration from rank 0
        self.assertEqual(self.progress_tracker.current_max_iteration, 100)

        # Update to higher iteration
        self.launcher._update_progress_iteration(0, 200)
        self.assertEqual(self.progress_tracker.current_max_iteration, 200)

    def test_multiple_ranks_min_aggregation(self):
        """Test that MIN aggregation computes minimum across all ranks."""
        # All ranks start together
        self.launcher._update_progress_iteration(0, 100)
        self.launcher._update_progress_iteration(1, 95)
        self.launcher._update_progress_iteration(2, 105)

        # Verify launcher correctly tracks each rank's max
        self.assertEqual(self.launcher._rank_iterations[0], 100)
        self.assertEqual(self.launcher._rank_iterations[1], 95)
        self.assertEqual(self.launcher._rank_iterations[2], 105)

        # Progress tracker receives MAX of all MINs seen (defensive against out-of-order)
        # First update with rank 0 (100), then updates with lower MINs are ignored by tracker
        self.assertEqual(self.progress_tracker.current_max_iteration, 100)

    def test_min_aggregation_monotonic_progress(self):
        """Test MIN aggregation with monotonic progress (realistic scenario)."""
        # All ranks start at 0 and progress together
        self.launcher._update_progress_iteration(0, 10)
        self.launcher._update_progress_iteration(1, 10)
        self.launcher._update_progress_iteration(2, 10)
        # After all 3 ranks report 10, MIN = 10, tracker = 10
        self.assertEqual(self.progress_tracker.current_max_iteration, 10)

        # All ranks progress, rank 1 is slightly slower
        self.launcher._update_progress_iteration(0, 50)  # MIN now = 10 (ranks 1,2 still at 10)
        self.launcher._update_progress_iteration(1, 45)  # MIN now = 10 (rank 2 still at 10)
        self.launcher._update_progress_iteration(2, 48)  # MIN now = 45 (slowest)
        # After all updates, MIN = 45, tracker = 45
        self.assertEqual(self.progress_tracker.current_max_iteration, 45)

        # All ranks continue progressing
        self.launcher._update_progress_iteration(0, 100)  # MIN = 45 (ranks 1,2 still at 45,48)
        self.launcher._update_progress_iteration(1, 95)  # MIN = 48 (rank 2 still at 48)
        self.launcher._update_progress_iteration(2, 98)  # MIN = 95 (slowest)
        # After all updates, MIN = 95, tracker = 95
        self.assertEqual(self.progress_tracker.current_max_iteration, 95)

    def test_rank_iterations_tracking(self):
        """Test that launcher correctly tracks per-rank iterations."""
        # Update ranks at different iterations
        self.launcher._update_progress_iteration(0, 100)
        self.launcher._update_progress_iteration(1, 95)
        self.launcher._update_progress_iteration(2, 105)

        # Verify each rank's iteration is tracked
        self.assertEqual(self.launcher._rank_iterations[0], 100)
        self.assertEqual(self.launcher._rank_iterations[1], 95)
        self.assertEqual(self.launcher._rank_iterations[2], 105)

    def test_rank_max_iteration_tracking(self):
        """Test that each rank tracks its max iteration correctly."""
        # Rank 0 reports iterations in non-monotonic order (e.g., out of order messages)
        self.launcher._update_progress_iteration(0, 100)
        self.assertEqual(self.launcher._rank_iterations[0], 100)

        # Lower value shouldn't decrease the max for this rank
        self.launcher._update_progress_iteration(0, 50)
        self.assertEqual(self.launcher._rank_iterations[0], 100)

        # Higher value should update
        self.launcher._update_progress_iteration(0, 150)
        self.assertEqual(self.launcher._rank_iterations[0], 150)

    def test_empty_rank_iterations(self):
        """Test behavior when no ranks have reported yet."""
        # With no rank updates, min should be 0
        self.assertEqual(len(self.launcher._rank_iterations), 0)
        self.assertEqual(self.progress_tracker.current_max_iteration, 0)

    def test_8_ranks_realistic_scenario(self):
        """Test realistic 8-rank training scenario."""
        # All 8 ranks start at 0 and progress together
        for rank in range(8):
            self.launcher._update_progress_iteration(rank, 100)
        # All at 100, MIN = 100, tracker = 100
        self.assertEqual(self.progress_tracker.current_max_iteration, 100)

        # All progress, with slight variations
        iterations = [500, 498, 502, 495, 501, 499, 503, 497]
        for rank, iteration in enumerate(iterations):
            self.launcher._update_progress_iteration(rank, iteration)
        # After all ranks update, MIN = 495 (rank 3 is slowest)
        # Tracker receives MIN at each update, keeps max = 495 (final MIN)
        self.assertEqual(self.progress_tracker.current_max_iteration, 495)

    def test_straggler_scenario(self):
        """Test scenario with one straggling rank."""
        # All ranks start at 100
        for rank in range(4):
            self.launcher._update_progress_iteration(rank, 100)
        self.assertEqual(self.progress_tracker.current_max_iteration, 100)

        # 3 ranks progress to 200, 1 straggler stays at 110
        self.launcher._update_progress_iteration(0, 200)  # MIN = 100 (others still at 100)
        self.launcher._update_progress_iteration(1, 200)  # MIN = 100 (ranks 2,3 still at 100)
        self.launcher._update_progress_iteration(2, 200)  # MIN = 100 (rank 3 still at 100)
        self.launcher._update_progress_iteration(3, 110)  # MIN = 110 (straggler)
        # Tracker keeps max of MINs seen = 110
        self.assertEqual(self.progress_tracker.current_max_iteration, 110)

        # Verify MIN is 110 (straggler)
        min_iter = min(self.launcher._rank_iterations.values())
        self.assertEqual(min_iter, 110)

    def test_zero_iteration_handling(self):
        """Test handling of zero iterations (training start)."""
        # Ranks start at iteration 0 or 1
        self.launcher._update_progress_iteration(0, 0)
        self.launcher._update_progress_iteration(1, 1)
        self.launcher._update_progress_iteration(2, 0)

        # MIN should be 0
        self.assertEqual(self.progress_tracker.current_max_iteration, 0)

    def test_integration_with_progress_analysis(self):
        """Test integration of MIN aggregation with progress analysis."""
        # Simulate a training cycle with 4 ranks
        # Cycle 0: Initial run - all ranks start at 0 and progress to ~500
        for rank in range(4):
            self.launcher._update_progress_iteration(rank, 500)
        self.assertEqual(self.progress_tracker.current_max_iteration, 500)

        self.progress_tracker.analyze_previous_cycle()

        # Cycle 1: All ranks make good progress to ~800
        for rank in range(4):
            self.launcher._update_progress_iteration(rank, 800)
        # Progress = 800 - 500 = 300 (> 200 threshold)
        self.assertEqual(self.progress_tracker.current_max_iteration, 800)

        self.progress_tracker.analyze_previous_cycle()
        self.assertEqual(self.progress_tracker.no_progress_count, 0)

    def test_progress_analysis_with_iteration_tracking(self):
        """Test that progress analysis uses tracker's max iteration correctly."""
        # Cycle 0: All ranks reach 100
        for rank in range(4):
            self.launcher._update_progress_iteration(rank, 100)
        self.assertEqual(self.progress_tracker.current_max_iteration, 100)
        self.progress_tracker.analyze_previous_cycle()

        # Cycle 1: All ranks reach 350 (progress = 250 >= 200)
        for rank in range(4):
            self.launcher._update_progress_iteration(rank, 350)
        self.assertEqual(self.progress_tracker.current_max_iteration, 350)
        self.progress_tracker.analyze_previous_cycle()

        # Should detect good progress
        self.assertEqual(self.progress_tracker.no_progress_count, 0)
