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
from unittest.mock import MagicMock, patch

import pytest
from torch.distributed.elastic.rendezvous.api import RendezvousGracefulExitError

from nvidia_resiliency_ext import fault_tolerance
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig

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
# Unit tests for launcher cycle_info_writer interaction
# ==============================================================================


def _make_agent_spec(rdzv_round=1):
    """Minimal WorkerSpec-like object for testing cycle_info_writer interaction."""
    spec = MagicMock()
    spec.rdzv_handler = MagicMock()
    spec.rdzv_handler.round.return_value = rdzv_round
    spec.rdzv_handler.get_last_rendezvous_participant_addrs.return_value = ["node001", "node002"]
    spec.rdzv_handler.get_last_rendezvous_standby_participant_addrs.return_value = ["node003"]
    spec.max_restarts = 3
    return spec


class TestLauncherCycleInfoWriterInteraction(unittest.TestCase):
    """Unit tests for launcher's interaction with CycleInfoWriter."""

    def setUp(self):
        """Set up test fixtures."""
        self.spec = _make_agent_spec(rdzv_round=1)
        self.fault_tol_cfg = FaultToleranceConfig()
        self.logs_specs = MagicMock()
        self.logs_specs.get_cycle_log_file.return_value = "/path/to/cycle_0.log"

    def test_on_cycle_end_when_writer_is_none(self):
        """_on_cycle_end does nothing when cycle_info_writer is None."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        agent = LocalElasticAgent(
            spec=self.spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
            cycle_info_writer=None,
        )
        # Should not raise; no writer to call
        agent._on_cycle_end()

    def test_on_cycle_end_calls_update_cycle_end_with_env_and_restart_count(self):
        """_on_cycle_end calls writer.update_cycle_end with job_id, attempt_index, cycle_number, cycle_end_time."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        writer = MagicMock()
        # rdzv_round=1 -> _get_global_restart_count() = max(0, 1-1) = 0
        spec = _make_agent_spec(rdzv_round=1)
        with patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_RESTART_CNT": "0",
            },
            clear=False,
        ):
            agent = LocalElasticAgent(
                spec=spec,
                fault_tol_cfg=self.fault_tol_cfg,
                logs_specs=self.logs_specs,
                cycle_info_writer=writer,
            )
            with patch(
                "nvidia_resiliency_ext.fault_tolerance.launcher.utc_iso_now",
                return_value="2024-01-01T12:00:00Z",
            ):
                agent._on_cycle_end()

        writer.update_cycle_end.assert_called_once()
        call_kw = writer.update_cycle_end.call_args[1]
        self.assertEqual(call_kw["job_id"], "12345")
        self.assertEqual(call_kw["attempt_index"], 0)
        self.assertEqual(call_kw["cycle_number"], 0)
        self.assertEqual(call_kw["cycle_end_time"], "2024-01-01T12:00:00Z")

    def test_on_cycle_end_uses_slurm_array_job_id_when_set(self):
        """_on_cycle_end uses SLURM_ARRAY_JOB_ID over SLURM_JOB_ID when present."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        writer = MagicMock()
        spec = _make_agent_spec(rdzv_round=2)  # restart_count = 1
        with patch.dict(
            os.environ,
            {
                "SLURM_ARRAY_JOB_ID": "array_99",
                "SLURM_JOB_ID": "12345",
                "SLURM_RESTART_CNT": "1",
            },
            clear=False,
        ):
            agent = LocalElasticAgent(
                spec=spec,
                fault_tol_cfg=self.fault_tol_cfg,
                logs_specs=self.logs_specs,
                cycle_info_writer=writer,
            )
            with patch(
                "nvidia_resiliency_ext.fault_tolerance.launcher.utc_iso_now",
                return_value="2024-01-01T12:00:00Z",
            ):
                agent._on_cycle_end()

        call_kw = writer.update_cycle_end.call_args[1]
        self.assertEqual(call_kw["job_id"], "array_99")
        self.assertEqual(call_kw["attempt_index"], 1)
        self.assertEqual(call_kw["cycle_number"], 1)

    def test_write_cycle_start_info_returns_none_when_writer_is_none(self):
        """_write_cycle_start_info returns None when cycle_info_writer is None."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        agent = LocalElasticAgent(
            spec=self.spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
            cycle_info_writer=None,
        )
        result = agent._write_cycle_start_info(current_cycle=0)
        self.assertIsNone(result)

    def test_write_cycle_start_info_calls_write_cycle_start_and_returns_current_path(self):
        """_write_cycle_start_info calls write_cycle_start and returns get_current_cycle_info_path."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        writer = MagicMock()
        writer.get_current_cycle_info_path.return_value = "/nvrx/cycle_info.job1.current"
        spec = _make_agent_spec(rdzv_round=1)
        with patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "job1", "SLURM_RESTART_CNT": "0"},
            clear=False,
        ):
            agent = LocalElasticAgent(
                spec=spec,
                fault_tol_cfg=self.fault_tol_cfg,
                logs_specs=self.logs_specs,
                cycle_info_writer=writer,
            )
            with patch(
                "nvidia_resiliency_ext.fault_tolerance.launcher.utc_iso_now",
                return_value="2024-01-01T12:00:00Z",
            ):
                with patch(
                    "nvidia_resiliency_ext.fault_tolerance.launcher.hostnames_to_slurm_nodelist",
                    side_effect=["node[001-002]", "node003"],  # active_addrs, then standby_addrs
                ):
                    result = agent._write_cycle_start_info(current_cycle=0)

        writer.write_cycle_start.assert_called_once()
        call_kw = writer.write_cycle_start.call_args[1]
        self.assertEqual(call_kw["job_id"], "job1")
        self.assertEqual(call_kw["attempt_index"], 0)
        self.assertEqual(call_kw["cycle_number"], 0)
        self.assertEqual(call_kw["cycle_start_time"], "2024-01-01T12:00:00Z")
        self.assertEqual(call_kw["cycle_log_file"], "/path/to/cycle_0.log")
        self.assertEqual(call_kw["active_nodes"], "node[001-002]")
        self.assertEqual(call_kw["standby_nodes"], "node003")
        writer.get_current_cycle_info_path.assert_called_once_with("job1")
        self.assertEqual(result, "/nvrx/cycle_info.job1.current")

    def test_write_cycle_start_info_standby_nodes_from_rdzv(self):
        """_write_cycle_start_info passes active and standby node lists from rdzv handler."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        writer = MagicMock()
        writer.get_current_cycle_info_path.return_value = "/nvrx/current"
        spec = _make_agent_spec(rdzv_round=1)
        spec.rdzv_handler.get_last_rendezvous_participant_addrs.return_value = ["host1"]
        spec.rdzv_handler.get_last_rendezvous_standby_participant_addrs.return_value = ["host2"]
        with patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "j", "SLURM_RESTART_CNT": "0"},
            clear=False,
        ):
            agent = LocalElasticAgent(
                spec=spec,
                fault_tol_cfg=self.fault_tol_cfg,
                logs_specs=self.logs_specs,
                cycle_info_writer=writer,
            )
            with patch(
                "nvidia_resiliency_ext.fault_tolerance.launcher.utc_iso_now",
                return_value="2024-01-01T12:00:00Z",
            ):
                with patch(
                    "nvidia_resiliency_ext.fault_tolerance.launcher.hostnames_to_slurm_nodelist",
                    side_effect=lambda addrs: (
                        "active"
                        if addrs == ["host1"]
                        else ("standby" if addrs == ["host2"] else "")
                    ),
                ):
                    agent._write_cycle_start_info(current_cycle=0)

        call_kw = writer.write_cycle_start.call_args[1]
        self.assertEqual(call_kw["active_nodes"], "active")
        self.assertEqual(call_kw["standby_nodes"], "standby")


class TestMaybeExitStandbyOnSuccess(unittest.TestCase):
    """Unit tests for _maybe_exit_standby_on_success (standby detects training success via exit barrier)."""

    def setUp(self):
        self.fault_tol_cfg = FaultToleranceConfig()
        self.logs_specs = MagicMock()

    def _make_agent_standby(self, min_nodes=2, group_rank=2, store_check_returns=False):
        """Create agent configured as standby (group_rank >= min_nodes) with optional store mock."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        spec = _make_agent_spec(rdzv_round=1)
        spec.rdzv_handler._settings = MagicMock()
        spec.rdzv_handler._settings.min_nodes = min_nodes

        agent = LocalElasticAgent(
            spec=spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
            cycle_info_writer=None,
        )
        agent._worker_group = MagicMock()
        agent._worker_group.group_rank = group_rank
        agent._store = MagicMock()
        agent._store.check.return_value = store_check_returns
        return agent

    def test_standby_raises_graceful_exit_when_exit_barrier_key_set(self):
        """Standby (group_rank >= min_nodes) raises RendezvousGracefulExitError when exit barrier is complete."""
        agent = self._make_agent_standby(min_nodes=2, group_rank=2, store_check_returns=True)
        with self.assertRaises(RendezvousGracefulExitError) as ctx:
            agent._maybe_exit_standby_on_success("trainer")
        self.assertIn("exit barrier", str(ctx.exception).lower())

    def test_standby_does_not_raise_when_exit_barrier_key_not_set(self):
        """Standby does not raise when exit barrier key is not set (keeps waiting)."""
        agent = self._make_agent_standby(min_nodes=2, group_rank=2, store_check_returns=False)
        agent._maybe_exit_standby_on_success("trainer")  # no raise
        agent._store.check.assert_called_once()

    def test_active_node_does_not_raise_even_when_key_set(self):
        """Active node (group_rank < min_nodes) returns without raising even if key is set."""
        agent = self._make_agent_standby(min_nodes=2, group_rank=1, store_check_returns=True)
        agent._maybe_exit_standby_on_success("trainer")  # no raise
        agent._store.check.assert_not_called()
