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
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest
from torch.distributed.elastic.agent.server.api import RunResult, WorkerState

from nvidia_resiliency_ext import fault_tolerance
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.utils import (
    DEFAULT_NO_RESTART_EXIT_CODE,
    RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP,
    RDZV_SHUTDOWN_REASON_NO_PROGRESS,
)
from nvidia_resiliency_ext.shared_utils.os_utils import resolve_under_allowed_roots

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
            shlex.split(cmd_to_run),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        return proc.returncode, stdout
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        output = exc.output or ""
        assert False, f"ft_launcher was still running after {timeout} seconds\n{output}"


def _save_ft_cfg(cfg, dirpath):
    cfg_path = os.path.join(dirpath, "_tmp_ft_cfg.yaml")
    cfg.to_yaml_file(cfg_path)
    return cfg_path


def _get_util_script_path():
    return os.path.join(os.path.dirname(__file__), "_launcher_test_util.py")


def _config_from_launcher_cli(cli_args):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    args = parser.parse_args(cli_args)
    with patch.object(
        launcher.LocalElasticAgent,
        "setup_rank_monitors_early",
        return_value={},
    ) as setup_rank_monitors_early:
        config, cmd, cmd_args = launcher.config_from_args(args)
    return config, cmd, cmd_args, setup_rank_monitors_early


def test_register_barrier_rdzv_handler_applies_c10d_patch():
    from torch.distributed.elastic.rendezvous import rendezvous_handler_registry

    from nvidia_resiliency_ext.fault_tolerance import c10d_monkey_patch, launcher

    with (
        patch.object(c10d_monkey_patch, "apply_c10d_patch") as apply_c10d_patch,
        patch.object(rendezvous_handler_registry, "_registry", {"c10d": object()}),
        patch.object(rendezvous_handler_registry, "register") as register,
    ):
        launcher._register_ft_rdzv_handler()

    apply_c10d_patch.assert_called_once()
    register.assert_called_once()
    assert register.call_args.args[0] == "c10d"


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], "barrier"),
        (["--ft-rdzv-impl", "barrier"], "barrier"),
    ],
)
def test_ft_rdzv_impl_accepts_barrier(extra_args, expected):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    args = parser.parse_args(
        [
            "--nnodes",
            "1",
            "--nproc-per-node",
            "1",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            *extra_args,
            "train.py",
        ]
    )

    assert args.ft_rdzv_impl == expected


def test_ft_rdzv_impl_rejects_legacy():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    parser = launcher.get_args_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ft-rdzv-impl", "legacy", "train.py"])


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
    config, _, _, setup_rank_monitors = _config_from_launcher_cli(
        [
            "--monitor-interval=1",
            f"--ft-cfg-path={empty_ft_cfg_path}",
            f"--nproc-per-node={WORLD_SIZE}",
            "--ft-rank-heartbeat-timeout=1.0",
            _get_util_script_path(),
            "--scenario=test_ranks_exit_gracefully",
        ]
    )
    assert config.fault_tol_cfg.rank_heartbeat_timeout == 1.0
    assert setup_rank_monitors.call_args.kwargs["ft_cfg"] is config.fault_tol_cfg

    # Invalid config file path - should fail despite FT args specified via CLI
    with pytest.raises(FileNotFoundError):
        _config_from_launcher_cli(
            [
                "--monitor-interval=1",
                "--ft-cfg-path=/not/there.yaml",
                "--ft-rank-heartbeat-timeout=1.0",
                f"--nproc-per-node={WORLD_SIZE}",
                _get_util_script_path(),
                "--scenario=test_ranks_exit_gracefully",
            ]
        )


def test_config_provided_via_cli(tmp_dir):
    # Check if FT args passed via CLI were propagated to the FT monitor process
    config, _, _, setup_rank_monitors = _config_from_launcher_cli(
        [
            "--ft-workload-check-interval=321.0",
            "--ft-initial-rank-heartbeat-timeout=1.0",
            "--ft-rank-heartbeat-timeout=2.0",
            "--ft-rank-termination-signal=SIGUSR2",
            "--ft-log-level=WARNING",
            f"--nproc-per-node={WORLD_SIZE}",
            _get_util_script_path(),
            "--scenario=dump_cfg",
            f"--tmp_dir={tmp_dir}",
        ]
    )

    assert config.fault_tol_cfg.workload_check_interval == 321.0
    assert config.fault_tol_cfg.initial_rank_heartbeat_timeout == 1.0
    assert config.fault_tol_cfg.rank_heartbeat_timeout == 2.0
    assert config.fault_tol_cfg.rank_termination_signal == signal.SIGUSR2
    assert config.fault_tol_cfg.log_level == logging.WARNING
    assert setup_rank_monitors.call_args.kwargs["ft_cfg"] is config.fault_tol_cfg


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

    config, _, _, setup_rank_monitors = _config_from_launcher_cli(
        [
            "--ft-rank-heartbeat-timeout=123.0",
            "--ft-safety-factor=7.7",
            "--ft-rank-termination-signal=SIGUSR1",
            "--ft-log-level=CRITICAL",
            f"--ft-cfg-path={ft_cfg_path}",
            f"--nproc-per-node={WORLD_SIZE}",
            _get_util_script_path(),
            "--scenario=dump_cfg",
            f"--tmp_dir={tmp_dir}",
        ]
    )

    assert config.fault_tol_cfg.workload_check_interval == 321.0
    assert config.fault_tol_cfg.initial_rank_heartbeat_timeout == 111.0
    assert config.fault_tol_cfg.rank_heartbeat_timeout == 123.0
    assert config.fault_tol_cfg.safety_factor == 7.7
    assert config.fault_tol_cfg.rank_termination_signal == signal.SIGUSR1
    assert config.fault_tol_cfg.log_level == logging.CRITICAL
    assert setup_rank_monitors.call_args.kwargs["ft_cfg"] is config.fault_tol_cfg


# ==============================================================================
# Unit tests for launcher cycle-info env path interaction
# ==============================================================================


def _make_agent_spec(rdzv_round=1):
    """Minimal WorkerSpec-like object for testing launcher cycle-info env interaction."""
    spec = MagicMock()
    spec.rdzv_handler = MagicMock()
    spec.rdzv_handler.round.return_value = rdzv_round
    spec.rdzv_handler.get_active_node_addrs.return_value = ["node001", "node002"]
    spec.rdzv_handler.get_standby_node_addrs.return_value = ["node003"]
    spec.rdzv_handler.get_active_ranks.return_value = [0, 1]
    spec.rdzv_handler._attribution_service = None
    # Only the rendezvous store host holds an attribution client, and a stop is the
    # exception rather than the rule, so default both attribution probes to "no stop".
    spec.rdzv_handler.attribution_stop_requested.return_value = False
    spec.rdzv_handler.no_restart_reason.return_value = None
    spec.max_restarts = 3
    return spec


class TestLauncherCycleInfoEnvPath(unittest.TestCase):
    """Unit tests for launcher's read-only cycle-info worker env plumbing."""

    def setUp(self):
        """Set up test fixtures."""
        self.spec = _make_agent_spec(rdzv_round=1)
        self.fault_tol_cfg = FaultToleranceConfig()
        self.logs_specs = MagicMock()
        self.logs_specs.get_cycle_log_file.return_value = "/path/to/cycle_0.log"

    def test_remaining_restarts_corrected_in_run(self):
        """run() re-syncs _remaining_restarts before delegating to _invoke_run.

        At __init__ time _round=0, so _remaining_restarts = max_restarts provisionally.
        run() re-computes using the post-sync cycle number before the monitor loop starts.
        """
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        # rdzv_round=2: replacement node synced to cycle 2, max_restarts=3 -> remaining=1
        spec = _make_agent_spec(rdzv_round=2)
        agent = LocalElasticAgent(
            spec=spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )

        captured = {}

        def fake_invoke_run(role):
            captured['remaining'] = agent._remaining_restarts
            return MagicMock()

        with (
            patch.object(agent, '_invoke_run', side_effect=fake_invoke_run),
            patch.object(agent, '_shutdown'),
            patch.object(agent, '_record_metrics'),
            patch.object(agent, '_record_worker_events'),
        ):
            agent.run()

        # At __init__ time round()=2, so provisional value is already 1 in this mock.
        # In production, round()=0 at init and round()=2 after _complete_initialization();
        # run() always re-computes so the value is guaranteed correct regardless.
        self.assertEqual(captured['remaining'], 1)  # max_restarts(3) - round()(2) = 1

    def test_current_cycle_info_path_returns_none_when_disabled(self):
        """No cycle-info env path is set when cycle info is disabled."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        agent = LocalElasticAgent(
            spec=self.spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )
        result = agent._current_cycle_info_path()
        self.assertIsNone(result)

    def test_current_cycle_info_path_uses_slurm_job_id(self):
        """Launchers compute the current symlink path without writing cycle info."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        fault_tol_cfg = FaultToleranceConfig(cycle_info_dir="/nvrx")
        with patch.dict(os.environ, {"SLURM_JOB_ID": "job1"}, clear=False):
            agent = LocalElasticAgent(
                spec=self.spec,
                fault_tol_cfg=fault_tol_cfg,
                logs_specs=self.logs_specs,
            )
            result = agent._current_cycle_info_path()

        self.assertEqual(result, "/nvrx/cycle_info.job1.current")

    def test_current_cycle_info_path_prefers_slurm_array_job_id(self):
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        fault_tol_cfg = FaultToleranceConfig(cycle_info_dir="/nvrx")
        with patch.dict(
            os.environ,
            {"SLURM_ARRAY_JOB_ID": "array1", "SLURM_JOB_ID": "job1"},
            clear=False,
        ):
            agent = LocalElasticAgent(
                spec=self.spec,
                fault_tol_cfg=fault_tol_cfg,
                logs_specs=self.logs_specs,
            )
            result = agent._current_cycle_info_path()

        self.assertEqual(result, "/nvrx/cycle_info.array1.current")


class TestLauncherRunBehavior(unittest.TestCase):
    """Unit tests for run() exception handling paths."""

    def setUp(self):
        self.spec = _make_agent_spec(rdzv_round=1)
        self.fault_tol_cfg = FaultToleranceConfig()
        self.logs_specs = MagicMock()
        self.logs_specs.get_cycle_log_file.return_value = "/path/to/cycle_0.log"

    def test_run_graceful_exit_returns_none_without_cycle_info_update(self):
        """RendezvousGracefulExitError is not a launcher-owned cycle-info path."""
        from torch.distributed.elastic.rendezvous.api import RendezvousGracefulExitError

        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        spec = _make_agent_spec(rdzv_round=3)
        spec.rdzv_handler.is_shutdown_due_to_failure.return_value = False
        agent = LocalElasticAgent(
            spec=spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )

        with (
            patch.object(
                agent, '_invoke_run', side_effect=RendezvousGracefulExitError("round closed")
            ),
            patch.object(agent, '_shutdown'),
        ):
            result = agent.run()

        self.assertIsNone(result)
        spec.rdzv_handler.is_shutdown_due_to_failure.assert_called_once_with()

    def test_run_failure_shutdown_raises_terminal_worker_group_failure(self):
        """A graceful rendezvous exception becomes failure when the store says why."""
        from torch.distributed.elastic.rendezvous.api import RendezvousGracefulExitError

        from nvidia_resiliency_ext.fault_tolerance import launcher
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        spec = _make_agent_spec(rdzv_round=3)
        spec.rdzv_handler.is_shutdown_due_to_failure.return_value = True
        agent = LocalElasticAgent(
            spec=spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )

        with (
            patch.object(
                agent, '_invoke_run', side_effect=RendezvousGracefulExitError("round closed")
            ),
            patch.object(agent, '_shutdown'),
        ):
            with self.assertRaises(launcher.TerminalWorkerGroupFailure):
                agent.run()

    def test_invoke_run_preserves_failures_when_restarts_exhausted(self):
        """The terminal failed result keeps the local child failure map."""
        from torch.distributed.elastic.agent.server.api import RunResult, WorkerState

        from nvidia_resiliency_ext.fault_tolerance import launcher
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        spec = _make_agent_spec(rdzv_round=3)
        spec.role = "trainer"
        spec.monitor_interval = 0
        agent = LocalElasticAgent(
            spec=spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )
        agent._worker_group.state = WorkerState.HEALTHY
        agent._worker_group.group_rank = 0
        failure = MagicMock()
        failures = {7: failure}

        with (
            patch.object(agent, '_initialize_workers'),
            patch.object(
                agent,
                '_monitor_workers',
                return_value=RunResult(state=WorkerState.FAILED, failures=failures),
            ),
            patch.object(agent, '_handle_restart_decision', return_value=False),
            patch.object(agent, '_stop_workers'),
            patch.object(launcher, 'record_profiling_event'),
            patch.object(launcher, 'put_metric'),
            patch.object(launcher.time, 'sleep'),
        ):
            result = agent._invoke_run_with_any_failed_policy()

        self.assertEqual(result.state, WorkerState.FAILED)
        self.assertEqual(result.failures, failures)


class TestHandleRestartDecision(unittest.TestCase):
    """Unit tests for _handle_restart_decision() and _open_rendezvous_for_restart()."""

    def setUp(self):
        self.spec = _make_agent_spec(rdzv_round=1)
        self.fault_tol_cfg = FaultToleranceConfig()
        self.logs_specs = MagicMock()
        self.logs_specs.get_cycle_log_file.return_value = "/path/to/cycle_0.log"

    def _make_agent(self):
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        return LocalElasticAgent(
            spec=self.spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )

    def test_handle_restart_decision_progress_terminate(self):
        """No-progress is a no-restart decision, reported like an attribution STOP.

        Both mean "NVRx decided, do not requeue", so they must be indistinguishable to
        downstream tooling rather than this one looking like an ordinary crash.
        """
        from nvidia_resiliency_ext.fault_tolerance.launcher import NoRestartRequested

        agent = self._make_agent()
        agent._is_store_host = True
        agent._progress_tracker = MagicMock()
        agent._progress_tracker.should_terminate_early.return_value = True
        agent._remaining_restarts = 2
        agent._rdzv_handler._attribution_service = MagicMock()

        with (
            patch.object(agent, '_restart_workers') as mock_restart,
            patch.object(agent, '_open_rendezvous_for_restart') as mock_open,
            patch.object(agent, '_stop_workers') as mock_stop,
        ):
            with self.assertRaises(NoRestartRequested) as ctx:
                agent._handle_restart_decision(
                    role="test", spec=self.spec, log_msg="[%s] restarting"
                )

        self.assertEqual(ctx.exception.reason, RDZV_SHUTDOWN_REASON_NO_PROGRESS)
        agent._rdzv_handler.signal_no_restart.assert_called_once_with(
            RDZV_SHUTDOWN_REASON_NO_PROGRESS
        )
        mock_stop.assert_called_once_with(agent._worker_group)
        agent._rdzv_handler._attribution_service.request_terminal_analysis.assert_not_called()
        mock_restart.assert_not_called()
        mock_open.assert_not_called()
        # The restart budget is untouched: this is a stop, not a consumed retry.
        self.assertEqual(agent._remaining_restarts, 2)

    def test_handle_restart_decision_restarts_remaining(self):
        """Returns True and decrements _remaining_restarts when restarts are available."""
        agent = self._make_agent()
        agent._progress_tracker = MagicMock()
        agent._progress_tracker.should_terminate_early.return_value = False
        agent._remaining_restarts = 2

        with (
            patch.object(agent, '_restart_workers') as mock_restart,
            patch.object(agent, '_open_rendezvous_for_restart') as mock_open,
        ):
            result = agent._handle_restart_decision(
                role="test", spec=self.spec, log_msg="[%s] restarting", open_rendezvous=False
            )

        self.assertTrue(result)
        self.assertEqual(agent._remaining_restarts, 1)
        mock_restart.assert_called_once()
        mock_open.assert_not_called()

    def test_handle_restart_decision_leaves_terminal_attribution_to_rendezvous_close(self):
        """Terminal attribution is owned by the rendezvous control-host close path."""
        agent = self._make_agent()
        agent._is_store_host = True
        agent._progress_tracker = MagicMock()
        agent._remaining_restarts = 2
        agent._rdzv_handler._attribution_service = MagicMock()
        calls = []
        agent._progress_tracker.analyze_previous_cycle.side_effect = lambda: calls.append("analyze")
        agent._progress_tracker.should_terminate_early.side_effect = (
            lambda: calls.append("progress-check") or False
        )

        with patch.object(
            agent, '_restart_workers', side_effect=lambda _wg: calls.append("restart")
        ):
            result = agent._handle_restart_decision(
                role="test", spec=self.spec, log_msg="[%s] restarting"
            )

        self.assertTrue(result)
        self.assertEqual(calls, ["analyze", "progress-check", "restart"])
        agent._rdzv_handler._attribution_service.request_terminal_analysis.assert_not_called()
        agent._rdzv_handler._attribution_service.get_last_result.assert_not_called()

    def test_handle_restart_decision_no_restarts_left(self):
        """Returns False when _remaining_restarts is 0."""
        agent = self._make_agent()
        agent._is_store_host = True
        agent._progress_tracker = MagicMock()
        agent._progress_tracker.should_terminate_early.return_value = False
        agent._remaining_restarts = 0
        agent._rdzv_handler._attribution_service = MagicMock()

        with patch.object(agent, '_restart_workers') as mock_restart:
            result = agent._handle_restart_decision(
                role="test", spec=self.spec, log_msg="[%s] restarting"
            )

        self.assertFalse(result)
        agent._rdzv_handler._attribution_service.request_terminal_analysis.assert_not_called()
        mock_restart.assert_not_called()

    def test_request_terminal_attribution_uses_barrier_state_helper(self):
        """Final-cycle terminal requests share the barrier attribution helper."""
        agent = self._make_agent()
        agent._is_store_host = True
        agent._rdzv_handler._barrier_state = MagicMock()

        agent._request_terminal_attribution()

        helper = (
            agent._rdzv_handler._barrier_state._request_terminal_attribution_for_submitted_cycle
        )
        helper.assert_called_once_with()

    def test_handle_restart_decision_open_rendezvous_called_when_requested(self):
        """Calls _open_rendezvous_for_restart() when open_rendezvous=True."""
        agent = self._make_agent()
        agent._progress_tracker = MagicMock()
        agent._progress_tracker.should_terminate_early.return_value = False
        agent._remaining_restarts = 1

        with (
            patch.object(agent, '_restart_workers'),
            patch.object(agent, '_open_rendezvous_for_restart') as mock_open,
        ):
            agent._handle_restart_decision(
                role="test", spec=self.spec, log_msg="[%s] restarting", open_rendezvous=True
            )

        mock_open.assert_called_once()

    def test_open_rendezvous_for_restart_barrier_handler(self):
        """Calls _barrier_state.open_rendezvous() when handler has _barrier_state."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        barrier_state = MagicMock()
        self.spec.rdzv_handler._barrier_state = barrier_state
        agent = LocalElasticAgent(
            spec=self.spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )
        agent._open_rendezvous_for_restart()

        barrier_state.open_rendezvous.assert_called_once()


def _healthy_agent(fault_tol_cfg, logs_specs, **rdzv_overrides):
    """Agent wired so one monitor-loop pass runs against a HEALTHY worker group."""
    from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

    spec = _make_agent_spec(rdzv_round=1)
    spec.role = "trainer"
    spec.monitor_interval = 0
    for name, value in rdzv_overrides.items():
        getattr(spec.rdzv_handler, name).return_value = value
    agent = LocalElasticAgent(
        spec=spec,
        fault_tol_cfg=fault_tol_cfg,
        logs_specs=logs_specs,
    )
    agent._worker_group.state = WorkerState.HEALTHY
    agent._worker_group.group_rank = 0
    return agent


class TestNoRestartTermination(unittest.TestCase):
    """The monitor loop consumes a latched STOP verdict and ends the job."""

    def setUp(self):
        self.fault_tol_cfg = FaultToleranceConfig()
        self.logs_specs = MagicMock()
        self.logs_specs.get_cycle_log_file.return_value = "/path/to/cycle_0.log"

    def _run_one_monitor_pass(self, agent):
        from nvidia_resiliency_ext.fault_tolerance import launcher

        with (
            patch.object(agent, '_initialize_workers'),
            patch.object(
                agent, '_monitor_workers', return_value=RunResult(state=WorkerState.HEALTHY)
            ),
            patch.object(agent, '_stop_workers') as stop_workers,
            patch.object(launcher, 'record_profiling_event'),
            patch.object(launcher, 'put_metric'),
            patch.object(launcher.time, 'sleep'),
        ):
            try:
                return agent._invoke_run_with_any_failed_policy(), stop_workers, None
            except Exception as exc:  # noqa: BLE001 - the raised type is the assertion
                return None, stop_workers, exc

    def test_store_host_terminates_job_when_attribution_latches_stop(self):
        """The latch is local, so acting on it needs no extra TCPStore poll."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import NoRestartRequested

        agent = _healthy_agent(self.fault_tol_cfg, self.logs_specs, attribution_stop_requested=True)
        rdzv_handler = agent._worker_group.spec.rdzv_handler

        _, stop_workers, exc = self._run_one_monitor_pass(agent)

        self.assertIsInstance(exc, NoRestartRequested)
        rdzv_handler.signal_no_restart.assert_called_once_with(
            RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP
        )
        stop_workers.assert_called_once_with(agent._worker_group)
        # The latch is checked first, so the stopping node skips the round-open store read.
        rdzv_handler.is_next_round_open.assert_not_called()

    def test_store_host_terminates_job_when_latch_set_during_a_failed_cycle(self):
        """The latch preempts the restart decision, not just the healthy path.

        A verdict often lands while the group is FAILED rather than HEALTHY, especially in
        a crash loop. Without preemption the node would spend a restart and a rendezvous
        round trip before noticing.
        """
        from nvidia_resiliency_ext.fault_tolerance import launcher
        from nvidia_resiliency_ext.fault_tolerance.launcher import NoRestartRequested

        agent = _healthy_agent(self.fault_tol_cfg, self.logs_specs, attribution_stop_requested=True)
        rdzv_handler = agent._worker_group.spec.rdzv_handler

        with (
            patch.object(agent, '_initialize_workers'),
            patch.object(
                agent, '_monitor_workers', return_value=RunResult(state=WorkerState.FAILED)
            ),
            patch.object(agent, '_stop_workers') as stop_workers,
            patch.object(agent, '_handle_restart_decision') as restart_decision,
            patch.object(launcher, 'record_profiling_event'),
            patch.object(launcher, 'put_metric'),
            patch.object(launcher.time, 'sleep'),
        ):
            with self.assertRaises(NoRestartRequested) as ctx:
                agent._invoke_run_with_any_failed_policy()

        self.assertEqual(ctx.exception.reason, RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP)
        rdzv_handler.signal_no_restart.assert_called_once_with(
            RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP
        )
        stop_workers.assert_called_once_with(agent._worker_group)
        restart_decision.assert_not_called()

    def test_successful_workers_are_not_killed_by_a_latched_stop(self):
        """A workload that finished is not killed because an earlier cycle was fatal."""
        from nvidia_resiliency_ext.fault_tolerance import launcher

        agent = _healthy_agent(self.fault_tol_cfg, self.logs_specs, attribution_stop_requested=True)
        rdzv_handler = agent._worker_group.spec.rdzv_handler
        succeeded = RunResult(state=WorkerState.SUCCEEDED)

        with (
            patch.object(agent, '_initialize_workers'),
            patch.object(agent, '_monitor_workers', return_value=succeeded),
            patch.object(agent, '_exit_barrier'),
            patch.object(agent, '_stop_workers') as stop_workers,
            patch.object(launcher, 'record_profiling_event'),
            patch.object(launcher, 'put_metric'),
            patch.object(launcher.time, 'sleep'),
        ):
            result = agent._invoke_run_with_any_failed_policy()

        self.assertEqual(result.state, WorkerState.SUCCEEDED)
        rdzv_handler.signal_no_restart.assert_not_called()
        stop_workers.assert_not_called()

    def test_peer_terminates_job_when_round_opens_for_attribution_stop(self):
        """Peers hold no attribution client; they learn about the stop from the store."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import NoRestartRequested

        agent = _healthy_agent(
            self.fault_tol_cfg,
            self.logs_specs,
            attribution_stop_requested=False,
            is_next_round_open=True,
            no_restart_reason=RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP,
        )

        with patch.object(agent, '_handle_restart_decision') as restart_decision:
            _, stop_workers, exc = self._run_one_monitor_pass(agent)

        self.assertIsInstance(exc, NoRestartRequested)
        stop_workers.assert_called_once_with(agent._worker_group)
        # A stop is not a restart: the restart budget must not be consumed.
        restart_decision.assert_not_called()

    def test_peer_restarts_normally_when_round_opens_without_attribution_stop(self):
        """An ordinary peer failure keeps the existing restart path."""
        agent = _healthy_agent(
            self.fault_tol_cfg,
            self.logs_specs,
            attribution_stop_requested=False,
            is_next_round_open=True,
            no_restart_reason=None,
        )

        with patch.object(agent, '_handle_restart_decision', return_value=False) as decision:
            result, _, exc = self._run_one_monitor_pass(agent)

        self.assertIsNone(exc)
        self.assertEqual(result.state, WorkerState.FAILED)
        decision.assert_called_once()

    def test_graceful_rendezvous_exit_maps_no_restart_to_dedicated_exception(self):
        """Hot spares exit via rendezvous, not the monitor loop, and must not report success."""
        from torch.distributed.elastic.rendezvous.api import RendezvousGracefulExitError

        from nvidia_resiliency_ext.fault_tolerance.launcher import (
            LocalElasticAgent,
            NoRestartRequested,
        )

        spec = _make_agent_spec(rdzv_round=3)
        spec.rdzv_handler.no_restart_reason.return_value = RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP
        agent = LocalElasticAgent(
            spec=spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )

        with (
            patch.object(
                agent, '_invoke_run', side_effect=RendezvousGracefulExitError("round closed")
            ),
            patch.object(agent, '_shutdown'),
        ):
            with self.assertRaises(NoRestartRequested):
                agent.run()


def test_no_restart_exit_code_default_is_outside_reserved_ranges():
    """Downstream tooling keys on this code to tell "do not requeue" from a failure."""
    assert DEFAULT_NO_RESTART_EXIT_CODE == 93
    assert DEFAULT_NO_RESTART_EXIT_CODE not in (0, 1, 2, 255)
    assert not 64 <= DEFAULT_NO_RESTART_EXIT_CODE <= 78
    assert not 126 <= DEFAULT_NO_RESTART_EXIT_CODE <= 165


def test_main_reports_same_exit_code_for_both_no_restart_reasons():
    """Attribution STOP and no-progress are one binary signal to the scheduler."""
    from nvidia_resiliency_ext.fault_tolerance.launcher import NoRestartRequested, main

    codes = []
    for reason in (RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP, RDZV_SHUTDOWN_REASON_NO_PROGRESS):
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.launcher.run",
                side_effect=NoRestartRequested(reason),
            ),
            pytest.raises(SystemExit) as excinfo,
        ):
            main(['train.py'])
        codes.append(excinfo.value.code)

    assert codes == [DEFAULT_NO_RESTART_EXIT_CODE, DEFAULT_NO_RESTART_EXIT_CODE]


def test_main_honors_no_restart_exit_code_override():
    from nvidia_resiliency_ext.fault_tolerance.launcher import NoRestartRequested, main

    with (
        patch(
            "nvidia_resiliency_ext.fault_tolerance.launcher.run",
            side_effect=NoRestartRequested(RDZV_SHUTDOWN_REASON_ATTRIBUTION_STOP),
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        main(['--ft-no-restart-exit-code', '17', 'train.py'])

    assert excinfo.value.code == 17


def test_ft_log_aggregator_count_rejects_negative():
    from nvidia_resiliency_ext.fault_tolerance.launcher import _validate_args, get_args_parser

    parser = get_args_parser()
    args = parser.parse_args(['--ft-log-aggregator-count', '-1', 'train.py'])
    with pytest.raises(ValueError, match='--ft-log-aggregator-count'):
        _validate_args(args)


def test_cli_cycle_info_dir_does_not_require_per_cycle_applog():
    from nvidia_resiliency_ext.fault_tolerance.launcher import _validate_args, get_args_parser

    parser = get_args_parser()
    args = parser.parse_args(['--ft-cycle-info-dir', '/nvrx', 'train.py'])

    _validate_args(args)


def test_segment_health_check_dir_is_passed_to_rendezvous(tmp_path):
    from nvidia_resiliency_ext.fault_tolerance.launcher import config_from_args, get_args_parser

    parser = get_args_parser()
    args = parser.parse_args(["--ft-segment-health-check-dir", str(tmp_path), "train.py"])

    with patch(
        "nvidia_resiliency_ext.fault_tolerance.launcher.LocalElasticAgent.setup_rank_monitors_early",
        return_value={},
    ):
        config, _, _ = config_from_args(args)

    assert config.fault_tol_cfg.segment_health_check_dir == str(tmp_path)
    assert config.rdzv_configs["segment_health_check_dir"] == str(tmp_path)


def test_segment_health_check_dir_must_be_absolute():
    from nvidia_resiliency_ext.fault_tolerance.launcher import config_from_args, get_args_parser

    parser = get_args_parser()
    args = parser.parse_args(["--ft-segment-health-check-dir", "relative/path", "train.py"])

    with pytest.raises(ValueError, match="must be an absolute path"):
        config_from_args(args)


def test_cli_attribution_endpoint_requires_per_cycle_applog():
    from nvidia_resiliency_ext.fault_tolerance.launcher import (
        _validate_attribution_requires_per_cycle_applog,
        get_args_parser,
    )

    parser = get_args_parser()
    args = parser.parse_args(['--ft-attribution-endpoint', 'localhost', 'train.py'])

    with pytest.raises(ValueError, match='--ft-attribution-endpoint requires'):
        _validate_attribution_requires_per_cycle_applog(args, FaultToleranceConfig())


def test_yaml_attribution_endpoint_requires_per_cycle_applog():
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance.launcher import (
        _validate_attribution_requires_per_cycle_applog,
    )

    args = SimpleNamespace(ft_attribution_endpoint=None, ft_per_cycle_applog_prefix=None)
    cfg = FaultToleranceConfig(attribution_endpoint='localhost')

    with pytest.raises(ValueError, match='--ft-attribution-endpoint requires'):
        _validate_attribution_requires_per_cycle_applog(args, cfg)


def test_per_cycle_applog_without_attribution_is_valid():
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance.launcher import (
        _validate_attribution_requires_per_cycle_applog,
    )

    args = SimpleNamespace(
        ft_attribution_endpoint=None,
        ft_per_cycle_applog_prefix='/tmp/train.log',
    )

    _validate_attribution_requires_per_cycle_applog(args, FaultToleranceConfig())


def test_attribution_endpoint_with_per_cycle_applog_is_valid():
    from nvidia_resiliency_ext.fault_tolerance.launcher import (
        _validate_attribution_requires_per_cycle_applog,
        get_args_parser,
    )

    parser = get_args_parser()
    args = parser.parse_args(
        [
            '--ft-per-cycle-applog-prefix',
            '/tmp/train.log',
            '--ft-attribution-endpoint',
            'localhost',
            'train.py',
        ]
    )

    _validate_attribution_requires_per_cycle_applog(args, FaultToleranceConfig())


def test_log_funnel_ports_from_launcher_args_auto():
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance.launcher import LogFunnelPorts

    # 0 = auto: single-level for small jobs, two-level for large jobs
    cases = [
        ("1", 1),
        ("1536", 1),
        ("1537", 2),
        ("3072", 2),
        ("3073", 3),
        ("4608", 3),
    ]
    for nnodes, expected_n in cases:
        ports = LogFunnelPorts.from_launcher_args(
            SimpleNamespace(ft_log_server_port=50051, ft_log_aggregator_count=0, nnodes=nnodes)
        )
        assert (
            ports.first_level_count == expected_n
        ), f"nnodes={nnodes}: expected n={expected_n}, got {ports.first_level_count}"


def test_log_funnel_ports_from_launcher_args_rejects_negative():
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance.launcher import LogFunnelPorts

    with pytest.raises(ValueError):
        LogFunnelPorts.from_launcher_args(
            SimpleNamespace(ft_log_server_port=50051, ft_log_aggregator_count=-1, nnodes="100")
        )


def test_grpc_log_server_log_prefix_resolution(tmp_path):
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance.launcher import _resolve_grpc_log_server_log_prefix

    assert _resolve_grpc_log_server_log_prefix(
        SimpleNamespace(
            ft_log_server_log_prefix=str(tmp_path / "explicit"),
            ft_nvrx_logfile=str(tmp_path / "nvrx.log"),
            ft_per_cycle_applog_prefix=str(tmp_path / "app.log"),
        )
    ) == str(tmp_path / "explicit")
    assert _resolve_grpc_log_server_log_prefix(
        SimpleNamespace(
            ft_log_server_log_prefix=None,
            ft_nvrx_logfile=str(tmp_path / "nvrx.log"),
            ft_per_cycle_applog_prefix=str(tmp_path / "app.log"),
        )
    ) == str(tmp_path / "nvrx_grpc")
    assert _resolve_grpc_log_server_log_prefix(
        SimpleNamespace(
            ft_log_server_log_prefix=None,
            ft_nvrx_logfile=None,
            ft_per_cycle_applog_prefix=str(tmp_path / "app.log"),
        )
    ) == str(tmp_path / "app_grpc")


def test_start_grpc_log_servers_uses_prefix_for_root_and_leaf_logs(tmp_path):
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance import launcher

    class FakePopen:
        next_pid = 1000

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.pid = FakePopen.next_pid
            FakePopen.next_pid += 1

        def kill(self):
            pass

        def wait(self):
            pass

    applog_dir = tmp_path / "applogs"
    applog_dir.mkdir()
    args = SimpleNamespace(
        nnodes="2",
        ft_log_server_graceful_shutdown_timeout=60.0,
        ft_log_leaf_max_queue_chunks=-1,
        ft_per_cycle_applog_prefix=str(applog_dir / "train.log"),
        ft_nvrx_logfile=None,
        ft_log_server_log_prefix=None,
    )
    log_dir = tmp_path / "missing" / "logs"
    log_prefix = str(log_dir / "grpc_diag")
    ports = launcher.LogFunnelPorts(base_port=50051, first_level_count=2)

    with patch.object(launcher.subprocess, "Popen", FakePopen):
        procs = launcher._start_grpc_log_servers(args, log_prefix, ports)

    assert len(procs) == 3
    assert log_dir.is_dir()
    assert (log_dir / "grpc_diag_root.log").is_file()
    assert (log_dir / "grpc_diag_leaf_0.log").is_file()
    assert (log_dir / "grpc_diag_leaf_1.log").is_file()

    # Every spawned server must be confined to the configured log directory: these bind
    # unauthenticated ports and write to a path chosen by the client on each chunk.
    for proc in procs:
        cmd = proc.args[0]
        roots = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--allowed-root"]
        assert roots == [os.path.realpath(str(applog_dir))], cmd


def test_start_grpc_log_servers_requires_resolvable_allowed_roots(tmp_path):
    """Refuse to start unconfined rather than accept client-chosen write destinations."""
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance import launcher

    class FakePopen:
        def __init__(self, *args, **kwargs):
            raise AssertionError("no server should be spawned without allowed roots")

    args = SimpleNamespace(
        nnodes="2",
        ft_log_server_graceful_shutdown_timeout=60.0,
        ft_log_leaf_max_queue_chunks=-1,
        ft_per_cycle_applog_prefix=None,
        ft_nvrx_logfile=None,
        ft_log_server_log_prefix=None,
    )
    ports = launcher.LogFunnelPorts(base_port=50051, first_level_count=1)

    with patch.object(launcher.subprocess, "Popen", FakePopen):
        # Failure is reported by returning no processes; the caller then disables gRPC
        # log aggregation and falls back to direct file writing.
        assert launcher._start_grpc_log_servers(args, str(tmp_path / "grpc_diag"), ports) == []


def test_managed_attribution_listen_port_rejects_log_funnel_overlap():
    from nvidia_resiliency_ext.fault_tolerance.launcher import (
        LogFunnelPorts,
        _validate_managed_attribution_listen_port_not_in_log_funnel,
    )

    funnel_ports = LogFunnelPorts(base_port=50051, first_level_count=3)

    with pytest.raises(ValueError, match="overlaps"):
        _validate_managed_attribution_listen_port_not_in_log_funnel(50053, funnel_ports)

    _validate_managed_attribution_listen_port_not_in_log_funnel(50050, funnel_ports)


def test_non_host_launcher_routes_logs_to_rendezvous_host_and_skips_host_services(tmp_path):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    class FakePipeBasedLogsSpecs:
        def __init__(
            self,
            base_log_file,
            launcher_pipe_fd=None,
            launcher_log_file=None,
            grpc_server_address=None,
            node_id=None,
        ):
            self.base_log_file = base_log_file
            self.grpc_server_address = grpc_server_address
            self.node_id = node_id

    parser = launcher.get_args_parser()
    args = parser.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "control.host:29500",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-enable-log-server",
            "true",
            "--ft-attribution-endpoint",
            "localhost",
            "train.py",
        ]
    )

    with (
        patch.object(launcher, "_matches_machine_hostname", return_value=False),
        patch.object(launcher, "PipeBasedLogsSpecs", FakePipeBasedLogsSpecs),
        patch.object(launcher.LocalElasticAgent, "setup_rank_monitors_early", return_value={}),
        patch.object(
            launcher,
            "_start_grpc_log_servers",
            side_effect=AssertionError("compute launcher must not start gRPC servers"),
        ),
        patch.object(
            launcher.AttributionManager,
            "start_if_needed",
            return_value=None,
        ),
    ):
        config, _, _ = launcher.config_from_args(args)

    assert "is_host" not in config.rdzv_configs
    assert "attribution_endpoint" not in config.rdzv_configs
    assert config.logs_specs.grpc_server_address == "control.host:50051"


def test_same_node_external_control_honors_is_host_false(tmp_path):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    class FakePipeBasedLogsSpecs:
        def __init__(
            self,
            base_log_file,
            launcher_pipe_fd=None,
            launcher_log_file=None,
            grpc_server_address=None,
            node_id=None,
        ):
            self.grpc_server_address = grpc_server_address

    parser = launcher.get_args_parser()
    args = parser.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--rdzv-conf",
            "is_host=false",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-enable-log-server",
            "true",
            "--ft-attribution-endpoint",
            "localhost",
            "train.py",
        ]
    )
    attribution_manager = MagicMock()
    attribution_manager.start_if_needed.return_value = None

    with (
        patch.object(launcher, "_matches_machine_hostname", return_value=True),
        patch.object(launcher, "PipeBasedLogsSpecs", FakePipeBasedLogsSpecs),
        patch.object(launcher.LocalElasticAgent, "setup_rank_monitors_early", return_value={}),
        patch.object(
            launcher,
            "_start_grpc_log_servers",
            side_effect=AssertionError("compute launcher must not start gRPC servers"),
        ),
        patch.object(
            launcher,
            "AttributionManager",
            return_value=attribution_manager,
        ) as attribution_manager_cls,
    ):
        config, _, _ = launcher.config_from_args(args)

    attribution_manager_cls.assert_called_once()
    assert attribution_manager_cls.call_args.kwargs["is_store_host"] is False
    assert config.rdzv_configs["is_host"] == "false"
    assert "attribution_endpoint" not in config.rdzv_configs
    assert config.logs_specs.grpc_server_address == "127.0.0.1:50051"


def test_nvrx_logfile_auto_enables_grpc_routing_to_rendezvous_host(tmp_path):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    class FakePipeBasedLogsSpecs:
        def __init__(
            self,
            base_log_file,
            launcher_pipe_fd=None,
            launcher_log_file=None,
            grpc_server_address=None,
            node_id=None,
        ):
            self.grpc_server_address = grpc_server_address

    parser = launcher.get_args_parser()
    args = parser.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "control.host:29500",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-nvrx-logfile",
            str(tmp_path / "nvrx.log"),
            "train.py",
        ]
    )
    assert args.ft_enable_log_server is None

    with (
        patch.object(launcher, "_matches_machine_hostname", return_value=False),
        patch.object(launcher, "PipeBasedLogsSpecs", FakePipeBasedLogsSpecs),
        patch.object(launcher.LocalElasticAgent, "setup_rank_monitors_early", return_value={}),
        patch.object(
            launcher,
            "_start_grpc_log_servers",
            side_effect=AssertionError("compute launcher must not start gRPC servers"),
        ),
    ):
        config, _, _ = launcher.config_from_args(args)

    assert args.ft_enable_log_server is True
    assert config.logs_specs.grpc_server_address == "control.host:50051"


def _managed_attribution_args(launcher, tmp_path):
    return launcher.get_args_parser().parse_args(
        [
            "--nnodes",
            "1",
            "--nproc-per-node",
            "1",
            "--rdzv-endpoint",
            "localhost:29500",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-nvrx-logfile",
            str(tmp_path / "nvrx.log"),
            "--ft-attribution-endpoint",
            "localhost",
            "--ft-attribution-llm-api-key-file",
            "/no/such/missing_key",
            "train.py",
        ]
    )


def test_store_host_writes_attribution_startup_failure_to_nvrx_log(tmp_path):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    args = _managed_attribution_args(launcher, tmp_path)
    with (
        patch.object(launcher, "_ATTRIBUTION_MANAGER", None),
        patch.object(launcher, "_matches_machine_hostname", return_value=True),
        pytest.raises(ValueError, match="is not a file"),
    ):
        launcher.config_from_args(args, launcher_log_file=args.ft_nvrx_logfile)

    record = (tmp_path / "nvrx.log").read_text(encoding="utf-8")
    assert "managed attribution service LLM API key file is not a file" in record
    assert "Agent's exit code = 1" in record


def test_non_store_host_does_not_write_attribution_startup_failure(tmp_path):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    args = _managed_attribution_args(launcher, tmp_path)
    with (
        patch.object(launcher, "_matches_machine_hostname", return_value=False),
        patch.object(
            launcher.AttributionConfig,
            "from_args",
            side_effect=ValueError("bad attribution config"),
        ),
        pytest.raises(ValueError, match="bad attribution config"),
    ):
        launcher.config_from_args(args, launcher_log_file=args.ft_nvrx_logfile)

    assert not (tmp_path / "nvrx.log").exists()


def test_attribution_startup_log_failure_does_not_mask_original_error(tmp_path):
    from nvidia_resiliency_ext.fault_tolerance import launcher

    args = _managed_attribution_args(launcher, tmp_path)
    original_error = ValueError("bad attribution config")
    with (
        patch.object(launcher, "_ATTRIBUTION_MANAGER", None),
        patch.object(launcher, "_matches_machine_hostname", return_value=True),
        patch.object(
            launcher.AttributionManager,
            "start_if_needed",
            side_effect=original_error,
        ),
        patch("builtins.open", side_effect=OSError("network filesystem unavailable")),
        pytest.raises(ValueError) as exc_info,
    ):
        launcher.config_from_args(args, launcher_log_file=args.ft_nvrx_logfile)

    assert exc_info.value is original_error


def _make_launch_agent_config(**overrides):
    from types import SimpleNamespace

    values = dict(
        run_id="run-a",
        rdzv_configs={},
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=1,
        rdzv_backend="c10d",
        rdzv_endpoint="host:29500",
        local_addr=None,
        fault_tol_cfg=FaultToleranceConfig(cycle_info_dir="/nvrx"),
        role="trainer",
        max_restarts=1,
        monitor_interval=1,
        logs_specs=SimpleNamespace(root_log_dir="/tmp/logs"),
        metrics_cfg={},
        start_method="spawn",
        log_line_prefix_template=None,
        term_timeout=1,
        workers_stop_timeout=1,
        restart_policy="any-failed",
        rank_monitors={},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shutdown_cycle_info_reporter_safely_logs_reporter_exception(caplog, capture_nvrx_logs):
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance import launcher

    reporter_error = RuntimeError("reporter failed")
    rdzv_handler = SimpleNamespace(
        shutdown_cycle_info_reporter=MagicMock(side_effect=reporter_error)
    )

    with caplog.at_level(logging.WARNING, logger=launcher.logger.name):
        launcher._shutdown_cycle_info_reporter_safely(rdzv_handler)

    rdzv_handler.shutdown_cycle_info_reporter.assert_called_once_with()
    assert "Failed to shut down cycle info reporter" in caplog.text


@pytest.mark.parametrize("is_job_array", [False, True])
def test_launch_agent_unhealthy_node_exits_failure_without_rdzv_shutdown(is_job_array):
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance import launcher

    rdzv_handler = MagicMock()
    rdzv_handler._attribution_service = None
    spec = SimpleNamespace(rdzv_handler=rdzv_handler)
    agent = MagicMock()
    agent.run.side_effect = launcher.UnhealthyNodeException("node unhealthy")
    agent._rdzv_handler = rdzv_handler

    config = _make_launch_agent_config()

    with (
        patch.object(launcher, "_get_addr_and_port", return_value=("host", 29500)),
        patch.object(launcher, "_is_store_host", return_value=True),
        patch.object(launcher, "WorkerSpec", return_value=spec),
        patch.object(launcher.rdzv_registry, "get_rendezvous_handler", return_value=rdzv_handler),
        patch.object(launcher, "LocalElasticAgent", return_value=agent),
        patch.object(launcher.metrics, "initialize_metrics"),
        patch.object(launcher.events, "record"),
        patch.object(launcher, "is_slurm_job_array", return_value=is_job_array),
        patch.object(launcher.time, "sleep", return_value=None),
    ):
        with pytest.raises(launcher.UnhealthyNodeException):
            launcher.launch_agent(config, "train.py", [])

    rdzv_handler.shutdown.assert_not_called()
    rdzv_handler.shutdown_due_to_failure.assert_not_called()
    rdzv_handler.shutdown_cycle_info_reporter.assert_called_once()


def test_launch_agent_failed_result_marks_rendezvous_shutdown_failure():
    from types import SimpleNamespace

    from torch.distributed.elastic.agent.server.api import RunResult, WorkerState

    from nvidia_resiliency_ext.fault_tolerance import launcher

    rdzv_handler = MagicMock()
    rdzv_handler._attribution_service = None
    spec = SimpleNamespace(rdzv_handler=rdzv_handler)
    agent = MagicMock()
    agent.run.return_value = RunResult(state=WorkerState.FAILED)
    agent._rdzv_handler = rdzv_handler

    config = _make_launch_agent_config()

    with (
        patch.object(launcher, "_get_addr_and_port", return_value=("host", 29500)),
        patch.object(launcher, "_is_store_host", return_value=True),
        patch.object(launcher, "WorkerSpec", return_value=spec),
        patch.object(launcher.rdzv_registry, "get_rendezvous_handler", return_value=rdzv_handler),
        patch.object(launcher, "LocalElasticAgent", return_value=agent),
        patch.object(launcher.metrics, "initialize_metrics"),
        patch.object(launcher.events, "record"),
        patch.object(launcher.time, "sleep", return_value=None),
    ):
        with pytest.raises(launcher.TerminalWorkerGroupFailure):
            launcher.launch_agent(config, "train.py", [])

    rdzv_handler.shutdown_due_to_failure.assert_called_once_with()
    rdzv_handler.shutdown.assert_not_called()


def test_launch_agent_child_failed_error_marks_rendezvous_shutdown_failure():
    from types import SimpleNamespace

    from torch.distributed.elastic.agent.server.api import RunResult, WorkerState
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, ProcessFailure

    from nvidia_resiliency_ext.fault_tolerance import launcher

    rdzv_handler = MagicMock()
    rdzv_handler._attribution_service = None
    spec = SimpleNamespace(rdzv_handler=rdzv_handler)
    failure = ProcessFailure(local_rank=0, pid=1234, exitcode=1, error_file="/tmp/missing.json")
    agent = MagicMock()
    agent.run.return_value = RunResult(state=WorkerState.FAILED, failures={0: failure})
    agent._rdzv_handler = rdzv_handler

    config = _make_launch_agent_config()

    with (
        patch.object(launcher, "_get_addr_and_port", return_value=("host", 29500)),
        patch.object(launcher, "_is_store_host", return_value=True),
        patch.object(launcher, "WorkerSpec", return_value=spec),
        patch.object(launcher.rdzv_registry, "get_rendezvous_handler", return_value=rdzv_handler),
        patch.object(launcher, "LocalElasticAgent", return_value=agent),
        patch.object(launcher.metrics, "initialize_metrics"),
        patch.object(launcher.events, "record"),
        patch.object(launcher.time, "sleep", return_value=None),
    ):
        with pytest.raises(ChildFailedError):
            launcher.launch_agent(config, "train.py", [])

    rdzv_handler.shutdown_due_to_failure.assert_called_once_with()
    rdzv_handler.shutdown.assert_not_called()


def test_launch_agent_success_result_returns_values_and_gracefully_shuts_down():
    from types import SimpleNamespace

    from torch.distributed.elastic.agent.server.api import RunResult, WorkerState

    from nvidia_resiliency_ext.fault_tolerance import launcher

    rdzv_handler = MagicMock()
    rdzv_handler._attribution_service = None
    spec = SimpleNamespace(rdzv_handler=rdzv_handler)
    agent = MagicMock()
    agent.run.return_value = RunResult(
        state=WorkerState.SUCCEEDED,
        return_values={0: "ok"},
    )
    agent._rdzv_handler = rdzv_handler

    config = _make_launch_agent_config()

    with (
        patch.object(launcher, "_get_addr_and_port", return_value=("host", 29500)),
        patch.object(launcher, "_is_store_host", return_value=False),
        patch.object(launcher, "WorkerSpec", return_value=spec),
        patch.object(launcher.rdzv_registry, "get_rendezvous_handler", return_value=rdzv_handler),
        patch.object(launcher, "LocalElasticAgent", return_value=agent),
        patch.object(launcher.metrics, "initialize_metrics"),
        patch.object(launcher.events, "record"),
    ):
        result = launcher.launch_agent(config, "train.py", [])

    assert result == {0: "ok"}
    rdzv_handler.shutdown.assert_called_once_with()
    rdzv_handler.shutdown_due_to_failure.assert_not_called()


def test_launch_agent_signal_exception_without_rank_failure_exits_normally():
    from types import SimpleNamespace

    from torch.distributed.elastic.multiprocessing import SignalException

    from nvidia_resiliency_ext.fault_tolerance import launcher

    rdzv_handler = MagicMock()
    rdzv_handler._attribution_service = None
    spec = SimpleNamespace(rdzv_handler=rdzv_handler)
    agent = MagicMock()
    agent.run.side_effect = SignalException(
        "simulated signal",
        sigval=signal.Signals(signal.SIGTERM),
    )
    agent._rdzv_handler = rdzv_handler
    agent.any_rank_failed.return_value = False

    config = _make_launch_agent_config(fault_tol_cfg=FaultToleranceConfig())

    with (
        patch.object(launcher, "_get_addr_and_port", return_value=("host", 29500)),
        patch.object(launcher, "_is_store_host", return_value=False),
        patch.object(launcher, "WorkerSpec", return_value=spec),
        patch.object(launcher.rdzv_registry, "get_rendezvous_handler", return_value=rdzv_handler),
        patch.object(launcher, "LocalElasticAgent", return_value=agent),
        patch.object(launcher.metrics, "initialize_metrics"),
        patch.object(launcher.events, "record") as record_event,
    ):
        result = launcher.launch_agent(config, "train.py", [])

    assert result is None
    agent.any_rank_failed.assert_called_once_with()
    record_event.assert_called_once_with(agent.get_event_failed.return_value)
    rdzv_handler.shutdown.assert_not_called()
    rdzv_handler.shutdown_due_to_failure.assert_not_called()


def test_launch_agent_signal_exception_with_rank_failure_re_raises():
    from types import SimpleNamespace

    from torch.distributed.elastic.multiprocessing import SignalException

    from nvidia_resiliency_ext.fault_tolerance import launcher

    rdzv_handler = MagicMock()
    rdzv_handler._attribution_service = None
    spec = SimpleNamespace(rdzv_handler=rdzv_handler)
    agent = MagicMock()
    agent.run.side_effect = SignalException(
        "simulated signal",
        sigval=signal.Signals(signal.SIGTERM),
    )
    agent._rdzv_handler = rdzv_handler
    agent.any_rank_failed.return_value = True

    config = _make_launch_agent_config(fault_tol_cfg=FaultToleranceConfig())

    with (
        patch.object(launcher, "_get_addr_and_port", return_value=("host", 29500)),
        patch.object(launcher, "_is_store_host", return_value=False),
        patch.object(launcher, "WorkerSpec", return_value=spec),
        patch.object(launcher.rdzv_registry, "get_rendezvous_handler", return_value=rdzv_handler),
        patch.object(launcher, "LocalElasticAgent", return_value=agent),
        patch.object(launcher.metrics, "initialize_metrics"),
        patch.object(launcher.events, "record") as record_event,
    ):
        with pytest.raises(SignalException):
            launcher.launch_agent(config, "train.py", [])

    agent.any_rank_failed.assert_called_once_with()
    record_event.assert_called_once_with(agent.get_event_failed.return_value)
    rdzv_handler.shutdown.assert_not_called()
    rdzv_handler.shutdown_due_to_failure.assert_not_called()


class TestLauncherAllowedRoots:
    """The launcher must derive roots covering every path its clients legitimately write."""

    def _args(self, **kwargs):
        from argparse import Namespace

        base = dict(
            ft_per_cycle_applog_prefix=None,
            ft_nvrx_logfile=None,
            ft_log_server_log_prefix=None,
        )
        base.update(kwargs)
        return Namespace(**base)

    def test_roots_are_dirnames_of_configured_log_paths(self, tmp_path):
        from nvidia_resiliency_ext.fault_tolerance.launcher import _resolve_grpc_log_allowed_roots

        applog = tmp_path / "app" / "train.log"
        nvrx = tmp_path / "launcher" / "nvrx.log"
        applog.parent.mkdir()
        nvrx.parent.mkdir()
        roots = _resolve_grpc_log_allowed_roots(
            self._args(
                ft_per_cycle_applog_prefix=str(applog),
                ft_nvrx_logfile=str(nvrx),
            )
        )
        assert roots == [
            os.path.realpath(str(applog.parent)),
            os.path.realpath(str(nvrx.parent)),
        ]

    def test_root_covers_every_per_cycle_log(self, tmp_path):
        """Per-cycle files are <prefix>_cycleN.log in the prefix dir, so one root suffices."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import _resolve_grpc_log_allowed_roots

        applog = tmp_path / "app" / "train.log"
        applog.parent.mkdir()
        (roots,) = _resolve_grpc_log_allowed_roots(
            self._args(ft_per_cycle_applog_prefix=str(applog))
        )
        for cycle in (0, 1, 42):
            cycle_log = str(applog).replace(".log", f"_cycle{cycle}.log")
            assert resolve_under_allowed_roots(cycle_log, [roots]) == cycle_log

    def test_duplicate_directories_collapse(self, tmp_path):
        from nvidia_resiliency_ext.fault_tolerance.launcher import _resolve_grpc_log_allowed_roots

        roots = _resolve_grpc_log_allowed_roots(
            self._args(
                ft_per_cycle_applog_prefix=str(tmp_path / "train.log"),
                ft_nvrx_logfile=str(tmp_path / "nvrx.log"),
            )
        )
        assert roots == [os.path.realpath(str(tmp_path))]

    def test_roots_are_pinned_against_later_symlink_creation(self, tmp_path):
        """realpath of a non-existent dir returns it unchanged, so a dir that only later
        appears as a symlink would resolve elsewhere at write time and the job's own logs
        would be rejected. Roots must be created before being resolved."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import _resolve_grpc_log_allowed_roots

        logdir = tmp_path / "logs"
        assert not logdir.exists()
        (roots,) = _resolve_grpc_log_allowed_roots(
            self._args(ft_per_cycle_applog_prefix=str(logdir / "train.log"))
        )
        assert logdir.is_dir(), "root directory must exist so its resolution is stable"
        # The job's own log path still validates.
        cycle_log = str(logdir / "train_cycle0.log")
        assert resolve_under_allowed_roots(cycle_log, [roots]) == cycle_log

    def test_server_own_log_dir_is_not_an_allowed_root(self, tmp_path):
        """`_root.log`/`_leaf_N.log` are written by the launcher redirecting each
        subprocess's stdout/stderr, never through a LogChunk, so that directory must not
        be exposed to gRPC clients."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import _resolve_grpc_log_allowed_roots

        applog_dir = tmp_path / "applogs"
        srv_dir = tmp_path / "serverlogs"
        roots = _resolve_grpc_log_allowed_roots(
            self._args(
                ft_per_cycle_applog_prefix=str(applog_dir / "train.log"),
                ft_log_server_log_prefix=str(srv_dir / "grpc"),
            )
        )
        assert roots == [os.path.realpath(str(applog_dir))]
        assert os.path.realpath(str(srv_dir)) not in roots

    def test_no_configured_paths_yields_no_roots(self):
        from nvidia_resiliency_ext.fault_tolerance.launcher import _resolve_grpc_log_allowed_roots

        assert _resolve_grpc_log_allowed_roots(self._args()) == []
