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
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext import fault_tolerance
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.shared_utils.health_check import AttrSvcResult

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
        launcher._register_ft_rdzv_handler("barrier")

    apply_c10d_patch.assert_called_once()
    register.assert_called_once()
    assert register.call_args.args[0] == "c10d"


def test_legacy_rdzv_impl_injects_use_libuv_false():
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
            "--ft-rdzv-impl",
            "legacy",
            "train.py",
        ]
    )

    with patch.object(launcher.LocalElasticAgent, "setup_rank_monitors_early", return_value={}):
        config, _, _ = launcher.config_from_args(args)

    assert config.rdzv_configs["use_libuv"] is False


def test_fact_url_starts_launcher_managed_fact_agent(tmp_path):
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
            "--ft-fact-url",
            "http://fact.example:8001/latest",
            "train.py",
        ]
    )
    fact_agent_manager = MagicMock()
    fact_agent_manager.start_if_needed.return_value = SimpleNamespace(
        socket_path=str(tmp_path / "managed-fact-agent.sock")
    )

    with (
        patch.object(
            launcher.LocalElasticAgent,
            "setup_rank_monitors_early",
            return_value={},
        ),
        patch.object(
            launcher,
            "FactAgentManager",
            return_value=fact_agent_manager,
        ) as manager_cls,
        patch.object(launcher, "_FACT_AGENT_MANAGER", None),
        patch.dict(
            os.environ,
            {"SLURM_JOB_USER": "slurm-user", "SLURM_CLUSTER_NAME": "slurm-cluster"},
            clear=False,
        ),
    ):
        config, _, _ = launcher.config_from_args(args)

    manager_cls.assert_called_once()
    manager_kwargs = manager_cls.call_args.kwargs
    assert manager_kwargs["fact_url"] == "http://fact.example:8001/latest"
    assert manager_kwargs["socket_path"] is None
    assert manager_kwargs["rpc_timeout_s"] == 2.0
    assert manager_kwargs["run_id"] == "none"
    assert manager_kwargs["rdzv_endpoint"] == "127.0.0.1:29500"
    assert manager_kwargs["store_timeout_s"] == 60.0
    assert manager_kwargs["is_store_host"] is True
    assert manager_kwargs["job_id"] == "none"
    assert manager_kwargs["ranks_per_node"] == 1
    assert manager_kwargs["username"] == "slurm-user"
    assert manager_kwargs["cluster"] == "slurm-cluster"
    assert manager_kwargs["health_log_prefix"] is None
    assert manager_kwargs["dmesg_artifact_enabled"] is False
    assert manager_kwargs["result_artifact_enabled"] is False
    assert manager_kwargs["grpc_server_address"] is None
    assert manager_kwargs["grpc_node_id"] is None
    assert manager_kwargs["fact_history_es_url"] is None
    assert manager_kwargs["fact_history_es_auth_file"] is None
    fact_agent_manager.start_if_needed.assert_called_once()
    assert config.fault_tol_cfg.fact_agent_socket_path == str(tmp_path / "managed-fact-agent.sock")


def test_fact_agent_start_failure_is_nonfatal(caplog):
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
            "--ft-fact-url",
            "http://fact.example:8001/latest",
            "train.py",
        ]
    )
    fact_agent_manager = MagicMock()
    fact_agent_manager.start_if_needed.side_effect = RuntimeError("boom")

    with (
        patch.object(
            launcher.LocalElasticAgent,
            "setup_rank_monitors_early",
            return_value={},
        ),
        patch.object(launcher, "FactAgentManager", return_value=fact_agent_manager),
        patch.object(launcher, "_FACT_AGENT_MANAGER", None),
        caplog.at_level(logging.WARNING),
    ):
        config, _, _ = launcher.config_from_args(args)

    fact_agent_manager.start_if_needed.assert_called_once()
    assert config.fault_tol_cfg.fact_url == "http://fact.example:8001/latest"
    assert "Failed to start local nvrx-fact-agent" in caplog.text


def test_fact_result_artifact_flag_passes_to_fact_agent_manager(tmp_path):
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
            "--ft-fact-url",
            "http://fact.example:8001/latest",
            "--ft-health-log-prefix",
            str(tmp_path / "job_health.log"),
            "--ft-enable-fact-result-artifact",
            "true",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-enable-log-server",
            "true",
            "train.py",
        ]
    )
    fact_agent_manager = MagicMock()
    grpc_proc = MagicMock()

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

    with (
        patch.object(
            launcher.LocalElasticAgent,
            "setup_rank_monitors_early",
            return_value={},
        ),
        patch.object(launcher, "PipeBasedLogsSpecs", FakePipeBasedLogsSpecs),
        patch.object(launcher, "_start_grpc_log_servers", return_value=[grpc_proc]),
        patch.object(launcher, "FactAgentManager", return_value=fact_agent_manager) as manager_cls,
        patch.object(launcher, "_FACT_AGENT_MANAGER", None),
        patch.object(launcher, "_GRPC_SERVER_PROCESSES", None),
    ):
        launcher.config_from_args(args)

    assert manager_cls.call_args.kwargs["result_artifact_enabled"] is True
    assert manager_cls.call_args.kwargs["grpc_server_address"] == "localhost:50051"


def test_fact_result_artifact_requires_grpc_log_aggregation(tmp_path):
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
            "--ft-fact-url",
            "http://fact.example:8001/latest",
            "--ft-health-log-prefix",
            str(tmp_path / "job_health.log"),
            "--ft-enable-fact-result-artifact",
            "true",
            "train.py",
        ]
    )

    with pytest.raises(ValueError, match="require gRPC log aggregation"):
        launcher.config_from_args(args)


def test_fact_dmesg_artifact_requires_grpc_log_aggregation(tmp_path):
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
            "--ft-fact-url",
            "http://fact.example:8001/latest",
            "--ft-health-log-prefix",
            str(tmp_path / "job_health.log"),
            "--ft-enable-health-log-dmesg",
            "true",
            "train.py",
        ]
    )

    with pytest.raises(ValueError, match="require gRPC log aggregation"):
        launcher.config_from_args(args)


def test_fact_agent_notification_includes_cycle_start_time():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    agent = launcher.LocalElasticAgent.__new__(launcher.LocalElasticAgent)
    agent._ft_cfg = SimpleNamespace(
        fact_url="http://fact.example/latest",
        fact_agent_socket_path="/tmp/fact-agent.sock",
        fact_agent_rpc_timeout=2.0,
        fact_policy_ready_timeout=60.0,
    )
    agent._is_store_host = True
    agent._node_id = "node-a"
    agent._fact_agent_cycle_start_time = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    agent._rdzv_handler = SimpleNamespace(
        _this_node=SimpleNamespace(addr="node-a"),
        get_active_node_addrs=lambda: ["node-a:29500", "node-b:29500"],
    )

    with patch.object(
        launcher,
        "notify_fact_agent",
        return_value={"accepted": True},
    ) as notify:
        launcher.LocalElasticAgent._notify_fact_agent(agent, SimpleNamespace(), 3)

    payload = notify.call_args.kwargs["payload"]
    cycle_end_time = datetime.fromisoformat(payload.pop("cycle_end_time"))
    assert cycle_end_time.tzinfo is not None
    assert payload == {
        "event": "cycle_failed",
        "cycle": 3,
        "cycle_start_time": "2026-05-10T12:00:00+00:00",
        "expected_nodes": ["node-a", "node-b"],
    }


def test_fact_agent_rpc_timeout_must_be_positive():
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
            "--ft-fact-url",
            "http://fact.example:8001/latest",
            "--ft-fact-agent-rpc-timeout",
            "0",
            "train.py",
        ]
    )

    with pytest.raises(ValueError, match="--ft-fact-agent-rpc-timeout must be positive"):
        launcher.config_from_args(args)


def test_fact_policy_ready_timeout_must_be_non_negative():
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
            "--ft-fact-url",
            "http://fact.example:8001/latest",
            "--ft-fact-policy-ready-timeout",
            "-1",
            "train.py",
        ]
    )

    with pytest.raises(ValueError, match="--ft-fact-policy-ready-timeout must be non-negative"):
        launcher.config_from_args(args)


def test_fact_avoid_nodes_waits_for_ready_policy():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    agent = launcher.LocalElasticAgent.__new__(launcher.LocalElasticAgent)
    agent._ft_cfg = SimpleNamespace(
        fact_url="http://fact.example/latest",
        fact_agent_socket_path="/tmp/fact-agent.sock",
        fact_agent_rpc_timeout=2.0,
        fact_policy_ready_timeout=1.0,
    )
    agent._is_store_host = True
    agent._last_fact_agent_cycle = 7

    with patch.object(
        launcher,
        "notify_fact_agent",
        side_effect=[
            {"cycle_id": "7", "status": "pending", "avoid_nodes": []},
            {"cycle_id": "7", "status": "ready", "avoid_nodes": ["node-a"]},
        ],
    ) as notify, patch.object(launcher.time, "sleep") as sleep:
        nodes = launcher.LocalElasticAgent.get_fact_avoid_nodes_for_rendezvous(agent)

    assert nodes == ["node-a"]
    assert notify.call_count == 2
    sleep.assert_called_once()


def test_fact_avoid_nodes_does_not_wait_for_skipped_policy():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    agent = launcher.LocalElasticAgent.__new__(launcher.LocalElasticAgent)
    agent._ft_cfg = SimpleNamespace(
        fact_url="http://fact.example/latest",
        fact_agent_socket_path="/tmp/fact-agent.sock",
        fact_agent_rpc_timeout=2.0,
        fact_policy_ready_timeout=60.0,
    )
    agent._is_store_host = True
    agent._last_fact_agent_cycle = 7

    with patch.object(
        launcher,
        "notify_fact_agent",
        return_value={"cycle_id": "7", "status": "skipped", "avoid_nodes": []},
    ) as notify, patch.object(launcher.time, "sleep") as sleep:
        nodes = launcher.LocalElasticAgent.get_fact_avoid_nodes_for_rendezvous(agent)

    assert nodes == []
    notify.assert_called_once()
    sleep.assert_not_called()


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
    spec.rdzv_handler._attribution_service = None
    spec.rdzv_handler.round.return_value = rdzv_round
    spec.rdzv_handler.get_active_node_addrs.return_value = ["node001", "node002"]
    spec.rdzv_handler.get_standby_node_addrs.return_value = ["node003"]
    spec.rdzv_handler.get_active_ranks.return_value = [0, 1]
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
        """Returns False without restarting when progress tracker says terminate early."""
        agent = self._make_agent()
        agent._rdzv_handler._attribution_service = None
        agent._progress_tracker = MagicMock()
        agent._progress_tracker.should_terminate_early.return_value = True
        agent._remaining_restarts = 2

        with (
            patch.object(agent, '_restart_workers') as mock_restart,
            patch.object(agent, '_open_rendezvous_for_restart') as mock_open,
        ):
            result = agent._handle_restart_decision(
                role="test", spec=self.spec, log_msg="[%s] restarting"
            )

        self.assertFalse(result)
        mock_restart.assert_not_called()
        mock_open.assert_not_called()

    def test_handle_restart_decision_restarts_remaining(self):
        """Returns True and decrements _remaining_restarts when restarts are available."""
        agent = self._make_agent()
        agent._rdzv_handler._attribution_service = None
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

    def test_handle_restart_decision_attrsvc_stop_blocks_restart(self):
        """Returns False without consuming a restart when attrsvc recommends STOP."""
        agent = self._make_agent()
        attrsvc = MagicMock()
        attrsvc.get_last_result.return_value = AttrSvcResult(
            result={"state": "STOP"},
            recommendation="STOP",
            should_stop=True,
            log_path="/path/to/cycle_0.log",
        )
        agent._rdzv_handler._attribution_service = attrsvc
        agent._progress_tracker = MagicMock()
        agent._progress_tracker.should_terminate_early.return_value = False
        agent._remaining_restarts = 2

        with (
            patch.object(agent, '_restart_workers') as mock_restart,
            patch.object(agent, '_open_rendezvous_for_restart') as mock_open,
        ):
            result = agent._handle_restart_decision(
                role="test", spec=self.spec, log_msg="[%s] restarting", open_rendezvous=True
            )

        self.assertFalse(result)
        self.assertEqual(agent._remaining_restarts, 2)
        attrsvc.get_last_result.assert_called_once()
        mock_restart.assert_not_called()
        mock_open.assert_not_called()

    def test_handle_restart_decision_no_restarts_left(self):
        """Returns False when _remaining_restarts is 0."""
        agent = self._make_agent()
        agent._rdzv_handler._attribution_service = None
        agent._progress_tracker = MagicMock()
        agent._progress_tracker.should_terminate_early.return_value = False
        agent._remaining_restarts = 0

        with patch.object(agent, '_restart_workers') as mock_restart:
            result = agent._handle_restart_decision(
                role="test", spec=self.spec, log_msg="[%s] restarting"
            )

        self.assertFalse(result)
        mock_restart.assert_not_called()

    def test_handle_restart_decision_open_rendezvous_called_when_requested(self):
        """Calls _open_rendezvous_for_restart() when open_rendezvous=True."""
        agent = self._make_agent()
        agent._rdzv_handler._attribution_service = None
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

    def test_open_rendezvous_for_restart_legacy_handler(self):
        """Does nothing (no error) when handler lacks _barrier_state (legacy rdzv)."""
        from nvidia_resiliency_ext.fault_tolerance.launcher import LocalElasticAgent

        # Use a spec-constrained mock so _barrier_state doesn't auto-exist
        legacy_rdzv = MagicMock(
            spec=[
                'round',
                'get_active_node_addrs',
                'get_standby_node_addrs',
            ]
        )
        legacy_rdzv.round.return_value = 1
        legacy_rdzv.get_active_node_addrs.return_value = []
        legacy_rdzv.get_standby_node_addrs.return_value = []
        self.spec.rdzv_handler = legacy_rdzv

        agent = LocalElasticAgent(
            spec=self.spec,
            fault_tol_cfg=self.fault_tol_cfg,
            logs_specs=self.logs_specs,
        )
        # Should not raise
        agent._open_rendezvous_for_restart()
        # No open_rendezvous() on the legacy handler
        self.assertFalse(hasattr(legacy_rdzv, '_barrier_state'))


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

    args = SimpleNamespace(
        nnodes="2",
        ft_log_server_graceful_shutdown_timeout=60.0,
        ft_log_leaf_max_queue_chunks=-1,
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


def test_main_stops_fact_agent_before_grpc_log_servers():
    from nvidia_resiliency_ext.fault_tolerance import launcher

    order = []
    args = SimpleNamespace(ft_log_server_graceful_shutdown_timeout=1.25)
    grpc_proc = object()
    fact_manager = MagicMock()
    attr_manager = MagicMock()
    fact_manager.stop.side_effect = lambda: order.append("fact")
    attr_manager.stop.side_effect = lambda: order.append("attr")

    def stop_grpc(procs, timeout):
        assert procs == [grpc_proc]
        assert timeout == 1.25
        order.append("grpc")

    with (
        patch.object(launcher, "parse_args", return_value=args),
        patch.object(launcher, "run", return_value=None),
        patch.object(launcher, "stop_grpc_log_servers", side_effect=stop_grpc),
        patch.object(launcher.sys, "exit", side_effect=SystemExit) as sys_exit,
        patch.object(launcher, "_FACT_AGENT_MANAGER", fact_manager),
        patch.object(launcher, "_ATTRIBUTION_MANAGER", attr_manager),
        patch.object(launcher, "_GRPC_SERVER_PROCESSES", [grpc_proc]),
        pytest.raises(SystemExit),
    ):
        launcher.main([])

    assert order == ["fact", "attr", "grpc"]
    sys_exit.assert_called_once_with(0)


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


def test_launch_agent_store_host_closes_cycle_info_reporter_without_rdzv_shutdown():
    from types import SimpleNamespace

    from nvidia_resiliency_ext.fault_tolerance import launcher

    rdzv_handler = MagicMock()
    rdzv_handler._attribution_service = None
    spec = SimpleNamespace(rdzv_handler=rdzv_handler)
    agent = MagicMock()
    agent.run.side_effect = launcher.UnhealthyNodeException("node unhealthy")
    agent._rdzv_handler = rdzv_handler

    config = SimpleNamespace(
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

    with (
        patch.object(launcher, "_get_addr_and_port", return_value=("host", 29500)),
        patch.object(launcher, "_is_store_host", return_value=True),
        patch.object(launcher, "WorkerSpec", return_value=spec),
        patch.object(launcher.rdzv_registry, "get_rendezvous_handler", return_value=rdzv_handler),
        patch.object(launcher, "LocalElasticAgent", return_value=agent),
        patch.object(launcher.metrics, "initialize_metrics"),
        patch.object(launcher.events, "record"),
        patch.object(launcher, "is_slurm_job_array", return_value=False),
        patch.object(launcher.time, "sleep", return_value=None),
    ):
        result = launcher.launch_agent(config, "train.py", [])

    assert result is None
    rdzv_handler.shutdown.assert_not_called()
    rdzv_handler.shutdown_cycle_info_reporter.assert_called_once()
