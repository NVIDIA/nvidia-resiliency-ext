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

import pytest

from nvidia_resiliency_ext import fault_tolerance

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
