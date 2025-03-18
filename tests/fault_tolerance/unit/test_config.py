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

import logging
import os
import signal
import tempfile
from argparse import ArgumentParser
from contextlib import contextmanager

import pytest

from nvidia_resiliency_ext import fault_tolerance


def test_from_kwargs():
    ref_conf = fault_tolerance.FaultToleranceConfig()

    conf_with_item_modified = fault_tolerance.FaultToleranceConfig.from_kwargs(
        rank_heartbeat_timeout=ref_conf.rank_heartbeat_timeout + 1,
        ignore_not_recognized=False,
    )
    assert conf_with_item_modified.rank_heartbeat_timeout == ref_conf.rank_heartbeat_timeout + 1

    not_modified_conf = fault_tolerance.FaultToleranceConfig.from_kwargs(
        ignore_not_recognized=False
    )
    assert not_modified_conf.rank_heartbeat_timeout == ref_conf.rank_heartbeat_timeout

    with pytest.raises(ValueError):
        _ = fault_tolerance.FaultToleranceConfig.from_kwargs(
            an_unknown_arg=True, ignore_not_recognized=False
        )

    # uknown arg should be ignored due to ignore_not_recognized=True
    _ = fault_tolerance.FaultToleranceConfig.from_kwargs(
        an_unknown_arg=True, ignore_not_recognized=True
    )


def test_from_args():
    parser = ArgumentParser(description="Test parser")
    parser.add_argument(
        "--ft-param-safety_factor",
        "--ft-param-safety_factor",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ft-param-rank_termination_signal",
        "--ft-param-rank_termination_signal",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ft-param-log_level",
        "--ft-param-log_level",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ft-param-rank_out_of_section_timeout",
        "--ft-param-rank_out_of_section_timeout",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ft-param-rank_section_timeouts",
        "--ft-param-rank_section_timeouts",
        type=str,
        default=None,
    )
    inp = [
        "--ft-param-safety_factor",
        "0.567",
        "--ft-param-rank_termination_signal",
        "SIGUSR2",
        "--ft-param-log_level",
        "DEBUG",
        "--ft-param-rank_out_of_section_timeout",
        "123.0",
        "--ft-param-rank_section_timeouts",
        "custom1:111.1,custom2:222.2",
    ]
    args = parser.parse_args(inp)
    ft = fault_tolerance.FaultToleranceConfig.from_args(
        args=args, cfg_file_arg=None, ft_args_prefix='ft_param_'
    )
    assert ft.safety_factor == 0.567
    assert ft.rank_termination_signal == signal.SIGUSR2
    assert ft.log_level == logging.DEBUG
    assert ft.rank_out_of_section_timeout == 123.0
    assert ft.rank_section_timeouts == {'custom1': 111.1, 'custom2': 222.2}


def test_signal_field_with_valid_values():
    ref_conf = fault_tolerance.FaultToleranceConfig()
    assert isinstance(ref_conf.rank_termination_signal, signal.Signals)
    assert (
        fault_tolerance.FaultToleranceConfig(rank_termination_signal=9).rank_termination_signal
        is signal.SIGKILL
    )
    assert (
        fault_tolerance.FaultToleranceConfig(
            rank_termination_signal='SIGKILL'
        ).rank_termination_signal
        is signal.SIGKILL
    )
    assert (
        fault_tolerance.FaultToleranceConfig(
            rank_termination_signal=signal.SIGKILL
        ).rank_termination_signal
        is signal.SIGKILL
    )


def test_signal_field_with_invalid_values():
    with pytest.raises(ValueError):
        fault_tolerance.FaultToleranceConfig(rank_termination_signal=None)
    with pytest.raises(ValueError):
        fault_tolerance.FaultToleranceConfig(rank_termination_signal=-999)
    with pytest.raises(ValueError):
        fault_tolerance.FaultToleranceConfig(rank_termination_signal="not a signal name")


@pytest.mark.parametrize(
    "level_to_set, expected_level",
    [
        ("deBUG", logging.DEBUG),
        ("Info", logging.INFO),
        ("WARNING", logging.WARNING),
        ("critical", logging.CRITICAL),
        (logging.INFO, logging.INFO),
    ],
)
def test_log_level_field_with_valid_values(level_to_set, expected_level):
    assert fault_tolerance.FaultToleranceConfig(log_level=level_to_set).log_level == expected_level


def test_log_level_field_with_invalid_values():
    with pytest.raises(ValueError):
        fault_tolerance.FaultToleranceConfig(log_level=None)
    with pytest.raises(ValueError):
        fault_tolerance.FaultToleranceConfig(log_level=-999)
    with pytest.raises(ValueError):
        fault_tolerance.FaultToleranceConfig(log_level="not a loglevel")


@contextmanager
def tmp_yaml_file(lines):
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_file.write('\n'.join(lines))
    temp_file.close()
    try:
        yield temp_file.name
    finally:
        os.remove(temp_file.name)


def test_read_from_yaml():
    YAML_LINES = [
        "fault_tolerance:",
        "    initial_rank_heartbeat_timeout: 987",
        "    rank_heartbeat_timeout: 121212",
        "    rank_termination_signal: SIGUSR2",
        "    rank_section_timeouts:",
        "        custom1: 111.0",
        "        custom2: 222.0",
        "    rank_out_of_section_timeout: 333.0",
    ]
    with tmp_yaml_file(YAML_LINES) as temp_file:
        ft = fault_tolerance.FaultToleranceConfig.from_yaml_file(temp_file)
        assert ft.initial_rank_heartbeat_timeout == 987
        assert ft.rank_heartbeat_timeout == 121212
        assert ft.rank_termination_signal == signal.SIGUSR2
        assert ft.rank_section_timeouts == {'custom1': 111.0, 'custom2': 222.0}
        assert ft.rank_out_of_section_timeout == 333.0


def test_read_from_yaml_nested():
    YAML_LINES = [
        "some_other_section:",
        "   fault_tolerance:",
        "       initial_rank_heartbeat_timeout: 987",
        "       rank_heartbeat_timeout: 121212",
        "       rank_termination_signal: SIGUSR2",
    ]
    with tmp_yaml_file(YAML_LINES) as temp_file:
        ft = fault_tolerance.FaultToleranceConfig.from_yaml_file(temp_file)
        assert ft.initial_rank_heartbeat_timeout == 987
        assert ft.rank_heartbeat_timeout == 121212
        assert ft.rank_termination_signal == signal.SIGUSR2


def test_fails_when_missing_ft_section():
    YAML_LINES = [
        "some_other_section:",
        "   almost_like_fault_tolerance:",
        "       initial_rank_heartbeat_timeout: 987",
        "       rank_heartbeat_timeout: 121212",
        "       rank_termination_signal: SIGUSR2",
    ]
    with tmp_yaml_file(YAML_LINES) as temp_file:
        with pytest.raises(ValueError):
            _ = fault_tolerance.FaultToleranceConfig.from_yaml_file(temp_file)


def test_fails_when_ft_section_has_invald_items():
    YAML_LINES = [
        "fault_tolerance:",
        "    unknown_config_item: 987",
        "    rank_heartbeat_timeout: 121212",
        "    rank_termination_signal: SIGUSR2",
    ]
    with tmp_yaml_file(YAML_LINES) as temp_file:
        with pytest.raises(ValueError):
            _ = fault_tolerance.FaultToleranceConfig.from_yaml_file(
                temp_file, ignore_not_recognized=False
            )


def test_to_yaml_file():
    ref_conf = fault_tolerance.FaultToleranceConfig()
    ref_conf.rank_termination_signal = signal.SIGUSR2
    ref_conf.log_level = logging.FATAL
    ref_conf.rank_heartbeat_timeout = 123.0
    ref_conf.rank_section_timeouts == {'custom1': 111.0, 'custom2': 222.0}
    with tempfile.NamedTemporaryFile() as temp_file:
        ref_conf.to_yaml_file(temp_file.name)
        restored_conf = fault_tolerance.FaultToleranceConfig.from_yaml_file(temp_file.name)
    assert restored_conf == ref_conf
