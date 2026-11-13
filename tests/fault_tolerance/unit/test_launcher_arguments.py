# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boolean CLI values must be explicit and consistent before services start."""

import pytest

from nvidia_resiliency_ext.fault_tolerance import launcher

BOOLEAN_OPTIONS = (
    "--ft-enable-log-server",
    "--ft-enable-nic-monitor",
    "--ft-enable-nic-healthcheck",
    "--ft-enable-dist-storage-healthcheck",
    "--ft-skip-section-response",
    "--ft-numa-bind-strict",
    "--ft-check-remaining-processes",
    "--ft-install-exception-hook",
)


@pytest.mark.parametrize("option", BOOLEAN_OPTIONS)
@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("FALSE", False), ("1", True), ("0", False), ("yes", True), ("no", False)],
)
def test_boolean_options_accept_consistent_explicit_values(option, value, expected):
    args = launcher.parse_args([f"{option}={value}", "train.py"])
    assert getattr(args, option[2:].replace("-", "_")) is expected


@pytest.mark.parametrize("option", BOOLEAN_OPTIONS)
@pytest.mark.parametrize("value", ["typo", "2", ""])
def test_invalid_boolean_values_fail_before_launch(option, value, capsys):
    with pytest.raises(SystemExit) as exc:
        launcher.parse_args([f"{option}={value}", "train.py"])
    assert exc.value.code == 2
    assert "expected a boolean value" in capsys.readouterr().err


@pytest.mark.parametrize("option", BOOLEAN_OPTIONS)
def test_omitted_boolean_options_keep_configuration_defaults(option):
    args = launcher.parse_args(["train.py"])
    assert getattr(args, option[2:].replace("-", "_")) is None


@pytest.mark.parametrize("value,expected", [("yes", True), ("0", False)])
def test_log_server_environment_default_uses_boolean_validation(monkeypatch, value, expected):
    monkeypatch.setenv("PET_FT_ENABLE_LOG_SERVER", value)
    assert launcher.parse_args(["train.py"]).ft_enable_log_server is expected


def test_invalid_log_server_environment_default_is_rejected_unless_overridden(monkeypatch):
    monkeypatch.setenv("PET_FT_ENABLE_LOG_SERVER", "typo")
    with pytest.raises(SystemExit) as exc:
        launcher.parse_args(["train.py"])
    assert exc.value.code == 2
    assert (
        launcher.parse_args(["--ft-enable-log-server=false", "train.py"]).ft_enable_log_server
        is False
    )
