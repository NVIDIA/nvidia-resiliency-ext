# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attrsvc process-start diagnostics and filesystem context."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvidia_resiliency_ext.attribution.restart_agent import load_restart_agent_config
from nvidia_resiliency_ext.services.attrsvc import app as attrsvc_app
from nvidia_resiliency_ext.services.attrsvc.config import ServiceEndpoint


def test_configuration_failure_logs_traceback_and_offending_path(
    tmp_path,
    monkeypatch,
    caplog,
):
    missing_root = tmp_path / "missing-log-root"

    def fail_setup():
        raise FileNotFoundError(2, "No such directory", str(missing_root))

    monkeypatch.setattr(attrsvc_app, "setup", fail_setup)
    run = Mock()
    monkeypatch.setattr(attrsvc_app.uvicorn, "run", run)
    caplog.set_level(logging.ERROR)

    with pytest.raises(SystemExit) as raised:
        attrsvc_app.main()

    assert raised.value.code == 1
    assert "event=attrsvc.startup.failed" in caplog.text
    assert "phase=configuration" in caplog.text
    assert str(missing_root) in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    run.assert_not_called()


def test_endpoint_failure_logs_socket_path(tmp_path, monkeypatch, caplog):
    socket_path = tmp_path / "attrsvc.sock"
    socket_path.write_text("not a socket", encoding="utf-8")
    cfg = SimpleNamespace(
        LOG_LEVEL="INFO",
        SERVICE_ENDPOINT=ServiceEndpoint(uds_path=str(socket_path)),
    )
    monkeypatch.setattr(attrsvc_app, "setup", lambda: cfg)
    run = Mock()
    monkeypatch.setattr(attrsvc_app.uvicorn, "run", run)
    caplog.set_level(logging.ERROR)

    with pytest.raises(SystemExit) as raised:
        attrsvc_app.main()

    assert raised.value.code == 1
    assert "phase=endpoint" in caplog.text
    assert str(socket_path) in caplog.text
    assert "endpoint exists and is not a socket" in caplog.text
    run.assert_not_called()


def test_invalid_restart_agent_config_reports_source_path(tmp_path):
    config_path = tmp_path / "restart_agent.json"
    config_path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(ValueError, match=str(config_path)) as raised:
        load_restart_agent_config(config_path)

    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
