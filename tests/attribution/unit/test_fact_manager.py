# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from nvidia_resiliency_ext.attribution.fact import manager as fact_manager


def test_fact_agent_manager_disabled_when_fact_url_absent():
    manager = fact_manager.FactAgentManager(fact_url=None)

    assert manager.start_if_needed() is None


def test_fact_agent_manager_starts_agent_and_waits_for_ping(tmp_path):
    socket_path = str(tmp_path / "fact-agent.sock")
    log_file = str(tmp_path / "fact-agent.log")
    process = MagicMock()
    process.pid = 123
    process.poll.return_value = None

    with (
        patch.object(fact_manager, "_fact_agent_command", return_value=["nvrx-fact-agent"]),
        patch.object(fact_manager.subprocess, "Popen", return_value=process) as popen,
        patch.object(fact_manager, "notify_fact_agent", return_value={"accepted": True}) as notify,
    ):
        manager = fact_manager.FactAgentManager(
            fact_url="http://fact.example:8001/latest",
            socket_path=socket_path,
            startup_timeout_s=0.5,
            log_file=log_file,
        )

        endpoint = manager.start_if_needed()

    assert endpoint is not None
    assert endpoint.socket_path == socket_path
    assert popen.call_args.args[0] == [
        "nvrx-fact-agent",
        "--fact-url",
        "http://fact.example:8001/latest",
        "--socket-path",
        socket_path,
    ]
    notify.assert_called_with(
        socket_path=socket_path,
        payload={"event": "ping"},
        timeout_s=manager.rpc_timeout_s,
    )


def test_fact_agent_manager_stops_agent_on_startup_failure(tmp_path):
    socket_path = str(tmp_path / "fact-agent.sock")
    log_file = str(tmp_path / "fact-agent.log")
    process = MagicMock()
    process.pid = 123
    process.poll.return_value = None

    with (
        patch.object(fact_manager, "_fact_agent_command", return_value=["nvrx-fact-agent"]),
        patch.object(fact_manager.subprocess, "Popen", return_value=process),
        patch.object(fact_manager, "notify_fact_agent", side_effect=ConnectionRefusedError("nope")),
        patch.object(fact_manager.time, "sleep", return_value=None),
    ):
        manager = fact_manager.FactAgentManager(
            fact_url="http://fact.example:8001/latest",
            socket_path=socket_path,
            startup_timeout_s=0.1,
            log_file=log_file,
        )

        try:
            manager.start_if_needed()
        except TimeoutError:
            pass
        else:
            raise AssertionError("expected startup timeout")

    process.terminate.assert_called_once()
    process.wait.assert_called()
