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


def test_fact_agent_manager_passes_session_args(tmp_path):
    socket_path = str(tmp_path / "fact-agent.sock")
    log_file = str(tmp_path / "fact-agent.log")
    process = MagicMock()
    process.pid = 123
    process.poll.return_value = None

    with (
        patch.object(fact_manager, "_fact_agent_command", return_value=["nvrx-fact-agent"]),
        patch.object(fact_manager.subprocess, "Popen", return_value=process) as popen,
        patch.object(fact_manager, "notify_fact_agent", return_value={"accepted": True}),
    ):
        manager = fact_manager.FactAgentManager(
            fact_url="http://fact.example:8001/latest",
            socket_path=socket_path,
            startup_timeout_s=0.5,
            log_file=log_file,
            run_id="run-1",
            rdzv_endpoint="store-host:29500",
            store_timeout_s=12.0,
            local_node="node-a",
            is_store_host=True,
            job_id="job-1",
            ranks_per_node=4,
            username="slurm-user",
            cluster="slurm-cluster",
            health_log_prefix="/logs/job_health.log",
            dmesg_artifact_enabled=True,
            result_artifact_enabled=True,
            grpc_server_address="store-host:50051",
            grpc_node_id="node-a_123",
            fact_history_es_url="http://history.example",
            fact_history_es_auth_file="/tmp/history.auth",
            fact_history_lookback="14d",
            fact_history_index="history-*",
            fact_history_max_candidate_nodes=16,
            fact_history_query_timeout_s=30.0,
            fact_min_repeat_count_for_avoid=2,
            fact_max_attribution_avoids_per_cycle=1,
        )

        manager.start_if_needed()

    assert popen.call_args.args[0] == [
        "nvrx-fact-agent",
        "--fact-url",
        "http://fact.example:8001/latest",
        "--socket-path",
        socket_path,
        "--run-id",
        "run-1",
        "--rdzv-endpoint",
        "store-host:29500",
        "--store-timeout",
        "12.0",
        "--local-node",
        "node-a",
        "--is-store-host",
        "--job-id",
        "job-1",
        "--ranks-per-node",
        "4",
        "--username",
        "slurm-user",
        "--cluster",
        "slurm-cluster",
        "--health-log-prefix",
        "/logs/job_health.log",
        "--dmesg-artifact-enabled",
        "--result-artifact-enabled",
        "--grpc-server-address",
        "store-host:50051",
        "--grpc-node-id",
        "node-a_123",
        "--fact-history-es-url",
        "http://history.example",
        "--fact-history-es-auth-file",
        "/tmp/history.auth",
        "--fact-history-lookback",
        "14d",
        "--fact-history-index",
        "history-*",
        "--fact-history-max-candidate-nodes",
        "16",
        "--fact-history-query-timeout",
        "30.0",
        "--fact-min-repeat-count-for-avoid",
        "2",
        "--fact-max-attribution-avoids-per-cycle",
        "1",
    ]


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


def test_fact_agent_manager_prefers_graceful_shutdown(tmp_path):
    socket_path = str(tmp_path / "fact-agent.sock")
    log_file = str(tmp_path / "fact-agent.log")
    process = MagicMock()
    process.pid = 123
    process.poll.return_value = None

    with patch.object(fact_manager, "notify_fact_agent", return_value={"accepted": True}) as notify:
        manager = fact_manager.FactAgentManager(
            fact_url="http://fact.example:8001/latest",
            socket_path=socket_path,
            log_file=log_file,
        )
        manager.process = process

        manager.stop()

    notify.assert_called_once_with(
        socket_path=socket_path,
        payload={"event": "shutdown"},
        timeout_s=manager.rpc_timeout_s,
    )
    process.terminate.assert_not_called()
    process.wait.assert_called_once_with(timeout=fact_manager._FACT_AGENT_STOP_TIMEOUT)
