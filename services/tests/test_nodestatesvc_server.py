# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from unittest.mock import patch

from nvidia_resiliency_ext.services.nodestatesvc.server import NodeStateService, run_server
from nvidia_resiliency_ext.services.nodestatesvc.slurm import NodeStateRecord


class FakeNodeStateClient:
    def __init__(self):
        self.queries = []

    def check_available(self):
        return True, None

    def get_node_states(self, nodes):
        self.queries.append(list(nodes))
        return {
            node: NodeStateRecord(
                node=node,
                state="DRAIN" if node == "node-b" else "ALLOCATED",
                raw_state="drain" if node == "node-b" else "allocated",
                bad=node == "node-b",
                reason="test",
            )
            for node in nodes
        }


class BlockingNodeStateClient(FakeNodeStateClient):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def get_node_states(self, nodes):
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().get_node_states(nodes)


def wait_for_cycle_status(service, job_id="job-123", cycle_id="3", *, timeout=5.0):
    deadline = time.monotonic() + timeout
    last_status = None
    last_body = None
    while time.monotonic() < deadline:
        last_status, last_body = service.query_cycle_node_states(job_id, cycle_id)
        if last_status != 409:
            return last_status, last_body
        time.sleep(0.01)
    return last_status, last_body


def test_node_state_service_health_readiness_and_query():
    service = NodeStateService(FakeNodeStateClient())

    assert service.health()["status"] == "ok"
    ready_status, ready_body = service.readiness()
    assert ready_status == 200
    assert ready_body["backend"] == "slurm"

    status, body = service.query_node_states(["node-a", "node-b"])
    assert status == 200
    assert body["bad_nodes"] == ["node-b"]
    assert json.dumps(body)


def test_node_state_service_cycle_start_end_and_compact_status():
    client = FakeNodeStateClient()
    service = NodeStateService(client)

    status, body = service.register_cycle_start("job-123", "3", ["node-a", "node-b", "node-a"])

    assert status == 200
    assert body["job_id"] == "job-123"
    assert body["cycle_id"] == "3"
    assert body["registered_nodes"] == 2

    status, body = service.register_cycle_end("job-123", "3")

    assert status == 202
    assert body["job_id"] == "job-123"
    assert body["accepted"] is True
    assert body["materializing"] is True

    status, body = wait_for_cycle_status(service)

    assert status == 200
    assert body["job_id"] == "job-123"
    assert body["cycle_id"] == "3"
    assert body["requested_nodes"] == 2
    assert body["bad_nodes"] == ["node-b"]
    assert body["unknown_nodes"] == []
    assert body["nodes"] == [
        {
            "node": "node-b",
            "state": "DRAIN",
            "raw_state": "drain",
            "bad": True,
            "reason": "test",
            "slurm_visible": True,
        }
    ]
    assert client.queries == [["node-a", "node-b"]]


def test_node_state_service_cycle_status_before_end_is_not_ready():
    service = NodeStateService(FakeNodeStateClient())

    status, _ = service.register_cycle_start("job-123", "3", ["node-a"])
    assert status == 200

    status, body = service.query_cycle_node_states("job-123", "3")

    assert status == 409
    assert body["error"] == "cycle_status_not_ready"


def test_node_state_service_duplicate_cycle_end_does_not_duplicate_query():
    client = BlockingNodeStateClient()
    service = NodeStateService(client)
    status, _ = service.register_cycle_start("job-123", "3", ["node-a", "node-b"])
    assert status == 200

    status, body = service.register_cycle_end("job-123", "3")
    assert status == 202
    assert body["accepted"] is True
    assert body["materializing"] is True
    assert client.started.wait(timeout=5)

    status, body = service.query_cycle_node_states("job-123", "3")
    assert status == 409
    assert body["error"] == "cycle_status_not_ready"

    status, body = service.register_cycle_end("job-123", "3")

    assert status == 202
    assert body["accepted"] is True
    assert body["materializing"] is True
    assert client.queries == []

    client.release.set()
    status, body = wait_for_cycle_status(service)

    assert status == 200
    assert client.queries == [["node-a", "node-b"]]


def test_node_state_service_unknown_cycle_returns_404():
    service = NodeStateService(FakeNodeStateClient())

    status, body = service.query_cycle_node_states("job-123", "missing")

    assert status == 404
    assert body["error"] == "cycle_not_found"


def test_node_state_service_evicts_old_cycles():
    service = NodeStateService(FakeNodeStateClient(), max_cycles=1)

    status, _ = service.register_cycle_start("job-123", "1", ["node-a"])
    assert status == 200
    status, _ = service.register_cycle_start("job-123", "2", ["node-b"])
    assert status == 200

    status, body = service.query_cycle_node_states("job-123", "1")
    assert status == 404
    assert body["error"] == "cycle_not_found"

    status, body = service.query_cycle_node_states("job-123", "2")
    assert status == 409
    assert body["error"] == "cycle_status_not_ready"


@patch("nvidia_resiliency_ext.services.nodestatesvc.server.ThreadingHTTPServer")
def test_run_server_binds_requested_host_and_port(mock_server):
    service = NodeStateService(FakeNodeStateClient())

    run_server("127.0.0.1", 8123, service)

    mock_server.assert_called_once()
    assert mock_server.call_args.args[0] == ("127.0.0.1", 8123)
