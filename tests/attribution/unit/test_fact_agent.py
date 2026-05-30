# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from nvidia_resiliency_ext.attribution.fact.agent import FactAgent, FactAgentKeys, FactAgentRequest
from nvidia_resiliency_ext.attribution.fact.client import FactAttributionResult
from nvidia_resiliency_ext.attribution.fact.models import FactHistoryRecord


class FakeStore:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value if isinstance(value, bytes) else str(value).encode("utf-8")

    def get(self, key):
        return self.data[key]

    def check(self, keys):
        return all(key in self.data for key in keys)

    def add(self, key, amount):
        value = int(self.data.get(key, b"0").decode("utf-8")) + amount
        self.data[key] = str(value).encode("utf-8")
        return value


class WaitStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.wait_calls = []

    def wait(self, keys, timeout):
        self.wait_calls.append((list(keys), timeout))
        if not self.check(keys):
            raise RuntimeError("timed out")


class FlakyCountReadStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.done_count_reads = 0

    def add(self, key, amount):
        if amount != 0:
            return super().add(key, amount)
        self.done_count_reads += 1
        if self.done_count_reads == 1:
            return 3
        raise RuntimeError("transient TCPStore read failure")


class FakeFactClient:
    def __init__(self):
        self.created = []
        self.submitted = []
        self.gets = []

    def create_failure_attributor(self, **kwargs):
        self.created.append(kwargs)
        return "att-1"

    def submit_dmesg_text_observation(self, **kwargs):
        self.submitted.append(kwargs)
        if "Xid" not in kwargs["dmesg_text"]:
            return None
        return f"obs-{kwargs['default_hostname']}"

    def get_attribution_result(self, *, attributor_id, observation_ids):
        self.gets.append({"attributor_id": attributor_id, "observation_ids": list(observation_ids)})
        return FactAttributionResult(
            attributor_id=attributor_id,
            observation_ids=list(observation_ids),
            faulty_nodes=["node-a"],
            attribution={"attributions": []},
        )


class FakeHistoryClient:
    def __init__(self, records):
        self.records = records
        self.queries = []

    def query_node_history(self, **kwargs):
        self.queries.append(kwargs)
        return list(self.records)


class FailingCreateFactClient(FakeFactClient):
    def create_failure_attributor(self, **kwargs):
        self.created.append(kwargs)
        raise RuntimeError("FACT unavailable")


class FlakyPostFactClient(FakeFactClient):
    def __init__(self):
        super().__init__()
        self.failures_left = 1

    def submit_dmesg_text_observation(self, **kwargs):
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("temporary overload")
        return super().submit_dmesg_text_observation(**kwargs)


class FakeGrpcWriter:
    def __init__(self):
        self.started = False
        self.shutdown_called = False
        self.join_timeout = None

    def start(self):
        self.started = True

    def shutdown(self):
        self.shutdown_called = True

    def join(self, timeout=None):
        self.join_timeout = timeout


def _recording_grpc_writer_factory():
    writer_records = []

    def grpc_writer_factory(write_queue, address, node_id, logger):
        writer = FakeGrpcWriter()
        writer_records.append(
            {
                "address": address,
                "node_id": node_id,
                "queue": write_queue,
                "writer": writer,
            }
        )
        return writer

    return writer_records, grpc_writer_factory


def _drain_grpc_writes(writer_records):
    writes = []
    for writer_record in writer_records:
        write_queue = writer_record["queue"]
        while not write_queue.empty():
            path, payload = write_queue.get_nowait()
            writes.append({"path": path, "payload": payload})
    return writes


def _request(**overrides):
    values = {
        "run_id": "run-1",
        "cycle": 2,
        "rdzv_endpoint": "127.0.0.1:29500",
        "local_node": "node-a",
        "store_timeout_s": 0.1,
        "job_id": "job-1",
        "dmesg_path": None,
        "result_path": None,
    }
    values.update(overrides)
    return FactAgentRequest(**values)


def test_cycle_payload_uses_session_context_and_derives_dmesg_artifact_path(tmp_path):
    cycle_start_time = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    service = FactAgent(
        fact_url="http://fact.example/latest",
        run_id="run-1",
        rdzv_endpoint="store-host:29500",
        store_timeout_s=12.0,
        local_node="node-a",
        is_store_host=True,
        job_id="job-1",
        ranks_per_node=4,
        health_log_prefix=str(tmp_path / "job_health.log"),
        dmesg_artifact_enabled=True,
        result_artifact_enabled=True,
        grpc_server_address="store-host:50051",
        grpc_node_id="node-a_123",
    )

    request = service._request_from_payload(
        {
            "event": "cycle_failed",
            "run_id": "payload-run-should-not-win",
            "cycle": 3,
            "cycle_start_time": cycle_start_time.isoformat(),
            "dmesg_path": str(tmp_path / "payload_dmesg.log"),
            "expected_nodes": ["node-a", "node-b"],
            "result_path": str(tmp_path / "payload_result.log"),
        }
    )

    assert request.run_id == "run-1"
    assert request.rdzv_endpoint == "store-host:29500"
    assert request.store_timeout_s == 12.0
    assert request.local_node == "node-a"
    assert request.is_store_host is True
    assert request.job_id == "job-1"
    assert request.ranks_per_node == 4
    assert request.cycle_start_time == cycle_start_time
    assert request.grpc_server_address == "store-host:50051"
    assert request.grpc_node_id == "node-a_123"
    assert request.expected_nodes == ("node-a", "node-b")
    assert request.dmesg_path == str(tmp_path / "job_health_dmesg_cycle3.log")
    assert request.result_path == str(tmp_path / "job_health_fact_cycle3.log")


def test_cycle_payload_does_not_derive_artifact_paths_without_grpc(tmp_path):
    service = FactAgent(
        fact_url="http://fact.example/latest",
        run_id="run-1",
        rdzv_endpoint="store-host:29500",
        local_node="node-a",
        is_store_host=True,
        health_log_prefix=str(tmp_path / "job_health.log"),
        dmesg_artifact_enabled=True,
        result_artifact_enabled=True,
    )

    request = service._request_from_payload(
        {
            "event": "cycle_failed",
            "cycle": 3,
            "expected_nodes": ["node-a", "node-b"],
        }
    )

    assert request.dmesg_path is None
    assert request.result_path is None


def test_leaf_cycle_payload_derives_shared_result_artifact_path(tmp_path):
    service = FactAgent(
        fact_url="http://fact.example/latest",
        run_id="run-1",
        rdzv_endpoint="store-host:29500",
        local_node="node-b",
        is_store_host=False,
        health_log_prefix=str(tmp_path / "job_health.log"),
        result_artifact_enabled=True,
        grpc_server_address="store-host:50051",
        grpc_node_id="node-b_123",
    )

    request = service._request_from_payload(
        {
            "event": "cycle_failed",
            "cycle": 3,
            "expected_nodes": [],
        }
    )

    assert request.result_path == str(tmp_path / "job_health_fact_cycle3.log")


def test_result_artifact_is_not_derived_without_grpc(tmp_path):
    service = FactAgent(
        fact_url="http://fact.example/latest",
        run_id="run-1",
        rdzv_endpoint="store-host:29500",
        local_node="node-b",
        is_store_host=False,
        health_log_prefix=str(tmp_path / "job_health.log"),
        result_artifact_enabled=True,
    )

    request = service._request_from_payload({"event": "cycle_failed", "cycle": 3})

    assert request.result_path is None


def test_leaf_submission_uses_minimal_store_completion():
    store = FakeStore()
    fact = FakeFactClient()
    keys = FactAgentKeys("run-1", 2)
    store.set(keys.attributor_id, b"att-1")

    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
    )

    service.process_cycle_failed(_request())

    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert sorted(store.data) == [keys.attributor_id, keys.done_count]
    assert fact.submitted[0]["attributor_id"] == "att-1"


def test_leaf_submission_appends_observation_jsonl(tmp_path):
    store = FakeStore()
    fact = FakeFactClient()
    keys = FactAgentKeys("run-1", 2)
    result_path = tmp_path / "fact_result.jsonl"
    writer_records, grpc_writer_factory = _recording_grpc_writer_factory()
    store.set(keys.attributor_id, b"att-1")

    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        grpc_writer_factory=grpc_writer_factory,
    )

    service.process_cycle_failed(
        _request(
            local_node="node-b",
            result_path=str(result_path),
            grpc_server_address="log-host:50051",
            grpc_node_id="node-b_123",
        )
    )

    writes = _drain_grpc_writes(writer_records)
    assert [write["path"] for write in writes] == [str(result_path)]
    records = [json.loads(write["payload"]) for write in writes]
    assert records == [
        {
            "record_type": "fact_observation",
            "run_id": "run-1",
            "cycle": 2,
            "job_id": "job-1",
            "node": "node-b",
            "source": "dmesg",
            "status": "submitted",
            "attributor_id": "att-1",
            "observation_id": "obs-node-b",
            "lines_collected": 1,
            "bytes_collected": len("node-b: [1.0] NVRM: Xid 95".encode("utf-8")),
            "dmesg_path": "",
            "dmesg_write_error": "",
            "error": "",
        }
    ]
    assert int(store.get(keys.done_count).decode("utf-8")) == 1


def test_empty_dmesg_writes_observation_jsonl_without_dmesg_file(tmp_path):
    store = FakeStore()
    fact = FakeFactClient()
    keys = FactAgentKeys("run-1", 2)
    dmesg_path = tmp_path / "job_health_dmesg_cycle2.log"
    result_path = tmp_path / "fact_result.jsonl"
    writer_records, grpc_writer_factory = _recording_grpc_writer_factory()
    store.set(keys.attributor_id, b"att-1")

    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: "",
        grpc_writer_factory=grpc_writer_factory,
    )

    service.process_cycle_failed(
        _request(
            local_node="node-b",
            dmesg_path=str(dmesg_path),
            result_path=str(result_path),
            grpc_server_address="log-host:50051",
            grpc_node_id="node-b_123",
        )
    )

    writes = _drain_grpc_writes(writer_records)
    records = [json.loads(write["payload"]) for write in writes]
    assert not dmesg_path.exists()
    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert records == [
        {
            "record_type": "fact_observation",
            "run_id": "run-1",
            "cycle": 2,
            "job_id": "job-1",
            "node": "node-b",
            "source": "dmesg",
            "status": "empty",
            "attributor_id": "att-1",
            "observation_id": None,
            "lines_collected": 0,
            "bytes_collected": 0,
            "dmesg_path": "",
            "dmesg_write_error": "",
            "error": "",
        }
    ]


def test_leaf_submission_retries_post_within_deadline(tmp_path):
    store = FakeStore()
    fact = FlakyPostFactClient()
    keys = FactAgentKeys("run-1", 2)
    result_path = tmp_path / "fact_result.jsonl"
    writer_records, grpc_writer_factory = _recording_grpc_writer_factory()
    store.set(keys.attributor_id, b"att-1")

    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=30.0,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        grpc_writer_factory=grpc_writer_factory,
    )

    with patch("nvidia_resiliency_ext.attribution.fact.agent.time.sleep"):
        service.process_cycle_failed(
            _request(
                result_path=str(result_path),
                grpc_server_address="log-host:50051",
                grpc_node_id="node-a_123",
            )
        )

    records = [json.loads(write["payload"]) for write in _drain_grpc_writes(writer_records)]
    assert len(fact.submitted) == 1
    assert records[0]["status"] == "submitted"
    assert records[0]["observation_id"] == "obs-node-a"


def test_leaf_uses_store_wait_for_attributor_id():
    store = WaitStore()
    fact = FakeFactClient()
    keys = FactAgentKeys("run-1", 2)
    store.set(keys.attributor_id, b"att-1")

    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
    )

    service.process_cycle_failed(_request())

    assert store.wait_calls[0][0] == [keys.attributor_id]
    assert fact.submitted[0]["attributor_id"] == "att-1"


def test_leaf_submission_queues_dmesg_evidence_file(tmp_path):
    store = FakeStore()
    fact = FakeFactClient()
    keys = FactAgentKeys("run-1", 2)
    dmesg_path = tmp_path / "job_health_dmesg_cycle2.log"
    writer_records, grpc_writer_factory = _recording_grpc_writer_factory()
    store.set(keys.attributor_id, b"att-1")

    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        grpc_writer_factory=grpc_writer_factory,
    )

    service.process_cycle_failed(
        _request(
            dmesg_path=str(dmesg_path),
            grpc_server_address="log-host:50051",
            grpc_node_id="node-a_123",
        )
    )

    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert sorted(store.data) == [keys.attributor_id, keys.done_count]
    assert not dmesg_path.exists()
    writes = _drain_grpc_writes(writer_records)
    assert [write["path"] for write in writes] == [str(dmesg_path)]
    assert writes[0]["payload"] == "node-a: [1.0] NVRM: Xid 95\n"


def test_store_host_gets_result_without_tcpstore_status_fan_in(tmp_path):
    store = FakeStore()
    fact = FakeFactClient()
    result_path = tmp_path / "fact_result.json"
    cycle_start_time = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    writer_records, grpc_writer_factory = _recording_grpc_writer_factory()
    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=0.01,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        grpc_writer_factory=grpc_writer_factory,
        username="slurm-user",
        cluster="slurm-cluster",
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a", "node-b"),
            ranks_per_node=4,
            cycle_start_time=cycle_start_time,
            result_path=str(result_path),
            grpc_server_address="log-host:50051",
            grpc_node_id="node-a_123",
        )
    )

    keys = FactAgentKeys("run-1", 2)
    assert sorted(store.data) == [keys.attributor_id, keys.done_count]
    assert fact.created[0]["nodes"] == ["node-a", "node-b"]
    assert fact.created[0]["ranks_per_node"] == 4
    assert fact.created[0]["nranks"] == 8
    assert fact.created[0]["start_time"] == cycle_start_time
    assert fact.created[0]["username"] == "slurm-user"
    assert fact.created[0]["cluster"] == "slurm-cluster"
    assert fact.gets[0]["observation_ids"] == []
    records = [json.loads(write["payload"]) for write in _drain_grpc_writes(writer_records)]
    assert [record["record_type"] for record in records] == ["fact_observation", "fact_result"]
    assert records[0]["status"] == "submitted"
    assert records[0]["observation_id"] == "obs-node-a"
    result_payload = records[1]
    assert result_payload["status"] == "complete"
    assert result_payload["fact_attribution_result"] == {
        "attributor_id": "att-1",
        "observation_ids": [],
        "faulty_nodes": ["node-a"],
        "attribution": {"attributions": []},
    }
    assert result_payload["expected_node_count"] == 2
    assert result_payload["completed_node_count"] == 1
    assert "faulty_nodes" not in result_payload
    assert "submission_statuses" not in result_payload


def test_store_host_computes_avoid_nodes_from_history():
    store = FakeStore()
    fact = FakeFactClient()
    cycle_start_time = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    cycle_end_time = cycle_start_time + timedelta(minutes=42)
    history = FakeHistoryClient(
        [
            FactHistoryRecord(
                cluster="slurm-cluster",
                node="node-a",
                episode_id="job-0_1",
                event_time=cycle_start_time - timedelta(hours=1),
            )
        ]
    )
    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=0.01,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        fact_history_client_factory=lambda: history,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        username="slurm-user",
        cluster="slurm-cluster",
        is_store_host=True,
        fact_history_es_url="http://history.example",
        fact_history_es_auth_file="/tmp/token",
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a", "node-b"),
            cycle_start_time=cycle_start_time,
            cycle_end_time=cycle_end_time,
        )
    )

    assert history.queries[0]["cluster"] == "slurm-cluster"
    assert history.queries[0]["nodes"] == ["node-a"]
    assert history.queries[0]["end_time"] == cycle_start_time
    assert service.handle_payload({"event": "get_avoid_nodes", "cycle": 2}) == {
        "cycle_id": "2",
        "status": "ready",
        "avoid_nodes": ["node-a"],
    }


def test_hot_cache_overlays_history_for_back_to_back_cycles():
    store = FakeStore()
    fact = FakeFactClient()
    history = FakeHistoryClient([])
    cycle_start_time = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=0.01,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        fact_history_client_factory=lambda: history,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        cluster="slurm-cluster",
        is_store_host=True,
        fact_history_es_url="http://history.example",
        fact_history_es_auth_file="/tmp/token",
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a",),
            cycle=2,
            cycle_start_time=cycle_start_time,
            cycle_end_time=cycle_start_time + timedelta(minutes=5, seconds=30),
        )
    )
    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a",),
            cycle=3,
            cycle_start_time=cycle_start_time + timedelta(minutes=5),
            cycle_end_time=cycle_start_time + timedelta(minutes=6),
        )
    )

    assert service.handle_payload({"event": "get_avoid_nodes", "cycle": 3})["avoid_nodes"] == [
        "node-a"
    ]


def test_hot_cache_computes_avoid_nodes_without_fact_history():
    store = FakeStore()
    fact = FakeFactClient()
    cycle_start_time = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=0.01,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        cluster="slurm-cluster",
        is_store_host=True,
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a",),
            cycle=2,
            cycle_start_time=cycle_start_time,
            cycle_end_time=cycle_start_time + timedelta(minutes=1),
        )
    )
    assert service.handle_payload({"event": "get_avoid_nodes", "cycle": 2}) == {
        "cycle_id": "2",
        "status": "ready",
        "avoid_nodes": [],
    }

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a",),
            cycle=3,
            cycle_start_time=cycle_start_time + timedelta(minutes=5),
            cycle_end_time=cycle_start_time + timedelta(minutes=6),
        )
    )

    assert service.handle_payload({"event": "get_avoid_nodes", "cycle": 3}) == {
        "cycle_id": "3",
        "status": "ready",
        "avoid_nodes": ["node-a"],
    }


def test_grpc_result_artifact_queues_shared_dmesg_artifact(tmp_path):
    store = FakeStore()
    fact = FakeFactClient()
    dmesg_path = tmp_path / "job_health_dmesg_cycle2.log"
    result_path = tmp_path / "fact_result.json"
    writer_records = []

    def grpc_writer_factory(write_queue, address, node_id, logger):
        writer = FakeGrpcWriter()
        writer_records.append(
            {
                "address": address,
                "node_id": node_id,
                "queue": write_queue,
                "writer": writer,
            }
        )
        return writer

    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=0.01,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        grpc_writer_factory=grpc_writer_factory,
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a", "node-b"),
            ranks_per_node=4,
            dmesg_path=str(dmesg_path),
            result_path=str(result_path),
            grpc_server_address="log-host:50051",
            grpc_node_id="node-a_123",
        )
    )

    keys = FactAgentKeys("run-1", 2)
    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert store.get(keys.attributor_id) == b"att-1"
    assert sorted(store.data) == [keys.attributor_id, keys.done_count]
    assert fact.gets[0]["observation_ids"] == []

    assert len(writer_records) == 1
    assert writer_records[0]["address"] == "log-host:50051"
    assert writer_records[0]["node_id"] == "node-a_123"
    assert writer_records[0]["writer"].started
    writes = []
    while not writer_records[0]["queue"].empty():
        path, payload = writer_records[0]["queue"].get_nowait()
        writes.append({"path": path, "payload": payload})

    assert not dmesg_path.exists()
    assert [write["path"] for write in writes] == [
        str(dmesg_path),
        str(result_path),
        str(result_path),
    ]
    assert writes[0]["payload"] == "node-a: [1.0] NVRM: Xid 95\n"
    observation_payload = json.loads(writes[1]["payload"])
    assert observation_payload["record_type"] == "fact_observation"
    assert observation_payload["status"] == "submitted"
    assert observation_payload["observation_id"] == "obs-node-a"
    result_payload = json.loads(writes[2]["payload"])
    assert result_payload["status"] == "complete"
    assert result_payload["fact_attribution_result"]["faulty_nodes"] == ["node-a"]
    assert "faulty_nodes" not in result_payload
    assert "submission_statuses" not in result_payload


def test_grpc_completion_uses_minimal_store_without_dmesg_artifact():
    store = FakeStore()
    fact = FakeFactClient()
    writer_records = []

    def grpc_writer_factory(write_queue, address, node_id, logger):
        writer = FakeGrpcWriter()
        writer_records.append({"queue": write_queue, "writer": writer})
        return writer

    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=0.01,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        grpc_writer_factory=grpc_writer_factory,
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a", "node-b"),
            grpc_server_address="log-host:50051",
            grpc_node_id="node-a_123",
        )
    )

    keys = FactAgentKeys("run-1", 2)
    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert store.get(keys.attributor_id) == b"att-1"
    assert sorted(store.data) == [keys.attributor_id, keys.done_count]
    assert writer_records == []


def test_scalable_attributor_failure_only_uses_attributor_key():
    store = FakeStore()
    fact = FailingCreateFactClient()
    writer_records = []

    def grpc_writer_factory(write_queue, address, node_id, logger):
        writer = FakeGrpcWriter()
        writer_records.append({"queue": write_queue, "writer": writer})
        return writer

    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
        grpc_writer_factory=grpc_writer_factory,
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a", "node-b"),
            grpc_server_address="log-host:50051",
            grpc_node_id="node-a_123",
        )
    )

    keys = FactAgentKeys("run-1", 2)
    assert (
        store.get(keys.attributor_id).decode("utf-8").startswith("__nvrx_fact_attributor_failed__:")
    )
    assert not store.check([keys.done_count])
    assert sorted(store.data) == [keys.attributor_id]
    assert writer_records == []


def test_store_host_attributor_failure_publishes_leaf_sentinel():
    store = FakeStore()
    fact = FailingCreateFactClient()
    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
    )

    service.process_cycle_failed(_request(is_store_host=True, expected_nodes=("node-a", "node-b")))

    keys = FactAgentKeys("run-1", 2)
    service.process_cycle_failed(_request(local_node="node-b"))

    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert fact.submitted == []


def test_leaf_observation_window_uses_collection_time_before_store_wait():
    keys = FactAgentKeys("run-1", 2)
    store = FakeStore()
    store.set(keys.attributor_id, b"att-1")
    store_wait_started = False
    original_check = store.check
    collection_end_time = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)

    def check(keys_to_check):
        nonlocal store_wait_started
        if keys_to_check == [keys.attributor_id]:
            store_wait_started = True
        return original_check(keys_to_check)

    class GuardedDateTime:
        @classmethod
        def now(cls, tz):
            assert not store_wait_started
            return collection_end_time

    store.check = check
    fact = FakeFactClient()
    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
    )

    with patch("nvidia_resiliency_ext.attribution.fact.agent.datetime", GuardedDateTime):
        service.process_cycle_failed(_request())

    assert fact.submitted[0]["end_time"] == collection_end_time


def test_completion_count_poll_keeps_last_known_count_on_store_exception():
    store = FlakyCountReadStore()
    service = FactAgent(fact_url="http://fact.example/latest", observation_deadline_s=1.0)
    keys = FactAgentKeys("run-1", 2)
    monotonic_values = iter([0.0, 0.0, 0.1, 2.0])

    with (
        patch(
            "nvidia_resiliency_ext.attribution.fact.agent.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ),
        patch("nvidia_resiliency_ext.attribution.fact.agent.time.sleep"),
    ):
        completed_count = service._wait_for_completion_count(
            store,
            keys,
            expected_node_count=5,
        )

    assert completed_count == 3


def test_stop_waits_for_executor_tasks_before_draining_grpc_writers():
    writer_records, grpc_writer_factory = _recording_grpc_writer_factory()
    service = FactAgent(
        fact_url="http://fact.example/latest",
        grpc_writer_factory=grpc_writer_factory,
    )

    service._executor.submit(
        service._enqueue_grpc_artifact,
        "log-host:50051",
        "node-a_123",
        "/tmp/fact_result.jsonl",
        "{}\n",
    )

    service.stop()

    assert len(writer_records) == 1
    assert writer_records[0]["writer"].started
    assert writer_records[0]["writer"].shutdown_called
    assert writer_records[0]["writer"].join_timeout == 5.0


def test_empty_collection_is_distinct_from_missing_node():
    store = FakeStore()
    fact = FakeFactClient()
    keys = FactAgentKeys("run-1", 2)
    store.set(keys.attributor_id, b"att-1")
    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: "plain kernel line",
    )

    service.process_cycle_failed(_request())

    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert fact.submitted[0]["dmesg_text"] == "plain kernel line"


def test_serve_forever_exits_after_stop_without_new_connection():
    socket_path = os.path.join(
        tempfile.gettempdir(),
        f"nvrx-fact-agent-test-{os.getpid()}-{time.monotonic_ns()}.sock",
    )
    service = FactAgent(fact_url="http://fact.example/latest", socket_path=socket_path)
    thread = threading.Thread(target=service.serve_forever, daemon=True)

    try:
        thread.start()
        deadline = time.monotonic() + 5.0
        while not os.path.exists(socket_path) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert os.path.exists(socket_path)

        service.stop()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
    finally:
        service.stop()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(socket_path)
