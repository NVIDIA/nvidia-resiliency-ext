# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch

from nvidia_resiliency_ext.attribution.fact.agent import FactAgent, FactAgentKeys, FactAgentRequest
from nvidia_resiliency_ext.attribution.fact.client import FactAttributionResult


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


class TransientStatusReadStore(FakeStore):
    def __init__(self, flaky_key):
        super().__init__()
        self.flaky_key = flaky_key
        self.failed_once = False

    def check(self, keys):
        if keys == [self.flaky_key] and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("transient TCPStore read error")
        return super().check(keys)


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


class FailingCreateFactClient(FakeFactClient):
    def create_failure_attributor(self, **kwargs):
        self.created.append(kwargs)
        raise RuntimeError("FACT unavailable")


def _request(**overrides):
    values = {
        "run_id": "run-1",
        "cycle": 2,
        "rdzv_endpoint": "127.0.0.1:29500",
        "local_node": "node-a",
        "store_timeout_s": 0.1,
        "role": "trainer",
        "job_id": "job-1",
        "dmesg_path": None,
    }
    values.update(overrides)
    return FactAgentRequest(**values)


def _load_status(store, keys, node):
    return json.loads(store.get(keys.status(node)).decode("utf-8"))


def test_leaf_submission_writes_status_observation_and_completion():
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

    status = _load_status(store, keys, "node-a")
    assert status["status"] == "submitted"
    assert status["observation_id"] == "obs-node-a"
    assert int(store.get(keys.done_count).decode("utf-8")) == 1
    assert store.get(keys.done(1)) == b"node-a"
    assert store.get(keys.observation("node-a")) == b"obs-node-a"
    assert fact.submitted[0]["attributor_id"] == "att-1"


def test_leaf_submission_can_write_dmesg_evidence_file(tmp_path):
    store = FakeStore()
    fact = FakeFactClient()
    keys = FactAgentKeys("run-1", 2)
    dmesg_path = tmp_path / "job_health_dmesg_node-a_cycle2.log"
    store.set(keys.attributor_id, b"att-1")

    service = FactAgent(
        fact_url="http://fact.example/latest",
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
    )

    service.process_cycle_failed(_request(dmesg_path=str(dmesg_path)))

    status = _load_status(store, keys, "node-a")
    assert status["status"] == "submitted"
    assert status["dmesg_path"] == str(dmesg_path)
    assert status["dmesg_write_error"] == ""
    assert dmesg_path.read_text(encoding="utf-8") == "node-a: [1.0] NVRM: Xid 95\n"


def test_store_host_gets_result_and_records_missing_nodes(tmp_path):
    store = FakeStore()
    fact = FakeFactClient()
    result_path = tmp_path / "fact_result.json"
    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=0.01,
        store_factory=lambda request: store,
        fact_client_factory=lambda: fact,
        dmesg_collector=lambda window_s, node: f"{node}: [1.0] NVRM: Xid 95",
    )

    service.process_cycle_failed(
        _request(
            is_store_host=True,
            expected_nodes=("node-a", "node-b"),
            ranks_per_node=4,
            nranks=8,
            result_path=str(result_path),
        )
    )

    keys = FactAgentKeys("run-1", 2)
    result_status = json.loads(store.get(keys.result_status).decode("utf-8"))
    assert result_status["status"] == "complete"
    assert result_status["faulty_nodes"] == ["node-a"]
    assert store.get(keys.faulty_nodes) == b'["node-a"]'
    assert store.get(keys.result_path).decode("utf-8") == str(result_path)
    assert fact.created[0]["nodes"] == ["node-a", "node-b"]
    assert fact.gets[0]["observation_ids"] == ["obs-node-a"]

    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_payload["faulty_nodes"] == ["node-a"]
    assert result_payload["submission_statuses"]["node-a"]["status"] == "submitted"
    assert result_payload["submission_statuses"]["node-b"]["status"] == "timeout"


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
    result_status = json.loads(store.get(keys.result_status).decode("utf-8"))
    assert result_status["status"] == "failed"

    service.process_cycle_failed(_request(local_node="node-b"))

    status = _load_status(store, keys, "node-b")
    assert status["status"] == "post_failed"
    assert "FACT attributor creation failed on store host" in status["error"]
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

    status = _load_status(store, keys, "node-a")
    assert status["status"] == "empty"
    assert status["observation_id"] is None
    assert not store.check([keys.observation("node-a")])


def test_collect_statuses_retries_pending_status_after_transient_read():
    keys = FactAgentKeys("run-1", 2)
    store = TransientStatusReadStore(keys.status("node-a"))
    store.set(keys.done_count, b"2")
    store.set(keys.done(1), b"node-a")
    store.set(keys.done(2), b"node-b")
    store.set(
        keys.status("node-a"),
        json.dumps({"node": "node-a", "status": "submitted", "observation_id": "obs-a"}),
    )
    store.set(
        keys.status("node-b"),
        json.dumps({"node": "node-b", "status": "submitted", "observation_id": "obs-b"}),
    )
    service = FactAgent(
        fact_url="http://fact.example/latest",
        observation_deadline_s=1.0,
        store_factory=lambda request: store,
        fact_client_factory=FakeFactClient,
        dmesg_collector=lambda window_s, node: "",
    )

    statuses = service._collect_statuses(store, keys, ["node-a", "node-b"])

    assert statuses["node-a"]["observation_id"] == "obs-a"
    assert statuses["node-b"]["observation_id"] == "obs-b"
    assert store.failed_once is True


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
