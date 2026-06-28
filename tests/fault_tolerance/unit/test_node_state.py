# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia_resiliency_ext.fault_tolerance.node_state import (
    DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_NODE_STATE_TIMEOUT_SECONDS,
    NodeStateCycleReporter,
    NodeStateCycleStatus,
    NodeStateServiceClient,
)


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeHttpClient:
    posts = []
    gets = []
    get_response = FakeResponse(200, {})
    post_response = FakeResponse(200, {})
    raise_on_get = None
    get_responses = []

    def __init__(self, base_url, timeout):
        self.base_url = base_url
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, path, json):
        self.posts.append((self.base_url, path, json, self.timeout))
        return self.post_response

    def get(self, path, params=None):
        self.gets.append((self.base_url, path, params, self.timeout))
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if self.get_responses:
            return self.get_responses.pop(0)
        return self.get_response


class FakeNodeStateClient:
    def __init__(self, status):
        self.status = status
        self.starts = []
        self.ends = []
        self.gets = []
        self.decision_timeout = DEFAULT_NODE_STATE_TIMEOUT_SECONDS
        self.request_timeout = DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS
        self.poll_interval = 0.0

    def post_cycle_start(self, job_id, cycle_id, nodes):
        self.starts.append((job_id, cycle_id, list(nodes)))
        return True

    def post_cycle_end(self, job_id, cycle_id):
        self.ends.append((job_id, cycle_id))
        return True

    def get_cycle_node_states(self, job_id, cycle_id, timeout=None):
        self.gets.append((job_id, cycle_id, timeout))
        return self.status


def test_node_state_client_posts_cycle_start_and_end(monkeypatch):
    FakeHttpClient.posts = []
    FakeHttpClient.post_response = FakeResponse(200, {})
    FakeHttpClient.get_responses = []
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.node_state.httpx.Client", FakeHttpClient
    )

    client = NodeStateServiceClient("http://node-state.test/")

    assert client.post_cycle_start("job-123", "3", ["node-a", "node-b"]) is True
    assert client.post_cycle_end("job-123", "3") is True
    assert FakeHttpClient.posts == [
        (
            "http://node-state.test",
            "/v1/cycles/start",
            {"job_id": "job-123", "cycle_id": "3", "nodes": ["node-a", "node-b"]},
            DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS,
        ),
        (
            "http://node-state.test",
            "/v1/cycles/end",
            {"job_id": "job-123", "cycle_id": "3"},
            DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS,
        ),
    ]


def test_node_state_client_accepts_async_cycle_end_post(monkeypatch):
    FakeHttpClient.posts = []
    FakeHttpClient.post_response = FakeResponse(
        202, {"accepted": True, "materializing": True}, "accepted"
    )
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.node_state.httpx.Client", FakeHttpClient
    )

    client = NodeStateServiceClient("http://node-state.test/")

    assert client.post_cycle_end("job-123", "3") is True
    assert FakeHttpClient.posts == [
        (
            "http://node-state.test",
            "/v1/cycles/end",
            {"job_id": "job-123", "cycle_id": "3"},
            DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS,
        )
    ]


def test_node_state_client_get_cycle_node_states(monkeypatch):
    FakeHttpClient.get_response = FakeResponse(
        200,
        {
            "job_id": "job-123",
            "cycle_id": "3",
            "bad_nodes": ["node-b"],
            "unknown_nodes": ["node-c"],
            "nodes": [{"node": "node-b", "bad": True}],
        },
    )
    FakeHttpClient.raise_on_get = None
    FakeHttpClient.gets = []
    FakeHttpClient.get_responses = []
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.node_state.httpx.Client", FakeHttpClient
    )

    client = NodeStateServiceClient("http://node-state.test")
    status = client.get_cycle_node_states("job-123", "3")

    assert status.available is True
    assert status.job_id == "job-123"
    assert status.bad_nodes == ("node-b",)
    assert status.unknown_nodes == ("node-c",)
    assert status.nodes == ({"node": "node-b", "bad": True},)
    assert FakeHttpClient.gets == [
        (
            "http://node-state.test",
            "/v1/cycles/3/node-states",
            {"job_id": "job-123"},
            DEFAULT_NODE_STATE_REQUEST_TIMEOUT_SECONDS,
        )
    ]


def test_node_state_client_get_fail_opens(monkeypatch):
    FakeHttpClient.raise_on_get = RuntimeError("boom")
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.node_state.httpx.Client", FakeHttpClient
    )

    client = NodeStateServiceClient("http://node-state.test")
    status = client.get_cycle_node_states("job-123", "3")

    assert status.available is False
    assert status.bad_nodes == ()
    assert "boom" in status.error
    assert status.retryable is True


def test_node_state_client_polls_until_cycle_status_ready(monkeypatch):
    FakeHttpClient.posts = []
    FakeHttpClient.gets = []
    FakeHttpClient.raise_on_get = None
    FakeHttpClient.post_response = FakeResponse(200, {})
    FakeHttpClient.get_responses = [
        FakeResponse(409, {"error": "cycle_status_not_ready"}, "cycle_status_not_ready"),
        FakeResponse(
            200,
            {
                "job_id": "job-123",
                "cycle_id": "3",
                "bad_nodes": ["node-b"],
                "unknown_nodes": [],
                "nodes": [{"node": "node-b", "bad": True}],
            },
        ),
    ]
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.node_state.httpx.Client", FakeHttpClient
    )
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.node_state.time.sleep", lambda _: None
    )

    client = NodeStateServiceClient(
        "http://node-state.test",
        timeout=DEFAULT_NODE_STATE_TIMEOUT_SECONDS,
        poll_interval=0.0,
    )
    status = client.end_cycle_and_get_node_states("job-123", "3")

    assert status.available is True
    assert status.bad_nodes == ("node-b",)
    assert [call[1] for call in FakeHttpClient.gets] == [
        "/v1/cycles/3/node-states",
        "/v1/cycles/3/node-states",
    ]


def test_node_state_client_fails_open_when_cycle_end_post_fails(monkeypatch):
    FakeHttpClient.posts = []
    FakeHttpClient.gets = []
    FakeHttpClient.raise_on_get = None
    FakeHttpClient.post_response = FakeResponse(503, {"error": "unavailable"}, "unavailable")
    FakeHttpClient.get_responses = [
        FakeResponse(409, {"error": "cycle_status_not_ready"}, "cycle_status_not_ready")
    ]
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.node_state.httpx.Client", FakeHttpClient
    )

    client = NodeStateServiceClient(
        "http://node-state.test",
        timeout=DEFAULT_NODE_STATE_TIMEOUT_SECONDS,
        poll_interval=0.0,
    )
    status = client.end_cycle_and_get_node_states("job-123", "3")

    assert status.available is False
    assert status.retryable is False
    assert status.error == "node-state cycle-end notification failed"
    assert [call[1] for call in FakeHttpClient.posts] == ["/v1/cycles/end"]
    assert FakeHttpClient.gets == []


def test_node_state_cycle_reporter_reports_start_and_end_once():
    status = NodeStateCycleStatus(job_id="job-123", cycle_id="3", available=True)
    fake_client = FakeNodeStateClient(status)
    reporter = NodeStateCycleReporter(None, job_id="job-123", is_enabled=False)
    reporter.client = fake_client

    reporter.report_cycle_start("3", ["node-a", "node-b"])
    request_ok = reporter.request_cycle_end("3")
    duplicate_request_ok = reporter.request_cycle_end("3")
    returned_status = reporter.get_cycle_status("3")
    cached_status = reporter.get_cycle_status("3")

    assert fake_client.starts == [("job-123", "3", ["node-a", "node-b"])]
    assert fake_client.ends == [("job-123", "3")]
    assert request_ok is True
    assert duplicate_request_ok is True
    assert returned_status is status
    assert cached_status is status
    assert len(fake_client.gets) == 1


def test_node_state_cycle_reporter_returns_none_while_status_pending():
    pending_status = NodeStateCycleStatus(
        job_id="job-123",
        cycle_id="3",
        available=False,
        retryable=True,
        error="cycle_status_not_ready",
    )
    fake_client = FakeNodeStateClient(pending_status)
    reporter = NodeStateCycleReporter(None, job_id="job-123", is_enabled=False)
    reporter.client = fake_client

    reporter.report_cycle_start("3", ["node-a"])
    assert reporter.request_cycle_end("3") is True
    assert reporter.get_cycle_status("3") is None

    assert fake_client.ends == [("job-123", "3")]
    assert len(fake_client.gets) == 1


def test_node_state_cycle_reporter_skips_end_when_start_was_not_reported():
    status = NodeStateCycleStatus(job_id="job-123", cycle_id="3", available=True)
    fake_client = FakeNodeStateClient(status)
    reporter = NodeStateCycleReporter(None, job_id="job-123", is_enabled=False)
    reporter.client = fake_client

    assert reporter.request_cycle_end("3") is False
    end_status = reporter.get_cycle_status("3")

    assert fake_client.ends == []
    assert end_status is not None
    assert end_status.available is False
    assert "cycle start was not reported" in end_status.error


def test_node_state_cycle_reporter_skips_empty_start():
    status = NodeStateCycleStatus(job_id="job-123", cycle_id="3", available=True)
    fake_client = FakeNodeStateClient(status)
    reporter = NodeStateCycleReporter(None, job_id="job-123", is_enabled=False)
    reporter.client = fake_client

    reporter.report_cycle_start("3", [])

    assert fake_client.starts == []
