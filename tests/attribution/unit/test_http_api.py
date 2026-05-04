from unittest.mock import MagicMock

from nvidia_resiliency_ext.attribution.orchestration.http_api import (
    get_log_response,
    log_result_params,
    log_submit_payload,
    post_log,
)


def test_log_submit_payload_omits_empty_optional_fields():
    assert log_submit_payload("/tmp/train.log") == {"log_path": "/tmp/train.log"}


def test_log_submit_payload_includes_smon_fields():
    assert log_submit_payload("/tmp/train.log", user="alice", job_id="123") == {
        "log_path": "/tmp/train.log",
        "user": "alice",
        "job_id": "123",
    }


def test_log_result_params_includes_splitlog_fields():
    assert log_result_params("/tmp/train.log", file="cycle-0.log", wl_restart=2) == {
        "log_path": "/tmp/train.log",
        "file": "cycle-0.log",
        "wl_restart": 2,
    }


def test_post_log_uses_shared_logs_route():
    client = MagicMock()

    post_log(client, "/tmp/train.log", user="alice", job_id="123")

    client.post.assert_called_once_with(
        "/logs",
        json={"log_path": "/tmp/train.log", "user": "alice", "job_id": "123"},
        headers={"accept": "application/json"},
    )


def test_get_log_response_uses_shared_logs_route():
    client = MagicMock()

    get_log_response(client, "/tmp/train.log", wl_restart=2)

    client.get.assert_called_once_with(
        "/logs",
        params={"log_path": "/tmp/train.log", "wl_restart": 2},
        headers={"accept": "application/json"},
    )
