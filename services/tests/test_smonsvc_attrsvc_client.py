# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.services.attrsvc.app import _prepare_uds_path
from nvidia_resiliency_ext.services.attrsvc.config import parse_service_endpoint
from nvidia_resiliency_ext.services.smonsvc.attrsvc_client import (
    AttrsvcClient,
    _parse_attrsvc_endpoint,
)


def test_parse_service_endpoint_unix():
    endpoint = parse_service_endpoint("unix:///tmp/nvrx-attrsvc.sock")

    assert endpoint.uds_path == "/tmp/nvrx-attrsvc.sock"
    assert endpoint.display_url == "unix:///tmp/nvrx-attrsvc.sock"


def test_parse_service_endpoint_http():
    endpoint = parse_service_endpoint("http://127.0.0.1:8123")

    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 8123
    assert endpoint.uds_path == ""
    assert endpoint.display_url == "http://127.0.0.1:8123"


@patch("nvidia_resiliency_ext.services.attrsvc.app.os.unlink")
@patch("nvidia_resiliency_ext.services.attrsvc.app.socket.socket")
@patch("nvidia_resiliency_ext.services.attrsvc.app.os.stat")
@patch("nvidia_resiliency_ext.services.attrsvc.app.os.path.exists", return_value=True)
def test_prepare_uds_path_unlinks_only_connection_refused(
    mock_exists, mock_stat, mock_socket, mock_unlink
):
    del mock_exists
    mock_stat.return_value = SimpleNamespace(st_mode=stat.S_IFSOCK)
    sock = MagicMock()
    sock.connect.side_effect = ConnectionRefusedError("stale")
    mock_socket.return_value.__enter__.return_value = sock

    _prepare_uds_path("/tmp/nvrx-attrsvc.sock")

    mock_unlink.assert_called_once_with("/tmp/nvrx-attrsvc.sock")


@patch("nvidia_resiliency_ext.services.attrsvc.app.os.unlink")
@patch("nvidia_resiliency_ext.services.attrsvc.app.socket.socket")
@patch("nvidia_resiliency_ext.services.attrsvc.app.os.stat")
@patch("nvidia_resiliency_ext.services.attrsvc.app.os.path.exists", return_value=True)
def test_prepare_uds_path_permission_error_does_not_unlink(
    mock_exists, mock_stat, mock_socket, mock_unlink
):
    del mock_exists
    mock_stat.return_value = SimpleNamespace(st_mode=stat.S_IFSOCK)
    sock = MagicMock()
    sock.connect.side_effect = PermissionError("denied")
    mock_socket.return_value.__enter__.return_value = sock

    with pytest.raises(SystemExit, match="could not be probed safely"):
        _prepare_uds_path("/tmp/nvrx-attrsvc.sock")

    mock_unlink.assert_not_called()


def test_parse_unix_attrsvc_endpoint():
    endpoint = _parse_attrsvc_endpoint("unix:///tmp/nvrx-attrsvc.sock")

    assert endpoint.base_url == "http://nvrx-attrsvc"
    assert endpoint.display_url == "unix:///tmp/nvrx-attrsvc.sock"
    assert endpoint.uds_path == "/tmp/nvrx-attrsvc.sock"


def test_parse_host_port_attrsvc_endpoint():
    endpoint = _parse_attrsvc_endpoint("localhost:8000")

    assert endpoint.base_url == "http://localhost:8000"
    assert endpoint.display_url == "http://localhost:8000"
    assert endpoint.uds_path is None


def test_parse_https_attrsvc_endpoint_is_rejected():
    with pytest.raises(ValueError, match="must use http://"):
        _parse_attrsvc_endpoint("https://localhost:8000")


@patch("nvidia_resiliency_ext.services.smonsvc.attrsvc_client.httpx")
def test_unix_attrsvc_client_uses_uds_transport(mock_httpx):
    transport = MagicMock()
    client = MagicMock()
    mock_httpx.HTTPTransport.return_value = transport
    mock_httpx.Client.return_value = client

    AttrsvcClient("unix:///tmp/nvrx-attrsvc.sock")

    mock_httpx.HTTPTransport.assert_called_once_with(uds="/tmp/nvrx-attrsvc.sock")
    mock_httpx.Client.assert_called_once_with(
        transport=transport,
        base_url="http://nvrx-attrsvc",
        timeout=AttrsvcClient.DEFAULT_TIMEOUT,
    )


@patch("nvidia_resiliency_ext.services.smonsvc.attrsvc_client.httpx")
def test_health_probe_uses_healthz_over_configured_client(mock_httpx):
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.get.return_value = response
    mock_httpx.Client.return_value = client

    attrsvc = AttrsvcClient("unix:///tmp/nvrx-attrsvc.sock")

    assert attrsvc.check_health_cached() is True
    client.get.assert_called_once_with("/healthz", timeout=5.0)


@patch("nvidia_resiliency_ext.services.smonsvc.attrsvc_client.httpx")
def test_request_with_retry_uses_relative_logs_route(mock_httpx):
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.get.return_value = response
    mock_httpx.Client.return_value = client
    on_success = MagicMock()

    attrsvc = AttrsvcClient("unix:///tmp/nvrx-attrsvc.sock", request_throttle=0)
    attrsvc.request_with_retry(
        "GET",
        job_id="123",
        log_path="/tmp/job.log",
        on_success=on_success,
        on_client_error=MagicMock(),
    )

    client.get.assert_called_once_with(
        "/logs",
        params={"log_path": "/tmp/job.log"},
        headers={"accept": "application/json"},
    )
    on_success.assert_called_once_with(response)


@patch("nvidia_resiliency_ext.services.smonsvc.attrsvc_client.httpx")
def test_not_readable_403_warns_by_default(mock_httpx, caplog):
    response = MagicMock()
    response.status_code = 403
    response.json.return_value = {
        "error_code": "errorcode.not_readable",
        "message": "permission denied",
        "details": None,
    }
    client = MagicMock()
    client.post.return_value = response
    mock_httpx.Client.return_value = client
    on_client_error = MagicMock()

    attrsvc = AttrsvcClient("unix:///tmp/nvrx-attrsvc.sock", request_throttle=0)
    with caplog.at_level(
        logging.WARNING,
        logger="nvidia_resiliency_ext.services.smonsvc.attrsvc_client",
    ):
        attrsvc.request_with_retry(
            "POST",
            job_id="123",
            log_path="/tmp/job.log",
            on_success=MagicMock(),
            on_client_error=on_client_error,
        )

    assert "[123] POST failed (403)" in caplog.text
    on_client_error.assert_called_once()


@patch("nvidia_resiliency_ext.services.smonsvc.attrsvc_client.httpx")
def test_not_readable_403_logs_at_debug_when_expected(mock_httpx, caplog):
    response = MagicMock()
    response.status_code = 403
    response.json.return_value = {
        "error_code": "errorcode.not_readable",
        "message": "permission denied",
        "details": None,
    }
    client = MagicMock()
    client.post.return_value = response
    mock_httpx.Client.return_value = client
    on_client_error = MagicMock()

    attrsvc = AttrsvcClient(
        "unix:///tmp/nvrx-attrsvc.sock",
        request_throttle=0,
        permission_denied_is_expected=True,
    )
    with caplog.at_level(
        logging.DEBUG,
        logger="nvidia_resiliency_ext.services.smonsvc.attrsvc_client",
    ):
        attrsvc.request_with_retry(
            "POST",
            job_id="123",
            log_path="/tmp/job.log",
            on_success=MagicMock(),
            on_client_error=on_client_error,
        )

    matching_records = [
        record for record in caplog.records if "[123] POST failed (403)" in record.message
    ]
    assert matching_records
    assert {record.levelno for record in matching_records} == {logging.DEBUG}
    on_client_error.assert_called_once()


@patch("nvidia_resiliency_ext.services.smonsvc.attrsvc_client.httpx")
def test_non_json_client_error_still_calls_error_callback(mock_httpx, caplog):
    response = MagicMock()
    response.status_code = 403
    response.json.side_effect = ValueError("not json")
    response.text = "<html>forbidden</html>"
    client = MagicMock()
    client.post.return_value = response
    mock_httpx.Client.return_value = client
    on_client_error = MagicMock()

    attrsvc = AttrsvcClient(
        "unix:///tmp/nvrx-attrsvc.sock",
        request_throttle=0,
        permission_denied_is_expected=True,
    )
    with caplog.at_level(
        logging.WARNING,
        logger="nvidia_resiliency_ext.services.smonsvc.attrsvc_client",
    ):
        attrsvc.request_with_retry(
            "POST",
            job_id="123",
            log_path="/tmp/job.log",
            on_success=MagicMock(),
            on_client_error=on_client_error,
        )

    assert "[123] POST failed (403): <html>forbidden</html>" in caplog.text
    on_client_error.assert_called_once_with("<html>forbidden</html>")


@patch("nvidia_resiliency_ext.services.smonsvc.attrsvc_client.httpx")
def test_expected_client_errors_are_sampled(mock_httpx, caplog):
    response = MagicMock()
    response.status_code = 403
    response.json.return_value = {
        "error_code": "errorcode.not_readable",
        "message": "permission denied",
        "details": None,
    }
    client = MagicMock()
    client.post.return_value = response
    mock_httpx.Client.return_value = client
    on_client_error = MagicMock()

    attrsvc = AttrsvcClient(
        "unix:///tmp/nvrx-attrsvc.sock",
        request_throttle=0,
        permission_denied_is_expected=True,
    )
    with caplog.at_level(
        logging.DEBUG,
        logger="nvidia_resiliency_ext.services.smonsvc.attrsvc_client",
    ):
        for idx in range(6):
            attrsvc.request_with_retry(
                "POST",
                job_id=str(123 + idx),
                log_path="/tmp/job.log",
                on_success=MagicMock(),
                on_client_error=on_client_error,
            )

    matching_records = [
        record
        for record in caplog.records
        if "POST failed (403)" in record.message
        and "expected client error count=" in record.message
    ]
    assert len(matching_records) == 5
    assert on_client_error.call_count == 6
