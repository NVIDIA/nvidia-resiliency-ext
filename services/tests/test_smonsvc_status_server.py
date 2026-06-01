# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from nvidia_resiliency_ext.services.smonsvc.status_server import DEFAULT_STATUS_HOST, StatusServer


@patch("nvidia_resiliency_ext.services.smonsvc.status_server.threading.Thread")
@patch("nvidia_resiliency_ext.services.smonsvc.status_server.HTTPServer")
def test_status_server_binds_loopback_by_default(mock_http_server, mock_thread):
    server = StatusServer(
        port=8100,
        get_stats=lambda: {},
        get_jobs=lambda: [],
        get_health=lambda: (True, {}),
    )

    server.start()

    mock_http_server.assert_called_once()
    assert mock_http_server.call_args.args[0] == (DEFAULT_STATUS_HOST, 8100)
    mock_thread.return_value.start.assert_called_once()


@patch("nvidia_resiliency_ext.services.smonsvc.status_server.threading.Thread")
@patch("nvidia_resiliency_ext.services.smonsvc.status_server.HTTPServer")
def test_status_server_accepts_explicit_bind_host(mock_http_server, mock_thread):
    server = StatusServer(
        host="0.0.0.0",  # nosec B104
        port=8100,
        get_stats=lambda: {},
        get_jobs=lambda: [],
        get_health=lambda: (True, {}),
    )
    mock_http_server.return_value = MagicMock()

    server.start()

    mock_http_server.assert_called_once()
    assert mock_http_server.call_args.args[0] == ("0.0.0.0", 8100)  # nosec B104
    mock_thread.return_value.start.assert_called_once()
