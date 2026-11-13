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


def test_log_path_counters_stay_cumulative_across_cleanup():
    """The stats contract calls this section cumulative, but two of its values
    were set lengths that cleanup prunes.

    A dashboard plotting them would read a routine cleanup as a process
    restart. The live sizes are still reported, under names that say so.
    """
    import threading

    from nvidia_resiliency_ext.services.smonsvc.models import MonitorState
    from nvidia_resiliency_ext.services.smonsvc.stats import get_stats_dict

    state, lock = MonitorState(), threading.Lock()
    for index in range(5):
        path = f"/logs/c{index}.log"
        state.submitted_log_paths.add(path)
        state.log_paths_claimed_total += 1
        state.analyzed_log_paths.add(path)
        state.log_paths_analyzed_total += 1

    before = get_stats_dict(state, lock)["log_paths"]
    assert (before["claimed"], before["analyzed"]) == (5, 5)
    assert before["currently_tracked_claims"] == 5

    # What _cleanup_old_jobs does once no retained job resolves to these.
    state.submitted_log_paths.clear()
    state.analyzed_log_paths.clear()

    after = get_stats_dict(state, lock)["log_paths"]
    assert (after["claimed"], after["analyzed"]) == (5, 5), "cumulative must not shrink"
    assert after["currently_tracked_claims"] == 0, "the live size should drop"
