# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for wait-for-TCPStore-server logic in c10d_monkey_patch."""

import socket
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch import _wait_for_tcp_store_server


def _monotonic_sequence(*values):
    """Return an iterator that yields the given values, then repeats the last."""
    it = iter(values)
    last = next(it, None)
    if last is None:
        raise ValueError("need at least one value")
    yield last
    for v in it:
        last = v
        yield v
    while True:
        yield last


class TestWaitForTcpStoreServer:
    """Tests for _wait_for_tcp_store_server with mocked socket and time."""

    def test_success_immediately(self):
        """Server is available on first probe; no retries."""
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                return_value=MagicMock(),
            ) as create_conn,
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0,
            ),
        ):
            _wait_for_tcp_store_server(
                "host",
                29400,
                max_wait_seconds=5,
                probe_timeout_seconds=1,
                probe_interval_seconds=2,
            )
        create_conn.assert_called_once()
        call = create_conn.call_args
        assert call.args == (("host", 29400),)
        assert call.kwargs["timeout"] == 1

    def test_success_after_retries(self):
        """Server comes up after two failed probes; third succeeds."""
        fail = socket.timeout("connect timed out")
        succeed = MagicMock()
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                side_effect=[fail, fail, succeed],
            ) as create_conn,
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0, 0.5, 0.5, 1.0, 1.0),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0,
            ),
        ):
            _wait_for_tcp_store_server(
                "node1",
                29400,
                max_wait_seconds=2,
                probe_timeout_seconds=1,
                probe_interval_seconds=0.5,
            )
        assert create_conn.call_count == 3

    def test_timeout_server_never_up(self):
        """Server never comes up; TimeoutError with message including host/port and last error."""
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                side_effect=socket.timeout("timed out"),
            ),
            # deadline=0.2: attempt 1 at t=0 fails, sleep; attempt 2 at t=0.2 fails, remaining=0 -> raise
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0, 0.2, 0.2),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0,
            ),
        ):
            with pytest.raises(TimeoutError) as exc_info:
                _wait_for_tcp_store_server(
                    "nvl72112-T01",
                    29400,
                    max_wait_seconds=0.2,
                    probe_timeout_seconds=0.5,
                    probe_interval_seconds=0.1,
                )
        msg = str(exc_info.value)
        assert "nvl72112-T01" in msg
        assert "29400" in msg
        assert "0.2" in msg
        assert "timed out" in msg or "last error" in msg

    def test_timeout_sets_cause(self):
        """TimeoutError chains the last socket error as __cause__."""
        orig = socket.timeout("connect failed")
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                side_effect=orig,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0, 0.1, 0.1),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0,
            ),
        ):
            with pytest.raises(TimeoutError) as exc_info:
                _wait_for_tcp_store_server(
                    "h",
                    1,
                    max_wait_seconds=0.1,
                    probe_timeout_seconds=0.05,
                    probe_interval_seconds=0.05,
                )
        assert exc_info.value.__cause__ is orig

    def test_jitter_sleep_duration_in_bounds(self):
        """On failure, sleep is between probe_interval and probe_interval + jitter."""
        sleep_calls = []
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                side_effect=[socket.timeout(""), MagicMock()],
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0, 2.0, 2.0, 4.0, 4.0),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
                side_effect=lambda x: sleep_calls.append(x),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0.5,
            ),
        ):
            _wait_for_tcp_store_server(
                "h",
                1,
                max_wait_seconds=10,
                probe_timeout_seconds=1,
                probe_interval_seconds=2,
                probe_interval_jitter_fraction=0.5,
            )
        # No initial stagger (uniform=0). One failure -> one retry sleep. jitter = 2*0.5*0.5 = 0.5, so sleep = 2.5
        assert len(sleep_calls) >= 1
        retry_sleep = sleep_calls[-1]
        assert 2.0 <= retry_sleep <= 3.0

    def test_edge_very_small_max_wait_stagger_capped(self):
        """With very small max_wait_seconds, initial stagger is capped so at least one probe runs."""
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                return_value=MagicMock(),
            ) as create_conn,
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0.1,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0,
            ),
        ):
            _wait_for_tcp_store_server(
                "h",
                1,
                max_wait_seconds=2,
                probe_timeout_seconds=1,
                probe_interval_seconds=2,
            )
        # max_stagger = min(1, 0.4) = 0.4; we pass with one probe
        create_conn.assert_called_once()

    def test_edge_large_max_wait_times_out_when_server_never_up(self):
        """With large max_wait_seconds, we still raise TimeoutError when server never comes up."""
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                side_effect=socket.timeout(""),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0, 60, 60, 120, 120, 181),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0,
            ),
        ):
            with pytest.raises(TimeoutError, match="did not become reachable within 120"):
                _wait_for_tcp_store_server(
                    "big",
                    29400,
                    max_wait_seconds=120,
                    probe_timeout_seconds=5,
                    probe_interval_seconds=2,
                )

    def test_oserror_also_retried(self):
        """OSError (e.g. ConnectionRefused) is caught and retried like socket.timeout."""
        with (
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.socket.create_connection",
                side_effect=[ConnectionRefusedError("refused"), MagicMock()],
            ) as create_conn,
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.monotonic",
                side_effect=_monotonic_sequence(0, 0, 0, 1.0, 1.0),
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.time.sleep",
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.uniform",
                return_value=0,
            ),
            patch(
                "nvidia_resiliency_ext.fault_tolerance.c10d_monkey_patch.random.random",
                return_value=0,
            ),
        ):
            _wait_for_tcp_store_server(
                "h",
                1,
                max_wait_seconds=5,
                probe_timeout_seconds=1,
                probe_interval_seconds=0.5,
            )
        assert create_conn.call_count == 2
