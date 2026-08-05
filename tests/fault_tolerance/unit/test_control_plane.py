# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance import control_plane
from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import CycleInfoRoundSnapshot


@pytest.fixture(autouse=True)
def _no_real_sleep():
    """Keep the store-read grace period from actually sleeping in unit tests.

    run() waits out the remainder of the grace window on every exit path, so without
    this each test that calls run() would burn the full _STORE_READ_GRACE_SECONDS.
    Tests that assert on the wait patch time.sleep themselves and shadow this.
    """
    with patch.object(control_plane.time, "sleep"):
        yield


def test_nvrx_control_starts_owned_services_without_worker_agent(tmp_path):
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--rdzv-id",
            "job-a",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-enable-log-server",
            "true",
            "--ft-attribution-endpoint",
            "localhost",
            "--ft-cycle-info-dir",
            str(tmp_path / "nvrx"),
            "--ft-cycle-info-job-id",
            "job-a",
        ]
    )
    manager = MagicMock()
    # Mirror the real AttributionEndpoint dataclass, including the enforcement flag.
    manager.start_if_needed.return_value = SimpleNamespace(
        endpoint="http://localhost:50050", enforce_stop=False
    )
    grpc_proc = MagicMock()
    cycle_reporter = MagicMock()

    with (
        patch.object(control_plane, "_create_tcp_store", return_value=object()) as create_store,
        patch.object(control_plane, "AttributionManager", return_value=manager) as manager_cls,
        patch.object(
            control_plane, "CycleInfoReporter", return_value=cycle_reporter
        ) as reporter_cls,
        patch.object(
            control_plane, "_start_grpc_log_servers", return_value=[grpc_proc]
        ) as start_grpc,
        patch.object(control_plane, "stop_grpc_log_servers") as stop_grpc,
        patch.object(control_plane, "_run_control_rendezvous_loop") as run_loop,
    ):
        control_plane.run(args)

    create_store.assert_called_once()
    manager_cls.assert_called_once()
    manager.start_if_needed.assert_called_once()
    start_grpc.assert_called_once()
    assert start_grpc.call_args.args[1] == str(tmp_path / "train_grpc")
    reporter_cls.assert_called_once_with(
        str(tmp_path / "nvrx"),
        cycle_log_prefix=str(tmp_path / "train.log"),
        cycle_info_job_id="job-a",
        attempt_index=0,
    )
    run_loop.assert_called_once()
    stop_grpc.assert_called_once_with([grpc_proc], 60.0)
    manager.stop.assert_called_once()
    cycle_reporter.shutdown.assert_called_once()
    assert not hasattr(control_plane, "LocalElasticAgent")


def test_nvrx_control_stops_the_attribution_poller_before_the_service(tmp_path):
    """nvrx-control owns the client, so nothing else will stop its poller thread.

    The embedded launcher does this from FtRendezvousBarrierHandler._close(), which has
    no equivalent here. Without it the daemon keeps polling attrsvc through its shutdown.
    """
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--rdzv-id",
            "job-a",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-attribution-endpoint",
            "localhost",
        ]
    )
    manager = MagicMock()
    manager.start_if_needed.return_value = SimpleNamespace(
        endpoint="http://localhost:50050", enforce_stop=False
    )
    service = MagicMock()
    order = []
    service.stop_poller.side_effect = lambda *a, **k: order.append("stop_poller")
    manager.stop.side_effect = lambda *a, **k: order.append("manager_stop")

    with (
        patch.object(control_plane, "_create_tcp_store", return_value=object()),
        patch.object(control_plane, "AttributionManager", return_value=manager),
        patch.object(control_plane, "AttributionService", return_value=service),
        patch.object(control_plane, "_run_control_rendezvous_loop"),
    ):
        control_plane.run(args)

    service.start_poller.assert_called_once()
    service.stop_poller.assert_called_once()
    assert order == ["stop_poller", "manager_stop"]


def test_nvrx_control_stops_the_poller_when_startup_fails(tmp_path):
    """A later startup failure must not abandon an already-running poller thread."""
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--rdzv-id",
            "job-a",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-attribution-endpoint",
            "localhost",
            "--ft-enable-log-server",
            "true",
        ]
    )
    manager = MagicMock()
    manager.start_if_needed.return_value = SimpleNamespace(
        endpoint="http://localhost:50050", enforce_stop=False
    )
    service = MagicMock()

    with (
        patch.object(control_plane, "_create_tcp_store", return_value=object()),
        patch.object(control_plane, "AttributionManager", return_value=manager),
        patch.object(control_plane, "AttributionService", return_value=service),
        patch.object(control_plane, "_start_grpc_log_servers", return_value=[]),
        patch.object(control_plane, "stop_grpc_log_servers"),
        patch.object(control_plane, "_run_control_rendezvous_loop"),
        pytest.raises(RuntimeError, match="log funnel"),
    ):
        control_plane.run(args)

    service.stop_poller.assert_called_once()


def test_store_read_grace_waits_out_the_remainder():
    """Peers map the stored shutdown reason to their exit code, so the store must outlive them."""
    with (
        patch.object(control_plane.time, "monotonic", return_value=100.0),
        patch.object(control_plane.time, "sleep") as sleep,
    ):
        control_plane._wait_for_store_read_grace(since=99.0)

    sleep.assert_called_once()
    assert sleep.call_args.args[0] == pytest.approx(2.0)


def test_store_read_grace_is_skipped_when_teardown_was_already_slow():
    """A slow gRPC or attrsvc shutdown already held the window open; do not add to it."""
    with (
        patch.object(control_plane.time, "monotonic", return_value=100.0),
        patch.object(control_plane.time, "sleep") as sleep,
    ):
        control_plane._wait_for_store_read_grace(since=90.0)

    sleep.assert_not_called()


def test_nvrx_control_holds_the_store_open_before_exit(tmp_path):
    """The grace wait must run after service teardown but before run() drops the store."""
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--rdzv-id",
            "job-a",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
        ]
    )

    with (
        patch.object(control_plane, "_create_tcp_store", return_value=object()),
        patch.object(control_plane, "_run_control_rendezvous_loop"),
        patch.object(control_plane, "_wait_for_store_read_grace") as grace,
    ):
        control_plane.run(args)

    grace.assert_called_once()


def test_nvrx_control_holds_the_store_open_even_when_the_loop_raises(tmp_path):
    """An aborted control loop is exactly when peers still need to read the reason."""
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--rdzv-id",
            "job-a",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
        ]
    )

    with (
        patch.object(control_plane, "_create_tcp_store", return_value=object()),
        patch.object(
            control_plane, "_run_control_rendezvous_loop", side_effect=RuntimeError("boom")
        ),
        patch.object(control_plane, "_wait_for_store_read_grace") as grace,
        pytest.raises(RuntimeError, match="boom"),
    ):
        control_plane.run(args)

    grace.assert_called_once()


def test_nvrx_control_does_not_start_attribution_without_endpoint(tmp_path):
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
        ]
    )
    grpc_proc = MagicMock()

    with (
        patch.object(control_plane, "_create_tcp_store", return_value=object()),
        patch.object(control_plane, "AttributionManager") as manager_cls,
        patch.object(control_plane, "_start_grpc_log_servers", return_value=[grpc_proc]),
        patch.object(control_plane, "stop_grpc_log_servers"),
        patch.object(control_plane, "_run_control_rendezvous_loop"),
    ):
        control_plane.run(args)

    manager_cls.assert_not_called()


def test_control_parser_rejects_log_server_without_diagnostic_prefix():
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
        ]
    )
    ft_cfg = control_plane.FaultToleranceConfig.from_args(args)

    try:
        control_plane._validate_args(args, ft_cfg)
    except ValueError as exc:
        assert "--ft-log-server-log-prefix" in str(exc)
    else:
        raise AssertionError("expected missing log server diagnostic prefix to be rejected")


def test_control_parser_rejects_attribution_without_applog_prefix(tmp_path):
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--ft-log-server-log-prefix",
            str(tmp_path / "grpc"),
            "--ft-attribution-endpoint",
            "localhost",
        ]
    )
    ft_cfg = control_plane.FaultToleranceConfig.from_args(args)

    try:
        control_plane._validate_args(args, ft_cfg)
    except ValueError as exc:
        assert "--ft-per-cycle-applog-prefix" in str(exc)
    else:
        raise AssertionError("expected missing applog prefix to be rejected")


def test_control_parser_accepts_cycle_info_dir_with_job_id_without_applog_prefix(tmp_path):
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--ft-log-server-log-prefix",
            str(tmp_path / "grpc"),
            "--ft-cycle-info-dir",
            str(tmp_path / "nvrx"),
            "--ft-cycle-info-job-id",
            "job-a",
        ]
    )
    ft_cfg = control_plane.FaultToleranceConfig.from_args(args)

    control_plane._validate_args(args, ft_cfg)

    assert ft_cfg.cycle_info_dir == str(tmp_path / "nvrx")
    assert args.ft_cycle_info_job_id == "job-a"


def test_control_parser_rejects_cycle_info_dir_without_job_id(tmp_path):
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--ft-log-server-log-prefix",
            str(tmp_path / "grpc"),
            "--ft-cycle-info-dir",
            str(tmp_path / "nvrx"),
        ]
    )
    ft_cfg = control_plane.FaultToleranceConfig.from_args(args)

    try:
        control_plane._validate_args(args, ft_cfg)
    except ValueError as exc:
        assert "--ft-cycle-info-job-id" in str(exc)
    else:
        raise AssertionError("expected missing cycle-info job id to be rejected")


def test_control_parser_uses_launcher_config_file_alias(tmp_path):
    cfg_path = tmp_path / "ft.yaml"
    args = control_plane.parse_args(
        [
            "--nnodes",
            "2",
            "--rdzv-endpoint",
            "127.0.0.1:29500",
            "--ft-log-server-log-prefix",
            str(tmp_path / "grpc"),
            "--ft-cfg_path",
            str(cfg_path),
        ]
    )

    assert args.ft_cfg_path == str(cfg_path)


def test_control_rendezvous_loop_wires_attribution_to_barrier_state(tmp_path):
    args = SimpleNamespace(
        nnodes="2",
        rdzv_conf="",
        rdzv_id="job-a",
        rdzv_endpoint="127.0.0.1:29500",
        local_addr=None,
        ft_per_cycle_applog_prefix=str(tmp_path / "train.log"),
    )
    ft_cfg = SimpleNamespace(segment=None)
    attribution_service = MagicMock()
    services = control_plane.ControlServices(
        attribution_service=attribution_service,
        cycle_info_reporter=MagicMock(),
    )
    stop_event = threading.Event()
    states = []

    class FakeState:
        def __init__(self, *args, **kwargs):
            states.append(self)
            self._rounds = iter([0, 1])
            self._cycle_info_reporter = None
            self._attribution_service = None
            self._cycle_log_prefix = kwargs.get("cycle_log_prefix")
            self._active_node_addrs = ["node-a", "node-b"]
            self._standby_node_addrs = ["node-c"]
            self._active_ranks = [0, 1]
            self.final_terminal_requested = False

        def close_current_round_as_host(self, *args, **kwargs):
            try:
                round_id = next(self._rounds)
            except StopIteration as exc:
                raise control_plane.RendezvousClosedError("done") from exc

            self._cycle_info_reporter.report_cycle_start(
                CycleInfoRoundSnapshot(
                    cycle_number=round_id,
                    active_node_addrs=self._active_node_addrs,
                    standby_node_addrs=self._standby_node_addrs,
                    active_ranks=self._active_ranks,
                )
            )
            return round_id

        def _request_terminal_attribution_for_submitted_cycle(self):
            self.final_terminal_requested = True

    node_generator = MagicMock()
    node_generator.generate.return_value = object()

    with (
        patch.object(control_plane, "_RendezvousBarrierState", FakeState),
        patch.object(control_plane, "_NodeDescGenerator", return_value=node_generator),
    ):
        control_plane._run_control_rendezvous_loop(
            args,
            ft_cfg,
            store=object(),
            services=services,
            stop_event=stop_event,
        )

    reported_rounds = [
        call.args[0].cycle_number
        for call in services.cycle_info_reporter.report_cycle_start.call_args_list
    ]
    assert reported_rounds == [0, 1]
    assert states[0]._cycle_info_reporter is services.cycle_info_reporter
    assert states[0]._attribution_service is attribution_service
    assert states[0]._cycle_log_prefix == str(tmp_path / "train.log")
    assert states[0].final_terminal_requested
    attribution_service._submit_log.assert_not_called()
