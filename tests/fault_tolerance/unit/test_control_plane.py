# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nvidia_resiliency_ext.fault_tolerance import control_plane
from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import CycleInfoRoundSnapshot


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
    manager.start_if_needed.return_value = SimpleNamespace(endpoint="http://localhost:50050")
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


def test_control_rendezvous_loop_submits_attribution_and_reports_cycle_info(tmp_path):
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
            self._active_node_addrs = ["node-a", "node-b"]
            self._standby_node_addrs = ["node-c"]
            self._active_ranks = [0, 1]

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
    attribution_service._submit_log.assert_any_call(str(tmp_path / "train_cycle0.log"))
    attribution_service._submit_log.assert_any_call(str(tmp_path / "train_cycle1.log"))
