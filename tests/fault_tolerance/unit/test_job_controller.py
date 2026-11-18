# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance import control_supervisor, job_controller, launcher
from nvidia_resiliency_ext.fault_tolerance.child_process import spawn_child
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.cycle_info_writer import CycleInfoRoundSnapshot
from nvidia_resiliency_ext.shared_utils import telemetry


def _args(*options):
    return control_supervisor._job_controller_args(
        launcher.parse_args(
            [
                "--nnodes=2",
                "--rdzv-endpoint=127.0.0.1:29500",
                "--rdzv-id=job-a",
                *options,
                "train.py",
            ]
        )
    )


@pytest.fixture(autouse=True)
def _isolated_controller_runtime():
    real_setup = telemetry.setup_telemetry
    # Providers are process-global; unit-test invocations must not initialize
    # Lens in the pytest process. The subprocess export test uses real providers.
    with (
        patch.object(job_controller.time, "sleep"),
        patch.object(telemetry, "setup_telemetry"),
        patch.object(telemetry, "shutdown"),
    ):
        yield real_setup


def test_controller_owns_clients_and_reporter_without_service_processes(tmp_path):
    args = _args(
        "--ft-per-cycle-applog-prefix",
        str(tmp_path / "train.log"),
        "--ft-cycle-info-dir",
        str(tmp_path / "cycle-info"),
        "--ft-attribution-endpoint=localhost",
        "--ft-attribution-stop-action=no-restart",
    )
    client, reporter = MagicMock(), MagicMock()
    with (
        patch.object(job_controller, "_create_tcp_store", return_value=object()),
        patch.object(job_controller, "AttributionService", return_value=client) as client_class,
        patch.object(job_controller, "CycleInfoReporter", return_value=reporter) as reporter_class,
        patch.object(job_controller, "_run_control_rendezvous_loop", return_value=0),
        patch("subprocess.Popen") as spawn,
    ):
        assert job_controller.run(args) == 0
    client_class.assert_called_once_with(endpoint="http://127.0.0.1:50050", enforce_stop=True)
    reporter_class.assert_called_once_with(
        str(tmp_path / "cycle-info"), cycle_log_prefix=str(tmp_path / "train.log")
    )
    client.start_poller.assert_called_once_with(
        node_id=job_controller.socket.getfqdn(job_controller.socket.gethostname())
    )
    client.stop_poller.assert_called_once()
    reporter.shutdown.assert_called_once()
    spawn.assert_not_called()


def test_controller_without_attribution_does_not_create_a_client():
    args = _args()
    with patch.object(job_controller, "AttributionService") as client:
        resources = job_controller._start_controller_resources(
            args, FaultToleranceConfig.from_args(args), node_id="control.example"
        )
    assert resources.attribution_service is None
    client.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "loop_failure", "store_failure", "signal"])
def test_controller_owns_telemetry_before_polling_and_shuts_it_down_last(tmp_path, outcome):
    args = _args(
        f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
        "--ft-attribution-endpoint=http://attr.example:8000",
    )
    events = []
    handle, client = MagicMock(), MagicMock()
    client.start_poller.side_effect = lambda **_: events.append("poller-start")
    client.stop_poller.side_effect = lambda: events.append("poller-stop")

    def setup(*_):
        events.append("telemetry-start")
        return handle

    def create_store(*_):
        events.append("store-start")
        if outcome == "store_failure":
            raise RuntimeError("store failed")
        return object()

    def run_loop(*_):
        events.append("loop")
        if outcome == "loop_failure":
            raise RuntimeError("loop failed")
        if outcome == "signal":
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return 0

    with (
        patch.object(job_controller.socket, "getfqdn", return_value="control.example"),
        patch.object(telemetry, "setup_telemetry", side_effect=setup) as setup_call,
        patch.object(telemetry, "shutdown", side_effect=lambda _: events.append("telemetry-stop")),
        patch.object(job_controller, "_create_tcp_store", side_effect=create_store),
        patch.object(job_controller, "AttributionService", return_value=client),
        patch.object(job_controller, "_run_control_rendezvous_loop", side_effect=run_loop),
        patch.object(job_controller, "_wait_for_store_read_grace"),
    ):
        if outcome in ("loop_failure", "store_failure"):
            with pytest.raises(RuntimeError, match="failed"):
                job_controller.run(args)
        else:
            assert job_controller.run(args) == (143 if outcome == "signal" else 0)

    setup_call.assert_called_once_with(
        "nvrx.job_controller",
        "job-controller-control.example",
        {"nv.nvrx.ftl.node": "control.example"},
    )
    assert events[:2] == ["telemetry-start", "store-start"]
    assert events[-1] == "telemetry-stop"
    if outcome == "store_failure":
        client.start_poller.assert_not_called()
    else:
        client.start_poller.assert_called_once_with(node_id="control.example")
        assert events.index("poller-start") < events.index("loop")
        assert events.index("poller-stop") < events.index("telemetry-stop")


@pytest.mark.parametrize("exit_code", [0, 93, 143])
def test_telemetry_shutdown_failure_preserves_controller_exit_and_signal_handlers(exit_code):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    with (
        patch.object(telemetry, "setup_telemetry", return_value=MagicMock()),
        patch.object(telemetry, "shutdown", side_effect=RuntimeError("export failed")),
        patch.object(job_controller, "_create_tcp_store", return_value=object()),
        patch.object(
            job_controller,
            "_start_controller_resources",
            return_value=job_controller.ControllerResources(),
        ),
        patch.object(job_controller, "_run_control_rendezvous_loop", return_value=exit_code),
    ):
        assert job_controller.run(_args()) == exit_code
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_controller_works_when_telemetry_is_unavailable(_isolated_controller_runtime):
    with (
        patch.object(telemetry, "_AVAILABLE", False),
        patch.object(telemetry, "setup_telemetry", _isolated_controller_runtime),
        patch.object(job_controller, "_create_tcp_store", return_value=object()),
        patch.object(
            job_controller,
            "_start_controller_resources",
            return_value=job_controller.ControllerResources(),
        ),
        patch.object(job_controller, "_run_control_rendezvous_loop", return_value=93),
    ):
        assert job_controller.run(_args()) == 93


@pytest.mark.skipif(not telemetry._AVAILABLE, reason="requires nemo-lens")
def test_spawned_controller_exports_attribution_with_inherited_job_and_own_identity(
    tmp_path, monkeypatch
):
    pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    (tmp_path / "controller_otel_probe.py").write_text(
        """
import json, os, socket, threading
from unittest.mock import patch
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from nvidia_resiliency_ext.fault_tolerance import job_controller
from nvidia_resiliency_ext.fault_tolerance.child_process import read_child_args
from nvidia_resiliency_ext.shared_utils import telemetry
from nvidia_resiliency_ext.shared_utils.health_check import AttributionService, _AttributionGetOutcome

exporter = InMemorySpanExporter()
polled = threading.Event()
original_setup = telemetry._setup_telemetry
def setup(config, **kwargs):
    config.metrics_enabled = False
    config.logs_enabled = False
    return original_setup(config, span_exporter=exporter, **kwargs)
def get_results(self, *args, **kwargs):
    polled.set()
    return _AttributionGetOutcome.CONTINUE
def rendezvous(args, cfg, store, resources, stop_event):
    resources.attribution_service._terminal_pending = '/logs/train_cycle0.log'
    assert polled.wait(5), 'attribution poller did not run'
    return 0
with (
    patch.object(telemetry, '_setup_telemetry', setup),
    patch.object(job_controller, '_create_tcp_store', return_value=object()),
    patch.object(job_controller, '_run_control_rendezvous_loop', rendezvous),
    patch.object(job_controller, '_wait_for_store_read_grace'),
    patch.object(AttributionService, '_get_results', get_results),
    patch.object(AttributionService, '_RESULT_POLL_INTERVAL_SECONDS', .01),
):
    assert job_controller.run(read_child_args()) == 0
spans = [s for s in exporter.get_finished_spans() if s.name == 'nv.nvrx.ftl.attribution']
assert len(spans) == 1, [s.name for s in exporter.get_finished_spans()]
span = spans[0]
keys = ('nv.dl.job.uuid', 'example.attribute', 'service.name', 'service.instance.id', 'nv.nvrx.ftl.node')
print(json.dumps({
    'resource': {key: span.resource.attributes.get(key) for key in keys},
    'node': span.attributes.get('nv.nvrx.ftl.node'),
    'hostname': socket.getfqdn(socket.gethostname()),
    'pid': os.getpid(),
}))
"""
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    carrier = (
        "nv.dl.job.uuid=job-test,example.attribute=preserved,"
        "service.instance.id=parent-instance,nv.nvrx.ftl.node=parent-node"
    )
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", carrier)
    monkeypatch.setenv("NVRX_OTEL_ENABLED", "1")
    monkeypatch.setenv("NVRX_OTEL_SPAN_GROUPS", "default")
    args = _args(
        f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
        "--ft-attribution-endpoint=http://attr.example:8000",
    )
    proc = spawn_child(
        "controller_otel_probe", args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)
        raise
    assert proc.returncode == 0, stderr.decode()
    data = json.loads(stdout)
    assert data["pid"] != os.getpid()
    assert data["node"] == data["hostname"]
    assert data["resource"] == {
        "nv.dl.job.uuid": "job-test",
        "example.attribute": "preserved",
        "service.name": "nvrx.job_controller",
        "service.instance.id": f"job-controller-{data['hostname']}",
        "nv.nvrx.ftl.node": data["hostname"],
    }
    assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == carrier


def test_partial_startup_stops_client_and_reporter(tmp_path):
    args = _args(
        "--ft-per-cycle-applog-prefix",
        str(tmp_path / "train.log"),
        "--ft-cycle-info-dir",
        str(tmp_path / "cycle-info"),
        "--ft-attribution-endpoint=localhost",
    )
    client, reporter = MagicMock(), MagicMock()
    client.start_poller.side_effect = RuntimeError("poller startup failed")
    with (
        patch.object(job_controller, "AttributionService", return_value=client),
        patch.object(job_controller, "CycleInfoReporter", return_value=reporter),
        pytest.raises(RuntimeError, match="poller startup failed"),
    ):
        job_controller._start_controller_resources(
            args, FaultToleranceConfig.from_args(args), node_id="control.example"
        )
    client.stop_poller.assert_called_once()
    reporter.shutdown.assert_called_once()


@pytest.mark.parametrize("failure", [False, True])
def test_controller_cleanup_holds_store_open_after_stopping_its_resources(failure):
    args = _args()
    client, reporter = MagicMock(), MagicMock()
    events = []
    client.stop_poller.side_effect = lambda: events.append("client-stopped")
    reporter.shutdown.side_effect = lambda: events.append("reporter-stopped")
    resources = job_controller.ControllerResources(
        attribution_service=client, cycle_info_reporter=reporter
    )
    with (
        patch.object(job_controller, "_create_tcp_store", return_value=object()),
        patch.object(job_controller, "_start_controller_resources", return_value=resources),
        patch.object(
            job_controller,
            "_run_control_rendezvous_loop",
            return_value=0,
            side_effect=RuntimeError("loop failed") if failure else None,
        ),
        patch.object(
            job_controller,
            "_wait_for_store_read_grace",
            side_effect=lambda _: events.append("store-grace"),
        ),
    ):
        if failure:
            with pytest.raises(RuntimeError, match="loop failed"):
                job_controller.run(args)
        else:
            assert job_controller.run(args) == 0
    assert events == ["reporter-stopped", "client-stopped", "store-grace"]


def test_store_read_grace_waits_out_the_remainder():
    with (
        patch.object(job_controller.time, "monotonic", return_value=100.0),
        patch.object(job_controller.time, "sleep") as sleep,
    ):
        job_controller._wait_for_store_read_grace(since=99.0)
    sleep.assert_called_once_with(2.0)


def test_store_read_grace_is_skipped_after_slow_resource_cleanup():
    with (
        patch.object(job_controller.time, "monotonic", return_value=100.0),
        patch.object(job_controller.time, "sleep") as sleep,
    ):
        job_controller._wait_for_store_read_grace(since=90.0)
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "options,expected",
    [
        (["--rdzv-backend=etcd"], "supports only"),
        (["--ft-attribution-endpoint=localhost"], "requires --ft-per-cycle-applog-prefix"),
        (["--ft-per-cycle-applog-prefix=relative/train.log"], "must be an absolute path"),
    ],
)
def test_controller_validates_its_configuration(options, expected):
    args = _args(*options)
    with pytest.raises(ValueError, match=expected):
        job_controller._validate_args(args, FaultToleranceConfig.from_args(args))


@pytest.mark.parametrize(
    "options,expected_segment",
    [
        ([], None),
        (["--rdzv-conf=segment=2"], None),
        (["--ft-segment=2"], 2),
        (["--rdzv-conf=segment=2", "--ft-segment=1"], 1),
    ],
)
def test_controller_and_agent_use_the_same_effective_segment(options, expected_segment):
    args = _args(*options)
    with (
        patch.object(launcher.LocalElasticAgent, "setup_rank_monitors_early", return_value={}),
        patch.object(launcher, "_start_diagnostic_collector"),
    ):
        agent_config, _, _ = launcher.config_from_args(args)

    state = MagicMock()
    state.close_current_round_as_host.side_effect = job_controller.RendezvousClosedError("done")
    state.get_shutdown_reason.return_value = job_controller.RDZV_SHUTDOWN_REASON_GRACEFUL
    with patch.object(job_controller, "_RendezvousBarrierState", return_value=state) as state_type:
        assert (
            job_controller._run_control_rendezvous_loop(
                args,
                FaultToleranceConfig.from_args(args),
                store=object(),
                services=job_controller.ControllerResources(),
                stop_event=threading.Event(),
            )
            == 0
        )

    assert state_type.call_args.kwargs["segment"] == expected_segment
    assert agent_config.rdzv_configs["segment"] == expected_segment


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
    services = job_controller.ControllerResources(
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
                raise job_controller.RendezvousClosedError("done") from exc

            self._cycle_info_reporter.report_cycle_start(
                CycleInfoRoundSnapshot(
                    cycle_number=round_id,
                    active_node_addrs=self._active_node_addrs,
                    standby_node_addrs=self._standby_node_addrs,
                    active_ranks=self._active_ranks,
                )
            )
            return round_id

        def get_shutdown_reason(self):
            return job_controller.RDZV_SHUTDOWN_REASON_GRACEFUL

        def _request_terminal_attribution_for_submitted_cycle(self):
            self.final_terminal_requested = True

    node_generator = MagicMock()
    node_generator.generate.return_value = object()

    with (
        patch.object(job_controller, "_RendezvousBarrierState", FakeState),
        patch.object(job_controller, "_NodeDescGenerator", return_value=node_generator),
    ):
        job_controller._run_control_rendezvous_loop(
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


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_controller_signal_exit_is_not_success(signum):
    args = _args()
    previous = signal.getsignal(signum)

    def interrupted(*_):
        signal.getsignal(signum)(signum, None)
        return 0

    with (
        patch.object(job_controller, "_create_tcp_store", return_value=object()),
        patch.object(
            job_controller,
            "_start_controller_resources",
            return_value=job_controller.ControllerResources(),
        ),
        patch.object(job_controller, "_run_control_rendezvous_loop", side_effect=interrupted),
    ):
        assert job_controller.run(args) == 128 + signum
    assert signal.getsignal(signum) == previous


@pytest.mark.parametrize("exit_code", [0, 42, 77, 143])
def test_controller_main_preserves_its_exit_status(exit_code):
    with (
        patch.object(job_controller, "read_child_args", return_value=_args()),
        patch.object(job_controller, "run", return_value=exit_code),
        pytest.raises(SystemExit) as exited,
    ):
        job_controller.main()
    assert exited.value.code == exit_code


def test_controller_rejects_independent_module_invocation():
    result = subprocess.run(
        [sys.executable, "-m", job_controller.__name__, "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "job-controller must be started by ft_launcher" in result.stderr
    assert "Starting control TCPStore" not in result.stdout + result.stderr


def test_external_cycle_reporter_uses_explicit_compute_job_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-allocation")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "9")
    args = launcher.parse_args(
        [
            "--ft-control-only",
            "--rdzv-endpoint=localhost:29500",
            "--ft-cycle-info-dir",
            str(tmp_path / "cycles"),
            "--ft-cycle-info-job-id=training-job",
        ]
    )
    with patch.object(job_controller, "CycleInfoReporter") as reporter:
        job_controller._start_controller_resources(
            args, FaultToleranceConfig.from_args(args), node_id="control.example"
        )
    reporter.assert_called_once_with(
        str(tmp_path / "cycles"),
        cycle_log_prefix=None,
        cycle_info_job_id="training-job",
        attempt_index=0,
    )
