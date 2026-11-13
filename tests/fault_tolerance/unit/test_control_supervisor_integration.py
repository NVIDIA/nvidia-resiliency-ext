# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU subprocess checks through the unchanged public ft_launcher entrypoint."""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

import psutil
import pytest


@pytest.fixture
def launchers(tmp_path):
    processes = []

    def start(options, worker_source=None, *, standalone=False, bootstrap=None):
        output_path = tmp_path / f"console-{len(processes)}.log"
        command = [
            sys.executable,
            *(
                ["-c", bootstrap]
                if bootstrap
                else ["-m", "nvidia_resiliency_ext.fault_tolerance.launcher"]
            ),
            *(
                ["--standalone"]
                if standalone
                else [
                    f"--rdzv-endpoint=localhost:{_ports(1, start=61000)}",
                    f"--rdzv-id={uuid.uuid4().hex}",
                ]
            ),
            "--nnodes=1",
            "--nproc-per-node=1",
            "--monitor-interval=0.1",
            "--term_timeout=2",
            "--max-restarts=0",
            *options,
        ]
        if worker_source is not None:
            command.extend(["--no-python", sys.executable, "-c", worker_source])
        with output_path.open("w") as output:
            proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        processes.append(proc)
        return proc, output_path

    yield start
    for proc in processes:
        if proc.poll() is None:
            with contextlib.suppress(psutil.NoSuchProcess):
                descendants = psutil.Process(proc.pid).children(recursive=True)
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    for child in descendants:
                        with contextlib.suppress(psutil.NoSuchProcess):
                            child.kill()
                    proc.kill()
                    proc.wait(timeout=5)


def _ports(count, *, start=42000):
    for port in range(start, 65000, count):
        sockets = []
        try:
            for offset in range(count):
                stream = socket.socket()
                sockets.append(stream)
                stream.bind(("127.0.0.1", port + offset))
            return port
        except OSError:
            pass
        finally:
            for stream in sockets:
                stream.close()
    raise RuntimeError("no free consecutive ports")


@pytest.mark.parametrize(
    "levels,filename", [(0, "nvrx.log"), (1, "nvrx.log"), (2, "nvrx.log"), (1, "nvrx")]
)
def test_public_launcher_completion_and_log_routing(launchers, tmp_path, levels, filename):
    options = []
    if levels:
        options = [
            "--ft-nvrx-logfile",
            str(tmp_path / filename),
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-log-server-port",
            str(_ports(levels + 1)),
            "--ft-log-aggregator-count",
            str(levels),
            "--ft-log-server-graceful-shutdown-timeout=0.2",
        ]
    proc, console = launchers(options, "print('training-output-token', flush=True)")
    assert proc.wait(timeout=35) == 0, console.read_text()
    if levels:
        control = (tmp_path / filename).read_text()
        compute = (tmp_path / "nvrx_agent.log").read_text()
        assert "supervisor final exit code=0" in control
        assert "job_controller.py:" in control
        assert "Server shutdown complete" in control
        assert "worker group successfully finished" in compute
        assert "Control-host supervisor" not in compute
        assert "worker group successfully finished" not in control
        assert "training-output-token" in (tmp_path / "train_cycle0.log").read_text()
        assert not list(tmp_path.glob("*_grpc_*.log"))
        assert not list(tmp_path.glob("*_control.log"))
        if levels == 2:
            assert "gRPC leaf log server 0" in control
            assert "gRPC leaf log server 1" in control
    else:
        assert "training-output-token" in console.read_text()
        assert "supervisor final exit code=0" in console.read_text()


@pytest.mark.parametrize(
    "target,death_signal",
    [
        ("job_controller", signal.SIGKILL),
        ("node_agent", signal.SIGKILL),
        ("supervisor", signal.SIGTERM),
        ("supervisor", signal.SIGINT),
        ("supervisor", signal.SIGHUP),
        ("supervisor", signal.SIGQUIT),
    ],
)
def test_control_or_agent_loss_ends_generation_and_reaps_workers(
    launchers, tmp_path, target, death_signal
):
    marker = tmp_path / "worker.pid"
    source = (
        "import os,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    options = []
    if target == "supervisor" and death_signal == signal.SIGHUP:
        options = [
            f"--ft-nvrx-logfile={tmp_path}/nvrx.log",
            f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
            f"--ft-log-server-port={_ports(1)}",
            "--ft-log-aggregator-count=1",
            "--ft-log-server-graceful-shutdown-timeout=.2",
        ]
    proc, console = launchers(options, source)
    deadline = time.monotonic() + 20
    while not marker.exists():
        assert proc.poll() is None, console.read_text()
        assert time.monotonic() < deadline, console.read_text()
        time.sleep(0.05)
    worker = psutil.Process(int(marker.read_text()))
    direct_children = psutil.Process(proc.pid).children()
    modules = {child.pid: " ".join(child.cmdline()) for child in direct_children}
    assert any("fault_tolerance.job_controller" in cmd for cmd in modules.values())
    assert any("fault_tolerance.node_agent" in cmd for cmd in modules.values())
    if target == "supervisor":
        proc.send_signal(death_signal)
    else:
        child = next(p for p in direct_children if f"fault_tolerance.{target}" in modules[p.pid])
        child.send_signal(death_signal)
    expected = 128 + death_signal
    assert proc.wait(timeout=20) == expected, console.read_text()
    assert not worker.is_running(), console.read_text()
    assert all(not child.is_running() for child in direct_children), console.read_text()


@pytest.mark.parametrize("mode", ["stdin", "batch"])
@pytest.mark.parametrize("dumps_enabled", [False, True])
def test_cuda_gdb_mode_allows_supervised_training(launchers, tmp_path, mode, dumps_enabled):
    command_file = tmp_path / "commands.gdb"
    command_file.write_text("thread apply all bt\n")
    options = [
        f"--ft-timeout-dump-cuda-gdb-mode={mode}",
        f"--ft-timeout-dump-enabled={str(dumps_enabled).lower()}",
        "--ft-timeout-dump-tools=cuda-gdb",
        # No timeout occurs: a known executable is enough for startup validation.
        f"--ft-timeout-dump-cuda-gdb={sys.executable}",
        f"--ft-timeout-dump-cuda-gdb-command-file={command_file}",
        f"--ft-nvrx-logfile={tmp_path}/nvrx/logs/nvrx.log",
        f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
        f"--ft-log-server-port={_ports(1)}",
        "--ft-log-server-graceful-shutdown-timeout=.2",
    ]
    proc, console = launchers(options, "print('training-with-cuda-gdb-mode', flush=True)")
    assert proc.wait(timeout=35) == 0, console.read_text()
    assert "training-with-cuda-gdb-mode" in (tmp_path / "train_cycle0.log").read_text()
    assert "supervisor final exit code=0" in (tmp_path / "nvrx/logs/nvrx.log").read_text()


def test_worker_restart_does_not_restart_supervisor_or_control(launchers, tmp_path):
    attempts = tmp_path / "attempts"
    source = (
        "import sys; from pathlib import Path; "
        f"p=Path({str(attempts)!r}); first=not p.exists(); "
        "p.write_text(p.read_text()+'retry\\n' if not first else 'initial\\n'); "
        "sys.exit(1 if first else 0)"
    )
    proc, console = launchers(["--max-restarts=1"], source)
    assert proc.wait(timeout=35) == 0, console.read_text()
    assert attempts.read_text().splitlines() == ["initial", "retry"]
    output = console.read_text()
    assert output.count("Starting control-host supervisor") == 1
    assert output.count("job-controller rendezvous loop started") == 1


def test_no_progress_preserves_configured_no_restart_exit_code(launchers, tmp_path):
    checkpoint = tmp_path / "latest_checkpointed_iteration.txt"
    checkpoint.write_text("0")
    proc, console = launchers(
        [
            "--max-restarts=5",
            "--ft-max-no-progress-cycles=1",
            "--ft-no-restart-exit-code=77",
            "--ft-checkpoint-iteration-file",
            str(checkpoint),
        ],
        "raise SystemExit(1)",
    )
    assert proc.wait(timeout=30) == 77, console.read_text()
    assert "supervisor final exit code=77" in console.read_text()


@pytest.mark.parametrize("segment_conf", ["", ",segment=2"])
def test_two_compute_participants_share_control_without_extra_nonhost_supervisor(
    launchers, tmp_path, monkeypatch, segment_conf
):
    monkeypatch.setenv("NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE", "1")
    options = [
        "--nnodes=2",
        "--rdzv-endpoint",
        f"localhost:{_ports(1, start=62000)}",
        "--rdzv-id",
        uuid.uuid4().hex,
        f"--ft-nvrx-logfile={tmp_path}/nvrx.log",
        f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
        f"--ft-log-server-port={_ports(3)}",
        "--ft-log-aggregator-count=2",
        "--ft-log-server-graceful-shutdown-timeout=.2",
    ]
    source = "import os; print('training rank='+os.environ['RANK'], flush=True)"
    host, host_log = launchers(
        options + [f"--rdzv-conf=is_host=true,join_timeout=15{segment_conf}"],
        source,
        standalone=False,
    )
    peer, peer_log = launchers(
        options + [f"--rdzv-conf=is_host=false,join_timeout=15{segment_conf}"],
        source,
        standalone=False,
    )
    assert host.wait(timeout=35) == 0, host_log.read_text() + peer_log.read_text()
    assert peer.wait(timeout=10) == 0, peer_log.read_text()
    app = (tmp_path / "train_cycle0.log").read_text()
    assert "training rank=0" in app
    assert "training rank=1" in app
    control = (tmp_path / "nvrx.log").read_text()
    compute = (tmp_path / "nvrx_agent.log").read_text()
    assert compute.count("worker group successfully finished") == 2
    assert "Starting control-host supervisor" in control
    assert "Starting control-host supervisor" not in compute
    assert "Starting control-host supervisor" not in peer_log.read_text()


@pytest.mark.parametrize("service", ["control", "grpc"])
def test_required_service_startup_failure_ends_generation(launchers, tmp_path, service):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        if service == "control":
            options = ["--rdzv-endpoint", f"127.0.0.1:{port}", "--rdzv-conf=is_host=true"]
        else:
            options = [
                "--ft-enable-log-server=true",
                "--ft-log-server-port",
                str(port),
                "--ft-per-cycle-applog-prefix",
                str(tmp_path / "train.log"),
                "--ft-log-server-graceful-shutdown-timeout=0.1",
            ]
        proc, console = launchers(options, "import time; time.sleep(60)")
        assert proc.wait(timeout=20) == 1, console.read_text()


def test_managed_attribution_is_a_sibling_and_uses_combined_control_log(tmp_path, launchers):
    # Exercise a real managed child and readiness endpoint without an LLM dependency.
    # The attribution analysis itself is outside the supervisor's responsibility.
    with socket.socket() as check_port:
        try:
            check_port.bind(("127.0.0.1", 50050))
        except OSError:
            pytest.skip("managed attribution port is already occupied")
    fake_service = tmp_path / "attr.py"
    fake_service.write_text(
        '''
import signal, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')
print('managed-attribution-process-marker', file=sys.stderr, flush=True)
def stop(*_):
    print('managed-attribution-final-marker', file=sys.stderr, flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
HTTPServer(('127.0.0.1', 50050), Health).serve_forever()
'''
    )
    key_file = tmp_path / "test-key"
    key_file.write_text("unused-test-key")
    log_path = tmp_path / "nvrx.log"
    script = f'''
import sys
from nvidia_resiliency_ext.fault_tolerance import attribution_manager, launcher
attribution_manager._attribution_command = lambda: [sys.executable, '-u', {str(fake_service)!r}]
launcher.main()
'''
    worker = (
        "import os,psutil; parent=psutil.Process(os.getppid()); "
        "siblings=parent.parent().children(); "
        f"assert sum({str(fake_service)!r} in p.cmdline() for p in siblings)==1; "
        "print('managed-worker-finished')"
    )
    proc, console = launchers(
        [
            "--ft-nvrx-logfile",
            str(log_path),
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-enable-log-server=false",
            "--ft-attribution-endpoint=localhost",
            "--ft-attribution-llm-api-key-file",
            str(key_file),
        ],
        worker,
        bootstrap=script,
    )
    assert proc.wait(timeout=30) == 0, console.read_text() + (tmp_path / "nvrx.log").read_text()
    control = (tmp_path / "nvrx.log").read_text()
    assert control.count("managed-attribution-process-marker") == 1
    assert "managed-attribution-final-marker" in control
    assert "managed-attribution-process-marker" not in (tmp_path / "nvrx_agent.log").read_text()
    assert not list(tmp_path.glob("*_attribution.log"))


@pytest.mark.parametrize("local_logs", [False, True])
def test_standalone_keeps_embedded_control_without_aggregation(launchers, tmp_path, local_logs):
    options = []
    if local_logs:
        options = [
            f"--ft-nvrx-logfile={tmp_path}/nvrx.log",
            f"--ft-per-cycle-applog-prefix={tmp_path}/train.log",
            "--ft-enable-log-server=true",
        ]
    proc, console = launchers(options, "print('standalone-finished', flush=True)", standalone=True)
    assert proc.wait(timeout=25) == 0, console.read_text()
    output = console.read_text()
    if local_logs:
        assert "standalone-finished" in (tmp_path / "train_cycle0.log").read_text()
        output += (tmp_path / "nvrx_agent.log").read_text()
    else:
        assert "standalone-finished" in output
    assert "Starting control-host supervisor" not in output
    assert "node-agent started" not in output
    assert "gRPC log server started" not in output
    assert not (tmp_path / "nvrx.log").exists()
    assert not list(tmp_path.glob("*_grpc_*.log"))


def test_agent_waits_for_delayed_controller_without_readiness_ipc(launchers):
    bootstrap = """
import time
from nvidia_resiliency_ext.fault_tolerance import launcher
# Delay only the controller interpreter; the agent must tolerate startup ordering.
import nvidia_resiliency_ext.fault_tolerance.control_supervisor as supervisor
import subprocess, sys
original = supervisor.spawn_child
def spawn(module, args, **kwargs):
    if module.endswith('.job_controller'):
        # Use the regular config transport, adding delay before executing the module.
        original_popen = subprocess.Popen
        def delayed(command, **options):
            code = "import time,runpy; time.sleep(2); runpy.run_module(" + repr(module) + ", run_name='__main__')"
            return original_popen([sys.executable, '-u', '-c', code], **options)
        subprocess.Popen = delayed
        try:
            return original(module, args, **kwargs)
        finally:
            subprocess.Popen = original_popen
    return original(module, args, **kwargs)
supervisor.spawn_child = spawn
launcher.main()
"""
    proc, console = launchers(
        [], "print('delayed-control-finished', flush=True)", bootstrap=bootstrap
    )
    assert proc.wait(timeout=30) == 0, console.read_text()
    assert "delayed-control-finished" in console.read_text()


@pytest.mark.parametrize("death_signal", [signal.SIGTERM, signal.SIGKILL])
def test_auxiliary_signal_exit_fails_generation_and_reaps_children(
    launchers, tmp_path, death_signal
):
    marker = tmp_path / "worker.pid"
    source = (
        "import os,time,signal; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, lambda *_: (print('final-worker-log', flush=True), exit(0))); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    proc, console = launchers(
        [
            "--ft-nvrx-logfile",
            str(tmp_path / "nvrx.log"),
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-log-server-port",
            str(_ports(1)),
            "--ft-log-aggregator-count=1",
            "--ft-log-server-graceful-shutdown-timeout=.2",
        ],
        source,
    )
    deadline = time.monotonic() + 20
    while not marker.exists():
        assert proc.poll() is None, console.read_text()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    children = psutil.Process(proc.pid).children()
    root = next(
        child for child in children if 'shared_utils.grpc_log_server' in ' '.join(child.cmdline())
    )
    root.send_signal(death_signal)
    # The gRPC server handles SIGTERM and exits zero; an unexpected clean
    # auxiliary exit is still a service failure. SIGKILL preserves signal status.
    expected = 1 if death_signal == signal.SIGTERM else 128 + death_signal
    assert proc.wait(timeout=25) == expected, console.read_text()
    assert all(not child.is_running() for child in children)
    assert not psutil.pid_exists(int(marker.read_text()))


@pytest.mark.parametrize("target", ["node-agent", "supervisor"])
def test_agent_sigterm_drains_final_logs_before_aggregators_stop(
    launchers, tmp_path, monkeypatch, target
):
    # Keep real agent cleanup running past the workers' two-second allowance.
    (tmp_path / "slow_cleanup_agent.py").write_text(
        """
import time
from nvidia_resiliency_ext.fault_tolerance import launcher, node_agent
original_cleanup = launcher.LocalElasticAgent.shutdown_rank_monitors
def cleanup(self):
    time.sleep(3)
    original_cleanup(self)
    print('agent-monitor-cleanup-finished', flush=True)
launcher.LocalElasticAgent.shutdown_rank_monitors = cleanup
node_agent.main()
"""
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    bootstrap = """
from nvidia_resiliency_ext.fault_tolerance import control_supervisor, launcher
original_spawn = control_supervisor.spawn_child
def spawn(module, args, **kwargs):
    if module.endswith('.node_agent'):
        module = 'slow_cleanup_agent'
    return original_spawn(module, args, **kwargs)
control_supervisor.spawn_child = spawn
launcher.main()
"""
    marker = tmp_path / "worker.pid"
    source = """
import os, signal, time
from pathlib import Path
def stop(*_):
    print('worker-shutdown-started', flush=True)
    time.sleep(.5)
    print('worker-final-log-before-exit', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(MARKER).write_text(str(os.getpid()))
time.sleep(60)
""".replace(
        'MARKER', repr(str(marker))
    )
    proc, console = launchers(
        [
            "--ft-nvrx-logfile",
            str(tmp_path / "nvrx.log"),
            "--ft-per-cycle-applog-prefix",
            str(tmp_path / "train.log"),
            "--ft-log-server-port",
            str(_ports(3)),
            "--ft-log-aggregator-count=2",
            "--ft-log-server-graceful-shutdown-timeout=.2",
        ],
        source,
        bootstrap=bootstrap,
    )
    deadline = time.monotonic() + 20
    while not marker.exists():
        assert proc.poll() is None, console.read_text()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    if target == "supervisor":
        proc.terminate()
    else:
        agent = next(
            child
            for child in psutil.Process(proc.pid).children()
            if 'slow_cleanup_agent' in child.cmdline()
        )
        agent.terminate()
    assert proc.wait(timeout=25) == 143, console.read_text()
    compute = (tmp_path / "nvrx_agent.log").read_text()
    assert "agent-monitor-cleanup-finished" in compute
    assert "worker-final-log-before-exit" in (tmp_path / "train_cycle0.log").read_text()
    assert "Agent's exit code" in compute or "exit code=143" in compute
    assert "Server shutdown complete" in (tmp_path / "nvrx.log").read_text()


def _allocation_bootstrap(job_id, rank=0):
    return f"""
import os
os.environ.pop('SLURM_ARRAY_JOB_ID', None)
os.environ['SLURM_JOB_ID'] = {job_id!r}
os.environ['SLURM_PROCID'] = {str(rank)!r}
os.environ['SLURM_RESTART_COUNT'] = '7'
from nvidia_resiliency_ext.fault_tolerance import launcher
launcher.main()
"""


def test_external_control_serves_two_compute_nodes_and_uses_training_job_identity(
    launchers, tmp_path, monkeypatch
):
    monkeypatch.setenv("NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE", "1")
    cycle_dir = tmp_path / "cycles"
    gate = tmp_path / "finish-workers"
    common = [
        "--nnodes=2",
        f"--rdzv-endpoint=localhost:{_ports(1, start=62000)}",
        f"--rdzv-id={uuid.uuid4().hex}",
        "--ft-nvrx-logfile",
        str(tmp_path / "nvrx.log"),
        "--ft-per-cycle-applog-prefix",
        str(tmp_path / "train.log"),
        "--ft-cycle-info-dir",
        str(cycle_dir),
        "--ft-log-server-port",
        str(_ports(3)),
        "--ft-log-aggregator-count=2",
        "--ft-log-server-graceful-shutdown-timeout=.5",
    ]
    control, control_console = launchers(
        common
        + [
            "--ft-control-only",
            "--rdzv-conf=is_host=true",
            "--ft-cycle-info-job-id=12345",
            "--nproc-per-node=gpu",
        ],
        bootstrap=_allocation_bootstrap("99999"),
    )
    source = """
import os,time
from pathlib import Path
rank = os.environ['RANK']
Path(MARKER_PREFIX + rank).write_text(str(os.getpid()))
deadline = time.monotonic() + 20
while not Path(GATE).exists():
    assert time.monotonic() < deadline
    time.sleep(.05)
print('external-training-rank=' + rank, flush=True)
print('cycle-info-path=' + os.environ['NVRX_CURRENT_CYCLE_INFO'], flush=True)
""".replace(
        'MARKER_PREFIX', repr(str(tmp_path / 'worker-'))
    ).replace(
        'GATE', repr(str(gate))
    )
    peers = [
        launchers(
            common + ["--rdzv-conf=is_host=false"],
            source,
            bootstrap=_allocation_bootstrap("12345", rank),
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 30
    while not all((tmp_path / f"worker-{rank}").exists() for rank in range(2)):
        assert control.poll() is None, (
            control_console.read_text() + (tmp_path / "nvrx.log").read_text()
        )
        for peer, console in peers:
            assert peer.poll() is None, console.read_text()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    children = psutil.Process(control.pid).children()
    commands = [' '.join(child.cmdline()) for child in children]
    assert sum('fault_tolerance.job_controller' in cmd for cmd in commands) == 1
    assert not any('fault_tolerance.node_agent' in cmd for cmd in commands)
    assert sum('shared_utils.grpc_log_' in cmd for cmd in commands) == 3
    gate.touch()
    for peer, console in peers:
        assert peer.wait(timeout=30) == 0, console.read_text()
    assert control.wait(timeout=30) == 0, (
        control_console.read_text() + (tmp_path / "nvrx.log").read_text()
    )
    app_log = (tmp_path / "train_cycle0.log").read_text()
    assert "external-training-rank=0" in app_log
    assert "external-training-rank=1" in app_log
    assert f"cycle-info-path={cycle_dir}/cycle_info.12345.current" in app_log
    assert (cycle_dir / "cycle_info.12345.0.0").exists()
    assert (cycle_dir / "cycle_info.12345.current").exists()
    assert not list(cycle_dir.glob("cycle_info.99999.*"))
    control_log = (tmp_path / "nvrx.log").read_text()
    agent_log = (tmp_path / "nvrx_agent.log").read_text()
    assert "External control host: no local node-agent is started" in control_log
    assert agent_log.count("worker group successfully finished") == 2
    assert "job_controller.py:" not in agent_log
    assert not (tmp_path / "nvrx_control.log").exists()
    assert all(not child.is_running() for child in children)


def test_external_control_preserves_no_restart_exit_code(launchers, tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE", "1")
    checkpoint = tmp_path / "latest_checkpointed_iteration.txt"
    checkpoint.write_text("0")
    common = [
        f"--rdzv-endpoint=localhost:{_ports(1, start=62000)}",
        f"--rdzv-id={uuid.uuid4().hex}",
        "--max-restarts=5",
        "--ft-max-no-progress-cycles=1",
        "--ft-no-restart-exit-code=77",
        "--ft-checkpoint-iteration-file",
        str(checkpoint),
    ]
    control, control_console = launchers(common + ["--ft-control-only", "--rdzv-conf=is_host=true"])
    peer, peer_console = launchers(common + ["--rdzv-conf=is_host=false"], "raise SystemExit(1)")
    assert peer.wait(timeout=30) == 77, peer_console.read_text()
    assert control.wait(timeout=15) == 77, control_console.read_text()
    assert "node-agent started" not in control_console.read_text()


def test_external_control_sigterm_cleans_up_without_a_node_agent(launchers):
    control, console = launchers(["--ft-control-only", "--nnodes=2", "--rdzv-conf=is_host=true"])
    deadline = time.monotonic() + 20
    while "job-controller rendezvous loop started" not in console.read_text():
        assert control.poll() is None, console.read_text()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    children = psutil.Process(control.pid).children()
    assert len(children) == 1
    control.terminate()
    assert control.wait(timeout=15) == 143, console.read_text()
    assert all(not child.is_running() for child in children)
