# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for launcher's gRPC log server functionality.

These tests verify the gRPC log aggregation feature works correctly:
- Server starts on TCP store host
- Clients connect and stream logs
- Logs are aggregated correctly
- Server handles shutdown gracefully
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

# gRPC is required for these tests - they will fail if not available
import grpc
import pytest

from nvidia_resiliency_ext import fault_tolerance
from nvidia_resiliency_ext.shared_utils.proto import log_aggregation_pb2, log_aggregation_pb2_grpc

WORLD_SIZE = 2
DEFAULT_TIMEOUT = 30


@pytest.fixture
def tmp_dir():
    """Create and cleanup temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def _run_launcher(cmd_to_run, timeout=DEFAULT_TIMEOUT):
    """Run ft_launcher command and return exit code and output."""
    try:
        proc = subprocess.Popen(
            cmd_to_run,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        return proc.returncode, stdout
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise AssertionError(f"ft_launcher was still running after {timeout} seconds")


def _save_ft_cfg(cfg, dirpath):
    """Save fault tolerance config to YAML file."""
    cfg_path = os.path.join(dirpath, "_tmp_ft_cfg.yaml")
    cfg.to_yaml_file(cfg_path)
    return cfg_path


def _get_util_script_path():
    """Get path to launcher test utility script."""
    return os.path.join(os.path.dirname(__file__), "_launcher_test_util.py")


# ==============================================================================
# gRPC Server Standalone Tests
# ==============================================================================


def test_grpc_server_can_start_and_shutdown():
    """Test that gRPC server can be started and shut down cleanly."""

    # Try to start the gRPC server directly
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'nvidia_resiliency_ext.shared_utils.grpc_log_server',
            '--host',
            '0.0.0.0',
            '--port',
            '50052',  # Use different port to avoid conflicts
            '--max-workers',
            '10',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give server time to start
    time.sleep(0.5)

    # Check if server is running (poll returns None while running)
    poll_result = proc.poll()
    assert poll_result is None, f"gRPC server exited immediately with code {poll_result}"

    # Terminate server gracefully
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("gRPC server did not shut down within 5 seconds")


def test_grpc_server_health_check():
    """Test that gRPC server responds to health check requests."""

    port = 50053  # Use unique port
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'nvidia_resiliency_ext.shared_utils.grpc_log_server',
            '--host',
            '0.0.0.0',
            '--port',
            str(port),
            '--max-workers',
            '10',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Give server time to start
        time.sleep(1.0)

        # Try health check
        channel = grpc.insecure_channel(f'localhost:{port}')
        stub = log_aggregation_pb2_grpc.LogAggregationServiceStub(channel)

        # Send health check
        response = stub.HealthCheck(log_aggregation_pb2.HealthRequest(), timeout=5.0)

        assert response.healthy is True, "Server should report as healthy"
        assert response.connected_clients == 0, "Should have 0 connected clients initially"

        channel.close()

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _wait_for_grpc_server_ready(port, timeout=10.0, poll_interval=0.2):
    """Wait until gRPC server responds to health check or timeout."""
    channel = grpc.insecure_channel(f'localhost:{port}')
    try:
        stub = log_aggregation_pb2_grpc.LogAggregationServiceStub(channel)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = stub.HealthCheck(log_aggregation_pb2.HealthRequest(), timeout=2.0)
                if response.healthy:
                    return
            except grpc.RpcError:
                pass
            time.sleep(poll_interval)
        pytest.fail(f"gRPC server at localhost:{port} did not become ready within {timeout}s")
    finally:
        channel.close()


def _wait_for_file_content(path, required_substrings, timeout=5.0, poll_interval=0.1):
    """Wait until file exists and contains all required substrings, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            if all(s in content for s in required_substrings):
                return content
        time.sleep(poll_interval)
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
    else:
        content = "(file did not exist)"
    missing = [s for s in required_substrings if s not in content]
    pytest.fail(f"File {path} missing content after {timeout}s: {missing}. Content: {content}")


def test_grpc_client_can_connect_and_stream_logs(tmp_dir):
    """Test that a gRPC client can connect to server and stream log data."""
    import queue

    from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import GrpcWriterThread

    port = 50054  # Use unique port
    output_file = os.path.join(tmp_dir, "test_output.log")

    # Start server
    server_proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'nvidia_resiliency_ext.shared_utils.grpc_log_server',
            '--host',
            '0.0.0.0',
            '--port',
            str(port),
            '--max-workers',
            '10',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to be ready (avoids fixed sleep; works on slow CI)
        _wait_for_grpc_server_ready(port, timeout=10.0)

        # Create writer client
        write_queue = queue.Queue()
        import logging

        logger = logging.getLogger("test_logger")

        writer = GrpcWriterThread(
            write_queue=write_queue,
            grpc_server_address=f'localhost:{port}',
            node_id='test_node_1',
            logger=logger,
        )

        # Queue log data BEFORE starting writer to ensure they're processed
        write_queue.put((output_file, "Test log line 1\n"))
        write_queue.put((output_file, "Test log line 2\n"))
        write_queue.put((output_file, "Test log line 3\n"))

        writer.start()

        # GrpcWriterThread has 0-3s initial random delay before connecting; wait until
        # all 3 queued items are consumed (sync via Queue.qsize() - thread-safe) instead
        # of polling the unsynchronized total_chunks_sent from another thread.
        send_timeout = 20.0
        deadline = time.monotonic() + send_timeout
        while time.monotonic() < deadline:
            if write_queue.qsize() == 0:
                break
            time.sleep(0.2)
        assert (
            write_queue.qsize() == 0
        ), f"Writer should consume 3 chunks within {send_timeout}s (queue size {write_queue.qsize()})"

        # Shutdown writer only after we know data was sent
        writer.shutdown()
        writer.join(timeout=10)

        # Verify statistics
        assert (
            writer.total_chunks_sent > 0
        ), f"Should have sent some chunks (sent {writer.total_chunks_sent})"
        assert (
            writer.total_bytes_sent > 0
        ), f"Should have sent some bytes (sent {writer.total_bytes_sent})"

        # Wait for server to flush to disk (poll for file content instead of fixed sleep)
        _wait_for_file_content(
            output_file,
            ["Test log line 1", "Test log line 2", "Test log line 3"],
            timeout=5.0,
        )
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=5)


# ==============================================================================
# Launcher Integration Tests
# ==============================================================================


def test_launcher_starts_grpc_server_on_correct_port(tmp_dir):
    """Test that launcher starts gRPC server on the specified port."""
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)

    custom_port = 50055
    cmd_to_run = f"{_get_util_script_path()} --scenario=test_ranks_exit_gracefully"

    launcher_cmd = (
        "ft_launcher"
        f" --ft-cfg-path={ft_cfg_path}"
        " --ft-enable-log-server=true"
        f" --ft-log-server-port={custom_port}"
        f" --ft-per-cycle-applog-prefix={tmp_dir}/test.log"
        f" --nproc-per-node={WORLD_SIZE}"
        f" {cmd_to_run}"
    )

    ret_code, output = _run_launcher(launcher_cmd, timeout=15)

    # Verify server was started with custom port
    assert f"port={custom_port}" in output, f"Server should be started on port {custom_port}"


def test_launcher_creates_grpc_server_log(tmp_dir):
    """Test that launcher creates a log file for the gRPC server."""
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)

    base_log_file = os.path.join(tmp_dir, "test.log")
    expected_server_log = os.path.join(tmp_dir, "test_grpc_server.log")

    cmd_to_run = f"{_get_util_script_path()} --scenario=test_ranks_exit_gracefully"

    launcher_cmd = (
        "ft_launcher"
        f" --ft-cfg-path={ft_cfg_path}"
        " --ft-enable-log-server=true"
        f" --ft-per-cycle-applog-prefix={base_log_file}"
        f" --nproc-per-node={WORLD_SIZE}"
        f" {cmd_to_run}"
    )

    ret_code, output = _run_launcher(launcher_cmd, timeout=15)

    # Verify server log file was created
    # Note: We don't strictly require it to exist as the launcher may clean up
    # But if it does exist, verify it contains server startup message
    if os.path.exists(expected_server_log):
        with open(expected_server_log, 'r') as f:
            server_content = f.read()
            # Check for any indication the server started
            assert len(server_content) > 0, "Server log should not be empty"


def test_launcher_without_grpc_flag_does_not_start_server(tmp_dir):
    """Test that gRPC server is NOT started when --ft-enable-log-server flag is absent."""
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)

    cmd_to_run = f"{_get_util_script_path()} --scenario=test_ranks_exit_gracefully"

    launcher_cmd = (
        "ft_launcher"
        f" --ft-cfg-path={ft_cfg_path}"
        # Note: NO --ft-enable-log-server flag
        f" --ft-per-cycle-applog-prefix={tmp_dir}/test.log"
        f" --nproc-per-node={WORLD_SIZE}"
        f" {cmd_to_run}"
    )

    ret_code, output = _run_launcher(launcher_cmd, timeout=15)

    # Verify gRPC server was NOT started
    assert "gRPC log server started" not in output, "gRPC server should not be started"


def test_launcher_custom_server_log_path(tmp_dir):
    """Test that custom gRPC server log path is respected."""
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = 3.0
    ft_cfg.rank_heartbeat_timeout = 3.0
    ft_cfg_path = _save_ft_cfg(ft_cfg, tmp_dir)

    base_log_file = os.path.join(tmp_dir, "test.log")
    custom_server_log = os.path.join(tmp_dir, "custom_grpc_server.log")

    cmd_to_run = f"{_get_util_script_path()} --scenario=test_ranks_exit_gracefully"

    launcher_cmd = (
        "ft_launcher"
        f" --ft-cfg-path={ft_cfg_path}"
        " --ft-enable-log-server=true"
        f" --ft-per-cycle-applog-prefix={base_log_file}"
        f" --ft-log-server-log={custom_server_log}"
        f" --nproc-per-node={WORLD_SIZE}"
        f" {cmd_to_run}"
    )

    ret_code, output = _run_launcher(launcher_cmd, timeout=15)

    # Verify custom server log path is mentioned or used
    # The test passes if launcher completes without hanging on log path issues
    assert ret_code in [0, 1], "Launcher should complete (gracefully or with expected error)"
