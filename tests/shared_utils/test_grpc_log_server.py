# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gRPC log aggregation server."""

import os
import tempfile
import time
from unittest.mock import MagicMock

import grpc
import pytest

from nvidia_resiliency_ext.shared_utils.grpc_log_server import LogAggregationServicer
from nvidia_resiliency_ext.shared_utils.proto import log_aggregation_pb2


class TestLogAggregationServicer:
    """Tests for LogAggregationServicer server."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def servicer(self):
        """Create a LogAggregationServicer instance."""
        return LogAggregationServicer(max_buffer_size=1024 * 1024)  # 1MB

    def test_multi_file_support(self, servicer, temp_log_dir):
        """Test that server writes to different files based on file_path."""
        file1 = os.path.join(temp_log_dir, "file1.log")
        file2 = os.path.join(temp_log_dir, "file2.log")

        # Create mock chunks for different files
        chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Data for file1\n", file_path=file1
            ),
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"More data for file1\n", file_path=file1
            ),
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Data for file2\n", file_path=file2
            ),
        ]

        # Mock the gRPC context
        mock_context = MagicMock()

        # Stream the chunks
        response = servicer.StreamLogs(iter(chunks), mock_context)

        # Force flush
        servicer.shutdown()

        # Verify file1 contains its data
        with open(file1, 'rb') as f:
            content1 = f.read()
        assert b"Data for file1\n" in content1
        assert b"More data for file1\n" in content1
        assert b"file2" not in content1

        # Verify file2 contains its data
        with open(file2, 'rb') as f:
            content2 = f.read()
        assert b"Data for file2\n" in content2
        assert b"file1" not in content2

    def test_buffering_and_flush(self, servicer, temp_log_dir):
        """Test that data is buffered and flushed at max_buffer_size."""
        log_file = os.path.join(temp_log_dir, "buffered.log")

        # Create servicer with small buffer for testing
        servicer_small_buffer = LogAggregationServicer(
            max_buffer_size=100  # Small buffer (periodic flush will handle flushing)
        )

        # Create chunks that exceed buffer size
        chunks = []
        for i in range(5):
            chunks.append(
                log_aggregation_pb2.LogChunk(
                    node_id="node_1",
                    data=f"Line {i}: {'X' * 30}\n".encode('utf-8'),  # ~40 bytes each
                    file_path=log_file,
                )
            )

        mock_context = MagicMock()

        # Stream chunks
        response = servicer_small_buffer.StreamLogs(iter(chunks), mock_context)

        # Shutdown to flush
        servicer_small_buffer.shutdown()

        # Verify all data was written
        with open(log_file, 'r') as f:
            content = f.read()
        for i in range(5):
            assert f"Line {i}:" in content

    def test_flush_on_disconnect(self, servicer, temp_log_dir):
        """Test that buffer is eventually flushed (by periodic flush or shutdown)."""
        log_file = os.path.join(temp_log_dir, "disconnect.log")

        chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Log before disconnect\n", file_path=log_file
            ),
        ]

        mock_context = MagicMock()

        # Stream and complete (simulates disconnect)
        response = servicer.StreamLogs(iter(chunks), mock_context)

        # Wait for periodic flush (happens every 1s)
        time.sleep(1.5)

        # Verify data was flushed by periodic flush
        servicer.shutdown()
        with open(log_file, 'r') as f:
            content = f.read()
        assert "Log before disconnect\n" in content

    def test_multiple_concurrent_clients(self, servicer, temp_log_dir):
        """Test server handling multiple clients writing to same file."""
        log_file = os.path.join(temp_log_dir, "concurrent.log")

        # Simulate 3 clients
        client1_chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Client 1 log\n", file_path=log_file
            ),
        ]

        client2_chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_2", data=b"Client 2 log\n", file_path=log_file
            ),
        ]

        client3_chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_3", data=b"Client 3 log\n", file_path=log_file
            ),
        ]

        mock_context = MagicMock()

        # Stream from all clients
        servicer.StreamLogs(iter(client1_chunks), mock_context)
        servicer.StreamLogs(iter(client2_chunks), mock_context)
        servicer.StreamLogs(iter(client3_chunks), mock_context)

        # Shutdown and flush
        servicer.shutdown()

        # Verify all clients' logs are present
        with open(log_file, 'r') as f:
            content = f.read()
        assert "Client 1 log\n" in content
        assert "Client 2 log\n" in content
        assert "Client 3 log\n" in content

    def test_invalid_file_path_raises_value_error(self, servicer):
        """Test that empty file_path raises ValueError."""
        chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Data\n", file_path=""  # Empty file_path
            ),
        ]

        mock_context = MagicMock()

        # StreamLogs should call context.abort with INVALID_ARGUMENT
        servicer.StreamLogs(iter(chunks), mock_context)

        # Verify context.abort was called with INVALID_ARGUMENT
        assert mock_context.abort.called, "context.abort should have been called"
        args = mock_context.abort.call_args
        assert (
            args[0][0] == grpc.StatusCode.INVALID_ARGUMENT
        ), f"Expected INVALID_ARGUMENT, got {args[0][0]}"

    def test_health_check_endpoint(self, servicer):
        """Test the HealthCheck RPC endpoint."""
        mock_context = MagicMock()

        # Call HealthCheck
        response = servicer.HealthCheck(log_aggregation_pb2.HealthRequest(), mock_context)

        # Should return healthy
        assert response.healthy is True

    def test_graceful_shutdown_flushes_all_buffers(self, servicer, temp_log_dir):
        """Test that shutdown flushes all pending buffers."""
        file1 = os.path.join(temp_log_dir, "shutdown1.log")
        file2 = os.path.join(temp_log_dir, "shutdown2.log")

        # Write to multiple files without exceeding buffer
        chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Pending data file1\n", file_path=file1
            ),
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Pending data file2\n", file_path=file2
            ),
        ]

        mock_context = MagicMock()
        servicer.StreamLogs(iter(chunks), mock_context)

        # Data may still be in buffer, not flushed yet
        # Now shutdown - should flush everything
        servicer.shutdown()

        # Verify both files have data
        with open(file1, 'r') as f:
            assert "Pending data file1\n" in f.read()
        with open(file2, 'r') as f:
            assert "Pending data file2\n" in f.read()

    def test_large_data_stream(self, servicer, temp_log_dir):
        """Test handling of large data stream."""
        log_file = os.path.join(temp_log_dir, "large.log")

        # Stream 1000 chunks
        def chunk_generator():
            for i in range(1000):
                yield log_aggregation_pb2.LogChunk(
                    node_id="node_1", data=f"Line {i}\n".encode('utf-8'), file_path=log_file
                )

        mock_context = MagicMock()
        response = servicer.StreamLogs(chunk_generator(), mock_context)

        # Shutdown to flush
        servicer.shutdown()

        # Verify all lines present
        with open(log_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1000
        assert "Line 0\n" in lines
        assert "Line 999\n" in lines

    def test_binary_data_preserved(self, servicer, temp_log_dir):
        """Test that binary data is written correctly without decoding."""
        log_file = os.path.join(temp_log_dir, "binary.log")

        # Binary data (not necessarily valid UTF-8)
        binary_data = b'\x00\x01\x02\xff\xfe\n'

        chunks = [
            log_aggregation_pb2.LogChunk(node_id="node_1", data=binary_data, file_path=log_file),
        ]

        mock_context = MagicMock()
        servicer.StreamLogs(iter(chunks), mock_context)
        servicer.shutdown()

        # Read as binary and verify
        with open(log_file, 'rb') as f:
            content = f.read()
        assert content == binary_data

    def test_automatic_directory_creation(self, temp_log_dir):
        """Test that server creates directories if they don't exist."""
        # Use nested directory that doesn't exist
        nested_file = os.path.join(temp_log_dir, "subdir", "nested", "file.log")

        servicer = LogAggregationServicer()

        chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Data in nested dir\n", file_path=nested_file
            ),
        ]

        mock_context = MagicMock()
        servicer.StreamLogs(iter(chunks), mock_context)
        servicer.shutdown()

        # Verify file was created
        assert os.path.exists(nested_file)
        with open(nested_file, 'r') as f:
            assert "Data in nested dir\n" in f.read()


class TestPeriodicFlush:
    """Tests for periodic flush functionality."""

    def test_periodic_flush_timer(self, tmp_path):
        """Test that periodic flush happens on schedule."""
        log_file = tmp_path / "periodic.log"

        # Create servicer with short flush interval
        servicer = LogAggregationServicer(max_buffer_size=1024 * 1024)  # Large buffer

        # Manually set flush interval to 0.5s for testing
        servicer.flush_interval = 0.5

        # Write small amount of data (won't trigger buffer flush)
        chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Small data\n", file_path=str(log_file)
            ),
        ]

        mock_context = MagicMock()
        servicer.StreamLogs(iter(chunks), mock_context)

        # Wait for periodic flush
        time.sleep(1.0)

        # Shutdown
        servicer.shutdown()

        # Data should have been flushed by periodic timer
        assert log_file.exists()
        with open(log_file, 'r') as f:
            assert "Small data\n" in f.read()


class TestGracefulShutdown:
    """Tests for graceful shutdown functionality."""

    def test_graceful_shutdown_waits_for_clients(self, tmp_path):
        """Test that graceful shutdown waits for all clients to disconnect."""
        log_file = tmp_path / "graceful.log"

        # Create servicer with short timeout and fast flush for testing
        servicer = LogAggregationServicer(graceful_shutdown_timeout=2.0, flush_interval=0.1)

        # Simulate client connection by manually incrementing counter
        with servicer.clients_lock:
            servicer.connected_clients = 1

        import threading

        # Start graceful shutdown in background thread
        shutdown_complete = threading.Event()
        shutdown_start_time = [None]
        shutdown_end_time = [None]

        def run_graceful_shutdown():
            shutdown_start_time[0] = time.time()
            servicer.graceful_shutdown()
            shutdown_end_time[0] = time.time()
            shutdown_complete.set()

        shutdown_thread = threading.Thread(target=run_graceful_shutdown)
        shutdown_thread.start()

        # Wait a bit, then disconnect the client
        time.sleep(0.5)
        with servicer.clients_lock:
            servicer.connected_clients = 0

        # Wait for shutdown to complete
        shutdown_complete.wait(timeout=5.0)
        shutdown_thread.join(timeout=5.0)

        # Verify shutdown completed before timeout (should be ~0.5s, not 2.0s)
        elapsed = shutdown_end_time[0] - shutdown_start_time[0]
        assert elapsed < 1.5, f"Shutdown took {elapsed}s, should exit early when clients disconnect"
        assert elapsed >= 0.5, f"Shutdown took {elapsed}s, should wait for clients"

    def test_graceful_shutdown_respects_timeout(self, tmp_path):
        """Test that graceful shutdown respects timeout when clients don't disconnect."""

        # Create servicer with very short timeout and fast flush for testing
        servicer = LogAggregationServicer(graceful_shutdown_timeout=1.0, flush_interval=0.1)

        # Simulate client that doesn't disconnect
        with servicer.clients_lock:
            servicer.connected_clients = 2

        # Run graceful shutdown
        start_time = time.time()
        servicer.graceful_shutdown()
        elapsed = time.time() - start_time

        # Should respect timeout (allow some margin for execution time)
        assert 0.9 <= elapsed <= 1.5, f"Shutdown took {elapsed}s, expected ~1.0s"

    def test_graceful_shutdown_with_no_clients(self, tmp_path):
        """Test that graceful shutdown exits immediately when no clients connected."""

        servicer = LogAggregationServicer(graceful_shutdown_timeout=10.0)  # Long timeout

        # No clients connected (default state)
        assert servicer.connected_clients == 0

        # Run graceful shutdown
        start_time = time.time()
        servicer.graceful_shutdown()
        elapsed = time.time() - start_time

        # Should exit immediately (within 0.5s for checking)
        assert elapsed < 1.0, f"Shutdown took {elapsed}s with no clients, should be immediate"

    def test_graceful_shutdown_flushes_data(self, tmp_path):
        """Test that graceful shutdown properly flushes all buffered data."""
        log_file = tmp_path / "graceful_flush.log"

        servicer = LogAggregationServicer(graceful_shutdown_timeout=1.0)

        # Write data
        chunks = [
            log_aggregation_pb2.LogChunk(
                node_id="node_1", data=b"Data before graceful shutdown\n", file_path=str(log_file)
            ),
        ]

        mock_context = MagicMock()
        servicer.StreamLogs(iter(chunks), mock_context)

        # Graceful shutdown should flush
        servicer.graceful_shutdown()

        # Verify data was written
        assert log_file.exists()
        with open(log_file, 'r') as f:
            content = f.read()
        assert "Data before graceful shutdown\n" in content
