# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GrpcWriterThread client functionality."""

import logging
import queue
import time
from unittest.mock import MagicMock, patch

import grpc
import pytest

from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import GrpcWriterThread


class TestGrpcWriterThread:
    """Tests for GrpcWriterThread client."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def write_queue(self):
        """Create a write queue for testing."""
        return queue.Queue()

    def test_initialization(self, write_queue, mock_logger):
        """Test that GrpcWriterThread initializes correctly."""

        writer = GrpcWriterThread(
            write_queue=write_queue,
            grpc_server_address="localhost:50051",
            node_id="test_node_0",
            logger=mock_logger,
        )

        assert writer.write_queue is write_queue
        assert writer.grpc_server_address == "localhost:50051"
        assert writer.node_id == "test_node_0"
        assert writer.logger is mock_logger
        assert writer.shutdown_requested is False
        assert writer.total_bytes_sent == 0
        assert writer.total_chunks_sent == 0
        assert writer.connection_errors == 0

    def test_node_id_conversion_to_string(self, write_queue, mock_logger):
        """Test that node_id is converted to string (supports int ranks)."""

        writer = GrpcWriterThread(
            write_queue=write_queue,
            grpc_server_address="localhost:50051",
            node_id=42,  # Pass as int
            logger=mock_logger,
        )

        assert writer.node_id == "42"  # Should be converted to string

    def test_chunk_creation_with_correct_metadata(self, write_queue, mock_logger):
        """Test that LogChunks are created with correct node_id and file_path."""

        test_node_id = "test_node_42"
        test_file_path = "/tmp/test.log"
        test_data = "Test log line\n"

        # Add data to queue
        write_queue.put((test_file_path, test_data))

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            chunks_sent = []

            def capture_chunks(chunk_iterator):
                """Capture chunks for verification."""
                for chunk in chunk_iterator:
                    chunks_sent.append(chunk)
                    # After receiving one chunk, signal writer to shutdown
                    writer.shutdown_requested = True
                response = MagicMock()
                response.status = "ok"
                response.bytes_received = len(test_data)
                return response

            mock_stub.StreamLogs.side_effect = capture_chunks
            mock_stub.HealthCheck.return_value = MagicMock(healthy=True, connected_clients=0)

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id=test_node_id,
                    logger=mock_logger,
                )

                # Reduce initial delay for testing
                with patch('time.sleep'):
                    writer.run()

                # Verify chunk metadata
                assert len(chunks_sent) == 1
                chunk = chunks_sent[0]
                assert chunk.node_id == test_node_id
                assert chunk.file_path == test_file_path
                assert chunk.data == test_data.encode('utf-8')

    def test_utf8_encoding(self, write_queue, mock_logger):
        """Test that data is properly UTF-8 encoded."""

        # Test with emoji and Chinese characters
        test_data = "Test 🚀 emoji and 测试中文\n"
        write_queue.put(("/tmp/test.log", test_data))

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            chunks_sent = []

            def capture_chunks(chunk_iterator):
                for chunk in chunk_iterator:
                    chunks_sent.append(chunk)
                    writer.shutdown_requested = True
                return MagicMock(status="ok", bytes_received=len(test_data))

            mock_stub.StreamLogs.side_effect = capture_chunks
            mock_stub.HealthCheck.return_value = MagicMock(healthy=True, connected_clients=0)

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                with patch('time.sleep'):
                    writer.run()

                # Verify UTF-8 encoding
                assert len(chunks_sent) == 1
                chunk = chunks_sent[0]
                # Should decode cleanly without replacement characters
                decoded = chunk.data.decode('utf-8')
                assert decoded == test_data
                assert '\ufffd' not in decoded  # No replacement characters

    def test_health_check_retry_on_unavailable(self, write_queue, mock_logger):
        """Test that client retries health check when server unavailable."""

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()

            # Simulate server unavailable for first 2 attempts, then success
            attempts = [0]

            def health_check_side_effect(*args, **kwargs):
                attempts[0] += 1
                if attempts[0] <= 2:
                    raise grpc.RpcError("Server unavailable")
                # Third attempt succeeds
                response = MagicMock()
                response.healthy = True
                response.connected_clients = 0
                return response

            mock_stub.HealthCheck.side_effect = health_check_side_effect

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                # Mock sleep to speed up test
                with patch('time.sleep'):
                    channel = writer._wait_for_server_ready()

                # Should succeed after retries
                assert channel is not None
                assert attempts[0] == 3  # 2 failures + 1 success

    def test_health_check_respects_shutdown(self, write_queue, mock_logger):
        """Test that health check loop exits when shutdown requested."""

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            # Always fail health check
            mock_stub.HealthCheck.side_effect = grpc.RpcError("Server unavailable")

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                # Request shutdown after first attempt
                attempts = [0]
                original_sleep = time.sleep

                def sleep_with_shutdown(duration):
                    attempts[0] += 1
                    if attempts[0] == 1:
                        writer.shutdown_requested = True
                    original_sleep(0.001)  # Small delay for test

                with patch('time.sleep', side_effect=sleep_with_shutdown):
                    channel = writer._wait_for_server_ready()

                # Should return None due to shutdown
                assert channel is None

    def test_multiple_chunks_sent(self, write_queue, mock_logger):
        """Test that multiple log entries are sent as separate chunks."""

        # Add multiple log entries
        logs = [
            ("/tmp/test.log", "Line 1\n"),
            ("/tmp/test.log", "Line 2\n"),
            ("/tmp/test.log", "Line 3\n"),
        ]
        for file_path, data in logs:
            write_queue.put((file_path, data))

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            chunks_sent = []

            def capture_chunks(chunk_iterator):
                for chunk in chunk_iterator:
                    chunks_sent.append(chunk)
                    if len(chunks_sent) == 3:  # After 3 chunks, shutdown
                        writer.shutdown_requested = True
                return MagicMock(status="ok", bytes_received=100)

            mock_stub.StreamLogs.side_effect = capture_chunks
            mock_stub.HealthCheck.return_value = MagicMock(healthy=True, connected_clients=0)

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                with patch('time.sleep'):
                    writer.run()

                # Verify all chunks were sent
                assert len(chunks_sent) == 3
                for i, (expected_path, expected_data) in enumerate(logs):
                    assert chunks_sent[i].file_path == expected_path
                    assert chunks_sent[i].data == expected_data.encode('utf-8')

    def test_statistics_tracking(self, write_queue, mock_logger):
        """Test that bytes and chunks statistics are tracked correctly."""

        test_data = "A" * 100 + "\n"  # 101 bytes
        write_queue.put(("/tmp/test.log", test_data))
        write_queue.put(("/tmp/test.log", test_data))

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            chunks_sent = [0]

            def capture_chunks(chunk_iterator):
                for chunk in chunk_iterator:
                    chunks_sent[0] += 1
                    if chunks_sent[0] == 2:
                        writer.shutdown_requested = True
                return MagicMock(status="ok", bytes_received=202)

            mock_stub.StreamLogs.side_effect = capture_chunks
            mock_stub.HealthCheck.return_value = MagicMock(healthy=True, connected_clients=0)

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                with patch('time.sleep'):
                    writer.run()

                # Verify statistics
                assert writer.total_chunks_sent == 2
                assert writer.total_bytes_sent == 202  # 101 * 2

    def test_shutdown_method(self, write_queue, mock_logger):
        """Test that shutdown() sets the shutdown flag."""

        writer = GrpcWriterThread(
            write_queue=write_queue,
            grpc_server_address="localhost:50051",
            node_id="test_node",
            logger=mock_logger,
        )

        assert writer.shutdown_requested is False
        writer.shutdown()
        assert writer.shutdown_requested is True

        # Verify shutdown is logged
        assert mock_logger.info.called

    def test_channel_reuse_from_health_check(self, write_queue, mock_logger):
        """Test that channel from health check is reused for streaming."""

        write_queue.put(("/tmp/test.log", "test\n"))

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            mock_stub.HealthCheck.return_value = MagicMock(healthy=True, connected_clients=0)

            def capture_chunks(chunk_iterator):
                for chunk in chunk_iterator:
                    writer.shutdown_requested = True
                return MagicMock(status="ok", bytes_received=5)

            mock_stub.StreamLogs.side_effect = capture_chunks

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                with patch('time.sleep'):
                    writer.run()

                # Verify channel was created only once (reused)
                assert mock_channel_fn.call_count == 1
                # Verify channel was closed after use
                assert mock_channel.close.called

    def test_connection_error_handling(self, write_queue, mock_logger):
        """Test that connection errors are handled and logged."""

        write_queue.put(("/tmp/test.log", "test\n"))

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            mock_stub.HealthCheck.return_value = MagicMock(healthy=True, connected_clients=0)

            # First stream attempt fails, second succeeds
            attempts = [0]

            def stream_logs_effect(chunk_iterator):
                attempts[0] += 1
                if attempts[0] == 1:
                    raise grpc.RpcError("Connection lost")
                # Second attempt succeeds
                for chunk in chunk_iterator:
                    writer.shutdown_requested = True
                return MagicMock(status="ok", bytes_received=5)

            mock_stub.StreamLogs.side_effect = stream_logs_effect

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                with patch('time.sleep'):
                    writer.run()

                # Verify error was logged
                assert writer.connection_errors == 1
                # Verify warning was logged
                assert any(
                    'connection lost' in str(call).lower()
                    for call in mock_logger.warning.call_args_list
                )

    def test_shutdown_exits_without_draining_queue(self, write_queue, mock_logger):
        """Test that shutdown exits immediately without draining remaining queue items.

        The gRPC writer does not drain on shutdown because:
        1. During shutdown, the server may already be closing/closed
        2. Draining adds delay that can exceed the reader's shutdown timeout
        3. Any logs still in queue at shutdown are not guaranteed to be delivered anyway
        """

        # Add items that will be in queue during shutdown
        write_queue.put(("/tmp/test.log", "Line 1\n"))
        write_queue.put(("/tmp/test.log", "Line 2\n"))
        write_queue.put(("/tmp/test.log", "Line 3\n"))

        with patch('grpc.insecure_channel') as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            mock_stub.HealthCheck.return_value = MagicMock(healthy=True, connected_clients=0)

            chunks_received = []

            def capture_chunks(chunk_iterator):
                """Capture chunks - should exit immediately when shutdown is set."""
                # Immediately request shutdown
                writer.shutdown_requested = True

                # Try to consume - generator should exit immediately
                for chunk in chunk_iterator:
                    chunks_received.append(chunk)

                return MagicMock(
                    status="ok",
                    bytes_received=(
                        sum(len(c.data) for c in chunks_received) if chunks_received else 0
                    ),
                )

            mock_stub.StreamLogs.side_effect = capture_chunks

            with patch(
                'nvidia_resiliency_ext.shared_utils.proto.log_aggregation_pb2_grpc.LogAggregationServiceStub',
                return_value=mock_stub,
            ):

                writer = GrpcWriterThread(
                    write_queue=write_queue,
                    grpc_server_address="localhost:50051",
                    node_id="test_node",
                    logger=mock_logger,
                )

                with patch('time.sleep'):
                    writer.run()

                # Generator should exit immediately on shutdown, not drain queue
                # Queue items remain unprocessed
                assert (
                    len(chunks_received) == 0
                ), f"Expected 0 chunks (no drain), got {len(chunks_received)}"
                assert write_queue.qsize() == 3, "Queue items should remain"
