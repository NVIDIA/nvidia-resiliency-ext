# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for grpc_log_leaf_server.py and two-level log aggregation."""

import threading
import time
from concurrent import futures
from unittest.mock import MagicMock

import grpc
import pytest

from nvidia_resiliency_ext.shared_utils.grpc_log_leaf_server import (
    _STOP,
    LeafLogServicer,
    _LeafChunkQueue,
    _UpstreamForwarder,
)
from nvidia_resiliency_ext.shared_utils.grpc_log_server import LogAggregationServicer
from nvidia_resiliency_ext.shared_utils.proto import log_aggregation_pb2, log_aggregation_pb2_grpc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(data: bytes, file_path: str, node_id: str = "node_0") -> log_aggregation_pb2.LogChunk:
    return log_aggregation_pb2.LogChunk(node_id=node_id, data=data, file_path=file_path)


def _start_root_server(servicer, host: str = "127.0.0.1"):
    """Bind a gRPC root server on an ephemeral port; return (server, port)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    log_aggregation_pb2_grpc.add_LogAggregationServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port(f"{host}:0")
    server.start()
    return server, port


def _start_leaf_stack(upstream_addr: str, host: str = "127.0.0.1"):
    """Start a full leaf stack (queue + forwarder + servicer + gRPC server).

    Returns (server, leaf_port, forwarder, servicer, chunk_q, stop_ev, reject_ev).
    """
    chunk_q = _LeafChunkQueue(max_chunks=256)
    stop_ev = threading.Event()
    reject_ev = threading.Event()
    forwarder = _UpstreamForwarder(
        upstream=upstream_addr,
        chunk_queue=chunk_q,
        stop_event=stop_ev,
        reconnect_sleep=0.1,
    )
    forwarder.start()
    servicer = LeafLogServicer(chunk_q, reject_ev, forwarder.upstream_ready)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    log_aggregation_pb2_grpc.add_LogAggregationServiceServicer_to_server(servicer, server)
    leaf_port = server.add_insecure_port(f"{host}:0")
    server.start()
    return server, leaf_port, forwarder, servicer, chunk_q, stop_ev, reject_ev


def _stop_leaf_stack(server, forwarder, chunk_q, stop_ev):
    """Gracefully drain the leaf queue then stop the gRPC server."""
    stop_ev.set()
    try:
        chunk_q.put_stop(timeout=2.0)
    except Exception:
        pass
    forwarder.join(timeout=5.0)
    server.stop(grace=1.0)
    server.wait_for_termination(timeout=5.0)


def _stream_to_addr(addr: str, chunks):
    """Open a client channel to addr and stream chunks."""
    ch = grpc.insecure_channel(addr)
    stub = log_aggregation_pb2_grpc.LogAggregationServiceStub(ch)
    stub.StreamLogs(iter(chunks))
    ch.close()


# ---------------------------------------------------------------------------
# _LeafChunkQueue unit tests
# ---------------------------------------------------------------------------


class TestLeafChunkQueue:
    def test_put_and_get(self):
        q = _LeafChunkQueue(max_chunks=10)
        reject = threading.Event()
        ok = q.put_chunk(_chunk(b"hello", "/tmp/x.log"), reject, peer="p", source_node_id="n")
        assert ok
        item = q.get_upstream(timeout=1.0)
        assert item.data == b"hello"

    def test_maxsize_property(self):
        q = _LeafChunkQueue(max_chunks=42)
        assert q.maxsize == 42

    def test_qsize_reflects_contents(self):
        q = _LeafChunkQueue(max_chunks=10)
        reject = threading.Event()
        assert q.qsize() == 0
        q.put_chunk(_chunk(b"a", "/tmp/x.log"), reject, peer="p", source_node_id="n")
        assert q.qsize() == 1

    def test_put_returns_false_when_rejected(self):
        q = _LeafChunkQueue(max_chunks=1)
        reject = threading.Event()
        # Fill queue
        q.put_chunk(_chunk(b"fill", "/tmp/x.log"), reject, peer="p", source_node_id="n")
        # Now queue is full; a second put with reject already set must return False immediately
        reject.set()
        ok = q.put_chunk(_chunk(b"blocked", "/tmp/x.log"), reject, peer="p", source_node_id="n")
        assert not ok

    def test_put_stop_sentinel(self):
        q = _LeafChunkQueue(max_chunks=10)
        q.put_stop(timeout=1.0)
        item = q.get_upstream(timeout=1.0)
        assert item is _STOP

    def test_invalid_max_chunks_raises(self):
        with pytest.raises(ValueError):
            _LeafChunkQueue(max_chunks=0)


# ---------------------------------------------------------------------------
# LeafLogServicer unit tests  (no network – call RPC methods directly)
# ---------------------------------------------------------------------------


class TestLeafLogServicer:
    def _make(self, upstream_set: bool = False, rejected: bool = False):
        chunk_q = _LeafChunkQueue(max_chunks=64)
        reject = threading.Event()
        upstream_ready = threading.Event()
        if upstream_set:
            upstream_ready.set()
        if rejected:
            reject.set()
        return LeafLogServicer(chunk_q, reject, upstream_ready), chunk_q, reject, upstream_ready

    # --- HealthCheck ---------------------------------------------------------

    def test_health_unhealthy_when_upstream_not_ready(self):
        svc, *_ = self._make(upstream_set=False)
        resp = svc.HealthCheck(log_aggregation_pb2.HealthRequest(), MagicMock())
        assert resp.healthy is False

    def test_health_healthy_when_upstream_ready(self):
        svc, *_ = self._make(upstream_set=True)
        resp = svc.HealthCheck(log_aggregation_pb2.HealthRequest(), MagicMock())
        assert resp.healthy is True

    def test_health_reports_connected_clients(self):
        svc, *_ = self._make(upstream_set=True)
        with svc.clients_lock:
            svc.connected_clients = 5
        resp = svc.HealthCheck(log_aggregation_pb2.HealthRequest(), MagicMock())
        assert resp.connected_clients == 5

    # --- StreamLogs ----------------------------------------------------------

    def test_stream_rejected_during_shutdown(self):
        svc, *_ = self._make(rejected=True)
        ctx = MagicMock()
        svc.StreamLogs(iter([_chunk(b"data", "/tmp/x.log")]), ctx)
        ctx.abort.assert_called_once()
        code, _msg = ctx.abort.call_args[0]
        assert code == grpc.StatusCode.UNAVAILABLE

    def test_stream_enqueues_chunks(self):
        svc, chunk_q, *_ = self._make()
        chunks = [_chunk(b"line1\n", "/tmp/x.log"), _chunk(b"line2\n", "/tmp/x.log")]
        svc.StreamLogs(iter(chunks), MagicMock())
        assert chunk_q.qsize() == 2

    def test_stream_chunks_are_copies(self):
        svc, chunk_q, *_ = self._make()
        original = _chunk(b"payload", "/tmp/x.log")
        svc.StreamLogs(iter([original]), MagicMock())
        queued = chunk_q.get_upstream(timeout=1.0)
        assert queued is not original
        assert queued.data == b"payload"

    def test_stream_client_count_zero_after_completion(self):
        svc, *_ = self._make()
        assert svc.connected_clients == 0
        svc.StreamLogs(iter([_chunk(b"x", "/tmp/x.log")]), MagicMock())
        assert svc.connected_clients == 0

    def test_stream_stops_mid_stream_on_reject(self):
        svc, chunk_q, reject, _ = self._make()

        def gen():
            yield _chunk(b"first\n", "/tmp/x.log")
            reject.set()
            yield _chunk(b"second\n", "/tmp/x.log")

        svc.StreamLogs(gen(), MagicMock())
        # Only the first chunk is queued; second arrives after reject is set
        assert chunk_q.qsize() == 1

    def test_stream_returns_ok_response(self):
        svc, *_ = self._make()
        resp = svc.StreamLogs(iter([_chunk(b"data\n", "/tmp/x.log")]), MagicMock())
        assert resp.status == "OK"
        assert resp.bytes_received == len(b"data\n")


# ---------------------------------------------------------------------------
# Two-level integration tests  (real gRPC servers, client → leaf → root)
# ---------------------------------------------------------------------------


class TestTwoLevelLogAggregation:
    """End-to-end: client streams to leaf, leaf forwards to root, root writes file."""

    def test_chunks_reach_root_file(self, tmp_path):
        log_file = str(tmp_path / "output.log")
        root_svc = LogAggregationServicer(flush_interval=0.1)
        root_server, root_port = _start_root_server(root_svc)

        server, leaf_port, forwarder, _, chunk_q, stop_ev, _ = _start_leaf_stack(
            f"127.0.0.1:{root_port}"
        )
        try:
            assert forwarder.upstream_ready.wait(timeout=5.0), "leaf did not connect to root"

            _stream_to_addr(
                f"127.0.0.1:{leaf_port}",
                [_chunk(b"hello from leaf\n", log_file), _chunk(b"world\n", log_file)],
            )
            time.sleep(1.5)  # allow forwarder to drain + root flush_interval
            root_svc.shutdown()

            content = (tmp_path / "output.log").read_bytes()
            assert b"hello from leaf\n" in content
            assert b"world\n" in content
        finally:
            _stop_leaf_stack(server, forwarder, chunk_q, stop_ev)
            root_server.stop(grace=1.0)
            root_server.wait_for_termination(timeout=5.0)

    def test_leaf_health_unhealthy_before_root_ready(self):
        """Leaf HealthCheck returns unhealthy until upstream root accepts connections."""
        chunk_q = _LeafChunkQueue(max_chunks=16)
        stop_ev = threading.Event()
        forwarder = _UpstreamForwarder(
            upstream="127.0.0.1:19871",  # nothing listening here
            chunk_queue=chunk_q,
            stop_event=stop_ev,
            reconnect_sleep=0.1,
        )
        forwarder.start()
        svc = LeafLogServicer(chunk_q, threading.Event(), forwarder.upstream_ready)

        resp = svc.HealthCheck(log_aggregation_pb2.HealthRequest(), MagicMock())
        assert resp.healthy is False

        stop_ev.set()
        forwarder.join(timeout=3.0)

    def test_leaf_health_becomes_healthy_after_root_starts(self):
        """Leaf HealthCheck flips to healthy once upstream root is reachable."""
        root_svc = LogAggregationServicer(flush_interval=0.1)
        root_server, root_port = _start_root_server(root_svc)

        server, leaf_port, forwarder, svc, chunk_q, stop_ev, _ = _start_leaf_stack(
            f"127.0.0.1:{root_port}"
        )
        try:
            assert forwarder.upstream_ready.wait(timeout=5.0)
            resp = svc.HealthCheck(log_aggregation_pb2.HealthRequest(), MagicMock())
            assert resp.healthy is True
        finally:
            _stop_leaf_stack(server, forwarder, chunk_q, stop_ev)
            root_svc.shutdown()
            root_server.stop(grace=1.0)
            root_server.wait_for_termination(timeout=5.0)

    def test_multiple_clients_via_leaf(self, tmp_path):
        """Three concurrent clients → leaf → root; all logs reach file."""
        log_file = str(tmp_path / "multi.log")
        root_svc = LogAggregationServicer(flush_interval=0.1)
        root_server, root_port = _start_root_server(root_svc)

        server, leaf_port, forwarder, _, chunk_q, stop_ev, _ = _start_leaf_stack(
            f"127.0.0.1:{root_port}"
        )
        leaf_addr = f"127.0.0.1:{leaf_port}"
        try:
            assert forwarder.upstream_ready.wait(timeout=5.0)

            threads = []
            for i in range(3):
                t = threading.Thread(
                    target=_stream_to_addr,
                    args=(leaf_addr, [_chunk(f"client{i} log\n".encode(), log_file, f"node_{i}")]),
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=5.0)

            time.sleep(1.5)
            root_svc.shutdown()

            content = (tmp_path / "multi.log").read_bytes()
            for i in range(3):
                assert f"client{i} log\n".encode() in content
        finally:
            _stop_leaf_stack(server, forwarder, chunk_q, stop_ev)
            root_server.stop(grace=1.0)
            root_server.wait_for_termination(timeout=5.0)

    def test_leaf_drains_queue_on_shutdown(self, tmp_path):
        """Chunks buffered in leaf queue before shutdown all reach root file."""
        log_file = str(tmp_path / "drain.log")
        root_svc = LogAggregationServicer(flush_interval=0.1)
        root_server, root_port = _start_root_server(root_svc)

        server, leaf_port, forwarder, _, chunk_q, stop_ev, _ = _start_leaf_stack(
            f"127.0.0.1:{root_port}"
        )
        try:
            assert forwarder.upstream_ready.wait(timeout=5.0)

            # Directly enqueue 5 chunks (bypass leaf gRPC layer)
            reject = threading.Event()
            for i in range(5):
                chunk_q.put_chunk(
                    _chunk(f"queued_line_{i}\n".encode(), log_file),
                    reject,
                    peer="test",
                    source_node_id="n0",
                )

            # Drain leaf: stop_ev + _STOP cause forwarder gen() to exhaust
            _stop_leaf_stack(server, forwarder, chunk_q, stop_ev)

            time.sleep(1.5)  # root flush_interval
            root_svc.shutdown()

            content = (tmp_path / "drain.log").read_bytes()
            for i in range(5):
                assert f"queued_line_{i}\n".encode() in content
        finally:
            root_server.stop(grace=1.0)
            root_server.wait_for_termination(timeout=5.0)

    def test_large_volume_via_leaf(self, tmp_path):
        """500 chunks streamed through leaf all reach root file."""
        log_file = str(tmp_path / "large.log")
        root_svc = LogAggregationServicer(flush_interval=0.1)
        root_server, root_port = _start_root_server(root_svc)

        server, leaf_port, forwarder, _, chunk_q, stop_ev, _ = _start_leaf_stack(
            f"127.0.0.1:{root_port}"
        )
        try:
            assert forwarder.upstream_ready.wait(timeout=5.0)

            chunks = [_chunk(f"line {i}\n".encode(), log_file) for i in range(500)]
            _stream_to_addr(f"127.0.0.1:{leaf_port}", chunks)

            time.sleep(2.0)
            root_svc.shutdown()

            content = (tmp_path / "large.log").read_bytes()
            for i in range(500):
                assert f"line {i}\n".encode() in content
        finally:
            _stop_leaf_stack(server, forwarder, chunk_q, stop_ev)
            root_server.stop(grace=1.0)
            root_server.wait_for_termination(timeout=5.0)
