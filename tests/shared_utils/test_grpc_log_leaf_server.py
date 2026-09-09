# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for grpc_log_leaf_server.py and two-level log aggregation."""

import os
import subprocess
import sys
import tempfile
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

# These tests exercise queueing/forwarding behaviour, not path confinement (see
# test_grpc_log_server_confinement.py). Every path they write to lives under the temp dir.
_TEST_ROOTS = ["/tmp", tempfile.gettempdir()]

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
    servicer = LeafLogServicer(
        chunk_q, reject_ev, forwarder.upstream_ready, allowed_roots=_TEST_ROOTS
    )
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
        return (
            LeafLogServicer(chunk_q, reject, upstream_ready, allowed_roots=_TEST_ROOTS),
            chunk_q,
            reject,
            upstream_ready,
        )

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
        root_svc = LogAggregationServicer(flush_interval=0.1, allowed_roots=_TEST_ROOTS)
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
        svc = LeafLogServicer(
            chunk_q, threading.Event(), forwarder.upstream_ready, allowed_roots=_TEST_ROOTS
        )

        resp = svc.HealthCheck(log_aggregation_pb2.HealthRequest(), MagicMock())
        assert resp.healthy is False

        stop_ev.set()
        forwarder.join(timeout=3.0)

    def test_leaf_health_becomes_healthy_after_root_starts(self):
        """Leaf HealthCheck flips to healthy once upstream root is reachable."""
        root_svc = LogAggregationServicer(flush_interval=0.1, allowed_roots=_TEST_ROOTS)
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
        root_svc = LogAggregationServicer(flush_interval=0.1, allowed_roots=_TEST_ROOTS)
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
        root_svc = LogAggregationServicer(flush_interval=0.1, allowed_roots=_TEST_ROOTS)
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
        root_svc = LogAggregationServicer(flush_interval=0.1, allowed_roots=_TEST_ROOTS)
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


def leaf_mod_cache_max():
    import nvidia_resiliency_ext.shared_utils.grpc_log_leaf_server as leaf_mod

    return leaf_mod._LEAF_VALIDATED_PATH_CACHE_MAX


@pytest.fixture
def log_root(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    # realpath: on some platforms the pytest tmp dir itself sits behind a symlink.
    return os.path.realpath(str(root))


@pytest.fixture
def outside_dir(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    return os.path.realpath(str(outside))


class TestLeafServerConfinement:
    """The leaf must reject out-of-root chunks before they consume queue capacity."""

    def _servicer(self, allowed_roots, max_chunks=64):
        # Nothing drains the queue in these tests, so it must hold every chunk sent:
        # put_chunk blocks for backpressure once full.
        chunk_q = _LeafChunkQueue(max_chunks=max_chunks)
        upstream_ready = threading.Event()
        upstream_ready.set()
        servicer = LeafLogServicer(
            chunk_q, threading.Event(), upstream_ready, allowed_roots=allowed_roots
        )
        return servicer, chunk_q

    @pytest.mark.parametrize("roots", [None, [], [""]])
    def test_leaf_servicer_refuses_to_construct_unconfined(self, roots):
        with pytest.raises(ValueError, match="at least one directory"):
            self._servicer(roots)

    def test_leaf_servicer_requires_allowed_roots_explicitly(self):
        with pytest.raises(TypeError, match="allowed_roots"):
            LeafLogServicer(_LeafChunkQueue(max_chunks=4), threading.Event(), threading.Event())

    def test_forwards_chunk_inside_root(self, log_root):
        servicer, chunk_q = self._servicer([log_root])
        target = os.path.join(log_root, "train.log")
        resp = servicer.StreamLogs(iter([_chunk(b"hi\n", target)]), MagicMock())
        assert resp.status == "OK"
        assert chunk_q.qsize() == 1
        assert chunk_q.get_upstream(timeout=1.0).file_path == target

    def test_rejects_chunk_outside_root_without_enqueueing(self, log_root, outside_dir):
        servicer, chunk_q = self._servicer([log_root])
        context = MagicMock()
        context.abort.side_effect = grpc.RpcError("aborted")
        with pytest.raises(grpc.RpcError):
            servicer.StreamLogs(
                iter([_chunk(b"payload\n", os.path.join(outside_dir, "owned.txt"))]), context
            )
        context.abort.assert_called_once()
        assert context.abort.call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT
        assert chunk_q.qsize() == 0

    def test_rejects_empty_file_path(self, log_root):
        servicer, chunk_q = self._servicer([log_root])
        context = MagicMock()
        context.abort.side_effect = grpc.RpcError("aborted")
        with pytest.raises(grpc.RpcError):
            servicer.StreamLogs(iter([_chunk(b"payload\n", "")]), context)
        assert context.abort.call_args[0][0] == grpc.StatusCode.INVALID_ARGUMENT
        assert chunk_q.qsize() == 0

    def test_forwards_resolved_path(self, log_root):
        """Leaf hands the root an already-resolved path, not the raw client string."""
        servicer, chunk_q = self._servicer([log_root])
        raw = os.path.join(log_root, "sub", "..", "train.log")
        resp = servicer.StreamLogs(iter([_chunk(b"hi\n", raw)]), MagicMock())
        assert resp.status == "OK"
        assert chunk_q.get_upstream(timeout=1.0).file_path == os.path.join(log_root, "train.log")

    def test_cache_survives_many_restart_cycles(self, log_root, monkeypatch):
        """A long-lived stream rotates <prefix>_cycleN.log once per restart cycle.

        Far more cycles than the cache holds must not degrade the hot path: the current
        cycle log is the newest entry, so a "stop inserting when full" policy would leave
        it permanently uncached and re-resolve it on every chunk.
        """
        import nvidia_resiliency_ext.shared_utils.grpc_log_leaf_server as leaf_mod

        launcher_log = os.path.join(log_root, "nvrx.log")
        cycles = leaf_mod._LEAF_VALIDATED_PATH_CACHE_MAX * 8
        chunks_per_cycle = 20
        servicer, chunk_q = self._servicer([log_root], max_chunks=cycles * chunks_per_cycle * 2 + 1)

        calls = []
        original = leaf_mod.resolve_under_allowed_roots
        monkeypatch.setattr(
            leaf_mod,
            "resolve_under_allowed_roots",
            lambda path, roots: (calls.append(path), original(path, roots))[1],
        )

        def stream():
            for cycle in range(cycles):
                cycle_log = os.path.join(log_root, f"train_cycle{cycle}.log")
                for _ in range(chunks_per_cycle):
                    yield _chunk(b"step\n", cycle_log)
                    yield _chunk(b"agent\n", launcher_log)

        total_chunks = cycles * chunks_per_cycle * 2
        resp = servicer.StreamLogs(stream(), MagicMock())
        assert resp.status == "OK"
        assert chunk_q.qsize() == total_chunks

        # Resolves must scale with the number of distinct paths, not the number of chunks:
        # one per cycle log, plus occasional re-resolves of the launcher log as FIFO ages
        # it out. Under a "stop inserting once full" policy this would be O(total_chunks).
        distinct_paths = cycles + 1
        assert len(calls) < 2 * distinct_paths
        assert len(calls) < total_chunks / 10

    def test_cache_is_bounded(self, log_root):
        """Per-stream memory stays bounded no matter how many paths a peer cycles through."""
        servicer, _ = self._servicer([log_root])
        memo = {}
        for i in range(leaf_mod_cache_max() * 4):
            servicer._resolve_target_path(os.path.join(log_root, f"f{i}.log"), memo)
        assert len(memo) == leaf_mod_cache_max()

    def test_validation_is_memoized_per_stream(self, log_root, monkeypatch):
        servicer, chunk_q = self._servicer([log_root])
        import nvidia_resiliency_ext.shared_utils.grpc_log_leaf_server as leaf_mod

        calls = []
        original = leaf_mod.resolve_under_allowed_roots

        def counting(path, roots):
            calls.append(path)
            return original(path, roots)

        monkeypatch.setattr(leaf_mod, "resolve_under_allowed_roots", counting)
        target = os.path.join(log_root, "train.log")
        servicer.StreamLogs(iter([_chunk(b"a\n", target) for _ in range(50)]), MagicMock())
        assert len(calls) == 1
        assert chunk_q.qsize() == 50


class TestLeafCliRequiresAllowedRoots:
    def test_cli_requires_allowed_root(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "nvidia_resiliency_ext.shared_utils.grpc_log_leaf_server",
                "--port",
                "0",
                "--upstream",
                "127.0.0.1:1",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode != 0
        assert "--allowed-root" in proc.stderr
