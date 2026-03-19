#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
First-level gRPC log aggregator (leaf): accepts StreamLogs from training nodes,
forwards chunks to the root log server over a single outbound StreamLogs RPC.

Uses a bounded queue for backpressure when the root is slow. Downstream handlers
retry enqueue until space exists or shutdown (they do not discard the current
chunk because the queue is full); gRPC flow-control slows senders until the
forwarder drains toward root. That is **not** an end-to-end delivery guarantee:
if the leaf→root ``StreamLogs`` RPC fails after items were ``get`` from the
queue and yielded into gRPC but not yet committed at the root, they are not
replayed on reconnect—treat leaf→root as best-effort (see also usage docs).
Short ``put`` timeouts only exist so threads can observe ``reject_new_streams``
during shutdown. Capacity is ``max-queue-chunks`` (item count); each ``put`` is O(1).

The upstream forwarder polls the chunk queue with a timeout and exits when
``stop_forwarder`` is set, so the leaf→root ``StreamLogs`` RPC can finish even
if the ``_STOP`` sentinel never enters a full queue. Root must be started before
leaves (launcher order). The leaf does not bind its downstream TCP port until the
upstream root is healthy and ``StreamLogs`` is active, so clients may see
``ECONNREFUSED`` until then (see log line before the upstream-ready wait).
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
import urllib.parse
from concurrent import futures
from typing import Any, Iterator, Optional, Union

import grpc

try:
    from nvidia_resiliency_ext.shared_utils.proto import (
        log_aggregation_pb2,
        log_aggregation_pb2_grpc,
    )
except ImportError:
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils', 'proto'))
    import log_aggregation_pb2
    import log_aggregation_pb2_grpc

logger = logging.getLogger("LogLeafServer")

# Sentinel placed on queue after graceful drain to end upstream StreamLogs
_STOP = object()

# Slice timeout for Queue.put / Condition.wait so StreamLogs handlers can poll
# ``reject_new_streams`` and exit during shutdown (no uninterruptible block).
_LEAF_QUEUE_PUT_SLICE_S = 0.5
_STOP_SENTINEL_PUT_TIMEOUT_S = 5.0

# Throttled diagnostics when a chunk waits a long time (backpressure / slow root).
_LEAF_QUEUE_BACKPRESSURE_LOG_FIRST_S = 5.0
_LEAF_QUEUE_BACKPRESSURE_LOG_INTERVAL_S = 30.0

# Upstream generator must not block forever on Queue.get when ``_STOP`` could not be
# enqueued (e.g. queue full); poll and exit if ``stop_forwarder`` is set.
_UPSTREAM_QUEUE_GET_TIMEOUT_S = 0.5

# Max seconds to wait for root health + active leaf→root ``StreamLogs`` before binding
# the downstream port and accepting compute-node clients.
_LEAF_UPSTREAM_ROOT_READY_TIMEOUT_S = 120.0

_GRPC_OPTS = [
    ('grpc.max_send_message_length', 10 * 1024 * 1024),
    ('grpc.max_receive_message_length', 10 * 1024 * 1024),
    ('grpc.http2.min_time_between_pings_ms', 5000),
    ('grpc.http2.min_ping_interval_without_data_ms', 5000),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.keepalive_permit_without_calls', 1),
]


def _copy_chunk(chunk: log_aggregation_pb2.LogChunk) -> log_aggregation_pb2.LogChunk:
    return log_aggregation_pb2.LogChunk(
        node_id=chunk.node_id,
        data=chunk.data,
        file_path=chunk.file_path,
    )


def _format_grpc_peer(raw: Optional[str]) -> str:
    """Decode URL-encoded characters in ``context.peer()`` for readable log lines."""
    if not raw:
        return "unknown_peer"
    return urllib.parse.unquote(raw)


class _LeafChunkQueue:
    """Per-leaf buffer from many StreamLogs handlers to one upstream forwarder."""

    def __init__(self, max_chunks: int):
        if max_chunks < 1:
            raise ValueError('max_chunks must be >= 1')
        self.max_chunks = max_chunks
        self._q: queue.Queue = queue.Queue(maxsize=max_chunks)

    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def maxsize(self) -> int:
        return self._q.maxsize

    def put_stop(self, timeout: float) -> None:
        self._q.put(_STOP, timeout=timeout)

    def get_upstream(self, timeout: float) -> Union[log_aggregation_pb2.LogChunk, object]:
        return self._q.get(timeout=timeout)

    def put_chunk(
        self,
        chunk: log_aggregation_pb2.LogChunk,
        reject: threading.Event,
        *,
        peer: str,
        source_node_id: str,
    ) -> bool:
        """
        Enqueue ``chunk`` (already a copy). Blocks until space or ``reject`` is set.

        Returns:
            True if enqueued, False if ``reject`` was set while waiting (shutdown).
        """
        sz = len(chunk.data)
        wait_start = time.monotonic()
        next_log_at = wait_start + _LEAF_QUEUE_BACKPRESSURE_LOG_FIRST_S

        def maybe_log() -> None:
            nonlocal next_log_at
            now = time.monotonic()
            if now < next_log_at:
                return
            next_log_at = now + _LEAF_QUEUE_BACKPRESSURE_LOG_INTERVAL_S
            elapsed = now - wait_start
            logger.warning(
                f"Leaf queue backpressure (chunk count) after {elapsed:.1f}s "
                f"(peer={peer} source_node_id={source_node_id} "
                f"max_chunks={self.max_chunks} qsize≈{self._q.qsize()} chunk_bytes={sz})"
            )

        while True:
            if reject.is_set():
                return False
            try:
                self._q.put(chunk, timeout=_LEAF_QUEUE_PUT_SLICE_S)
                return True
            except queue.Full:
                maybe_log()


class _UpstreamForwarder:
    """Runs stub.StreamLogs in a thread; reconnects on failure until stopped."""

    def __init__(
        self,
        upstream: str,
        chunk_queue: _LeafChunkQueue,
        stop_event: threading.Event,
        reconnect_sleep: float = 2.0,
    ):
        self._upstream = upstream
        self._q = chunk_queue
        self._stop = stop_event
        self._reconnect_sleep = reconnect_sleep
        self._thread: Optional[threading.Thread] = None
        #: Set while an active upstream StreamLogs RPC is in progress (root can receive).
        self.upstream_ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name='LeafUpstream', daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _wait_root_ready(self, stub: Any) -> bool:
        while not self._stop.is_set():
            try:
                r = stub.HealthCheck(log_aggregation_pb2.HealthRequest(), timeout=2.0)
                if r.healthy:
                    return True
            except grpc.RpcError:
                pass
            time.sleep(0.5)
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            ch = None
            try:
                ch = grpc.insecure_channel(
                    self._upstream,
                    options=[
                        *_GRPC_OPTS,
                        ('grpc.keepalive_time_ms', 60000),
                        ('grpc.keepalive_timeout_ms', 20000),
                    ],
                )
                stub = log_aggregation_pb2_grpc.LogAggregationServiceStub(ch)
                if not self._wait_root_ready(stub):
                    break

                def gen() -> Iterator[log_aggregation_pb2.LogChunk]:
                    while True:
                        try:
                            item = self._q.get_upstream(timeout=_UPSTREAM_QUEUE_GET_TIMEOUT_S)
                        except queue.Empty:
                            if self._stop.is_set():
                                logger.info(
                                    "Upstream StreamLogs generator exiting: stop requested "
                                    f"and no chunk within {_UPSTREAM_QUEUE_GET_TIMEOUT_S}s "
                                    "(STOP sentinel may be missing if queue was full)"
                                )
                                return
                            continue
                        if item is _STOP:
                            return
                        yield item  # type: ignore[misc]

                logger.info(f"Connected upstream StreamLogs to {self._upstream}")
                self.upstream_ready.set()
                # Chunks leave _LeafChunkQueue when gen() yields them. If this RPC errors
                # mid-flight, yields already handed to gRPC (send path / HTTP/2 buffers) may
                # never reach the root and are not put back on the queue—the next loop
                # iteration starts a fresh gen() at the new head. Not end-to-end lossless.
                stub.StreamLogs(gen())
                logger.info("Upstream StreamLogs ended")
            except grpc.RpcError as e:
                if not self._stop.is_set():
                    logger.warning(f"Upstream gRPC error, will retry: {e.code()} {e.details()}")
            except Exception as e:
                if not self._stop.is_set():
                    logger.error(f"Upstream forwarder error: {e}", exc_info=True)
            finally:
                self.upstream_ready.clear()
                if ch is not None:
                    ch.close()
            if not self._stop.is_set():
                time.sleep(self._reconnect_sleep)
        logger.info("Upstream forwarder stopped")


class LeafLogServicer(log_aggregation_pb2_grpc.LogAggregationServiceServicer):
    def __init__(
        self,
        chunk_queue: _LeafChunkQueue,
        reject_new_streams: threading.Event,
        upstream_ready: threading.Event,
    ):
        self._chunkq = chunk_queue
        self._reject = reject_new_streams
        self._upstream_ready = upstream_ready
        self.connected_clients = 0
        self.clients_lock = threading.Lock()

    def StreamLogs(self, request_iterator, context):
        if self._reject.is_set():
            context.abort(grpc.StatusCode.UNAVAILABLE, "leaf server is shutting down")
            # abort() may return without raising on older grpcio; do not fall through.
            return log_aggregation_pb2.StreamResponse(status="UNAVAILABLE", bytes_received=0)
        with self.clients_lock:
            self.connected_clients += 1
        peer = _format_grpc_peer(context.peer())
        source_node_id = None
        total = 0
        nchunks = 0
        try:
            for chunk in request_iterator:
                if self._reject.is_set():
                    break
                if source_node_id is None:
                    source_node_id = chunk.node_id
                    logger.info(
                        f"Compute node StreamLogs opened peer={peer} source_node_id={source_node_id}"
                    )
                copied = _copy_chunk(chunk)
                if not self._chunkq.put_chunk(
                    copied,
                    self._reject,
                    peer=peer,
                    source_node_id=source_node_id or 'n/a',
                ):
                    break
                total += len(chunk.data)
                nchunks += 1
            return log_aggregation_pb2.StreamResponse(status="OK", bytes_received=total)
        finally:
            end_src = (
                f"source_node_id={source_node_id}"
                if source_node_id is not None
                else "source_node_id=n/a"
            )
            logger.info(
                f"Compute node StreamLogs ended peer={peer} {end_src} "
                f"chunks={nchunks} received_KiB={total / 1024:.1f}"
            )
            with self.clients_lock:
                self.connected_clients -= 1

    def HealthCheck(self, request, context):
        with self.clients_lock:
            n = self.connected_clients
        ok = self._upstream_ready.is_set()
        return log_aggregation_pb2.HealthResponse(healthy=ok, connected_clients=n)


def serve(
    host: str,
    port: int,
    upstream: str,
    max_workers: int,
    max_queue_chunks: int,
    graceful_shutdown_timeout: float,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(process)5s] %(filename)s:%(lineno)d %(message)s',
    )
    chunk_queue = _LeafChunkQueue(max_queue_chunks)
    stop_forwarder = threading.Event()
    reject_new = threading.Event()
    addr = f'{host}:{port}'

    forwarder = _UpstreamForwarder(upstream, chunk_queue, stop_forwarder)
    forwarder.start()
    logger.info(
        f"Leaf waiting for upstream root {upstream!r} to become healthy "
        f"(timeout={int(_LEAF_UPSTREAM_ROOT_READY_TIMEOUT_S)}s); downstream port {addr} "
        "will not accept connections until upstream is ready — clients may get ECONNREFUSED."
    )
    if not forwarder.upstream_ready.wait(timeout=_LEAF_UPSTREAM_ROOT_READY_TIMEOUT_S):
        logger.error("Timeout waiting for upstream root; exiting")
        stop_forwarder.set()
        try:
            chunk_queue.put_stop(timeout=1.0)
        except queue.Full:
            pass
        forwarder.join(timeout=10.0)
        sys.exit(1)

    servicer = LeafLogServicer(chunk_queue, reject_new, forwarder.upstream_ready)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=_GRPC_OPTS,
    )
    log_aggregation_pb2_grpc.add_LogAggregationServiceServicer_to_server(servicer, server)
    server.add_insecure_port(addr)
    server.start()
    logger.info(
        f"Leaf log server on {addr}, upstream={upstream}, max_queue_chunks={max_queue_chunks}"
    )

    def _leaf_shutdown_snapshot(phase: str, **kwargs: Any) -> None:
        with servicer.clients_lock:
            downstream = servicer.connected_clients
        qdepth = chunk_queue.qsize()
        upstream_ok = forwarder.upstream_ready.is_set()
        fwd_alive = bool(forwarder._thread and forwarder._thread.is_alive())
        parts = [
            f"phase={phase!r}",
            f"downstream_StreamLogs={downstream}",
            f"chunk_queue_depth={qdepth}",
            f"chunk_queue_maxsize={max_queue_chunks}",
            f"upstream_StreamLogs_active={upstream_ok}",
            f"forwarder_thread_alive={fwd_alive}",
        ]
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}={v}")
        logger.info("Leaf shutdown snapshot: " + ", ".join(parts))

    def shutdown_leaf():
        logger.info("Graceful shutdown: waiting for downstream clients...")
        reject_new.set()
        _leaf_shutdown_snapshot("after_reject_new_streams")
        start = time.time()
        exit_reason = "running"
        while True:
            with servicer.clients_lock:
                ac = servicer.connected_clients
            if ac == 0:
                exit_reason = "all_downstream_disconnected"
                break
            if time.time() - start >= graceful_shutdown_timeout:
                exit_reason = f"timeout_after_{graceful_shutdown_timeout}s"
                logger.warning(
                    f"{ac} downstream StreamLogs client(s) still connected after "
                    f"{graceful_shutdown_timeout}s — closing may drop in-flight leaf→root data"
                )
                break
            time.sleep(0.1)
        _leaf_shutdown_snapshot(
            "after_downstream_wait",
            exit_reason=exit_reason,
            downstream_streams_remaining=ac,
        )
        stop_forwarder.set()
        try:
            chunk_queue.put_stop(timeout=_STOP_SENTINEL_PUT_TIMEOUT_S)
        except queue.Full:
            logger.error(
                f"Could not enqueue upstream STOP sentinel within {_STOP_SENTINEL_PUT_TIMEOUT_S}s "
                "(queue full). Upstream forwarder may not exit cleanly; logs may be stuck in "
                "leaf queue."
            )
        join_timeout = graceful_shutdown_timeout + 5.0
        forwarder.join(timeout=join_timeout)
        fwd_still_alive = bool(forwarder._thread and forwarder._thread.is_alive())
        if fwd_still_alive:
            logger.warning(
                f"Upstream forwarder thread still alive after join(timeout={join_timeout}s); "
                "root may already be gone or RPC stalled — possible log loss on this leaf."
            )
        _leaf_shutdown_snapshot(
            "after_forwarder_join",
            forwarder_thread_still_alive=fwd_still_alive,
            chunk_queue_depth_after_join=chunk_queue.qsize(),
        )
        # stop() returns before the internal _serve thread and executor finish; exiting the
        # process immediately races with late RPC dispatch → RuntimeError: cannot schedule
        # new futures after shutdown. Wait for full termination after stop().
        stop_grace_seconds = 5.0
        termination_wait_timeout_seconds = stop_grace_seconds + 60.0
        logger.info(
            f"Stopping leaf gRPC server (grace={stop_grace_seconds}s); open RPCs may be cancelled mid-stream."
        )
        server.stop(grace=stop_grace_seconds)
        # wait_for_termination returns True iff the wait timed out (per grpc API).
        if server.wait_for_termination(timeout=termination_wait_timeout_seconds):
            logger.warning(
                "Timed out waiting for leaf gRPC server to finish termination after stop() "
                f"(>{termination_wait_timeout_seconds}s)"
            )
        logger.info("Leaf server exit")

    def on_signal(signum, frame):
        logger.info(f"Signal {signum}, shutting down leaf...")
        shutdown_leaf()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        shutdown_leaf()


def main() -> None:
    p = argparse.ArgumentParser(description="First-level gRPC log aggregator (forwards to root).")
    p.add_argument('--host', type=str, default='0.0.0.0')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--upstream', type=str, required=True, help='Root server host:port')
    p.add_argument('--max-workers', type=int, default=100)
    p.add_argument(
        '--max-queue-chunks',
        type=int,
        default=8192,
        help='Max queued chunks (item count) before blocking handlers (backpressure). '
        'Clients batch lines up to 256 KiB per chunk (see per_cycle_logs); typical '
        'chunks are often smaller.',
    )
    p.add_argument('--graceful-shutdown-timeout', type=float, default=60.0)
    args = p.parse_args()
    serve(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        max_workers=args.max_workers,
        max_queue_chunks=args.max_queue_chunks,
        graceful_shutdown_timeout=args.graceful_shutdown_timeout,
    )


if __name__ == '__main__':
    main()
