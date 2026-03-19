#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
First-level gRPC log aggregator (leaf): accepts StreamLogs from training nodes,
forwards chunks to the root log server over a single outbound StreamLogs RPC.

Uses a bounded queue for backpressure when the root is slow. Root must be
started before leaves (launcher order).
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
from typing import Any, Iterator, Optional

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


class _UpstreamForwarder:
    """Runs stub.StreamLogs in a thread; reconnects on failure until stopped."""

    def __init__(
        self,
        upstream: str,
        chunk_queue: queue.Queue,
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
                        item = self._q.get()
                        if item is _STOP:
                            return
                        yield item  # type: ignore[misc]

                logger.info(f"Connected upstream StreamLogs to {self._upstream}")
                self.upstream_ready.set()
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
        chunk_queue: queue.Queue,
        reject_new_streams: threading.Event,
        upstream_ready: threading.Event,
    ):
        self._q = chunk_queue
        self._reject = reject_new_streams
        self._upstream_ready = upstream_ready
        self.connected_clients = 0
        self.clients_lock = threading.Lock()

    def StreamLogs(self, request_iterator, context):
        if self._reject.is_set():
            context.abort(grpc.StatusCode.UNAVAILABLE, "leaf server is shutting down")
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
                self._q.put(_copy_chunk(chunk))
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
    chunk_queue: queue.Queue = queue.Queue(maxsize=max_queue_chunks)
    stop_forwarder = threading.Event()
    reject_new = threading.Event()

    forwarder = _UpstreamForwarder(upstream, chunk_queue, stop_forwarder)
    forwarder.start()
    if not forwarder.upstream_ready.wait(timeout=120.0):
        logger.error("Timeout waiting for upstream root; exiting")
        stop_forwarder.set()
        try:
            chunk_queue.put(_STOP, timeout=1.0)
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
    addr = f'{host}:{port}'
    server.add_insecure_port(addr)
    server.start()
    logger.info(
        f"Leaf log server on {addr}, upstream={upstream}, max_queue_chunks={max_queue_chunks}"
    )

    def shutdown_leaf():
        logger.info("Graceful shutdown: waiting for downstream clients...")
        reject_new.set()
        start = time.time()
        while True:
            with servicer.clients_lock:
                ac = servicer.connected_clients
            if ac == 0:
                break
            if time.time() - start >= graceful_shutdown_timeout:
                logger.warning(f"{ac} client(s) still connected after {graceful_shutdown_timeout}s")
                break
            time.sleep(0.1)
        stop_forwarder.set()
        chunk_queue.put(_STOP)
        forwarder.join(timeout=graceful_shutdown_timeout + 5.0)
        server.stop(grace=5.0)
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
        help='Max queued chunks before blocking clients (backpressure). Default 8192.',
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
