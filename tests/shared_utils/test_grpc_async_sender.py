# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
import threading
from concurrent import futures
from unittest.mock import MagicMock

import grpc
import pytest

from nvidia_resiliency_ext.shared_utils.grpc_async_sender import AsyncUnarySender
from nvidia_resiliency_ext.shared_utils.proto import cycle_info_pb2 as pb
from nvidia_resiliency_ext.shared_utils.proto import cycle_info_pb2_grpc as pb_grpc

REQUEST = pb.CycleOutcome(cycle=3, node="node0")


class _Receiver(pb_grpc.CycleInfoServiceServicer):
    def __init__(self):
        self.outcomes = []
        self.received = threading.Event()

    def PublishCycleOutcome(self, request, context):
        self.outcomes.append(request)
        self.received.set()
        return pb.PublishReply()


@pytest.fixture
def receiver():
    """A CycleInfoService on localhost, and a stub connected to it."""
    servicer = _Receiver()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_CycleInfoServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield servicer, pb_grpc.CycleInfoServiceStub(channel)
    finally:
        channel.close()
        server.stop(grace=None)


def _unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _never_completing_stub():
    stub = MagicMock()
    # A new future per call, never completing.
    stub.PublishCycleOutcome.future.side_effect = lambda *args, **kwargs: MagicMock()
    return stub


def test_request_reaches_the_receiver(receiver):
    servicer, stub = receiver
    sender = AsyncUnarySender(MagicMock())

    assert sender.send(stub.PublishCycleOutcome, REQUEST, "outcome")
    sender.close()

    assert servicer.received.wait(timeout=5)
    assert servicer.outcomes == [REQUEST]


def test_send_starts_the_call_without_waiting():
    stub = MagicMock()
    sender = AsyncUnarySender(MagicMock(), rpc_timeout=2.0)

    assert sender.send(stub.PublishCycleOutcome, REQUEST, "outcome")

    stub.PublishCycleOutcome.future.assert_called_once_with(REQUEST, timeout=2.0)
    stub.PublishCycleOutcome.assert_not_called()  # no blocking call


def test_unreachable_receiver_is_logged_without_raising():
    channel = grpc.insecure_channel(f"127.0.0.1:{_unused_port()}")
    logger = MagicMock()
    sender = AsyncUnarySender(logger, close_timeout=5)
    try:
        assert sender.send(pb_grpc.CycleInfoServiceStub(channel).PublishCycleOutcome, REQUEST, "x")
        sender.close()  # waits for the call, whose failure is logged
        logger.warning.assert_called_once()
        assert "UNAVAILABLE" in logger.warning.call_args.args[0]
    finally:
        channel.close()


def test_requests_are_dropped_beyond_max_pending():
    stub = _never_completing_stub()
    logger = MagicMock()
    sender = AsyncUnarySender(logger, max_pending=2)

    sent = [sender.send(stub.PublishCycleOutcome, REQUEST, "outcome") for _ in range(3)]

    assert sent == [True, True, False]
    assert stub.PublishCycleOutcome.future.call_count == 2
    logger.warning.assert_called_once()


def test_nothing_is_sent_after_close():
    stub = MagicMock()
    sender = AsyncUnarySender(MagicMock())
    sender.close()
    assert not sender.send(stub.PublishCycleOutcome, REQUEST, "outcome")
    stub.PublishCycleOutcome.future.assert_not_called()


def test_close_cancels_calls_still_pending_after_the_timeout():
    future = MagicMock()
    future.result.side_effect = grpc.FutureTimeoutError()
    stub = MagicMock()
    stub.PublishCycleOutcome.future.return_value = future
    sender = AsyncUnarySender(MagicMock())
    sender.send(stub.PublishCycleOutcome, REQUEST, "outcome")

    sender.close(timeout=0)

    future.cancel.assert_called_once_with()


def test_call_failing_to_start_is_logged_without_raising():
    stub = MagicMock()
    stub.PublishCycleOutcome.future.side_effect = RuntimeError("channel closed")
    logger = MagicMock()
    sender = AsyncUnarySender(logger)

    assert not sender.send(stub.PublishCycleOutcome, REQUEST, "outcome")
    logger.warning.assert_called_once()
