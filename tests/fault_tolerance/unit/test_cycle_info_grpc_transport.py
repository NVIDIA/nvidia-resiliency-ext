# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info import grpc_transport
from nvidia_resiliency_ext.fault_tolerance.cycle_info.grpc_transport import (
    GrpcCycleInfoPublisher,
    GrpcCycleInfoReader,
    cycle_start_from_proto,
    cycle_start_to_proto,
    outcome_from_proto,
    outcome_to_proto,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    MAX_EXIT_DETAILS_LEN,
    MAX_SECTION_NAME_LEN,
    MAX_TIMEOUT_SECTIONS,
    CycleEndReason,
    CycleStartInfo,
    LocalCycleOutcome,
    RankEndReason,
    RankOutcome,
    TimeoutKind,
)
from nvidia_resiliency_ext.shared_utils.proto import cycle_info_pb2 as pb
from nvidia_resiliency_ext.shared_utils.proto import cycle_info_pb2_grpc as pb_grpc

START = CycleStartInfo(3, 10.5, ("node0", "node1"), ("node2",), (0, 1), "/logs/train_cycle3.log")
OUTCOME = LocalCycleOutcome(
    cycle=3,
    node="node0",
    node_end=CycleEndReason.FAILED,
    detected_at=12.25,
    ranks=(
        RankOutcome(0, 0, 100, RankEndReason.STOPPED_BY_LAUNCHER),
        RankOutcome(
            1,
            1,
            101,
            RankEndReason.STOPPED_BY_RMON,
            exitcode=-9,
            timeout_kind=TimeoutKind.SECTION,
            timeout_sections=("step",),
        ),
    ),
)


def test_cycle_start_to_proto():
    message = cycle_start_to_proto(START)
    assert (message.cycle, message.started_at, message.cycle_log_file) == (
        3,
        10.5,
        "/logs/train_cycle3.log",
    )
    assert list(message.active_nodes) == ["node0", "node1"]
    assert list(message.standby_nodes) == ["node2"]
    assert list(message.active_ranks) == [0, 1]


def test_outcome_to_proto():
    message = outcome_to_proto(OUTCOME)

    assert (message.cycle, message.node, message.node_end, message.detected_at) == (
        3,
        "node0",
        pb.CYCLE_END_REASON_FAILED,
        12.25,
    )
    stopped, timed_out = message.ranks
    assert (stopped.global_rank, stopped.local_rank, stopped.pid) == (0, 0, 100)
    assert stopped.end == pb.RANK_END_REASON_STOPPED_BY_LAUNCHER
    assert not stopped.HasField("exitcode")  # had not exited
    assert stopped.timeout_kind == pb.TIMEOUT_KIND_UNSPECIFIED
    assert timed_out.end == pb.RANK_END_REASON_STOPPED_BY_RANK_MONITOR
    assert timed_out.exitcode == -9
    assert timed_out.timeout_kind == pb.TIMEOUT_KIND_SECTION
    assert list(timed_out.timeout_sections) == ["step"]


def test_outcome_to_proto_bounds_workload_strings():
    sections = tuple("s" * 100 + str(i) for i in range(MAX_TIMEOUT_SECTIONS + 2))
    rank = RankOutcome(
        0,
        0,
        100,
        RankEndReason.STOPPED_BY_RMON,
        exitcode=-9,
        exitdetails="x" * 1000,
        timeout_kind=TimeoutKind.SECTION,
        timeout_sections=sections,
    )
    outcome = LocalCycleOutcome(3, "node0", CycleEndReason.FAILED, 1.0, (rank,))

    message = outcome_to_proto(outcome).ranks[0]

    assert message.exit_details == "x" * MAX_EXIT_DETAILS_LEN
    assert list(message.timeout_sections) == [
        name[:MAX_SECTION_NAME_LEN] for name in sections[:MAX_TIMEOUT_SECTIONS]
    ]


@pytest.mark.parametrize(
    ("enum", "mapping"),
    [
        (CycleEndReason, grpc_transport._CYCLE_END_TO_PROTO),
        (RankEndReason, grpc_transport._RANK_END_TO_PROTO),
        (TimeoutKind, grpc_transport._TIMEOUT_KIND_TO_PROTO),
    ],
)
def test_every_enum_member_has_a_distinct_wire_value(enum, mapping):
    assert set(enum) <= set(mapping)
    wire_values = list(mapping.values())
    assert len(set(wire_values)) == len(wire_values)


class _Receiver(pb_grpc.CycleInfoServiceServicer):
    def __init__(self):
        self.starts = []
        self.outcomes = []
        self.received = threading.Event()

    def PublishCycleStart(self, request, context):
        self.starts.append(request)
        self.received.set()
        return pb.PublishReply()

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


@pytest.mark.parametrize("kind", ["start", "outcome"])
def test_records_reach_the_receiver(receiver, kind):
    servicer, stub = receiver
    publisher = GrpcCycleInfoPublisher(stub)

    if kind == "start":
        publisher.publish_cycle_start(START)
    else:
        publisher.publish_outcome(OUTCOME)
    publisher.close()

    assert servicer.received.wait(timeout=5)
    if kind == "start":
        assert servicer.starts == [cycle_start_to_proto(START)]
    else:
        assert servicer.outcomes == [outcome_to_proto(OUTCOME)]


def test_publish_does_not_wait_for_the_call():
    stub = MagicMock()
    publisher = GrpcCycleInfoPublisher(stub, rpc_timeout=2.0)

    publisher.publish_outcome(OUTCOME)

    stub.PublishCycleOutcome.future.assert_called_once_with(outcome_to_proto(OUTCOME), timeout=2.0)
    stub.PublishCycleOutcome.assert_not_called()  # no blocking call


def test_nothing_is_published_after_close():
    stub = MagicMock()
    publisher = GrpcCycleInfoPublisher(stub)
    publisher.close()
    publisher.publish_cycle_start(START)
    publisher.publish_outcome(OUTCOME)
    stub.PublishCycleStart.future.assert_not_called()
    stub.PublishCycleOutcome.future.assert_not_called()


def test_record_failing_to_encode_is_logged_without_raising():
    stub = MagicMock()
    publisher = GrpcCycleInfoPublisher(stub)
    with patch.dict(grpc_transport._CYCLE_END_TO_PROTO, clear=True):
        with patch.object(grpc_transport, "logger") as logger:
            publisher.publish_outcome(OUTCOME)
    logger.warning.assert_called_once()
    stub.PublishCycleOutcome.future.assert_not_called()


def test_records_round_trip_through_proto():
    assert cycle_start_from_proto(cycle_start_to_proto(START)) == START
    assert outcome_from_proto(outcome_to_proto(OUTCOME)) == OUTCOME


@pytest.mark.parametrize("field", ["node_end", "rank_end"])
def test_outcome_with_unspecified_reason_is_rejected(field):
    message = outcome_to_proto(OUTCOME)
    if field == "node_end":
        message.node_end = pb.CYCLE_END_REASON_UNSPECIFIED
    else:
        message.ranks[0].end = pb.RANK_END_REASON_UNSPECIFIED
    with pytest.raises(ValueError):
        outcome_from_proto(message)


@pytest.mark.parametrize("transport", ["unix", "tcp"])
def test_reader_receives_what_the_publisher_sends(tmp_path, transport):
    address = f"unix:{tmp_path}/analyzer.sock" if transport == "unix" else "127.0.0.1:0"
    reader = GrpcCycleInfoReader(address)
    target = address if transport == "unix" else f"127.0.0.1:{reader.port}"
    channel = grpc.insecure_channel(target)
    publisher = GrpcCycleInfoPublisher(pb_grpc.CycleInfoServiceStub(channel), close_timeout=5)
    try:
        publisher.publish_cycle_start(START)
        publisher.publish_outcome(OUTCOME)
        publisher.close()  # waits until both calls completed

        assert reader.read_cycle_starts() == [START]
        assert reader.read_outcomes() == [OUTCOME]
        assert reader.read_cycle_starts() == []  # each record is read once
        assert reader.read_outcomes() == []
    finally:
        channel.close()
        reader.close()


def test_reader_rejects_an_undecodable_record():
    reader = GrpcCycleInfoReader("127.0.0.1:0")
    channel = grpc.insecure_channel(f"127.0.0.1:{reader.port}")
    try:
        message = outcome_to_proto(OUTCOME)
        message.node_end = pb.CYCLE_END_REASON_UNSPECIFIED
        with pytest.raises(grpc.RpcError) as error:
            pb_grpc.CycleInfoServiceStub(channel).PublishCycleOutcome(message, timeout=5)
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert reader.read_outcomes() == []
    finally:
        channel.close()
        reader.close()
