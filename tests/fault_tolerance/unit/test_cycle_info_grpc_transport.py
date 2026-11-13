# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent import futures
from dataclasses import replace
from unittest.mock import MagicMock, patch

import grpc
import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info import grpc_transport
from nvidia_resiliency_ext.fault_tolerance.cycle_info.grpc_transport import (
    GrpcCycleInfoPublisher,
    GrpcCycleInfoReader,
    cycle_info_from_proto,
    cycle_info_to_proto,
    end_report_from_proto,
    end_report_to_proto,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    CycleInfo,
    EndReason,
    NodeCycleEndReport,
)
from nvidia_resiliency_ext.fault_tolerance.utils import range_str_to_ranks
from nvidia_resiliency_ext.shared_utils.proto import log_aggregation_pb2, log_aggregation_pb2_grpc
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2_grpc as pb_grpc

# Nodes as the store host knows them (FQDN, rank order); ranks aligned with them.
START = CycleInfo(
    job_id="job1",
    attempt_index=2,
    cycle_number=3,
    started_at=1_700_000_000.5,
    cycle_log_file="/logs/train_cycle3.log",
    active_nodes=("node002.cluster", "node001.cluster"),
    active_ranks=(0, 1),
    standby_nodes=("node003.cluster",),
    timestamp=1_700_000_000_500,
)
END = replace(START, ended_at=1_700_000_100.25, timestamp=1_700_000_100_250)
REPORT = NodeCycleEndReport(
    job_id="job1",
    attempt_index=2,
    cycle_number=3,
    node_name="node001.cluster",
    end_reason=EndReason.TERMINATED_BY_RANK_MONITOR,
    detected_timestamp=1_700_000_099_123,
    exit_details='[{"rank":1}]',
)


def test_cycle_info_to_proto():
    message = cycle_info_to_proto(END)

    assert (message.job_id, message.attempt_index, message.cycle_number) == ("job1", 2, 3)
    assert message.cycle_start_time == "2023-11-14T22:13:20.500000Z"
    assert message.cycle_end_time == "2023-11-14T22:15:00.250000Z"
    assert message.cycle_log_file == "/logs/train_cycle3.log"
    # Compressed: hosts sorted and without their domain.
    assert message.active_nodes == "node[001-002]"
    assert message.standby_nodes == "node[003]"
    assert message.timestamp == 1_700_000_100_250


def test_cycle_info_ranks_follow_the_sorted_hosts():
    message = cycle_info_to_proto(START)
    assert message.active_nodes == "node[001-002]"
    # node001 has rank 1, node002 rank 0.
    assert range_str_to_ranks(message.active_ranks) == [1, 0]


def test_cycle_info_round_trip_normalizes_the_node_lists():
    received = cycle_info_from_proto(cycle_info_to_proto(START))

    assert received == replace(
        START,
        active_nodes=("node001", "node002"),
        active_ranks=(1, 0),
        standby_nodes=("node003",),
    )


def test_cycle_info_without_end_round_trips_unset():
    message = cycle_info_to_proto(START)
    assert message.cycle_end_time == ""
    assert cycle_info_from_proto(message).ended_at is None


def test_cycle_info_end_round_trips():
    assert cycle_info_from_proto(cycle_info_to_proto(END)).ended_at == END.ended_at


def test_cycle_info_without_start_time_is_rejected():
    message = cycle_info_to_proto(START)
    message.cycle_start_time = ""
    with pytest.raises(ValueError):
        cycle_info_from_proto(message)


def test_end_report_to_proto():
    message = end_report_to_proto(REPORT)

    assert (message.job_id, message.attempt_index, message.cycle_number) == ("job1", 2, 3)
    assert message.node_name == "node001.cluster"
    assert message.end_reason == pb.CycleEndReasonWorkerTerminatedByRankMonitor
    assert message.detected_timestamp == 1_700_000_099_123
    assert message.exit_details == '[{"rank":1}]'


def test_end_report_round_trips():
    assert end_report_from_proto(end_report_to_proto(REPORT)) == REPORT


def test_end_report_without_detection_time_is_rejected():
    message = end_report_to_proto(REPORT)
    message.detected_timestamp = 0
    with pytest.raises(ValueError):
        end_report_from_proto(message)


def test_end_report_with_unknown_end_reason_is_rejected():
    message = end_report_to_proto(REPORT)
    message.end_reason = 99  # a value from a newer version
    with pytest.raises(ValueError):
        end_report_from_proto(message)


def test_every_end_reason_has_a_distinct_wire_value():
    mapping = grpc_transport._END_REASON_TO_PROTO
    assert set(mapping) == set(EndReason)
    assert sorted(mapping.values()) == sorted(pb.EndReason.values())


def _rpc_error(code, details, debug=None):
    class _Error(grpc.RpcError):
        def code(self):
            return code

        def details(self):
            return details

        def debug_error_string(self):
            return debug

    return _Error()


class TestPublisher:
    def test_cycle_infos_are_blocking_calls_with_the_timeout(self):
        stub = MagicMock()
        publisher = GrpcCycleInfoPublisher(cycle_info_stub=stub, rpc_timeout=2.0)

        assert publisher.publish_cycle_info(START) is True
        assert publisher.update_cycle_info(END) is True

        stub.PublishCycleInfo.assert_called_once_with(cycle_info_to_proto(START), timeout=2.0)
        stub.UpdateCycleInfo.assert_called_once_with(cycle_info_to_proto(END), timeout=2.0)

    def test_end_report_is_a_one_record_stream(self):
        stub = MagicMock()
        publisher = GrpcCycleInfoPublisher(data_record_stub=stub, rpc_timeout=2.0)

        assert publisher.publish_end_report(REPORT) is True

        (requests,), kwargs = stub.StreamDataRecords.call_args
        assert kwargs == {"timeout": 2.0}
        assert list(requests) == [
            log_aggregation_pb2.DataRecord(node_cycle_end_report=end_report_to_proto(REPORT))
        ]

    @pytest.mark.parametrize("kind", ["cycle_info", "end_report"])
    def test_publish_without_the_stub_is_dropped(self, kind):
        publisher = GrpcCycleInfoPublisher()
        with patch.object(grpc_transport, "logger") as logger:
            if kind == "cycle_info":
                assert publisher.publish_cycle_info(START) is False
            else:
                assert publisher.publish_end_report(REPORT) is False
        logger.warning.assert_called_once()

    def test_failed_call_logs_the_grpc_core_debug_string_without_raising(self):
        stub = MagicMock()
        stub.PublishCycleInfo.side_effect = _rpc_error(
            grpc.StatusCode.CANCELLED, "CANCELLED", "Received RST_STREAM with error code 8"
        )
        with patch.object(grpc_transport, "logger") as logger:
            assert GrpcCycleInfoPublisher(cycle_info_stub=stub).publish_cycle_info(START) is False
        message = logger.warning.call_args.args[0]
        assert "CANCELLED" in message
        assert "Received RST_STREAM with error code 8" in message

    def test_unexpected_error_is_logged_without_raising(self):
        stub = MagicMock()
        stub.StreamDataRecords.side_effect = ValueError("Cannot invoke RPC on closed channel!")
        with patch.object(grpc_transport, "logger") as logger:
            publisher = GrpcCycleInfoPublisher(data_record_stub=stub)
            assert publisher.publish_end_report(REPORT) is False
        logger.warning.assert_called_once()

    def test_record_failing_to_encode_is_logged_without_raising(self):
        stub = MagicMock()
        publisher = GrpcCycleInfoPublisher(data_record_stub=stub)
        with patch.dict(grpc_transport._END_REASON_TO_PROTO, clear=True):
            with patch.object(grpc_transport, "logger") as logger:
                assert publisher.publish_end_report(REPORT) is False
        logger.warning.assert_called_once()
        stub.StreamDataRecords.assert_not_called()


class _DataRecordReceiver(log_aggregation_pb2_grpc.DataRecordAggregationServiceServicer):
    def __init__(self):
        self.records = []

    def StreamDataRecords(self, request_iterator, context):
        self.records.extend(request_iterator)
        return log_aggregation_pb2.StreamResponse(status="OK")


def test_end_report_reaches_a_log_server():
    receiver = _DataRecordReceiver()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    log_aggregation_pb2_grpc.add_DataRecordAggregationServiceServicer_to_server(receiver, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = log_aggregation_pb2_grpc.DataRecordAggregationServiceStub(channel)
        assert GrpcCycleInfoPublisher(data_record_stub=stub).publish_end_report(REPORT)

        # The publish returned once the server had the record.
        (record,) = receiver.records
        assert end_report_from_proto(record.node_cycle_end_report) == REPORT
    finally:
        channel.close()
        server.stop(grace=None)


@pytest.mark.parametrize("transport", ["unix", "tcp"])
def test_reader_receives_each_kind_once(tmp_path, transport):
    address = f"unix:{tmp_path}/analyzer.sock" if transport == "unix" else "127.0.0.1:0"
    reader = GrpcCycleInfoReader(address)
    target = address if transport == "unix" else f"127.0.0.1:{reader.port}"
    channel = grpc.insecure_channel(target)
    stub = pb_grpc.CycleInfoServiceStub(channel)
    publisher = GrpcCycleInfoPublisher(cycle_info_stub=stub)
    try:
        assert publisher.publish_cycle_info(START)
        assert publisher.update_cycle_info(END)
        # End reports reach the analyzer from the log root, directly over this RPC.
        stub.PublishNodeCycleEndReport(end_report_to_proto(REPORT), timeout=5)

        assert reader.read_cycle_infos() == [cycle_info_from_proto(cycle_info_to_proto(START))]
        assert reader.read_cycle_info_updates() == [cycle_info_from_proto(cycle_info_to_proto(END))]
        assert reader.read_end_reports() == [REPORT]
        assert reader.read_cycle_infos() == []  # each record is read once
        assert reader.read_cycle_info_updates() == []
        assert reader.read_end_reports() == []
    finally:
        channel.close()
        reader.close()


@pytest.mark.parametrize(
    "rpc", ["PublishCycleInfo", "UpdateCycleInfo", "PublishNodeCycleEndReport"]
)
def test_reader_rejects_an_undecodable_record(rpc):
    reader = GrpcCycleInfoReader("127.0.0.1:0")
    channel = grpc.insecure_channel(f"127.0.0.1:{reader.port}")
    if rpc == "PublishNodeCycleEndReport":
        message = end_report_to_proto(REPORT)
        message.detected_timestamp = 0
    else:
        message = cycle_info_to_proto(START)
        message.cycle_start_time = "not a time"
    try:
        with pytest.raises(grpc.RpcError) as error:
            getattr(pb_grpc.CycleInfoServiceStub(channel), rpc)(message, timeout=5)
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert reader.read_cycle_infos() == []
        assert reader.read_cycle_info_updates() == []
        assert reader.read_end_reports() == []
    finally:
        channel.close()
        reader.close()
