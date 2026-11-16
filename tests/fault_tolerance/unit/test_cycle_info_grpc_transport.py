# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info import grpc_transport
from nvidia_resiliency_ext.fault_tolerance.cycle_info.grpc_transport import (
    GrpcCycleInfoPublisher,
    GrpcCycleInfoReader,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.messages import (
    build_cycle_info,
    ended_cycle_info,
)
from nvidia_resiliency_ext.shared_utils.proto import log_aggregation_pb2, log_aggregation_pb2_grpc
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2_grpc as pb_grpc

START = build_cycle_info(
    job_id="job1",
    attempt_index=2,
    cycle_number=3,
    started_at_ms=1_700_000_000_500,
    cycle_log_file="/logs/train_cycle3.log",
    active_nodes=("node002.cluster", "node001.cluster"),
    active_ranks=(0, 1),
    standby_nodes=("node003.cluster",),
)
END = ended_cycle_info(START, 1_700_000_100_250)
REPORT = pb.NodeCycleEndReport(
    job_id="job1",
    attempt_index=2,
    cycle_number=3,
    node_name="node001.cluster",
    end_reason=pb.CycleEndReasonWorkerTerminatedByRankMonitor,
    detected_timestamp=1_700_000_099_123,
    exit_details='[{"rank":1}]',
)


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

        # Sent as given, without a conversion.
        stub.PublishCycleInfo.assert_called_once_with(START, timeout=2.0)
        stub.UpdateCycleInfo.assert_called_once_with(END, timeout=2.0)
        assert stub.PublishCycleInfo.call_args.args[0] is START

    def test_end_report_is_a_one_record_stream(self):
        stub = MagicMock()
        publisher = GrpcCycleInfoPublisher(data_record_stub=stub, rpc_timeout=2.0)

        assert publisher.publish_end_report(REPORT) is True

        (requests,), kwargs = stub.StreamDataRecords.call_args
        assert kwargs == {"timeout": 2.0}
        assert list(requests) == [log_aggregation_pb2.DataRecord(node_cycle_end_report=REPORT)]

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

    def test_record_failing_to_build_its_request_is_logged_without_raising(self):
        stub = MagicMock()
        publisher = GrpcCycleInfoPublisher(data_record_stub=stub)
        with patch.object(grpc_transport, "logger") as logger:
            # A cycle info where an end report is expected cannot go into a DataRecord.
            assert publisher.publish_end_report(START) is False
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
        assert record.node_cycle_end_report == REPORT
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
        stub.PublishNodeCycleEndReport(REPORT, timeout=5)

        assert reader.read_cycle_infos() == [START]
        assert reader.read_cycle_info_updates() == [END]
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
def test_reader_rejects_a_record_failing_its_check(rpc):
    reader = GrpcCycleInfoReader("127.0.0.1:0")
    channel = grpc.insecure_channel(f"127.0.0.1:{reader.port}")
    if rpc == "PublishNodeCycleEndReport":
        message = pb.NodeCycleEndReport()
        message.CopyFrom(REPORT)
        message.detected_timestamp = 0
    else:
        message = pb.CycleInfo()
        message.CopyFrom(START)
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
