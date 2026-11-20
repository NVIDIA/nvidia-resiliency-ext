# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gRPC transport of the cycle records: cycle infos over CycleInfoService (nvrx_interface.proto),
end reports as records of the launchers' log streams (DataRecordAggregationService,
data_records_aggregation.proto)."""

import logging
import queue
from concurrent import futures
from typing import Callable, List, Optional

import grpc

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig
from nvidia_resiliency_ext.shared_utils.proto import data_records_aggregation_pb2
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2_grpc as pb_grpc

from .messages import check_cycle_info, check_node_cycle_end_report
from .transport import CycleInfoPublisher, CycleInfoReader

logger = logging.getLogger(LogConfig.name)


class GrpcCycleInfoPublisher(CycleInfoPublisher):
    """Publishes the cycle infos with a blocking call to the CycleAnalyzer's CycleInfoService
    (`cycle_info_stub`): the store host publishes them directly to the analyzer.

    A publish returns once the analyzer accepted the record, or after `rpc_timeout`;
    failures are logged and the record dropped, never raised. The caller owns the stub's
    channel. End reports are not published here: launchers send them on their log stream
    (LogStreamEndReportPublisher).
    """

    def __init__(
        self,
        *,
        cycle_info_stub: Optional[pb_grpc.CycleInfoServiceStub] = None,
        rpc_timeout: float = 5.0,
    ):
        self._cycle_info_stub = cycle_info_stub
        self._rpc_timeout = rpc_timeout

    def publish_cycle_info(self, info: pb.NVRxCycleInfo) -> bool:
        what = f"start of cycle {info.cycle_number}"
        stub = self._cycle_info_stub
        rpc = stub.PublishCycleInfo if stub is not None else None
        return self._publish(rpc, lambda: info, what)

    def update_cycle_info(self, info: pb.NVRxCycleInfo) -> bool:
        what = f"end of cycle {info.cycle_number}"
        stub = self._cycle_info_stub
        rpc = stub.UpdateCycleInfo if stub is not None else None
        return self._publish(rpc, lambda: info, what)

    def publish_end_report(self, report: pb.NodeCycleEndReport) -> bool:
        logger.warning(
            f"Cannot publish the end report of cycle {report.cycle_number}: end reports are "
            "sent on the launchers' log streams"
        )
        return False

    def close(self) -> None:
        """Nothing to release: calls complete before a publish returns."""

    def _publish(self, rpc, build_request, what: str) -> bool:
        if rpc is None:
            logger.warning(f"Cannot publish the {what}: no stub for its destination")
            return False
        try:
            request = build_request()
        except Exception as e:  # e.g. a record of the wrong message type
            logger.warning(f"Failed to build the request of the {what}: {e!r}")
            return False
        try:
            rpc(request, timeout=self._rpc_timeout)
            return True
        except grpc.RpcError as e:
            # The gRPC core's own account of the failure, e.g. a reset stream or a GOAWAY;
            # details() alone can be as terse as "CANCELLED".
            debug = e.debug_error_string() if hasattr(e, "debug_error_string") else ""
            logger.warning(
                f"Failed to publish the {what}: {e.code().name}: {e.details()} ({debug})"
            )
        except Exception as e:  # e.g. the caller closed the channel
            logger.warning(f"Failed to publish the {what}: {e!r}")
        return False


class LogStreamEndReportPublisher(CycleInfoPublisher):
    """Publishes a launcher's end reports as records of its gRPC log stream
    (StreamDataRecords to its leaf or root log server, which forward them to the analyzer).

    No call per report, so no server thread per report: the record is queued and sent
    ahead of the queued log chunks, by the thread that streams the logs. A publish only
    queues and never blocks; flush() waits until the queued reports were handed to the
    stream. Best effort, like the log chunks: a report is lost if the stream breaks after it
    was handed to gRPC; while the stream is down it waits for the next one.

    Args:
        put_record: Queues a DataRecord on the log stream; False when it cannot (logs are
            not forwarded over gRPC). E.g. PipeBasedLogsSpecs.publish_record.
        wait_for_records: Waits up to a timeout until the queued records were handed to the
            stream; True if none is left. E.g. PipeBasedLogsSpecs.wait_for_records.
    """

    def __init__(
        self,
        put_record: Callable[[data_records_aggregation_pb2.DataRecord], bool],
        wait_for_records: Callable[[float], bool],
    ):
        self._put_record = put_record
        self._wait_for_records = wait_for_records

    def publish_cycle_info(self, info: pb.NVRxCycleInfo) -> bool:
        logger.debug(f"Not publishing the start of cycle {info.cycle_number}: not a store host")
        return False

    def update_cycle_info(self, info: pb.NVRxCycleInfo) -> bool:
        logger.debug(f"Not publishing the end of cycle {info.cycle_number}: not a store host")
        return False

    def publish_end_report(self, report: pb.NodeCycleEndReport) -> bool:
        try:
            if self._put_record(
                data_records_aggregation_pb2.DataRecord(node_cycle_end_report=report)
            ):
                return True
            logger.warning(
                f"Cannot publish the end report of cycle {report.cycle_number}: logs are not "
                "forwarded over gRPC"
            )
        except Exception as e:
            logger.warning(
                f"Failed to publish the end report of cycle {report.cycle_number}: {e!r}"
            )
        return False

    def flush(self, timeout: float) -> bool:
        try:
            return self._wait_for_records(timeout)
        except Exception as e:
            logger.warning(f"Failed to wait for the end reports to be sent: {e!r}")
            return False


def _drain(records: queue.SimpleQueue) -> List:
    drained = []
    while True:
        try:
            drained.append(records.get_nowait())
        except queue.Empty:
            return drained


class _ReaderServicer(pb_grpc.CycleInfoServiceServicer):
    """Checks each received record on the server's threads and queues it for reading."""

    def __init__(self, cycle_infos: queue.SimpleQueue, end_reports: queue.SimpleQueue):
        self._cycle_infos = cycle_infos
        self._end_reports = end_reports

    def PublishCycleInfo(self, request, context):
        return self._accept(check_cycle_info, request, self._cycle_infos, context)

    def UpdateCycleInfo(self, request, context):
        return self._accept(check_cycle_info, request, self._cycle_infos, context)

    def PublishNodeCycleEndReport(self, request, context):
        return self._accept(check_node_cycle_end_report, request, self._end_reports, context)

    @staticmethod
    def _accept(check, request, records: queue.SimpleQueue, context):
        try:
            check(request)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return pb.PublishReply()  # abort() raises; older grpcio may return instead
        records.put(request)  # the received message itself; gRPC does not use it afterwards
        return pb.PublishReply()


class GrpcCycleInfoReader(CycleInfoReader):
    """Receives the records as a CycleInfoService listening on `address`.

    Records are checked on the gRPC server's threads and queued until read, so reading
    never waits on the network. A record that fails its check (see messages.py) is
    rejected with INVALID_ARGUMENT; its sender logs the failure.

    Args:
        address: gRPC listen address: ``unix:/path`` for a Unix domain socket, or
            ``host:port`` for TCP (port 0 picks a free one, see `port`).
        max_workers: Server threads; handlers only check and queue.

    Raises:
        RuntimeError: The address cannot be bound.
    """

    def __init__(self, address: str, max_workers: int = 4):
        self._cycle_infos: queue.SimpleQueue = queue.SimpleQueue()
        self._end_reports: queue.SimpleQueue = queue.SimpleQueue()
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        pb_grpc.add_CycleInfoServiceServicer_to_server(
            _ReaderServicer(self._cycle_infos, self._end_reports), self._server
        )
        self.port = self._server.add_insecure_port(address)
        self._server.start()

    def read_cycle_infos(self) -> List[pb.NVRxCycleInfo]:
        return _drain(self._cycle_infos)

    def read_end_reports(self) -> List[pb.NodeCycleEndReport]:
        return _drain(self._end_reports)

    def close(self, grace: float = 1.0) -> None:
        """Stop the server, letting calls in progress finish for up to `grace` seconds."""
        self._server.stop(grace=grace).wait()
