# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gRPC transport of the cycle records: cycle infos over CycleInfoService (nvrx_interface.proto),
end reports over DataRecordAggregationService (log_aggregation.proto)."""

import logging
import queue
from concurrent import futures
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import grpc

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig
from nvidia_resiliency_ext.shared_utils.proto import log_aggregation_pb2, log_aggregation_pb2_grpc
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2_grpc as pb_grpc

from ..utils import (
    hostnames_to_slurm_nodelist,
    range_str_to_ranks,
    ranks_to_range_str,
    slurm_nodelist_to_hostnames,
    slurm_sort_addrs,
)
from .records import CycleInfo, EndReason, NodeCycleEndReport
from .transport import CycleInfoPublisher, CycleInfoReader

logger = logging.getLogger(LogConfig.name)

_END_REASON_TO_PROTO = {
    EndReason.UNSPECIFIED: pb.CycleEndReasonUnspecified,
    EndReason.COMPLETED: pb.CycleEndReasonWorkerCompleted,
    EndReason.FAILED: pb.CycleEndReasonWorkerFailed,
    EndReason.TERMINATED_DUE_TO_PEER: pb.CycleEndReasonWorkerTerminatedDueToPeer,
    EndReason.TERMINATED_BY_RANK_MONITOR: pb.CycleEndReasonWorkerTerminatedByRankMonitor,
    EndReason.TERMINATED_BY_ATTRIBUTION: pb.CycleEndReasonWorkerTerminatedByAttribution,
}
# Wire values back to records; a value unknown to this version is rejected.
_END_REASON_FROM_PROTO = {wire: reason for reason, wire in _END_REASON_TO_PROTO.items()}


def _to_iso(timestamp: float) -> str:
    """Seconds since the epoch as ISO 8601 UTC, e.g. "2026-11-13T05:55:28.123456Z"."""
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _from_iso(value: str) -> Optional[float]:
    """ISO 8601 UTC back to seconds since the epoch; None for "" (not set).

    Raises:
        ValueError: `value` is not an ISO 8601 time.
    """
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _ranks_in_host_order(nodes: tuple[str, ...], ranks: tuple[int, ...]) -> List[int]:
    """`ranks`, reordered like the compressed node list (sorted hosts), so the i-th rank
    belongs to the i-th expanded host. Launchers of one host keep their order."""
    if len(nodes) != len(ranks):
        return list(ranks)
    host_order = {node: i for i, node in enumerate(slurm_sort_addrs(list(dict.fromkeys(nodes))))}
    return [rank for _, rank in sorted(zip(nodes, ranks), key=lambda pair: host_order[pair[0]])]


def cycle_info_to_proto(info: CycleInfo) -> pb.CycleInfo:
    """Convert `info`, compressing its node and rank lists (see CycleInfo)."""
    return pb.CycleInfo(
        job_id=info.job_id,
        attempt_index=info.attempt_index,
        cycle_number=info.cycle_number,
        cycle_start_time=_to_iso(info.started_at),
        cycle_end_time=_to_iso(info.ended_at) if info.ended_at is not None else "",
        cycle_log_file=info.cycle_log_file,
        active_nodes=hostnames_to_slurm_nodelist(list(info.active_nodes)),
        standby_nodes=hostnames_to_slurm_nodelist(list(info.standby_nodes)),
        active_ranks=ranks_to_range_str(_ranks_in_host_order(info.active_nodes, info.active_ranks)),
        timestamp=info.timestamp,
    )


def cycle_info_from_proto(message: pb.CycleInfo) -> CycleInfo:
    """Convert a received cycle info back to its record, expanding the node and rank lists.

    Raises:
        ValueError: A time or a list is malformed, or the start time is missing.
    """
    started_at = _from_iso(message.cycle_start_time)
    if started_at is None:
        raise ValueError(f"cycle info of cycle {message.cycle_number} has no start time")
    return CycleInfo(
        job_id=message.job_id,
        attempt_index=message.attempt_index,
        cycle_number=message.cycle_number,
        started_at=started_at,
        ended_at=_from_iso(message.cycle_end_time),
        cycle_log_file=message.cycle_log_file,
        active_nodes=tuple(slurm_nodelist_to_hostnames(message.active_nodes)),
        active_ranks=tuple(range_str_to_ranks(message.active_ranks)),
        standby_nodes=tuple(slurm_nodelist_to_hostnames(message.standby_nodes)),
        timestamp=message.timestamp,
    )


def end_report_to_proto(report: NodeCycleEndReport) -> pb.NodeCycleEndReport:
    """Convert `report`.

    Raises:
        KeyError: Its end reason has no wire value.
    """
    return pb.NodeCycleEndReport(
        job_id=report.job_id,
        attempt_index=report.attempt_index,
        cycle_number=report.cycle_number,
        node_name=report.node_name,
        end_reason=_END_REASON_TO_PROTO[report.end_reason],
        detected_timestamp=report.detected_timestamp,
        exit_details=report.exit_details,
    )


def end_report_from_proto(message: pb.NodeCycleEndReport) -> NodeCycleEndReport:
    """Convert a received end report back to its record.

    Raises:
        ValueError: The end reason is unknown to this version, or the detection time is
            missing (0).
    """
    try:
        end_reason = _END_REASON_FROM_PROTO[message.end_reason]
    except KeyError as e:
        raise ValueError(f"unknown end reason {e} in an end report") from e
    if not message.detected_timestamp:
        raise ValueError(f"end report of cycle {message.cycle_number} has no detection time")
    return NodeCycleEndReport(
        job_id=message.job_id,
        attempt_index=message.attempt_index,
        cycle_number=message.cycle_number,
        node_name=message.node_name,
        end_reason=end_reason,
        detected_timestamp=message.detected_timestamp,
        exit_details=message.exit_details,
    )


def _end_report_stream(report: NodeCycleEndReport) -> Iterator[log_aggregation_pb2.DataRecord]:
    """`report` as a StreamDataRecords request: a stream of one DataRecord.

    Raises:
        KeyError: Its end reason has no wire value.
    """
    record = log_aggregation_pb2.DataRecord(node_cycle_end_report=end_report_to_proto(report))
    return iter([record])


class GrpcCycleInfoPublisher(CycleInfoPublisher):
    """Publishes each record with a blocking call, on the stub of its destination.

    - Cycle infos go to the CycleAnalyzer's CycleInfoService (`cycle_info_stub`): the
      store host publishes them directly to the analyzer.
    - End reports go to a log server's DataRecordAggregationService (`data_record_stub`):
      launchers publish them to the server they stream logs to, as a one-record stream.
      The call returns once that server queued the report, not once the analyzer has it.

    Pass the stub of each record the caller publishes; publishing a record without its
    stub is logged and dropped. A publish returns once the record was accepted, or after
    `rpc_timeout`; failures are logged and the record dropped, never raised. The caller
    owns the stubs' channels.
    """

    def __init__(
        self,
        *,
        cycle_info_stub: Optional[pb_grpc.CycleInfoServiceStub] = None,
        data_record_stub: Optional[
            log_aggregation_pb2_grpc.DataRecordAggregationServiceStub
        ] = None,
        rpc_timeout: float = 5.0,
    ):
        self._cycle_info_stub = cycle_info_stub
        self._data_record_stub = data_record_stub
        self._rpc_timeout = rpc_timeout

    def publish_cycle_info(self, info: CycleInfo) -> bool:
        what = f"start of cycle {info.cycle_number}"
        stub = self._cycle_info_stub
        rpc = stub.PublishCycleInfo if stub is not None else None
        return self._publish(rpc, cycle_info_to_proto, info, what)

    def update_cycle_info(self, info: CycleInfo) -> bool:
        what = f"end of cycle {info.cycle_number}"
        stub = self._cycle_info_stub
        rpc = stub.UpdateCycleInfo if stub is not None else None
        return self._publish(rpc, cycle_info_to_proto, info, what)

    def publish_end_report(self, report: NodeCycleEndReport) -> bool:
        what = f"end report of cycle {report.cycle_number}"
        stub = self._data_record_stub
        rpc = stub.StreamDataRecords if stub is not None else None
        return self._publish(rpc, _end_report_stream, report, what)

    def close(self) -> None:
        """Nothing to release: calls complete before a publish returns."""

    def _publish(self, rpc, encode, record, what: str) -> bool:
        if rpc is None:
            logger.warning(f"Cannot publish the {what}: no stub for its destination")
            return False
        try:
            request = encode(record)
        except Exception as e:
            logger.warning(f"Failed to encode the {what}: {e!r}")
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


def _drain(records: queue.SimpleQueue) -> List:
    drained = []
    while True:
        try:
            drained.append(records.get_nowait())
        except queue.Empty:
            return drained


class _ReaderServicer(pb_grpc.CycleInfoServiceServicer):
    """Decodes each received record on the server's threads and queues it for reading."""

    def __init__(
        self,
        cycle_infos: queue.SimpleQueue,
        cycle_info_updates: queue.SimpleQueue,
        end_reports: queue.SimpleQueue,
    ):
        self._cycle_infos = cycle_infos
        self._cycle_info_updates = cycle_info_updates
        self._end_reports = end_reports

    def PublishCycleInfo(self, request, context):
        return self._accept(cycle_info_from_proto, request, self._cycle_infos, context)

    def UpdateCycleInfo(self, request, context):
        return self._accept(cycle_info_from_proto, request, self._cycle_info_updates, context)

    def PublishNodeCycleEndReport(self, request, context):
        return self._accept(end_report_from_proto, request, self._end_reports, context)

    @staticmethod
    def _accept(decode, request, records: queue.SimpleQueue, context):
        try:
            records.put(decode(request))
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        return pb.PublishReply()


class GrpcCycleInfoReader(CycleInfoReader):
    """Receives the records as a CycleInfoService listening on `address`.

    Records are decoded on the gRPC server's threads and queued until read, so reading
    never waits on the network. A record that cannot be decoded is rejected with
    INVALID_ARGUMENT; its sender logs the failure.

    Args:
        address: gRPC listen address: ``unix:/path`` for a Unix domain socket, or
            ``host:port`` for TCP (port 0 picks a free one, see `port`).
        max_workers: Server threads; handlers only decode and queue.

    Raises:
        RuntimeError: The address cannot be bound.
    """

    def __init__(self, address: str, max_workers: int = 4):
        self._cycle_infos: queue.SimpleQueue = queue.SimpleQueue()
        self._cycle_info_updates: queue.SimpleQueue = queue.SimpleQueue()
        self._end_reports: queue.SimpleQueue = queue.SimpleQueue()
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        pb_grpc.add_CycleInfoServiceServicer_to_server(
            _ReaderServicer(self._cycle_infos, self._cycle_info_updates, self._end_reports),
            self._server,
        )
        self.port = self._server.add_insecure_port(address)
        self._server.start()

    def read_cycle_infos(self) -> List[CycleInfo]:
        return _drain(self._cycle_infos)

    def read_cycle_info_updates(self) -> List[CycleInfo]:
        return _drain(self._cycle_info_updates)

    def read_end_reports(self) -> List[NodeCycleEndReport]:
        return _drain(self._end_reports)

    def close(self, grace: float = 1.0) -> None:
        """Stop the server, letting calls in progress finish for up to `grace` seconds."""
        self._server.stop(grace=grace).wait()
