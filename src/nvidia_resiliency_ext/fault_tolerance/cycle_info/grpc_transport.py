# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gRPC transport of the cycle records, over CycleInfoService (cycle_info.proto)."""

import logging
import queue
from concurrent import futures
from typing import List

import grpc

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig
from nvidia_resiliency_ext.shared_utils.proto import cycle_info_pb2 as pb
from nvidia_resiliency_ext.shared_utils.proto import cycle_info_pb2_grpc as pb_grpc

from .records import (
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
from .transport import CycleInfoPublisher, CycleInfoReader

logger = logging.getLogger(LogConfig.name)

_CYCLE_END_TO_PROTO = {
    CycleEndReason.COMPLETED: pb.CYCLE_END_REASON_COMPLETED,
    CycleEndReason.FAILED: pb.CYCLE_END_REASON_FAILED,
    CycleEndReason.PEER_RESTART: pb.CYCLE_END_REASON_PEER_RESTART,
    CycleEndReason.TERMINATED: pb.CYCLE_END_REASON_TERMINATED,
}
_RANK_END_TO_PROTO = {
    RankEndReason.COMPLETED: pb.RANK_END_REASON_COMPLETED,
    RankEndReason.FAILED: pb.RANK_END_REASON_FAILED,
    RankEndReason.STOPPED_BY_LAUNCHER: pb.RANK_END_REASON_STOPPED_BY_LAUNCHER,
    RankEndReason.STOPPED_BY_RMON: pb.RANK_END_REASON_STOPPED_BY_RANK_MONITOR,
}
_TIMEOUT_KIND_TO_PROTO = {
    None: pb.TIMEOUT_KIND_UNSPECIFIED,
    TimeoutKind.HEARTBEAT: pb.TIMEOUT_KIND_HEARTBEAT,
    TimeoutKind.SECTION: pb.TIMEOUT_KIND_SECTION,
    TimeoutKind.OUT_OF_SECTION: pb.TIMEOUT_KIND_OUT_OF_SECTION,
}
# Wire values back to records; an unspecified reason is not in these, so it is rejected.
_CYCLE_END_FROM_PROTO = {wire: reason for reason, wire in _CYCLE_END_TO_PROTO.items()}
_RANK_END_FROM_PROTO = {wire: reason for reason, wire in _RANK_END_TO_PROTO.items()}
_TIMEOUT_KIND_FROM_PROTO = {wire: kind for kind, wire in _TIMEOUT_KIND_TO_PROTO.items()}


def cycle_start_to_proto(info: CycleStartInfo) -> pb.CycleStart:
    return pb.CycleStart(
        cycle=info.cycle,
        started_at=info.started_at,
        active_nodes=info.active_nodes,
        standby_nodes=info.standby_nodes,
        active_ranks=info.active_ranks,
        cycle_log_file=info.cycle_log_file,
    )


def outcome_to_proto(outcome: LocalCycleOutcome) -> pb.CycleOutcome:
    """Convert `outcome`, bounding its workload-provided strings (see records.MAX_*).

    Raises:
        KeyError: An enum member has no wire value.
    """
    message = pb.CycleOutcome(
        cycle=outcome.cycle,
        node=outcome.node,
        node_end=_CYCLE_END_TO_PROTO[outcome.node_end],
        detected_at=outcome.detected_at,
    )
    for rank in outcome.ranks:
        rank_message = message.ranks.add(
            global_rank=rank.global_rank,
            local_rank=rank.local_rank,
            pid=rank.pid,
            end=_RANK_END_TO_PROTO[rank.end],
            exit_details=rank.exitdetails[:MAX_EXIT_DETAILS_LEN],
            timeout_kind=_TIMEOUT_KIND_TO_PROTO[rank.timeout_kind],
            timeout_sections=[
                name[:MAX_SECTION_NAME_LEN] for name in rank.timeout_sections[:MAX_TIMEOUT_SECTIONS]
            ],
        )
        # Optional on the wire: unset means the worker had not exited.
        if rank.exitcode is not None:
            rank_message.exitcode = rank.exitcode
    return message


def cycle_start_from_proto(message: pb.CycleStart) -> CycleStartInfo:
    return CycleStartInfo(
        cycle=message.cycle,
        started_at=message.started_at,
        active_nodes=tuple(message.active_nodes),
        standby_nodes=tuple(message.standby_nodes),
        active_ranks=tuple(message.active_ranks),
        cycle_log_file=message.cycle_log_file,
    )


def outcome_from_proto(message: pb.CycleOutcome) -> LocalCycleOutcome:
    """Convert a received outcome back to its record.

    Raises:
        ValueError: A reason is unspecified or unknown to this version.
    """
    try:
        ranks = tuple(
            RankOutcome(
                global_rank=rank.global_rank,
                local_rank=rank.local_rank,
                pid=rank.pid,
                end=_RANK_END_FROM_PROTO[rank.end],
                exitcode=rank.exitcode if rank.HasField("exitcode") else None,
                exitdetails=rank.exit_details,
                timeout_kind=_TIMEOUT_KIND_FROM_PROTO[rank.timeout_kind],
                timeout_sections=tuple(rank.timeout_sections),
            )
            for rank in message.ranks
        )
        node_end = _CYCLE_END_FROM_PROTO[message.node_end]
    except KeyError as e:
        raise ValueError(f"unspecified or unknown enum value {e} in a cycle outcome") from e
    return LocalCycleOutcome(
        cycle=message.cycle,
        node=message.node,
        node_end=node_end,
        detected_at=message.detected_at,
        ranks=ranks,
    )


class GrpcCycleInfoPublisher(CycleInfoPublisher):
    """Publishes each record with a blocking unary call on a CycleInfoService stub.

    A publish returns once the record has been forwarded all the way to the receiver, or
    after `rpc_timeout`; failures are logged and the record dropped, never raised. The
    caller owns the stub's channel.
    """

    def __init__(self, stub: pb_grpc.CycleInfoServiceStub, rpc_timeout: float = 5.0):
        self._stub = stub
        self._rpc_timeout = rpc_timeout

    def publish_cycle_start(self, info: CycleStartInfo) -> None:
        what = f"start of cycle {info.cycle}"
        self._publish(self._stub.PublishCycleStart, cycle_start_to_proto, info, what)

    def publish_outcome(self, outcome: LocalCycleOutcome) -> None:
        what = f"outcome of cycle {outcome.cycle}"
        self._publish(self._stub.PublishCycleOutcome, outcome_to_proto, outcome, what)

    def close(self) -> None:
        """Nothing to release: calls complete before a publish returns."""

    def _publish(self, rpc, encode, record, what: str) -> None:
        try:
            request = encode(record)
        except Exception as e:
            logger.warning(f"Failed to encode the {what}: {e!r}")
            return
        try:
            rpc(request, timeout=self._rpc_timeout)
        except grpc.RpcError as e:
            logger.warning(f"Failed to publish the {what}: {e.code().name}: {e.details()}")
        except Exception as e:  # e.g. the caller closed the channel
            logger.warning(f"Failed to publish the {what}: {e!r}")


def _drain(records: queue.SimpleQueue) -> List:
    drained = []
    while True:
        try:
            drained.append(records.get_nowait())
        except queue.Empty:
            return drained


class _ReaderServicer(pb_grpc.CycleInfoServiceServicer):
    """Decodes each received record on the server's threads and queues it for reading."""

    def __init__(self, starts: queue.SimpleQueue, outcomes: queue.SimpleQueue):
        self._starts = starts
        self._outcomes = outcomes

    def PublishCycleStart(self, request, context):
        return self._accept(cycle_start_from_proto, request, self._starts, context)

    def PublishCycleOutcome(self, request, context):
        return self._accept(outcome_from_proto, request, self._outcomes, context)

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
        self._starts: queue.SimpleQueue = queue.SimpleQueue()
        self._outcomes: queue.SimpleQueue = queue.SimpleQueue()
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        pb_grpc.add_CycleInfoServiceServicer_to_server(
            _ReaderServicer(self._starts, self._outcomes), self._server
        )
        self.port = self._server.add_insecure_port(address)
        self._server.start()

    def read_cycle_starts(self) -> List[CycleStartInfo]:
        return _drain(self._starts)

    def read_outcomes(self) -> List[LocalCycleOutcome]:
        return _drain(self._outcomes)

    def close(self, grace: float = 1.0) -> None:
        """Stop the server, letting calls in progress finish for up to `grace` seconds."""
        self._server.stop(grace=grace).wait()
