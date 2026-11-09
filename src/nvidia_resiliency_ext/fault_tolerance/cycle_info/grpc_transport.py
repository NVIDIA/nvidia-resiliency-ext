# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gRPC transport of the cycle records, over CycleInfoService (cycle_info.proto)."""

import logging

from nvidia_resiliency_ext.shared_utils.grpc_async_sender import AsyncUnarySender
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
    TimeoutKind,
)
from .transport import CycleInfoPublisher

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


class GrpcCycleInfoPublisher(CycleInfoPublisher):
    """Publishes each record with an asynchronous unary call on a CycleInfoService stub.

    A publish converts the record and starts the call without waiting for it (see
    `AsyncUnarySender`): failures are logged and the record dropped, and at most
    `max_pending` calls are in flight. The caller owns the stub's channel.
    """

    def __init__(
        self,
        stub: pb_grpc.CycleInfoServiceStub,
        rpc_timeout: float = 5.0,
        max_pending: int = 64,
        close_timeout: float = 1.0,
    ):
        self._stub = stub
        self._sender = AsyncUnarySender(
            logger,
            rpc_timeout=rpc_timeout,
            max_pending=max_pending,
            close_timeout=close_timeout,
        )

    def publish_cycle_start(self, info: CycleStartInfo) -> None:
        what = f"start of cycle {info.cycle}"
        try:
            request = cycle_start_to_proto(info)
        except Exception as e:
            logger.warning(f"Failed to encode the {what}: {e!r}")
            return
        self._sender.send(self._stub.PublishCycleStart, request, what)

    def publish_outcome(self, outcome: LocalCycleOutcome) -> None:
        what = f"outcome of cycle {outcome.cycle}"
        try:
            request = outcome_to_proto(outcome)
        except Exception as e:
            logger.warning(f"Failed to encode the {what}: {e!r}")
            return
        self._sender.send(self._stub.PublishCycleOutcome, request, what)

    def close(self) -> None:
        """Wait up to `close_timeout` for pending calls, then cancel the rest."""
        self._sender.close()
