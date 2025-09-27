# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: BSD-3-Clause
# Modifications made by NVIDIA
# This file implements a new fault-tolerant rendezvous barrier design
# that eliminates the need for compare_set operations by using atomic
# increments and a simpler key-based approach.

import inspect
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import NodeState, construct_and_record_rdzv_event
from torch.distributed.elastic.rendezvous.api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousHandler,
    RendezvousParameters,
    RendezvousTimeoutError,
)

# Try to import newer PyTorch features, fall back gracefully if not available
try:
    from torch.distributed.elastic.rendezvous.api import RendezvousInfo, RendezvousStoreInfo

    _RENDEZVOUS_INFO_AVAILABLE = True
except ImportError:
    _RENDEZVOUS_INFO_AVAILABLE = False
    RendezvousInfo = None
    RendezvousStoreInfo = None

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

from ..shared_utils.health_check import GPUHealthCheck
from ..shared_utils.profiling import ProfilingEvent, record_profiling_event
from .data import WorkloadAction
from .ipc_connector import IpcConnector
from .launcher import FT_LAUNCHER_IPC_SOCKET, UnhealthyNodeException

log = logging.getLogger(LogConfig.name)


def get_method_name(depth=2):
    if len(inspect.stack()) > depth:
        return inspect.stack()[depth].function
    return "no_method_name"


Token = Any
"""Represent an opaque fencing token used by the rendezvous backend."""


class RendezvousTimeout:
    """Hold the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        keep_alive:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """

    _ZERO = timedelta(0)

    _DEFAULT_TIMEOUTS = {
        "join": timedelta(seconds=600),
        "last_call": timedelta(seconds=30),
        "close": timedelta(seconds=30),
        "heartbeat": timedelta(seconds=5),
    }

    _join: timedelta
    _last_call: timedelta
    _close: timedelta
    _heartbeat: timedelta

    def __init__(
        self,
        join: Optional[timedelta] = None,
        last_call: Optional[timedelta] = None,
        close: Optional[timedelta] = None,
        heartbeat: Optional[timedelta] = None,
    ) -> None:
        self._set_timeouts(join=join, last_call=last_call, close=close, heartbeat=heartbeat)

    @property
    def join(self) -> timedelta:
        """Get the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        """Get the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        """Get the close timeout."""
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        """Get the keep-alive heartbeat timeout."""
        return self._heartbeat

    def _set_timeouts(self, **timeouts: Optional[timedelta]):
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f"The {name} timeout ({timeout}) must be positive.")
            setattr(self, "_" + name, timeout)


@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    """Hold the settings of the rendezvous.

    Attributes:
        run_id:
            The run id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
        keep_alive_interval:
            The amount of time a node waits before sending a heartbeat to keep
            it alive in the rendezvous.
        keep_alive_max_attempt:
            The maximum number of failed heartbeat attempts after which a node
            is considered dead.
    """

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int


@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    """Describe a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.addr}_{self.pid}_{self.local_id}"


class _NodeDescGenerator:
    """Generate node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an auto-
    incremented integer that uniquely identifies a node in the rendezvous.
    """

    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # An integer that is incremented with each call to generate().
        self._local_id = 0

    def generate(self, local_addr: Optional[str] = None) -> _NodeDesc:
        # This method can be called by multiple threads concurrently; therefore,
        # we must increment the integer atomically.
        with self._lock:
            local_id = self._local_id

            self._local_id += 1

        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)


class ParticipantStatus(Enum):
    """Participant status in the rendezvous."""

    ACTIVE = "Active"
    STANDBY = "Standby"


class GroupRankStatus(Enum):
    """Group rank status for participants."""

    UNASSIGNED = -1  # Initially unassigned
    STANDBY = -2  # Standby participant (not active)


class RendezvousParticipantInfo:
    """Participant information for storage in arrived_<count> keys.

    A rendezvous participant can be a physical node or a process in a physical node
    in the simulation case. This class provides a JSON-based format that can store:
    - NodeDesc (addr, pid, local_id)
    - Participant status (Active or Standby)

    The format is designed to support up to 4K participants efficiently.
    In future, this can be changed to Protobuf for better efficiency and performance.
    """

    @staticmethod
    def pack(node_desc: _NodeDesc, status: ParticipantStatus) -> str:
        """Pack participant information into JSON format."""
        data = {
            "addr": node_desc.addr,
            "pid": node_desc.pid,
            "local_id": node_desc.local_id,
            "status": status.value,
        }
        return json.dumps(data)

    @staticmethod
    def unpack(data: str) -> Tuple[_NodeDesc, ParticipantStatus]:
        """Unpack participant information from JSON format."""
        try:
            info = json.loads(data)
            node_desc = _NodeDesc(addr=info["addr"], pid=info["pid"], local_id=info["local_id"])
            status = ParticipantStatus(info["status"])
            return node_desc, status
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid participant info data: {e}")


class _RendezvousBarrierState:
    """Hold the state of a rendezvous barrier.

    This class manages the barrier state using atomic operations and
    simple key-based coordination instead of compare_set operations.
    """

    def __init__(
        self,
        store: Store,
        run_id: str,
        is_store_host: bool = False,
        join_timeout_seconds: float = 600.0,
    ):
        self.store = store
        self.run_id = run_id
        self.is_store_host = is_store_host
        self.join_timeout_seconds = join_timeout_seconds
        self._rendezvous_start_time = None
        self._prev_participants = {}  # Store previous round's participants and their ranks

        # Key prefixes for the barrier
        self.prefix = f"ft_rendezvous_barrier:{run_id}"
        self.arrived_count_key = f"{self.prefix}:arrived_count"
        self.last_participant_arrived_key = f"{self.prefix}:last_participant_arrived"
        self.ack_count_key = f"{self.prefix}:ack_count"
        self.closed_key = f"{self.prefix}:closed"

    @staticmethod
    def _assign_group_ranks_preserving_previous(
        participants: List[Tuple[_NodeDesc, ParticipantStatus, int]],
        prev_participants: Dict[_NodeDesc, int],
    ) -> Dict[_NodeDesc, int]:
        """Assign group ranks while preserving previous group rank assignments as much as possible.

        Args:
            participants: List of (node_desc, status, current_group_rank) tuples
            prev_participants: Dictionary of previous round's node_desc -> group_rank mapping

        Returns:
            Dictionary of node_desc -> assigned_group_rank
        """
        # Filter only active participants for group rank assignment
        active_participants = [
            (node_desc, status, current_group_rank)
            for node_desc, status, current_group_rank in participants
            if status == ParticipantStatus.ACTIVE
        ]

        world_size = len(active_participants)
        sorted_participants = sorted(active_participants, key=lambda x: x[0])  # Sort by node_desc
        free_group_ranks = set(range(world_size))
        result = {}

        # First pass: try to reuse previous group rank assignments
        for node_desc, status, current_group_rank in sorted_participants:
            prev_group_rank = prev_participants.get(node_desc, -1)
            if (
                prev_group_rank >= 0
                and prev_group_rank < world_size
                and prev_group_rank in free_group_ranks
            ):
                # This node can have the same group rank as before
                result[node_desc] = prev_group_rank
                free_group_ranks.remove(prev_group_rank)
            else:
                # Mark as unassigned for now
                result[node_desc] = -1

        # Second pass: fill gaps with remaining free group ranks
        free_group_ranks = sorted(free_group_ranks)
        for node_desc, status, current_group_rank in sorted_participants:
            if result[node_desc] < 0:
                result[node_desc] = free_group_ranks.pop(0)

        assert (
            not free_group_ranks
        ), f"Should have assigned all group ranks, but {free_group_ranks} remain"
        return result

    def _check_timeout_and_closure(self, node_desc: _NodeDesc) -> None:
        """Check for early closure and timeout, raising appropriate exceptions if detected.

        Args:
            node_desc: Node descriptor for logging

        Raises:
            RendezvousClosedError: If rendezvous was closed
            RendezvousTimeoutError: If rendezvous has timed out
        """
        # Check for early closure
        if self.is_closed():
            msg = f"The node '{node_desc}' detected that rendezvous was closed"
            log.info(msg)
            raise RendezvousClosedError(msg)

        # Check for timeout
        elapsed = time.monotonic() - self._rendezvous_start_time
        if elapsed > self.join_timeout_seconds:
            msg = (
                f"The node '{node_desc}' rendezvous has timed out after "
                f"{elapsed:.2f} seconds (timeout: {self.join_timeout_seconds}s)."
            )
            log.error(msg)
            raise RendezvousTimeoutError(msg)

    def perform_rendezvous(
        self, node_desc: _NodeDesc, min_nodes: int, max_nodes: int, last_call_timeout: timedelta
    ) -> int:
        """Perform the complete rendezvous process: join, wait for completion, acknowledge, and get rank.

        Args:
            node_desc: Node descriptor for this participant
            min_nodes: Minimum number of nodes required
            max_nodes: Maximum number of nodes allowed
            last_call_timeout: Additional wait time after min_nodes reached

        Returns:
            group_rank
        """
        # Check if rendezvous is closed
        if self.is_closed():
            raise RendezvousClosedError()

        # Record start time for timeout monitoring
        self._rendezvous_start_time = time.monotonic()

        # Step 1: Join the rendezvous and get unique identifier
        self._arrived_count = self.store.add(self.arrived_count_key, 1)

        # Store participant information in arrived_<count> key using the unique identifier
        arrived_key = f"{self.prefix}:arrived_{self._arrived_count}"
        # Participants 1 to min_nodes are active, participants min_nodes+1 to max_nodes are standby
        # Note: _arrived_count starts at 1 (not 0) due to store.add() atomic increment
        is_active = self._arrived_count <= min_nodes
        status = ParticipantStatus.ACTIVE if is_active else ParticipantStatus.STANDBY
        participant_data = RendezvousParticipantInfo.pack(node_desc, status)
        self.store.set(arrived_key, participant_data)

        # Set initial group rank
        rank_key = f"{self.prefix}:arrived_{self._arrived_count}_group_rank"
        self.store.set(rank_key, str(GroupRankStatus.UNASSIGNED.value))  # Initially unassigned

        log.debug(
            f"[{node_desc}] [Step 1] Joined rendezvous with arrived_count={self._arrived_count}, status={status.value}"
        )

        # Step 2: Wait for rendezvous completion
        last_call_deadline = None
        if self._arrived_count >= min_nodes:
            last_call_deadline = datetime.utcnow() + last_call_timeout

        while True:
            # Check for early closure and timeout
            self._check_timeout_and_closure(node_desc)

            # Check if rendezvous is already complete
            if self.store.check([self.last_participant_arrived_key]):
                break

            # Check if we should mark completion now
            should_complete = False

            if self._arrived_count >= max_nodes:
                # Max nodes reached
                should_complete = True
            elif (
                self._arrived_count >= min_nodes
                and last_call_deadline
                and datetime.utcnow() >= last_call_deadline
            ):
                # Min nodes reached and deadline passed
                should_complete = True

            if should_complete:
                # Mark rendezvous as complete
                self.store.set(self.last_participant_arrived_key, "1")
                log.debug(
                    f"[{node_desc}] [Step 2] Rendezvous marked as complete with arrived_count={self._arrived_count}"
                )
                break

            # Small delay before next check
            time.sleep(0.1)

        # Step 3: Perform two-step acknowledge phase
        # Step 3a: All participants acknowledge completion
        ack_count = self.store.add(self.ack_count_key, 1)
        log.debug(f"[{node_desc}] [Step 3a] Acknowledged completion, ack_count={ack_count}")

        # Step 3b: TCPStore host participant waits for all acknowledgments, assigns ranks, and clears keys
        if self.is_store_host:
            log.debug(
                f"[{node_desc}] [Step 3b] Store host waiting for acknowledgments and assigning ranks"
            )
            while True:
                # Check for early closure and timeout
                self._check_timeout_and_closure(node_desc)

                current_count = int(self.store.get(self.ack_count_key))
                # arrived_count_key is guaranteed to exist since we're in perform_rendezvous()
                total_participants = int(self.store.get(self.arrived_count_key))
                if current_count >= total_participants:
                    log.debug(
                        f"[{node_desc}] [Step 3b] All {total_participants} participants acknowledged (ack_count={current_count}), proceeding to rank assignment"
                    )

                    # Clear barrier keys before group rank assignments. So other
                    # launchers would not think new participants are joining during rank assignment.
                    log.debug(
                        f"[{node_desc}] [Step 3b] Clearing barrier keys before rank assignment to prevent race conditions"
                    )
                    self._clear_barrier_keys(node_desc)

                    # Assign group ranks to all participants
                    self.assign_group_ranks(min_nodes, total_participants, node_desc)
                    break

                time.sleep(0.1)
        else:
            # Non-first participants just wait for rank assignment
            pass

        # Step 4: Wait for group rank assignment
        log.debug(f"[{node_desc}] [Step 4] Waiting for group rank assignment")
        while True:
            # Check for early closure and timeout
            self._check_timeout_and_closure(node_desc)

            rank_str = self.store.get(rank_key)
            rank = int(rank_str)

            # Check if rank has been assigned (not unassigned)
            if rank != GroupRankStatus.UNASSIGNED.value:
                log.debug(f"[{node_desc}] [Step 4] Received group rank {rank}")
                return rank

            # Delay before next check
            time.sleep(1)

    def get_all_participants(
        self, total_participants: int
    ) -> List[Tuple[_NodeDesc, ParticipantStatus, int]]:
        """Get all participants that have arrived using multi_get.

        Args:
            total_participants: Total number of participants

        Returns:
            List of tuples: (node_desc, status, group_rank)
        """
        arrived_count = total_participants

        # Prepare keys for multi_get
        participant_keys = [f"{self.prefix}:arrived_{i}" for i in range(1, arrived_count + 1)]
        rank_keys = [f"{self.prefix}:arrived_{i}_group_rank" for i in range(1, arrived_count + 1)]

        # Use multi_get to fetch all data at once
        all_keys = participant_keys + rank_keys
        all_data = self.store.multi_get(all_keys)

        # Split the results
        participant_data_list = all_data[:arrived_count]
        rank_data_list = all_data[arrived_count:]

        # Unpack participant information
        participants = []
        for i in range(arrived_count):
            if participant_data_list[i] and rank_data_list[i]:
                try:
                    node_desc, status = RendezvousParticipantInfo.unpack(participant_data_list[i])
                    group_rank = int(rank_data_list[i])
                    participants.append((node_desc, status, group_rank))
                except Exception as e:
                    log.warning(f"Failed to unpack participant data for arrived_{i+1}: {e}")

        return participants

    def _clear_barrier_keys(self, node_desc: _NodeDesc):
        """Clear main barrier keys."""
        # Clear main keys - individual arrived_<count> keys don't need clearing
        keys_to_clear = [
            self.last_participant_arrived_key,
            self.arrived_count_key,
            self.ack_count_key,
        ]

        # Delete main keys
        for key in keys_to_clear:
            try:
                self.store.delete_key(key)
            except Exception as e:
                log.debug(f"[{node_desc}] Failed to delete key {key}: {e}")

        log.debug(
            f"[{node_desc}] [Step 3b] Cleared {len(keys_to_clear)} barrier keys: {keys_to_clear}"
        )

    def assign_group_ranks(
        self, min_nodes: int, total_participants: int, node_desc: _NodeDesc
    ) -> bool:
        """Assign group ranks to all participants while preserving previous assignments. Called by Rank 0 (TCPStore host).

        Args:
            min_nodes: Minimum number of active participants
            total_participants: Total number of participants (passed to avoid race condition)
        """
        all_participants = self.get_all_participants(total_participants)
        # Assert that we have participants - if arrived_count > 0, we should have participants
        assert (
            len(all_participants) > 0
        ), f"Expected participants but got empty list. total_participants={total_participants}"
        assert (
            len(all_participants) == total_participants
        ), f"Expected {total_participants} participants, got {len(all_participants)}"

        assigned_group_ranks = self._assign_group_ranks_preserving_previous(
            all_participants, self._prev_participants
        )

        # Store the assigned ranks in the store
        for i, (node_desc, status, current_rank) in enumerate(all_participants):
            rank_key = f"{self.prefix}:arrived_{i+1}_group_rank"

            if status == ParticipantStatus.ACTIVE:
                # Use preserved group rank assignment
                assigned_group_rank = assigned_group_ranks.get(node_desc, -1)
                self.store.set(rank_key, str(assigned_group_rank))
            elif status == ParticipantStatus.STANDBY:
                # Mark standby participant
                self.store.set(rank_key, str(GroupRankStatus.STANDBY.value))

        # Save current participants for next round (similar to _ft_rendezvous.py)
        self._prev_participants = assigned_group_ranks.copy()

        log.debug(
            f"[{node_desc}] [Step 3b] Assigned group ranks to {len(assigned_group_ranks)} active participants, preserving previous assignments"
        )

    def is_closed(self) -> bool:
        """Check if rendezvous is closed."""
        try:
            return self.store.check([self.closed_key])
        except Exception:
            return False

    def set_closed(self):
        """Mark rendezvous as closed."""
        try:
            self.store.set(self.closed_key, "1")
        except Exception as e:
            log.error(f"Failed to set closed: {e}")


class FtRendezvousBarrierHandler(RendezvousHandler):
    """Represent a handler that sets up a rendezvous among a set of nodes using barrier-based coordination."""

    # Static
    _node_desc_generator = _NodeDescGenerator()

    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _barrier_state: _RendezvousBarrierState

    @classmethod
    def from_backend(
        cls,
        run_id: str,
        store: Store,
        backend: Any,  # We don't use the backend in this implementation
        min_nodes: int,
        max_nodes: int,
        local_addr: Optional[str] = None,
        timeout: Optional[RendezvousTimeout] = None,
        is_store_host: bool = False,
    ):
        """Create a new :py:class:`FtRendezvousBarrierHandler`.

        Args:
            run_id:
                The run id of the rendezvous.
            store:
                The C10d store to return as part of the rendezvous.
            backend:
                The backend (not used in this implementation).
            min_nodes:
                The minimum number of nodes to admit to the rendezvous.
            max_nodes:
                The maximum number of nodes to admit to the rendezvous.
            local_addr:
                The local node address.
            timeout:
                The timeout configuration of the rendezvous.
        """
        # We associate each handler instance with a unique node descriptor.
        node = cls._node_desc_generator.generate(local_addr)

        settings = RendezvousSettings(
            run_id,
            min_nodes,
            max_nodes,
            timeout or RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=5),
            keep_alive_max_attempt=3,
        )

        return cls(node, settings, "c10d", store, is_store_host)

    def __init__(
        self,
        node: _NodeDesc,
        settings: RendezvousSettings,
        backend_name: str,
        store: Store,
        is_store_host: bool = False,
    ) -> None:
        if not settings.run_id:
            raise ValueError("The run id must be a non-empty string.")

        if settings.min_nodes < 1:
            raise ValueError(
                f"The minimum number of nodes ({settings.min_nodes}) must be greater than zero."
            )

        if settings.max_nodes < settings.min_nodes:
            raise ValueError(
                f"The maximum number of nodes ({settings.max_nodes}) must be greater than or equal "
                f"to the minimum number of nodes ({settings.min_nodes})."
            )

        self._this_node = node
        self._settings = settings
        self._backend_name = backend_name
        self._store = store
        self._barrier_state = _RendezvousBarrierState(
            store, settings.run_id, is_store_host, settings.timeout.join.total_seconds()
        )
        self._assigned_rank = None
        self._world_size = None

        self._ranks_connector = IpcConnector(FT_LAUNCHER_IPC_SOCKET)
        self._ranks_connector.start_receiving()

    def _record(
        self,
        message: str,
        node_state: NodeState = NodeState.RUNNING,
        rank: Optional[int] = None,
    ) -> None:
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._this_node.addr,
            pid=self._this_node.pid,
            local_id=self._this_node.local_id,
            rank=rank,
        )

    @property
    def settings(self) -> RendezvousSettings:
        """Get the settings of the rendezvous."""
        return self._settings

    def get_backend(self) -> str:
        """See base class."""
        return self._backend_name

    @property
    def use_agent_store(self) -> bool:
        """See base class."""
        return False

    def ensure_node_is_healthy(self) -> None:
        """Perform GPU health check for this node."""
        # Record the health check message
        msg = f"Checking health status of {self._this_node}."
        self._record(message=msg)
        # Perform GPU health check
        health_checker = GPUHealthCheck()
        try:
            health_status = health_checker()
            if not health_status:
                raise UnhealthyNodeException(f"Node {self._this_node} has an unhealthy GPU.")
        except UnhealthyNodeException as e:
            # Log specific health check failure
            log.error(f"Health check failed for node {self._this_node}: {str(e)}")
            raise
        except Exception as e:
            # General exception for unexpected issues during health check
            log.error(f"Unexpected error during health check for node {self._this_node}: {str(e)}")
            raise UnhealthyNodeException(str(e))

    def handle_control_requests_from_rank(self) -> None:
        """Check control messages received from local ranks."""
        # Check control messages received from local ranks
        excl_this_node = False
        shutdown_workload = False
        for rank, req in self._ranks_connector.fetch_received():
            log.error(f"Received request from rank={rank}: req={req}")
            if req.action == WorkloadAction.ExcludeThisNode:
                excl_this_node = True
            if req.action == WorkloadAction.ShutdownWorkload:
                shutdown_workload = True
        if shutdown_workload:
            self._close()
        if excl_this_node:
            raise UnhealthyNodeException(
                f"Node {self._this_node} is excluded from the training due to an user request."
            )

    def _perform_rendezvous(self) -> None:
        """Perform the complete rendezvous process."""
        # Perform complete rendezvous process
        group_rank = self._barrier_state.perform_rendezvous(
            self._this_node,
            self._settings.min_nodes,
            self._settings.max_nodes,
            self._settings.timeout.last_call,
        )

        # Store the assigned rank and calculate world size
        self._assigned_rank = group_rank
        self._world_size = self._settings.min_nodes  # World size = number of active participants

        if group_rank == GroupRankStatus.UNASSIGNED.value:
            log.warning("Failed to get group rank assignment, but continuing")
        elif group_rank == GroupRankStatus.STANDBY.value:
            log.info(
                f"Node {self._this_node} is in standby mode (rank {GroupRankStatus.STANDBY.value})"
            )
        else:
            log.info(f"Node {self._this_node} assigned group rank {group_rank}")

    def next_rendezvous(self) -> Union[RendezvousInfo, Tuple[Store, int, int]]:
        """See base class.

        Returns:
            RendezvousInfo object if supported by PyTorch version,
            otherwise tuple of (store, rank, world_size)
        """

        msg = (
            f"The node '{self._this_node}' attempts to join the next round of the rendezvous "
            f"'{self._settings.run_id}'."
        )
        self._record(message=msg)
        log.info(msg)

        # Record rendezvous start event
        rendezvous_start_event_id = record_profiling_event(
            ProfilingEvent.RENDEZVOUS_STARTED,
            node_id=self._this_node,
        )

        try:
            # Check node health and control requests before starting rendezvous
            self.ensure_node_is_healthy()
            self.handle_control_requests_from_rank()

            # Perform the complete rendezvous process
            self._perform_rendezvous()

            # Use stored rank and world size
            rank = self._assigned_rank
            world_size = self._world_size
            store = self._get_store()

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

        msg = (
            f"The node '{self._this_node}' has joined the rendezvous "
            f"'{self._settings.run_id}' as rank {rank} in a world of size "
            f"{world_size}."
        )
        self._record(message=msg, rank=rank)
        log.info(msg)

        # Record rendezvous completion event
        rendezvous_completion_event_id = record_profiling_event(
            ProfilingEvent.RENDEZVOUS_COMPLETED,
            node_id=self._this_node,
        )

        # Use RendezvousInfo if available (newer PyTorch versions >= 2.4.0)
        # Fall back to tuple format if RendezvousInfo is not supported
        if _RENDEZVOUS_INFO_AVAILABLE:
            # TCPStore sharing is disabled, TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1.
            # Handle backward compatibility for RendezvousStoreInfo.build
            try:
                bootstrap_store_info = RendezvousStoreInfo.build(
                    rank, store, local_addr=self._this_node.addr
                )
            except TypeError:
                # For older PyTorch versions (<= 2.5.1), local_addr parameter is not supported
                bootstrap_store_info = RendezvousStoreInfo.build(rank, store)
            return RendezvousInfo(
                store,
                rank,
                world_size,
                bootstrap_store_info,
            )
        else:
            # RendezvousInfo not supported, use tuple format
            log.debug("RendezvousInfo not available, using tuple format")
            return store, rank, world_size

    def is_closed(self) -> bool:
        """See base class."""
        try:
            return self._barrier_state.is_closed()
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def set_closed(self) -> None:
        """See base class."""
        try:
            self._close()
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def num_nodes_waiting(self) -> int:
        """See base class."""
        # Return the arrived count as the number of nodes waiting
        if not self._barrier_state.store.check([self._barrier_state.arrived_count_key]):
            return 0

        return int(self._barrier_state.store.get(self._barrier_state.arrived_count_key))

    def remove_this_node(self):
        raise NotImplementedError("Not implemented")

    def num_nodes(self) -> int:
        raise NotImplementedError("Not implemented")

    def round(self) -> int:
        raise NotImplementedError("Not implemented")

    def get_run_id(self) -> str:
        """See base class."""
        return self._settings.run_id

    def shutdown(self) -> bool:
        """See base class."""
        try:
            self._close()
            return True
        except RendezvousError as ex:
            msg = (
                f"The node '{self._this_node}' has failed to shutdown the rendezvous "
                f"'{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            )
            self._record(message=msg, node_state=NodeState.FAILED)
            log.warning(msg)
            return False
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def _close(self) -> None:
        """Close the rendezvous."""
        self._barrier_state.set_closed()

        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.info(msg)

    def _get_store(self) -> Store:
        """Get the store for this rendezvous."""
        key_prefix = f"torch.rendezvous.{self._settings.run_id}.0"

        return PrefixStore(key_prefix, self._store)

    def should_create_workers(self) -> bool:
        """Check if this node should create workers (only active nodes should)."""
        # Only active nodes (rank >= 0) should create workers
        # Standby nodes should not create workers
        return self._assigned_rank is not None and self._assigned_rank >= 0


def _get_timeout(params: RendezvousParameters, key: str) -> Optional[timedelta]:
    timeout = params.get_as_int(key + "_timeout")
    if timeout is None:
        return None
    return timedelta(seconds=timeout)


def create_handler(
    store: Store, backend: Any, params: RendezvousParameters
) -> FtRendezvousBarrierHandler:
    """Create a new :py:class:`FtRendezvousBarrierHandler` from the specified parameters.

    Args:
        store:
            The C10d store to return as part of the rendezvous.
        backend:
            The backend (not used in this implementation).

    +-------------------+------------------------------------------------------+
    | Parameter         | Description                                          |
    +===================+======================================================+
    | join_timeout      | The total time, in seconds, within which the         |
    |                   | rendezvous is expected to complete. Defaults to 600  |
    |                   | seconds.                                             |
    +-------------------+------------------------------------------------------+
    | last_call_timeout | An additional wait amount, in seconds, before        |
    |                   | completing the rendezvous once the minimum number of |
    |                   | nodes has been reached. Defaults to 30 seconds.      |
    +-------------------+------------------------------------------------------+
    | close_timeout     | The time, in seconds, within which the rendezvous is |
    |                   | expected to close after a call to                    |
    |                   | :py:meth:`RendezvousHandler.set_closed` or           |
    |                   | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |
    |                   | 30 seconds.                                          |
    +-------------------+------------------------------------------------------+
    """
    try:
        timeout = RendezvousTimeout(
            _get_timeout(params, "join"),
            _get_timeout(params, "last_call"),
            _get_timeout(params, "close"),
        )

        # Get is_store_host from parameters
        is_store_host = params.config.get('is_store_host', False)

        return FtRendezvousBarrierHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
            is_store_host=is_store_host,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
