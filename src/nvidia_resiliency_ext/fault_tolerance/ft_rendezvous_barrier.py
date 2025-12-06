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

from ..shared_utils.health_check import GPUHealthCheck, InfraNodeHealthCheck
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
        use_infra_group_rank:
            Whether to use infrastructure group rank for rank assignment instead of
            arrival-based assignment. If True, ranks are read from SLURM_PROCID (in SLURM
            environments) or GROUP_RANK (set by launcher) environment variables.
    """

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int
    use_infra_group_rank: bool = True


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


class GroupRankStatus(Enum):
    """Group rank status for participants."""

    UNASSIGNED = -1  # Initially unassigned


class RendezvousParticipantInfo:
    """Participant information for storage in arrived_<count> keys.

    A rendezvous participant can be a physical node or a process in a physical node
    in the simulation case. This class provides a JSON-based format that can store:
    - NodeDesc (addr, pid, local_id)
    - Infrastructure rank (optional, used when use_infra_group_rank is enabled)

    The format is designed to support up to 4K participants efficiently.
    In future, this can be changed to Protobuf for better efficiency and performance.
    """

    @staticmethod
    def pack(node_desc: _NodeDesc, infra_rank: int = -1) -> str:
        """Pack participant information into JSON format."""
        data = {
            "addr": node_desc.addr,
            "pid": node_desc.pid,
            "local_id": node_desc.local_id,
            "infra_rank": infra_rank,
        }
        return json.dumps(data)

    @staticmethod
    def unpack(data: str) -> Tuple[_NodeDesc, int]:
        """Unpack participant information from JSON format.

        Returns:
            Tuple of (node_desc, infra_rank)
        """
        try:
            info = json.loads(data)
            node_desc = _NodeDesc(addr=info["addr"], pid=info["pid"], local_id=info["local_id"])
            # Support old format without infra_rank field
            infra_rank = info.get("infra_rank", -1)
            return node_desc, infra_rank
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid participant info data: {e}")


class _RendezvousBarrierState:
    """Hold the state of a rendezvous barrier.

    This class manages the barrier state using atomic operations and
    simple key-based coordination instead of compare_set operations.

    DESIGN PHILOSOPHY:
    This implementation prioritizes flexibility and fault tolerance over strict
    synchronization. Key design principles:

    1. ATOMIC BARRIER: Uses atomic increments instead of compare-and-swap operations
       for better performance and simpler coordination

    2. GRACEFUL COMPLETION: Completes rendezvous on last_call_timeout rather than
       waiting for max_nodes, enabling hot-fix scenarios and flexible scaling

    3. EVENTUAL CONVERGENCE: New comers trigger restarts of existing participants,
       ensuring all nodes eventually participate in the same rendezvous round

    4. RACE CONDITION TOLERANCE: The last_call_timeout provides a grace period
       for restarting nodes, preventing premature completion

    This design trades some synchronization guarantees for better fault tolerance
    and operational flexibility in production environments.
    """

    def __init__(
        self,
        store: Store,
        run_id: str,
        is_store_host: bool = False,
        join_timeout_seconds: float = 600.0,
        use_infra_group_rank: bool = True,
    ):
        self.store = store
        self.run_id = run_id
        self.is_store_host = is_store_host
        self.join_timeout_seconds = join_timeout_seconds
        self.use_infra_group_rank = use_infra_group_rank
        self._rendezvous_start_time = None
        self._prev_participants = {}  # Store previous round's participants and their ranks
        self._attempted_open = False  # Track if this node tried to open rendezvous

        # Key prefixes for the barrier
        self.prefix = f"ft_rendezvous_barrier:{run_id}"
        self.arrived_count_key = f"{self.prefix}:arrived_count"
        self.last_participant_arrived_key = f"{self.prefix}:last_participant_arrived"
        self.ack_count_key = f"{self.prefix}:ack_count"
        self.closed_key = f"{self.prefix}:closed"
        self.unhealthy_count_key = f"{self.prefix}:unhealthy_count"
        self.peer_aborted_count_key = f"{self.prefix}:peer_aborted_count"

        # Initialize last_participant_arrived_key to 0 (open) if it doesn't exist
        # This key is always present and serves as the open/close indicator:
        #   0 = rendezvous is OPEN (accepting new participants)
        #   1 = rendezvous is CLOSED (training in progress, hot spares should wait)
        if not self.store.check([self.last_participant_arrived_key]):
            self.store.set(self.last_participant_arrived_key, "0".encode('utf-8'))

    def _assign_group_ranks(
        self,
        participants: List[Tuple[_NodeDesc, int]],
        prev_participants: Dict[_NodeDesc, int],
        min_nodes: int,
    ) -> Dict[_NodeDesc, int]:
        """Assign group ranks to participants, preserving previous assignments when possible.

        Args:
            participants: List of (node_desc, infra_rank) tuples
            prev_participants: Dictionary of previous round's node_desc -> group_rank mapping
            min_nodes: Minimum number of active nodes

        Returns:
            Dictionary of node_desc -> assigned_group_rank
        """
        if self.use_infra_group_rank:
            return self._assign_group_ranks_with_infra_rank(participants, min_nodes)

        # Original logic for non-infrastructure rank mode (arrival-based assignment)
        # First min_nodes arrivals are active (ranks 0 to min_nodes-1)
        # Later arrivals are standby (sequential ranks starting from min_nodes)
        # Note: len(participants) >= min_nodes is guaranteed by rendezvous completion logic

        result = {}
        active_count = min_nodes

        # Sort all participants by node_desc for deterministic assignment
        sorted_participants = sorted(participants, key=lambda x: x[0])

        # Pass 1: Try to preserve previous ranks for active participants (first min_nodes)
        free_ranks = set(range(active_count))
        for idx, (node_desc, _) in enumerate(sorted_participants):
            if idx < active_count:
                # Active participant - try to reuse previous rank
                prev_rank = prev_participants.get(node_desc, -1)
                if 0 <= prev_rank < active_count and prev_rank in free_ranks:
                    result[node_desc] = prev_rank
                    free_ranks.remove(prev_rank)
                else:
                    result[node_desc] = -1  # Assign later
            else:
                # Standby participant - sequential assignment
                result[node_desc] = idx

        # Pass 2: Assign free ranks to unassigned active participants
        if free_ranks:
            free_ranks = sorted(free_ranks)
            for idx, (node_desc, _) in enumerate(sorted_participants[:active_count]):
                if result[node_desc] == -1:
                    result[node_desc] = free_ranks.pop(0)

        standby_count = len(sorted_participants) - active_count
        log.debug(
            f"Assigned group_ranks by arrival order: "
            f"{active_count} active, {standby_count} standby"
        )

        return result

    def _assign_group_ranks_with_infra_rank(
        self,
        participants: List[Tuple[_NodeDesc, int]],
        min_nodes: int,
    ) -> Dict[_NodeDesc, int]:
        """Assign group ranks using infrastructure ranks with hardware failure resilience.

        Core principle: group_rank = infra_rank (strict deterministic mapping)
        Exception: Fill gaps in [0, min_nodes) caused by HW failures using spare nodes

        Example with HW failure:
            min_nodes=5, arrivals: infra_ranks [0, 1, 3, 4, 10, 11]  (rank 2 missing due to HW failure)
            - Node(infra=0) → group_rank=0  (direct mapping, primary active)
            - Node(infra=1) → group_rank=1  (direct mapping, primary active)
            - Node(infra=3) → group_rank=3  (direct mapping, primary active)
            - Node(infra=4) → group_rank=4  (direct mapping, primary active)
            - Node(infra=10) → group_rank=2 (promoted to fill gap at rank 2)
            - Node(infra=11) → group_rank=11 (direct mapping, unpromoted spare)

        Args:
            participants: List of (node_desc, infra_rank) tuples
            min_nodes: Minimum number of active nodes

        Returns:
            Dictionary of node_desc -> assigned_group_rank

        Raises:
            RuntimeError: If insufficient spare nodes to fill gaps in [0, min_nodes)
            ValueError: If duplicate infra_ranks detected
        """
        result = {}
        infra_ranks_seen = set()
        gaps = set(range(min_nodes))  # Start with all primary ranks as potential gaps
        spare_nodes = []  # Collect spare nodes during iteration

        # Single pass: Assign ranks, detect duplicates, collect spares, build gaps
        for node_desc, infra_rank in participants:
            # Validate infrastructure rank
            if infra_rank < 0:
                raise ValueError(
                    f"Invalid infrastructure rank {infra_rank} for participant {node_desc}. "
                    f"Expected non-negative integer from SLURM_PROCID or GROUP_RANK."
                )

            # Check for duplicate infra_ranks (deployment error)
            if infra_rank in infra_ranks_seen:
                raise RuntimeError(
                    f"Duplicate infrastructure rank {infra_rank} detected. "
                    f"This indicates a deployment error - each node must have a unique "
                    f"SLURM_PROCID or GROUP_RANK value."
                )
            infra_ranks_seen.add(infra_rank)

            # Default assignment: group_rank = infra_rank (direct mapping)
            result[node_desc] = infra_rank

            if infra_rank < min_nodes:
                gaps.discard(infra_rank)  # Remove from gaps as we see primary ranks
            else:
                spare_nodes.append((node_desc, infra_rank))

        # Log summary instead of per-participant logs to avoid spam with large participant counts
        primary_count = min_nodes - len(gaps)
        log.debug(
            f"Assigned group_ranks using infra_rank: "
            f"{primary_count} primary active, {len(spare_nodes)} spare, {len(gaps)} gaps"
        )

        # Handle spare nodes: fill gaps (if any) and ensure contiguous group_rank assignment
        if spare_nodes:
            # Sort spare nodes by infra_rank for deterministic promotion
            spare_nodes.sort(key=lambda x: x[1])

            # Sort gaps for deterministic processing
            gaps = sorted(gaps)

            if gaps:
                # Defensive check: This should never happen because rendezvous guarantees
                # total_participants >= min_nodes, which means spare_nodes >= gaps
                # Proof: P + S >= min_nodes => S >= min_nodes - P => S >= gaps
                assert len(gaps) <= len(spare_nodes), (
                    f"INTERNAL ERROR: Insufficient spare nodes to fill gaps. "
                    f"gaps={len(gaps)} at ranks {gaps}, spare_nodes={len(spare_nodes)}. "
                    f"This should never happen if rendezvous guarantees total_participants >= min_nodes."
                )

                # Promote spare nodes to fill gaps
                for gap_rank, (node_desc, original_infra_rank) in zip(gaps, spare_nodes):
                    result[node_desc] = gap_rank
                    log.warning(
                        f"PROMOTED spare node {node_desc} (infra_rank={original_infra_rank}) to "
                        f"group_rank={gap_rank} to fill gap caused by hardware failure"
                    )
                # Note: Unpromoted spare nodes keep their original infra_rank as group_rank
                # They were already assigned result[node_desc] = infra_rank in the first pass

        return result

    def _get_unhealthy_count(self) -> int:
        """Get the current unhealthy node count.

        Returns:
            The number of unhealthy nodes reported so far, or 0 if the key doesn't exist
        """
        if not self.store.check([self.unhealthy_count_key]):
            return 0

        unhealthy_count_bytes = self.store.get(self.unhealthy_count_key)
        return int(unhealthy_count_bytes.decode('utf-8'))

    def _increment_peer_aborted_count(self) -> int:
        """Increment the peer aborted count to signal a restart.

        This is called by a launcher when it detects local worker failure and decides
        to restart, notifying other healthy nodes to restart as well for faster
        failure propagation.

        Returns:
            The new peer aborted count after incrementing
        """
        new_count = self.store.add(self.peer_aborted_count_key, 1)
        return new_count

    def _get_peer_aborted_count(self) -> int:
        """Get the current peer aborted count.

        This count tracks how many peers have detected local worker failures and
        decided to restart, enabling faster failure propagation across the cluster.

        Returns:
            The number of peers that have aborted so far, or 0 if the key doesn't exist
        """
        if not self.store.check([self.peer_aborted_count_key]):
            return 0

        peer_aborted_count_bytes = self.store.get(self.peer_aborted_count_key)
        return int(peer_aborted_count_bytes.decode('utf-8'))

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

    def _wait_for_rendezvous_open(self, node_desc: _NodeDesc) -> None:
        """Step 0: Wait if rendezvous is closed. Hot spares will wait here.

        This prevents hot spares that arrive late from disrupting ongoing training.
        The rendezvous is considered:
        - OPEN (value=0): Accepting new participants for a new rendezvous round
        - CLOSED (value=1): Training in progress, wait for next round

        Args:
            node_desc: Node descriptor for logging

        Raises:
            RendezvousClosedError: If rendezvous is explicitly closed (not just waiting)

        Note:
            This wait does NOT timeout unless the rendezvous is explicitly closed.
            Hot spares can wait indefinitely until a failure opens a new round.

            DEADLOCK PREVENTION:
            If this node previously tried to open but was deferred (because arrived_count_key
            existed), it will retry opening periodically. This prevents deadlock where:
            1. Node fails after training starts
            2. Tries to open but defers (Step 3b in progress)
            3. Store host clears keys (slow)
            4. Node stuck waiting, but arrived_count_key now gone
            Without retry, no one would know to open for this node.
        """
        wait_count = 0
        logged_waiting = False  # Track if we've logged the waiting message

        while True:
            # Check for explicit closure (permanent shutdown)
            if self.is_closed():
                msg = f"The node '{node_desc}' detected that rendezvous was closed"
                log.info(msg)
                raise RendezvousClosedError(msg)

            # Read the open/close indicator
            # The key is always present (initialized in __init__)
            value_bytes = self.store.get(self.last_participant_arrived_key)
            value = int(value_bytes.decode('utf-8'))

            if value == 0:
                # Rendezvous is open, can proceed
                log.debug(
                    f"[{node_desc}] [Step 0] Rendezvous is open (value=0), proceeding to join"
                )
                break

            # value == 1: Rendezvous is closed (training in progress)
            # Log once when we first start waiting, then every 60 seconds
            if not logged_waiting:
                log.info(
                    f"[{node_desc}] [Step 0] Rendezvous closed (training in progress), waiting..."
                )
                logged_waiting = True
            elif wait_count % 60 == 0:  # Log every 60 seconds
                log.debug(
                    f"[{node_desc}] [Step 0] Still waiting for rendezvous to open "
                    f"(waited {wait_count} seconds)"
                )

            # Retry opening every 10 seconds ONLY if we previously attempted to open
            # This distinguishes between:
            # - Failed nodes that tried to open but were deferred → SHOULD retry
            # - Hot spares that never tried to open → SHOULD NOT retry (would disrupt training!)
            wait_count += 1
            if self._attempted_open and wait_count % 10 == 0:  # Every 10 seconds
                log.debug(
                    f"[{node_desc}] [Step 0] Retrying open_rendezvous() "
                    f"(this node previously attempted to open and was deferred)"
                )
                self.open_rendezvous()
                # If open succeeds, value will be 0 on next iteration and we'll break
                # If still deferred, we keep waiting

            time.sleep(1.0)  # Poll every 1 second

    def perform_rendezvous(
        self, node_desc: _NodeDesc, min_nodes: int, max_nodes: int, last_call_timeout: timedelta
    ) -> Tuple[int, int]:
        """Perform the complete rendezvous process: join, wait for completion, acknowledge, and get rank.

        DESIGN RATIONALE:
        This atomic barrier-based rendezvous design balances flexibility with convergence guarantees:

        1. FLEXIBILITY: We use last_call_timeout (not max_nodes) as completion criteria because:
           - Future hot-fix scenarios may need to exclude broken participants without failing
           - We cannot wait for max_nodes (active + standby) as this would block legitimate completions
           - The system needs to be resilient to partial failures while maintaining progress

        2. CONVERGENCE: The rendezvous will complete when either:
           - max_nodes is reached (immediate completion), OR
           - min_nodes is reached AND last_call_timeout expires (graceful completion)

        3. NEW COMER HANDLING:
           - New comers arriving during active rendezvous participate in the current round
           - New comers arriving after completion trigger the next rendezvous round
           - All existing participants detect new comers and restart to join the new round
           - This ensures eventual convergence with the new comer

        4. RACE CONDITION MITIGATION:
           - The last_call_timeout provides a grace period for nodes that are restarting
           - This prevents premature completion when nodes are in the process of joining
           - The timeout balances responsiveness with inclusion of restarting nodes

        5. HOT SPARE HANDLING (Step 0):
           - Hot spares arriving after training starts will wait at Step 0
           - They wait until a failure opens the rendezvous (value=0)
           - This prevents disruption of ongoing training

        Args:
            node_desc: Node descriptor for this participant
            min_nodes: Minimum number of nodes required for training to proceed
            max_nodes: Maximum number of nodes allowed (active + standby)
            last_call_timeout: Grace period after min_nodes reached to allow restarting nodes to join

        Returns:
            Tuple of (group_rank, total_participants)
        """
        # Step 0: Wait if rendezvous is closed (training in progress)
        # Hot spares arriving late will wait here until a failure opens a new round
        # Note: This also checks for explicit closure (is_closed()), no need to check again
        self._wait_for_rendezvous_open(node_desc)

        # Record start time for timeout monitoring
        # Start timing AFTER Step 0 completes, since hot spares may wait indefinitely at Step 0
        self._rendezvous_start_time = time.monotonic()

        # Step 1: Join the rendezvous and get unique identifier
        self._arrived_count = self.store.add(self.arrived_count_key, 1)

        # Check if we exceed max_nodes (can happen due to race conditions or user misconfiguration)
        if self._arrived_count > max_nodes:
            msg = (
                f"Maximum number of nodes ({max_nodes}) exceeded. "
                f"Participant count: {self._arrived_count}. "
                f"This is likely a configuration error - please check max_nodes setting."
            )
            log.error(f"[{node_desc}] {msg}")
            # Set closed to notify other nodes before exiting
            self.set_closed()
            raise RendezvousClosedError(msg)

        # Determine infrastructure rank
        infra_rank = -1
        if self.use_infra_group_rank:
            # Try SLURM_PROCID first (set by SLURM), then fall back to GROUP_RANK (set by launcher)
            infra_rank_str = os.getenv('SLURM_PROCID', os.getenv('GROUP_RANK', '-1'))
            infra_rank = int(infra_rank_str)
            if infra_rank < 0:
                raise ValueError(
                    "use_infra_group_rank is enabled but neither SLURM_PROCID nor GROUP_RANK "
                    "environment variable is set. Please set one of these environment variables "
                    "or disable use_infra_group_rank."
                )
            log.debug(f"[{node_desc}] Using infrastructure rank {infra_rank} from environment")

        # Store participant information in arrived_<count> key using the unique identifier
        arrived_key = f"{self.prefix}:arrived_{self._arrived_count}"
        participant_data = RendezvousParticipantInfo.pack(node_desc, infra_rank)
        self.store.set(arrived_key, participant_data)

        # Set initial group rank (unassigned)
        rank_key = f"{self.prefix}:arrived_{self._arrived_count}_group_rank"
        # Initially unassigned - use format "unassigned,0" for consistency
        self.store.set(rank_key, f"{GroupRankStatus.UNASSIGNED.value},0".encode('utf-8'))

        log.debug(
            f"[{node_desc}] [Step 1] Joined rendezvous with arrived_count={self._arrived_count}"
        )

        # Step 2: Wait for rendezvous completion
        last_call_deadline = None
        if self._arrived_count >= min_nodes:
            last_call_deadline = datetime.utcnow() + last_call_timeout

        while True:
            # Check for early closure and timeout
            self._check_timeout_and_closure(node_desc)

            # STORE HOST: Check if too many nodes are unhealthy (early termination condition)
            # If unhealthy_count > max_nodes - min_nodes, then it's mathematically impossible
            # to complete the rendezvous with min_nodes participants
            if self.is_store_host:
                unhealthy_count = self._get_unhealthy_count()
                if unhealthy_count > (max_nodes - min_nodes):
                    msg = (
                        f"Rendezvous cannot complete: {unhealthy_count} unhealthy nodes detected, "
                        f"max possible healthy nodes = {max_nodes - unhealthy_count} < {min_nodes} required. "
                        f"Closing rendezvous to terminate the job."
                    )
                    log.error(msg)
                    self.set_closed()
                    # Continue to next iteration to detect closure immediately.
                    continue

            # Check if rendezvous is already complete
            # IMPORTANT: We check the VALUE, not just existence, since the key is always present
            # Value "1" means complete, value "0" means open/in-progress
            value_bytes = self.store.get(self.last_participant_arrived_key)
            value = int(value_bytes.decode('utf-8'))
            if value == 1:
                log.debug(
                    f"[{node_desc}] [Step 2] Detected rendezvous completion (last_participant_arrived_key=1)"
                )
                break

            # Check if we should mark completion now
            # DESIGN NOTE: We complete on last_call_timeout (not max_nodes) to support:
            # 1. Hot-fix scenarios where broken participants are excluded without failing the job
            # 2. Flexibility in participant count while ensuring convergence
            # 3. Grace period for restarting nodes to join before completion
            should_complete = False

            if self._arrived_count >= max_nodes:
                # Max nodes reached - immediate completion
                should_complete = True
            elif (
                self._arrived_count >= min_nodes
                and last_call_deadline
                and datetime.utcnow() >= last_call_deadline
            ):
                # Min nodes reached and grace period expired - graceful completion
                # This allows the system to proceed even if not all potential nodes have joined
                should_complete = True

            if should_complete:
                # Mark rendezvous as complete
                self.store.set(self.last_participant_arrived_key, "1".encode('utf-8'))
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
            while True:
                # Check for early closure and timeout
                self._check_timeout_and_closure(node_desc)

                current_count = int(self.store.get(self.ack_count_key))
                # arrived_count_key is guaranteed to exist since we're in perform_rendezvous()
                total_participants = int(self.store.get(self.arrived_count_key))
                if current_count >= total_participants:
                    log.debug(
                        f"[{node_desc}] [Step 3b] All {total_participants} participants acknowledged (ack_count={current_count}), proceeding to clear keys and assign ranks"
                    )

                    # Clear barrier keys first to prevent false positives in launcher's
                    # num_nodes_waiting() detection. If arrived_count_key persists after
                    # participants return to training, launcher may incorrectly detect a
                    # new rendezvous and trigger unnecessary restarts.
                    self._clear_barrier_keys(node_desc)

                    # Assign group ranks after clearing keys
                    # New comers are blocked at Step 0 by closed rendezvous, so no race condition
                    self.assign_group_ranks(min_nodes, max_nodes, total_participants, node_desc)
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

            rank_value_bytes = self.store.get(rank_key)
            rank_value = rank_value_bytes.decode('utf-8')

            # Parse the combined rank value: "group_rank,total_participants"
            try:
                rank_str, total_participants_str = rank_value.split(',', 1)
                rank = int(rank_str)
                total_participants = int(total_participants_str)
            except (ValueError, AttributeError) as e:
                raise RuntimeError(
                    f"[{node_desc}] Failed to parse rank value '{rank_value}': {e}. "
                    f"Expected format 'group_rank,total_participants' but got malformed data."
                )

            # Check if rank has been assigned (not unassigned)
            if rank != GroupRankStatus.UNASSIGNED.value:
                log.debug(
                    f"[{node_desc}] [Step 4] Received group rank {rank}, total participants {total_participants}"
                )
                # Reset the attempted_open flag after successful rendezvous completion
                # This ensures the flag only applies to the current rendezvous attempt
                # On the next rendezvous, if this node is late, it should behave like a hot spare
                self._attempted_open = False
                return rank, total_participants

            # Delay before next check
            time.sleep(1)

    def get_all_participants(self, total_participants: int) -> List[Tuple[_NodeDesc, int]]:
        """Get all participants that have arrived using multi_get.

        Args:
            total_participants: Total number of participants

        Returns:
            List of tuples: (node_desc, infra_rank)
        """
        arrived_count = total_participants

        # Prepare keys for multi_get
        participant_keys = [f"{self.prefix}:arrived_{i}" for i in range(1, arrived_count + 1)]

        # Use multi_get to fetch all data at once
        participant_data_list = self.store.multi_get(participant_keys)

        # Unpack participant information
        participants = []
        for i in range(arrived_count):
            if participant_data_list[i]:
                try:
                    # Handle bytes to string conversion
                    participant_data = participant_data_list[i].decode('utf-8')
                    node_desc, infra_rank = RendezvousParticipantInfo.unpack(participant_data)
                    participants.append((node_desc, infra_rank))
                except Exception as e:
                    log.warning(f"Failed to unpack participant data for arrived_{i+1}: {e}")

        return participants

    def _clear_barrier_keys(self, node_desc: _NodeDesc):
        """Clear main barrier keys to prepare for next rendezvous round.

        Note: We do NOT clear last_participant_arrived_key because it serves as the
        open/close indicator for the rendezvous. It remains set to 1 (closed) during
        training and is reset to 0 (open) by the launcher when a failure is detected.

        This method is called by the last participant to acknowledge in Step 3b, BEFORE
        rank assignment. This ensures all participants are still waiting and cannot
        start the next cycle yet, making it safe to clear the global counters.
        """
        # Clear main keys - individual arrived_<count> keys don't need clearing
        # DO NOT clear last_participant_arrived_key - it indicates open/close state
        keys_to_clear = [
            self.arrived_count_key,
            self.ack_count_key,
            self.unhealthy_count_key,  # Clear unhealthy counter for next round
            self.peer_aborted_count_key,  # Clear peer aborted counter for next round
        ]

        # Delete main keys
        for key in keys_to_clear:
            try:
                self.store.delete_key(key)
            except Exception as e:
                log.debug(f"[{node_desc}] Failed to delete key {key}: {e}")

    def assign_group_ranks(
        self, min_nodes: int, max_nodes: int, total_participants: int, node_desc: _NodeDesc
    ) -> bool:
        """Assign group ranks to all participants while preserving previous assignments. Called by Rank 0 (TCPStore host).

        Args:
            min_nodes: Minimum number of active participants
            max_nodes: Maximum number of participants allowed
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

        # Ensure we don't exceed max_nodes
        if total_participants > max_nodes:
            # This indicates a deployment error - fail fast with clear message
            raise RuntimeError(
                f"Total participants ({total_participants}) exceeds max_nodes ({max_nodes}). "
                f"This indicates a deployment/configuration error. Common causes:\n"
                f"  - SLURM is launching more instances than configured (check SLURM job configuration)\n"
                f"  - Multiple ft_launcher processes running on the same node\n"
                f"  - Mismatch between --max-nodes setting and actual node count\n"
                f"Please verify your deployment configuration and ensure max_nodes matches "
                f"the actual number of nodes being launched."
            )

        assigned_group_ranks = self._assign_group_ranks(
            all_participants, self._prev_participants, min_nodes
        )

        # Store the assigned ranks and total participants in the store
        for i, (node_desc_item, infra_rank) in enumerate(all_participants):
            rank_key = f"{self.prefix}:arrived_{i+1}_group_rank"
            assigned_group_rank = assigned_group_ranks.get(node_desc_item, -1)

            # Ensure every participant gets a valid rank assignment
            if assigned_group_rank == -1:
                raise RuntimeError(
                    f"Failed to assign group rank to participant {node_desc_item}. "
                    f"This should never happen - all participants should be assigned ranks."
                )

            # Store both group_rank and total_participants in the rank key
            # Format: "group_rank,total_participants"
            rank_value = f"{assigned_group_rank},{total_participants}"
            self.store.set(rank_key, rank_value.encode('utf-8'))

        # Save current participants for next round
        self._prev_participants = assigned_group_ranks.copy()

        log.debug(
            f"[{node_desc}] [Step 3b] Assigned group ranks to {len(assigned_group_ranks)} participants, preserving previous assignments"
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
            self.store.set(self.closed_key, "1".encode('utf-8'))
        except Exception as e:
            log.error(f"Failed to set closed: {e}")

    def open_rendezvous(self):
        """Open rendezvous for new participants to join (typically after failure detection).

        This is called by the launcher when it detects a worker failure and needs to
        restart. Setting last_participant_arrived_key to 0 signals to all participants
        (including hot spares waiting at Step 0) that a new rendezvous round can begin.

        RACE CONDITION PROTECTION:
        We check if a rendezvous is currently in progress or being finalized.
        If arrived_count_key exists, it means:
        - Either a rendezvous is actively in progress (Step 1-2), OR
        - Step 3b is in progress but hasn't cleared the keys yet
        In this case, we DO NOT modify last_participant_arrived_key to avoid
        racing with the active rendezvous.

        BEHAVIOR AFTER DEFER:
        The calling node will proceed to Step 0 and check last_participant_arrived_key:
        - If value=0 (rendezvous still open/joining), node can join the current rendezvous
        - If value=1 (rendezvous complete):
          * If this node called open_rendezvous() → will retry opening every 10 seconds
          * If this node never called open_rendezvous() (hot spare) → will wait passively
        """
        # Mark that this node attempted to open (for retry logic in Step 0)
        self._attempted_open = True

        # Check if a rendezvous is in progress or being finalized
        # If arrived_count_key exists, it means participants are joining (Step 1) or
        # finalizing (Step 3b before clearing). The key only exists with value >= 1.
        if self.store.check([self.arrived_count_key]):
            log.debug(
                "open_rendezvous() deferred - rendezvous in progress. "
                "Not modifying last_participant_arrived_key to avoid race with active rendezvous. "
                "The node may still join the current rendezvous if it's still open (value=0)."
            )
            # Don't modify last_participant_arrived_key - let the current rendezvous manage it
            # The calling node will check the value at Step 0:
            # - If value=0 (still joining), the node can join the current rendezvous
            # - If value=1 (completed), the node will wait for the next round
            return

        # Safe to open - no rendezvous in progress
        self.store.set(self.last_participant_arrived_key, "0".encode('utf-8'))
        log.debug(f"Opened rendezvous for new round (set {self.last_participant_arrived_key}=0)")


class FtRendezvousBarrierHandler(RendezvousHandler):
    """Represent a handler that sets up a rendezvous among a set of nodes using barrier-based coordination."""

    # Static
    _node_desc_generator = _NodeDescGenerator()

    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _barrier_state: _RendezvousBarrierState
    _worker_group: Optional[Any] = None  # Store reference to worker group

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
        use_infra_group_rank: bool = True,
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
            is_store_host:
                Whether this node is the TCPStore host.
            use_infra_group_rank:
                Whether to use infrastructure group rank for rank assignment.
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
            use_infra_group_rank=use_infra_group_rank,
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
            store,
            settings.run_id,
            is_store_host,
            settings.timeout.join.total_seconds(),
            settings.use_infra_group_rank,
        )
        self._assigned_rank = None
        self._world_size = None

        self._ranks_connector = IpcConnector(FT_LAUNCHER_IPC_SOCKET)
        self._ranks_connector.start_receiving()

    def set_worker_group(self, worker_group: Any) -> None:
        """Set the worker group reference for this handler."""
        self._worker_group = worker_group

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

        # Perform GPU and Infra node health checks
        health_checker = GPUHealthCheck()
        _infrahc_socket = os.environ.get("INFRAHCD_SOCKET") or os.environ.get("INFRAHC_SOCKET")
        infrahc_checker = (
            InfraNodeHealthCheck(socket_path=_infrahc_socket)
            if _infrahc_socket
            else InfraNodeHealthCheck()
        )
        try:
            health_status = health_checker()
            infrahc_status = infrahc_checker()
        except Exception as e:
            # Unexpected error during health check
            self._barrier_state.store.add(self._barrier_state.unhealthy_count_key, 1)
            log.error(f"Unexpected error during health check for node {self._this_node}: {str(e)}")
            raise UnhealthyNodeException(str(e)) from e

        if not health_status:
            # Health check failed
            self._barrier_state.store.add(self._barrier_state.unhealthy_count_key, 1)
            log.error(f"Health check failed for node {self._this_node}: Node has an unhealthy GPU.")
            raise UnhealthyNodeException(f"Node {self._this_node} has an unhealthy GPU.")
        if not infrahc_status:
            # Infra node health check failed
            self._barrier_state.store.add(self._barrier_state.unhealthy_count_key, 1)
            log.error(
                f"Health check failed for node {self._this_node}: Infra node health check reported unhealthy."
            )
            raise UnhealthyNodeException(
                f"Node {self._this_node} failed Infra node health check."
            )

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
            # Report unhealthy to the store before raising exception
            self._barrier_state.store.add(self._barrier_state.unhealthy_count_key, 1)
            raise UnhealthyNodeException(
                f"Node {self._this_node} is excluded from the training due to an user request."
            )

    def _perform_rendezvous(self) -> None:
        """Perform the complete rendezvous process."""
        # Perform complete rendezvous process
        group_rank, total_participants = self._barrier_state.perform_rendezvous(
            self._this_node,
            self._settings.min_nodes,
            self._settings.max_nodes,
            self._settings.timeout.last_call,
        )

        # Store the assigned rank and world size
        self._assigned_rank = group_rank

        # World size should be the total number of participants (active + standby)
        # This represents the total number of groups in the distributed system
        # PyTorch will calculate global_world_size = sum of local_world_size across all groups
        # Standby participants will report local_world_size = 0
        self._world_size = total_participants

        if group_rank == GroupRankStatus.UNASSIGNED.value:
            log.warning("Failed to get group rank assignment, but continuing")
        elif group_rank >= self._settings.min_nodes:
            log.info(f"Node {self._this_node} is in standby mode (rank {group_rank})")
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

            # If this is a standby participant, modify the worker group's local_world_size
            if self._worker_group is not None and rank >= self._settings.min_nodes:
                # This is a standby participant, set local_world_size to 0
                self._worker_group.spec.local_world_size = 0
                log.info(f"Set local_world_size to 0 for standby participant with rank {rank}")

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

        arrived_count_bytes = self._barrier_state.store.get(self._barrier_state.arrived_count_key)
        arrived_count_str = arrived_count_bytes.decode('utf-8')
        return int(arrived_count_str)

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
    | use_infra_group_  | Whether to use infrastructure group rank for rank    |
    | rank              | assignment. Defaults to True.                        |
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
        use_infra_group_rank = params.config.get('use_infra_group_rank', True)

        return FtRendezvousBarrierHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
            is_store_host=is_store_host,
            use_infra_group_rank=use_infra_group_rank,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
