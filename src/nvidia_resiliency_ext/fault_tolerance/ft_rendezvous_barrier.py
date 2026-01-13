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
from collections import defaultdict
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

from ..inprocess.utils import format_rank_set_verbose
from ..shared_utils.health_check import (
    DistributedStorageHealthCheck,
    GPUHealthCheck,
    NicLinkStateHealthCheck,
    StoragePathHealthCheck,
)
from ..shared_utils.profiling import ProfilingEvent, record_profiling_event, set_profiling_cycle
from .data import WorkloadAction
from .ipc_connector import IpcConnector
from .launcher import FT_LAUNCHER_IPC_SOCKET, UnhealthyNodeException, get_node_health_check
from .utils import get_infrastructure_rank, is_slurm_job_array

# Conditionally import health check injector for testing/debugging
# This only activates if NVRX_INJECT_GPU_FAILURE environment variable is set
if os.environ.get("NVRX_INJECT_GPU_FAILURE"):
    from ..testing_utils import health_check_injector

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
            rendezvous has the minimum number of required participants. Note: this
            only applies to SUBSEQUENT rendezvous after the first one. The first
            rendezvous always waits for max_nodes.
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
        "last_call": timedelta(seconds=10),
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
        segment:
            Number of nodes to select from each domain. None disables segment awareness.
        nproc_per_node:
            Number of processes per node (local_world_size). Used to restore local_world_size
            when a standby node becomes active.
    """

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int
    segment: Optional[int] = None
    nproc_per_node: int = 1  # Default to 1 if not specified


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
    - Infrastructure rank (from SLURM_PROCID/GROUP_RANK or deterministically assigned)
    - Domain ID (from node name or ClusterUUID, "none" if segment not configured)

    The format is designed to support up to 4K participants efficiently.
    In future, this can be changed to Protobuf for better efficiency and performance.
    """

    @staticmethod
    def pack(node_desc: _NodeDesc, infra_rank: int = -1, domain_id: str = "none") -> str:
        """Pack participant information into JSON format.

        Args:
            node_desc: Node descriptor
            infra_rank: Infrastructure rank (-1 if not assigned)
            domain_id: Domain ID string, or "none" if segment not configured
        """
        data = {
            "addr": node_desc.addr,
            "pid": node_desc.pid,
            "local_id": node_desc.local_id,
            "infra_rank": infra_rank,
            "domain_id": domain_id,
        }
        return json.dumps(data)

    @staticmethod
    def unpack(data: str) -> Tuple[_NodeDesc, int, str]:
        """Unpack participant information from JSON format.

        Returns:
            Tuple of (node_desc, infra_rank, domain_id)
            domain_id will be "none" if not present (backward compatibility)
        """
        try:
            info = json.loads(data)
            node_desc = _NodeDesc(addr=info["addr"], pid=info["pid"], local_id=info["local_id"])
            # Support old format without infra_rank field
            infra_rank = info.get("infra_rank", -1)
            # Support old format without domain_id field - default to "none"
            domain_id = info.get("domain_id", "none")
            return node_desc, infra_rank, domain_id
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid participant info data: {e}")


def _parse_domain_id_from_nvidia_smi() -> str:
    """Parse domain ID from GPU using nvidia-smi to query ClusterUUID.

    The ClusterUUID serves as the domain identifier.
    All GPUs in the same NVLink domain share the same ClusterUUID.

    Returns:
        The ClusterUUID as the domain ID string.

    Raises:
        RuntimeError: If ClusterUUID cannot be retrieved.

    Example:
        >>> domain_id = _parse_domain_id_from_nvidia_smi()
        >>> # domain_id is "abc9829a-d4c8-491c-8da5-ad28fb34876b"
    """
    import subprocess

    try:
        # Run nvidia-smi to query ClusterUUID
        result = subprocess.run(
            ['nvidia-smi', '-q'],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi command failed with return code {result.returncode}. "
                f"stderr: {result.stderr}"
            )

        # Parse output to find ClusterUUID
        cluster_uuid = None
        for line in result.stdout.split('\n'):
            if 'ClusterUUID' in line:
                # Format: "        ClusterUUID                       : abc9829a-d4c8-491c-8da5-ad28fb34876b"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    cluster_uuid = parts[1].strip()
                    break

        if not cluster_uuid:
            raise RuntimeError(
                "ClusterUUID not found in nvidia-smi output. "
                "This may indicate that GPU fabric (NVLink domain) is not configured on this system. "
                "ClusterUUID is only available on systems with NVSwitch/NVLink fabrics (e.g., DGX, HGX systems)."
            )

        log.debug(f"Retrieved domain ID from nvidia-smi ClusterUUID: {cluster_uuid}")
        return cluster_uuid

    except FileNotFoundError:
        raise RuntimeError("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi command timed out after 10 seconds")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Failed to query domain ID from nvidia-smi: {e}")


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
        segment: Optional[int] = None,
    ):
        self.store = store
        self.run_id = run_id
        self.is_store_host = is_store_host
        self.join_timeout_seconds = join_timeout_seconds
        self.segment = segment
        self._rendezvous_start_time = None
        self._attempted_open = False  # Track if this node tried to open rendezvous

        # Cache for domain_id to avoid re-parsing on every rendezvous
        # Only populated if segment is configured
        self._cached_domain_id: Optional[str] = None

        # Key prefixes for the barrier
        self.prefix = f"ft_rendezvous_barrier:{run_id}"
        self.arrived_count_key = f"{self.prefix}:arrived_count"
        self.last_participant_arrived_key = f"{self.prefix}:last_participant_arrived"
        self.ack_count_key = f"{self.prefix}:ack_count"
        self.closed_key = f"{self.prefix}:closed"
        self.unhealthy_count_key = f"{self.prefix}:unhealthy_count"
        self.peer_aborted_count_key = f"{self.prefix}:peer_aborted_count"
        self.global_cycle_key = f"{self.prefix}:global_cycle"

        # Initialize last_participant_arrived_key to 0 (open) if it doesn't exist
        # This key is always present and serves as the open/close indicator:
        #   0 = rendezvous is OPEN (accepting new participants)
        #   1 = rendezvous is CLOSED (training in progress, hot spares should wait)
        if not self.store.check([self.last_participant_arrived_key]):
            self.store.set(self.last_participant_arrived_key, "0".encode('utf-8'))

    def _assign_group_ranks(
        self,
        participants: List[Tuple[_NodeDesc, int, str]],
        world_size: int,
    ) -> Dict[_NodeDesc, int]:
        """Assign group ranks using infrastructure ranks with hardware failure resilience.

        Simplified unified logic (works for both segment=None and segment-aware cases):
        1. Group participants by domain
        2. Sort domains by SLURM topology (smallest infra_rank in each domain)
        3. Single pass walk through sorted domains:
           - Get complete segments from each domain
           - Assign active ranks [0..world_size) directly
           - Assign standby ranks [world_size..) directly
        4. Treat segment=None as segment=1 (every node is its own segment)

        Args:
            participants: List of (node_desc, infra_rank, domain_id) tuples
                         domain_id is "none" if segment not configured
            world_size: Target number of active nodes (training world size)

        Returns:
            Dictionary of node_desc -> assigned_group_rank

        Raises:
            RuntimeError: If insufficient nodes available
            ValueError: If duplicate infra_ranks or world_size not divisible by segment
        """
        # Treat segment=None as segment=1
        segment = self.segment if self.segment is not None else 1
        if world_size % segment != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by segment ({segment})")

        # Handle unassigned infra_ranks and sort by infra_rank (SLURM topology order)
        has_unassigned = any(infra_rank == -1 for _, infra_rank, _ in participants)
        if has_unassigned:
            # Sort by node_desc and assign sequential infra_ranks [0, 1, 2, ...]
            # No need to sort again - already in infra_rank order
            # No duplicates possible since we assign sequential indices
            sorted_by_node = sorted(participants, key=lambda x: x[0])
            sorted_participants = [
                (node_desc, idx, domain_id)
                for idx, (node_desc, _, domain_id) in enumerate(sorted_by_node)
            ]
        else:
            # Sort by existing infra_rank
            sorted_participants = sorted(participants, key=lambda x: x[1])

            # Check for duplicate infra_ranks early (deployment error)
            infra_ranks = {infra_rank for _, infra_rank, _ in sorted_participants}
            if len(infra_ranks) != len(sorted_participants):
                raise RuntimeError(
                    "Duplicate infrastructure ranks detected. "
                    "Each node must have a unique SLURM_PROCID or GROUP_RANK."
                )

        # Step 1: Group by domain
        # When segment is None, treat each node as its own domain (non-segmented deployment)
        if self.segment is not None:
            domain_to_participants: Dict[str, List[Tuple[_NodeDesc, int]]] = defaultdict(list)
            for node_desc, infra_rank, domain_id in sorted_participants:
                # Validate that domain_id is not "none" when segment is configured
                if domain_id == "none":
                    raise RuntimeError(
                        f"Domain ID is required when segment is configured, but got 'none' for node {node_desc.addr}"
                    )
                domain_to_participants[domain_id].append((node_desc, infra_rank))
        else:
            # Each node is its own domain (use infra_rank as domain key for consistent typing)
            domain_to_participants = {
                str(infra_rank): [(node_desc, infra_rank)]
                for node_desc, infra_rank, _ in sorted_participants
            }

        # Step 2: Sort domains by SLURM topology (first node's infra_rank in each domain)
        # Since participants are already sorted by infra_rank, domain_participants[0][1] is the smallest
        sorted_domains = sorted(
            domain_to_participants.items(), key=lambda x: x[1][0][1]
        )  # x[1][0][1] = first participant's infra_rank

        # Step 3: Single pass - walk through sorted domains and assign ranks directly
        # Work in segments for cleaner arithmetic
        num_segments = world_size // segment
        result = {}
        active_rank = 0
        active_segments = 0
        standby_rank = world_size
        standby_infra_ranks = []  # Collect infra_ranks of standby ranks

        for domain_id, domain_participants in sorted_domains:
            domain_segments = len(domain_participants) // segment

            # Calculate how many nodes to take for active ranks
            if active_segments < num_segments and domain_segments > 0:
                segments_to_take = min(domain_segments, num_segments - active_segments)
            else:
                segments_to_take = 0

            nodes_to_take = segments_to_take * segment

            # Assign active ranks
            for node_desc, infra_rank in domain_participants[:nodes_to_take]:
                result[node_desc] = active_rank
                log.debug(
                    f"Rank: {node_desc.addr} (infra={infra_rank}) -> group_rank={active_rank} (active)"
                )
                active_rank += 1

            active_segments += segments_to_take

            # Assign standby ranks to rest from this domain
            for node_desc, infra_rank in domain_participants[nodes_to_take:]:
                result[node_desc] = standby_rank
                standby_infra_ranks.append(infra_rank)  # Collect standby infra_rank
                log.debug(
                    f"Rank: {node_desc.addr} (infra={infra_rank}) -> group_rank={standby_rank} (standby)"
                )
                standby_rank += 1

            if nodes_to_take > 0:
                log.debug(
                    f"Domain {domain_id}: {segments_to_take} segments ({nodes_to_take} nodes) active, "
                    f"{len(domain_participants) - nodes_to_take} standby"
                )
            else:
                log.debug(f"Domain {domain_id}: all {len(domain_participants)} nodes to standby")

        # Validate we have enough active segments
        if active_segments < num_segments:
            raise RuntimeError(
                f"Insufficient nodes: assigned {active_segments} segments ({active_rank} nodes), "
                f"need {num_segments} segments ({world_size} nodes). "
                f"Each domain must have at least {segment} nodes."
            )

        standby_info = ""
        if standby_rank > world_size:
            # Format standby infra_ranks as ranges
            # Strip the outer curly braces from format_rank_set_verbose output
            infra_range_str = format_rank_set_verbose(standby_infra_ranks).strip('{}')
            standby_info = f" and {standby_rank - world_size} standby ranks [{infra_range_str}]"

        log.info(f"Assigned segments(segment={segment}): {active_segments} segments{standby_info}")

        return result

    def _get_unhealthy_count(self) -> int:
        """Get the global unhealthy node count.

        This counter tracks unhealthy nodes across the entire job lifetime, not per cycle.
        Once a node is marked unhealthy, it permanently reduces the effective max_nodes.

        Returns:
            The total number of unhealthy nodes in the job, or 0 if the key doesn't exist
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
        self,
        node_desc: _NodeDesc,
        min_nodes: int,
        max_nodes: int,
        last_call_timeout: timedelta,
        is_first_rendezvous: bool = True,
    ) -> Tuple[int, int]:
        """Perform the complete rendezvous process: join, wait for completion, acknowledge, and get rank.

        DESIGN RATIONALE:
        This atomic barrier-based rendezvous design balances flexibility with convergence guarantees:

        1. FIRST RENDEZVOUS ENFORCEMENT: The first rendezvous (initial startup) REQUIRES max_nodes
           to join before completion. This ensures all expected nodes participate during initial
           startup when all nodes should be available. This prevents premature start with missing nodes.

        2. FLEXIBILITY (SUBSEQUENT RENDEZVOUS): After the first rendezvous, we use last_call_timeout
           (not max_nodes) as completion criteria because:
           - Hot-fix scenarios may need to exclude broken participants without failing the job
           - We cannot wait for max_nodes after failures as some nodes may be permanently down
           - The system needs to be resilient to partial failures while maintaining progress

        3. CONVERGENCE: The rendezvous will complete when either:
           - FIRST RENDEZVOUS: max_nodes is reached (strict requirement)
           - SUBSEQUENT RENDEZVOUS: max_nodes is reached (immediate completion), OR
             min_nodes is reached AND last_call_timeout expires (graceful completion)

        4. NEW COMER HANDLING:
           - New comers arriving during active rendezvous participate in the current round
           - New comers arriving after completion trigger the next rendezvous round
           - All existing participants detect new comers and restart to join the new round
           - This ensures eventual convergence with the new comer

        5. RACE CONDITION MITIGATION:
           - The last_call_timeout provides a grace period for nodes that are restarting
           - This prevents premature completion when nodes are in the process of joining
           - The timeout balances responsiveness with inclusion of restarting nodes

        6. HOT SPARE HANDLING (Step 0):
           - Hot spares arriving after training starts will wait at Step 0
           - They wait until a failure opens the rendezvous (value=0)
           - This prevents disruption of ongoing training

        Args:
            node_desc: Node descriptor for this participant
            min_nodes: Minimum number of nodes required for training to proceed
            max_nodes: Maximum number of nodes allowed (active + standby)
            last_call_timeout: Grace period after min_nodes reached to allow restarting nodes to join
                              (only applies to subsequent rendezvous, not the first one)
            is_first_rendezvous: True if this is the first rendezvous cycle (enforces max_nodes),
                                False for subsequent cycles (allows graceful completion)

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
        infra_rank = get_infrastructure_rank()

        # Determine domain ID (with caching to avoid re-parsing on every rendezvous)
        if self._cached_domain_id is None:
            if self.segment is not None:
                # Segment is configured - domain_id is required, always use ClusterUUID
                try:
                    self._cached_domain_id = _parse_domain_id_from_nvidia_smi()
                except Exception as e:
                    raise RuntimeError(
                        f"Domain ID is required when --ft-segment is specified, but failed to parse: {e}"
                    )
            else:
                # Segment not configured - domain_id not needed
                self._cached_domain_id = "none"

        domain_id = self._cached_domain_id

        # Store participant information in arrived_<count> key using the unique identifier
        arrived_key = f"{self.prefix}:arrived_{self._arrived_count}"
        participant_data = RendezvousParticipantInfo.pack(node_desc, infra_rank, domain_id)
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
        if self._arrived_count >= min_nodes and not is_first_rendezvous:
            # Only set deadline for non-first rendezvous
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
            # DESIGN NOTE:
            # - FIRST RENDEZVOUS: We require max_nodes to ensure all nodes join during initial startup
            # - SUBSEQUENT RENDEZVOUS: We complete on last_call_timeout to support:
            #   1. Hot-fix scenarios where broken participants are excluded without failing the job
            #   2. Flexibility in participant count while ensuring convergence
            #   3. Grace period for restarting nodes to join before completion
            should_complete = False

            # Get unhealthy count to calculate effective max nodes
            # Since unhealthy nodes won't participate, we should complete when all healthy nodes arrive
            effective_max_nodes = max_nodes - self._get_unhealthy_count()

            if self._arrived_count >= effective_max_nodes:
                # Max healthy nodes reached - immediate completion (both first and subsequent)
                # We don't wait for unhealthy nodes that will never join
                should_complete = True
            elif (
                not is_first_rendezvous
                and self._arrived_count >= min_nodes
                and last_call_deadline
                and datetime.utcnow() >= last_call_deadline
            ):
                # Min nodes reached and grace period expired - graceful completion
                # ONLY allowed for subsequent rendezvous (not first)
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

    def get_all_participants(self, total_participants: int) -> List[Tuple[_NodeDesc, int, str]]:
        """Get all participants that have arrived using multi_get.

        Args:
            total_participants: Total number of participants

        Returns:
            List of tuples: (node_desc, infra_rank, domain_id) in arrival order.
            domain_id is "none" if segment not configured.
            Note: Consumers of this method sort the list according to their needs.
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
                    node_desc, infra_rank, domain_id = RendezvousParticipantInfo.unpack(
                        participant_data
                    )
                    participants.append((node_desc, infra_rank, domain_id))
                except Exception as e:
                    log.warning(f"Failed to unpack participant data for arrived_{i+1}: {e}")

        return participants

    def _clear_barrier_keys(self, node_desc: _NodeDesc):
        """Clear main barrier keys to prepare for next rendezvous round.

        Note: We do NOT clear last_participant_arrived_key because it serves as the
        open/close indicator for the rendezvous. It remains set to 1 (closed) during
        training and is reset to 0 (open) by the launcher when a failure is detected.

        Note: We do NOT clear unhealthy_count_key because it is a global counter that
        tracks unhealthy nodes across the entire job lifetime, not per cycle.

        This method is called by the last participant to acknowledge in Step 3b, BEFORE
        rank assignment. This ensures all participants are still waiting and cannot
        start the next cycle yet, making it safe to clear the global counters.
        """
        # Clear main keys - individual arrived_<count> keys don't need clearing
        # DO NOT clear last_participant_arrived_key - it indicates open/close state
        # DO NOT clear unhealthy_count_key - it is a global job-level counter
        keys_to_clear = [
            self.arrived_count_key,
            self.ack_count_key,
            self.peer_aborted_count_key,  # Clear peer aborted counter for next round
        ]

        # Delete main keys
        for key in keys_to_clear:
            try:
                self.store.delete_key(key)
            except Exception as e:
                log.debug(f"[{node_desc}] Failed to delete key {key}: {e}")

    def assign_group_ranks(
        self, world_size: int, max_nodes: int, total_participants: int, node_desc: _NodeDesc
    ) -> bool:
        """Assign group ranks to all participants. Called by Rank 0 (TCPStore host).

        Args:
            world_size: Target world size for training (number of active participants)
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

        assigned_group_ranks = self._assign_group_ranks(all_participants, world_size)

        # Store the assigned ranks and total participants in the store
        for i, (node_desc_item, infra_rank, _) in enumerate(all_participants):
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

        log.debug(
            f"[{node_desc}] [Step 3b] Assigned group ranks to {len(assigned_group_ranks)} participants"
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
        segment: Optional[int] = None,
        nproc_per_node: int = 1,  # Number of processes per node
        enable_nic_healthcheck: bool = False,
        enable_dist_storage_healthcheck: bool = False,
        link_state_path_template: Optional[str] = None,
        storage_healthcheck_paths: Optional[list] = None,
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
            segment:
                Number of nodes to select from each domain.
            enable_nic_healthcheck:
                Whether to enable all NIC link state health check before rendezvous.
            link_state_path_template:
                Template path for NIC link state files.
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
            segment=segment,
            nproc_per_node=nproc_per_node,
        )

        return cls(
            node,
            settings,
            "c10d",
            store,
            is_store_host,
            enable_nic_healthcheck=enable_nic_healthcheck,
            enable_dist_storage_healthcheck=enable_dist_storage_healthcheck,
            link_state_path_template=link_state_path_template,
            storage_healthcheck_paths=storage_healthcheck_paths,
        )

    def __init__(
        self,
        node: _NodeDesc,
        settings: RendezvousSettings,
        backend_name: str,
        store: Store,
        is_store_host: bool = False,
        enable_nic_healthcheck: bool = False,
        enable_dist_storage_healthcheck: bool = False,
        link_state_path_template: Optional[str] = None,
        storage_healthcheck_paths: Optional[list] = None,
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
            settings.segment,
        )
        self._assigned_rank = None
        self._world_size = None

        # Track rendezvous round number in memory (increments on each next_rendezvous call)
        # Used to isolate MASTER_ADDR/MASTER_PORT between rendezvous rounds
        # Round 0 indicates the first rendezvous (enforces max_nodes requirement)
        self._rendezvous_round = 0

        # Sync with global cycle for cross-array task coordination
        # This ensures all nodes (including newly joined nodes from different array tasks)
        # use the same cycle number, which is critical for PyTorch's role_info coordination
        if self._barrier_state.store.check([self._barrier_state.global_cycle_key]):
            stored_cycle_bytes = self._barrier_state.store.get(self._barrier_state.global_cycle_key)
            stored_cycle = int(stored_cycle_bytes.decode('utf-8'))
            if stored_cycle > self._rendezvous_round:
                log.info(
                    f"Syncing _rendezvous_round from {self._rendezvous_round} to {stored_cycle} "
                    f"(cross-array task coordination during initialization)"
                )
                self._rendezvous_round = stored_cycle

                # Also sync the profiling cycle to match
                # This ensures newly joining nodes (e.g., replacement array tasks) continue
                # with the correct cycle number instead of restarting from 0
                set_profiling_cycle(self._rendezvous_round)

        self._ranks_connector = IpcConnector(FT_LAUNCHER_IPC_SOCKET)
        self._ranks_connector.start_receiving()

        # Store NIC health check configuration
        self._enable_nic_healthcheck = enable_nic_healthcheck
        self._link_state_path_template = link_state_path_template

        # Initialize NIC link state health checker (single instance to maintain baseline)
        self._nic_link_state_checker = None
        if self._enable_nic_healthcheck:
            self._nic_link_state_checker = NicLinkStateHealthCheck(
                link_state_path_template=self._link_state_path_template
            )

        # Distributed storage health checker instance (enabled via boolean flag)
        self._dist_storage_state_checker = (
            DistributedStorageHealthCheck() if enable_dist_storage_healthcheck else None
        )
        # Storage path health checker instance
        self._storage_path_checker = (
            StoragePathHealthCheck(storage_healthcheck_paths) if storage_healthcheck_paths else None
        )

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

    def _maybe_increment_unhealthy_count(self) -> None:
        """Increment unhealthy count in the store, but skip if in SLURM job array.

        In SLURM job array deployments, we rely on SLURM to quickly restart/requeue
        failing job array elements. The unhealthy_count logic should not kick in for
        job array deployments:
        1) A node failing health check will exit the ft_launcher itself. The srun will
           run with "--kill-on-bad-exit" to kill the job array element.
        2) No unhealthy count should be increased. So the rest of the nodes will proceed
           to wait for rendezvous and eventually timeout on rendezvous if worst case we
           don't have refill coming in.
        """
        if is_slurm_job_array():
            return

        # Normal case: increment unhealthy count
        self._barrier_state.store.add(self._barrier_state.unhealthy_count_key, 1)

    def _run_health_check(self, health_checker, check_name: str, failure_message: str) -> None:
        """Helper method to run a health check with consistent error handling.

        Args:
            health_checker: Health checker instance to call
            check_name: Name of the health check (for logging)
            failure_message: Error message to use if health check fails
        """
        try:
            if not health_checker():
                self._maybe_increment_unhealthy_count()
                log.error(f"{check_name} failed for node {self._this_node}: {failure_message}")
                raise UnhealthyNodeException(failure_message)
        except UnhealthyNodeException:
            raise
        except Exception as e:
            self._maybe_increment_unhealthy_count()
            log.error(f"Unexpected error during {check_name} for node {self._this_node}: {str(e)}")
            raise UnhealthyNodeException(str(e)) from e

    def ensure_node_is_healthy(self) -> None:
        """Perform GPU, NIC link state, and Node health checks for this node."""
        # Record the health check message
        msg = f"Checking health status of {self._this_node}."
        self._record(message=msg)

        # Set current cycle for health check injection (if enabled)
        if os.environ.get("NVRX_INJECT_GPU_FAILURE"):
            health_check_injector.set_current_cycle(self._rendezvous_round)

        # Perform GPU health check
        self._run_health_check(
            GPUHealthCheck(), "GPU health check", f"Node {self._this_node} has an unhealthy GPU."
        )

        # Perform NIC link state health check if enabled
        if self._nic_link_state_checker is not None:
            self._run_health_check(
                self._nic_link_state_checker,
                "NIC link state health check",
                f"Node {self._this_node} has unhealthy NIC link(s).",
            )
        # Perform distributed storage (Lustre/NFS) health check if enabled
        if self._dist_storage_state_checker is not None:
            self._run_health_check(
                self._dist_storage_state_checker,
                "Storage health check",
                f"Node {self._this_node} has unhealthy storage state.",
            )
        # Perform storage path health check if paths were provided
        if self._storage_path_checker is not None:
            self._run_health_check(
                self._storage_path_checker,
                "Storage path health check",
                f"Node {self._this_node} has invalid or unreadable paths.",
            )

        # Perform Node health check (external service if available)
        _nodehealth_checker = get_node_health_check()
        if _nodehealth_checker is not None:
            self._run_health_check(
                _nodehealth_checker,
                "Node health check",
                f"Node {self._this_node} is unhealthy.",
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
            self._maybe_increment_unhealthy_count()
            raise UnhealthyNodeException(
                f"Node {self._this_node} is excluded from the training due to an user request."
            )

    def _perform_rendezvous(self) -> None:
        """Perform the complete rendezvous process."""
        # Perform complete rendezvous process
        # Pass flag indicating if this is the first rendezvous (round 0)
        is_first_rendezvous = self._rendezvous_round == 0

        group_rank, total_participants = self._barrier_state.perform_rendezvous(
            self._this_node,
            self._settings.min_nodes,
            self._settings.max_nodes,
            self._settings.timeout.last_call,
            is_first_rendezvous,
        )

        # Increment round number for the next rendezvous
        # This ensures each rendezvous round uses an isolated namespace for MASTER_ADDR/MASTER_PORT
        self._rendezvous_round += 1

        # AFTER rendezvous: Store host updates global cycle
        # This allows new nodes to sync to the current cycle before their first rendezvous
        if self._barrier_state.is_store_host:
            self._barrier_state.store.set(
                self._barrier_state.global_cycle_key, str(self._rendezvous_round).encode('utf-8')
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

            # Adjust local_world_size based on whether this is a standby or active participant
            assert (
                self._worker_group is not None
            ), "set_worker_group must be called before next_rendezvous"
            if rank >= self._settings.min_nodes:
                # This is a standby participant, set local_world_size to 0
                self._worker_group.spec.local_world_size = 0
            else:
                # This is an active participant, ensure local_world_size is correct
                if self._worker_group.spec.local_world_size == 0:
                    # Restore from the configured value in settings
                    self._worker_group.spec.local_world_size = self._settings.nproc_per_node

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
        """Return the current rendezvous round.

        The round number starts at 0 and increments after each successful rendezvous.
        This is useful for debugging and understanding the rendezvous lifecycle.
        """
        return self._rendezvous_round

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
        """Get the store for this rendezvous.

        Uses round number in the prefix to isolate MASTER_ADDR/MASTER_PORT keys
        between rendezvous rounds, preventing race conditions when rank 0 changes.

        The round number is tracked in memory and increments each time next_rendezvous()
        completes. This ensures each handler instance uses progressively isolated namespaces
        for bootstrap keys without requiring TCPStore coordination.
        """
        key_prefix = f"torch.rendezvous.{self._settings.run_id}.{self._rendezvous_round}"

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

    +----------------------------+------------------------------------------------------+
    | Parameter                  | Description                                          |
    +============================+======================================================+
    | join_timeout               | The total time, in seconds, within which the         |
    |                            | rendezvous is expected to complete. Defaults to 600  |
    |                            | seconds.                                             |
    +----------------------------+------------------------------------------------------+
    | last_call_timeout          | An additional wait amount, in seconds, before        |
    |                            | completing the rendezvous once the minimum number of |
    |                            | nodes has been reached. Defaults to 10 seconds.      |
    |                            | NOTE: This only applies to SUBSEQUENT rendezvous     |
    |                            | after the first one. The FIRST rendezvous always     |
    |                            | waits for max_nodes to ensure all nodes join during  |
    |                            | initial startup.                                     |
    +----------------------------+------------------------------------------------------+
    | close_timeout              | The time, in seconds, within which the rendezvous is |
    |                            | expected to close after a call to                    |
    |                            | :py:meth:`RendezvousHandler.set_closed` or           |
    |                            | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |
    |                            | 30 seconds.                                          |
    +----------------------------+------------------------------------------------------+
    | segment                    | Minimum number of nodes required per domain for      |
    |                            | segment-aware rank assignment. Domains with fewer    |
    |                            | nodes are excluded. As many complete segments as     |
    |                            | possible are selected from each domain. min_nodes    |
    |                            | must be divisible by segment. Defaults to None       |
    |                            | (disabled).                                          |
    +----------------------------+------------------------------------------------------+
    | enable_nic_healthcheck     | Whether to enable NIC link state health check before |
    |                            | rendezvous. Defaults to False.                       |
    +----------------------------+------------------------------------------------------+
    | link_state_path_template   | Template path for NIC link state files. Should       |
    |                            | contain {nic} placeholder. Defaults to None (uses    |
    |                            | default path /sys/class/infiniband/{nic}/ports/1/    |
    |                            | state).                                              |
    +----------------------------+------------------------------------------------------+
    """
    try:
        timeout = RendezvousTimeout(
            _get_timeout(params, "join"),
            _get_timeout(params, "last_call"),
            _get_timeout(params, "close"),
        )

        # Get is_store_host from parameters
        is_store_host = params.config.get('is_store_host', False)
        segment = params.config.get('segment', None)
        nproc_per_node = params.config.get('nproc_per_node', 1)
        enable_nic_healthcheck = params.config.get('enable_nic_healthcheck', False)
        enable_dist_storage_healthcheck = params.config.get(
            'enable_dist_storage_healthcheck', False
        )
        storage_healthcheck_paths = params.config.get('storage_healthcheck_paths', None)
        link_state_path_template = params.config.get('link_state_path_template', None)

        return FtRendezvousBarrierHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
            is_store_host=is_store_host,
            segment=segment,
            nproc_per_node=nproc_per_node,
            enable_nic_healthcheck=enable_nic_healthcheck,
            enable_dist_storage_healthcheck=enable_dist_storage_healthcheck,
            link_state_path_template=link_state_path_template,
            storage_healthcheck_paths=storage_healthcheck_paths,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
