# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: BSD-3-Clause
# Modifications made by NVIDIA
# This file implements a new fault-tolerant rendezvous barrier design
# that uses atomic increments for slot allocation and compare_set for
# round-fenced writes to reused slot/rank keys.

import inspect
import json
import logging
import os
import shutil
import signal
import socket

# More Info: https://bandit.readthedocs.io/en/latest/blacklists/blacklist_imports.html#b404-import-subprocess
import subprocess  # nosec B404
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import NodeState, construct_and_record_rdzv_event
from torch.distributed.elastic.multiprocessing import SignalException
from torch.distributed.elastic.rendezvous.api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousGracefulExitError,
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
    AttributionService,
    DistributedStorageHealthCheck,
    GPUHealthCheck,
    NicLinkStateHealthCheck,
    StoragePathHealthCheck,
)
from ..shared_utils.profiling import (
    ProfilingEvent,
    get_profiling_cycle,
    record_profiling_event,
    set_profiling_cycle,
)
from .cycle_info_writer import CycleInfoReporter, CycleInfoRoundSnapshot
from .data import WorkloadAction
from .ipc_connector import IpcConnector
from .launcher import FT_LAUNCHER_IPC_SOCKET, UnhealthyNodeException, get_node_health_check

# Conditionally import failure injector (only active when NVRX_INJECT_GPU_FAILURE is set)
if os.environ.get("NVRX_INJECT_GPU_FAILURE"):
    from ..testing_utils import health_check_injector

from .utils import get_infrastructure_rank, is_slurm_job_array, parse_bool_env, slurm_sort_addrs

log = logging.getLogger(LogConfig.name)

RDZV_SHUTDOWN_REASON_GRACEFUL = "graceful"
RDZV_SHUTDOWN_REASON_FAILURE = "failure"


def _rdzv_signal_exception_handler(sig: int, frame: Optional[FrameType]) -> None:
    del frame
    raise SignalException(f"Received signal {sig} during rendezvous", signal.Signals(sig))


def _install_rdzv_signal_handlers() -> Dict[signal.Signals, Any]:
    prev_handlers: Dict[signal.Signals, Any] = {}
    for sig_to_handle in (signal.SIGTERM, signal.SIGINT):
        prev_handlers[sig_to_handle] = signal.getsignal(sig_to_handle)
        signal.signal(sig_to_handle, _rdzv_signal_exception_handler)
    return prev_handlers


def _restore_rdzv_signal_handlers(prev_handlers: Dict[signal.Signals, Any]) -> None:
    for sig_to_handle, handler in prev_handlers.items():
        signal.signal(sig_to_handle, handler)


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
            An additional wait amount before completing a restart rendezvous
            once enough hot-spare candidates are present but previous active
            nodes are still missing.
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
        "last_call": timedelta(seconds=1),
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
            if name == "last_call":
                if timeout < self._ZERO:
                    raise ValueError(f"The {name} timeout ({timeout}) must be non-negative.")
            elif timeout <= self._ZERO:
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
        replacement_group_size:
            Number of nodes required for a complete replacement group. None disables
            replacement-group-aware active selection.
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
    replacement_group_size: Optional[int] = None
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


Participant = Tuple[_NodeDesc, int, str, Optional[str]]
SlotParticipant = Optional[Participant]


class GroupRankStatus(Enum):
    """Group rank status for participants."""

    UNASSIGNED = -1  # Initially unassigned


class RendezvousParticipantInfo:
    """Participant metadata stored inside a round-fenced TCPStore value.

    A rendezvous participant can be a physical node or a process in a physical node
    in the simulation case. This class provides a JSON-based format that can store:
    - NodeDesc (addr, pid, local_id)
    - Infrastructure rank (from SLURM_PROCID/GROUP_RANK or deterministically assigned)
    - Domain ID (from node name or ClusterUUID, "none" if segment not configured)
    - Replacement group ID (from NVRX_REPLACEMENT_GROUP_ID or SLURM_ARRAY_TASK_ID)

    The format is designed to support up to 4K participants efficiently.
    In future, this can be changed to Protobuf for better efficiency and performance.
    """

    @staticmethod
    def to_payload(
        node_desc: _NodeDesc,
        infra_rank: int = -1,
        domain_id: str = "none",
        replacement_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert participant information into the store wrapper payload."""
        return {
            "addr": node_desc.addr,
            "pid": node_desc.pid,
            "local_id": node_desc.local_id,
            "infra_rank": infra_rank,
            "domain_id": domain_id,
            "replacement_group_id": (
                None if replacement_group_id is None else str(replacement_group_id)
            ),
        }

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> Participant:
        """Convert a store wrapper payload into participant information."""
        try:
            node_desc = _NodeDesc(
                addr=payload["addr"], pid=payload["pid"], local_id=payload["local_id"]
            )
            infra_rank = payload["infra_rank"]
            domain_id = payload["domain_id"]
            replacement_group_id = payload["replacement_group_id"]
            if replacement_group_id is not None:
                replacement_group_id = str(replacement_group_id)
            return node_desc, infra_rank, domain_id, replacement_group_id
        except KeyError as e:
            raise ValueError(f"Invalid participant info payload: {e}")

    @staticmethod
    def pack(
        node_desc: _NodeDesc,
        infra_rank: int = -1,
        domain_id: str = "none",
        replacement_group_id: Optional[str] = None,
    ) -> str:
        """Pack participant information into JSON format.

        Args:
            node_desc: Node descriptor
            infra_rank: Infrastructure rank (-1 if not assigned)
            domain_id: Domain ID string, or "none" if segment not configured
            replacement_group_id: Replacement group id, or None if replacement groups are disabled
        """
        return json.dumps(
            RendezvousParticipantInfo.to_payload(
                node_desc, infra_rank, domain_id, replacement_group_id
            )
        )

    @staticmethod
    def unpack(data: str) -> Participant:
        """Unpack participant information from JSON format.

        Returns:
            Tuple of (node_desc, infra_rank, domain_id, replacement_group_id)
        """
        try:
            return RendezvousParticipantInfo.from_payload(json.loads(data))
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid participant info data: {e}")


class RendezvousStoreValue:
    """Round-fenced TCPStore value wrapper for reused rendezvous keys."""

    @staticmethod
    def pack(round_id: int, payload: Dict[str, Any]) -> str:
        return json.dumps({"round": round_id, "payload": payload})

    @staticmethod
    def unpack(data: str) -> Tuple[int, Dict[str, Any]]:
        try:
            value = json.loads(data)
            round_id = int(value["round"])
            payload = value["payload"]
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            return round_id, payload
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid rendezvous store value: {e}")


@dataclass(frozen=True)
class _CloseRoundDecision:
    """Decision returned by the previous-active last-call gate."""

    should_close: bool
    missing_previous_active: Tuple[int, ...] = ()
    last_call_deadline: Optional[float] = None


class _PreviousActiveLastCallGate:
    """Delay restart-round closure while previous active nodes may still rejoin."""

    def __init__(
        self,
        previous_active_by_infra_rank: Optional[Dict[int, Optional[str]]],
        timeout_seconds: float,
        enabled: bool,
    ) -> None:
        self._previous_active_by_infra_rank = dict(previous_active_by_infra_rank or {})
        self._timeout_seconds = timeout_seconds
        self._enabled = (
            enabled and bool(self._previous_active_by_infra_rank) and self._timeout_seconds > 0.0
        )
        self._deadline: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(
        self,
        participants: List[Participant],
        now: float,
        unhealthy_replacement_groups: Optional[Set[str]] = None,
    ) -> _CloseRoundDecision:
        if not self._enabled:
            return _CloseRoundDecision(should_close=True)

        current_infra_ranks = {
            infra_rank for _, infra_rank, _, _ in participants if infra_rank >= 0
        }
        unhealthy_replacement_groups = unhealthy_replacement_groups or set()
        # The last-call gate should only wait for previous active nodes that can
        # still be eligible in this round. If a previous active node belonged to a
        # replacement group already marked unhealthy, waiting for it would delay the
        # replacement path without improving correctness.
        expected_previous_active = {
            infra_rank
            for infra_rank, replacement_group_id in self._previous_active_by_infra_rank.items()
            if replacement_group_id is None
            or replacement_group_id not in unhealthy_replacement_groups
        }
        missing = tuple(sorted(expected_previous_active - current_infra_ranks))
        if not missing:
            return _CloseRoundDecision(should_close=True)

        if self._deadline is None:
            self._deadline = now + self._timeout_seconds

        if now >= self._deadline:
            return _CloseRoundDecision(
                should_close=True,
                missing_previous_active=missing,
                last_call_deadline=self._deadline,
            )

        return _CloseRoundDecision(
            should_close=False,
            missing_previous_active=missing,
            last_call_deadline=self._deadline,
        )


class _StaleRendezvousRoundError(RuntimeError):
    """Raised when a reused key already contains data from a newer rendezvous round."""

    def __init__(self, observed_round: int, attempted_round: int, key: str):
        self.observed_round = observed_round
        self.attempted_round = attempted_round
        self.key = key
        super().__init__(
            f"Stale rendezvous write for {key}: attempted round {attempted_round}, "
            f"observed round {observed_round}"
        )


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
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")

    try:
        # Run nvidia-smi to query ClusterUUID (full path via shutil.which avoids PATH hijack).
        # More Info: https://bandit.readthedocs.io/en/latest/plugins/b603_subprocess_without_shell_equals_true.html
        result = subprocess.run(  # nosec B603
            [nvidia_smi, "-q"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
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

        log.debug(f"domain_id={cluster_uuid} (nvidia-smi ClusterUUID)")
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
    simple key-based coordination.

    DESIGN PHILOSOPHY:
    This implementation prioritizes flexibility and fault tolerance over strict
    synchronization. Key design principles:

    1. ATOMIC BARRIER: Uses atomic increments for slot allocation and per-key
       compare-and-swap writes for reused slot/rank state

    2. GRACEFUL COMPLETION: Completes rendezvous once enough valid participants
       are present; restart rounds with hot spares can use a bounded
       last_call_timeout so previous active nodes can rejoin before spares are
       accepted.

    3. EVENTUAL CONVERGENCE: New comers trigger restarts of existing participants,
       ensuring all nodes eventually participate in the same rendezvous round

    4. RACE CONDITION TOLERANCE: When configured, the previous-active
       last_call_timeout prevents early hot spares from closing a restart round
       ahead of slower survivors.

    5. STALE ROUND DETECTION: Periodically checks if the local rendezvous round
       is behind the global cycle, allowing nodes to recover from desynchronization

    Args:
        store: The C10d store instance
        run_id: The run id of the rendezvous
        is_store_host: Whether this node is the TCPStore host
        join_timeout_seconds: Maximum time to wait for rendezvous completion
        last_call_timeout_seconds: Restart-round grace window after enough
            hot-spare candidates are available while previous active nodes are
            still missing. A value of 0 disables this wait.
        segment: Number of nodes per domain for segment-aware rank assignment
        replacement_group_size: Number of nodes in a complete replacement group. None disables
            replacement-group-aware active selection.
        stale_check_interval: How often (in seconds) to check for stale rounds.
            Default is 10 seconds. Lower values increase store load but reduce
            detection latency. Higher values reduce store load but increase latency.

    This design trades some synchronization guarantees for better fault tolerance
    and operational flexibility in production environments.
    """

    def __init__(
        self,
        store: Store,
        run_id: str,
        is_store_host: bool = False,
        join_timeout_seconds: float = 600.0,
        last_call_timeout_seconds: float = 1.0,
        segment: Optional[int] = None,
        replacement_group_size: Optional[int] = None,
        stale_check_interval: float = 10.0,
    ):
        self.store = store
        self.run_id = run_id
        self.is_store_host = is_store_host
        self.join_timeout_seconds = join_timeout_seconds
        self.last_call_timeout_seconds = last_call_timeout_seconds
        self.segment = segment
        self.replacement_group_size = replacement_group_size
        self.stale_check_interval = stale_check_interval
        self._rendezvous_start_time = None
        self._last_stale_check_time = 0.0  # Track last stale round check time for rate limiting

        # Track rendezvous round number. Each round corresponds to one rendezvous barrier
        # invocation; keys are namespaced by round number to prevent cross-round contamination.
        self._round = 0
        # Current slot in the join order (1-based). Set in Step 1 of perform_rendezvous.
        self._slot: Optional[int] = None
        # Active/standby node addrs (set by store host in assign_group_ranks for cycle info)
        self._active_node_addrs: Optional[List[str]] = None
        self._standby_node_addrs: Optional[List[str]] = None
        # Group ranks of active nodes, parallel to _active_node_addrs (same slot order)
        self._active_ranks: Optional[List[int]] = None
        self._cycle_info_reporter: Optional[CycleInfoReporter] = None

        # Reference to agent (set via set_agent() method in handler)
        # Will be set by handler via set_agent() method
        self._agent = None

        # Cache for domain_id to avoid re-parsing on every rendezvous
        # Only populated if segment is configured
        self._cached_domain_id: Optional[str] = None

        # Key prefixes for the barrier
        self.prefix = f"ft_rendezvous_barrier:{run_id}"
        # Permanent shutdown reason: empty means open; non-empty means no further rounds.
        self.shutdown_key = f"{self.prefix}:shutdown"
        self._attribution_service: Optional[AttributionService] = None
        self._attribution_settled_for_round: int = -1
        # Job-level unhealthy counter; persists across all rounds.
        self.unhealthy_count_key = f"{self.prefix}:unhealthy_count"
        # Latest closed round's active membership. This is host-local by design:
        # the barrier rendezvous currently has a single long-lived rendezvous host.
        self._previous_active_by_infra_rank: Optional[Dict[int, Optional[str]]] = None

        # Per-round open/close indicator. round_done_key value:
        #   0 = round is OPEN (accepting new participants)
        #   1 = round is CLOSED (ranks assigned, training in progress)
        # Initialize round_done_0 to 0 (open) if this is the first time any node connects.
        if not self.store.check([self.round_done_key]):
            self.store.set(self.round_done_key, "0".encode('utf-8'))
        if self.is_store_host and not self.store.check([self.shutdown_key]):
            # Empty value is the open-state sentinel. Consumers must check the value,
            # not just key existence, to decide whether rendezvous is shut down.
            self.store.set(self.shutdown_key, b"")

    @property
    def join_count_key(self) -> str:
        """TCPStore key for the number of participants that have joined this round."""
        return f"{self.prefix}:join_count_{self._round}"

    @property
    def round_done_key(self) -> str:
        """TCPStore key for the open/close indicator of this round.

        Value semantics:
          0 = round is OPEN (accepting new participants)
          1 = round is CLOSED (all ranks assigned, training in progress)

        Writing "1" only happens AFTER all rank keys are written by the store host,
        so any participant that sees "1" can immediately read its rank without racing.
        """
        return f"{self.prefix}:round_done_{self._round}"

    @property
    def unhealthy_replacement_groups_key(self) -> str:
        """TCPStore key for round-scoped unhealthy replacement group ids."""
        return f"{self.prefix}:unhealthy_replacement_groups_{self._round}"

    @staticmethod
    def _current_replacement_group_id() -> Optional[str]:
        replacement_group_id = os.getenv("NVRX_REPLACEMENT_GROUP_ID")
        if replacement_group_id is not None:
            return replacement_group_id
        return os.getenv("SLURM_ARRAY_TASK_ID")

    @staticmethod
    def _pack_unhealthy_replacement_groups(group_ids: Set[str]) -> bytes:
        return json.dumps(
            {"groups": sorted(group_ids)}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def _unpack_unhealthy_replacement_groups(cls, value: Union[str, bytes, bytearray]) -> Set[str]:
        if not value:
            return set()
        try:
            payload = json.loads(cls._store_value_to_str(value))
            groups = payload.get("groups", [])
            if not isinstance(groups, list):
                raise ValueError("groups must be a list")
            return {str(group_id) for group_id in groups}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid unhealthy replacement-group set: {e}")

    def _mark_replacement_group_unhealthy(self, replacement_group_id: str) -> None:
        """Add a replacement group id to this round's unhealthy-group set."""
        replacement_group_id = str(replacement_group_id)
        key = self.unhealthy_replacement_groups_key
        expected_bytes = b""

        while True:
            if expected_bytes:
                current_groups = self._unpack_unhealthy_replacement_groups(expected_bytes)
            elif self.store.check([key]):
                expected_bytes = self.store.get(key)
                current_groups = self._unpack_unhealthy_replacement_groups(expected_bytes)
            else:
                current_groups = set()

            if replacement_group_id in current_groups:
                return

            desired_bytes = self._pack_unhealthy_replacement_groups(
                current_groups | {replacement_group_id}
            )
            result = self.store.compare_set(key, expected_bytes, desired_bytes)
            result_bytes = self._store_value_to_bytes(result)
            if result_bytes == desired_bytes:
                log.info(
                    "Marked replacement group unhealthy: round=%s group_id=%s "
                    "unhealthy_replacement_groups_key=%s",
                    self._round,
                    replacement_group_id,
                    key,
                )
                return

            expected_bytes = result_bytes

    def _maybe_mark_current_replacement_group_unhealthy(self) -> None:
        replacement_group_id = self._current_replacement_group_id()
        if replacement_group_id is None:
            return
        self._mark_replacement_group_unhealthy(replacement_group_id)

    def _get_unhealthy_replacement_groups(self) -> Set[str]:
        key = self.unhealthy_replacement_groups_key
        if not self.store.check([key]):
            return set()
        return self._unpack_unhealthy_replacement_groups(self.store.get(key))

    def _split_by_complete_replacement_groups(
        self,
        participants: List[Participant],
    ) -> Tuple[List[Participant], List[Participant]]:
        """Split participants into complete replacement groups and standby-only leftovers."""
        if self.replacement_group_size is None:
            return list(participants), []

        replacement_group_to_count: Dict[str, int] = defaultdict(int)
        for _, _, _, replacement_group_id in participants:
            if replacement_group_id is not None:
                replacement_group_to_count[replacement_group_id] += 1

        complete_participants = []
        incomplete_participants = []
        for participant in participants:
            replacement_group_id = participant[3]
            if (
                replacement_group_id is not None
                and replacement_group_to_count[replacement_group_id] >= self.replacement_group_size
            ):
                complete_participants.append(participant)
            else:
                incomplete_participants.append(participant)

        return complete_participants, incomplete_participants

    def _can_meet_segment_constraint(
        self,
        participants: List[Participant],
        world_size: int,
    ) -> bool:
        """Check if current participants can meet the segment constraint.

        This performs similar logic to _assign_group_ranks but only checks feasibility
        without actually assigning ranks. Returns True if the provided participants
        can supply enough complete segments across domains to fill world_size positions.

        This is used by the store host to incrementally check if rendezvous can complete
        as participants arrive, enabling optimal performance by completing as soon as
        the segment constraint is satisfied.

        Args:
            participants: List of (node_desc, infra_rank, domain_id) tuples
            world_size: Target number of active nodes (training world size)

        Returns:
            True if participants can form enough complete segments to meet world_size

        Note:
            This method assumes world_size % segment == 0, which is validated at
            handler initialization time in __init__.
        """
        # Treat segment=None as segment=1
        segment = self.segment if self.segment is not None else 1

        # Group by domain and count available complete segments.
        # Slot tombstones and incomplete replacement groups have already been excluded by the caller.
        if self.segment is not None:
            domain_to_count: Dict[str, int] = defaultdict(int)
            for node_desc, infra_rank, domain_id, _ in participants:
                # domain_id validation happens in _assign_group_ranks which will be called
                # after this check passes. If segment is configured, domain_id must not be "none"
                if domain_id == "none":
                    # This indicates a bug - segment is configured but domain_id wasn't obtained
                    raise RuntimeError(
                        f"Domain ID is required when segment is configured, but got 'none' for node {node_desc.addr}"
                    )
                domain_to_count[domain_id] += 1
        else:
            # Each node is its own domain - just count total nodes
            domain_to_count = {"all": len(participants)}

        # Calculate total available complete segments across all domains
        total_segments = sum(count // segment for count in domain_to_count.values())
        required_segments = world_size // segment

        return total_segments >= required_segments

    def _sort_rankable_participants(
        self,
        active_candidate_participants: List[Participant],
        standby_only_participants: List[Participant],
    ) -> Tuple[List[Participant], List[Participant]]:
        """Sort active/standby rank buckets in one deterministic infrastructure order."""
        rankable_participants = active_candidate_participants + standby_only_participants
        active_node_descs = {participant[0] for participant in active_candidate_participants}
        standby_node_descs = {participant[0] for participant in standby_only_participants}
        if active_node_descs & standby_node_descs:
            raise RuntimeError("A participant cannot be both an active candidate and standby-only.")

        has_unassigned = any(infra_rank == -1 for _, infra_rank, _, _ in rankable_participants)
        if has_unassigned:
            rankable_participants_by_node = sorted(rankable_participants, key=lambda x: x[0])
            infra_rank_by_node = {
                node_desc: idx
                for idx, (node_desc, _, _, _) in enumerate(rankable_participants_by_node)
            }

            def with_synthetic_infra_ranks(participants: List[Participant]) -> List[Participant]:
                return [
                    (node_desc, infra_rank_by_node[node_desc], domain_id, replacement_group_id)
                    for node_desc, _, domain_id, replacement_group_id in sorted(
                        participants, key=lambda x: infra_rank_by_node[x[0]]
                    )
                ]

            return (
                with_synthetic_infra_ranks(active_candidate_participants),
                with_synthetic_infra_ranks(standby_only_participants),
            )

        infra_ranks = {infra_rank for _, infra_rank, _, _ in rankable_participants}
        if len(infra_ranks) != len(rankable_participants):
            raise RuntimeError(
                "Duplicate infrastructure ranks detected. "
                "Each node must have a unique SLURM_PROCID or GROUP_RANK."
            )

        return (
            sorted(active_candidate_participants, key=lambda x: x[1]),
            sorted(standby_only_participants, key=lambda x: x[1]),
        )

    def _assign_group_ranks(
        self,
        active_candidate_participants: List[Participant],
        world_size: int,
        standby_only_participants: Optional[List[Participant]] = None,
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
            active_candidate_participants: List of active-candidate (node_desc, infra_rank,
                         domain_id, replacement_group_id) tuples. domain_id is "none" if
                         segment not configured.
            world_size: Target number of active nodes (training world size)
            standby_only_participants: Participants that should be assigned standby ranks
                         after active candidates.

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

        standby_only_participants = standby_only_participants or []
        sorted_active_candidate_participants, standby_only_participants = (
            self._sort_rankable_participants(
                active_candidate_participants,
                standby_only_participants,
            )
        )

        # Step 1: Group by domain
        # When segment is None, treat each node as its own domain (non-segmented deployment)
        if self.segment is not None:
            domain_to_participants: Dict[str, List[Tuple[_NodeDesc, int]]] = defaultdict(list)
            for node_desc, infra_rank, domain_id, _ in sorted_active_candidate_participants:
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
                for node_desc, infra_rank, _, _ in sorted_active_candidate_participants
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
                active_rank += 1

            active_segments += segments_to_take

            # Assign standby ranks to rest from this domain
            for node_desc, infra_rank in domain_participants[nodes_to_take:]:
                result[node_desc] = standby_rank
                standby_infra_ranks.append(infra_rank)  # Collect standby infra_rank
                standby_rank += 1

            if nodes_to_take > 0:
                log.debug(
                    f"Domain {domain_id[:8]}: {segments_to_take} segs ({nodes_to_take} nodes) active, "
                    f"{len(domain_participants) - nodes_to_take} standby"
                )
            else:
                log.debug(
                    f"Domain {domain_id[:8]}: all {len(domain_participants)} nodes to standby"
                )

        # Validate we have enough active segments
        if active_segments < num_segments:
            raise RuntimeError(
                f"Insufficient nodes: assigned {active_segments} segments ({active_rank} nodes), "
                f"need {num_segments} segments ({world_size} nodes). "
                f"Each domain must have at least {segment} nodes."
            )

        for node_desc, infra_rank, _, _ in standby_only_participants:
            result[node_desc] = standby_rank
            standby_infra_ranks.append(infra_rank)
            standby_rank += 1

        standby_info = ""
        if standby_rank > world_size:
            # Format standby infra_ranks as ranges
            # Strip the outer curly braces from format_rank_set_verbose output
            infra_range_str = format_rank_set_verbose(standby_infra_ranks).strip('{}')
            standby_info = f" and {standby_rank - world_size} standby ranks [{infra_range_str}]"

        log.info(
            f"Rank assignment (seg_size={segment}): {active_segments} active segs{standby_info}"
        )

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

    def _remember_previous_active_membership(
        self,
        active_candidate_participants: List[Participant],
        assigned_group_ranks: Dict[_NodeDesc, int],
        world_size: int,
    ) -> None:
        """Remember active membership for restart-round close gating.

        Infrastructure rank is the stable physical identity across launcher
        restarts. If any active participant lacks one, the snapshot is treated as
        unstable and the previous-active last-call gate will be disabled.
        """
        active_by_infra_rank: Dict[int, Optional[str]] = {}
        for node_desc, infra_rank, _, replacement_group_id in active_candidate_participants:
            group_rank = assigned_group_ranks.get(node_desc)
            if group_rank is None or group_rank >= world_size:
                continue
            if infra_rank < 0:
                log.warning(
                    "Previous active snapshot is not stable; disabling previous-active "
                    "last-call wait until stable infrastructure ranks are available."
                )
                self._previous_active_by_infra_rank = None
                return
            active_by_infra_rank[infra_rank] = replacement_group_id

        if len(active_by_infra_rank) != world_size:
            log.warning(
                "Previous active snapshot has %s stable infra ranks for world_size=%s; "
                "disabling previous-active last-call wait.",
                len(active_by_infra_rank),
                world_size,
            )
            self._previous_active_by_infra_rank = None
            return

        self._previous_active_by_infra_rank = active_by_infra_rank

    def _make_previous_active_last_call_gate(
        self,
        min_nodes: int,
        max_nodes: int,
    ) -> _PreviousActiveLastCallGate:
        enabled_by_config = (
            self._round > 0 and max_nodes > min_nodes and self.last_call_timeout_seconds > 0.0
        )
        if not enabled_by_config:
            return _PreviousActiveLastCallGate(
                previous_active_by_infra_rank=None,
                timeout_seconds=self.last_call_timeout_seconds,
                enabled=False,
            )

        return _PreviousActiveLastCallGate(
            previous_active_by_infra_rank=self._previous_active_by_infra_rank,
            timeout_seconds=self.last_call_timeout_seconds,
            enabled=self._previous_active_by_infra_rank is not None,
        )

    def is_next_round_open(self) -> bool:
        """Return True if the next rendezvous round has been opened by any node in the cluster.

        open_rendezvous() sets round_done_{N+1}=0 as the canonical signal that a failure
        was detected and a new round is ready for joining.  Healthy nodes in the monitor
        loop can poll this instead of a separate peer_restart_count key.
        """
        next_key = f"{self.prefix}:round_done_{self._round + 1}"
        if not self.store.check([next_key]):
            return False
        # The next round may be opened by a non-store-host. The store host closes the
        # active cycle when it creates or observes round N+1.
        self._report_cycle_end_as_host(self._round + 1)
        return True

    def _sync_from_per_round_state(self) -> bool:
        """Sync local _round by scanning round_done_N keys forward from current round.

        Scans `round_done_{N}` keys starting at the current `_round`. For each round
        whose key exists and has value 1 (closed), advances N. Stops when the key does
        not exist (round not yet opened) or has value 0 (round in progress).

        This replaces the old `global_cycle_key` approach: instead of a single
        monotonically-increasing counter written by the store host, replacement nodes
        can reconstruct the current round by scanning per-round keys. The scan costs
        O(restarts) round trips but is only performed during Step 0 rate-limited checks.

        Returns:
            True if _round was advanced (was behind), False if already current
        """
        N = self._round
        observed_open_round = None
        while True:
            key = f"{self.prefix}:round_done_{N}"
            if not self.store.check([key]):
                break  # Round N not opened yet
            val = int(self.store.get(key).decode('utf-8'))
            if val == 1:
                N += 1  # Round N completed, check N+1
            else:
                observed_open_round = N
                break  # Round N is open or in progress (val==0)

        if N > self._round:
            old_round = self._round
            self._round = N
            set_profiling_cycle(N)
            # Sync agent's progress tracker if agent reference is available
            if self._agent is not None:
                self._agent._progress_tracker.sync_cycle_number(N)
            log.debug(f"Synced round {old_round} -> {N} (per-round scan)")
            if observed_open_round is not None:
                self._report_cycle_end_as_host(observed_open_round)
            return True

        return False

    @staticmethod
    def _compute_step2_poll_interval(min_nodes: int, segment_check_interval: float) -> float:
        """Compute adaptive Step 2 polling interval from rendezvous size.

        Small rendezvous should converge quickly (1s polling), while large rendezvous
        can use a slower poll to reduce store pressure. We cap by the configured
        segment_check_interval.
        """
        # Scale from 1s up to the configured max, reaching max around 10K nodes.
        scaled_interval = min_nodes / 2000.0
        return max(1.0, min(segment_check_interval, max(1.0, scaled_interval)))

    @staticmethod
    def _store_value_to_bytes(value: Union[str, bytes, bytearray]) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        return str(value).encode('utf-8')

    @classmethod
    def _store_value_to_str(cls, value: Union[str, bytes, bytearray]) -> str:
        return cls._store_value_to_bytes(value).decode('utf-8')

    @classmethod
    def _store_value_round(cls, value: Union[str, bytes, bytearray]) -> int:
        try:
            round_id, _ = RendezvousStoreValue.unpack(cls._store_value_to_str(value))
            return round_id
        except Exception:
            return -1

    @staticmethod
    def _rank_payload(rank: int, total_participants: int) -> Dict[str, int]:
        return {"rank": rank, "total": total_participants}

    @classmethod
    def _pack_rank_value(cls, rank: int, total_participants: int, round_id: int) -> bytes:
        return cls._store_value_to_bytes(
            RendezvousStoreValue.pack(round_id, cls._rank_payload(rank, total_participants))
        )

    @classmethod
    def _unpack_rank_value(cls, value: Union[str, bytes, bytearray]) -> Tuple[int, int, int]:
        round_id, payload = RendezvousStoreValue.unpack(cls._store_value_to_str(value))
        try:
            return int(payload["rank"]), int(payload["total"]), round_id
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid rank payload: {e}")

    @classmethod
    def _pack_participant_value(
        cls,
        node_desc: _NodeDesc,
        infra_rank: int,
        domain_id: str,
        round_id: int,
        replacement_group_id: Optional[str] = None,
    ) -> bytes:
        return cls._store_value_to_bytes(
            RendezvousStoreValue.pack(
                round_id,
                RendezvousParticipantInfo.to_payload(
                    node_desc, infra_rank, domain_id, replacement_group_id
                ),
            )
        )

    @classmethod
    def _unpack_participant_value(
        cls, value: Union[str, bytes, bytearray]
    ) -> Tuple[int, _NodeDesc, int, str, Optional[str]]:
        round_id, payload = RendezvousStoreValue.unpack(cls._store_value_to_str(value))
        node_desc, infra_rank, domain_id, replacement_group_id = (
            RendezvousParticipantInfo.from_payload(payload)
        )
        return round_id, node_desc, infra_rank, domain_id, replacement_group_id

    def _round_fenced_compare_set(
        self,
        key: str,
        desired_value: Union[str, bytes, bytearray],
    ) -> None:
        """CAS a reused key once, refusing newer-round state.

        Slot and rank keys are intentionally reused across rounds.  The value stored
        in each key embeds the round that wrote it; this makes the check and write
        atomic on the same key and prevents a slow writer from round N from clobbering
        round N+1 data.  A same-round conflict is a protocol invariant violation.
        """
        desired_bytes = self._store_value_to_bytes(desired_value)
        desired_round = self._store_value_round(desired_bytes)
        if desired_round < 0:
            raise ValueError(f"Desired value for {key} must be round-fenced")

        expected_bytes = b""
        if self.store.check([key]):
            expected_bytes = self.store.get(key)
            current_round = self._store_value_round(expected_bytes)
            if current_round > desired_round:
                raise _StaleRendezvousRoundError(current_round, desired_round, key)
            if current_round == desired_round and expected_bytes != desired_bytes:
                raise RuntimeError(
                    f"Unexpected same-round rendezvous write conflict for {key} "
                    f"in round {desired_round}"
                )

        result = self.store.compare_set(key, expected_bytes, desired_bytes)
        result_bytes = self._store_value_to_bytes(result)
        if result_bytes == desired_bytes:
            return

        current_round = self._store_value_round(result_bytes)
        if current_round > desired_round:
            raise _StaleRendezvousRoundError(current_round, desired_round, key)
        raise RuntimeError(
            f"Unexpected rendezvous CAS conflict for {key}: attempted round "
            f"{desired_round}, observed round {current_round}"
        )

    def _check_timeout_and_closure(self, node_desc: _NodeDesc) -> None:
        """Check for shutdown and timeout.

        Args:
            node_desc: Node descriptor for logging

        Raises:
            RendezvousClosedError: If rendezvous was permanently shut down.
            RendezvousTimeoutError: If rendezvous has timed out
        """
        # Check for permanent shutdown
        if self.is_shutdown():
            msg = f"The node '{node_desc}' detected that rendezvous was permanently closed"
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

    def _wait_for_rendezvous_open(
        self, node_desc: _NodeDesc, stop_event: Optional[threading.Event] = None
    ) -> None:
        """Step 0: Wait until the current round is open. Hot spares and late arrivals wait here.

        This prevents hot spares and late-arriving nodes from disrupting ongoing training.
        A round is considered:
        - NOT YET OPENED: round_done_key does not exist — wait for open_rendezvous() to call
        - OPEN (value=0): Accepting new participants for a new rendezvous round
        - CLOSED (value=1): Training in progress — advance round via _sync_from_per_round_state()
          and wait for the next open round

        For rounds N > 0, round_done_key is only created by open_rendezvous(), which is called
        when a launcher detects a failure. For round 0, the key is initialized in __init__.

        Args:
            node_desc: Node descriptor for logging

        Raises:
            RendezvousGracefulExitError: If rendezvous is permanently shut down. The
                launcher maps the recorded shutdown reason to the final process exit code.

        Note:
            This wait does NOT timeout unless the rendezvous is permanently closed.
            Waiting nodes (hot spares, late arrivals) wait passively until an
            active participant detects a failure and opens the next round.
        """
        wait_count = 0
        logged_waiting = False  # Track if we've logged the waiting message

        while True:
            if stop_event is not None and stop_event.is_set():
                raise RendezvousGracefulExitError("Control rendezvous stop requested")

            # Check for permanent shutdown (no further rounds)
            if self.is_shutdown():
                msg = f"The node '{node_desc}' detected that rendezvous was permanently closed"
                log.info(msg)
                raise RendezvousGracefulExitError(msg)

            # Check if round_done_key exists for the current round.
            # For round 0 it always exists (initialized in __init__). For round N > 0 it is
            # created by open_rendezvous() when a launcher detects a failure.
            key = self.round_done_key
            if not self.store.check([key]):
                # Round not yet opened; wait for open_rendezvous()
                wait_count += 1
                time.sleep(1.0)
                continue

            value = int(self.store.get(key).decode('utf-8'))

            if value == 0:
                # Round is open — proceed to Step 1
                log.debug(f"[{node_desc}] [Step 0] Round {self._round} is open, proceeding to join")
                break

            # value == 1: Round is closed (training in progress).
            # Rate-limited scan to detect if we're behind and need to advance _round.
            if not logged_waiting:
                log.info(
                    f"[{node_desc}] [Step 0] Round {self._round} closed (training in progress), "
                    f"waiting for next round..."
                )
                logged_waiting = True
            elif wait_count % 60 == 0:  # Log every 60 seconds
                log.debug(
                    f"[{node_desc}] [Step 0] Still waiting for next round to open "
                    f"(round={self._round}, waited {wait_count} seconds)"
                )

            current_time = time.monotonic()
            if current_time - self._last_stale_check_time >= self.stale_check_interval:
                self._last_stale_check_time = current_time
                if self._sync_from_per_round_state():
                    # Synced to a newer round, reset logged_waiting flag for new round
                    logged_waiting = False

            wait_count += 1
            time.sleep(1.0)  # Poll every 1 second

    def _attribution_gate(self, node_desc: _NodeDesc) -> bool:
        """Return whether attribution permits closing the current round.

        Raises ``RendezvousGracefulExitError`` if attribution recommends stopping.
        Returns ``False`` while attribution is still pending.
        """
        if self._attribution_service is None or self._round == 0:
            return True

        if self._attribution_settled_for_round == self._round:
            return True

        result = self._attribution_service.get_last_result(node_id=node_desc)

        if result is None:
            return False

        if result is True:
            msg = (
                f"The node '{node_desc}' is closing rendezvous because attribution "
                f"recommended stopping the job."
            )
            log.error(msg)
            self.set_shutdown(RDZV_SHUTDOWN_REASON_FAILURE)
            raise RendezvousGracefulExitError(msg)

        self._attribution_settled_for_round = self._round
        return True

    def _host_close_round(
        self,
        node_desc: _NodeDesc,
        min_nodes: int,
        max_nodes: int,
        segment_check_interval: float,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Step 2 (store host only): Poll until segment constraint is met, assign ranks, close round.

        Loops checking the current participant snapshot. When enough healthy nodes are present
        and the segment constraint is satisfied, calls assign_group_ranks() and sets
        round_done_key=1.  All rank keys are written BEFORE round_done_key=1 is set, so
        any participant that sees round_done=1 in Step 3 can immediately read its rank.

        Raises:
            RendezvousClosedError: If shutdown is detected (too many unhealthy nodes).
            RendezvousTimeoutError: If the rendezvous join timeout is exceeded.
        """
        last_segment_check_time = 0.0
        effective_check_interval = self._compute_step2_poll_interval(
            min_nodes, segment_check_interval
        )
        # Track how long a count mismatch has persisted so we can escalate to WARNING
        # after a threshold and give actionable diagnostics before the join timeout fires.
        mismatch_first_seen: Optional[float] = None
        close_gate = self._make_previous_active_last_call_gate(min_nodes, max_nodes)
        last_call_logged_deadline: Optional[float] = None

        while True:
            if stop_event is not None and stop_event.is_set():
                raise RendezvousGracefulExitError("Control rendezvous stop requested")

            self._check_timeout_and_closure(node_desc)

            # Early termination: if too many nodes are unhealthy it is mathematically
            # impossible to complete with min_nodes active participants.
            unhealthy_count = self._get_unhealthy_count()
            if unhealthy_count > (max_nodes - min_nodes):
                msg = (
                    f"Rendezvous cannot complete: {unhealthy_count} unhealthy nodes detected, "
                    f"max possible healthy nodes = {max_nodes - unhealthy_count} < {min_nodes} required. "
                    f"Permanently closing rendezvous to terminate the job."
                )
                log.error(msg)
                self.set_shutdown(RDZV_SHUTDOWN_REASON_FAILURE)
                # Loop: _check_timeout_and_closure will raise RendezvousClosedError next iteration.
                continue

            if not self.store.check([self.join_count_key]):
                time.sleep(0.1)
                continue

            current_joined = int(self.store.get(self.join_count_key).decode('utf-8'))

            # Only attempt snapshot + constraint check when enough nodes have joined.
            if current_joined < min_nodes:
                time.sleep(0.1)
                continue

            if not self._attribution_gate(node_desc):
                time.sleep(0.1)
                continue

            current_time = time.monotonic()
            if (current_time - last_segment_check_time) < effective_check_interval:
                time.sleep(0.1)
                continue
            last_segment_check_time = current_time

            # Fetch a full snapshot to avoid races with partially written slots.
            fetch_start = time.monotonic()
            slot_participants = self.get_all_participants(
                total_participants=current_joined,
                expected_round_id=self._round,
            )
            fetch_elapsed = time.monotonic() - fetch_start

            current_round_participant_count = sum(
                1 for participant in slot_participants if participant is not None
            )

            # join_count counts every node that claimed a slot, including surplus
            # survivors of a failed array task that may never finish writing their
            # slot (SLURM kills them mid-join). A snapshot smaller than the counter
            # is therefore expected and benign: we proceed using the slots that are
            # actually present and let the segment/replacement-group constraint below
            # decide whether enough real nodes have arrived. Missing slots are simply
            # excluded from selection. This is diagnostic-only and no longer blocks.
            if current_round_participant_count != current_joined:
                now = time.monotonic()
                if mismatch_first_seen is None:
                    mismatch_first_seen = now
                mismatch_duration = now - mismatch_first_seen
                if mismatch_duration > 30.0:
                    log.warning(
                        f"[{node_desc}] [Step 2] Snapshot below join_count for "
                        f"{mismatch_duration:.0f}s "
                        f"(present={current_round_participant_count}, "
                        f"join_count={current_joined}, round={self._round}). "
                        f"Likely surplus survivors that incremented join_count without "
                        f"completing their slot write; proceeding with present nodes "
                        f"once the constraint is satisfied."
                    )
                else:
                    log.debug(
                        f"[{node_desc}] [Step 2] Snapshot below join_count "
                        f"(present={current_round_participant_count}, "
                        f"join_count={current_joined}, round={self._round}); "
                        f"proceeding with present nodes once the constraint is satisfied."
                    )
            else:
                mismatch_first_seen = None  # full snapshot; reset diagnostic timer

            # Stability check (anti-race guard, replaces the old exact snapshot==counter
            # gate): ensure join_count did not move during the fetch. While nodes are
            # still actively joining the counter keeps changing and we retry; once it
            # settles -- including the steady inflated value left by dead survivors --
            # we evaluate the constraint.
            current_joined_after = int(self.store.get(self.join_count_key).decode('utf-8'))
            if current_joined_after != current_joined:
                log.debug(
                    f"[{node_desc}] [Step 2] Counters changed during snapshot "
                    f"(joined {current_joined}->{current_joined_after}); retrying."
                )
                continue

            current_round_participants = [
                participant for participant in slot_participants if participant is not None
            ]

            # Check if the segment/replacement-group constraint is satisfied.
            unhealthy_replacement_groups = self._get_unhealthy_replacement_groups()
            complete_replacement_group_participants, incomplete_replacement_group_participants = (
                self._split_by_complete_replacement_groups(current_round_participants)
            )
            check_start = time.monotonic()
            constraint_satisfied = self._can_meet_segment_constraint(
                complete_replacement_group_participants, min_nodes
            )
            check_elapsed = time.monotonic() - check_start

            if not constraint_satisfied:
                continue

            decision = close_gate.evaluate(
                current_round_participants, time.monotonic(), unhealthy_replacement_groups
            )
            if not decision.should_close:
                if decision.last_call_deadline != last_call_logged_deadline:
                    last_call_logged_deadline = decision.last_call_deadline
                    log.info(
                        "[slot=%s] [Step 2] Enough participants are present, but "
                        "waiting up to %.1fs for previous active infra ranks %s "
                        "before accepting hot spares",
                        self._slot,
                        self.last_call_timeout_seconds,
                        list(decision.missing_previous_active),
                    )
                continue
            if decision.missing_previous_active:
                log.info(
                    "[slot=%s] [Step 2] Previous-active last_call expired; "
                    "closing round without infra ranks %s",
                    self._slot,
                    list(decision.missing_previous_active),
                )

            # Defensive misconfiguration guard: two ft_launchers on the same physical
            # node appear as two participants sharing an addr (FQDN). This is the case
            # the old slot>max_nodes check was meant to catch. Legitimate survivors and
            # replacements always live on different physical nodes, so they never
            # collide here. Allowed only in simulation, matching the launch-time
            # _validate_slurm_single_launcher_per_node() override. Evaluated here, on the
            # authoritative snapshot we are about to close with, so it fires once rather
            # than on every poll and never hard-fails on a transient pre-close state.
            if not parse_bool_env("NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE"):
                seen_addrs: Set[str] = set()
                duplicate_addr: Optional[str] = None
                for participant in current_round_participants:
                    addr = participant[0].addr
                    if addr in seen_addrs:
                        # One collision is enough to declare the misconfiguration; stop.
                        duplicate_addr = addr
                        break
                    seen_addrs.add(addr)
                if duplicate_addr is not None:
                    log.error(
                        f"[{node_desc}] [Step 2] Detected multiple participants on the "
                        f"same node ({duplicate_addr}) in round {self._round}. "
                        f"This indicates more than one ft_launcher per node "
                        f"(misconfiguration). Permanently closing the rendezvous. If "
                        f"this is intentional simulation, set "
                        f"NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE=1."
                    )
                    self.set_shutdown(RDZV_SHUTDOWN_REASON_FAILURE)
                    # Loop: _check_timeout_and_closure raises RendezvousClosedError next.
                    continue

            replacement_group_info = ""
            if self.replacement_group_size is not None:
                replacement_group_info = (
                    f", complete_replacement_group_participants="
                    f"{len(complete_replacement_group_participants)} "
                    f"incomplete_replacement_group_participants="
                    f"{len(incomplete_replacement_group_participants)}"
                )
            unhealthy_replacement_group_info = ""
            if unhealthy_replacement_groups:
                unhealthy_replacement_group_info = (
                    f", unhealthy_replacement_groups={sorted(unhealthy_replacement_groups)}"
                )

            log.info(
                f"[slot={self._slot}] [Step 2] Constraint ok: {len(current_round_participants)} "
                f"participants{replacement_group_info}{unhealthy_replacement_group_info}, "
                f"min={min_nodes} (fetch {fetch_elapsed*1000:.1f}ms, check {check_elapsed*1000:.1f}ms)"
            )
            # Assign ranks BEFORE setting round_done=1 so Step 3 readers can get their rank
            # immediately without any additional waiting.
            # Pass the full slot-indexed snapshot (may contain None for missing/late
            # slots) so rank keys are written to the actual slot each participant
            # claimed. Selection buckets are derived from the present participants.
            self.assign_group_ranks(
                min_nodes,
                max_nodes,
                node_desc,
                slot_participants=slot_participants,
                active_candidate_participants=complete_replacement_group_participants,
                standby_only_participants=incomplete_replacement_group_participants,
            )
            self._report_cycle_start_as_host(self._round)
            self.store.set(self.round_done_key, "1".encode('utf-8'))
            return

    def _report_cycle_start_as_host(self, round_id: int) -> None:
        """Report cycle start from the rendezvous-host rank assignment snapshot."""
        if self._cycle_info_reporter is None:
            return
        try:
            self._cycle_info_reporter.report_cycle_start(
                CycleInfoRoundSnapshot(
                    cycle_number=round_id,
                    active_node_addrs=self._active_node_addrs,
                    standby_node_addrs=self._standby_node_addrs,
                    active_ranks=self._active_ranks,
                )
            )
        except Exception as e:
            log.warning(
                "Failed to report cycle info after closing rendezvous round %s: %s",
                round_id,
                e,
                exc_info=True,
            )

    def _report_cycle_end_as_host(self, opened_round_id: int) -> None:
        """Report cycle end from the rendezvous host when a later round is observed."""
        if self._cycle_info_reporter is None:
            return
        try:
            self._cycle_info_reporter.report_cycle_end()
        except Exception as e:
            log.warning(
                "Failed to report cycle info end after opening rendezvous round %s: %s",
                opened_round_id,
                e,
                exc_info=True,
            )

    def _wait_for_round_done(self, node_desc: _NodeDesc, rank_key: str) -> Tuple[int, int]:
        """Step 3 (all participants): Wait for round_done_key=1 then read the assigned rank.

        The store host set round_done_key=1 only after writing all rank keys, so the rank
        read here is guaranteed to be the final assigned value (no polling needed).

        The rank value is wrapped with the round number that wrote it.
        If the embedded round doesn't match self._round, the slot was overwritten by a later
        round (e.g. same port reuse caused a stale slow read); treat as UNASSIGNED so the
        caller loops to the next round.

        Returns:
            Tuple of (group_rank, total_participants). group_rank may be UNASSIGNED
            (late comer), a stale-round sentinel, or >= min_nodes (standby); callers
            handle all non-active cases by looping.

        Raises:
            RendezvousClosedError: If shutdown is detected.
            RendezvousTimeoutError: If the join timeout is exceeded.
            RuntimeError: If the rank value cannot be parsed.
        """
        while True:
            self._check_timeout_and_closure(node_desc)

            # round_done_key always exists here (set to "0" at Step 0).
            value = int(self.store.get(self.round_done_key).decode('utf-8'))
            if value == 1:
                break

            time.sleep(0.1)

        # Race 3 fix: rank_key may not exist when our add() arrived after the store
        # host's double-check (round closed without us, so the host never wrote our
        # rank). A blocking store.get() on a missing key would hang; detect first.
        if not self.store.check([rank_key]):
            log.debug(
                f"[{node_desc}] rank_key {rank_key} absent — late comer, treating as UNASSIGNED"
            )
            return GroupRankStatus.UNASSIGNED.value, 0

        rank_value = self.store.get(rank_key)
        try:
            rank, total, written_round = self._unpack_rank_value(rank_value)
        except (ValueError, AttributeError) as e:
            raise RuntimeError(
                f"[{node_desc}] Failed to parse rank value "
                f"'{self._store_value_to_str(rank_value)}': {e}. "
                f"Expected a round-fenced rank value."
            )

        if written_round != self._round:
            # Slot was overwritten by a later round (key reuse race). Treat as UNASSIGNED
            # so perform_rendezvous() advances _round and retries.
            log.warning(
                f"[{node_desc}] Rank key stale: written_round={written_round} != "
                f"self._round={self._round}. Treating as UNASSIGNED."
            )
            return GroupRankStatus.UNASSIGNED.value, 0

        return rank, total

    def close_current_round_as_host(
        self,
        node_desc: _NodeDesc,
        min_nodes: int,
        max_nodes: int,
        segment_check_interval: float = 5.0,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        """Close one open rendezvous round without joining as a participant.

        This is used by an external control process that owns TCPStore. The
        control process is the store host, but it does not increment
        join_count_key, write a slot, or receive a rank. Participant accounting
        remains compute-only.

        Returns:
            The rendezvous round number that was closed.
        """
        if not self.is_store_host:
            raise RuntimeError("host-only rendezvous close must run on the TCPStore host")

        self._slot = None
        self._last_stale_check_time = 0.0

        self._wait_for_rendezvous_open(node_desc, stop_event=stop_event)
        self._rendezvous_start_time = time.monotonic()
        self._host_close_round(
            node_desc,
            min_nodes,
            max_nodes,
            segment_check_interval,
            stop_event=stop_event,
        )
        return self._round

    def perform_rendezvous(
        self,
        node_desc: _NodeDesc,
        min_nodes: int,
        max_nodes: int,
        segment_check_interval: float = 5.0,
        pre_join_hook: Optional[Callable[[], None]] = None,
    ) -> Tuple[int, int]:
        """Perform the complete rendezvous for this node, returning an active group rank.

        Protocol steps (repeated until an active rank is obtained):

          Step 0 — Wait for round open (all participants):
            Hot spares and late-arriving nodes block here until open_rendezvous() sets
            round_done_N=0, signalling that round N is ready for joining.

          Step 1 — Join (all participants):
            Atomically claim a slot via join_count_key. Write participant metadata and
            an initial UNASSIGNED rank to the slot. A slot index beyond max_nodes
            is benign (surplus/transient node) and is handled as standby; it does
            not fail the rendezvous.

          Step 2 — Close round (store host only):
            Poll the participant snapshot until the segment constraint is satisfied
            (see _host_close_round). Then assign group ranks to all slots and set
            round_done_N=1. Non-hosts skip this step entirely.

          Step 3 — Read rank assignment (all participants):
            Poll round_done_N until it equals 1 (see _wait_for_round_done). Because
            all rank keys are written before round_done_N=1 is set, the rank read is
            always the final assigned value with no additional waiting.

        After Step 3:
          - Active rank (< min_nodes): return to launcher → start training workers.
          - Standby rank (>= min_nodes) or UNASSIGNED (late comer): advance _round and
            loop back to Step 0 to wait for the next round.

        Args:
            node_desc: Node descriptor for this participant
            min_nodes: Minimum number of nodes required for training to proceed
            max_nodes: Target number of nodes (active + standby) selected per round.
                Not a hard admission cap: join_count may transiently exceed it when
                surplus nodes co-join; the surplus is handled as standby.
            segment_check_interval: Base interval in seconds for the store host's snapshot
                check in Step 2. Actual interval is adaptive by min_nodes (1s floor).

        Returns:
            Tuple of (group_rank, total_participants) where group_rank < min_nodes.
        """
        while True:
            # Reset per-round mutable state at the top of each attempt.
            # Reset _last_stale_check_time so _sync_from_per_round_state() fires
            # immediately at Step 0 instead of waiting up to stale_check_interval.
            self._last_stale_check_time = 0.0

            # Step 0: Wait until the current round is open.
            # Hot spares and late-arriving nodes wait here indefinitely until a failure
            # opens the next round. Also checks for permanent shutdown.
            # Note: _wait_for_rendezvous_open() raises RendezvousGracefulExitError on shutdown.
            self._wait_for_rendezvous_open(node_desc)

            # Record start time for timeout monitoring.
            # Start timing AFTER Step 0 completes, since nodes may wait indefinitely at Step 0.
            self._rendezvous_start_time = time.monotonic()

            # Record rendezvous start event — start profiling AFTER waiting for round to open.
            # This ensures hot spares waiting at Step 0 don't skew the rendezvous measurement.
            rendezvous_start_event_id = record_profiling_event(
                ProfilingEvent.RENDEZVOUS_STARTED,
                node_id=node_desc,
            )

            if pre_join_hook is not None:
                try:
                    pre_join_hook()
                except UnhealthyNodeException:
                    try:
                        self._maybe_mark_current_replacement_group_unhealthy()
                    except Exception as e:
                        log.warning("Failed to mark current replacement group unhealthy: %s", e)
                    raise

            # Step 1: Join the rendezvous and get a unique slot identifier.
            # join_count_key is per-round, so each round starts counting from 1.
            self._slot = self.store.add(self.join_count_key, 1)

            # NOTE: a slot index beyond max_nodes is intentionally NOT treated as an
            # error here (an earlier version shut the rendezvous down on slot >
            # max_nodes, which tore down the whole job). It is benign and expected: in
            # SLURM job-array deployments, surviving nodes of a failed task can briefly
            # co-join the round before SLURM terminates them, inflating join_count and
            # pushing legitimate nodes past max_nodes. The slot index is only a
            # join-sequence artifact: it does NOT decide active vs standby. The store
            # host selects the active set from the present participants via the
            # replacement_group/segment logic and writes the resulting rank to each
            # node's actual slot, so a node at a slot > max_nodes may still be chosen
            # active this round (or be standby) -- that is determined by its assigned
            # rank, not its slot. Genuine misconfiguration (multiple ft_launchers per
            # node) is caught at launch time by _validate_slurm_single_launcher_per_node()
            # and, defensively, by the duplicate-addr check in the store host's close path.

            # Determine infrastructure rank
            infra_rank = get_infrastructure_rank()
            replacement_group_id = self._current_replacement_group_id()

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

            slot_key = f"{self.prefix}:slot_{self._slot}"
            rank_key = f"{self.prefix}:slot_{self._slot}_rank"

            # SLOT WRITE PROTOCOL — key reuse safety
            #
            # slot_key and rank_key are reused across rounds (normally bounded by
            # max_nodes) to avoid TCPStore keyspace bloat.  A round with surplus
            # joiners may briefly create slots beyond max_nodes; these are transient
            # and self-clear once the surplus nodes leave and join_count resets next
            # round.  Each value embeds the round that wrote it,
            # so participant-published writes use per-key CAS:
            #
            #   old_value, old_round = get(key)
            #   if old_round > this_round: go back to Step 0
            #   compare_set(key, old_value, new_value_for_this_round)
            #
            # This makes the freshness check and write atomic on the reused key.  A slow
            # writer from round N can refresh older values, but it cannot overwrite
            # state that has already been published for round N+1.
            try:
                self._round_fenced_compare_set(
                    rank_key,
                    self._pack_rank_value(
                        GroupRankStatus.UNASSIGNED.value,
                        0,
                        self._round,
                    ),
                )
                participant_data = self._pack_participant_value(
                    node_desc,
                    infra_rank,
                    domain_id,
                    self._round,
                    replacement_group_id,
                )
                self._round_fenced_compare_set(
                    slot_key,
                    participant_data,
                )
            except _StaleRendezvousRoundError as e:
                log.info(
                    f"[{node_desc}] Detected newer rendezvous round {e.observed_round} "
                    f"while joining round {e.attempted_round}; retrying"
                )
                continue

            log.debug(f"[slot={self._slot}] [Step 1] Joined round {self._round}")

            # Step 2 (store host only): Poll until the segment constraint is satisfied,
            # assign group ranks to all participants, then set round_done_key=1.
            # Non-hosts skip this step entirely and proceed directly to Step 3.
            if self.is_store_host:
                self._host_close_round(node_desc, min_nodes, max_nodes, segment_check_interval)

            # Step 3 (all participants): Wait for round_done_key=1, then read rank.
            # The store host set round_done_key=1 only AFTER writing all rank keys, so
            # any participant that sees round_done=1 can immediately read its rank.
            rank, total_participants = self._wait_for_round_done(node_desc, rank_key)

            if rank != GroupRankStatus.UNASSIGNED.value and rank < min_nodes:
                # Active rank: return to launcher to start training workers.
                return rank, total_participants

            # rank == UNASSIGNED: late comer that joined after the store host's snapshot.
            # rank >= min_nodes: standby node — too many nodes for this round.
            # Both cases: advance to next round and wait at Step 0 for it to open.
            if rank == GroupRankStatus.UNASSIGNED.value:
                log.info(
                    f"[{node_desc}] Late joiner detected for round {self._round} "
                    f"(rank=UNASSIGNED); retrying in round {self._round + 1}"
                )
            else:
                log.info(
                    f"[{node_desc}] Standby (rank={rank}) for round {self._round}; "
                    f"waiting for round {self._round + 1} to open"
                )
            # Loop back to Step 0; _sync_from_per_round_state() will advance _round
            # from N to N+1 when it sees round_done_N=1 (closed).

    def _present_slot_keys(self, keys: List[str]) -> List[str]:
        """Return the subset of ``keys`` that currently exist in the store.

        Used only on the degraded close path, where a dead survivor may have bumped
        join_count without writing its slot, leaving holes in an otherwise dense slot
        range. ``store.check(list)`` is all-or-nothing (True only if every key exists),
        so we divide-and-conquer: a sub-range that checks True is wholly present and
        kept in one call; only sub-ranges containing a hole are split further. This is
        O((holes+1)*log N) round-trips rather than the O(N) of a per-key probe -- it
        must stay cheap at up to 32K participants, where holes are rare.
        """
        present: List[str] = []
        # Half-open [lo, hi) index ranges into keys still to resolve.
        ranges: List[Tuple[int, int]] = [(0, len(keys))]
        while ranges:
            lo, hi = ranges.pop()
            if lo >= hi:
                continue
            segment = keys[lo:hi]
            if self.store.check(segment):
                present.extend(segment)  # entire sub-range present
            elif hi - lo == 1:
                continue  # single key absent -> hole, drop it
            else:
                mid = (lo + hi) // 2
                ranges.append((lo, mid))
                ranges.append((mid, hi))
        return present

    def get_all_participants(
        self,
        total_participants: int,
        start_index: int = 1,
        expected_round_id: Optional[int] = None,
        existing_participants: Optional[List[SlotParticipant]] = None,
    ) -> List[SlotParticipant]:
        """Get participants that have arrived using multi_get.

        Supports incremental fetching by allowing caller to specify a start index
        and provide a list of existing participants to append to.

        Args:
            total_participants: Total number of participants to fetch up to
            start_index: Starting index for fetching (1-based, inclusive). Defaults to 1 (fetch all).
            expected_round_id: If provided, only participants from this round are accepted.
                               Mismatched entries are returned as None.
            existing_participants: Optional list of already-fetched participants to extend.
                                  If provided, new participants are appended to this list.
                                  If None, a new list is created.

        Returns:
            Slot-indexed list in arrival order. Each current-round entry is a tuple
            (node_desc, infra_rank, domain_id, replacement_group_id). Missing,
            malformed, or wrong-round slot values are returned as None.
            domain_id is "none" if segment is not configured.

        Example:
            # Fetch all participants
            all_participants = get_all_participants(100)

            # Incremental fetch: first get 50, then get next 50
            participants = get_all_participants(50)
            participants = get_all_participants(100, start_index=51, existing_participants=participants)
        """
        if start_index < 1:
            raise ValueError(f"start_index must be >= 1, got {start_index}")

        if start_index > total_participants:
            # Nothing to fetch, return existing or empty list
            return existing_participants if existing_participants is not None else []

        # Initialize result list
        participants = existing_participants if existing_participants is not None else []

        # Prepare keys for multi_get (only fetch new participants).
        # Slot keys (slot_N) are reused across rounds; they hold data for the current round
        # and are overwritten at Step 1 of each new round for the same slot number.
        participant_keys = [
            f"{self.prefix}:slot_{i}" for i in range(start_index, total_participants + 1)
        ]

        if not participant_keys:
            # Nothing to fetch
            return participants

        # Fetch slot data without blocking on gaps. TCPStore.multi_get waits until
        # *every* requested key exists, but a dead survivor can bump join_count and
        # then die before writing its slot, leaving a permanent gap; a blind
        # multi_get would stall the close loop until the join timeout fires.
        # Fast path: a single check() confirms all slots are present, then one
        # multi_get fetches them. Degraded path (some slot missing): locate the holes
        # without a per-slot probe storm, fetch only the slots that exist; absent slots
        # stay None and are excluded from selection below.
        if self.store.check(participant_keys):
            participant_data_list = list(self.store.multi_get(participant_keys))
        else:
            present_keys = self._present_slot_keys(participant_keys)
            present_data = self.store.multi_get(present_keys) if present_keys else []
            data_by_key = dict(zip(present_keys, present_data))
            # get() returns None for hole keys absent from the map -> treated as missing.
            participant_data_list = [data_by_key.get(key) for key in participant_keys]

        # Unpack participant information; one entry per slot so list index maps to slot
        # (participants[i] = slot start_index + i). Use None for missing, stale,
        # or malformed slot values; the host only closes on current-round entries.
        for i, participant_data_bytes in enumerate(participant_data_list):
            actual_index = start_index + i

            if not participant_data_bytes:
                log.warning(f"No participant data for slot_{actual_index}; treating as missing")
                participants.append(None)
                continue

            try:
                (
                    round_id,
                    node_desc,
                    infra_rank,
                    domain_id,
                    replacement_group_id,
                ) = self._unpack_participant_value(participant_data_bytes)
                if expected_round_id is not None and round_id != expected_round_id:
                    log.debug(
                        f"Round mismatch for slot_{actual_index}: expected round "
                        f"{expected_round_id}, got {round_id}; treating as missing"
                    )
                    participants.append(None)
                    continue
                participants.append((node_desc, infra_rank, domain_id, replacement_group_id))
            except Exception as e:
                log.warning(f"Failed to unpack participant data for slot_{actual_index}: {e}")
                participants.append(None)

        return participants

    def assign_group_ranks(
        self,
        world_size: int,
        max_nodes: int,
        node_desc: _NodeDesc,
        slot_participants: List[Participant],
        active_candidate_participants: List[Participant],
        standby_only_participants: List[Participant],
    ) -> bool:
        """Assign group ranks to all participants. Called by the store host in Step 2.

        This is called BEFORE setting round_done_key=1 so that any participant reading
        round_done=1 can immediately read its rank without waiting further.

        Args:
            world_size: Target world size for training (number of active participants)
            max_nodes: Target number of selected nodes (active + standby). Not a hard
                cap: the slot space (and join_count) may transiently exceed it when
                surplus nodes co-join; the surplus is simply not selected as active.
            slot_participants: Full slot-indexed snapshot for this round. Entry i maps
                to slot i+1 and may be None for missing/late/dead slots, which are
                skipped (their slot keeps its UNASSIGNED rank from Step 1).
            active_candidate_participants: Participants already selected for active rank
                consideration.
            standby_only_participants: Participants that must receive standby ranks.
        """
        slot_count = len(slot_participants)
        assert slot_count > 0, "Expected at least one participant"

        assigned_group_ranks = self._assign_group_ranks(
            active_candidate_participants,
            world_size,
            standby_only_participants=standby_only_participants,
        )

        # Number of participants actually assigned a rank (present active + standby);
        # reported back to nodes as total_participants. May be < slot_count when the
        # snapshot has gaps (surplus survivors that never finished their slot write).
        total_assigned = len(assigned_group_ranks)

        # Write rank to each active participant's slot_{N}_rank key.
        # These writes happen BEFORE round_done_key is set to 1, ensuring all ranks
        # are visible to participants as soon as they see round_done=1.
        rank_keys = []
        rank_values = []
        self._active_node_addrs = []
        self._standby_node_addrs = []
        self._active_ranks = []
        for slot in range(1, slot_count + 1):
            entry = slot_participants[slot - 1]
            if entry is None:
                # Missing/late/dead slot: leave its UNASSIGNED rank so the node (if it
                # ever materializes) treats itself as a late joiner and waits.
                continue
            node_desc_item = entry[0]
            if node_desc_item in assigned_group_ranks:
                assigned_group_rank = assigned_group_ranks[node_desc_item]

                # Collect addrs for cycle info (active vs standby by group_rank vs world_size)
                if assigned_group_rank < world_size:
                    self._active_node_addrs.append(node_desc_item.addr)
                    self._active_ranks.append(assigned_group_rank)
                else:
                    self._standby_node_addrs.append(node_desc_item.addr)

                rank_key = f"{self.prefix}:slot_{slot}_rank"
                rank_value = self._pack_rank_value(
                    assigned_group_rank,
                    total_assigned,
                    self._round,
                )
                rank_keys.append(rank_key)
                rank_values.append(rank_value)

        self.store.multi_set(rank_keys, rank_values)
        self._remember_previous_active_membership(
            active_candidate_participants, assigned_group_ranks, world_size
        )

    def is_shutdown(self) -> bool:
        """Check if rendezvous is permanently shut down (no further rounds)."""
        if not self.store.check([self.shutdown_key]):
            return False
        return bool(self.store.get(self.shutdown_key))

    def get_shutdown_reason(self) -> Optional[str]:
        """Return the recorded shutdown reason, if one was written."""
        if not self.store.check([self.shutdown_key]):
            return None
        return self.store.get(self.shutdown_key).decode('utf-8') or None

    def set_shutdown(self, reason: str = RDZV_SHUTDOWN_REASON_GRACEFUL) -> None:
        """Mark rendezvous as permanently shut down (graceful exit, no further rounds)."""
        try:
            self.store.compare_set(
                self.shutdown_key,
                b"",
                reason.encode('utf-8'),
            )
        except Exception as e:
            log.error(f"Failed to set shutdown: {e}")

    def open_rendezvous(self):
        """Open the next round (N+1) for participants to join after failure detection.

        After completing round N, _round=N and round_done_N=1 (closed). This creates
        round_done_{N+1}=0, which is the key polled by:
          - is_next_round_open() on healthy training nodes
          - _wait_for_rendezvous_open() (via _sync_from_per_round_state) on hot spares
            and restarting nodes

        RACE CONDITION SAFETY:
        Multiple launchers detecting the same failure concurrently all target the same
        round_done_{N+1} key and write the same "0" value — this is idempotent.

        The check() guard ensures only the first caller writes; subsequent callers defer:
        - If round_done_{N+1} does not exist: safe to open → set to "0"
        - If round_done_{N+1} already exists: another node already opened it → defer
        """
        next_round_done_key = f"{self.prefix}:round_done_{self._round + 1}"
        if self.store.check([next_round_done_key]):
            log.debug(f"open_rendezvous() deferred - round_done_{self._round + 1} already exists.")
            self._report_cycle_end_as_host(self._round + 1)
            return

        self.store.set(next_round_done_key, "0".encode('utf-8'))
        log.debug(f"Opened round {self._round + 1} for joining")
        self._report_cycle_end_as_host(self._round + 1)


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
        replacement_group_size: Optional[int] = None,
        nproc_per_node: int = 1,  # Number of processes per node
        enable_nic_healthcheck: bool = False,
        enable_dist_storage_healthcheck: bool = False,
        link_state_path_template: Optional[str] = None,
        storage_healthcheck_paths: Optional[list] = None,
        attribution_endpoint: Optional[str] = None,
        attribution_decision_timeout: Optional[float] = None,
        cycle_info_dir: Optional[str] = None,
        cycle_log_prefix: Optional[str] = None,
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
            replacement_group_size:
                Number of nodes required for a complete replacement group.
            enable_nic_healthcheck:
                Whether to enable all NIC link state health check before rendezvous.
            enable_dist_storage_healthcheck:
                Whether to enable distributed storage health check before rendezvous.
            link_state_path_template:
                Template path for NIC link state files.
            storage_healthcheck_paths:
                List of storage paths to check for health.
            attribution_endpoint:
                Endpoint of the attribution service.
            attribution_decision_timeout:
                Launcher-side attribution decision budget in seconds.
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
            replacement_group_size=replacement_group_size,
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
            attribution_endpoint=attribution_endpoint,
            attribution_decision_timeout=attribution_decision_timeout,
            cycle_info_dir=cycle_info_dir,
            cycle_log_prefix=cycle_log_prefix,
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
        attribution_endpoint: Optional[str] = None,
        attribution_decision_timeout: Optional[float] = None,
        cycle_info_dir: Optional[str] = None,
        cycle_log_prefix: Optional[str] = None,
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

        if settings.segment is not None and settings.min_nodes % settings.segment != 0:
            raise ValueError(
                f"The minimum number of nodes ({settings.min_nodes}) must be divisible by "
                f"segment size ({settings.segment})."
            )
        if settings.replacement_group_size is not None and settings.replacement_group_size < 1:
            raise ValueError("replacement_group_size must be a positive integer when configured.")

        self._this_node = node
        self._settings = settings
        self._backend_name = backend_name
        self._store = store
        self._barrier_state = _RendezvousBarrierState(
            store,
            settings.run_id,
            is_store_host,
            settings.timeout.join.total_seconds(),
            settings.timeout.last_call.total_seconds(),
            settings.segment,
            settings.replacement_group_size,
            stale_check_interval=10.0,  # Check for stale rounds every 10 seconds
        )
        self._cycle_info_reporter: Optional[CycleInfoReporter] = None
        if is_store_host and cycle_info_dir:
            self._cycle_info_reporter = CycleInfoReporter(
                cycle_info_dir,
                cycle_log_prefix=cycle_log_prefix,
            )
            self._barrier_state._cycle_info_reporter = self._cycle_info_reporter
        self._assigned_rank = None
        self._world_size = None
        self._agent = None  # Reference to the agent (set via set_agent())

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

        # Attribution service client (optional, only on master node)
        if is_store_host and attribution_endpoint:
            self._attribution_service = AttributionService(
                endpoint=attribution_endpoint,
                decision_timeout=attribution_decision_timeout,
            )
        else:
            self._attribution_service = None
        self._barrier_state._attribution_service = self._attribution_service

    @property
    def _rendezvous_round(self) -> int:
        """Delegate to barrier state's round counter."""
        return self._barrier_state._round

    @_rendezvous_round.setter
    def _rendezvous_round(self, value: int) -> None:
        """Delegate to barrier state's round counter."""
        self._barrier_state._round = value

    def set_worker_group(self, worker_group: Any) -> None:
        """Set the worker group reference for this handler."""
        self._worker_group = worker_group

    def set_agent(self, agent: Any) -> None:
        """Set the agent reference for this handler.

        This allows the handler to call back to the agent when important events occur,
        such as rendezvous round updates that require agent state synchronization.

        Args:
            agent: The LocalElasticAgent instance
        """
        self._agent = agent
        # Pass agent reference to barrier state for progress tracking
        self._barrier_state._agent = agent
        # Complete initialization now that agent is set
        self._complete_initialization()

    def _complete_initialization(self) -> None:
        """Complete initialization tasks that require agent reference.

        This is called after set_agent() to perform initialization tasks that
        need to sync with the agent, such as syncing the rendezvous round for
        cross-job coordination (replacement nodes joining an ongoing job).
        """
        old_round = self._rendezvous_round
        if self._barrier_state._sync_from_per_round_state():
            log.info(
                f"[{self._this_node}] Synced rendezvous round at initialization "
                f"({old_round} -> {self._rendezvous_round})"
            )

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

        # Perform failure injection check (testing only; no-op unless NVRX_INJECT_GPU_FAILURE is set)
        if os.environ.get("NVRX_INJECT_GPU_FAILURE"):
            cycle = get_profiling_cycle()
            infra_rank = get_infrastructure_rank(skip_nodename_logic=True)
            if health_check_injector.should_inject_failure(cycle, infra_rank):
                raise UnhealthyNodeException(
                    f"Injected failure: infra_rank={infra_rank}, cycle={cycle}"
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

        def pre_join_hook() -> None:
            health_check_start = time.monotonic()
            try:
                self.ensure_node_is_healthy()
            finally:
                health_check_elapsed = time.monotonic() - health_check_start
                record_profiling_event(
                    ProfilingEvent.HEALTH_CHECK_COMPLETED,
                    node_id=self._this_node,
                )
                log.debug(
                    f"[{self._this_node}] Node health check completed in {health_check_elapsed:.3f}s"
                )
            self.handle_control_requests_from_rank()

        # Perform complete rendezvous process
        group_rank, total_participants = self._barrier_state.perform_rendezvous(
            self._this_node,
            self._settings.min_nodes,
            self._settings.max_nodes,
            pre_join_hook=pre_join_hook,
        )

        # Store the assigned rank and world size
        self._assigned_rank = group_rank

        # World size for the training job is the number of *active* participants (min_nodes).
        # perform_rendezvous() only returns with an active rank (< min_nodes); standby nodes
        # loop inside perform_rendezvous() until they become active. All workers therefore
        # see WORLD_SIZE = min_nodes * nproc_per_node (e.g. 32*4=128), not total_participants.
        # This is required when TORCH_ELASTIC_WORKER_IDENTICAL=1.
        self._world_size = self._settings.min_nodes

        log.info(f"Assigned active rank {group_rank} (round={self._barrier_state._round})")

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

        prev_signal_handlers = _install_rdzv_signal_handlers()
        try:
            # Perform the complete rendezvous process
            # Stale round detection and sync happens automatically in _wait_for_rendezvous_open()
            self._perform_rendezvous()

            # Use stored rank and world size
            rank = self._assigned_rank
            world_size = self._world_size
            store = self._get_store()

            # perform_rendezvous() only returns with an active rank (< min_nodes).
            # Standby nodes loop inside perform_rendezvous() until they become active.
            # Restore local_world_size for nodes that were previously standby (had 0 workers).
            assert (
                self._worker_group is not None
            ), "set_worker_group must be called before next_rendezvous"
            if self._worker_group.spec.local_world_size == 0:
                self._worker_group.spec.local_world_size = self._settings.nproc_per_node

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise
        finally:
            _restore_rdzv_signal_handlers(prev_signal_handlers)

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
        """See base class. Returns True if rendezvous is permanently shut down."""
        try:
            return self._barrier_state.is_shutdown()
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
        """See base class.

        Returns the number of nodes currently waiting to join the rendezvous.
        Uses per-round join_count_key; if the key doesn't exist for the current
        round (e.g. during training between rounds), returns 0 with no false positives.
        """
        if not self._barrier_state.store.check([self._barrier_state.join_count_key]):
            return 0
        joined_count_bytes = self._barrier_state.store.get(self._barrier_state.join_count_key)
        return int(joined_count_bytes.decode('utf-8'))

    def is_next_round_open(self) -> bool:
        """Return True if the next rendezvous round has been opened by any node in the cluster.

        Healthy nodes in the monitor loop call this to detect that a peer has triggered
        a restart (open_rendezvous() sets round_done_{N+1}=0 as the signal).
        """
        return self._barrier_state.is_next_round_open()

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

    def get_active_node_addrs(self) -> Optional[List[str]]:
        """Return active node addresses from the last completed rendezvous.

        Only the TCPStore host populates this (in assign_group_ranks). Other nodes
        will get None. Used by the launcher to build NVRx cycle info (active_nodes).
        """
        return self._barrier_state._active_node_addrs

    def get_standby_node_addrs(self) -> Optional[List[str]]:
        """Return standby (hot spare) node addresses from the last rendezvous.

        Only the TCPStore host populates this. Other nodes will get None.
        Used by the launcher to build NVRx cycle info (standby_nodes).
        """
        return self._barrier_state._standby_node_addrs

    def get_active_ranks(self) -> Optional[List[int]]:
        """Return group ranks of active nodes in SLURM hostname sort order.

        active_ranks[i] corresponds to the i-th node in hostnames_to_slurm_nodelist()
        expansion, enabling strict per-element rank-to-node alignment.
        Only the TCPStore host populates this; other nodes return None.
        """
        addrs = self._barrier_state._active_node_addrs
        ranks = self._barrier_state._active_ranks
        if addrs is None or ranks is None:
            return None
        addr_to_rank = dict(zip(addrs, ranks))
        return [addr_to_rank[a] for a in slurm_sort_addrs(addrs)]

    def shutdown(self, reason: str = RDZV_SHUTDOWN_REASON_GRACEFUL) -> bool:
        """See base class."""
        try:
            self._close(reason)
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

    def shutdown_due_to_failure(self) -> bool:
        return self.shutdown(reason=RDZV_SHUTDOWN_REASON_FAILURE)

    def is_shutdown_due_to_failure(self) -> bool:
        return self._barrier_state.get_shutdown_reason() == RDZV_SHUTDOWN_REASON_FAILURE

    def shutdown_cycle_info_reporter(self) -> None:
        if self._cycle_info_reporter is not None:
            self._cycle_info_reporter.shutdown()

    def _close(self, reason: str = RDZV_SHUTDOWN_REASON_GRACEFUL) -> None:
        """Permanently shut down the rendezvous (no further rounds)."""
        self._barrier_state.set_shutdown(reason)
        self.shutdown_cycle_info_reporter()

        msg = f"The node '{self._this_node}' has permanently closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.info(msg)

    def _get_store(self) -> Store:
        """Get the store for this rendezvous.

        Uses round number in the prefix to isolate MASTER_ADDR/MASTER_PORT keys
        between rendezvous rounds, preventing race conditions when rank 0 changes.

        _rendezvous_round == N after completing round N, so each training session
        uses namespace N — a unique, naturally incrementing prefix per round.
        """
        key_prefix = f"torch.rendezvous.{self._settings.run_id}.{self._rendezvous_round}"

        return PrefixStore(key_prefix, self._store)


def _get_timeout(params: RendezvousParameters, key: str) -> Optional[timedelta]:
    timeout = params.get_as_int(key + "_timeout")
    if timeout is None:
        return None
    return timedelta(seconds=timeout)


def _get_replacement_group_size(params: RendezvousParameters) -> Optional[int]:
    replacement_group_size = params.get_as_int("replacement_group_size")
    if replacement_group_size is not None:
        return replacement_group_size

    if not is_slurm_job_array():
        return None

    nnodes_per_group = os.getenv("SLURM_NNODES", os.getenv("SLURM_JOB_NUM_NODES"))
    if nnodes_per_group is None:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is set but SLURM_NNODES/SLURM_JOB_NUM_NODES is not "
            "defined. Cannot derive replacement_group_size for SLURM job-array rendezvous."
        )
    return int(nnodes_per_group)


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
    | last_call_timeout          | Restart-round grace period, in seconds, after enough |
    |                            | hot-spare candidates are present while previous      |
    |                            | active nodes are still missing. Defaults to 1        |
    |                            | second. A value of 0 disables the wait. Only applies |
    |                            | when max_nodes > min_nodes.                          |
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
    |                            | (disabled). Rendezvous completes as soon as enough   |
    |                            | participants arrive to satisfy the segment           |
    |                            | constraint.                                          |
    +----------------------------+------------------------------------------------------+
    | replacement_group_size         | Static node count for a complete replacement group. When |
    |                            | set, active rank assignment only considers complete  |
    |                            | replacement groups. Defaults to SLURM_NNODES/            |
    |                            | SLURM_JOB_NUM_NODES for SLURM job arrays and None    |
    |                            | otherwise.                                           |
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
            join=_get_timeout(params, "join"),
            last_call=_get_timeout(params, "last_call"),
            close=_get_timeout(params, "close"),
        )

        # Get is_store_host from parameters
        is_store_host = params.config.get('is_store_host', False)
        segment = params.config.get('segment', None)
        replacement_group_size = _get_replacement_group_size(params)
        nproc_per_node = params.config.get('nproc_per_node', 1)
        enable_nic_healthcheck = params.config.get('enable_nic_healthcheck', False)
        enable_dist_storage_healthcheck = params.config.get(
            'enable_dist_storage_healthcheck', False
        )
        storage_healthcheck_paths = params.config.get('storage_healthcheck_paths', None)
        link_state_path_template = params.config.get('link_state_path_template', None)
        attribution_endpoint = params.config.get('attribution_endpoint', None)
        attribution_decision_timeout = params.config.get('attribution_decision_timeout', None)
        if attribution_decision_timeout is not None:
            attribution_decision_timeout = float(attribution_decision_timeout)
        cycle_info_dir = params.config.get('cycle_info_dir', None)
        cycle_log_prefix = params.config.get('cycle_log_prefix', None)

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
            replacement_group_size=replacement_group_size,
            nproc_per_node=nproc_per_node,
            enable_nic_healthcheck=enable_nic_healthcheck,
            enable_dist_storage_healthcheck=enable_dist_storage_healthcheck,
            link_state_path_template=link_state_path_template,
            storage_healthcheck_paths=storage_healthcheck_paths,
            attribution_endpoint=attribution_endpoint,
            attribution_decision_timeout=attribution_decision_timeout,
            cycle_info_dir=cycle_info_dir,
            cycle_log_prefix=cycle_log_prefix,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
