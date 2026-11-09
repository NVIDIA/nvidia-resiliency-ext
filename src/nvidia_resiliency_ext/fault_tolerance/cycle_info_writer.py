# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NVRx cycle info writer: writes NVRxCycleInfo as JSON to Lustre (or local).

Uses the NVRxCycleInfo protobuf and protobuf JSON serialization. A dedicated background
thread performs file I/O so the main launcher thread is not blocked. Writes are
atomic (temp file + os.replace) for updates.
"""

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from google.protobuf import json_format

from nvidia_resiliency_ext.shared_utils.job_metadata import job_id_from_env
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2

from .cycle_outcome import CycleEndReason, LocalCycleOutcome
from .utils import hostnames_to_slurm_nodelist, ranks_to_range_str, slurm_sort_addrs

logger = logging.getLogger(LogConfig.name)


def _cycle_info_filename(job_id: str, attempt_index: int, cycle_number: int) -> str:
    return f"cycle_info.{job_id}.{attempt_index}.{cycle_number}"


def _current_symlink_name(job_id: str) -> str:
    return f"cycle_info.{job_id}.current"


def cycle_log_file(base_log_file: str, cycle_index: int) -> str:
    base_without_ext, ext = os.path.splitext(os.path.abspath(base_log_file))
    return f"{base_without_ext}_cycle{cycle_index}{ext or '.log'}"


def _cycle_info_job_id_from_env() -> str:
    return job_id_from_env() or ""


def _cycle_info_attempt_index_from_env() -> int:
    raw_attempt_index = os.environ.get("SLURM_RESTART_COUNT", "0")
    try:
        return int(raw_attempt_index)
    except ValueError:
        logger.warning(
            "Invalid SLURM_RESTART_COUNT=%r for cycle info; using 0",
            raw_attempt_index,
        )
        return 0


@dataclass(frozen=True)
class CycleInfoRoundSnapshot:
    cycle_number: int
    active_node_addrs: Optional[Sequence[str]] = None
    standby_node_addrs: Optional[Sequence[str]] = None
    active_ranks: Optional[Sequence[int]] = None
    cycle_log_file: str = ""


@dataclass(frozen=True)
class _ReportedCycle:
    job_id: str
    attempt_index: int
    cycle_number: int


class _CycleInfoTask:
    """Single task for the writer thread."""

    CREATE = "create"
    UPDATE = "update"
    SHUTDOWN = "shutdown"

    def __init__(self, op: str, **kwargs: Any):
        self.op = op
        self.kwargs = kwargs
        self.enqueued_at = time.monotonic()


class CycleInfoWriter:
    """
    Writes NVRx cycle info JSON files under <base_dir>/nvrx/ on a dedicated thread.

    The rendezvous host should call write_cycle_start() after rendezvous and
    update_cycle_end() when the cycle ends.
    """

    def __init__(self, cycle_info_dir: str) -> None:
        """Initialize with the full path to the cycle info directory (e.g. <base>/nvrx/)."""
        if not cycle_info_dir or not cycle_info_dir.strip():
            raise ValueError("cycle_info_dir must be non-empty")
        self._nvrx_dir = os.path.abspath(cycle_info_dir.rstrip("/"))
        self._queue: queue.Queue[Optional[_CycleInfoTask]] = queue.Queue()
        self._shutdown_requested = False
        # Track the current cycle (job_id, attempt_index, cycle_number) from write_cycle_start
        # so shutdown() can write cycle_end_time if the launcher never called update_cycle_end.
        self._current_cycle: Optional[Tuple[str, int, int]] = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.debug("CycleInfoWriter started with cycle_info_dir=%s", self._nvrx_dir)

    def write_cycle_start(
        self,
        job_id: str,
        attempt_index: int,
        cycle_number: int,
        cycle_start_time: str,
        cycle_log_file: str,
        active_nodes: str,
        standby_nodes: str = "",
        active_ranks: str = "",
    ) -> None:
        """Enqueue writing the initial cycle info file and updating the .current symlink."""
        if self._shutdown_requested:
            logger.warning("CycleInfoWriter already shutdown, ignoring write_cycle_start")
            return
        payload: Dict[str, Any] = {
            "job_id": job_id,
            "attempt_index": attempt_index,
            "cycle_number": cycle_number,
            "cycle_start_time": cycle_start_time,
            "cycle_end_time": "",
            "cycle_log_file": cycle_log_file,
            "active_nodes": active_nodes,
            "standby_nodes": standby_nodes,
            "active_ranks": active_ranks,
        }
        self._current_cycle = (job_id, attempt_index, cycle_number)
        self._queue.put(_CycleInfoTask(op=_CycleInfoTask.CREATE, payload=payload))

    def update_cycle_end(
        self,
        job_id: str,
        attempt_index: int,
        cycle_number: int,
        cycle_end_time: str,
    ) -> None:
        """Enqueue an atomic update of the cycle file to set cycle_end_time."""
        if self._shutdown_requested:
            logger.warning("CycleInfoWriter already shutdown, ignoring update_cycle_end")
            return
        # Clear current cycle so shutdown() will not write cycle_end again
        if self._current_cycle == (job_id, attempt_index, cycle_number):
            self._current_cycle = None
        update_payload: Dict[str, Any] = {"cycle_end_time": cycle_end_time}
        self._queue.put(
            _CycleInfoTask(
                op=_CycleInfoTask.UPDATE,
                job_id=job_id,
                attempt_index=attempt_index,
                cycle_number=cycle_number,
                update_payload=update_payload,
            )
        )

    def update_cycle_later(
        self,
        job_id: str,
        attempt_index: int,
        cycle_number: int,
        build_update: Callable[[], Optional[Dict[str, Any]]],
    ) -> None:
        """Enqueue an atomic update of the cycle file whose payload is built on the writer thread.

        For updates that are expensive to prepare: the caller only enqueues `build_update`.
        If it returns no payload, the file is left unchanged.
        """
        if self._shutdown_requested:
            logger.warning("CycleInfoWriter already shutdown, ignoring update_cycle_later")
            return
        self._queue.put(
            _CycleInfoTask(
                op=_CycleInfoTask.UPDATE,
                job_id=job_id,
                attempt_index=attempt_index,
                cycle_number=cycle_number,
                build_update=build_update,
            )
        )

    def shutdown(self) -> None:
        """Signal the worker to exit and wait for it (drains queue).
        If a cycle was started (write_cycle_start) but cycle end was never recorded
        (update_cycle_end not called, e.g. launcher error path), enqueue an update
        to set cycle_end_time to now so the cycle file is still complete.
        """
        self._shutdown_requested = True
        if self._current_cycle is not None:
            job_id, attempt_index, cycle_number = self._current_cycle
            self._current_cycle = None
            self._queue.put(
                _CycleInfoTask(
                    op=_CycleInfoTask.UPDATE,
                    job_id=job_id,
                    attempt_index=attempt_index,
                    cycle_number=cycle_number,
                    update_payload={"cycle_end_time": utc_iso_now()},
                )
            )
        self._queue.put(_CycleInfoTask(op=_CycleInfoTask.SHUTDOWN))
        join_timeout = 10.0
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            logger.warning("CycleInfoWriter thread did not exit within %.0fs", join_timeout)

    def _worker(self) -> None:
        while True:
            try:
                if self._shutdown_requested:
                    try:
                        task = self._queue.get_nowait()
                    except queue.Empty:
                        break
                else:
                    task = self._queue.get()
                if task is None:
                    continue
                if task.op == _CycleInfoTask.SHUTDOWN:
                    continue
                if task.op == _CycleInfoTask.CREATE:
                    self._do_create(task.kwargs["payload"])
                elif task.op == _CycleInfoTask.UPDATE:
                    enqueued_at = task.enqueued_at
                    prepare_seconds = 0.0
                    update_payload = task.kwargs.get("update_payload")
                    build_update = task.kwargs.get("build_update")
                    if build_update is not None:
                        start = time.monotonic()
                        update_payload = build_update()
                        prepare_seconds = time.monotonic() - start
                        # Report queueing and preparation separately.
                        enqueued_at += prepare_seconds
                    if not update_payload:
                        continue
                    self._do_update(
                        task.kwargs["job_id"],
                        task.kwargs["attempt_index"],
                        task.kwargs["cycle_number"],
                        update_payload,
                        enqueued_at=enqueued_at,
                        prepare_ms=prepare_seconds * 1000,
                    )
            except Exception as e:
                logger.exception("CycleInfoWriter task failed: %s", e)

    def _do_create(self, payload: Dict[str, Any]) -> None:
        """Write a new cycle info file from a payload dict (keys match NVRxCycleInfo fields)."""
        msg = nvrx_interface_pb2.NVRxCycleInfo()
        for key, value in payload.items():
            if hasattr(msg, key):
                setattr(msg, key, value)
        msg.generation = 0
        job_id = msg.job_id
        attempt_index = msg.attempt_index
        cycle_number = msg.cycle_number
        os.makedirs(self._nvrx_dir, exist_ok=True)
        filename = _cycle_info_filename(job_id, attempt_index, cycle_number)
        path = os.path.join(self._nvrx_dir, filename)
        try:
            with open(path, "w") as f:
                f.write(
                    json_format.MessageToJson(
                        msg,
                        indent=2,
                        preserving_proto_field_name=True,
                        always_print_fields_with_no_presence=True,
                    )
                )
                f.flush()
        except OSError as e:
            logger.warning("Failed to write cycle info file %s: %s", path, e)
            try:
                os.remove(path)
            except OSError:
                pass
            return
        # Update symlink to point to this cycle file (atomic replace on same filesystem)
        symlink_name = _current_symlink_name(job_id)
        symlink_path = os.path.join(self._nvrx_dir, symlink_name)
        tmp_symlink_path = symlink_path + ".tmp"
        try:
            os.symlink(filename, tmp_symlink_path)
            os.replace(tmp_symlink_path, symlink_path)
        except OSError as e:
            logger.warning("Failed to update cycle_info symlink %s: %s", symlink_path, e)
            try:
                os.remove(tmp_symlink_path)
            except OSError:
                pass

    def _do_update(
        self,
        job_id: str,
        attempt_index: int,
        cycle_number: int,
        update_payload: Dict[str, Any],
        enqueued_at: Optional[float] = None,
        prepare_ms: float = 0.0,
    ) -> None:
        """Update the cycle info file by applying update_payload; increment generation by one."""
        start = time.monotonic()
        filename = _cycle_info_filename(job_id, attempt_index, cycle_number)
        path = os.path.join(self._nvrx_dir, filename)
        if not os.path.isfile(path):
            logger.warning("Cycle info file not found for update: %s", path)
            return

        tmp_path = path + ".tmp"
        try:
            with open(path) as f:
                msg = json_format.Parse(f.read(), nvrx_interface_pb2.NVRxCycleInfo())
            for key, value in update_payload.items():
                if not hasattr(msg, key):
                    continue
                if isinstance(value, list):
                    # Repeated fields cannot be assigned; replace their contents.
                    msg.ClearField(key)
                    json_format.ParseDict({key: value}, msg)
                else:
                    setattr(msg, key, value)
            msg.generation = msg.generation + 1
            text = json_format.MessageToJson(
                msg,
                indent=2,
                preserving_proto_field_name=True,
                always_print_fields_with_no_presence=True,
            )
            with open(tmp_path, "w") as f:
                f.write(text)
                f.flush()
            os.replace(tmp_path, path)
            logger.debug(
                "Updated %s (%s, %d bytes) in %.1fms after %.1fms queued, %.1fms preparing",
                filename,
                ", ".join(update_payload),
                len(text),
                (time.monotonic() - start) * 1000,
                (start - enqueued_at) * 1000 if enqueued_at is not None else 0.0,
                prepare_ms,
            )
        except (OSError, json_format.ParseError) as e:
            logger.warning("Failed to update cycle info in %s: %s", path, e)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def get_cycle_info_dir(self) -> str:
        """Return the cycle info directory path."""
        return self._nvrx_dir

    def get_current_cycle_info_path(self, job_id: str) -> str:
        """Return the full path to the cycle_info.<job_id>.current symlink (for NVRX_CURRENT_CYCLE_INFO env)."""
        return os.path.join(self._nvrx_dir, _current_symlink_name(job_id))


class CycleInfoReporter:
    """Owns cycle-info lifecycle reporting for a rendezvous host."""

    def __init__(
        self,
        cycle_info_dir: str,
        cycle_log_prefix: Optional[str] = None,
        cycle_info_job_id: Optional[str] = None,
        attempt_index: Optional[int] = None,
        writer: Optional[CycleInfoWriter] = None,
    ) -> None:
        if cycle_info_job_id is not None and not cycle_info_job_id.strip():
            raise ValueError("cycle_info_job_id must be non-empty when set")
        self._writer = writer or CycleInfoWriter(cycle_info_dir)
        self._cycle_log_prefix = cycle_log_prefix
        self._cycle_info_job_id = cycle_info_job_id
        self._attempt_index = attempt_index
        self._current_cycle: Optional[_ReportedCycle] = None
        self._started_cycles: set[int] = set()
        # Active launchers of each started cycle, until its outcomes are reported.
        self._active_launchers: Dict[int, int] = {}
        self._shutdown = False

    @staticmethod
    def current_cycle_info_path(cycle_info_dir: str, job_id: str) -> str:
        return os.path.join(
            os.path.abspath(cycle_info_dir.rstrip("/")),
            _current_symlink_name(job_id),
        )

    def report_cycle_start(
        self,
        snapshot: CycleInfoRoundSnapshot,
    ) -> bool:
        if self._shutdown:
            logger.warning("CycleInfoReporter already shutdown, ignoring cycle start")
            return False
        if snapshot.cycle_number in self._started_cycles:
            return False

        job_id, attempt_index = self._resolve_file_namespace()
        if self._current_cycle is not None:
            self.report_cycle_end()

        self._writer.write_cycle_start(
            job_id=job_id,
            attempt_index=attempt_index,
            cycle_number=snapshot.cycle_number,
            cycle_start_time=utc_iso_now(),
            cycle_log_file=self._resolve_cycle_log_file(snapshot),
            active_nodes=self._format_nodes(snapshot.active_node_addrs),
            standby_nodes=self._format_nodes(snapshot.standby_node_addrs),
            active_ranks=self._format_active_ranks(snapshot),
        )
        self._current_cycle = _ReportedCycle(job_id, attempt_index, snapshot.cycle_number)
        self._started_cycles.add(snapshot.cycle_number)
        self._active_launchers[snapshot.cycle_number] = len(snapshot.active_ranks or ())
        return True

    def report_cycle_end(self) -> bool:
        if self._shutdown or self._current_cycle is None:
            return False
        current_cycle = self._current_cycle
        self._current_cycle = None
        self._writer.update_cycle_end(
            job_id=current_cycle.job_id,
            attempt_index=current_cycle.attempt_index,
            cycle_number=current_cycle.cycle_number,
            cycle_end_time=utc_iso_now(),
        )
        return True

    def report_cycle_outcomes(
        self, cycle_number: int, collect: Callable[[], Sequence[LocalCycleOutcome]]
    ) -> bool:
        """Add the launchers' outcomes of `cycle_number` to its cycle info file.

        `collect` returns the outcomes; it runs on the writer thread, together with the
        formatting and the write, so the caller only enqueues (e.g. decoding records it
        already read). Launchers that only followed
        a peer's restart are counted but not listed; the listed ones are sorted by when they
        observed the end.
        """
        if self._shutdown:
            return False
        job_id, attempt_index = self._resolve_file_namespace()
        launchers_expected = self._active_launchers.pop(cycle_number, 0)

        def build_update() -> Optional[Dict[str, Any]]:
            outcomes = collect()
            if not outcomes:
                return None
            listed = sorted(
                (
                    outcome
                    for outcome in outcomes
                    if outcome.node_end is not CycleEndReason.PEER_RESTART
                ),
                key=lambda outcome: (outcome.detected_at, outcome.node),
            )
            return {
                "launcher_outcomes": [self._format_outcome(outcome) for outcome in listed],
                "outcome_launchers_expected": launchers_expected,
                "outcome_launchers_reported": len(outcomes),
            }

        self._writer.update_cycle_later(job_id, attempt_index, cycle_number, build_update)
        return True

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self.report_cycle_end()
        self._shutdown = True
        self._writer.shutdown()

    def _resolve_file_namespace(self) -> Tuple[str, int]:
        if self._cycle_info_job_id is not None:
            return self._cycle_info_job_id, self._attempt_index or 0
        return _cycle_info_job_id_from_env(), _cycle_info_attempt_index_from_env()

    def _resolve_cycle_log_file(self, snapshot: CycleInfoRoundSnapshot) -> str:
        if snapshot.cycle_log_file:
            return snapshot.cycle_log_file
        if self._cycle_log_prefix:
            return cycle_log_file(self._cycle_log_prefix, snapshot.cycle_number)
        return ""

    @staticmethod
    def _format_nodes(addrs: Optional[Sequence[str]]) -> str:
        return hostnames_to_slurm_nodelist(list(addrs)) if addrs else ""

    @staticmethod
    def _format_active_ranks(snapshot: CycleInfoRoundSnapshot) -> str:
        if snapshot.active_ranks is None:
            return ""
        active_ranks = list(snapshot.active_ranks)
        active_addrs = list(snapshot.active_node_addrs or [])
        if active_addrs and len(active_addrs) == len(active_ranks):
            addr_to_rank = dict(zip(active_addrs, active_ranks))
            active_ranks = [addr_to_rank[addr] for addr in slurm_sort_addrs(active_addrs)]
        return ranks_to_range_str(active_ranks)

    @staticmethod
    def _format_outcome(outcome: LocalCycleOutcome) -> Dict[str, Any]:
        """NVRxLauncherOutcome fields of `outcome`, as a JSON-style dict."""
        ranks = []
        for rank in outcome.ranks:
            fields: Dict[str, Any] = {
                "global_rank": rank.global_rank,
                "local_rank": rank.local_rank,
                "pid": rank.pid,
                "end": rank.end.value,
                "timeout_sections": list(rank.timeout_sections),
            }
            if rank.exitcode is not None:
                fields["exitcode"] = rank.exitcode
            if rank.exitdetails:
                fields["exit_details"] = rank.exitdetails
            if rank.timeout_kind is not None:
                fields["timeout_kind"] = rank.timeout_kind.value
            ranks.append(fields)
        return {
            "node": outcome.node,
            "end": outcome.node_end.value,
            "detected_at": utc_iso(outcome.detected_at),
            "ranks": ranks,
        }


def utc_iso(timestamp: float) -> str:
    """Return `timestamp` (seconds since the epoch) in ISO 8601 UTC, like utc_iso_now()."""
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def utc_iso_now() -> str:
    """Return current UTC time in ISO 8601 format (e.g. for cycle_start_time / cycle_end_time)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
