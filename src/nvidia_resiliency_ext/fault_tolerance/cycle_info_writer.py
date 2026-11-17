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
NVRx cycle info writer: writes CycleInfo as JSON to Lustre (or local).

Uses the CycleInfo protobuf and protobuf JSON serialization. A dedicated background
thread performs file I/O so the main launcher thread is not blocked. Writes are
atomic (temp file + os.replace) for updates.
"""

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from google.protobuf import json_format
from google.protobuf.message import Message

from nvidia_resiliency_ext.shared_utils.job_metadata import job_id_from_env
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2

logger = logging.getLogger(LogConfig.name)


def _cycle_info_filename(job_id: str, attempt_index: int, cycle_number: int) -> str:
    return f"cycle_info.{job_id}.{attempt_index}.{cycle_number}"


def _current_symlink_name(job_id: str) -> str:
    return f"cycle_info.{job_id}.current"


def cycle_log_file(base_log_file: str, cycle_index: int) -> str:
    base_without_ext, ext = os.path.splitext(os.path.abspath(base_log_file))
    return f"{base_without_ext}_cycle{cycle_index}{ext or '.log'}"


# Fields of the launchers' end reports that the cycle info file leaves out: they repeat the
# cycle's own job id, attempt index and cycle number.
_FILE_OMITTED_END_REPORT_FIELDS = ("job_id", "attempt_index", "cycle_number")


def _cycle_info_to_json(msg: nvrx_interface_pb2.CycleInfo) -> str:
    """The cycle info file's JSON of `msg`: every field by its proto name, except the
    internal `timestamp` (for the CycleAnalyzer's checks only). The end report is {} until
    the cycle's end reports are written, and stays {} when none were received. Its
    launchers' end reports leave out the cycle's job id, attempt index and cycle number, and
    carry detected_timestamp in ISO 8601 UTC (millisecond precision); read back with
    _cycle_info_from_json."""
    data = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    data.pop("timestamp", None)
    if not msg.HasField("end_report") or msg.end_report.ByteSize() == 0:
        data["end_report"] = {}
    else:
        entries = data["end_report"].get("launcher_end_reports", [])
        for entry, report in zip(entries, msg.end_report.launcher_end_reports):
            for key in _FILE_OMITTED_END_REPORT_FIELDS:
                entry.pop(key, None)
            entry["detected_timestamp"] = utc_iso_ms(report.detected_timestamp)
    return json.dumps(data, indent=2)


def _cycle_info_from_json(text: str) -> nvrx_interface_pb2.CycleInfo:
    """The CycleInfo of a cycle info file written by _cycle_info_to_json. The omitted fields of
    the launchers' end reports come back unset.

    Raises:
        json_format.ParseError: `text` is not a cycle info.
        ValueError: `text` is not JSON, or a detected_timestamp is not an ISO 8601 time.
    """
    data = json.loads(text)
    for entry in (data.get("end_report") or {}).get("launcher_end_reports", []):
        detected = entry.get("detected_timestamp")
        if isinstance(detected, str) and "T" in detected:
            entry["detected_timestamp"] = utc_iso_to_ms(detected)
    return json_format.ParseDict(data, nvrx_interface_pb2.CycleInfo())


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

    def write_cycle_start(self, info: nvrx_interface_pb2.CycleInfo) -> None:
        """Enqueue writing `info` as a new cycle info file (generation 0) and updating the
        .current symlink. The file is named by the job id, attempt index and cycle number
        of `info`. The writer takes ownership of `info`: it sets its generation on the
        writer thread, so the caller must pass a message it does not use afterwards."""
        if self._shutdown_requested:
            logger.warning("CycleInfoWriter already shutdown, ignoring write_cycle_start")
            return
        self._current_cycle = (info.job_id, info.attempt_index, info.cycle_number)
        self._queue.put(_CycleInfoTask(op=_CycleInfoTask.CREATE, info=info))

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
                    self._do_create(task.kwargs["info"])
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

    def _do_create(self, msg: nvrx_interface_pb2.CycleInfo) -> None:
        """Write `msg` as a new cycle info file, at generation 0."""
        msg.generation = 0
        job_id = msg.job_id
        attempt_index = msg.attempt_index
        cycle_number = msg.cycle_number
        os.makedirs(self._nvrx_dir, exist_ok=True)
        filename = _cycle_info_filename(job_id, attempt_index, cycle_number)
        path = os.path.join(self._nvrx_dir, filename)
        try:
            with open(path, "w") as f:
                f.write(_cycle_info_to_json(msg))
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
                msg = _cycle_info_from_json(f.read())
            for key, value in update_payload.items():
                if not hasattr(msg, key):
                    continue
                if isinstance(value, Message):
                    # Message fields cannot be assigned; replace their contents. Set even
                    # when `value` is empty, so the field is written (e.g. end_report {}).
                    field = getattr(msg, key)
                    field.CopyFrom(value)
                    field.SetInParent()
                else:
                    setattr(msg, key, value)
            msg.generation = msg.generation + 1
            text = _cycle_info_to_json(msg)
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
        except (OSError, ValueError, json_format.ParseError) as e:
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
        end_previous_on_start: bool = True,
    ) -> None:
        if cycle_info_job_id is not None and not cycle_info_job_id.strip():
            raise ValueError("cycle_info_job_id must be non-empty when set")
        self._writer = writer or CycleInfoWriter(cycle_info_dir)
        self._cycle_log_prefix = cycle_log_prefix
        self._cycle_info_job_id = cycle_info_job_id
        self._attempt_index = attempt_index
        self._end_previous_on_start = end_previous_on_start
        self._current_cycle: Optional[_ReportedCycle] = None
        self._started_cycles: set[int] = set()
        self._shutdown = False

    @staticmethod
    def current_cycle_info_path(cycle_info_dir: str, job_id: str) -> str:
        return os.path.join(
            os.path.abspath(cycle_info_dir.rstrip("/")),
            _current_symlink_name(job_id),
        )

    def report_cycle_start(self, info: nvrx_interface_pb2.CycleInfo) -> bool:
        """Write the cycle info file of the cycle `info` starts, as the store host built it.

        A copy of `info` is written, with this reporter's file namespace (job id and
        attempt index) and, when `info` has none, the cycle log file of its prefix. `info`
        itself is not modified. Its cycle_end_time is written as received: empty for a
        start.
        """
        if self._shutdown:
            logger.warning("CycleInfoReporter already shutdown, ignoring cycle start")
            return False
        if info.cycle_number in self._started_cycles:
            return False

        job_id, attempt_index = self._resolve_file_namespace()
        if self._current_cycle is not None and self._end_previous_on_start:
            self.report_cycle_end()

        msg = nvrx_interface_pb2.CycleInfo()
        msg.CopyFrom(info)
        msg.job_id = job_id
        msg.attempt_index = attempt_index
        if not msg.cycle_log_file and self._cycle_log_prefix:
            msg.cycle_log_file = cycle_log_file(self._cycle_log_prefix, info.cycle_number)
        self._writer.write_cycle_start(msg)
        self._current_cycle = _ReportedCycle(job_id, attempt_index, info.cycle_number)
        self._started_cycles.add(info.cycle_number)
        return True

    def report_cycle_end(
        self, cycle_number: Optional[int] = None, cycle_end_time: Optional[str] = None
    ) -> bool:
        """Write the end time of `cycle_number`, by default the open cycle.

        Args:
            cycle_number: The cycle to end; any started cycle, not only the open one.
            cycle_end_time: When it ended, in ISO 8601 UTC (as CycleInfo.cycle_end_time);
                None stamps the time of reporting.
        """
        if self._shutdown:
            return False
        if cycle_number is None:
            if self._current_cycle is None:
                return False
            cycle_number = self._current_cycle.cycle_number
        if cycle_number not in self._started_cycles:
            return False
        if self._current_cycle is not None and self._current_cycle.cycle_number == cycle_number:
            reported = self._current_cycle
            self._current_cycle = None
            job_id, attempt_index = reported.job_id, reported.attempt_index
        else:
            job_id, attempt_index = self._resolve_file_namespace()
        self._writer.update_cycle_end(
            job_id=job_id,
            attempt_index=attempt_index,
            cycle_number=cycle_number,
            cycle_end_time=cycle_end_time if cycle_end_time is not None else utc_iso_now(),
        )
        return True

    def report_cycle_end_report(
        self, cycle_number: int, end_report: nvrx_interface_pb2.JobCycleEndReport
    ) -> bool:
        """Write how `cycle_number` ended for the job, `end_report`, in one update of its file.

        Sets the end report only; cycle_end_time is written from the store host's end
        (report_cycle_end). The caller only enqueues, and must not modify `end_report`
        afterwards: it is written on the writer thread.
        """
        if self._shutdown or cycle_number not in self._started_cycles:
            return False
        job_id, attempt_index = self._resolve_file_namespace()
        if self._current_cycle is not None and self._current_cycle.cycle_number == cycle_number:
            # Ended here, so the writer's shutdown does not stamp it again.
            self._current_cycle = None
        self._writer.update_cycle_later(
            job_id, attempt_index, cycle_number, lambda: {"end_report": end_report}
        )
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


def utc_iso(timestamp: float) -> str:
    """Return `timestamp` (seconds since the epoch) in ISO 8601 UTC, like utc_iso_now()."""
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def utc_iso_ms(timestamp_ms: int) -> str:
    """Return `timestamp_ms` (milliseconds since the epoch) in ISO 8601 UTC, exactly (e.g.
    "2025-02-05T14:30:00.123Z")."""
    moment = datetime.fromtimestamp(0, timezone.utc) + timedelta(milliseconds=timestamp_ms)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_iso_to_ms(value: str) -> int:
    """Return the ISO 8601 UTC time `value` in milliseconds since the epoch; the exact inverse
    of utc_iso_ms.

    Raises:
        ValueError: `value` is not an ISO 8601 time.
    """
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (moment - datetime.fromtimestamp(0, timezone.utc)) // timedelta(milliseconds=1)


def utc_iso_now() -> str:
    """Return current UTC time in ISO 8601 format (e.g. for cycle_start_time / cycle_end_time)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
