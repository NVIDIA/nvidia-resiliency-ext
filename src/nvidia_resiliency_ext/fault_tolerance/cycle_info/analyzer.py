# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CycleAnalyzer: builds the job-wide cycle info from the cycle records.

It is the only writer of the cycle info files. It runs in its own process, started by
nvrx-control next to the gRPC log servers, so its work never competes with the
rendezvous for a GIL.
"""

import contextlib
import logging
import multiprocessing
import os
import signal
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig, setup_logger
from nvidia_resiliency_ext.shared_utils.proto import nvrx_interface_pb2 as pb

from ..cycle_info_writer import CycleInfoReporter
from .aggregator import CycleEndReportAggregator
from .messages import iso_to_seconds
from .transport import CycleInfoReader

logger = logging.getLogger(LogConfig.name)


class CycleAnalyzer:
    """Turns the received cycle records into cycle info, written through a reporter.

    `run` owns all the state on one thread. Receiving and decoding happen in the reader
    and file writes on the reporter's writer thread, so this loop only aggregates.

    The cycle's end time (cycle_end_time) is written when the store host's end arrives,
    before or after its end reports. The end reports of cycle N are written once, at the
    earliest of:
    - every active launcher of N has sent its end report;
    - `next_start_grace` after a later cycle's start arrived. Launchers report before
      rejoining, so by then every launcher that rejoined has reported, but its end report
      can still be in flight on another path than the store host's start;
    - `end_report_quiet_period` after the last end report of N arrived, e.g. for the
      final cycle, which has no next start;
    - the analyzer stopping.
    Launchers that have not reported by then are listed as missing.

    Args:
        reader: Source of the store host's cycle infos and the launchers' end reports.
        cycle_info_reporter: Writes the cycle info files.
        aggregator: Draws the job-wide conclusions of a cycle from its end reports.
        poll_interval: Seconds between reads of the reader.
        next_start_grace: Seconds to wait for a cycle's end reports after a later start.
        end_report_quiet_period: Seconds without a new end report after which a cycle is
            written.
    """

    def __init__(
        self,
        reader: CycleInfoReader,
        cycle_info_reporter: CycleInfoReporter,
        aggregator: Optional[CycleEndReportAggregator] = None,
        poll_interval: float = 0.5,
        next_start_grace: float = 10.0,
        end_report_quiet_period: float = 60.0,
    ):
        self._reader = reader
        self.cycle_info_reporter = cycle_info_reporter
        self._aggregator = aggregator or CycleEndReportAggregator()
        self._poll_interval = poll_interval
        self._next_start_grace = next_start_grace
        self._end_report_quiet_period = end_report_quiet_period
        # The store host's latest CycleInfo of each cycle, as received: its start, replaced
        # by its update once the cycle ended (cycle_end_time set). With the end reports
        # received so far, by cycle; both dropped once the cycle's end reports are written
        # (update_cycle_end_reports).
        self._cycle_infos: Dict[int, pb.CycleInfo] = {}
        self._end_reports: Dict[int, List[pb.NodeCycleEndReport]] = defaultdict(list)
        # Launchers that reported, by cycle, to detect duplicates in O(1). TODO: keyed by
        # node name, as end reports do not tell launchers of one host apart.
        self._reported: Dict[int, Set[str]] = defaultdict(set)
        # Latest cycle opened at the reporter, from the store host's CycleInfo.
        self._started_cycle: Optional[int] = None
        # Cycle the launchers are currently sending end reports for: the latest cycle any
        # end report was received for.
        self._end_report_cycle: Optional[int] = None
        # When to write a cycle whose next start arrived, and when its last end report did.
        self._write_deadlines: Dict[int, float] = {}
        self._last_end_report_at: Dict[int, float] = {}
        # Cycles whose end reports were written; their late end reports are ignored.
        self._written: Set[int] = set()
        # The store host's start of each written cycle whose end has not arrived yet, so
        # that end is still written when it arrives (report_cycle_info_update). Dropped then.
        self._awaiting_end: Dict[int, pb.CycleInfo] = {}

    def run(self, stop_event) -> None:
        """Process the received records until `stop_event` is set, then finish."""
        while not stop_event.is_set():
            self._process_received_safely()
            stop_event.wait(self._poll_interval)
        # Records that arrived while stopping, then every cycle still open.
        self._process_received_safely()
        try:
            self.finish()
        except Exception as e:
            logger.warning(f"Failed to write the open cycles: {e}", exc_info=True)

    def process_received(self) -> None:
        """Handle the records received since the previous call, then write the cycles due."""
        now = time.monotonic()
        for info in self._reader.read_cycle_infos():
            if self.report_cycle_start(info):
                self._schedule_earlier_cycles(info.cycle_number, now)
        for info in self._reader.read_cycle_info_updates():
            self.report_cycle_info_update(info)
        updated = set()
        for report in self._reader.read_end_reports():
            if self.add_end_report(report):
                updated.add(report.cycle_number)
                self._last_end_report_at[report.cycle_number] = now
        for cycle in self._due_cycles(updated, now):
            self.update_cycle_end_reports(cycle)

    def finish(self) -> None:
        """Write every cycle still open, e.g. when stopping."""
        for cycle in sorted(self._open_cycles()):
            self.update_cycle_end_reports(cycle)

    def report_cycle_start(self, info: pb.CycleInfo) -> bool:
        """Open the cycle `info.cycle_number` at the reporter, from the store host's CycleInfo,
        passed as received.

        A start is ignored, with a warning, if its cycle was already started (still open,
        or already written) or is older than the latest started cycle: the store host
        starts each cycle once, in order. A start without active nodes is logged but still
        opens the cycle: it is the only start the cycle gets, and the workload reads its
        cycle info.

        Returns:
            True if the cycle was opened.
        """
        cycle = info.cycle_number
        if cycle in self._cycle_infos or cycle in self._written:
            logger.warning("Ignoring a duplicate start of cycle %s", cycle)
            return False
        if self._started_cycle is not None and cycle < self._started_cycle:
            logger.warning(
                "Ignoring the start of cycle %s: older than the latest started cycle %s",
                cycle,
                self._started_cycle,
            )
            return False
        if not info.active_nodes:
            logger.warning("The start of cycle %s has no active nodes", cycle)
        self._cycle_infos[cycle] = info
        started = self.cycle_info_reporter.report_cycle_start(info)
        if started:
            self._started_cycle = cycle
            logger.debug("Opened the cycle info of cycle %s", cycle)
        return started

    def report_cycle_info_update(self, info: pb.CycleInfo) -> bool:
        """End cycle `info.cycle_number` at the reporter, from the store host's updated CycleInfo.

        The end is accepted even when the cycle's end reports were already written: the
        store host's end can arrive after them, e.g. for the final cycle.

        The update is ignored, with a warning, if:
        - its cycle was never started, or already ended;
        - it is not newer than the cycle's start (`timestamp`; both are stamped by the
          store host, so comparable);
        - it has no end time, or one not after the start time.
        Otherwise the end time is written. For an open cycle, the update replaces the
        cycle's CycleInfo, which marks the cycle ended; a written cycle is then forgotten.

        Returns:
            False if the update was ignored.
        """
        cycle = info.cycle_number
        written = cycle in self._written
        prev_info = self._awaiting_end.get(cycle) if written else self._cycle_infos.get(cycle)
        if prev_info is None:
            if written:
                # Its end was already written, or its start never arrived.
                logger.warning(
                    "Ignoring the end of cycle %s: already ended or never started", cycle
                )
            else:
                logger.warning("Ignoring the end of cycle %s: the cycle was never started", cycle)
            return False
        if prev_info.cycle_end_time:
            logger.warning("Ignoring the end of cycle %s: the cycle already ended", cycle)
            return False
        if info.timestamp <= prev_info.timestamp:
            logger.warning(
                "Ignoring the end of cycle %s: its timestamp %s is not after the start's %s",
                cycle,
                info.timestamp,
                prev_info.timestamp,
            )
            return False
        # Both parse: the reader checked them (check_cycle_info).
        ended_at = iso_to_seconds(info.cycle_end_time)
        if ended_at is None or ended_at <= iso_to_seconds(prev_info.cycle_start_time):
            logger.warning(
                "Ignoring the end of cycle %s: end time %r is not after the start time %r",
                cycle,
                info.cycle_end_time,
                prev_info.cycle_start_time,
            )
            return False
        if not self.cycle_info_reporter.report_cycle_end(cycle, info.cycle_end_time):
            logger.warning("The end of cycle %s was not written", cycle)
        if written:
            # Its end reports are written too: nothing more to keep. Not reopened in
            # _cycle_infos, so they are not written again.
            del self._awaiting_end[cycle]
        else:
            self._cycle_infos[cycle] = info
        logger.info("Cycle %s ended at %s", cycle, info.cycle_end_time)
        return True

    def add_end_report(self, report: pb.NodeCycleEndReport) -> bool:
        """Keep `report` with the end reports received for its cycle.

        An end report is ignored, with a warning, if:
        - its cycle is already written (a late report).
        - its cycle was never started: it is older than the latest started cycle and was
          not started itself, and starts arrive in order. A report of a later cycle is
          kept, as its start may still be on its way;
        - its launcher already reported for the cycle (a duplicate); the first report is
          kept. TODO: launchers are told apart by node name only.

        Returns:
            False if the end report was ignored.
        """
        cycle, node = report.cycle_number, report.node_name
        if cycle in self._written:
            logger.warning(
                "Ignoring the end report of cycle %s from %s: the cycle is already written",
                cycle,
                node,
            )
            return False
        if (
            self._started_cycle is not None
            and cycle < self._started_cycle
            and cycle not in self._cycle_infos
        ):
            logger.warning(
                "Ignoring the end report of cycle %s from %s: the cycle was never started",
                cycle,
                node,
            )
            return False
        if node in self._reported[cycle]:
            logger.warning("Ignoring a duplicate end report of cycle %s from %s", cycle, node)
            return False
        self._reported[cycle].add(node)
        self._end_reports[cycle].append(report)
        if self._end_report_cycle is None or cycle > self._end_report_cycle:
            self._end_report_cycle = cycle
            logger.debug("Receiving the end reports of cycle %s", cycle)
        return True

    def create_job_end_report(self, cycle: int) -> Optional[pb.JobCycleEndReport]:
        """Aggregate the end reports received for `cycle` into the job-wide end report; None
        when none were received."""
        return self._aggregator.aggregate(self._end_reports.get(cycle, ()))

    def update_cycle_end_reports(self, cycle: int) -> None:
        """Write the end reports of `cycle` to its cycle info, then forget the cycle."""
        end_report = self.create_job_end_report(cycle)
        reported = self._aggregator.deduplicate(self._end_reports.get(cycle, ()))
        missing = self._aggregator.missing_nodes(self._cycle_infos.get(cycle), reported)
        missing_text = f", missing {', '.join(missing)}" if missing else ""
        if end_report is None:
            # No end report received: nothing to write; the file keeps no end_report.
            logger.info("Cycle %s ended: no launcher reported%s", cycle, missing_text)
        else:
            # One update: the job-wide end report. The cycle's end time comes only from
            # the store host's update (report_cycle_info_update).
            self.cycle_info_reporter.report_cycle_end_report(cycle, end_report)
            logger.info(
                "Cycle %s ended: %s, first detected at %s by %s; %d launchers reported%s",
                cycle,
                pb.EndReason.Name(end_report.end_reason),
                end_report.first_detected_timestamp or "unknown",
                ", ".join(end_report.first_detected_nodes) or "none",
                len(reported),
                missing_text,
            )
        # Forgotten either way: its late end reports are ignored. If the store host's end
        # has not arrived yet, its start is kept so that end is still written.
        info = self._cycle_infos.pop(cycle, None)
        if info is not None and not info.cycle_end_time:
            self._awaiting_end[cycle] = info
        self._end_reports.pop(cycle, None)
        self._reported.pop(cycle, None)
        self._write_deadlines.pop(cycle, None)
        self._last_end_report_at.pop(cycle, None)
        self._written.add(cycle)

    def _open_cycles(self) -> Set[int]:
        """Cycles with a cycle info or end reports received and not written yet."""
        return set(self._cycle_infos) | set(self._end_reports)

    def _schedule_earlier_cycles(self, started_cycle: int, now: float) -> None:
        """A cycle started: the open cycles before it are written after the grace period."""
        deadline = now + self._next_start_grace
        for cycle in self._open_cycles():
            if cycle < started_cycle:
                self._write_deadlines.setdefault(cycle, deadline)

    def _due_cycles(self, updated: Set[int], now: float) -> List[int]:
        """The open cycles to write now, oldest first."""
        due = set()
        # Complete: every active node of the start has reported. Checked only for the
        # cycles that just received end reports.
        for cycle in updated:
            start = self._cycle_infos.get(cycle)
            if start is not None and not self._aggregator.missing_nodes(
                start, self._aggregator.deduplicate(self._end_reports[cycle])
            ):
                due.add(cycle)
        # check older cycles for which deadline has passed
        due.update(cycle for cycle, at in self._write_deadlines.items() if now >= at)
        due.update(
            cycle
            for cycle, at in self._last_end_report_at.items()
            if now - at >= self._end_report_quiet_period
        )
        return sorted(due)

    def _process_received_safely(self) -> None:
        # Best effort: one bad record must not stop the analysis of the next cycles.
        try:
            self.process_received()
        except Exception as e:
            logger.warning(f"Failed to process the received cycle records: {e}", exc_info=True)


class _StopRequest:
    """Set by the parent's event or by a signal.

    The signal handler only sets a flag: setting the event takes a lock that the
    interrupted thread may be holding.
    """

    def __init__(self, parent_event):
        self._parent_event = parent_event
        self.signalled = False

    def is_set(self) -> bool:
        return self.signalled or self._parent_event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._parent_event.wait(timeout) or self.signalled


def _run_cycle_analyzer(
    cycle_info_dir: str,
    cycle_info_job_id: str,
    cycle_log_prefix: Optional[str],
    address: str,
    ready_event,
    stop_event,
) -> None:
    """Entry point of the CycleAnalyzer process; receives the records on `address`."""
    setup_logger(force_reset=True, node_local_tmp_prefix="cycleanalyzer")
    # The parent stops this process through stop_event; a signal sent to the whole process
    # group (e.g. by Slurm) stops it the same way, so the reporter can finish its writes.
    stop_request = _StopRequest(stop_event)

    def _on_signal(signum, frame):
        stop_request.signalled = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)

    reporter = CycleInfoReporter(
        cycle_info_dir,
        cycle_log_prefix=cycle_log_prefix,
        cycle_info_job_id=cycle_info_job_id,
        attempt_index=0,
        # Cycle ends come from the store host's updates and the end reports
        # (update_cycle_end_reports), not from the next cycle's start.
        end_previous_on_start=False,
    )
    # Deferred: grpc is only needed in this process.
    from .grpc_transport import GrpcCycleInfoReader

    socket_path = address[len("unix:") :] if address.startswith("unix:") else None
    reader = None
    try:
        if socket_path is not None and os.path.exists(socket_path):
            os.unlink(socket_path)  # left behind by an earlier run
        reader = GrpcCycleInfoReader(address)
        ready_event.set()
        logger.info("CycleAnalyzer receiving cycle records on %s", address)
        CycleAnalyzer(reader, reporter).run(stop_request)
    finally:
        if reader is not None:
            reader.close()
        reporter.shutdown()
        if socket_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(socket_path)
        logger.info("CycleAnalyzer stopped")


class CycleAnalyzerProcess:
    """Runs a CycleAnalyzer in a spawned process that receives the records on `address`.

    The gRPC log root, on the same node, forwards the records there.

    Args:
        cycle_info_dir: Directory of the cycle info files.
        cycle_info_job_id: Job id in the cycle info file names.
        cycle_log_prefix: Prefix of the per-cycle application logs, if any.
        address: gRPC listen address; by default a Unix domain socket private to this
            process (``unix:<tmpdir>/nvrx_cycle_analyzer_<pid>.sock``).
    """

    START_TIMEOUT = 120.0
    STOP_TIMEOUT = 10.0

    def __init__(
        self,
        cycle_info_dir: str,
        cycle_info_job_id: str,
        cycle_log_prefix: Optional[str] = None,
        address: Optional[str] = None,
    ):
        self.address = address or (
            f"unix:{tempfile.gettempdir()}/nvrx_cycle_analyzer_{os.getpid()}.sock"
        )
        self._kwargs = dict(
            cycle_info_dir=cycle_info_dir,
            cycle_info_job_id=cycle_info_job_id,
            cycle_log_prefix=cycle_log_prefix,
            address=self.address,
        )
        self._ctx = multiprocessing.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._process = None

    def start(self) -> None:
        """Start the process and wait until it receives records on `address`.

        Raises:
            RuntimeError: The process exited or was not ready within START_TIMEOUT.
        """
        ready_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_run_cycle_analyzer,
            kwargs=dict(self._kwargs, ready_event=ready_event, stop_event=self._stop_event),
            name="CycleAnalyzer",
            daemon=True,
        )
        self._process.start()
        deadline = time.monotonic() + self.START_TIMEOUT
        while not ready_event.wait(timeout=0.5):
            if not self._process.is_alive():
                raise RuntimeError(
                    f"CycleAnalyzer exited during startup (exit code {self._process.exitcode})"
                )
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"CycleAnalyzer not ready after {self.START_TIMEOUT}s")
        logger.info("Started CycleAnalyzer (PID=%s) on %s", self._process.pid, self.address)

    def stop(self) -> None:
        """Ask the process to stop, then force it if it does not exit within STOP_TIMEOUT."""
        if self._process is None:
            return
        self._stop_event.set()
        self._process.join(timeout=self.STOP_TIMEOUT)
        if self._process.is_alive():
            logger.warning("CycleAnalyzer did not stop within %ss; killing it", self.STOP_TIMEOUT)
            self._process.kill()
            self._process.join(timeout=2.0)
        self._process = None
