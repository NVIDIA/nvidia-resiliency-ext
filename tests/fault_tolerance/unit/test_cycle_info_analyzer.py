# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info import analyzer as analyzer_module
from nvidia_resiliency_ext.fault_tolerance.cycle_info.analyzer import (
    CycleAnalyzer,
    CycleAnalyzerProcess,
    _StopRequest,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    CycleInfo,
    EndReason,
    NodeCycleEndReport,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.transport import CycleInfoReader


class _ListReader(CycleInfoReader):
    def __init__(self):
        self.starts = []
        self.updates = []
        self.end_reports = []

    def read_cycle_infos(self):
        starts, self.starts = self.starts, []
        return starts

    def read_cycle_info_updates(self):
        updates, self.updates = self.updates, []
        return updates

    def read_end_reports(self):
        end_reports, self.end_reports = self.end_reports, []
        return end_reports


def _start(cycle, nodes=("node0",)):
    return CycleInfo(
        job_id="job1",
        attempt_index=0,
        cycle_number=cycle,
        started_at=10.0 + cycle,
        active_nodes=tuple(nodes),
        active_ranks=tuple(range(len(nodes))),
        timestamp=10_000 + cycle * 1000,
    )


def _end(cycle, nodes=("node0",)):
    """The store host's update of `_start(cycle)`: stamped later, with its end time."""
    return replace(_start(cycle, nodes), ended_at=30.0 + cycle, timestamp=30_000 + cycle * 1000)


def _end_report(cycle, node):
    return NodeCycleEndReport("job1", 0, cycle, node, EndReason.FAILED, 20_000 + cycle)


def _analyzer(reader):
    analyzer = CycleAnalyzer(reader, MagicMock(), poll_interval=0.01)
    analyzer.report_cycle_start = MagicMock()
    return analyzer


def test_cycle_starts_are_reported_in_arrival_order():
    reader = _ListReader()
    analyzer = _analyzer(reader)
    reader.starts = [_start(0), _start(1)]

    analyzer.process_received()

    assert [c.args[0].cycle_number for c in analyzer.report_cycle_start.call_args_list] == [0, 1]


def test_end_reports_are_kept_by_cycle():
    reader = _ListReader()
    analyzer = _analyzer(reader)
    reader.end_reports = [_end_report(0, "n0"), _end_report(1, "n0")]
    analyzer.process_received()
    reader.end_reports = [_end_report(0, "n1")]
    analyzer.process_received()

    assert analyzer._end_reports == {
        0: [_end_report(0, "n0"), _end_report(0, "n1")],
        1: [_end_report(1, "n0")],
    }


def test_cycle_info_updates_are_processed():
    reader = _ListReader()
    analyzer = CycleAnalyzer(reader, MagicMock())
    reader.starts = [_start(3, ("n0", "n1"))]
    reader.updates = [_end(3, ("n0", "n1"))]

    analyzer.process_received()

    assert analyzer._cycle_infos == {3: _end(3, ("n0", "n1"))}


def test_run_processes_records_until_stopped_then_writes_the_open_cycles():
    reader = _ListReader()
    analyzer = _analyzer(reader)
    stop_event = threading.Event()

    def stop_after_first_records(info):
        # Records that arrive while stopping are still processed.
        reader.end_reports.append(_end_report(0, "late"))
        stop_event.set()

    analyzer.report_cycle_start.side_effect = stop_after_first_records
    reader.starts = [_start(0)]
    thread = threading.Thread(target=analyzer.run, args=(stop_event,))
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    summary = analyzer.cycle_info_reporter.report_cycle_summary.call_args.args[0]
    assert (summary.cycle, summary.end_reports) == (0, (_end_report(0, "late"),))


def test_a_failing_record_is_logged_and_the_loop_continues():
    reader = MagicMock()
    reader.read_cycle_infos.side_effect = [RuntimeError("boom"), [], []]
    reader.read_cycle_info_updates.return_value = []
    reader.read_end_reports.return_value = []
    analyzer = CycleAnalyzer(reader, MagicMock(), poll_interval=0.01)
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, False, True]
    stop_event.wait.return_value = False

    with patch.object(analyzer_module, "logger") as logger:
        analyzer.run(stop_event)

    logger.warning.assert_called_once()
    assert reader.read_cycle_infos.call_count == 3  # two loop passes and the final one


def test_stop_request_is_set_by_the_parent_or_a_signal():
    parent = threading.Event()
    stop = _StopRequest(parent)
    assert not stop.is_set()
    stop.signalled = True
    assert stop.is_set() and stop.wait(0)

    stop = _StopRequest(parent)
    parent.set()
    assert stop.is_set() and stop.wait(0)


class TestCycleAnalyzerProcess:
    def _process(self, ready, alive=True):
        ctx = MagicMock()
        ctx.Event.return_value.wait.return_value = ready
        ctx.Process.return_value.is_alive.return_value = alive
        with patch.object(analyzer_module.multiprocessing, "get_context", return_value=ctx):
            process = CycleAnalyzerProcess(
                "/nvrx", "job-a", "/logs/train.log", address="unix:/tmp/a.sock"
            )
        return process, ctx

    def test_start_spawns_the_analyzer_and_waits_until_ready(self):
        process, ctx = self._process(ready=True)

        process.start()

        kwargs = ctx.Process.call_args.kwargs["kwargs"]
        assert ctx.Process.call_args.kwargs["target"] is analyzer_module._run_cycle_analyzer
        assert (
            kwargs["cycle_info_dir"],
            kwargs["cycle_info_job_id"],
            kwargs["cycle_log_prefix"],
            kwargs["address"],
        ) == ("/nvrx", "job-a", "/logs/train.log", "unix:/tmp/a.sock")
        ctx.Process.return_value.start.assert_called_once_with()

    def test_default_address_is_a_unix_socket(self):
        address = CycleAnalyzerProcess("/nvrx", "job-a").address
        assert address.startswith("unix:") and address.endswith(".sock")

    def test_start_fails_when_the_process_exits(self):
        process, ctx = self._process(ready=False, alive=False)
        with pytest.raises(RuntimeError, match="exited during startup"):
            process.start()

    def test_stop_asks_first_then_kills(self):
        process, ctx = self._process(ready=True)
        process.start()
        child = ctx.Process.return_value

        process.stop()

        ctx.Event.return_value.set.assert_called()  # the stop event
        child.join.assert_called()
        child.kill.assert_called_once_with()  # still alive after the join, in this mock


class TestUpdateCycleEndReports:
    def _analyzer_with_cycle(self, started_cycle=3):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(_ListReader(), reporter)
        analyzer.report_cycle_start(_start(3))
        analyzer.add_end_report(_end_report(3, "node0"))
        if started_cycle != 3:
            analyzer.report_cycle_start(_start(started_cycle))
        return analyzer, reporter

    def test_writes_the_end_and_the_end_reports_then_forgets_the_cycle(self):
        analyzer, reporter = self._analyzer_with_cycle()
        analyzer.report_cycle_info_update(_end(3))

        analyzer.update_cycle_end_reports(3)

        summary = reporter.report_cycle_summary.call_args.args[0]
        assert (summary.cycle, summary.end_reports) == (3, (_end_report(3, "node0"),))
        assert summary.ended_at == _end_report(3, "node0").detected_timestamp
        # Only the store host's end, when the update arrived: not again when written.
        reporter.report_cycle_end.assert_called_once_with(3, ended_at=_end(3).ended_at)
        assert 3 not in analyzer._end_reports and 3 not in analyzer._cycle_infos

    def test_writes_the_cycle_even_after_a_later_cycle_started(self):
        analyzer, reporter = self._analyzer_with_cycle(started_cycle=4)

        analyzer.update_cycle_end_reports(3)

        assert reporter.report_cycle_summary.call_args.args[0].cycle == 3
        reporter.report_cycle_end.assert_not_called()

    def test_without_end_reports_the_end_is_stamped_when_written(self):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(_ListReader(), reporter)
        analyzer.report_cycle_start(_start(3))

        analyzer.update_cycle_end_reports(3)

        reporter.report_cycle_end.assert_called_once_with(3)
        reporter.report_cycle_summary.assert_called_once()

    def test_the_start_of_the_store_host_is_reported(self):
        _, reporter = self._analyzer_with_cycle()
        snapshot = reporter.report_cycle_start.call_args.args[0]
        assert (snapshot.cycle_number, snapshot.started_at) == (3, _start(3).started_at)
        assert (snapshot.active_node_addrs, snapshot.active_ranks) == (["node0"], [0])

    def test_summary_uses_the_cycle_start(self):
        analyzer, _ = self._analyzer_with_cycle()
        summary = analyzer.create_cycle_summary(3)
        assert summary.launchers_expected == len(_start(3).active_nodes)
        assert summary.missing == ()  # its only active node reported


class TestReportCycleStart:
    """A start is accepted once per cycle, in cycle order."""

    def _analyzer(self):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        return CycleAnalyzer(_ListReader(), reporter), reporter

    def test_a_start_opens_its_cycle(self):
        analyzer, reporter = self._analyzer()
        with patch.object(analyzer_module, "logger") as logger:
            assert analyzer.report_cycle_start(_start(3))
        reporter.report_cycle_start.assert_called_once()
        logger.warning.assert_not_called()

    def test_a_start_without_active_nodes_is_logged_and_still_opens_its_cycle(self):
        analyzer, reporter = self._analyzer()
        with patch.object(analyzer_module, "logger") as logger:
            assert analyzer.report_cycle_start(_start(3, nodes=()))
        reporter.report_cycle_start.assert_called_once()
        logger.warning.assert_called_once()

    @pytest.mark.parametrize("written", [False, True], ids=["open", "already_written"])
    def test_a_duplicate_start_is_ignored(self, written):
        analyzer, reporter = self._analyzer()
        analyzer.report_cycle_start(_start(3))
        if written:
            analyzer.update_cycle_end_reports(3)

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.report_cycle_start(_start(3, nodes=("other",)))

        reporter.report_cycle_start.assert_called_once()
        logger.warning.assert_called_once()
        if not written:
            assert analyzer._cycle_infos[3] == _start(3)  # the first start is kept

    def test_a_start_older_than_the_latest_is_ignored(self):
        analyzer, reporter = self._analyzer()
        analyzer.report_cycle_start(_start(4))

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.report_cycle_start(_start(3))

        reporter.report_cycle_start.assert_called_once()
        logger.warning.assert_called_once()
        assert 3 not in analyzer._cycle_infos

    def test_an_ignored_start_does_not_schedule_earlier_cycles(self):
        reader = _ListReader()
        analyzer = CycleAnalyzer(reader, MagicMock())
        reader.starts = [_start(4)]
        analyzer.process_received()
        reader.end_reports = [_end_report(2, "node0")]  # an open earlier cycle
        analyzer.process_received()

        reader.starts = [_start(3)]  # older than 4: ignored
        analyzer.process_received()

        assert 2 not in analyzer._write_deadlines


class TestReportCycleInfoUpdate:
    """The store host's update ends its cycle at the reporter and marks the cycle ended."""

    def _analyzer(self):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(_ListReader(), reporter)
        analyzer.report_cycle_start(_start(3))
        return analyzer, reporter

    def _assert_ignored(self, analyzer, reporter, update):
        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.report_cycle_info_update(update)
        logger.warning.assert_called_once()
        reporter.report_cycle_end.assert_not_called()

    def test_the_end_is_written_and_the_cycle_marked_ended(self):
        analyzer, reporter = self._analyzer()

        assert analyzer.report_cycle_info_update(_end(3))

        reporter.report_cycle_end.assert_called_once_with(3, ended_at=_end(3).ended_at)
        assert analyzer._cycle_infos[3] == _end(3)

    def test_the_cycle_is_kept_for_its_end_reports(self):
        analyzer, _ = self._analyzer()
        analyzer.report_cycle_info_update(_end(3))
        assert analyzer.add_end_report(_end_report(3, "node0"))

    def test_a_reporter_failure_is_logged_and_the_cycle_still_marked_ended(self):
        analyzer, reporter = self._analyzer()
        reporter.report_cycle_end.return_value = False
        with patch.object(analyzer_module, "logger") as logger:
            assert analyzer.report_cycle_info_update(_end(3))
        logger.warning.assert_called_once()
        assert analyzer._cycle_infos[3] == _end(3)

    def test_an_update_of_a_cycle_never_started_is_ignored(self):
        analyzer, reporter = self._analyzer()
        self._assert_ignored(analyzer, reporter, _end(4))
        assert 4 not in analyzer._cycle_infos

    def test_an_update_of_a_written_cycle_is_ignored(self):
        analyzer, reporter = self._analyzer()
        analyzer.update_cycle_end_reports(3)
        reporter.report_cycle_end.reset_mock()  # stamped when written, without end reports
        self._assert_ignored(analyzer, reporter, _end(3))

    def test_a_second_update_is_ignored(self):
        analyzer, reporter = self._analyzer()
        analyzer.report_cycle_info_update(_end(3))
        later = replace(_end(3), ended_at=40.0, timestamp=40_000)
        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.report_cycle_info_update(later)
        logger.warning.assert_called_once()
        reporter.report_cycle_end.assert_called_once()  # the first update only
        assert analyzer._cycle_infos[3] == _end(3)

    @pytest.mark.parametrize("timestamp", [5_000, 13_000], ids=["older", "same"])
    def test_an_update_not_newer_than_the_start_is_ignored(self, timestamp):
        analyzer, reporter = self._analyzer()
        self._assert_ignored(analyzer, reporter, replace(_end(3), timestamp=timestamp))
        assert analyzer._cycle_infos[3] == _start(3)

    @pytest.mark.parametrize("ended_at", [None, 5.0, 13.0], ids=["none", "before", "at_start"])
    def test_an_update_without_a_valid_end_time_is_ignored(self, ended_at):
        analyzer, reporter = self._analyzer()
        self._assert_ignored(analyzer, reporter, replace(_end(3), ended_at=ended_at))
        assert analyzer._cycle_infos[3] == _start(3)


class TestAddEndReport:
    """An end report is kept once per launcher, for a cycle that can still be written."""

    def _analyzer(self, started=()):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(_ListReader(), reporter)
        for cycle in started:
            analyzer.report_cycle_start(_start(cycle))
        return analyzer

    def test_a_duplicate_is_ignored_and_the_first_kept(self):
        analyzer = self._analyzer(started=[3])
        first = _end_report(3, "node0")
        assert analyzer.add_end_report(first)

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.add_end_report(replace(first, detected_timestamp=1))

        assert analyzer._end_reports[3] == [first]
        logger.warning.assert_called_once()

    def test_launchers_sharing_a_host_are_duplicates(self):
        """Known limitation: end reports are told apart by node name only."""
        analyzer = self._analyzer(started=[3])
        assert analyzer.add_end_report(_end_report(3, "host"))
        assert not analyzer.add_end_report(_end_report(3, "host"))
        assert len(analyzer._end_reports[3]) == 1

    def test_an_end_report_of_a_cycle_never_started_is_ignored(self):
        analyzer = self._analyzer(started=[4])  # cycle 3 was skipped

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.add_end_report(_end_report(3, "node0"))

        assert 3 not in analyzer._end_reports
        logger.warning.assert_called_once()

    def test_an_end_report_before_its_cycle_start_is_kept(self):
        """Starts and end reports travel on different paths: the start may come later."""
        analyzer = self._analyzer(started=[3])
        assert analyzer.add_end_report(_end_report(4, "node0"))
        assert analyzer._end_reports[4] == [_end_report(4, "node0")]

    def test_a_late_end_report_is_ignored(self):
        analyzer = self._analyzer(started=[3])
        analyzer.update_cycle_end_reports(3)

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.add_end_report(_end_report(3, "node0"))

        assert 3 not in analyzer._end_reports
        logger.warning.assert_called_once()


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class TestWhenCyclesAreWritten:
    """The end of a cycle is written once, at the earliest of its triggers."""

    @pytest.fixture(autouse=True)
    def _fake_clock(self):
        # The analyzer times its triggers with time.monotonic(); these tests run synchronously.
        self._clock = _Clock()
        with patch.object(analyzer_module.time, "monotonic", self._clock):
            yield

    def _analyzer(self):
        reader, reporter = _ListReader(), MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(
            reader, reporter, next_start_grace=10.0, end_report_quiet_period=60.0
        )
        analyzer.update_cycle_end_reports = MagicMock(side_effect=analyzer.update_cycle_end_reports)
        return analyzer, reader, self._clock

    @staticmethod
    def _written(analyzer):
        return [c.args[0] for c in analyzer.update_cycle_end_reports.call_args_list]

    def test_when_every_active_launcher_reported(self):
        analyzer, reader, _ = self._analyzer()
        reader.starts = [_start(3, ("n0", "n1"))]
        reader.end_reports = [_end_report(3, "n0")]
        analyzer.process_received()
        assert self._written(analyzer) == []

        reader.end_reports = [_end_report(3, "n1")]
        analyzer.process_received()
        assert self._written(analyzer) == [3]

    def test_launchers_reporting_their_fqdn_complete_the_cycle(self):
        """The start's node lists carry short host names; end reports may carry the FQDN."""
        analyzer, reader, _ = self._analyzer()
        reader.starts = [_start(3, ("n0", "n1"))]
        reader.end_reports = [_end_report(3, "n0.cluster"), _end_report(3, "n1.cluster")]
        analyzer.process_received()
        assert self._written(analyzer) == [3]

    def test_after_the_grace_period_of_the_next_start(self):
        analyzer, reader, clock = self._analyzer()
        reader.starts = [_start(3, ("n0", "n1"))]
        reader.end_reports = [_end_report(3, "n0")]  # n1 never reports
        analyzer.process_received()
        reader.starts = [_start(4, ("n0", "n1"))]
        analyzer.process_received()

        clock.now += 9.9
        analyzer.process_received()
        assert self._written(analyzer) == []
        clock.now += 0.1
        analyzer.process_received()
        assert self._written(analyzer) == [3]  # not the running cycle 4

    def test_after_a_quiet_period_without_end_reports(self):
        analyzer, reader, clock = self._analyzer()
        reader.starts = [_start(3, ("n0", "n1"))]
        reader.end_reports = [_end_report(3, "n0")]
        analyzer.process_received()

        clock.now += 59.9
        analyzer.process_received()
        assert self._written(analyzer) == []
        clock.now += 0.1
        analyzer.process_received()
        assert self._written(analyzer) == [3]

    def test_the_end_update_alone_does_not_write_the_cycle(self):
        analyzer, reader, _ = self._analyzer()
        reader.starts = [_start(3, ("n0", "n1"))]
        reader.updates = [_end(3, ("n0", "n1"))]
        analyzer.process_received()
        assert self._written(analyzer) == []

    def test_every_open_cycle_when_finishing(self):
        analyzer, reader, _ = self._analyzer()
        reader.starts = [_start(3), _start(4)]
        analyzer.process_received()

        analyzer.finish()

        assert self._written(analyzer) == [3, 4]

    def test_a_cycle_is_written_once_and_late_end_reports_are_ignored(self):
        analyzer, reader, clock = self._analyzer()
        reader.starts = [_start(3, ("n0",))]
        reader.end_reports = [_end_report(3, "n0")]
        analyzer.process_received()
        assert self._written(analyzer) == [3]

        reader.end_reports = [_end_report(3, "n0")]
        with patch.object(analyzer_module, "logger") as logger:
            analyzer.process_received()
        clock.now += 100
        analyzer.process_received()
        analyzer.finish()

        assert self._written(analyzer) == [3]
        logger.warning.assert_called_once()
