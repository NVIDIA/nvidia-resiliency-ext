# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from unittest.mock import MagicMock, patch

import pytest

from nvidia_resiliency_ext.fault_tolerance.cycle_info import analyzer as analyzer_module
from nvidia_resiliency_ext.fault_tolerance.cycle_info.analyzer import (
    CycleAnalyzer,
    CycleAnalyzerProcess,
    _StopRequest,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.records import (
    CycleEndReason,
    CycleStartInfo,
    LocalCycleOutcome,
    RankEndReason,
    RankOutcome,
)
from nvidia_resiliency_ext.fault_tolerance.cycle_info.transport import CycleInfoReader


class _ListReader(CycleInfoReader):
    def __init__(self):
        self.starts = []
        self.outcomes = []

    def read_cycle_starts(self):
        starts, self.starts = self.starts, []
        return starts

    def read_outcomes(self):
        outcomes, self.outcomes = self.outcomes, []
        return outcomes


def _start(cycle):
    return CycleStartInfo(cycle, 10.0 + cycle, ("node0",), active_ranks=(0,))


def _outcome(cycle, node):
    return LocalCycleOutcome(cycle, node, CycleEndReason.FAILED, 20.0 + cycle, ())


def _analyzer(reader):
    analyzer = CycleAnalyzer(reader, MagicMock(), poll_interval=0.01)
    analyzer.report_cycle_start = MagicMock()
    return analyzer


def test_cycle_starts_are_reported_in_arrival_order():
    reader = _ListReader()
    analyzer = _analyzer(reader)
    reader.starts = [_start(0), _start(1)]

    analyzer.process_received()

    assert [c.args[0].cycle for c in analyzer.report_cycle_start.call_args_list] == [0, 1]


def test_outcomes_are_kept_by_cycle():
    reader = _ListReader()
    analyzer = _analyzer(reader)
    reader.outcomes = [_outcome(0, "n0"), _outcome(1, "n0")]
    analyzer.process_received()
    reader.outcomes = [_outcome(0, "n1")]
    analyzer.process_received()

    assert analyzer._outcomes == {
        0: [_outcome(0, "n0"), _outcome(0, "n1")],
        1: [_outcome(1, "n0")],
    }


def test_run_processes_records_until_stopped_then_writes_the_open_cycles():
    reader = _ListReader()
    analyzer = _analyzer(reader)
    stop_event = threading.Event()

    def stop_after_first_records(info):
        # Records that arrive while stopping are still processed.
        reader.outcomes.append(_outcome(0, "late"))
        stop_event.set()

    analyzer.report_cycle_start.side_effect = stop_after_first_records
    reader.starts = [_start(0)]
    thread = threading.Thread(target=analyzer.run, args=(stop_event,))
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    summary = analyzer.cycle_info_reporter.report_cycle_summary.call_args.args[0]
    assert (summary.cycle, summary.outcomes) == (0, (_outcome(0, "late"),))


def test_a_failing_record_is_logged_and_the_loop_continues():
    reader = MagicMock()
    reader.read_cycle_starts.side_effect = [RuntimeError("boom"), [], []]
    reader.read_outcomes.return_value = []
    analyzer = CycleAnalyzer(reader, MagicMock(), poll_interval=0.01)
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, False, True]
    stop_event.wait.return_value = False

    with patch.object(analyzer_module, "logger") as logger:
        analyzer.run(stop_event)

    logger.warning.assert_called_once()
    assert reader.read_cycle_starts.call_count == 3  # two loop passes and the final one


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


class TestUpdateCycleEndOutcomes:
    def _analyzer_with_cycle(self, started_cycle=3):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(_ListReader(), reporter)
        analyzer.report_cycle_start(_start(3))
        analyzer.add_outcome(_outcome(3, "node0"))
        if started_cycle != 3:
            analyzer.report_cycle_start(_start(started_cycle))
        return analyzer, reporter

    def test_writes_the_end_and_the_outcomes_then_forgets_the_cycle(self):
        analyzer, reporter = self._analyzer_with_cycle()

        analyzer.update_cycle_end_outcomes(3)

        summary = reporter.report_cycle_summary.call_args.args[0]
        assert (summary.cycle, summary.outcomes) == (3, (_outcome(3, "node0"),))
        reporter.report_cycle_end.assert_not_called()  # the summary carries the end time
        assert 3 not in analyzer._outcomes and 3 not in analyzer._starts

    def test_writes_the_cycle_even_after_a_later_cycle_started(self):
        analyzer, reporter = self._analyzer_with_cycle(started_cycle=4)

        analyzer.update_cycle_end_outcomes(3)

        assert reporter.report_cycle_summary.call_args.args[0].cycle == 3
        reporter.report_cycle_end.assert_not_called()

    def test_without_outcomes_the_end_is_stamped_when_written(self):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(_ListReader(), reporter)
        analyzer.report_cycle_start(_start(3))

        analyzer.update_cycle_end_outcomes(3)

        reporter.report_cycle_end.assert_called_once_with(3)
        reporter.report_cycle_summary.assert_called_once()

    def test_the_start_time_of_the_store_host_is_reported(self):
        analyzer, reporter = self._analyzer_with_cycle()
        assert reporter.report_cycle_start.call_args.args[0].started_at == _start(3).started_at

    def test_summary_uses_the_cycle_start(self):
        analyzer, _ = self._analyzer_with_cycle()
        summary = analyzer.create_job_cycle_outcomes(3)
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
            assert analyzer.report_cycle_start(CycleStartInfo(3, 1.0, ()))
        reporter.report_cycle_start.assert_called_once()
        logger.warning.assert_called_once()

    @pytest.mark.parametrize("written", [False, True], ids=["open", "already_written"])
    def test_a_duplicate_start_is_ignored(self, written):
        analyzer, reporter = self._analyzer()
        analyzer.report_cycle_start(_start(3))
        if written:
            analyzer.update_cycle_end_outcomes(3)

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.report_cycle_start(CycleStartInfo(3, 99.0, ("other",)))

        reporter.report_cycle_start.assert_called_once()
        logger.warning.assert_called_once()
        if not written:
            assert analyzer._starts[3] == _start(3)  # the first start is kept

    def test_a_start_older_than_the_latest_is_ignored(self):
        analyzer, reporter = self._analyzer()
        analyzer.report_cycle_start(_start(4))

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.report_cycle_start(_start(3))

        reporter.report_cycle_start.assert_called_once()
        logger.warning.assert_called_once()
        assert 3 not in analyzer._starts

    def test_an_ignored_start_does_not_schedule_earlier_cycles(self):
        reader = _ListReader()
        analyzer = CycleAnalyzer(reader, MagicMock())
        reader.starts = [_start(4)]
        analyzer.process_received()
        reader.outcomes = [_outcome(2, "node0")]  # an open earlier cycle
        analyzer.process_received()

        reader.starts = [_start(3)]  # older than 4: ignored
        analyzer.process_received()

        assert 2 not in analyzer._write_deadlines


class TestAddOutcome:
    """An outcome is kept once per launcher, for a cycle that can still be written."""

    def _analyzer(self, started=()):
        reporter = MagicMock()
        reporter.report_cycle_start.return_value = True
        analyzer = CycleAnalyzer(_ListReader(), reporter)
        for cycle in started:
            analyzer.report_cycle_start(_start(cycle))
        return analyzer

    @staticmethod
    def _ranked(cycle, node, global_rank):
        rank = RankOutcome(global_rank, 0, 100 + global_rank, RankEndReason.FAILED, 1)
        return LocalCycleOutcome(cycle, node, CycleEndReason.FAILED, 20.0, (rank,))

    def test_a_duplicate_is_ignored_and_the_first_kept(self):
        analyzer = self._analyzer(started=[3])
        first = _outcome(3, "node0")
        assert analyzer.add_outcome(first)

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.add_outcome(_outcome(3, "node0"))

        assert analyzer._outcomes[3] == [first]
        logger.warning.assert_called_once()

    def test_launchers_sharing_a_host_are_not_duplicates(self):
        analyzer = self._analyzer(started=[3])
        assert analyzer.add_outcome(self._ranked(3, "host", 0))
        assert analyzer.add_outcome(self._ranked(3, "host", 1))
        assert len(analyzer._outcomes[3]) == 2

    def test_an_outcome_of_a_cycle_never_started_is_ignored(self):
        analyzer = self._analyzer(started=[4])  # cycle 3 was skipped

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.add_outcome(_outcome(3, "node0"))

        assert 3 not in analyzer._outcomes
        logger.warning.assert_called_once()

    def test_an_outcome_before_its_cycle_start_is_kept(self):
        """Starts and outcomes travel on different paths: the start may come later."""
        analyzer = self._analyzer(started=[3])
        assert analyzer.add_outcome(_outcome(4, "node0"))
        assert analyzer._outcomes[4] == [_outcome(4, "node0")]

    def test_a_late_outcome_is_ignored(self):
        analyzer = self._analyzer(started=[3])
        analyzer.update_cycle_end_outcomes(3)

        with patch.object(analyzer_module, "logger") as logger:
            assert not analyzer.add_outcome(_outcome(3, "node0"))

        assert 3 not in analyzer._outcomes
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
        analyzer = CycleAnalyzer(reader, reporter, next_start_grace=10.0, outcome_quiet_period=60.0)
        analyzer.update_cycle_end_outcomes = MagicMock(
            side_effect=analyzer.update_cycle_end_outcomes
        )
        return analyzer, reader, self._clock

    @staticmethod
    def _written(analyzer):
        return [c.args[0] for c in analyzer.update_cycle_end_outcomes.call_args_list]

    def test_when_every_active_launcher_reported(self):
        analyzer, reader, _ = self._analyzer()
        reader.starts = [CycleStartInfo(3, 1.0, ("n0", "n1"))]
        reader.outcomes = [_outcome(3, "n0")]
        analyzer.process_received()
        assert self._written(analyzer) == []

        reader.outcomes = [_outcome(3, "n1")]
        analyzer.process_received()
        assert self._written(analyzer) == [3]

    def test_after_the_grace_period_of_the_next_start(self):
        analyzer, reader, clock = self._analyzer()
        reader.starts = [CycleStartInfo(3, 1.0, ("n0", "n1"))]
        reader.outcomes = [_outcome(3, "n0")]  # n1 never reports
        analyzer.process_received()
        reader.starts = [CycleStartInfo(4, 2.0, ("n0", "n1"))]
        analyzer.process_received()

        clock.now += 9.9
        analyzer.process_received()
        assert self._written(analyzer) == []
        clock.now += 0.1
        analyzer.process_received()
        assert self._written(analyzer) == [3]  # not the running cycle 4

    def test_after_a_quiet_period_without_outcomes(self):
        analyzer, reader, clock = self._analyzer()
        reader.starts = [CycleStartInfo(3, 1.0, ("n0", "n1"))]
        reader.outcomes = [_outcome(3, "n0")]
        analyzer.process_received()

        clock.now += 59.9
        analyzer.process_received()
        assert self._written(analyzer) == []
        clock.now += 0.1
        analyzer.process_received()
        assert self._written(analyzer) == [3]

    def test_every_open_cycle_when_finishing(self):
        analyzer, reader, _ = self._analyzer()
        reader.starts = [CycleStartInfo(3, 1.0, ("n0",)), CycleStartInfo(4, 2.0, ("n0",))]
        analyzer.process_received()

        analyzer.finish()

        assert self._written(analyzer) == [3, 4]

    def test_a_cycle_is_written_once_and_late_outcomes_are_ignored(self):
        analyzer, reader, clock = self._analyzer()
        reader.starts = [CycleStartInfo(3, 1.0, ("n0",))]
        reader.outcomes = [_outcome(3, "n0")]
        analyzer.process_received()
        assert self._written(analyzer) == [3]

        reader.outcomes = [_outcome(3, "n0")]
        with patch.object(analyzer_module, "logger") as logger:
            analyzer.process_received()
        clock.now += 100
        analyzer.process_received()
        analyzer.finish()

        assert self._written(analyzer) == [3]
        logger.warning.assert_called_once()
