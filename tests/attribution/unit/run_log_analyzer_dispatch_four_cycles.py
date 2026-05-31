"""Script 2/2: drive ``NVRxLogAnalyzer`` start + end phases over 5 cycles.

Pairs with ``run_log_writer_dispatch_four_cycles.py``. A single analyzer
instance is reused so ``attribution_dict`` / ``cycle_counter_dict``
persist across cycles. For each cycle ``i`` in ``[0..4]`` the script:

  1. Calls ``analyze_logs_rt_start`` directly. Polls the cycle's file
     every ``--poll-interval`` seconds for ``--cycle-duration`` seconds
     total (default 60s / 300s = poll 1×/min for 5 min), accumulating
     per-poll entries on ``job_inline_data_dict[path]``, then returns
     ``None``.
  2. Calls ``analyze_logs`` (history-aware branch — rebuilds the LLM
     input as ``tail + last --end-window-minutes of history``, default
     2 min, and runs extraction) → ``list[ApplicationData]``. Feeds the
     single result into ``_streaming_attribution`` to produce an
     ``ErrorAttribution``.

With the writer producing identical errors across cycles and default
``repeated_amount=3``, the third identical attribution should override
``auto_resume`` to ``STOP - DONT RESTART IMMEDIATE``.

Workarounds applied (each is a current source quirk in ``nvrx_logsage.py``):

  - ``__init__`` assigns ``self.stop_accumulating_count``,
    ``self.chunks_per_time``, ``self.logs_minutes_before_job_end``, and
    ``self.repeated_amount`` with trailing commas → 1-tuples. We
    rewrite them to scalar ints after construction.
  - ``analyze_logs_rt_start`` appends to ``self.job_inline_data_dict[path]``
    without initializing the list — we pre-seed it.
  - ``_streaming_attribution`` reads ``cycle_counter_dict[ck]`` without
    a ``setdefault`` — we pre-seed the shared key to 0.
  - The start phase's ``while True`` polling loop is bounded by
    patching ``time.sleep`` with a deadline. When the deadline elapses
    the sleep raises ``_Phase1Deadline``, ``analyze_logs_rt_start``
    exits, and the script moves on to the end phase.

Usage:
    # Real timing (60s polls, 300s cycles = 20 min total):
    python tests/attribution/unit/run_log_analyzer_dispatch_four_cycles.py
    # Fast local iteration (5s polls, 25s cycles):
    python tests/attribution/unit/run_log_analyzer_dispatch_four_cycles.py \\
        --poll-interval 5 --cycle-duration 25
"""

import argparse
import asyncio
import importlib.util
import os
import sys
import time
from unittest.mock import patch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = _HERE
NUM_CYCLES = 5


def cycle_log_path(log_dir: str, cycle: int) -> str:
    return os.path.join(log_dir, f"nvrx_{cycle}.log")


def _logsage_available() -> bool:
    return importlib.util.find_spec("logsage") is not None


def _api_key_available() -> bool:
    try:
        from nvidia_resiliency_ext.attribution.api_keys import load_llm_api_key
    except ImportError:
        return False
    return bool(load_llm_api_key())


class _Phase1Deadline(Exception):
    """Raised inside the patched sleep when the per-cycle budget expires."""


def _make_deadline_sleep(start_duration_sec: float, poll_interval_sec: float):
    # ``analyze_logs_rt_start`` awaits ``asyncio.sleep`` between polls, so this
    # replacement must be a coroutine. We patch ``asyncio.sleep`` (not
    # ``time.sleep``) at the call site below.
    real_sleep = asyncio.sleep
    deadline = time.monotonic() + start_duration_sec

    async def deadline_sleep(*_args, **_kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Phase1Deadline()
        await real_sleep(min(poll_interval_sec, remaining))

    return deadline_sleep


def _override_scalar_config(
    analyzer,
    poll_interval_sec: float,
    end_window_minutes: float,
) -> None:
    """Undo the 1-tuple bug in ``NVRxLogAnalyzer.__init__`` and pin the
    polling cadence + end-phase history window.

    ``end_window_minutes`` is how many minutes of history the end phase
    glues to the tail before re-running extraction. With the writer's
    1-chunk-per-minute pacing and the OOM in the last chunk, a value of
    2 means the end phase always sees the OOM (last minute) plus the
    minute before it.
    """
    # poll_interval is the wall-clock sleep between reads. The source
    # uses ``chunks_per_time * 60`` seconds, so chunks_per_time is in
    # minutes.
    analyzer.chunks_per_time = max(poll_interval_sec / 60.0, 0.0)
    # Generous counter — the deadline patch is what bounds the loop.
    analyzer.stop_accumulating_count = 1000
    # End-phase grabs the last
    # ``int(logs_minutes_before_job_end / chunks_per_time)`` history
    # entries and concatenates them with the freshly read tail.
    analyzer.logs_minutes_before_job_end = end_window_minutes
    analyzer.repeated_amount = 3


def _prime_state(analyzer, path: str, cycle: int) -> None:
    """Set per-call cfg + seed shared state the analyzer assumes exists.

    ``analyze_logs_rt_start`` appends to ``job_inline_data_dict[path]`` without
    initializing it, and ``_streaming_attribution`` reads
    ``cycle_counter_dict[ck]`` without a setdefault. Seed both here.
    """
    analyzer._init_config["log_path"] = path
    analyzer._init_config["cycle_counter"] = cycle
    analyzer._init_config["attribution"] = True

    from nvidia_resiliency_ext.attribution.log_analyzer import nvrx_logsage

    analyzer.job_inline_data_dict.setdefault(path, [])
    ck = nvrx_logsage._cycle_counter_key(path)
    analyzer.cycle_counter_dict.setdefault(ck, 0)


def _run_start_phase(analyzer, path: str, cycle: int):
    """Drive the streaming start phase — direct call to analyze_logs_rt_start."""
    _prime_state(analyzer, path, cycle)
    return asyncio.run(analyzer.analyze_logs_rt_start())


def _run_end_phase(analyzer, path: str, cycle: int):
    """Drive the end-of-cycle attribution.

    Calls ``analyze_logs`` (the unified method's history-aware branch
    rebuilds the LLM input as ``tail + last-N history chunks``) to get a
    single ``ApplicationData``, then routes that through
    ``_streaming_attribution`` to obtain an ``ErrorAttribution``.
    """
    _prime_state(analyzer, path, cycle)

    from nvidia_resiliency_ext.attribution.base import effective_run_or_init_config

    output_list = asyncio.run(analyzer.analyze_logs())
    if not output_list:
        return None
    cfg = effective_run_or_init_config(analyzer._init_config)
    return analyzer._streaming_attribution(output_list[0], cfg, path)


def _wait_for_file(path: str, timeout_sec: float = 30.0) -> bool:
    """Block until ``path`` exists, up to ``timeout_sec``."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if os.path.isfile(path):
            return True
        time.sleep(0.5)
    return False


def run_dispatch_all_cycles(
    log_dir: str,
    num_cycles: int,
    cycle_duration_sec: float,
    poll_interval_sec: float,
    end_window_minutes: float,
) -> None:
    from nvidia_resiliency_ext.attribution.log_analyzer import nvrx_logsage

    first_path = cycle_log_path(log_dir, 0)
    # Writer truncates files up front, so the file is expected to exist
    # by the time we start dispatching cycle 0. Be tolerant of a small
    # startup race when launched in parallel.
    if not _wait_for_file(first_path, timeout_sec=30):
        print(
            f"[dispatch] {first_path} did not appear within 30s — start "
            "the writer first or pass --log-dir.",
            file=sys.stderr,
        )
        sys.exit(2)

    analyzer = nvrx_logsage.NVRxLogAnalyzer(
        {
            "log_path": first_path,
        }
    )
    _override_scalar_config(analyzer, poll_interval_sec, end_window_minutes)

    per_cycle_result: list[tuple[int, str, object]] = []

    for cycle in range(num_cycles):
        path = cycle_log_path(log_dir, cycle)
        if not _wait_for_file(path, timeout_sec=poll_interval_sec):
            print(
                f"[dispatch] cycle {cycle}: {path} missing; skipping",
                file=sys.stderr,
                flush=True,
            )
            continue

        print(
            f"\n[dispatch] === cycle {cycle} " f"({os.path.basename(path)}) ===",
            flush=True,
        )

        # Phase 1 — analyze_logs_rt_start, bounded by deadline-patched sleep.
        print(
            f"[dispatch] cycle {cycle}: phase=start → "
            f"analyze_logs_rt_start (deadline {cycle_duration_sec:.0f}s, "
            f"poll {poll_interval_sec:.0f}s)",
            flush=True,
        )
        start_t0 = time.monotonic()
        with patch.object(
            nvrx_logsage.asyncio,
            "sleep",
            _make_deadline_sleep(cycle_duration_sec, poll_interval_sec),
        ):
            try:
                _run_start_phase(analyzer, path, cycle)
            except _Phase1Deadline:
                print(
                    f"[dispatch] cycle {cycle}: start deadline reached",
                    flush=True,
                )
        start_elapsed = time.monotonic() - start_t0

        history_len = len(analyzer.job_inline_data_dict.get(path, []))
        print(
            f"[dispatch] cycle {cycle}: start took {start_elapsed:.2f}s, "
            f"job_inline_data_dict[{os.path.basename(path)}] len="
            f"{history_len}",
            flush=True,
        )

        # Phase 2 — analyze_logs (history-aware) + _streaming_attribution.
        print(
            f"[dispatch] cycle {cycle}: phase=end   → " "analyze_logs + _streaming_attribution",
            flush=True,
        )
        end_t0 = time.monotonic()
        end_result = _run_end_phase(analyzer, path, cycle)
        end_elapsed = time.monotonic() - end_t0
        print(
            f"[dispatch] cycle {cycle}: end took {end_elapsed:.2f}s",
            flush=True,
        )
        # The end phase OR-reduces checkpoint_saved across all per-poll
        # entries in job_inline_data_dict[path]; recompute the same way
        # for visibility.
        ckpt_saved_in_history = any(
            getattr(item[2], "checkpoint_saved", False)
            for item in analyzer.job_inline_data_dict.get(path, [])
        )

        if end_result is None:
            print(
                f"[dispatch] cycle {cycle}: end returned None "
                f"(checkpoint_saved_in_history={ckpt_saved_in_history})",
                file=sys.stderr,
                flush=True,
            )
        else:
            auto_resume = getattr(end_result, "auto_resume", "?")
            attribution = getattr(end_result, "attribution", "?")
            verbose = getattr(end_result, "auto_resume_verbose", "")
            ckpt_on_result = getattr(end_result, "checkpoint_saved", None)
            print(
                f"[dispatch] cycle {cycle}: end → "
                f"auto_resume={auto_resume!r}, "
                f"attribution={attribution!r}, "
                f"verbose={verbose!r}, "
                f"checkpoint_saved={ckpt_on_result!r} "
                f"(history={ckpt_saved_in_history})",
                flush=True,
            )
        per_cycle_result.append(
            (cycle, path, end_result, ckpt_saved_in_history, start_elapsed, end_elapsed)
        )

    print("\n[dispatch] ====== summary ======", flush=True)
    print("[dispatch] attribution_dict (per path):", flush=True)
    for path, attribution in analyzer.attribution_dict.items():
        print(f"  {os.path.basename(path)}: {attribution!r}", flush=True)

    print("[dispatch] cycle_counter_dict (per stripped key):", flush=True)
    for key, counter in analyzer.cycle_counter_dict.items():
        print(f"  {os.path.basename(key)}: {counter}", flush=True)

    print("[dispatch] per-cycle auto_resume:", flush=True)
    for cycle, _path, result, ckpt, start_s, end_s in per_cycle_result:
        ar = getattr(result, "auto_resume", None) if result else None
        verb = getattr(result, "auto_resume_verbose", "") if result else ""
        ckpt_tag = " [checkpoint_saved]" if ckpt else ""
        print(
            f"  cycle {cycle}: {ar!r}  ({verb!r}){ckpt_tag} "
            f"[start={start_s:.2f}s, end={end_s:.2f}s]",
            flush=True,
        )

    _assert_canonical_run(analyzer, per_cycle_result, log_dir, num_cycles)


def _assert_canonical_run(
    analyzer,
    per_cycle_result: list[tuple],
    log_dir: str,
    num_cycles: int,
) -> None:
    """Assert behavior of the canonical 5-cycle scenario.

    Assumes the writer defaults (``--checkpoint-cycles second-last`` for the
    minute-3 placement, ``--checkpoint-early-cycles last`` for the minute-0
    placement), ``repeated_amount=3`` (set by ``_override_scalar_config``),
    and that ``CHUNK_TEMPLATES`` produce identical OOMs across all cycles.
    Other invocations skip with a summary line so the script remains usable
    for ad-hoc runs.

    Per-cycle expectations:
      cycle 0, 1 → LLM-attributed STOP (no override yet; counter climbs 0→2)
      cycle 2   → repeated-issue STOP override (counter at threshold)
      cycle 3   → bypass + checkpoint override → RESTART IMMEDIATE; the
                  checkpoint flag appears mid-stream (chunk index 3)
      cycle 4   → bypass + checkpoint override → RESTART IMMEDIATE; the
                  checkpoint flag appears in the very first poll (chunk
                  index 0), i.e. at the head of job_inline_data_dict[path]
    """
    if num_cycles != NUM_CYCLES:
        print(
            f"[dispatch] assertions skipped (num_cycles={num_cycles}, "
            f"canonical requires {NUM_CYCLES})",
            flush=True,
        )
        return

    from nvidia_resiliency_ext.attribution.log_analyzer import nvrx_logsage

    assert (
        len(per_cycle_result) == num_cycles
    ), f"expected {num_cycles} per-cycle results, got {len(per_cycle_result)}"

    # All cycles must produce an ErrorAttribution and attribute the OOM.
    for cycle, _path, result, _ckpt, _start_s, _end_s in per_cycle_result:
        assert result is not None, f"cycle {cycle}: end_result is None"
        attribution_text = str(getattr(result, "attribution", ""))
        assert (
            "OutOfMemoryError" in attribution_text or "out of memory" in attribution_text.lower()
        ), f"cycle {cycle}: attribution missing OOM token: {attribution_text!r}"

    # Per-cycle history: cycles 0..2 see no checkpoint; cycles 3 and 4 each
    # see at least one history entry with checkpoint_saved=True. For cycle 4
    # specifically the flag must be on the *first* history entry (writer's
    # minute-0 placement), which is what makes this scenario distinct from
    # cycle 3.
    for cycle in range(num_cycles - 2):
        cpath = cycle_log_path(log_dir, cycle)
        history = analyzer.job_inline_data_dict.get(cpath, [])
        assert not any(item[2].checkpoint_saved for item in history), (
            f"cycle {cycle}: unexpected checkpoint_saved=True in history "
            f"({len(history)} entries)"
        )

    cycle3_path = cycle_log_path(log_dir, num_cycles - 2)
    cycle3_history = analyzer.job_inline_data_dict.get(cycle3_path, [])
    assert any(item[2].checkpoint_saved for item in cycle3_history), (
        f"cycle {num_cycles - 2}: expected checkpoint_saved=True in at least "
        f"one history entry ({len(cycle3_history)} entries)"
    )

    cycle4_path = cycle_log_path(log_dir, num_cycles - 1)
    cycle4_history = analyzer.job_inline_data_dict.get(cycle4_path, [])
    assert any(item[2].checkpoint_saved for item in cycle4_history), (
        f"cycle {num_cycles - 1}: expected checkpoint_saved=True in at least "
        f"one history entry ({len(cycle4_history)} entries)"
    )
    assert cycle4_history and cycle4_history[0][2].checkpoint_saved, (
        f"cycle {num_cycles - 1}: expected the FIRST history entry to carry "
        f"checkpoint_saved=True (minute-0 placement); got history of length "
        f"{len(cycle4_history)} with first entry checkpoint_saved="
        f"{cycle4_history[0][2].checkpoint_saved if cycle4_history else 'N/A'}"
    )

    # Cycle 2 (third identical attribution, repeated_amount=3) → repeated-issue
    # STOP override fires; verbose is overwritten deterministically.
    cycle2_result = per_cycle_result[2][2]
    assert cycle2_result is not None
    assert (
        cycle2_result.auto_resume == "STOP - DONT RESTART IMMEDIATE"
    ), f"cycle 2 expected STOP, got {cycle2_result.auto_resume!r}"
    assert (
        cycle2_result.auto_resume_verbose == "Stop job due to repeated issue"
    ), f"cycle 2 verbose mismatch: {cycle2_result.auto_resume_verbose!r}"

    # Cycle 3 and cycle 4 (both checkpoint_saved=True) must end with RESTART
    # IMMEDIATE — either because the LLM said it, or because the
    # checkpoint_saved override flipped an LLM-STOP.
    cycle3_result = per_cycle_result[3][2]
    assert cycle3_result is not None
    assert cycle3_result.auto_resume == "RESTART IMMEDIATE", (
        f"cycle 3 expected RESTART IMMEDIATE (LLM or checkpoint override), "
        f"got {cycle3_result.auto_resume!r}"
    )

    cycle4_result = per_cycle_result[4][2]
    assert cycle4_result is not None
    assert cycle4_result.auto_resume == "RESTART IMMEDIATE", (
        f"cycle 4 expected RESTART IMMEDIATE (LLM or checkpoint override; "
        f"checkpoint sits in the first poll of history), "
        f"got {cycle4_result.auto_resume!r}"
    )

    # cycle_counter_dict reset to 1 after cycle 3's bypass (then re-reset
    # on cycle 4's bypass) — final value 1.
    ck = nvrx_logsage._cycle_counter_key(cycle4_path)
    counter_final = analyzer.cycle_counter_dict.get(ck)
    assert counter_final == 1, (
        f"cycle_counter_dict[{os.path.basename(ck)}] expected 1 after bypass, "
        f"got {counter_final}"
    )

    print("[dispatch] all assertions passed.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=NUM_CYCLES,
        help=f"Number of cycle files (default {NUM_CYCLES})",
    )
    parser.add_argument(
        "--cycle-duration",
        type=float,
        default=300.0,
        help="Seconds the start phase polls per cycle before exiting " "(default 300s = 5 min)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="Seconds between polls inside the start phase (default 60s = 1 min)",
    )
    parser.add_argument(
        "--end-window-minutes",
        type=float,
        default=2.0,
        help="Minutes of streaming history the end phase glues to the "
        "freshly read tail before re-running extraction. With the "
        "writer at 1 chunk/min, '2' = tail + last 2 minutes "
        "(default 2)",
    )
    args = parser.parse_args()

    if not _logsage_available():
        print("logsage package not installed; aborting.", file=sys.stderr)
        return 2
    if not _api_key_available():
        print("LLM API key not configured; aborting.", file=sys.stderr)
        return 2

    run_dispatch_all_cycles(
        args.log_dir,
        args.num_cycles,
        args.cycle_duration,
        args.poll_interval,
        args.end_window_minutes,
    )
    print("[dispatch] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
