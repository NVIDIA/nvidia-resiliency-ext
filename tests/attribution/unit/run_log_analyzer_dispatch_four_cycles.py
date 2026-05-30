"""Script 2/2: drive ``NVRxLogAnalyzer._analyze_logs_rt_dispatch`` over 4 cycles.

Pairs with ``run_log_writer_dispatch_four_cycles.py``. A single analyzer
instance is reused so ``attribution_dict`` / ``cycle_counter_dict``
persist across cycles. For each cycle ``i`` in ``[0..3]`` the script:

  1. Sets ``cfg["job_stage"] = "start"`` and calls
     ``_analyze_logs_rt_dispatch`` → routes to
     ``analyze_logs_rt_start``. Polls the cycle's file every
     ``--poll-interval`` seconds for ``--cycle-duration`` seconds total
     (default 60s / 300s = poll 1×/min for 5 min), then returns
     ``None``.
  2. Sets ``cfg["job_stage"] = "end"`` and calls it again → routes to
     ``analyze_logs_rt_end``. The end phase rebuilds its LLM input as
     ``tail + last --end-window-minutes of history`` (default 2 min),
     re-runs extraction, and returns an ``ErrorAttribution``.

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
  - ``analyze_logs_rt_end`` does ``cycle_counter_dict[ck] += 1`` without
    a ``setdefault`` — we pre-seed the shared key to 0.
  - The start branch's ``while True`` polling loop is bounded by
    patching ``time.sleep`` with a deadline (matches the existing
    ``run_log_analyzer_start_four_cycles.py`` pattern). When the
    deadline elapses the sleep raises ``_Phase1Deadline``, the start
    coroutine exits, and the script moves on to the end phase.

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
NUM_CYCLES = 4


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
    real_sleep = time.sleep
    deadline = time.monotonic() + start_duration_sec

    def deadline_sleep(*_args, **_kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Phase1Deadline()
        real_sleep(min(poll_interval_sec, remaining))

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


def _dispatch(analyzer, path: str, job_stage: str, cycle: int):
    """Configure cfg + minimal state, then call _analyze_logs_rt_dispatch."""
    analyzer._init_config["log_path"] = path
    analyzer._init_config["job_stage"] = job_stage
    analyzer._init_config["cycle_counter"] = cycle
    analyzer._init_config["attribution"] = True

    from nvidia_resiliency_ext.attribution.log_analyzer import nvrx_logsage

    analyzer.job_inline_data_dict.setdefault(path, [])
    ck = nvrx_logsage._cycle_counter_key(path)
    analyzer.cycle_counter_dict.setdefault(ck, 0)

    return asyncio.run(analyzer._analyze_logs_rt_dispatch())


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
            "job_stage": "start",
            "is_streaming_logs": True,
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

        # Phase 1 — start branch via dispatcher, bounded by deadline.
        print(
            f"[dispatch] cycle {cycle}: job_stage=start → "
            f"analyze_logs_rt_start (deadline {cycle_duration_sec:.0f}s, "
            f"poll {poll_interval_sec:.0f}s)",
            flush=True,
        )
        start_t0 = time.monotonic()
        with patch.object(
            nvrx_logsage.time,
            "sleep",
            _make_deadline_sleep(cycle_duration_sec, poll_interval_sec),
        ):
            try:
                _dispatch(analyzer, path, "start", cycle)
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

        # Phase 2 — end branch via dispatcher.
        print(
            f"[dispatch] cycle {cycle}: job_stage=end   → " "analyze_logs_rt_end",
            flush=True,
        )
        end_t0 = time.monotonic()
        end_result = _dispatch(analyzer, path, "end", cycle)
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
