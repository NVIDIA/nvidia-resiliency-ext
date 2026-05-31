"""Run the writer + dispatch analyzer in parallel.

Launches ``run_log_writer_dispatch_four_cycles.py`` and
``run_log_analyzer_dispatch_four_cycles.py`` as concurrent subprocesses,
sharing the same log directory. Both pace themselves so cycles align:

  - Writer: 5 chunks × ``--chunk-interval`` (default 60s) = 300s per cycle
  - Analyzer: polls every ``--poll-interval`` (60s) for ``--cycle-duration``
    (300s) per cycle, then runs the end phase over
    ``tail + --end-window-minutes`` of history (default 2 min)

Default canonical scenario is 5 cycles: cycles 0-2 with no checkpoint,
cycle 3 with a minute-3 (late) checkpoint, and cycle 4 with a minute-0
(boot) checkpoint. Cycles 3 and 4 both exercise the checkpoint_saved
bypass; cycle 4 specifically exercises the case where the flag sits at
the head of ``job_inline_data_dict[path]`` rather than the tail.

Each subprocess's stdout is forwarded line-by-line with a ``[writer]`` /
``[reader]`` prefix so the output is greppable.

The driver exits with the worst of the two return codes.

Usage:
    # Realistic 25-min run:
    python tests/attribution/unit/run_dispatch_test_parallel.py
    # Fast iteration (5s polls → ~125 s total):
    python tests/attribution/unit/run_dispatch_test_parallel.py \\
        --chunk-interval 5 --poll-interval 5 --cycle-duration 25
"""

import argparse
import os
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
WRITER_SCRIPT = os.path.join(_HERE, "run_log_writer_dispatch_four_cycles.py")
ANALYZER_SCRIPT = os.path.join(_HERE, "run_log_analyzer_dispatch_four_cycles.py")


def _stream(proc: subprocess.Popen, label: str) -> None:
    """Forward proc.stdout to our stdout, prefixed by ``label``."""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        # Each subprocess already prefixes its lines (``[writer] ...``,
        # ``[dispatch] ...``) — add the outer label so it's obvious
        # which subprocess emitted the line in interleaved output.
        print(f"{label} {line}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=_HERE)
    parser.add_argument("--num-cycles", type=int, default=5)
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=60.0,
        help="Seconds between writer chunks (default 60s = 1 min)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="Seconds between analyzer polls (default 60s = 1 min)",
    )
    parser.add_argument(
        "--cycle-duration",
        type=float,
        default=300.0,
        help="Analyzer's per-cycle deadline (default 300s = 5 min)",
    )
    parser.add_argument(
        "--end-window-minutes",
        type=float,
        default=2.0,
        help="Minutes of history the end phase reuses (default 2 = " "tail + last 2 minutes)",
    )
    parser.add_argument(
        "--checkpoint-cycles",
        default="second-last",
        help="Which cycle indices get a minute-3 'Saved checkpoint' line "
        "(late-training placement). Default 'second-last' = cycle 3 in "
        "the canonical 5-cycle setup; exercises the bypass when the "
        "checkpoint flag arrives late in the polling window.",
    )
    parser.add_argument(
        "--checkpoint-early-cycles",
        default="last",
        help="Which cycle indices get a minute-0 'Saved checkpoint' line "
        "(boot-chunk placement). Default 'last' = cycle 4 in the "
        "canonical 5-cycle setup; exercises the bypass when the "
        "checkpoint flag is captured by the very first poll.",
    )
    parser.add_argument(
        "--writer-head-start",
        type=float,
        default=0.5,
        help="Seconds the writer leads the analyzer so the first poll "
        "finds a non-empty file (default 0.5s)",
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    # Clean stale cycle files so the analyzer's first read sees fresh
    # writer output rather than a prior run's content.
    for cycle in range(args.num_cycles):
        path = os.path.join(args.log_dir, f"nvrx_{cycle}.log")
        if os.path.exists(path):
            os.remove(path)
    # Drop the pickled state file too if a previous run left one.
    stale = os.path.join(args.log_dir, "nvrx_four_cycles.state.pkl")
    if os.path.exists(stale):
        os.remove(stale)

    common_env = os.environ.copy()
    # Force unbuffered Python so the line streamer sees output promptly.
    common_env["PYTHONUNBUFFERED"] = "1"

    writer_cmd = [
        sys.executable,
        WRITER_SCRIPT,
        "--log-dir",
        args.log_dir,
        "--num-cycles",
        str(args.num_cycles),
        "--chunk-interval",
        str(args.chunk_interval),
        "--checkpoint-cycles",
        args.checkpoint_cycles,
        "--checkpoint-early-cycles",
        args.checkpoint_early_cycles,
    ]
    analyzer_cmd = [
        sys.executable,
        ANALYZER_SCRIPT,
        "--log-dir",
        args.log_dir,
        "--num-cycles",
        str(args.num_cycles),
        "--cycle-duration",
        str(args.cycle_duration),
        "--poll-interval",
        str(args.poll_interval),
        "--end-window-minutes",
        str(args.end_window_minutes),
    ]

    print(f"[driver] writer:   {' '.join(writer_cmd)}", flush=True)
    print(f"[driver] analyzer: {' '.join(analyzer_cmd)}", flush=True)
    print(
        f"[driver] launching in parallel, writer head start " f"{args.writer_head_start:.1f}s",
        flush=True,
    )

    writer = subprocess.Popen(
        writer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=common_env,
    )
    # Brief head start so the writer creates / writes to nvrx_0.log
    # before the analyzer's first read.
    if args.writer_head_start > 0:
        time.sleep(args.writer_head_start)
    analyzer = subprocess.Popen(
        analyzer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=common_env,
    )

    threads = [
        threading.Thread(target=_stream, args=(writer, "[W]"), daemon=True),
        threading.Thread(target=_stream, args=(analyzer, "[R]"), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        writer_rc = writer.wait()
        analyzer_rc = analyzer.wait()
    except KeyboardInterrupt:
        print("\n[driver] interrupted; terminating children", flush=True)
        writer.terminate()
        analyzer.terminate()
        writer_rc = writer.wait()
        analyzer_rc = analyzer.wait()

    # Both children have already exited (wait() returned), so their stdout
    # pipes are at EOF and each stream thread will drain the remaining buffered
    # output and finish on its own. Give them a few seconds to flush the final
    # cycle's result + the summary block before we exit (daemon threads would
    # otherwise be killed mid-flush, truncating the tail).
    for t in threads:
        t.join(timeout=5.0)

    print(
        f"\n[driver] writer exit={writer_rc}, analyzer exit={analyzer_rc}",
        flush=True,
    )
    return max(writer_rc, analyzer_rc)


if __name__ == "__main__":
    sys.exit(main())
