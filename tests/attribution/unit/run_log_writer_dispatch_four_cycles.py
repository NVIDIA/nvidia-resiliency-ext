"""Script 1/2: write five 5-minute per-cycle log files for the dispatch test.

Pairs with ``run_log_analyzer_dispatch_four_cycles.py``. For each cycle
``i`` in ``[0..4]``:

  - Truncates ``nvrx_{i}.log``
  - Appends 5 chunks at ``--chunk-interval`` second intervals
    (default 60s → cycle is ``chunks_per_cycle * chunk_interval`` =
    300s = 5 minutes)
  - Sleeps after the *last* chunk too, so the per-cycle window is
    fully 5 minutes before the next cycle's file begins

Total wall time for the default 5 × 5 = 25 minutes. All five files end
up with the same OOM error, which is what drives the repeated-issue
stop guard in the analyzer.

Canonical checkpoint layout (defaults):
  - cycle 3: checkpoint at minute 3 (late-training chunk) — exercises
    the OR-reduced-history checkpoint_saved bypass when the flag
    arrives late in the polling window.
  - cycle 4: checkpoint at minute 0 (boot chunk) — exercises the same
    bypass when the flag arrives in the first poll, so it lives at the
    head of ``job_inline_data_dict[path]`` rather than the tail.

Usage:
    python tests/attribution/unit/run_log_writer_dispatch_four_cycles.py
    # Shorter intervals for local iteration:
    python tests/attribution/unit/run_log_writer_dispatch_four_cycles.py \\
        --chunk-interval 5
"""

import argparse
import datetime
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
NUM_CYCLES = 5
CHUNKS_PER_CYCLE = 5
CHECKPOINT_CHUNK_INDEX = 3  # minute 3 — the late-training chunk
CHECKPOINT_CHUNK_INDEX_EARLY = 0  # minute 0 — the boot/cycle-marker chunk

# 5 chunks per cycle, one per minute → 5-minute cycle window. The OOM
# lands in the last chunk so it falls inside the end phase's
# tail + 2-minute history slice.
CHUNK_TEMPLATES = [
    # minute 0 — boot / cycle marker
    "[{ts}] starting distributed training on 8 GPUs\n"
    "[{ts}] FT: initialized\n"
    "[{ts}] Cycle: {cycle} begin\n",
    # minute 1 — early training
    "[{ts}] step 100 loss=0.512 lr=1e-4\n" "[{ts}] step 200 loss=0.487 lr=1e-4\n",
    # minute 2 — mid training
    "[{ts}] step 300 loss=0.461 lr=1e-4\n" "[{ts}] step 400 loss=0.439 lr=1e-4\n",
    # minute 3 — late training (last clean window before failure).
    # In "checkpoint cycles" we append CHECKPOINT_LINE here so the
    # end phase observes checkpoint_saved=True alongside the OOM.
    "[{ts}] step 500 loss=0.421 lr=1e-4\n" "[{ts}] step 600 loss=0.408 lr=1e-4\n",
    # minute 4 — failure (this chunk must land inside the end phase's
    # tail + 2-minute window)
    "[{ts}] ERROR: torch.cuda.OutOfMemoryError: CUDA out of memory.\n"
    "Traceback (most recent call last):\n"
    "  File 'train.py', line 142, in train_step\n"
    "    loss = model(input_ids).loss\n"
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n",
]

# Sentinel line the LLM extraction layer recognises as a successful
# checkpoint save. Appended into minute 3 of any cycle listed in
# ``--checkpoint-cycles``. Wording is intentionally explicit so the
# extraction prompt has no ambiguity.
CHECKPOINT_LINE = (
    "[{ts}] [checkpointing.py:142] Saved global checkpoint at step 600 "
    "to /ckpt/global_step_600.pt (rank 0)\n"
)


def _parse_checkpoint_cycles(value: str, num_cycles: int) -> set[int]:
    """Resolve a checkpoint-cycles CLI value to a set of cycle indices.

    Accepts ``""``/``"none"`` (empty set), ``"all"``, ``"last"`` (last cycle),
    ``"second-last"`` (second-to-last cycle), or a comma-separated list of
    explicit indices like ``"0,2"``.
    """
    v = (value or "").strip().lower()
    if v in ("", "none"):
        return set()
    if v == "last":
        return {num_cycles - 1}
    if v == "second-last":
        return {num_cycles - 2}
    if v == "all":
        return set(range(num_cycles))
    out: set[int] = set()
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.add(int(piece))
    return out


def cycle_log_path(log_dir: str, cycle: int) -> str:
    return os.path.join(log_dir, f"nvrx_{cycle}.log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=_HERE)
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=NUM_CYCLES,
        help=f"Number of cycle files (default {NUM_CYCLES})",
    )
    parser.add_argument(
        "--chunk-interval",
        type=float,
        default=60.0,
        help="Seconds between chunks; also the gap after the last chunk "
        "of a cycle, so each cycle is chunks_per_cycle * "
        "chunk_interval long (default 60s).",
    )
    parser.add_argument(
        "--checkpoint-cycles",
        default="second-last",
        help="Which cycle indices include a 'Saved checkpoint' line in "
        "their minute-3 chunk (late-training placement). Accepts "
        "'last', 'second-last' (default — second-to-last cycle, i.e. "
        "cycle 3 in the canonical 5-cycle setup), 'all', 'none', or a "
        "comma-separated list like '0,2'.",
    )
    parser.add_argument(
        "--checkpoint-early-cycles",
        default="last",
        help="Which cycle indices include a 'Saved checkpoint' line in "
        "their minute-0 chunk (boot/cycle-marker placement). Accepts "
        "'last' (default — final cycle, i.e. cycle 4 in the canonical "
        "5-cycle setup), 'second-last', 'all', 'none', or a "
        "comma-separated list like '0,2'.",
    )
    args = parser.parse_args()

    checkpoint_cycles = _parse_checkpoint_cycles(args.checkpoint_cycles, args.num_cycles)
    checkpoint_early_cycles = _parse_checkpoint_cycles(
        args.checkpoint_early_cycles, args.num_cycles
    )

    os.makedirs(args.log_dir, exist_ok=True)
    # Truncate up front so the analyzer's first read against any cycle
    # file isn't tripped by stale content from a prior run.
    paths = [cycle_log_path(args.log_dir, c) for c in range(args.num_cycles)]
    for p in paths:
        open(p, "w").close()

    chunks_per_cycle = len(CHUNK_TEMPLATES)
    cycle_seconds = chunks_per_cycle * args.chunk_interval

    def _summarize(cset: set[int]) -> str:
        return ",".join(str(c) for c in sorted(cset)) if cset else "none"

    print(
        f"[writer] {args.num_cycles} cycles × {chunks_per_cycle} chunks @ "
        f"{args.chunk_interval:.0f}s = {cycle_seconds:.0f}s per cycle "
        f"({args.num_cycles * cycle_seconds:.0f}s total); "
        f"checkpoint cycles late(min3): {_summarize(checkpoint_cycles)}, "
        f"early(min0): {_summarize(checkpoint_early_cycles)}",
        flush=True,
    )

    try:
        for cycle, path in enumerate(paths):
            print(f"[writer] cycle {cycle}: {path}", flush=True)
            for chunk_idx, tmpl in enumerate(CHUNK_TEMPLATES):
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                content = tmpl.format(ts=ts, cycle=cycle)
                ckpt_tags = []
                if chunk_idx == CHECKPOINT_CHUNK_INDEX and cycle in checkpoint_cycles:
                    content += CHECKPOINT_LINE.format(ts=ts)
                    ckpt_tags.append("checkpoint")
                if chunk_idx == CHECKPOINT_CHUNK_INDEX_EARLY and cycle in checkpoint_early_cycles:
                    content += CHECKPOINT_LINE.format(ts=ts)
                    ckpt_tags.append("checkpoint-early")
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
                suffix = (" + " + " + ".join(ckpt_tags)) if ckpt_tags else ""
                print(
                    f"[writer]   cycle {cycle} chunk "
                    f"{chunk_idx + 1}/{chunks_per_cycle} @ {ts}{suffix}",
                    flush=True,
                )
                # Sleep after every chunk including the last so the
                # cycle window is fully `chunks_per_cycle *
                # chunk_interval` seconds before the next file begins.
                time.sleep(args.chunk_interval)
    except KeyboardInterrupt:
        print("\n[writer] interrupted; stopping.", flush=True)
    print("[writer] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
