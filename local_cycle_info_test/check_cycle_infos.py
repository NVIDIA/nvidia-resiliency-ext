#!/usr/bin/env python3
"""Sanity-check NVRx cycle info files written by the CycleAnalyzer.

For each file, recomputes from the listed launcher outcomes what the analyzer should have
written, and compares:
  - first_detected_nodes: the launchers that ended FAILED or TERMINATED earliest (exact ties)
  - ended_at: last launcher's detection if every listed launcher completed, otherwise the
    first detection of a FAILED/TERMINATED launcher, otherwise the earliest detection
  - end_reason: completed / rank_failure / terminated / unknown, from the first detectors

Launchers that only followed a peer's restart (peer_restart) are counted in
launchers_reported but not listed, so they do not take part in the checks.

Usage:
  check_cycle_infos.py <cycle_infos dir or files...> [-v]

Reads only the given files, or the entries of one directory (a single listing); it does
not walk directory trees.
"""

import argparse
import json
import os
import sys
from datetime import datetime


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def cycle_files(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            names = [n for n in os.listdir(path) if n.startswith("cycle_info.")]
            files += [os.path.join(path, n) for n in names if not n.endswith(".current")]
        else:
            files.append(path)

    def cycle_number(path):
        tail = path.rsplit(".", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return sorted(files, key=cycle_number)


def expected(launchers):
    """What the aggregator should conclude from the listed launcher outcomes."""
    if not launchers:
        return "unknown", [], None
    detected = [(parse_time(l["detected_at"]), l) for l in launchers]
    if all(l["end"] == "completed" for l in launchers):
        return "completed", [], max(t for t, _ in detected)
    candidates = [(t, l) for t, l in detected if l["end"] in ("failed", "terminated")]
    if not candidates:
        return "unknown", [], min(t for t, _ in detected)
    first_time = min(t for t, _ in candidates)
    first = [l for t, l in candidates if t == first_time]
    if any(l["end"] == "failed" for l in first):
        reason = "rank_failure"
    elif any(l["end"] == "terminated" for l in first):
        reason = "terminated"
    else:
        reason = "unknown"
    return reason, first, first_time


def launcher_label(launcher):
    ranks = [r["global_rank"] for r in launcher.get("ranks", [])]
    span = f"{min(ranks)}-{max(ranks)}" if ranks else "-"
    return f"{launcher['node']}[ranks {span}]"


def check(path, verbose):
    with open(path) as f:
        info = json.load(f)
    outcome = info.get("outcome") or {}
    launchers = outcome.get("launcher_outcomes", [])
    reason, first, first_time = expected(launchers)

    written_reason = outcome.get("end_reason", "")
    written_ended = parse_time(outcome.get("ended_at", ""))
    written_first = sorted(outcome.get("first_detected_nodes", []))
    expected_first = sorted(l["node"] for l in first)

    problems = []
    if written_reason != reason:
        problems.append(f"end_reason {written_reason!r} != expected {reason!r}")
    if written_ended != first_time:
        problems.append(f"ended_at {written_ended} != expected {first_time}")
    # The file keeps microseconds, the analyzer compares the full float: launchers tied here
    # may not have been tied there, so the written ones need only be among the tied ones.
    remaining = list(expected_first)
    within_ties = bool(written_first) == bool(expected_first)
    for node in written_first:
        if node in remaining:
            remaining.remove(node)
        else:
            within_ties = False
    if not within_ties:
        problems.append(f"first_detected_nodes {written_first} not among expected {expected_first}")
    cycle_end = parse_time(info.get("cycle_end_time", ""))
    if written_ended is not None and cycle_end != written_ended:
        problems.append(f"cycle_end_time {cycle_end} != ended_at {written_ended}")

    times = sorted(parse_time(l["detected_at"]) for l in launchers)
    spread = (times[-1] - times[0]).total_seconds() if times else 0.0
    start = parse_time(info.get("cycle_start_time", ""))
    duration = (written_ended - start).total_seconds() if written_ended and start else None
    ends = {}
    for l in launchers:
        ends[l["end"]] = ends.get(l["end"], 0) + 1

    status = "OK" if not problems else "MISMATCH"
    print(
        f"{os.path.basename(path)}: {status} reason={written_reason} "
        f"reported={outcome.get('launchers_reported')}/{outcome.get('launchers_expected')} "
        f"listed={len(launchers)} {ends} missing={len(outcome.get('missing_nodes', []))} "
        f"duration={duration if duration is None else round(duration, 3)}s "
        f"detection_spread={spread:.3f}s"
    )
    for problem in problems:
        print(f"    {problem}")
    if verbose and first:
        print(f"    first detector(s) at {first_time}: {[launcher_label(l) for l in first]}")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="cycle_infos directory or cycle_info files")
    parser.add_argument("-v", "--verbose", action="store_true", help="show the first detectors")
    args = parser.parse_args()
    files = cycle_files(args.paths)
    if not files:
        sys.exit("no cycle_info files found")
    results = [check(path, args.verbose) for path in files]
    print(f"{sum(results)}/{len(results)} files OK")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
