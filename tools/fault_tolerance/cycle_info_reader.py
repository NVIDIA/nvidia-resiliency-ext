#!/usr/bin/env python3
"""
Read and validate NVRx cycle_info.* files from collected artifacts.

Usage:
    python cycle_info_reader.py <artifacts_dir> [--job-id <id>] [--show-log-files]

Output: JSON with cycle sequence, anomalies, active_nodes/standby_nodes per cycle.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

CYCLE_INFO_RE = re.compile(r"cycle_info\.(?P<job_id>[^.]+)\.(?P<attempt>\d+)\.(?P<cycle>\d+)$")


@dataclass
class CycleInfo:
    job_id: str
    attempt_index: int
    cycle_number: int
    cycle_start_time: str
    cycle_end_time: str
    cycle_log_file: str
    active_nodes: str
    standby_nodes: str
    generation: int
    source_file: str


def find_cycle_info_dir(artifacts_dir: str) -> str | None:
    """Find the cycle_infos directory under artifacts."""
    candidates = [
        os.path.join(artifacts_dir, "nvrx", "cycle_infos"),
        os.path.join(artifacts_dir, "cycle_infos"),
        artifacts_dir,
    ]
    for c in candidates:
        if os.path.isdir(c) and any(
            CYCLE_INFO_RE.match(os.path.basename(f))
            for f in os.listdir(c)
            if not f.endswith(".current")
        ):
            return c
    return None


def load_cycle_infos(ci_dir: str, job_id: str | None) -> list[CycleInfo]:
    """Load all cycle_info files, optionally filtered by job_id."""
    infos = []
    for fname in sorted(os.listdir(ci_dir)):
        m = CYCLE_INFO_RE.match(fname)
        if not m:
            continue
        if job_id and m.group("job_id") != job_id:
            continue
        fpath = os.path.join(ci_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            continue

        infos.append(
            CycleInfo(
                job_id=data.get("job_id", m.group("job_id")),
                attempt_index=int(data.get("attempt_index", m.group("attempt"))),
                cycle_number=int(data.get("cycle_number", m.group("cycle"))),
                cycle_start_time=data.get("cycle_start_time", ""),
                cycle_end_time=data.get("cycle_end_time", ""),
                cycle_log_file=data.get("cycle_log_file", ""),
                active_nodes=data.get("active_nodes", ""),
                standby_nodes=data.get("standby_nodes", ""),
                generation=int(data.get("generation", 0)),
                source_file=fpath,
            )
        )

    infos.sort(key=lambda c: (c.attempt_index, c.cycle_number))
    return infos


def validate_sequence(infos: list[CycleInfo]) -> list[str]:
    """Return list of anomaly strings."""
    anomalies = []
    if not infos:
        return ["no cycle_info files found"]

    # Check for cycles without end time (crashed mid-cycle)
    for ci in infos:
        if not ci.cycle_end_time or ci.cycle_end_time == "":
            anomalies.append(
                f"cycle {ci.cycle_number} (attempt {ci.attempt_index}) has no cycle_end_time"
            )

    # Check for gaps in cycle sequence per attempt
    by_attempt: dict[int, list[int]] = {}
    for ci in infos:
        by_attempt.setdefault(ci.attempt_index, []).append(ci.cycle_number)

    for attempt, cycles in sorted(by_attempt.items()):
        cycles_sorted = sorted(cycles)
        for i in range(1, len(cycles_sorted)):
            if cycles_sorted[i] != cycles_sorted[i - 1] + 1:
                anomalies.append(
                    f"attempt {attempt}: gap between cycle {cycles_sorted[i-1]} and {cycles_sorted[i]}"
                )

    return anomalies


def build_output(infos: list[CycleInfo], anomalies: list[str], show_log_files: bool) -> dict:
    cycles = []
    for ci in infos:
        entry = {
            "attempt": ci.attempt_index,
            "cycle": ci.cycle_number,
            "start": ci.cycle_start_time,
            "end": ci.cycle_end_time,
            "active_nodes": ci.active_nodes,
            "standby_nodes": ci.standby_nodes,
            "has_standby": bool(ci.standby_nodes),
        }
        if show_log_files:
            entry["cycle_log_file"] = ci.cycle_log_file
        cycles.append(entry)

    total = len(infos)
    cycles_with_standby = sum(1 for ci in infos if ci.standby_nodes)
    cycle_numbers = sorted({ci.cycle_number for ci in infos if ci.cycle_number > 0})

    return {
        "total_cycle_infos": total,
        "cycle_count_excluding_warmup": len(cycle_numbers),
        "cycles_excluding_warmup": cycle_numbers,
        "cycles_with_standby_nodes": cycles_with_standby,
        "anomalies": anomalies,
        "sequence_valid": len(anomalies) == 0,
        "cycles": cycles,
    }


def main():
    parser = argparse.ArgumentParser(description="Read and validate NVRx cycle_info files")
    parser.add_argument("artifacts_dir", help="Local artifacts directory")
    parser.add_argument("--job-id", help="Filter by SLURM array job ID")
    parser.add_argument(
        "--show-log-files",
        action="store_true",
        help="Include cycle_log_file paths in output",
    )
    args = parser.parse_args()

    artifacts_dir = os.path.abspath(args.artifacts_dir)
    if not os.path.isdir(artifacts_dir):
        print(f"Error: {artifacts_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Auto-detect job_id from run_result.json if not provided
    job_id = args.job_id
    if not job_id:
        run_result_path = os.path.join(artifacts_dir, "run_result.json")
        if os.path.exists(run_result_path):
            with open(run_result_path) as f:
                rr = json.load(f)
            job_id = rr.get("array_job_id") or rr.get("job_id")

    ci_dir = find_cycle_info_dir(artifacts_dir)
    if not ci_dir:
        print(
            json.dumps(
                {
                    "error": "no cycle_info directory found",
                    "artifacts_dir": artifacts_dir,
                    "anomalies": ["cycle_info directory not found"],
                    "sequence_valid": False,
                }
            )
        )
        sys.exit(1)

    infos = load_cycle_infos(ci_dir, job_id)
    anomalies = validate_sequence(infos)
    output = build_output(infos, anomalies, args.show_log_files)
    output["artifacts_dir"] = artifacts_dir
    output["cycle_info_dir"] = ci_dir

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
