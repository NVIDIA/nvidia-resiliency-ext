#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict, deque
from datetime import datetime

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Regular expression to parse log lines (handles optional rank number prefix and concatenated entries)
LOG_PATTERN = r"(?:\d+:\s+)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?Cycle: (\d+).*?Event: (\w+).*?Node: ([^ ]+).*?Time: ([^ ]+ \d{2}:\d{2}:\d{2}\.\d{3} UTC)"

# Regular expression to extract rank number from beginning of log line
RANK_PATTERN = r"^\s*(\d+):\s+"

# Regular expression to extract run_id from log lines
RUN_ID_PATTERN = r"^\s*\d+:\s+run_id\s*:\s*(\d+)"


def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object."""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f UTC")
    except ValueError as e:
        logging.error(f"Failed to parse timestamp '{timestamp_str}': {e}")
        raise


def generate_run_id(log_file_prefix):
    """Generate a run_id from the log file prefix."""
    basename = os.path.basename(log_file_prefix)
    if basename:
        return basename
    else:
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def dump_failure_logs(rank, cycle, rank_lines, run_id):
    """Dump the last 1000 lines for a failing rank to the brief log file."""
    if not run_id:
        run_id = "unknown"

    output_file = f"/tmp/nvrx_{run_id}_brief.log"

    try:
        with open(output_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FAILURE DETECTED - Cycle: {cycle}, Rank: {rank}\n")
            f.write(f"Last {len(rank_lines)} lines from rank {rank}:\n")
            f.write(f"{'='*80}\n")

            for line in rank_lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")

            f.write(f"{'='*80}\n\n")

        logging.info(
            f"Dumped {len(rank_lines)} lines for failing rank {rank} (cycle {cycle}) to {output_file}"
        )
    except Exception as e:
        logging.error(f"Failed to dump failure logs to {output_file}: {e}")


def parse_log_files(log_file_prefix, dump_failures=False):
    """Parse log files with the given prefix and extract relevant metrics."""
    logging.info(f"Searching for log files with prefix: {log_file_prefix}")
    node_metrics = {}
    log_files = glob.glob(f"{log_file_prefix}*")

    # For failure dumping: track lines by rank and detected failures
    rank_lines = defaultdict(lambda: deque(maxlen=1000))  # Last 1000 lines per rank
    failure_cycles_seen = set()  # Track cycles where we've already dumped failures
    parsed_run_id = None  # Will be extracted from logs

    if not log_files:
        logging.warning(f"No log files found with prefix: {log_file_prefix}")
        return node_metrics

    logging.debug(f"Found {len(log_files)} log files: {log_files}")
    for log_file in log_files:
        logging.debug(f"Processing log file: {log_file}")
        try:
            with open(log_file, "r") as f:
                for line in f:
                    original_line = line
                    line = line.strip()

                    # If dumping failures, track lines by rank and extract run_id
                    if dump_failures:
                        # Try to extract run_id if we haven't found it yet
                        if parsed_run_id is None:
                            run_id_match = re.match(RUN_ID_PATTERN, original_line)
                            if run_id_match:
                                parsed_run_id = run_id_match.group(1)
                                logging.info(f"Found run_id: {parsed_run_id}")

                        # Track lines by rank
                        rank_match = re.match(RANK_PATTERN, original_line)
                        if rank_match:
                            rank_num = int(rank_match.group(1))
                            rank_lines[rank_num].append(original_line)

                    # Fast pre-filter: skip lines that don't contain both "Cycle:" and "Event:"
                    if "Cycle:" not in line or "Event:" not in line:
                        continue

                    logging.debug(f"Parsing line: {line}")
                    match = re.search(LOG_PATTERN, original_line)
                    if match:
                        timestamp_str, cycle, event, node, utc_time = match.groups()
                        logging.debug(
                            f"Parsed: cycle={cycle}, event={event}, node={node}, utc_time={utc_time}"
                        )
                        try:
                            timestamp = parse_timestamp(utc_time)
                            cycle_num = int(cycle)

                            # Check for first failure_detected event in this cycle
                            if (
                                dump_failures
                                and event == "failure_detected"
                                and cycle_num not in failure_cycles_seen
                            ):
                                failure_cycles_seen.add(cycle_num)

                                # Extract rank number from the original line
                                rank_match = re.match(RANK_PATTERN, original_line)
                                if rank_match:
                                    failing_rank = int(rank_match.group(1))
                                    effective_run_id = parsed_run_id or generate_run_id(
                                        log_file_prefix
                                    )
                                    dump_failure_logs(
                                        failing_rank,
                                        cycle_num,
                                        rank_lines[failing_rank],
                                        effective_run_id,
                                    )

                            if node not in node_metrics:
                                node_metrics[node] = []
                            node_metrics[node].append(
                                {
                                    "cycle": cycle_num,
                                    "event": event,
                                    "timestamp": timestamp,
                                }
                            )
                        except ValueError as e:
                            logging.error(f"Skipping line due to timestamp parsing error: {line}")
                            continue
                    else:
                        logging.debug(f"Line does not match pattern, skipping: {line}")
                        if "Cycle:" in line and "Event:" in line:
                            logging.info(
                                f"Found cycle/event line but regex didn't match: {original_line.strip()}"
                            )
        except Exception as e:
            logging.error(f"Error reading file {log_file}: {e}")
            continue

    if not node_metrics:
        logging.warning("No valid metrics parsed from log files.")
    else:
        logging.info(f"Parsed metrics for {len(node_metrics)} nodes")
    return node_metrics


STAGES = [
    ("terminate", "failure_detected", "worker_terminated"),
    ("health_check", "worker_terminated", "rendezvous_started"),
    ("rendezvous", "rendezvous_started", "rendezvous_completed"),
    ("worker_launch", "rendezvous_completed", "worker_start_started"),
    ("worker_startup", "worker_start_started", "worker_start_completed"),
]


def calculate_cycle_metrics(events):
    """Calculate rendezvous and restart times for a list of events grouped by cycle."""
    rendezvous_times = []
    total_restart_times = []
    stage_times = {name: [] for name, _, _ in STAGES}

    cycles = {}
    for event in events:
        cycle = event["cycle"]
        if cycle == 0:
            logging.debug(f"Skipping Cycle 0 event: {event['event']}")
            continue
        if cycle not in cycles:
            cycles[cycle] = {}
        cycles[cycle][event["event"]] = event["timestamp"]

    for cycle, cycle_events in cycles.items():
        logging.debug(f"Processing cycle {cycle} with events: {list(cycle_events.keys())}")
        if "rendezvous_started" in cycle_events and "rendezvous_completed" in cycle_events:
            rendezvous_time = (
                cycle_events["rendezvous_completed"] - cycle_events["rendezvous_started"]
            ).total_seconds()
            rendezvous_times.append(rendezvous_time)
            logging.debug(f"Cycle {cycle} node_rendezvous_time: {rendezvous_time:.3f} seconds")

        if "failure_detected" in cycle_events and "worker_start_completed" in cycle_events:
            total_restart_time = (
                cycle_events["worker_start_completed"] - cycle_events["failure_detected"]
            ).total_seconds()
            total_restart_times.append(total_restart_time)
            logging.debug(
                f"Cycle {cycle} node_total_restart_time: {total_restart_time:.3f} seconds"
            )

        for name, start_evt, end_evt in STAGES:
            if start_evt in cycle_events and end_evt in cycle_events:
                # Exclude teardown for cycles that didn't restart (final cycle has no rendezvous_started)
                if name == "terminate" and "rendezvous_started" not in cycle_events:
                    logging.debug(f"Skipping terminate stage for cycle {cycle} — no restart")
                    continue
                dt = (cycle_events[end_evt] - cycle_events[start_evt]).total_seconds()
                stage_times[name].append(dt)

    return rendezvous_times, total_restart_times, stage_times


def calculate_node_metrics(node_metrics):
    """Calculate node-level rendezvous and restart times across all nodes."""
    node_rendezvous_times = []
    node_total_restart_times = []
    all_stage_times = {name: [] for name, _, _ in STAGES}

    for node, events in node_metrics.items():
        logging.debug(f"Calculating metrics for node: {node}")
        rendezvous_times, total_restart_times, stage_times = calculate_cycle_metrics(events)
        node_rendezvous_times.extend(rendezvous_times)
        node_total_restart_times.extend(total_restart_times)
        for name in all_stage_times:
            all_stage_times[name].extend(stage_times[name])

    node_teardown_times = all_stage_times["terminate"]
    return node_rendezvous_times, node_total_restart_times, node_teardown_times, all_stage_times


def calculate_job_metrics(node_metrics):
    """Calculate job-level rendezvous and restart times per cycle."""
    cycle_events = {}
    for node, events in node_metrics.items():
        for event in events:
            cycle = event["cycle"]
            if cycle == 0:
                continue
            if cycle not in cycle_events:
                cycle_events[cycle] = []
            cycle_events[cycle].append(event)

    job_rendezvous_times = []
    job_total_restart_times = []
    job_teardown_times = []

    for cycle, events in cycle_events.items():
        logging.debug(f"Calculating job metrics for cycle {cycle}")
        cycle_event_dict = {}
        for event in events:
            cycle_event_dict[event["event"]] = cycle_event_dict.get(event["event"], []) + [
                event["timestamp"]
            ]

        if "rendezvous_started" in cycle_event_dict and "rendezvous_completed" in cycle_event_dict:
            rendezvous_started = min(cycle_event_dict["rendezvous_started"])
            rendezvous_completed = max(cycle_event_dict["rendezvous_completed"])
            job_rendezvous_time = (rendezvous_completed - rendezvous_started).total_seconds()
            job_rendezvous_times.append(job_rendezvous_time)
            logging.debug(f"Cycle {cycle} job_rendezvous_time: {job_rendezvous_time:.3f} seconds")

        if "failure_detected" in cycle_event_dict and "worker_start_completed" in cycle_event_dict:
            failure_detected = min(cycle_event_dict["failure_detected"])
            worker_start_completed = max(cycle_event_dict["worker_start_completed"])
            job_total_restart_time = (worker_start_completed - failure_detected).total_seconds()
            job_total_restart_times.append(job_total_restart_time)
            logging.debug(
                f"Cycle {cycle} job_total_restart_time: {job_total_restart_time:.3f} seconds"
            )

        if (
            "failure_detected" in cycle_event_dict
            and "worker_terminated" in cycle_event_dict
            and "rendezvous_started" in cycle_event_dict
        ):
            failure_detected = min(cycle_event_dict["failure_detected"])
            worker_terminated = max(cycle_event_dict["worker_terminated"])
            job_teardown_time = (worker_terminated - failure_detected).total_seconds()
            job_teardown_times.append(job_teardown_time)
            logging.debug(f"Cycle {cycle} job_teardown_time: {job_teardown_time:.3f} seconds")

    return job_rendezvous_times, job_total_restart_times, job_teardown_times, len(cycle_events)


def print_percentiles(times, metric_name):
    """Print 50th and 95th percentiles for given times."""
    if times:
        print(f"{metric_name}:")
        print(f"  50th percentile: {np.percentile(times, 50):.3f} seconds")
        print(f"  95th percentile: {np.percentile(times, 95):.3f} seconds")
    else:
        logging.warning(f"No data available for {metric_name}")
        print(f"{metric_name}: No data available")


def main(log_file_prefix, dump_failures=False, output_json=False):
    """Main function to parse logs and calculate metrics."""
    node_metrics = parse_log_files(log_file_prefix, dump_failures)

    if not node_metrics:
        logging.error("No valid metrics found in logs. Exiting.")
        if output_json:
            print(
                json.dumps({"error": "no valid metrics found", "log_file_prefix": log_file_prefix})
            )
        else:
            print("No valid metrics found in logs.")
        return

    node_rendezvous_times, node_total_restart_times, node_teardown_times, stage_times = (
        calculate_node_metrics(node_metrics)
    )
    job_rendezvous_times, job_total_restart_times, job_teardown_times, restart_cycles = (
        calculate_job_metrics(node_metrics)
    )

    if output_json:

        def pct(times, p):
            return float(np.percentile(times, p)) if times else None

        stages_json = {}
        for name, _, _ in STAGES:
            d = stage_times[name]
            stages_json[name] = {
                "p50_s": pct(d, 50),
                "p95_s": pct(d, 95),
                "count": len(d),
            }

        print(
            json.dumps(
                {
                    "restart_cycles": restart_cycles,
                    "node_teardown_time_p50_s": pct(node_teardown_times, 50),
                    "node_teardown_time_p95_s": pct(node_teardown_times, 95),
                    "node_rendezvous_time_p50_s": pct(node_rendezvous_times, 50),
                    "node_rendezvous_time_p95_s": pct(node_rendezvous_times, 95),
                    "node_total_restart_time_p50_s": pct(node_total_restart_times, 50),
                    "node_total_restart_time_p95_s": pct(node_total_restart_times, 95),
                    "job_teardown_time_p50_s": pct(job_teardown_times, 50),
                    "job_teardown_time_p95_s": pct(job_teardown_times, 95),
                    "job_rendezvous_time_p50_s": pct(job_rendezvous_times, 50),
                    "job_rendezvous_time_p95_s": pct(job_rendezvous_times, 95),
                    "job_total_restart_time_p50_s": pct(job_total_restart_times, 50),
                    "job_total_restart_time_p95_s": pct(job_total_restart_times, 95),
                    "stages": stages_json,
                    "log_file_prefix": log_file_prefix,
                },
                indent=2,
            )
        )
        return

    print("Node Metrics (across all restart cycles and nodes)")
    print(f"Restart Cycles: {restart_cycles}")
    print_percentiles(node_teardown_times, "node_teardown_time")
    print_percentiles(node_rendezvous_times, "node_rendezvous_time")
    print_percentiles(node_total_restart_times, "node_total_restart_time")

    print("\nJob Metrics (per cycle)")
    print_percentiles(job_teardown_times, "job_teardown_time")
    print_percentiles(job_rendezvous_times, "job_rendezvous_time")
    print_percentiles(job_total_restart_times, "job_total_restart_time")

    print("\nStage Breakdown (per node per cycle, excluding cycle 0)")
    print(f"  {'Stage':<18} {'count':>6}  {'p50':>7}  {'p95':>7}")
    print(f"  {'-'*18} {'-'*6}  {'-'*7}  {'-'*7}")
    for name, _, _ in STAGES:
        d = stage_times[name]
        if d:
            print(
                f"  {name:<18} {len(d):>6}  {np.percentile(d,50):>6.2f}s  {np.percentile(d,95):>6.2f}s"
            )
        else:
            print(f"  {name:<18} {'no data':>15}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse restart metrics from log files")
    parser.add_argument("log_file_prefix", help="Prefix of log files to parse")
    parser.add_argument(
        "--dump-failures",
        action="store_true",
        help="Dump failure logs to /tmp/nvrx_<run_id>_brief.log",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output metrics as JSON instead of human-readable text",
    )

    args = parser.parse_args()
    main(args.log_file_prefix, args.dump_failures, args.output_json)
