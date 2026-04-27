---
name: nvrx-validate
description: Validate InJob restart correctness from collected artifacts. Checks fault detection, restart timing, throughput, and membership. Writes report.md with PASS/FAIL per check. Use after a job completes and artifacts are collected.
argument-hint: <artifacts_dir> [--thresholds <thresholds.yaml>]
allowed-tools: [Read, Write, Bash, Glob, Grep]
---

Validate InJob restart correctness from collected artifacts.

Checks three things: (A) failure detection, (B) restart timing, (C) training throughput + membership.

## Usage
```
/nvrx-validate <artifacts_dir> [--thresholds <thresholds.yaml>]
```

- `<artifacts_dir>`: directory written by `/nvrx-watch` (contains run_result.json, logs, cycle_infos/)
- `--thresholds`: optional YAML with numeric thresholds (see schema below)

If no artifacts dir given, ask the user.

## Thresholds YAML schema

```yaml
max_job_teardown_time_p95_seconds: 30.0
max_job_rendezvous_time_p95_seconds: 75.0
max_job_total_restart_time_p95_seconds: 75.0
max_node_teardown_time_p95_seconds: 30.0
max_node_rendezvous_time_p95_seconds: 40.0
max_node_total_restart_time_p95_seconds: 75.0
min_cycles_excluding_warmup: 3
warmup_iters_per_cycle: 10        # skip first N iters per cycle for throughput
reference_throughput_tflops: null # set to baseline value to enable throughput check
throughput_tolerance_pct: 5.0
```

If `--thresholds` is not provided, use these defaults. Skip the throughput check if `reference_throughput_tflops` is null.

## Your task

Read `<artifacts_dir>/run_result.json` for job_id, final_state, paths.

Run all checks below using Bash commands to grep/parse artifacts. Interpret results and write a final report.

---

## Check A: Failure detection

### Hardware track (NVRX_INJECT_GPU_FAILURE)

Look for the injection spec in artifacts. Check the sbatch script or run_result meta:
```bash
grep "NVRX_INJECT_GPU_FAILURE" <artifacts_dir>/**/*.sh <artifacts_dir>/run_result.json 2>/dev/null | head -5
```

If a spec is found (e.g. `"2:33,4:33,6:33"`), parse the expected `(cycle, infra_rank)` pairs.

For each expected fault:
```bash
# Find log file for that cycle from cycle_info
python tools/fault_tolerance/cycle_info_reader.py <artifacts_dir> --job-id <job_id> --show-log-files

# Check injection was triggered
grep "INJECTING GPU HEALTH CHECK FAILURE" <cycle_log_file>
```

Also check `standby_nodes` non-empty in fault cycles (from cycle_info_reader output).

### Software track (auto-detect from inject_fault.py logs)

```bash
# Find injection discovery lines (logged at INFO level on injected rank)
grep -rh "num_ranks_to_inject=" <artifacts_dir>/logs/ 2>/dev/null | head -20

# Find fault-fired lines (logged at CRITICAL when fault triggers)
grep -rh "asynchronously raising\|locking GIL\|raising GPU error on\|raising workload exception\|raising segmentation fault\|aborting\|GPU sleep on" <artifacts_dir>/logs/ 2>/dev/null | head -20
```

For each discovered injection event:
- Extract: cycle number (from surrounding context or profiling events), fault type, affected ranks
- Verify a `failure_detected` profiling event exists in the same cycle

**Interpret:** for each fault, report whether it (1) was discovered, (2) actually fired, (3) was detected by FT.

---

## Check B: Restart timing

```bash
python tools/fault_tolerance/parse_restart_metrics.py <log_prefix> --json
```

Where `<log_prefix>` is the common prefix of log files in `<artifacts_dir>/logs/` (everything before `.log`, `.log.0`, etc.).

Parse the JSON output. Compare:
- `job_teardown_time_p95` vs `max_job_teardown_time_p95_seconds`
- `job_rendezvous_time_p95` vs `max_job_rendezvous_time_p95_seconds`
- `job_total_restart_time_p95` vs `max_job_total_restart_time_p95_seconds`
- `node_teardown_time_p95` vs `max_node_teardown_time_p95_seconds`
- `node_rendezvous_time_p95` vs `max_node_rendezvous_time_p95_seconds`
- `node_total_restart_time_p95` vs `max_node_total_restart_time_p95_seconds`
- `cycle_count_excluding_warmup` vs `min_cycles_excluding_warmup`

Also read `stages` from the JSON for the stage breakdown table (see report template).

---

## Check C: Training throughput

```bash
grep -rh "throughput per GPU (TFLOP/s)" <artifacts_dir>/logs/ 2>/dev/null | tail -500
```

Parse TFLOP/s values. Group by cycle (use surrounding `Cycle:` context if available, or assume all lines from a cycle log file belong to that cycle). Skip first `warmup_iters_per_cycle` iterations per cycle.

Compute median TFLOP/s across all stable-cycle iterations.

If `reference_throughput_tflops` is set:
- PASS if `median >= reference * (1 - tolerance/100)`
- FAIL otherwise, report the delta

---

## Check D: Membership + checkpoint

```bash
python tools/fault_tolerance/cycle_info_reader.py <artifacts_dir> --job-id <job_id>
```

From the output:
- Verify no gaps in cycle sequence
- Verify all cycles have `cycle_end_time` set
- For each cycle: verify `active_nodes` is a subset of the union of the original active set and all known standby nodes (no unknown nodes appeared)
- Note: node swaps can occur even with SW-only faults when the rendezvous host constraint policy is ANY — do NOT assume SW fault → same nodes. Node membership changes are not a reliable indicator of HW vs SW fault type.

Check checkpoint monotonicity:
```bash
cat <artifacts_dir>/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null || \
find <artifacts_dir> -name "latest_checkpointed_iteration.txt" | head -1 | xargs cat
```

The iteration number should be higher at end than at start (training made progress).

---

## Final job state

Report PASS/FAIL for overall job completion:
- COMPLETED → PASS
- FAILED/CANCELLED/TIMEOUT → FAIL (but continue running all checks on partial data)

---

## Write report

Write `~/.cache/nvrx_reports/<cluster>/<job_id>.md` (read `cluster` and `job_id` from `run_result.json`; create dir if needed):

```markdown
# InJob Validation Report — Job <job_id>

**Cluster:** <cluster>  **Completed:** <timestamp>  **Final state:** <state>

## Summary

| Check | Status | Detail |
|-------|--------|--------|
| Job completed | PASS/FAIL | final_state=COMPLETED |
| Failure detection (HW) | PASS/FAIL/SKIP | N faults expected, N confirmed |
| Failure detection (SW) | PASS/FAIL/SKIP | N injections found, N fired, N detected by FT |
| Min cycle count | PASS/FAIL | observed N >= threshold M |
| Cycle sequence integrity | PASS/FAIL | no gaps / N gaps found |
| job_teardown_time p95 | PASS/FAIL | Xs <= threshold Ys |
| job_rendezvous_time p95 | PASS/FAIL | Xs <= threshold Ys |
| job_total_restart_time p95 | PASS/FAIL | Xs <= threshold Ys |
| node_teardown_time p95 | PASS/FAIL | Xs <= threshold Ys |
| node_rendezvous_time p95 | PASS/FAIL | Xs <= threshold Ys |
| node_total_restart_time p95 | PASS/FAIL | Xs <= threshold Ys |
| Throughput | PASS/FAIL/SKIP | median N TFLOP/s vs reference M (delta D%) |
| Membership (active_nodes) | PASS/FAIL | <detail> |
| Checkpoint progress | PASS/FAIL | iter start→end |

## Timing Metrics

| Metric | p50 | p95 | Threshold |
|--------|-----|-----|-----------|
| job_teardown_time (s) | ... | ... | 30.0 |
| job_rendezvous_time (s) | ... | ... | 75.0 |
| job_total_restart_time (s) | ... | ... | 75.0 |
| node_teardown_time (s) | ... | ... | 30.0 |
| node_rendezvous_time (s) | ... | ... | 40.0 |
| node_total_restart_time (s) | ... | ... | 75.0 |

Cycles analyzed: N (excluding cycle 0 warmup)

## Stage Breakdown (per node per cycle)

| Stage | p50 | p95 |
|-------|-----|-----|
| terminate (failure_detected → worker_terminated) | ...s | ...s |
| health_check (worker_terminated → rendezvous_started) | ...s | ...s |
| rendezvous (rendezvous_started → rendezvous_completed) | ...s | ...s |
| worker_launch (rendezvous_completed → worker_start_started) | ...s | ...s |
| worker_startup (worker_start_started → worker_start_completed) | ...s | ...s |

## Fault Detection Details

<per-fault table>
```

Print the summary table to terminal. Tell the user where the full report is saved.

After writing the report, delete the artifacts directory:
```bash
rm -rf <artifacts_dir>
```
Print: `Artifacts deleted: <artifacts_dir>`
