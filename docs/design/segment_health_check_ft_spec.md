# Segment Health Check: FT Consumer Design

## Purpose

NVRx consumes a shared segment health decision before joining rendezvous.
The restart path performs filesystem I/O only; it does not query Slurm or call
an external service. The
[integration guide](../source/fault_tolerance/integration/segment_health_check.rst)
owns the user-facing configuration, artifact contract, and supported launch
shape.

## Internal Flow

```text
FtRendezvousBarrierHandler initialization
  -> get_segment_health_check()
       -> ineligible launcher: no check installed
       `-> eligible allocation-unit process 0: SegmentHealthCheck(directory, job_id, task_id)

pre_join_hook
  -> ensure_node_is_healthy()
       -> SegmentHealthCheck._perform_health_check()
       -> existing local node checks
       `-> _run_health_check() converts failure to UnhealthyNodeException
            -> array path marks replacement group unhealthy
            `-> regular-job path exits the workload step
```

### Installation Guard

`get_segment_health_check()` installs the check only on the launcher with
`SLURM_PROCID=0`. Arrays use `SLURM_ARRAY_JOB_ID` and `SLURM_ARRAY_TASK_ID`;
regular jobs use `SLURM_JOB_ID` as both the artifact scope and exclusion token.
Missing job identity leaves the check uninstalled; other launchers return
without logging. Successful installation emits one INFO record.

### Decision Evaluation

The concrete `SegmentHealthCheck` in
`fault_tolerance/segment_health_check.py` owns the configured directory and
Slurm job/task IDs. Its `_perform_health_check()`:

1. derives `segment_health_check.<job_id>.<task_id>` from the configured
   directory;
2. performs one metadata check; and
3. returns unhealthy only when the path is a non-empty regular file.

The file content is producer-owned diagnostic context and is not read by the
consumer. A missing or zero-byte file is healthy and quiet. An unreadable path
or unexpected file type logs a warning and fails open.

### Rendezvous Handoff

A matching allocation ID returns unhealthy through `_run_health_check()`, which
raises `UnhealthyNodeException`. For arrays, the existing pre-join exception
path uses `SLURM_ARRAY_TASK_ID` as the replacement-group token and adds it to
`unhealthy_replacement_groups_<round>`. A regular job has no replacement-group
token; process 0 exits nonzero and `srun --kill-on-bad-exit=1` terminates its
workload step. Segment Health Check adds no host-side selection path.

## Implementation Map

| Area | Responsibility |
| --- | --- |
| `fault_tolerance/segment_health_check.py` | Identify eligible process-0 launchers and implement `SegmentHealthCheck`. |
| `fault_tolerance/config.py` and launcher plumbing | Validate and pass the decision directory into rendezvous settings. |
| `fault_tolerance/ft_rendezvous_barrier.py` | Install the check and reuse the existing health-check exception path. |
| `tests/fault_tolerance/unit/` | Cover installation, per-task file state, failure handling, and rendezvous propagation. |

## Verification

Environment-specific E2E scenarios, commands, and result analysis live with
the deployment validation harness rather than this product design.
