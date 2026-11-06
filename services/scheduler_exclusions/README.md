# Scheduler Exclusion Publisher

An independent Python process that publishes scheduler exclusions for all running
array tasks (segments) in one Slurm array job, identified by `SLURM_ARRAY_JOB_ID`.
It is not one publisher per segment, and it does not automatically discover other
separately submitted array jobs belonging to the same workload.

It ports the deployed `scheduler_segment_health.sh` producer from PR #399,
including the later immediate-refresh support from PR #424.

The Bash producer remains available and unchanged. This is an opt-in replacement,
not an additional publisher to run alongside it. The launcher consumer and
history format are unchanged. The reusable component lives under
`nvidia_resiliency_ext.attribution.scheduler_exclusions`. There is no HTTP API,
log-analysis dependency, or node-state cache.

## Launcher-Managed Publishing

Add these options to the existing `ft_launcher` invocation on every node:

```bash
ft_launcher \
  --ft-segment-health-check-dir /shared/nvrx/123/segment_health \
  --ft-enable-segment-health-check-publisher \
  <other-options> train.py
```

These are presence-only flags: do not append `true` or `false`. Publishing is
disabled by default. `--ft-disable-segment-health-check-publisher` explicitly
disables it without disabling consumption of the shared decision artifacts.
The flags are mutually exclusive; an explicit flag overrides a parser-provided
default. Programmatic callers can set
`FaultToleranceConfig(enable_segment_health_check_publisher=True,
segment_health_check_dir="/shared/nvrx/123/segment_health")`.

Only the launcher identified as the rendezvous/TCP-store host starts a publisher,
using the same ownership decision as launcher-managed `attrsvc`. Its array task
and Slurm process indices need not be zero. It still requires
`SLURM_ARRAY_JOB_ID`, `SLURM_JOB_PARTITION`, and Slurm read access to monitor the
array. No separate publisher-startup code is needed in `sbatch`. The launcher
passes the configured directory to its child; there is no need to set
`NVRX_SEGMENT_HEALTH_CHECK_DIR` for this mode.

The publisher is a separate process in its own process group. All Slurm calls
and shared-directory I/O happen there. The launcher starts the child directly
without waiting for readiness or a first publication, and retains it across
training restarts. Spawn failures are logged without terminating training.
There is no supervisor thread or automatic restart if the publisher exits.
Process creation itself runs on the launcher thread; scheduler polling does not.
Launcher shutdown requests termination, escalates to group kill after two
seconds, and waits at most another half-second to reap the child. The child also
watches its parent and exits if the launcher dies.

The manager owns process startup and cleanup only. Restart supervision is a
separate responsibility, deferred to a future shared facility rather than
implemented specifically for this publisher.

This is single-owner startup, not leader election: if the owning launcher exits
or its task is excluded, no other task takes over. If the rendezvous store is
external and no participating launcher is its host, no launcher starts a
publisher. Use a standalone publisher for that deployment, or when publishing
must survive the owning launcher's lifetime.
There is no initial-publication barrier before the first rendezvous. Existing
markers remain when the publisher stops, just as with the Bash producer.

Run one publisher per Slurm array job, monitoring all its running array tasks.
Stop the previous publisher before enabling this mode. Do not run multiple
publisher modes (Bash, standalone Python, or launcher-managed Python) against
the same decision directory for that array job.

## Standalone Publishing

Run one publisher for the entire Slurm array job, under a batch supervisor or
another independent process owner. The process needs the array job ID and
partition, Slurm read access, and the shared decision directory:

```bash
export SLURM_ARRAY_JOB_ID=123
export SLURM_JOB_PARTITION=batch
export NVRX_SEGMENT_HEALTH_CHECK_DIR=/shared/nvrx/123/segment_health

# Immediate poll, then periodic polling until SIGTERM or SIGINT.
nvrx-scheduler-exclusions

# Alternatively, validate one poll and exit.
nvrx-scheduler-exclusions --once
```

`python -m nvidia_resiliency_ext.services.scheduler_exclusions` is equivalent to
the console command. The component uses only Python 3.10+ standard-library modules.
It does not import log analysis, PyTorch, FastAPI, or the fault-tolerance launcher.

In standalone mode the supervising deployment owns startup, restart, and
shutdown. `attrsvc` does not start this publisher.

For a container deployment, provide `sinfo`, `squeue`, and `scontrol` on `PATH`,
their matching plugins/libraries and cluster configuration, scheduler
authentication access, user resolution, and connectivity to the scheduler.
These can be supplied through cluster-specific bind mounts. No Slurm credentials
or internal cluster paths are bundled into NVRx.

Every consuming launcher must already receive:

```bash
ft_launcher --ft-segment-health-check-dir "$NVRX_SEGMENT_HEALTH_CHECK_DIR" ...
```

The publisher supports arrays, matching #399. Although the consumer also supports
regular jobs, this publisher does not add regular-job allocation discovery.

## Component Ownership

`attribution/scheduler_exclusions/` owns scheduler observations, publication,
the polling/signal runtime, and process lifecycle management. The service entry
point only adapts CLI/environment configuration. The launcher selects the
rendezvous-store owner and passes explicit job configuration to the manager;
the manager has no rendezvous-host or Slurm task-rank selection logic.

Other process owners can use the same API without a launcher:

```python
from pathlib import Path

from nvidia_resiliency_ext.attribution.scheduler_exclusions import (
    PublisherConfig,
    SegmentHealthPublisherManager,
)

manager = SegmentHealthPublisherManager(
    PublisherConfig(job_id="123", partition="batch", directory=Path("/shared/nvrx/123/segment_health"))
)
try:
    manager.start()
    # The owning application's lifetime runs here.
finally:
    manager.stop()
```

Explicit job configuration overrides the process owner's inherited Slurm job
identity. The child still inherits its Slurm clients, authentication, and polling
environment. A future attrsvc host would need job registration, Slurm access,
shared-directory access, and single-owner startup/shutdown integration. This PR
does not add that hosting or automatic restart supervision.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `SLURM_ARRAY_JOB_ID` | Required | Numeric array parent ID |
| `SLURM_JOB_PARTITION` | Required | Partition queried for unavailable nodes |
| `NVRX_SEGMENT_HEALTH_CHECK_DIR` | Required | Absolute shared decision directory |
| `SEGMENT_HEALTH_POLL_INTERVAL` | `300` | Seconds between completed polls; no jitter |
| `NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS` | `1800` | Clear exclusions after this long without a complete scheduler snapshot |
| `NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT` | `30` | Timeout in seconds for each `sinfo` / `squeue` call |

Hostlist expansion (`scontrol show hostnames`) has a fixed five-second timeout.
`--once` exits with 0 for a successful poll, 1 for a poll/publication failure,
and 2 for invalid configuration. Periodic mode continues after poll failures.

## Preserved Behavior

1. Read existing per-task markers before every poll, including after process restart.
2. Query `drain,down,fail,no_respond` nodes in the configured partition.
3. Skip allocation discovery when there are no unavailable nodes or existing markers.
4. Otherwise query running array tasks, expand each hostlist, and intersect its
   nodes with the unavailable set. Reconcile only after the entire snapshot succeeds.
5. Publish or update affected tasks. Clear a running task when no unavailable
   nodes remain; there is no additional recovery-verification query.
6. Retire a task only after it is absent from two consecutive complete task maps.
   Failed polls do not count as complete maps. An empty successful task map is valid.
7. Preserve decisions on incomplete/failed reads. Once the stale deadline passes,
   a running publisher clears stale exclusions and records `stale_decision_expired`.
8. Continue reconciling other tasks if one artifact operation fails. History is
   best-effort and unchanged decisions do not produce duplicate events.

The missing-task counter and stale deadline are process-local, as in Bash.
Restarting the publisher resets them, but does not discard existing markers.
Stopping or losing the publisher leaves markers in place: expiration is performed
by the polling process, not by the consumer or a separate timer.

## Artifacts And Refresh

```text
segment_health_check.<array_job_id>.<task_id>
segment_health_check.<array_job_id>.<task_id>.inactive
segment_health_check_history.<array_job_id>.log
segment_health_producer.<array_job_id>.pid
```

A nonempty current marker excludes its task. Missing or empty markers allow the
task to proceed. The marker's sorted node CSV is diagnostic; the consumer checks
file size only. Updating a nonempty marker does not truncate it before writing.
The `.inactive` file retains the last observation outside the consumer namespace.

History uses the existing `excluded`, `updated`, and `cleared` JSONL records with
`task_id`, `observed_at` in UTC, and `nodes` or `reason` where appropriate. Existing
collection and dashboards need no schema change.

Periodic mode writes `hostname PID` to the PID file. On the publisher host, send
`kill -USR1 <PID>` to request an immediate poll. A request during a poll schedules
another poll immediately afterward. `SIGTERM` / `SIGINT` stop the process and
remove its own PID file without clearing the exclusions.

## Verification

```bash
PYTHONPATH=src python -m pytest services/tests/test_scheduler_exclusions.py -q
```

The differential tests replay identical snapshots through the original Bash
producer and Python publisher, comparing task markers, history, query order,
timeouts passed to queries, and return status. Separate tests exercise artifact
failures, configuration, command handling, and the standalone signal lifecycle.
They use mocked scheduler responses and do not query a live cluster.

Launcher ownership, presence-only flags, config precedence, and failure isolation
are covered by `tests/fault_tolerance/unit/test_segment_health_publisher_manager.py`
and `tests/fault_tolerance/unit/test_launcher.py`.
Explicit configuration and dependency isolation are covered by
`tests/attribution/unit/test_scheduler_exclusion_component.py`.
