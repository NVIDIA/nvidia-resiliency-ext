# Scheduler Exclusion: FT Integration Specification

## Goal

Before forming a rendezvous round, NVRx must not reuse a Slurm array task whose
current allocation generation appears in the Scheduler Exclusion decision. The
FT path consumes the shared JSONL artifact; it does not query Slurm or call the
service at the restart boundary.

```text
nvrx-scheduler-exclusion-service
  -> atomically replaces scheduler_exclusion.<job_id>.jsonl
  -> first record: compact excluded task IDs
  -> later records: detailed task/node decisions and observations

array-task Node0
  -> SegmentHealthCheck runs the scheduler decision predicate
  -> predicate scans first JSONL line for its quoted task ID
  -> raises UnhealthyNodeException when excluded
  -> existing exception path marks its replacement group unhealthy
       |
       `-> Slurm terminates that array task; a spare task may join
```

The existing rendezvous host already ignores unhealthy replacement groups.
Scheduler Exclusion adds no new host-side selection logic or process.

## FT Boundary

The FT side receives an optional absolute decision directory through launcher
configuration. When configured, `SegmentHealthCheck` runs the scheduler
decision predicate through the existing pre-rendezvous health-check flow. The
predicate derives the job-specific artifact path and reads only the compact
task-ID record. It does not manage the service or make HTTP or Slurm calls.
Missing, stale, or invalid decisions fail open.

The [usage guide](../source/fault_tolerance/usage_guide.rst) owns CLI and YAML
enablement. The [integration guide](../source/fault_tolerance/integration/scheduler_exclusion_service.rst)
owns deployment, lifecycle, service configuration, decision format, and HTTP
behavior.

## Implementation

### 1. Segment health check

`SegmentHealthCheck` in `shared_utils/health_check.py` composes task- or
segment-level health predicates using the same callable boolean contract as
the existing health checks. The scheduler predicate in
`fault_tolerance/scheduler_exclusion.py` provides:

- shared Slurm metadata helpers for `SLURM_ARRAY_JOB_ID`,
  `SLURM_ARRAY_TASK_ID`, and strict `SLURM_RESTART_COUNT` parsing;
- local Node0 detection from `SLURM_NODEID`, with `SLURM_PROCID` as the
  one-launcher-per-node fallback;
- component-owned path derivation from the configured scheduler-exclusion
  directory;
- a bounded read of only the first JSONL line;
- a compact first-line contract containing only quoted task IDs;
- fixed byte-token matching equivalent to `grep -Fq '"<task_id>"'`;
- a 30-minute freshness bound derived from the opened file's modification time;
- the standard callable health-check result: healthy for a clean or fail-open
  decision, unhealthy for a matching task ID in a fresh artifact.

The filename supplies the array job ID. Detailed records after the first line
retain schema versions, task generations, per-entry expiry, node state, and
reasons for diagnostics. Malformed data, I/O errors, stale files, and absent
files log a concise warning and fail open.

Only Node0 of each array task reads the artifact. `SLURM_NODEID=0` is primary;
`SLURM_PROCID=0` is the supported fallback for one launcher per node.
Each check emits its outcome and monotonic elapsed time at INFO level. This
measures the synchronous FT overhead independently of total rendezvous time.

The segment health check consumes only the compact task-ID record. Node-scoped
consumption for regular jobs is a separate integration.

### 2. Generation-aware replacement-group identity

Represent a Slurm replacement group internally as an opaque generation token
derived from `(SLURM_ARRAY_TASK_ID, SLURM_RESTART_COUNT)`. All launchers in one
array task publish the same token. The compact external decision contains only
task IDs because the supported deployment uses `--no-requeue`; detailed records
retain restart counts for diagnostics.

Keep `unhealthy_replacement_groups_<round>` as the exclusion key. Existing
node-local health-check failures and `SegmentHealthCheck` both reach the
existing `UnhealthyNodeException` path. That path adds the current generation
token through the existing compare-and-set update.

### 3. Existing group-exclusion path

In the participant-side `pre_join_hook`:

1. Non-leader launchers perform no filesystem work and may publish their join.
2. Node0 reads and validates the decision.
3. For a clean or fail-open result, Node0 joins normally.
4. For an exclusion, Node0 raises `UnhealthyNodeException`. The existing catch
   path marks its generation in `unhealthy_replacement_groups_<round>` before
   propagating the exception.

With enough eligible spares, the round closes normally. Without enough nodes,
the round does not start and follows the existing rendezvous timeout behavior;
NVRx never weakens scheduler exclusion to manufacture quorum.

This first implementation relies on the normal full-task replacement-group
shape: the group cannot become complete without Node0. If
`replacement_group_size < SLURM_NNODES`, enough non-leader peers could become
eligible before Node0 publishes the unhealthy result. That configuration is
not protected by this implementation because there is no current deployment
that needs it. Supporting it later requires a round-scoped Node0-checked gate;
the service and decision artifact do not need to change.

### 4. Preserve existing lifecycle behavior

`UnhealthyNodeException` already exits `ft_launcher` non-zero without
permanently shutting down rendezvous. For a non-host array task, Slurm terminates
the task and a spare task can replace it.

Scheduler Exclusion does not change rendezvous-host recovery or implement
TCPStore host migration. Deployments use the existing lifecycle described in the
[singleton job-array guide](../source/fault_tolerance/examples/singleton_deployment.rst).

## Implementation Map

| Area | Responsibility |
| --- | --- |
| `shared_utils/health_check.py` | Provide the `SegmentHealthCheck` composition and callable health-check contract. |
| `fault_tolerance/scheduler_exclusion.py` | Evaluate scheduler decisions, including identity, bounded reads, validation, expiry, and fail-open behavior. |
| `fault_tolerance/config.py` and launcher plumbing | Validate `--ft-scheduler-exclusion-dir` as an absolute path and pass it into rendezvous settings. |
| `fault_tolerance/ft_rendezvous_barrier.py` | Compose `SegmentHealthCheck` with the existing health-check flow and unhealthy-group exception path without changing host-side selection. |
| `tests/shared_utils/` and `tests/fault_tolerance/unit/` | Cover health-check composition, parsing, environment handling, exception propagation, and task generations. |
| FT usage and integration guides | Own user configuration and deployment behavior, respectively. |

## Verification

### Unit tests

- a matching task ID in a fresh artifact is excluded;
- an omitted launcher option disables consumption and a relative directory is
  rejected;
- quoted task IDs do not produce substring matches and stale files are ignored;
- missing, malformed, oversized, or unsupported artifacts fail open;
- only array-task Node0 reads the file;
- an excluded Node0 raises `UnhealthyNodeException` and the existing exception
  path marks its generation unhealthy;
- existing local health-check exclusion behavior remains intact.

Environment-specific E2E scenarios, commands, and artifacts are maintained with
the deployment validation harness rather than duplicated in this specification.

## Invariants

- Only array-task Node0 reads the decision artifact.
- A matching quoted task ID in a fresh job-specific artifact prevents that task
  from joining rendezvous.
- The supported deployment uses `--no-requeue`; task IDs are not reused.
- Missing, expired, malformed, or unreadable decisions fail open.
- Scheduler exclusions are never weakened to manufacture quorum.
- The FT path performs no Slurm or service I/O at the rendezvous boundary.
- Under the supported full-task replacement-group shape, an excluded generation
  cannot receive an active rank.
