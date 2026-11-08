# NVRx Scheduler Exclusion Service

The NVRx Scheduler Exclusion Service is the host-side component of the
Scheduler Exclusion feature. For an array job, the first allocated node of the
lowest array task runs it outside the workload environment. Build it as a
standalone Python zipapp so that host does not need an NVRx installation or
virtual environment.

## Build

From the repository root:

```bash
python3 services/scheduler_exclusions/build_zipapp.py
dist/nvrx-scheduler-exclusion-service.pyz --help
```

The builder packages only the Scheduler Exclusion modules and validates the
artifact. It is an explicit build step: building the wheel does not create or
publish the `.pyz`, and generated `dist/` output is not committed. The
deployment owner builds it from the same NVRx revision as the workload, stores
it at an immutable version or commit-based path, and stages it on storage
visible to the service host.

## Run

The artifact is directly executable with Python 3.10 or newer:

```bash
/shared/nvrx/bin/nvrx-scheduler-exclusion-service.pyz \
  --output-dir /shared/nvrx/scheduler-exclusions
```

It requires the Slurm CLI and configuration from the host environment. It has
no third-party Python runtime dependencies.

`deploy/run_service.sh` is an optional foreground launcher. Set
`NVRX_SCHEDULER_EXCLUSION_ARTIFACT` when the artifact is outside the repository:

```bash
NVRX_SCHEDULER_EXCLUSION_ARTIFACT=/shared/nvrx/bin/nvrx-scheduler-exclusion-service.pyz \
  services/scheduler_exclusions/deploy/run_service.sh \
    --output-dir /shared/nvrx/scheduler-exclusions
```

`deploy/run_service.sh` stays in the foreground. The deployment owner provides
process supervision, restart policy, log redirection, and shutdown.

## Producer Contract

The service monitors `SLURM_ARRAY_JOB_ID`, falling back to `SLURM_JOB_ID`, and
refreshes scheduler state in the background. It atomically publishes
`segment_health_check.<job-id>.state` under `--output-dir`. The compact first
line contains excluded array-task IDs; detailed task, node, and observation
records follow it:

```text
["7"]
{"type":"decision","schema_version":1,"job_id":"12345","generated_at":"2026-08-04T19:00:00Z","scope":"array_task","excluded_array_tasks":[{"task_id":"7","restart_count":0,"valid_until":"2026-08-04T19:30:00Z"}]}
{"type":"decision","schema_version":1,"job_id":"12345","generated_at":"2026-08-04T19:00:00Z","scope":"node","excluded_nodes":[{"node":"node-a","valid_until":"2026-08-04T19:30:00Z"}]}
```

The HTTP interface exposes `GET /healthz`, `GET /stats`, and
`GET /scheduler-exclusions`. `POST /refresh` queues a nonblocking refresh.
Run the artifact with `--help` for configuration defaults and Slurm overrides.

## FT Consumer Integration

`deploy/slurm_array.sbatch` is the combined lifecycle reference. It supervises
the producer in the lowest array task, passes the shared output directory to
`ft_launcher`, and terminates the supervisor during batch cleanup. The FT
consumer remains independent of this service and depends only on the artifact
contract above.
