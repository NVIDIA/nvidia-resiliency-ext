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

`deploy/run_service.sh` stays in the foreground. Use
`deploy/slurm_array.sbatch` as the lifecycle reference: it runs
`deploy/run_service.sh` in the foreground beneath a batch-owned supervisor,
checks readiness, restarts with capped backoff, and terminates and waits for the
supervisor during batch cleanup. Adapt its Slurm allocation and workload
arguments to the target deployment.

See the
[integration guide](../../docs/source/fault_tolerance/integration/scheduler_exclusion_service.rst)
for deployment requirements, configuration, decision format, HTTP API, and
runtime behavior.
