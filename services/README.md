# NVRX Services

Service entry points and deployment assets for NVRX.

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **nvrx-attrsvc** | FastAPI server for LLM-based log analysis | [attrsvc/README.md](attrsvc/README.md) |
| **nvrx-smonsvc** | SLURM job monitor for automatic log submission | [smonsvc/README.md](smonsvc/README.md) |
| **nvrx-scheduler-exclusion-service** | Host-side Slurm monitor that publishes shared task-exclusion decisions | [scheduler_exclusion_service.rst](../docs/source/fault_tolerance/integration/scheduler_exclusion_service.rst) |

See the linked documentation for each service's contract. Scheduler Exclusion
build and deployment assets are in [`scheduler_exclusions/`](scheduler_exclusions/).

## Combined Attribution Deployment

Run attrsvc and smonsvc together on SLURM:

```bash
NVRX_ATTRSVC_ALLOWED_ROOT=/path/to/logs \
  sbatch --account=myaccount services/scripts/nvrx_services.sbatch
```

This starts `nvrx-attrsvc` and `nvrx-smonsvc` in a single job with health monitoring.

For individual attrsvc or smonsvc deployment, see its `deploy/` directory.

## Container Image

Build an enroot squash image containing the attribution services:

```bash
./services/scripts/build_enroot_image.sh
```

See [scripts/build_enroot_image.sh](scripts/build_enroot_image.sh) for usage with Slurm + pyxis.

## Monitoring

Periodically snapshot service endpoints for debugging:

```bash
# Attribution services
./scripts/snapshot_services.sh hostname

# Individual services (in respective directories)
./attrsvc/deploy/snapshot_attrsvc.sh hostname 8000
./smonsvc/deploy/snapshot_smonsvc.sh hostname 8100
```

Configure via environment: `SNAPSHOT_INTERVAL`, `SNAPSHOT_OUTPUT_DIR`.

## Files

| Path | Description |
|------|-------------|
| `attrsvc/` | Attribution service deployment docs and assets |
| `smonsvc/` | SLURM monitor deployment docs and assets |
| `scheduler_exclusions/` | Standalone Scheduler Exclusion build and deployment assets |
| `scripts/` | Shell scripts ([README](scripts/README.md)) |

## Library Layer

The core analysis functionality in `nvidia_resiliency_ext.attribution` can be used without HTTP.
See [attrsvc/README.md](attrsvc/README.md#python-api) for the Python API.
