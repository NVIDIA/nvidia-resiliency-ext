# NVRX Attribution

Automated log analysis and failure attribution for distributed training jobs.

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **nvrx-attrsvc** | FastAPI server for LLM-based log analysis | [attrsvc/README.md](attrsvc/README.md) |
| **nvrx-smonsvc** | SLURM job monitor for automatic log submission | [smonsvc/README.md](smonsvc/README.md) |

See component READMEs for quick start, configuration, and API details.

## Combined Deployment

Run both services together on SLURM:

```bash
NVRX_ATTRSVC_ALLOWED_ROOT=/path/to/logs \
  sbatch --account=myaccount services/scripts/nvrx_services.sbatch
```

This starts `nvrx-attrsvc` and `nvrx-smonsvc` in a single job with health monitoring.

For individual deployment, see each component's `deploy/` directory.

## Container Image

Build an enroot squash image containing both services:

```bash
./services/scripts/build_enroot_image.sh
```

See [scripts/build_enroot_image.sh](scripts/build_enroot_image.sh) for usage with Slurm + pyxis.

## Monitoring

Periodically snapshot service endpoints for debugging:

```bash
# Both services
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
| `scripts/` | Shell scripts ([README](scripts/README.md)) |

## Library Layer

The core analysis functionality in `nvidia_resiliency_ext.attribution` can be used without HTTP.
See [attrsvc/README.md](attrsvc/README.md#python-api) for the Python API.
