# NVRX Attribution Scripts

Shared shell scripts for deployment and monitoring.

## Scripts

| Script | Description |
|--------|-------------|
| `run_services.sh` | Manage services as background processes (install, start, stop, status) |
| `setup_systemd.sh` | Manage services via systemd (auto-detects user vs system mode) |
| `nvrx_services.sbatch` | SLURM batch script to run both services together |
| `build_enroot_image.sh` | Build enroot container image with both services |
| `snapshot_services.sh` | Periodically snapshot both service endpoints |
| `common.sh` | Shared functions used by all scripts |

## Quick Start

### Option 1: Background Processes (recommended for most users)

```bash
# Set required environment
export NVRX_ATTRSVC_ALLOWED_ROOT=/lustre/logs
# API key: set env var OR create ~/.nvidia_api_key file
export NVIDIA_API_KEY=nvapi-...

# Install, start, and manage
./scripts/run_services.sh install   # Install packages
./scripts/run_services.sh start     # Start in background
./scripts/run_services.sh status    # Check status
./scripts/run_services.sh logs      # Tail log files
./scripts/run_services.sh stop      # Stop services
./scripts/run_services.sh restart   # Restart services

# Foreground mode (Ctrl+C to stop)
./scripts/run_services.sh run
```

### Option 2: Systemd (requires root or user systemd)

```bash
# As root: system-wide installation
sudo ./scripts/setup_systemd.sh install
sudo ./scripts/setup_systemd.sh start

# As regular user: user-local installation (auto-detected)
./scripts/setup_systemd.sh install
./scripts/setup_systemd.sh start
```

### API Key

The API key can be provided in multiple ways (checked in order):
1. `NVIDIA_API_KEY` environment variable
2. `NVIDIA_API_KEY_FILE` environment variable (path to key file)
3. `~/.nvidia_api_key` file
4. `~/.config/nvrx/nvidia_api_key` file

**Output files** (in `~/nvrx_logs/` by default):
- `<timestamp>_attrsvc.log` - Attribution service stdout/stderr
- `<timestamp>_smonsvc.log` - SLURM monitor stdout/stderr
- `<timestamp>_snapshot.log` - Snapshot process stdout/stderr
- `<timestamp>_snapshot_attrsvc.log` - Attribution service endpoint snapshots
- `<timestamp>_snapshot_smonsvc.log` - Monitor service endpoint snapshots

**Configuration:**
```bash
export NVRX_LOGS_DIR=/custom/log/dir       # Logs directory
# NVRX_ATTRSVC_CLUSTER_NAME auto-detected from SLURM (override if needed)
export SNAPSHOT_INTERVAL=300               # Snapshot every 5 min

# Processed files ledger (optional - prevents re-analysis after restart)
# Tracks which files have been analyzed and posted to Elasticsearch
export NVRX_ATTRSVC_CACHE_FILE=${NVRX_LOGS_DIR}/attrsvc_cache.json
# Grace period before validating file changes (default 600s = 10 min)
# export NVRX_ATTRSVC_CACHE_GRACE_PERIOD_SECONDS=600
```

## Combined SLURM Deployment

Run both `nvrx-attrsvc` and `nvrx-smonsvc` in a single SLURM job:

```bash
NVRX_ATTRSVC_ALLOWED_ROOT=/path/to/logs \
  sbatch --account=myaccount scripts/nvrx_services.sbatch
```

See [nvrx_services.sbatch](nvrx_services.sbatch) header for configuration options.

## Container Image

Build an enroot squash image for containerized deployments:

```bash
# Run from repo root
./services/scripts/build_enroot_image.sh [output_path]
```

**Environment variables:**
- `BASE_IMAGE` - Base Docker image (default: `python:3.12-slim`)
- `CONTAINER_NAME` - Build container name (default: `nvrx-services-build`)

**Output:** `nvrx-services.sqsh` (or custom path)

## Monitoring

Snapshot service endpoints periodically for debugging:

```bash
# Both services
./scripts/snapshot_services.sh hostname

# Environment variables:
#   NVRX_HOST              - Common host (default: localhost)
#   NVRX_ATTRSVC_PORT      - Attribution service port (default: 8000)
#   NVRX_SMONSVC_PORT      - Monitor service port (default: 8100)
#   SNAPSHOT_INTERVAL      - Interval in seconds (default: 600)
#   SNAPSHOT_OUTPUT_DIR    - Output directory (default: ~/nvrx_snapshots)
```

For individual service snapshots, see:
- `nvrx_attrsvc/deploy/snapshot_attrsvc.sh`
- `nvrx_smonsvc/deploy/snapshot_smonsvc.sh`

## Common Functions (common.sh)

Shared functions sourced by other scripts:

### Setup Functions

| Function | Description |
|----------|-------------|
| `setup_nvidia_api_key` | Load API key from env, file, or default location |
| `install_nvrx_packages` | Install NVRX packages from local repo |
| `validate_commands` | Check required commands exist |

### Validation Functions

| Function | Description |
|----------|-------------|
| `validate_attrsvc_allowed_root` | Validate `NVRX_ATTRSVC_ALLOWED_ROOT` is set |
| `check_attrsvc_reachable` | Check if attribution service is reachable (warning only) |
| `ensure_directory` | Create directory if it doesn't exist |

### Service Startup Functions

| Function | Description |
|----------|-------------|
| `wait_for_service` | Wait for foreground service to respond on port |
| `wait_for_background_service` | Wait for background service (with PID/log file handling) |

### Snapshot Functions

| Function | Description |
|----------|-------------|
| `snapshot_endpoint` | Snapshot a single HTTP endpoint |
| `snapshot_header` | Write snapshot header to output file |
| `snapshot_attrsvc` | Snapshot all attrsvc endpoints |
| `snapshot_smonsvc` | Snapshot all smonsvc endpoints |

**Usage in other scripts:**
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
# or from service directories:
source "${SCRIPT_DIR}/../scripts/common.sh"
```
