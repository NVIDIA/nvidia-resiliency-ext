# NVRX SLURM Monitor Service (nvrx-smonsvc)

SLURM job monitor that automatically submits completed job logs to the Attribution Service for analysis.

## Quick Start

```bash
# Install
cd services
pip install -e ..

# Run (requires nvrx-attrsvc running)
export NVRX_ATTRSVC_ENDPOINT=http://localhost:8000
export NVRX_SMONSVC_PARTITIONS="batch batch_long"
nvrx-smonsvc
```

## Configuration

Environment variables (prefix: `NVRX_SMONSVC_`) or command-line arguments:

| Variable / Argument | Default | Description |
|---------------------|---------|-------------|
| `NVRX_ATTRSVC_ENDPOINT` / `--attrsvc-endpoint` | `http://localhost:8000` | Attribution service endpoint (`http://host:port` or `unix:///path.sock`) |
| `HOST` / `--host` | `127.0.0.1` | Host/interface for the HTTP status server. Deployments that need remote access should set `NVRX_SMONSVC_HOST=0.0.0.0` explicitly. |
| `PORT` / `--port` | `None` | Port for HTTP server (stats, health, jobs) |
| `INTERVAL` / `--interval` | `180` | Poll interval in seconds |
| `PARTITIONS` / `--partitions` | `batch batch_long` | SLURM partitions (space-separated) |
| `USER` / `--user` | all users | Filter jobs by user |
| `JOB_PATTERN` / `--job-pattern` | `None` | Regex to filter job names |
| `TIMEOUT` / `--timeout` | `60` | HTTP request timeout in seconds |
| `VERBOSE` / `-v, --verbose` | `false` | Enable verbose logging |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |

CLI arguments override environment variables.

## Application Log Resolution

Some launchers point SLURM `StdOut` at a batch wrapper rather than the training
log. A common layout:

```
<run_dir>/slurm_out/slurm-<jobid>_<task>.out          # wrapper: launcher banner, few KB
<run_dir>/logs/<name>_<jobid>_date_..._cycle<N>.log   # the training log
```

Both naming conventions are handled — multi-cycle runs (`..._cycle<N>.log`) and
single-cycle runs (`....log`). A cycle log outranks a plain one, and metadata
sidecars (`.env.log`, `.tasks.log`) are never selected. The wrapper may also sit
directly in the run directory rather than under `slurm_out/`.

Attributing the wrapper yields "no failure signature found" regardless of what
the job did. When enabled, the monitor maps the wrapper back to the newest cycle
log for the same SLURM job.

| Variable | Default | Description |
|----------|---------|-------------|
| `NVRX_SMONSVC_APP_LOG_RESOLUTION` | `false` | Enable resolution (`1`/`true`/`yes`/`on`) |
| `NVRX_SMONSVC_APP_LOG_STDOUT_SUBDIR` | `slurm_out` | Wrapper directory, stripped to find the run directory |
| `NVRX_SMONSVC_APP_LOG_SUBDIR` | `logs` | Application log directory, relative to the run directory |

**Off by default** — it encodes a site layout convention, and a deployment whose
`StdOut` already *is* the training log must not have its paths rewritten. When
the layout does not match or no cycle log exists, the monitor falls back to the
original `StdOut` path rather than skipping the job.

Application logs embed the **parent** job ID, so every array task of a job
resolves to the same log. The monitor claims each resolved path once; sibling
tasks are skipped rather than re-analyzing identical content (which would also
collide in the attrsvc registry, where a path maps to a single job ID). Both
counts appear under `log_paths` in `/stats`:

```json
{"log_paths": {"claimed": 12, "duplicates_skipped": 158}}
```

## Slack Notifications

The monitor posts attribution results to Slack. Credentials use unprefixed
environment variables (matching the equivalent attrsvc settings):

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_BOT_TOKEN` | `""` | Bot token. If empty, falls back to `SLACK_BOT_TOKEN_FILE`, then `~/.slack_bot_token`, `~/.slack_token`, `~/.config/nvrx/slack_bot_token` |
| `SLACK_BOT_TOKEN_FILE` | — | Path to a file containing the token |
| `SLACK_CHANNEL` | `""` | Channel ID or name (e.g. `#trng-alerts`). In `.env` files, quote values starting with `#` |
| `NVRX_SMONSVC_SLACK_NOTIFY_ACTIONS` | `STOP` | Comma- or space-separated recommendation actions that trigger a message. Valid: `STOP`, `RESTART`, `CONTINUE`, `UNKNOWN`, `TIMEOUT` |

Requires `slack-sdk`:

```bash
pip install 'nvidia-resiliency-ext[attribution]'
```

Notifications are **off by default**. They activate only when `slack-sdk` is
installed *and* both a token and a channel are configured; otherwise the
monitor logs `Slack alerts: disabled (...)` at startup and runs unchanged.

By default only `STOP` pages, since `RESTART` is the routine outcome and would
be noisy. To page on more actions:

```bash
export SLACK_BOT_TOKEN_FILE=/secure/slack_bot_token
export SLACK_CHANNEL="#trng-alerts"
export NVRX_SMONSVC_SLACK_NOTIFY_ACTIONS="STOP,TIMEOUT"
```

Each message carries the recommendation action and reason, the job ID and name,
the attributed issues, the terminal-issue explanation, and the log path. The
job owner is mentioned when their `{user}@nvidia.com` address resolves to a
Slack account.

Delivery is best effort: a Slack outage is counted and logged, never propagated
into the monitor's polling loop. Counters are exposed under `slack` in `/stats`:

```json
{"slack": {"attempts": 12, "sent": 11, "failed": 1, "skipped_action": 143}}
```

The bot must be invited to the target channel (`/invite @your-bot`) and needs
the `chat:write` scope, plus `users:read.email` for owner mentions.

## API Endpoints

When `PORT` is set, the monitor exposes an HTTP server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/stats` | GET | Monitor statistics |
| `/jobs` | GET | List tracked jobs |

## How It Works

1. Polls SLURM for completed/failed jobs in configured partitions
2. For each terminal job, extracts the output log path (optionally resolving it
   to the application log, see below)
3. Submits the log to the Attribution Service via POST /logs
4. Tracks job state to avoid duplicate submissions
5. Posts a Slack alert when the recommendation matches the configured actions

## Architecture

```
  __main__.py (CLI)
         |
         v
  +------------------+
  | SlurmJobMonitor  |  (monitor.py: poll loop, coordinates components)
  +--------+---------+
           |
     +-----+-----+--------------+--------------+
     |           |              |              |
     v           v              v              v
  +------+  +----------+  +------------+  +-------------+
  |Slurm |  |job_      |  |MonitorState|  |StatusServer |
  |Client|  |handlers  |  |(models.py) |  |(status_     |
  +--+---+  +----+-----+  +-----+------+  | server.py)  |
     |           |               ^        +------+------+
     |           |  submit_log,  |              |
     |           |  fetch_results|              v
     |           +------+--------+        +----------+
     |                  |                 | HTTP     |
     v                  v                 |/healthz  |
  +--------+         +----------+         |/stats    |
  | SLURM  |         |Attrsvc   |         |/jobs     |
  |squeue  |         |Client    |         +----------+
  |scontrol|         +----+-----+
  | sacct  |              |
  +--------+              v
                     +----------+
                     | nvrx-    |
                     | attrsvc  |
                     | POST/GET |
                     | /logs    |
                     +----------+
```

- **SlurmJobMonitor**: Main loop; every `INTERVAL` seconds polls SLURM, updates in-memory job state, submits new logs to attrsvc via **AttrsvcClient**, and fetches results for completed submissions.
- **SlurmClient**: Runs `squeue` (running/completing jobs), `scontrol` (output paths), `sacct` (batch path lookup); handles array and het job IDs.
- **AttrsvcClient**: HTTP client with retries and rate limiting; POST `/logs` to submit, GET `/logs` to fetch attribution results, GET `/stats` for status.
- **job_handlers**: Layer that interacts with nvrx-attrsvc: `submit_log()` and `fetch_results()` use **AttrsvcClient** (POST/GET), then update **MonitorState** and summarize `recommendation` from responses.
- **StatusServer**: Optional HTTP server (when `PORT` is set) serving `/healthz`, `/stats`, `/jobs` from **MonitorState**; can optionally proxy attrsvc `/stats`.

### SLURM Job ID Handling

The monitor handles various SLURM job ID formats:

| Format | Type | Handling |
|--------|------|----------|
| `12345` | Regular job | Processed normally |
| `12345_0` | Array task | Processed normally |
| `12345+0` | Heterogeneous (het) job component | Processed (sacct uses base ID) |
| `12345[0-10]` | Array summary | Skipped (individual tasks queried separately) |

**Heterogeneous jobs**: Het job components (e.g., `1234+0`, `1234+1`) are processed. When fetching
output paths via `sacct`, the base job ID is used since SLURM associates paths with the parent job.

**Array jobs**: Array task summaries with bracket notation are skipped since they don't represent
individual runnable jobs. Individual array tasks (e.g., `12345_0`) are processed normally.

## Usage Examples

```bash
# Monitor specific partitions
nvrx-smonsvc --partitions "gpu gpu_long"

# Monitor specific user's jobs
nvrx-smonsvc --user alice

# Filter jobs by name pattern
nvrx-smonsvc --job-pattern "training_.*"

# Enable HTTP status server
nvrx-smonsvc --port 8100

# Bind status server on all interfaces for remote access
nvrx-smonsvc --host 0.0.0.0 --port 8100

# Verbose logging
nvrx-smonsvc -v
```

## Files

| File | Description |
|------|-------------|
| `__main__.py` | CLI entry point |
| `monitor.py` | Main monitor class |
| `slurm.py` | SLURM subprocess calls (squeue, scontrol, sacct) with batching and het job support |
| `attrsvc_client.py` | HTTP client for Attribution Service |
| `status_server.py` | Status server (/stats, /jobs, /healthz) |
| `slack.py` | Slack notifications for attribution results |
| `log_resolver.py` | Maps SLURM stdout wrappers to application logs |
| `models.py` | Data models (JobState, SlurmJob, MonitorState) |
| `deploy/run_smonsvc.sh` | Run service with logging (background) |
| `deploy/snapshot_smonsvc.sh` | Periodic endpoint snapshot for debugging |

## Deployment

See `deploy/` directory:
- **SLURM**: `deploy/slurm.sbatch`

For combined deployment with attrsvc, see `../scripts/nvrx_services.sbatch`
