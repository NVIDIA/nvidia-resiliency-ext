# NVRX Attribution Service (nvrx-attrsvc)

FastAPI server that exposes log analysis over HTTP. It wraps the **`nvidia_resiliency_ext.attribution`** library (**`Analyzer`**, coalescing, postprocessing) with pydantic `Settings`, routes, and optional cache persistence.

---

## Library vs this service

| | **Library** (`nvidia_resiliency_ext.attribution`) | **This package** (`nvrx_attrsvc`) |
|---|--------------------------------------------------|-----------------------------------|
| **Role** | **`Analyzer`** (→ `LogAnalyzer`, pipelines, MCP/lib LogSage, FR analysis, jobs/splitlog) | HTTP API, env-based `Settings`, rate limits, ledger file |
| **Docs** | [`src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md`](../../src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md), [`README.md`](../../src/nvidia_resiliency_ext/attribution/README.md) | This file, [`ATTRSVC_SPEC.md`](ATTRSVC_SPEC.md) |

Do not duplicate library internals here—**ARCHITECTURE.md** is the source of truth for package layout, `LogAnalyzerConfig` / `AnalysisPipelineMode` (default `LOG_AND_TRACE` for this service), MCP vs in-process backends, and analysis flow.

---

## Quick Start

```bash
# Install
cd services
pip install -e .

# Run
export NVRX_ATTRSVC_ALLOWED_ROOT=/path/to/logs
# API key: set env var OR create ~/.nvidia_api_key file
export NVIDIA_API_KEY=nvapi-...
nvrx-attrsvc
```

## Configuration

Environment variables (prefix: `NVRX_ATTRSVC_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ROOT` | (required) | Base directory for allowed log paths |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, or `WARNING` for root logging and MCP; FastAPI `debug` when set to `DEBUG`. Legacy env: `LOG_LEVEL_NAME`. |
| `CLUSTER_NAME` | `""` | Cluster name for dataflow posting |
| `DATAFLOW_INDEX` | `""` | Elasticsearch index for result posting |
| `RATE_LIMIT_SUBMIT` | `1200/minute` | Rate limit for POST /logs |
| `RATE_LIMIT_ANALYZE` | `60/minute` | Rate limit for GET /logs |
| `RATE_LIMIT_PREVIEW` | `120/minute` | Rate limit for GET /print |

**LLM / analysis** (optional — unset vars keep library defaults). Use the prefixed names below in the environment or ``.env``; ``AttributionService`` passes them into ``LogSageExecutionConfig`` and the library ``Analyzer`` (LogSage / MCP / merge paths). See **ARCHITECTURE.md §7**.

| Variable (with prefix) | Description |
|----------|-------------|
| `NVRX_ATTRSVC_LLM_MODEL` | LLM model identifier |
| `NVRX_ATTRSVC_LLM_TEMPERATURE` | Temperature (0.0 = deterministic) |
| `NVRX_ATTRSVC_LLM_TOP_P` | Top-p for nucleus sampling |
| `NVRX_ATTRSVC_LLM_MAX_TOKENS` | Max tokens for response |
| `NVRX_ATTRSVC_COMPUTE_TIMEOUT` | Timeout for analysis in seconds |
| `NVRX_ATTRSVC_ANALYSIS_BACKEND` | `mcp` (subprocess MCP, default) or `lib` (in-process LogSage and flight-recorder analysis). Same setting for both; library behavior: **ARCHITECTURE.md §7**. Legacy env: `NVRX_ATTRSVC_LOG_ANALYSIS_BACKEND`. |

**NVIDIA API Key** (required, checked in order):
1. `NVIDIA_API_KEY` environment variable
2. `NVIDIA_API_KEY_FILE` environment variable (path to file)
3. `~/.nvidia_api_key` file
4. `~/.config/nvrx/nvidia_api_key` file

**Slack Notifications** (optional; no `NVRX_ATTRSVC_` prefix):

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_BOT_TOKEN` | `""` | Bot token (empty = try file fallbacks below via `setup()`) |
| `SLACK_BOT_TOKEN_FILE` | — | Path to a file containing the token (checked before `~/.slack_bot_token` / `~/.slack_token`) |
| `SLACK_CHANNEL` | `""` | Channel ID or name (e.g. `#trng-alerts`). In `.env`, quote values that start with `#`: `SLACK_CHANNEL="#trng-alerts"` |

When configured, sends alerts to Slack for jobs with `auto_resume = "STOP - DONT RESTART IMMEDIATE"`.

**Processed Files Ledger** (optional cache persistence):

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_FILE` | `""` | Path to ledger file for persistence (empty = disabled) |
| `CACHE_GRACE_PERIOD_SECONDS` | `600` | Grace period before validating file on cache hit (10 min) |

The cache acts as a **processed files ledger** - tracking which files have been analyzed
and posted to Elasticsearch. This prevents duplicate processing after service restarts.

**Why needed:** When the service restarts, smonsvc resubmits recently completed jobs.
Without the ledger, all files would be re-analyzed and re-posted to ES. The ledger
allows the service to recognize "I've already processed this file" and skip it.

**What's stored:** `(path, mtime, size, result)` for each processed file.

**Validation strategy:**

| Phase | Behavior |
|-------|----------|
| **Grace period** (first 10 min) | Serve from cache without file validation |
| **After grace period** | `stat()` file on each hit; if `(mtime, size)` changed, invalidate and re-analyze |
| **Eviction** | Remove entries when file mtime > 14 days (safeguard against unbounded growth) |
| **On import** | Validate `(mtime, size)`; skip if file changed or > 14 days old |

The grace period absorbs straggling writes at end of file, preventing unnecessary re-analysis.

**Note:** Ledger is saved only on graceful shutdown (SIGTERM). If the service crashes or is
killed (SIGKILL), in-memory entries are lost. Use a process manager with adequate stop timeout.

Example: `NVRX_ATTRSVC_CACHE_FILE=/var/lib/nvrx/attrsvc_cache.json`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/stats` | GET | Cache and request statistics |
| `/jobs` | GET | All tracked jobs (pending, single, splitlog modes) |
| `/logs` | POST | Submit log for analysis |
| `/logs` | GET | Retrieve analysis results |
| `/print` | GET | Preview first 4KB of file |
| `/inflight` | GET | In-flight requests |
| `/docs` | GET | OpenAPI documentation |

**POST /logs** body:
```json
{
  "log_path": "/path/to/slurm-12345.out",
  "user": "alice",
  "job_id": "12345"
}
```

**GET /logs** query params:
- `log_path` (required): Path to job output file
- `file` (optional): Filename for splitlog mode
- `wl_restart` (optional): Workload restart index within file (0-based; single-file supports multiple cycles)

---

### POST/GET API contract

All success responses use HTTP 200. Interpret outcome from the response body only.

#### POST /logs

| Request (JSON body) | Type | Required | Description |
|--------------------|------|----------|-------------|
| `log_path` | string | Yes | Absolute path to job output file under allowed root |
| `user` | string | Yes | Job owner |
| `job_id` | string | No | Job ID (used for splitlog detection) |

| Response (200) | Type | Description |
|----------------|------|-------------|
| `submitted` | bool | Always true on success |
| `normalized_path` | string | Resolved path used as job key |
| `mode` | string | `"PENDING"` \| `"SINGLE"` \| `"SPLITLOG"` |
| `logs_dir` | string \| null | Set when mode is SPLITLOG |
| `sched_restarts` | int | Scheduler restart count (SPLITLOG) |
| `files_analyzed` | int \| null | Number of files analyzed (SPLITLOG only; null for PENDING/SINGLE) |

4xx/5xx return `{ "error_code": string, "message": string }`.

#### GET /logs

| Query | Type | Required | Description |
|-------|------|----------|-------------|
| `log_path` | string | Yes | Same path as used in POST |
| `file` | string | No | Filename for splitlog (e.g. `model_12345_cycle0.log`) |
| `wl_restart` | int | No | Workload restart index within file (0-based). Omit to get all cycles (single-file) or use default. |

**Response (200)** — same top-level shape for single-file and splitlog; splitlog adds extra fields.

| Field | Type | Description |
|-------|------|-------------|
| `result` | object | **Inner result** from the analysis pipeline (see below). |
| `status` | string | Always `"completed"` (request completed). |
| `wl_restart` | int | Which workload cycle this result is for (0 when returning all). |
| `wl_restart_count` | int \| null | Total workload cycles in the file (single-file; null if N/A). |
| `mode` | string | Only for splitlog: `"splitlog"`. |
| `sched_restarts` | int | Only for splitlog. |
| `log_file` | string | Only for splitlog: path to the analyzed log file. |

**Inner `result` object** — client should branch on `result.state`:

| Field | Type | When | Description |
|-------|------|------|-------------|
| `module` | string | Always | e.g. `"log_analyzer"`. |
| `state` | string | Always | **Drives client decision: continue or stop.** One of `"timeout"` \| `"CONTINUE"` \| `"STOP"`. See below. |
| `result` | array | Always | Attribution items (one per cycle). Empty when `state === "timeout"`. |
| `error` | string | When `state === "timeout"` | Human-readable timeout message. |

Other fields (e.g. `result_id`, `resource_uri`) may be present.

**`result.state` — how the client should use it:**

1. **`"timeout"`** — Analysis did not complete. Use `result.error` for the message; do not treat as a successful attribution. Do not use `result.state` for a continue/stop decision.
2. **`"CONTINUE"`** — Analysis succeeded; attribution suggests it is safe to continue (e.g. resume/restart the job).
3. **`"STOP"`** — Analysis succeeded; attribution suggests **do not** continue (e.g. do not restart immediately; user intervention or different action required).

The client should branch on `result.state` to decide whether to continue or stop the workflow, and to choose between `result.error` (timeout) and `result.result` (attribution text).

**Type note:** `state` is a string in JSON for compatibility. The set of values is fixed; clients may treat it as an enum (e.g. in OpenAPI schema or client code: `"timeout" | "CONTINUE" | "STOP"`).

4xx/5xx return `{ "error_code": string, "message": string }`.

#### GET /print

| Query | Type | Required | Description |
|-------|------|----------|-------------|
| `log_path` | string | Yes | Absolute path to file under allowed root |

**Response (200):** `Content-Type: text/plain` — raw file content (first 4KB).  
4xx/5xx return JSON `{ "error_code": string, "message": string }`.

---

## Resource Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| CPU | 0.5 | 2 | Mostly I/O bound |
| Memory | 256MB | 1GB | Depends on MAX_JOBS and file sizes |
| Disk | 100MB | 500MB | Logs only (no persistence) |

## Deployment

All deployment and run scripts live under `deploy/`:
- **Docker**: `deploy/Dockerfile`
- **Kubernetes**: `deploy/kubernetes.yaml`
- **SLURM**: `deploy/slurm.sbatch`
- **Run (background)**: `deploy/run_attrsvc.sh [output_dir]`
- **Snapshot (debug)**: `deploy/snapshot_attrsvc.sh [host] [port]`

For combined deployment with monitor, see `../scripts/nvrx_services.sbatch`

## Python API

**Embedding the library** (no HTTP): use **`Analyzer`** (or `LogAnalyzer` / `LogAnalyzerConfig` for lower-level control) from `nvidia_resiliency_ext.attribution`. Overview and examples: **[attribution README](../../src/nvidia_resiliency_ext/attribution/README.md)** · **[ARCHITECTURE.md](../../src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md)**.

**In-process service wrapper** (same repo, after `pip install`):

```python
import asyncio
from nvrx_attrsvc import AttributionService, setup

async def main():
    cfg = setup()
    service = AttributionService(cfg)
    
    result = await service.analyze_log("/path/to/job.log")
    service.shutdown()

asyncio.run(main())
```

## Files

| File | Description |
|------|-------------|
| `app.py` | FastAPI routes and middleware |
| `service.py` | `AttributionService` — wraps **`Analyzer`** |
| `config.py` | `Settings` (pydantic), `setup()` wires postprocessing (poster via `post_backend.post`, Slack) from cfg |
| `deploy/run_attrsvc.sh` | Run service with logging (background) |
| `deploy/snapshot_attrsvc.sh` | Periodic endpoint snapshot for debugging |
| `deploy/Dockerfile` | Docker build instructions |
| `deploy/kubernetes.yaml` | Kubernetes deployment manifest |
| `deploy/slurm.sbatch` | SLURM batch script |

## Documentation

| Document | Audience |
|----------|----------|
| [ATTRSVC_SPEC.md](ATTRSVC_SPEC.md) | HTTP contract, service-level behavior; library internals in **ARCHITECTURE.md** |
| [../../src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md](../../src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md) | Library architecture, pipelines, MCP, coalescing |

## Configuration and postprocessing (summary)

- **Service config** is in `config.py` (`Settings` from env with prefix `NVRX_ATTRSVC_`).
- **`setup()`** in `config.py` loads settings, configures logging, and wires the library postprocessing singleton via `configure(default_poster=..., cluster_name=..., dataflow_index=..., slack_bot_token=..., slack_channel=...)`.
- The analyzer calls `post_results()` from the library when it has results; posting and Slack use the values set in `configure()`.
