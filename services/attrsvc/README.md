# NVRX Attribution Service (nvrx-attrsvc)

FastAPI server that exposes log analysis over HTTP. The service runs
`RestartAgentRuntime` directly through the `lib` backend. The lower-level
attribution package still retains the MCP/controller implementation for
manual use, but `nvrx-attrsvc` no longer exposes it as a backend selection.

---

## Library vs this service

| | **Library** (`nvidia_resiliency_ext.attribution`) | **This package** (`nvidia_resiliency_ext.services.attrsvc`) |
|---|--------------------------------------------------|-----------------------------------|
| **Role** | **`RestartAgentRuntime`** for direct analysis; **`AttributionController`** for manual MCP use | HTTP API, env-based `Settings`, and rate limits |
| **Docs** | [`restart_agent/README.md`](../../docs/design/attribution/restart_agent/README.md), [`ARCHITECTURE.md`](../../src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md) for the MCP/controller analysis path | This file, [`ATTRSVC_SPEC.md`](ATTRSVC_SPEC.md) |

The Restart Agent design directory is the source of truth for the service
backend. **ARCHITECTURE.md** remains the source of truth for the manual
controller/LogSage/Flight Recorder path.

---

## Quick Start

```bash
# Install
cd services
pip install -e ..

# Run
export NVRX_ATTRSVC_ALLOWED_ROOT=/path/to/logs
# API key: set env var OR create ~/.llm_api_key file
export LLM_API_KEY_FILE=/secure/llm_api_key
nvrx-attrsvc
```

## Configuration

Environment variables (prefix: `NVRX_ATTRSVC_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FAST_API_ROOT_PATH` | `""` | FastAPI root path when serving behind a path-prefixing proxy |
| `ALLOWED_ROOT` | (required) | Base directory for allowed log paths |
| `ENDPOINT` | `""` | Unified bind endpoint. Supports `http://host:port`, `host:port`, `unix:///absolute/path.sock`, or an absolute socket path. Overrides `HOST`/`PORT`. |
| `HOST` | `127.0.0.1` | Listen address. Deployments that need remote access should set `NVRX_ATTRSVC_HOST=0.0.0.0` explicitly. |
| `PORT` | `8000` | Listen port |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, or `WARNING` for root logging; FastAPI `debug` when set to `DEBUG`. |
| `CLUSTER_NAME` | `""` | Cluster name retained for controller dataflow configuration |
| `EXPORT_URL` | `""` | Controller complete result export URI. The direct backend does not post results. |
| `DATAFLOW_QUEUE` | `""` | Controller queue parameter for dataflow HTTP posting |
| `DATAFLOW_TIMEOUT_SECONDS` | `10.0` | Controller dataflow HTTP request timeout |
| `RATE_LIMIT_SUBMIT` | `1200/minute` | Rate limit for POST /logs |
| `RATE_LIMIT_ANALYZE` | `60/minute` | Rate limit for GET /logs |
| `RATE_LIMIT_PREVIEW` | `120/minute` | Rate limit for GET /print |

**LLM / analysis** (optional — unset vars keep library defaults).
`AttributionHttpAdapter` resolves these into `RestartAgentConfig` for the direct
Restart Agent backend.

| Variable (with prefix)          | Description                                                                                                                                                                                                   |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `NVRX_ATTRSVC_LLM_MODEL`        | Optional LLM model override                                                                                                                                                                                   |
| `NVRX_ATTRSVC_LLM_BASE_URL`     | Optional LLM base URL override                                                                                                                                                                                |
| `NVRX_ATTRSVC_LLM_TEMPERATURE`  | Temperature (0.0 = deterministic)                                                                                                                                                                             |
| `NVRX_ATTRSVC_LLM_TOP_P`        | Top-p for nucleus sampling                                                                                                                                                                                    |
| `NVRX_ATTRSVC_LLM_MAX_TOKENS`   | Max tokens for response                                                                                                                                                                                       |
| `NVRX_ATTRSVC_COMPUTE_TIMEOUT`  | Timeout for analysis in seconds                                                                                                                                                                               |
| `NVRX_ATTRSVC_ANALYSIS_BACKEND` | `lib` (direct Restart Agent backend). |
| `NVRX_ATTRSVC_RESTART_AGENT_CONFIG` | Optional authoritative `restart_agent_config.v1` JSON file for `lib`; the first attrsvc integration requires exactly one route. |
| `NVRX_ATTRSVC_RESTART_AGENT_LOG_QUIET_SECONDS` | Required unchanged-log interval after the internal 10-second minimum live observation period; default `5`. |
| `NVRX_ATTRSVC_RESTART_AGENT_LOG_MAX_WAIT_SECONDS` | Maximum live terminal observation period before the source freezes; default `40`. |
| `NVRX_ATTRSVC_RESTART_AGENT_LOG_POLL_SECONDS` | Log-drain polling interval; default `0.25`. |

**LLM API Key**: the default `lib` route requires `LLM_API_KEY_FILE`. A supplied
Restart Agent config may name another key-file environment variable through its
`credential_ref`.

**Slack Notifications** (optional; no `NVRX_ATTRSVC_` prefix):

These settings are retained for the controller/manual path. The current
`nvrx-attrsvc` direct backend does not send Slack notifications.

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_BOT_TOKEN` | `""` | Bot token (empty = controller path tries file fallbacks below) |
| `SLACK_BOT_TOKEN_FILE` | — | Path to a file containing the token (checked before `~/.slack_bot_token` / `~/.slack_token`) |
| `SLACK_CHANNEL` | `""` | Channel ID or name (e.g. `#trng-alerts`). In `.env`, quote values that start with `#`: `SLACK_CHANNEL="#trng-alerts"` |

**Processed Files Ledger** (optional cache persistence):

These settings are retained for the controller cache path. The current
direct backend uses an in-memory attempt registry and does not persist a ledger.

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_FILE` | `""` | Controller ledger file path (ignored by the direct backend) |
| `CACHE_GRACE_PERIOD_SECONDS` | `600` | Controller cache grace period (ignored by the direct backend) |

Example: `NVRX_ATTRSVC_CACHE_FILE=/var/lib/nvrx/attrsvc_cache.json`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/stats` | GET | Service and request statistics |
| `/jobs` | GET | All tracked jobs and attempts |
| `/logs` | POST | Submit log for analysis |
| `/logs` | GET | Retrieve analysis results |
| `/print` | GET | Preview first 4KB of file |
| `/inflight` | GET | In-flight requests |
| `/docs` | GET | OpenAPI documentation |

**POST /logs** body:
```json
{
  "log_path": "/path/to/train_cycle2.log",
  "user": "alice",
  "job_id": "12345",
  "cycle_id": 2,
  "analysis_intent": "progressive"
}
```

**GET /logs** query params:
- `log_path` (required): Path to job output file
- `wait` (optional, default `true`): Set `false` to probe cache/in-flight state without starting or waiting for analysis.

The direct Restart Agent backend does not support the legacy `file` or
`wl_restart` selectors.

---

### POST/GET API contract

All success responses use HTTP 200. Interpret outcome from the response body only.

#### POST /logs

| Request (JSON body) | Type | Required | Description |
|--------------------|------|----------|-------------|
| `log_path` | string | Yes | Absolute path to job output file under allowed root |
| `user` | string | Yes | Job owner |
| `job_id` | string | No | Job identity used for same-job history |
| `cycle_id` | integer | No | Explicit restart-attempt order. The direct Restart Agent backend infers `_cycle<N>.log` only when absent. |
| `analysis_intent` | string | No | `track_only` (default), `progressive`, or `terminal`. |

| Response (200) | Type | Description |
|----------------|------|-------------|
| `submitted` | bool | Always true on success |
| `normalized_path` | string | Resolved path used as job key |
| `mode` | string | Compatibility field; `"SINGLE"` for the direct backend |
| `logs_dir` | null | Legacy compatibility field |
| `sched_restarts` | int | Legacy compatibility field; `0` for the direct backend |
| `files_analyzed` | int | Legacy compatibility field; `0` for the direct backend |

4xx/5xx return `{ "error_code": string, "message": string }`.

#### GET /logs

| Query | Type | Required | Description |
|-------|------|----------|-------------|
| `log_path` | string | Yes | Same path as used in POST |
| `wait` | bool | No | Default `true`. Set `false` to return immediately with the registered result, `in_flight`, or `pending`. |

The direct backend rejects the legacy `file` and `wl_restart` selectors.

**Response (200)**

| Field | Type | Description |
|-------|------|-------------|
| `result` | object | Restart Agent response when available; empty while no result exists. |
| `status` | string | `"completed"` for a result. With `wait=false`, may be `"in_flight"` or `"pending"`. |
| `recommendation` | object | Completed recommendation consumed by NVRx. |
| `candidate_recommendation` | object | Early deterministic candidate for observation only; NVRx does not act on it. |
| `wl_restart` | int | Attempt cycle ID, or `0` when unavailable. |

**`recommendation` object** — clients should branch on this field:

| Field | Type | Description |
|-------|------|-------------|
| `action` | string | `"STOP"`, `"RESTART"`, or `"UNKNOWN"`. |
| `reason` | string | Restart Agent justification or lifecycle reason. |
| `source` | string | Result source, such as `"deterministic"` or `"l1_enriched:<route>"`. |

Only completed `"STOP"` is actionable at the NVRx boundary. Every other completed
action is non-stopping. Terminal analysis failure is also a completed request:
`result.analysis_outcome` is `"failed"` and the recommendation is `"UNKNOWN"`.

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

### Slurm process supervision

An externally deployed attrsvc needs to remain reachable at the endpoint given
to NVRx even if the service process crashes. The attrsvc-only Slurm deployment
therefore runs a supervisor as the batch payload. The supervisor retains the
same CPU-node allocation and endpoint while restarting attrsvc after `1`, `2`,
`5`, `10`, and `20` seconds. If all five restarted processes exit before
becoming stable, the supervisor exits nonzero and Slurm marks the job failed.
Running continuously for five minutes resets the consecutive restart count.

Slurm cancellation, timeout, and preemption signals are forwarded to attrsvc
and do not trigger a restart. The following optional environment variables
override the deployment defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `NVRX_ATTRSVC_SUPERVISOR_BACKOFF_SECONDS` | `1 2 5 10 20` | Space-separated base-10 delay before each allowed restart. The number of entries is the maximum restart count. |
| `NVRX_ATTRSVC_SUPERVISOR_STABLE_SECONDS` | `300` | Positive base-10 continuous runtime that resets the consecutive restart count. |

The supervisor preserves the Slurm allocation and network endpoint, but attrsvc
runtime state remains in memory and is not restored after a process restart.

## Python API

**Embedding the direct analyzer** (no HTTP): use `build_restart_agent_runtime()`
and `RestartAgentRuntime.analyze()`, or the `restart-agent` CLI. See the
**[Restart Agent design](../../docs/design/attribution/restart_agent/README.md)**.
For the MCP/controller analysis stack, use `AttributionController`, `Analyzer`, or
`LogAnalyzer`; see the **[attribution README](../../src/nvidia_resiliency_ext/attribution/README.md)**.

**In-process HTTP adapter** (same repo, after `pip install`):

```python
import asyncio
from nvidia_resiliency_ext.services.attrsvc import AttributionHttpAdapter, setup

async def main():
    cfg = setup()
    adapter = AttributionHttpAdapter(cfg)
    
    result = await adapter.analyze_log("/path/to/job.log")
    adapter.shutdown()

asyncio.run(main())
```

## Files

| File | Description |
|------|-------------|
| `app.py` | FastAPI routes and middleware |
| `service.py` | `AttributionHttpAdapter` and the common backend protocol |
| `config.py` | `Settings` (pydantic), `setup()` loads settings and configures logging |
| `restart_agent_config.py` | Resolves attrsvc settings into `RestartAgentConfig` |
| `restart_agent_backend.py` | Direct progressive/terminal HTTP lifecycle over `RestartAgentRuntime` |
| `deploy/run_attrsvc.sh` | Run service with logging (background) |
| `deploy/supervise_attrsvc.sh` | Restart an attrsvc Slurm child after transient exits |
| `deploy/snapshot_attrsvc.sh` | Periodic endpoint snapshot for debugging |
| `deploy/Dockerfile` | Docker build instructions |
| `deploy/kubernetes.yaml` | Kubernetes deployment manifest |
| `deploy/slurm.sbatch` | SLURM batch script |

## Documentation

| Document | Audience |
|----------|----------|
| [ATTRSVC_SPEC.md](ATTRSVC_SPEC.md) | HTTP contract, service-level behavior; library internals in **ARCHITECTURE.md** |
| [../../docs/design/attribution/restart_agent/ATTRSVC_INTEGRATION.md](../../docs/design/attribution/restart_agent/ATTRSVC_INTEGRATION.md) | Direct NVRx/attrsvc/Restart Agent composition |
| [../../src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md](../../src/nvidia_resiliency_ext/attribution/ARCHITECTURE.md) | Library architecture, pipelines, MCP, coalescing |

## Configuration and postprocessing (summary)

- **Service config** is in `config.py` (`Settings` from env with prefix `NVRX_ATTRSVC_`).
- **`setup()`** in `config.py` loads settings and configures logging only.
- **`AttributionHttpAdapter`** translates `Settings` into `RestartAgentConfig`
  for the direct Restart Agent backend.
- Dataflow and Slack postprocessing remain on the controller/manual path;
  the direct backend does not invoke them.
