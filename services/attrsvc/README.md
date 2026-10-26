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

### Slack Notifications

attrsvc posts completed terminal analyses to Slack. Alerting lives here, rather
than in a client, so both deployment shapes are covered by one implementation:

| Mode | Submits logs | Slack |
|------|--------------|-------|
| As a service | `nvrx-smonsvc` + `nvrx-attrsvc` | ✅ |
| Inline | NVRx + `nvrx-attrsvc` (no monitor) | ✅ |

Both submit through the same `POST /logs` and both carry `user` and `job_id`, so
the alert is identical. One analysis produces exactly one message no matter how
many clients fetch the result.

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_BOT_TOKEN` | `""` | Bot token. If empty, falls back to `SLACK_BOT_TOKEN_FILE`, then `~/.slack_bot_token`, `~/.slack_token`, `~/.config/nvrx/slack_bot_token` |
| `SLACK_BOT_TOKEN_FILE` | — | Path to a file containing the token |
| `SLACK_CHANNEL` | `""` | Channel ID or name (e.g. `#trng-alerts`). In `.env` files, quote values starting with `#` |
| `NVRX_ATTRSVC_SLACK_NOTIFY_ACTIONS` | `STOP` | Comma- or space-separated actions that trigger a message. Valid: `STOP`, `RESTART`, `CONTINUE`, `UNKNOWN`, `TIMEOUT` |
| `NVRX_ATTRSVC_SLACK_EMAIL_DOMAIN` | `""` | Domain used to turn a job owner into an address for an `@` mention, e.g. `example.com`. Unset means no mention is attempted |

`slack-sdk` ships as a regular dependency, so no extra is needed.

Notifications are **off by default**, activating only when both a token and a
channel are configured; otherwise attrsvc logs `Slack alerts: disabled (...)` at
startup and behaves unchanged. Delivery is best
effort — a Slack outage is logged, never propagated into the analysis path.

Each alert is **two messages**: a three-line summary in the channel and the
analysis as a threaded reply. A channel of full attributions is unscannable, so
the channel keeps one line per job and the detail stays one click away.

```
*NVRx attribution:* `RESTART` on *aws-cmh-slurm-1* _(source: l1_enriched:eccn-gpt-5.4)_
*Job ID:* `4103814_0 (nemotron4_ultra_60t_phase1_v0) cycle 3`
*Log path:* `/scratch/.../logs/..._cycle3.log`
```

The job line carries the restart cycle when the logs are numbered
(`..._cycle<N>.log`); logs without cycle numbers have no attempt ordering, so
nothing is shown rather than an invented zero.

The owner mention rides on the summary so the ping carries the visible line. A
failed reply is logged but does not count as a failed notification — the alert
is already delivered and only the detail is lost.

The threaded reply carries the recommendation and its source, the cluster
(`NVRX_ATTRSVC_CLUSTER_NAME`, so alerts from several clusters in one channel are
distinguishable), the job ID, the narrative root cause, the L4 decision trail
(rule, category, retry outlook), an evidence line locating the failure, the
first three log lines at that failure, and — when the cause is unconfirmed —
the plausible causes and missing evidence.

The decision cites a line number; quoting the lines there saves opening a
multi-megabyte log to see what it actually says. Overlong lines are truncated,
and the section is omitted when the log cannot be read.

A category of `0` is L1's sanctioned "no listed category matches" and is
reported as such. That is a different fact from no category at all, which
happens on deterministic-only results where L1 never ran.

The job owner is named in every message. Turning that owner into an `@` mention
additionally requires `NVRX_ATTRSVC_SLACK_EMAIL_DOMAIN`, since a SLURM account
name is not an email address and the domain is site-specific. Without it the
alert is sent unmentioned; a failed lookup never blocks delivery.

attrsvc never sees the SLURM job name, so the run name is recovered from the
application log filename (`<run>_<jobid>_date_...`) to keep alerts
self-describing in both modes.

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
| `slack.py` | Slack alerts for completed terminal analyses |
| `deploy/run_attrsvc.sh` | Run service with logging (background) |
| `deploy/supervise_attrsvc.sh` | Restart an attrsvc Slurm child after transient exits |
| `deploy/snapshot_attrsvc.sh` | Periodic endpoint snapshot for debugging |
| `deploy/Dockerfile` | Docker build instructions |
| `deploy/requirements.txt` | Standalone attrsvc image dependencies; excludes workload-side PyTorch/CUDA |
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
- Dataflow postprocessing remains on the controller/manual path; the direct
  backend does not invoke it. Slack alerting is implemented in `slack.py` and
  fires from the direct backend on terminal completion.
