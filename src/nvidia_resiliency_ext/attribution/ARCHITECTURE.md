# Attribution library architecture

This document describes how `nvidia_resiliency_ext.attribution` is structured and how its pieces fit together.

---

## 1. Scope

The package is **library-only**: no HTTP server. The usual entry point is **`Analyzer`** (`analyzer/engine.py`): request coalescing, `submit` / `analyze`, and delegation to **`LogAnalyzer`**. You can also embed **`LogAnalyzer`** directly if you do not need the coalescer. Path validation, job tracking, LLM calls, and optional posting hooks are expressed as Python APIs.

---

## 2. Diagrams

### 2.1 End-to-end flow

How a typical `analyze` path moves through the library (cache miss shown).

```mermaid
flowchart TB
    subgraph caller [Caller]
        APP[Your code]
    end

    subgraph analyzer_pkg [analyzer]
        AN[Analyzer]
    end

    subgraph orchestration_pkg [orchestration]
        LA[LogAnalyzer]
        SP[slurm_parser / SplitlogTracker]
    end

    subgraph coalesce [coalescing]
        RC[RequestCoalescer]
    end

    subgraph run [Analysis execution]
        RLLM["_run_llm_analysis"]
        RAP["run_attribution_pipeline"]
        LS["LogSage\n(MCP or in-process)"]
        FR["trace_analyzer\nextract + analyze_fr_dump"]
        MERGE["log_fr_analyzer\noptional merge_log_fr_llm"]
        PP[postprocessing]
    end

    APP -->|submit / analyze| AN
    AN --> LA
    LA --> SP
    AN --> RC
    RC -->|cache hit| OUT[LogAnalysisCoalesced]
    RC -->|cache miss| RLLM
    RLLM --> LA
    LA --> RAP
    RAP --> LS
    RAP --> FR
    RAP -.->|LOG_AND_TRACE_WITH_LLM| MERGE
    LA --> PP
    RLLM --> OUT
    OUT -->|LogAnalysisCycleResult / LogAnalysisSplitlogResult| APP
```

### 2.2 Pipeline modes

`LogAnalyzer.analysis_pipeline_mode` (on the **`LogAnalyzer`** instance) selects what `run_attribution_pipeline` runs.

```mermaid
flowchart LR
    subgraph modes [AnalysisPipelineMode]
        A[LOG_ONLY]
        B[TRACE_ONLY]
        C[LOG_AND_TRACE]
        D[LOG_AND_TRACE_WITH_LLM]
    end

    A --> LS2[LogSage only]
    B --> FR2[FR dump analysis only]
    C --> PAR[LogSage + FR when dump path found\nparallel when both]
    D --> PAR2[Same as LOG_AND_TRACE\nwhen FR data exists]
    PAR2 --> M[merge_log_fr_llm]
```

### 2.3 Major modules (dependency view)

```mermaid
flowchart TB
    AN[Analyzer]
    LA[LogAnalyzer]

    subgraph pipeline [orchestration/analysis_pipeline]
        AP[run_attribution_pipeline]
    end

    subgraph inputs [Inputs]
        MCP[mcp_integration]
        NV[NVRxLogAnalyzer]
        TA[trace_analyzer]
    end

    subgraph fusion [Optional fusion]
        CLF[log_fr_analyzer]
    end

    subgraph out [Outputs]
        CO[coalescing / LogAnalysisCoalesced]
        PP[postprocessing]
    end

    AN --> LA
    LA --> AP
    LA --> CO
    LA --> PP
    AP --> MCP
    AP --> NV
    AP --> TA
    AP --> CLF
```

### 2.4 Proposed attribution controller boundary

The long-term boundary should separate **frontends**, an **AttributionController**, and
**analysis engines**. Both `ft_launcher` and `nvrx-attrsvc` need attribution, but neither
should own the full set of attribution internals.

```text
[ft_launcher | nvrx-attrsvc | future callers]
        <>
AttributionController
        <>
[MCP tools | in-process analyzers | third-party services]
```

The **frontend** layer owns caller-specific lifecycle and transport:

- `ft_launcher`: worker monitoring, restart timing, restart budget, and applying the final restart/stop decision.
- `nvrx-attrsvc`: HTTP routing, FastAPI lifecycle, request/response models, service auth/rate limits, and service health endpoints.
- Future callers: CLI tools, batch jobs, notebook/debug workflows, or other control planes.

The **AttributionController** owns attribution semantics and state:

- request submission and job tracking
- path policy / allowed-root enforcement for the deployment
- request coalescing and cache lifecycle
- log analysis orchestration
- FR discovery and FR analysis orchestration
- result normalization into a caller-stable shape
- Slack and direct dataflow HTTP posting, with idempotency for side effects
- dry-run behavior and attribution-level status

The controller should have one contract with two deployment modes:

```text
Embedded mode:
external client <> nvrx-attrsvc process
                    └── AttributionController object <> MCP

Hosted mode:
ft_launcher <> AttributionController process <> MCP
```

For `nvrx-attrsvc`, the controller can be embedded because the service process already provides
the process boundary, HTTP lifecycle, and health surface. For `ft_launcher`, the controller should
be hosted as a separate long-lived process so launcher code does not own attribution state,
side-effect idempotency, cache, or MCP lifecycle.

Configuration belongs at controller construction or process startup, not as ordinary per-request
state. The concrete boundary is `AttributionControllerConfig`: path policy plus nested analysis,
cache, credentials, and postprocessing config. The analysis section uses
`analysis.engine_backend` (`lib` or `mcp`) for the LogSage/FR execution backend; that is separate
from whether the controller itself is embedded or hosted as a process. Controller config also
includes LLM knobs, cache persistence, timeouts, LLM API key policy, Slack,
direct dataflow HTTP, dry-run behavior, and non-secret deployment metadata. Avoid a mutable
public `configure(...)` API unless dynamic reconfiguration becomes a real requirement.

The controller contract should stay small:

```text
AttributionController(config)
  start(...)
  submit_log(...)
  analyze_log(...)
  get_result(...)
  status(...)
  shutdown(...)
```

`status(...)` is operational state, not attribution output. It should report readiness and
dependency health: `ready` / lifecycle state, sanitized config summary, cache and in-flight counts,
tracked-job summary, MCP health, Slack/dataflow configured-or-error state, and recent errors.

The **analysis engine** layer owns compute:

- MCP module tools such as `log_analyzer`, `fr_analyzer`, and `log_fr_analyzer`
- in-process equivalents loaded by the controller when a subprocess boundary is unnecessary
- future external attribution services or adapters

The key rule is that low-level MCP module tools should not become the owner of the whole
attribution product surface. MCP is a good process boundary for analysis engines; the controller
above it should own cache, orchestration, side effects, and policy. If the controller itself is
implemented using MCP transport, it should expose service-grade methods such as
`submit_log`, `analyze_log`, `get_result`, and `status` rather than overloading the existing
`log_analyzer` module tool.

---

## 3. Major subsystems

| Area | Responsibility |
|------|------------------|
| **`controller.py`** | **`AttributionController`** and nested config dataclasses: frontend-facing attribution boundary; owns startup config, LLM API key validation, cache persistence, health/status policy, postprocessing wiring, dataflow/Slack stats, and delegates analysis to **`Analyzer`**. |
| **`analyzer/`** | **`Analyzer`** (`engine.py`): path policy, `RequestCoalescer`, `submit` / `analyze`, splitlog branches; delegates heavy lifting to **`LogAnalyzer`**. Re-exports pipeline symbols from `orchestration.analysis_pipeline` for convenience. |
| **`log_analyzer/`** | LogSage package surface only: `NVRxLogAnalyzer` implementation plus its minimal `__init__.py` export. It no longer owns orchestration, parsing, splitlog, or config/types modules. |
| **`orchestration/`** | Log-side orchestration subsystem: **`LogAnalyzer`**, `Job` / `FileInfo`, SLURM parsing, splitlog polling, **`run_attribution_pipeline`** / **`AnalysisPipelineMode`**, `LogAnalyzerConfig` and result dataclasses, path validation, wire keys (`RESP_*`), and LogSage execution config/runtime helpers. |
| **`coalescing/`** | `RequestCoalescer`: dedupe concurrent analysis for the same path; cache entries as `LogAnalysisCoalesced` (LogSage dict + optional FR fields + optional LLM merge summary) |
| **`trace_analyzer/`** | Flight-recorder dump discovery/analysis (`extract_fr_dump_path`, `analyze_fr_dump`, `CollectiveAnalyzer`, etc.) |
| **`combined_log_fr/`** | MCP tool **`log_fr_analyzer`** for one-call LogSage + FR collection; optional **log + FR LLM fusion** (`CombinedLogFR`, `merge_log_fr_llm()`) for `LOG_AND_TRACE_WITH_LLM` |
| **`postprocessing/`** | Build records, optional direct dataflow HTTP post, Slack; `post_analysis_items` / `post_results` run as best-effort observability after analysis |
| **`mcp_integration/`** | Subprocess MCP client/server so LogSage can run isolated from the caller (see `mcp_integration/README.md`) |

---

## 4. Analysis pipeline (conceptual)

1. **Submit / classify** (`Analyzer.submit` → `LogAnalyzer.submit`): validate path under `allowed_root`, parse job output for `LOGS_DIR` and modes (`PENDING` → `SINGLE` / `SPLITLOG`).
2. **Analyze** (`Analyzer.analyze`): resolve job mode, file + optional `wl_restart`; on cache hit return from `RequestCoalescer`; on miss call **`_run_llm_analysis`** → **`LogAnalyzer.run_attribution_for_path`**.
3. **`run_attribution_for_path`** loads log text via **MCP** or **in-process** LogSage (`LogSageExecutionConfig.use_lib_log_analysis`), then runs **`run_attribution_pipeline`**:
   - **`LogAnalyzer.analysis_pipeline_mode`** selects log-only, trace-only, log+trace, or log+trace+LLM merge. The default **`Analyzer`** constructs **`LogAnalyzer`** with **`LOG_AND_TRACE`**; override by passing a custom **`log_analyzer`** (or embed **`LogAnalyzer`** directly).
4. **Validate** LogSage-shaped dict (unless trace-only); **schedule** best-effort observability through `postprocessing.post_analysis_items` (including the FR-only path when there are no cycles but FR data exists). Dataflow/Slack latency and failures are not on the launcher response path.
5. **Return** `LogAnalysisCoalesced` to the coalescer; callers receive `LogAnalysisCycleResult` / `LogAnalysisSplitlogResult`.

---

## 5. Terminology: scheduler vs workload restarts

These concepts drive `Job`, `FileInfo`, and HTTP query parameters such as `wl_restart`.

| Concept | Meaning | How it appears |
|--------|---------|----------------|
| **Scheduler restart** (`sched_restart`) | External orchestrator (SLURM, Kubernetes, …) re-runs the job | `<< START PATHS >>` in the job output file; new file in `LOGS_DIR` in splitlog setups |
| **Workload restart** (`wl_restart`) | Framework restarts *inside* the same allocation | `Cycle: N` in log content, or `_cycleN` in filenames |

Hierarchy (conceptual): **Job** → scheduler-restart blocks → within each, one or more **workload** cycles in the same file or across files.

Implementation details: `orchestration/job.py`, `orchestration/splitlog.py`,
`orchestration/slurm_parser.py`.

---

## 6. Content splitting (multi-cycle files)

For logs with multiple workload cycles in one file, analysis uses chunking (e.g. `chunk_logs_strict` in `log_analyzer`) guided by **`Cycle: N`** markers (pattern such as `profiling.py:.*Cycle: N`):

1. Find all cycle markers in the file.
2. Take lines from `Cycle: N` to `Cycle: N+1` (or EOF).
3. Analyze each chunk; combine results in the output.

If no markers are found, the whole file is treated as a single cycle.

---

## 7. Configuration: `Analyzer` ctor vs `LogAnalyzerConfig` vs `LogSageExecutionConfig`

Do **not** conflate the bundled type **`LogAnalyzerConfig`** (`orchestration/types.py`) with the **`Analyzer`** constructor. They overlap conceptually but are wired differently.

### 7.1 `Analyzer` (`analyzer/engine.py`)

Constructed with **keyword and positional arguments**, not `config=...`:

- **`allowed_root`** (required): absolute path prefix for validation.
- **`use_lib_log_analysis`** (default `False`): lib vs MCP when **`log_sage`** is omitted; ignored when **`log_sage`** is provided (then use `log_sage.use_lib_log_analysis`).
- **`log_sage`** (optional): a **`LogSageExecutionConfig`** (`orchestration/config.py`) — LLM knobs, MCP log level, lib/MCP switch. If omitted, defaults are built from **`use_lib_log_analysis`** only.
- **`compute_timeout`**, **`grace_period_seconds`**: passed into **`RequestCoalescer`** when you do not inject a custom **`coalescer`**.
- Optional DI: **`coalescer`**, **`log_analyzer`**, **`trace_analyzer`**.

The default **`Analyzer`** builds a **`LogAnalyzer`** with **`analysis_pipeline_mode=LOG_AND_TRACE`**. To use another mode, supply your own **`log_analyzer`** constructed with the desired **`analysis_pipeline_mode`**.

**Example** mapping from a **`LogAnalyzerConfig`** instance `cfg` (same file as the dataclass):

```python
Analyzer(
    allowed_root=cfg.allowed_root,
    log_sage=cfg.log_sage_execution(),
    compute_timeout=...,
    grace_period_seconds=...,
)
```

`cfg.analysis_pipeline_mode` is **not** applied unless you build **`LogAnalyzer(..., analysis_pipeline_mode=cfg.analysis_pipeline_mode)`** and pass it as **`log_analyzer=`**.

### 7.2 `LogAnalyzerConfig` (`orchestration/types.py`)

A **documentation / aggregation** dataclass: `allowed_root`, LLM defaults, `use_lib_log_analysis`, **`analysis_pipeline_mode`**. Its **`log_sage_execution()`** method returns **`LogSageExecutionConfig`** for **`LogAnalyzer`** / **`Analyzer(log_sage=...)`**. It is **not** passed wholesale into **`Analyzer`**.

### 7.3 `LogSageExecutionConfig` (`orchestration/config.py`)

The subset that **`LogSageRunner`** / **`LogAnalyzer`** actually consume for LogSage and MCP: **`use_lib_log_analysis`**, **`mcp_server_log_level`**, **`llm_model`**, **`llm_base_url`**, **`llm_temperature`**, **`llm_top_p`**, **`llm_max_tokens`**.

### 7.4 LogSage backends

**MCP (Model Context Protocol):** stdio subprocess to the MCP server in `mcp_integration/`; the client calls the `log_analyzer` tool with `log_path` and LLM parameters. No separate network server for MCP itself. Details: `mcp_integration/README.md`.

**In-process:** same LogSage logic without a subprocess; `LogAnalyzer` holds the analyzer under a lock.

---

## 8. LogSage output shape and parsing

**Parsing structured text:** `NVRxLogAnalyzer.print_output` converts each five-field LogSage cycle tuple into a structured `RawAnalysisResultItem` once. Downstream code reuses those parsed fields instead of reparsing `raw_text`.

**Typical success dict** from the log analyzer module (MCP or lib), simplified:

```text
{
  "module": "log_analyzer",
  "recommendation": {
    "action": "STOP" | "RESTART" | "CONTINUE" | "UNKNOWN" | "TIMEOUT",
    "source": "log_analyzer"
  },
  "result": [
    {
      "raw_text": "<raw_llm_text>",
      "auto_resume": "<parsed decision>",
      "auto_resume_explanation": "<parsed explanation>",
      "attribution_text": "<parsed attribution text>",
      "checkpoint_saved_flag": 0 | 1,
      "primary_issues": ["..."],
      "secondary_issues": ["..."],
      "action": "STOP" | "RESTART" | "CONTINUE" | "UNKNOWN"
    },
    ...
  ]   # per cycle
}
```

**Timeout / error** uses `"result": []` plus `recommendation`; timeout/error
payloads may also include `"state": "timeout"` and an `"error"` message for
diagnostics. Consumers should still branch on `recommendation.action`, not
`state`, for restart/stop semantics.

`log_fr_analyzer` uses the same top-level shape with
`"module": "log_fr_analyzer"`. It may also include `fr` and, when
`LOG_AND_TRACE_WITH_LLM` ran with FR data present, `llm_merged_summary`
alongside the LogSage `result` and `recommendation`.
`fr_analyzer` also includes `result` and `recommendation`; because FR is
monitor-only, its recommendation action is `UNKNOWN`.

**Flight recorder output:** FR analysis reports missing/hanging ranks for
monitoring and postprocessing (`fr_analysis`, `s_fr_*`). It does not decide
workload stop/restart policy and does not override the LogSage-derived
`recommendation.action`.

---

## 9. File size and concurrent writes

- No hard max file size in the library; very large files are read fully into memory for analysis (context window still limits effective LLM input).
- **While writer is active:** reads are single snapshot; UTF-8 with `errors='ignore'`; optional minimum size checks (`MIN_FILE_SIZE_KB` in `orchestration/config.py`).
- Splitlog timing (analyze after next file appears, etc.) interacts with **service** polling; the library reads whatever path it is given.

---

## 10. Caching and concurrency

- **Request coalescing**: One in-flight compute per cache key (file path); concurrent waiters share the same `asyncio` future.
- **Result cache**: `RequestCoalescer` stores `LogAnalysisCoalesced` with optional file `mtime`/`size` validation after a configurable grace period; optional persistence when the coalescer is configured with a cache file path.
- **`_log_analysis_lock`**: Serializes LogSage execution per process when using the lib or MCP path to avoid overlapping LLM calls on shared client state.

**Retries:** No automatic retry of failed LLM calls inside the library; failed/cached outcomes are a **service** concern for TTL and re-submission (see attrsvc spec).

---

## 11. Directory layout (current)

```
attribution/
├── ARCHITECTURE.md          ← this file
├── README.md
├── analyzer/
│   ├── engine.py            # Analyzer (coalescing, submit/analyze, delegates to LogAnalyzer)
│   └── __init__.py          # re-exports pipeline + result types for convenience
├── api_keys.py
├── base.py
├── combined_log_fr/         # CombinedLogFR + merge_log_fr_llm
├── coalescing/              # RequestCoalescer, LogAnalysisCoalesced, coalesced_cache
├── controller.py            # AttributionController boundary and config
├── log_analyzer/            # NVRxLogAnalyzer implementation and package export
├── mcp_integration/         # MCP client/server (see mcp_integration/README.md)
├── orchestration/           # LogAnalyzer, analysis_pipeline, job, splitlog, slurm_parser, types, config
├── postprocessing/          # Dataflow record, poster, Slack
├── trace_analyzer/          # FR dumps, collective analysis
└── straggler/               # Profiling utilities; separate from LogAnalyzer attribution
```

---

## 12. Further reading

| Document | Content |
|----------|---------|
| `mcp_integration/README.md` | MCP transport and server layout |
