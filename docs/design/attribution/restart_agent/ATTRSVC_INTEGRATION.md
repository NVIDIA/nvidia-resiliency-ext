# Attrsvc Integration

This document specifies the first production integration from NVRx through
`nvrx-attrsvc` to the Restart Agent. It is the source of truth for service
composition, request identity, configuration resolution, state ownership, and
result timing. The L0-L4 analysis contracts remain defined by the other Restart
Agent design documents.

## Scope

The first cut runs the Restart Agent directly in the attrsvc process. It does
not use MCP, LogSage, Flight Recorder analysis, or the legacy
`RequestCoalescer` cache. The existing legacy path remains available behind the
`mcp` backend value during migration.

A progressive start registers the attempt before the log file must exist.
Terminal-first execution is the default and performs no pre-end polling.
Explicit progressive enablement additionally schedules L0A-only
precomputation. Terminal submission performs bounded log drain, finalizes the
same L0A accumulator and publishes deterministic policy. Model enrichment starts
by default and may be explicitly disabled in the Restart Agent configuration.
`PROGRESSIVE.md` owns that lifecycle.

## Process Shape

```text
NVRx AttributionService
  -> attrsvc POST /logs and GET /logs
  -> AttributionHttpAdapter
  -> RestartAgentServiceBackend
  -> RestartAgentRuntime
  -> L0-L4 and AttemptRecord history
```

`AttributionHttpAdapter` selects the implementation before constructing any
legacy controller:

| `ANALYSIS_BACKEND` | Implementation |
| --- | --- |
| `lib` | Direct in-process Restart Agent; default |
| `mcp` | Existing LogSage/Flight Recorder MCP path |

For `lib`, attrsvc MUST NOT construct `AttributionController`, `Analyzer`,
`RequestCoalescer`, or an MCP client. The meaning of `lib` is intentionally
repurposed because the prior in-process LogSage path was not productized.

## Existing NVRx Lifecycle

NVRx can supply the required identity explicitly or through the existing path
convention:

1. Before workers start cycle `N`, NVRx creates the expected path
   `<prefix>_cycle<N>.log` and sends `POST /logs` with `analysis_intent=progressive`.
   The Restart Agent always registers the attempt; it performs pre-end L0A work
   only when its progressive optimization is explicitly enabled.
2. The POST includes `user` and `job_id` from the job environment.
3. When that cycle fails, NVRx sends the same path with
   `analysis_intent=terminal`.
4. During the next rendezvous, NVRx polls `GET /logs?wait=false` until attrsvc
   reports a completed result or the NVRx-owned decision deadline expires.

The progressive POST MAY arrive before the file exists. Terminal analysis also
MUST tolerate a missing or empty file and return the Restart Agent's explicit
log-unavailable result rather than failing the service request.

## Attempt Identity

The service constructs a normalized attempt identity as follows:

- `log_path`: normalized absolute path under attrsvc `ALLOWED_ROOT`;
- `job_id`: non-empty POST `job_id`, otherwise the existing path heuristic;
- `cycle_id`: explicit integer POST field when present, otherwise the
  `_cycle<N>.log` suffix parsed by the shared cycle-path helper;
- `user`: POST user, retained for service observability but not consumed by the
  stateless L0-L4 core.

The explicit cycle ID takes precedence when it differs from the path-derived
value. A path without either form has `cycle_id=None`; it MUST NOT be collapsed
to cycle zero. Such a request remains analyzable but is history-ineligible. A
literal explicit `0` or `_cycle0.log` has integer cycle ID zero and is
history-eligible when a job ID is also available.

The execution-registry key is `(job_id, cycle_id)` when both values exist.
Otherwise the normalized path is the service correlation key. Repeated
progressive and terminal POSTs for the same attempt are idempotent.

## Configuration Sources

All sources resolve to the same typed `RestartAgentConfig` and use the same
`build_restart_agent_runtime()` composition root.

```text
CLI --config file ---------------------+
library mapping or RestartAgentConfig -+-> RestartAgentConfig -> runtime
attrsvc settings/environment ----------+
```

### Attrsvc Resolution

For `ANALYSIS_BACKEND=lib`:

1. When `NVRX_ATTRSVC_RESTART_AGENT_CONFIG` is set, attrsvc loads that file.
2. Otherwise attrsvc constructs one route from its existing LLM settings and
   `LLM_API_KEY_FILE`, then passes the generated mapping through
   `parse_restart_agent_config()`.

The environment-derived route uses:

- `NVRX_ATTRSVC_LLM_MODEL`;
- `NVRX_ATTRSVC_LLM_BASE_URL`;
- optional temperature, top-p, and maximum-output-token overrides; and
- `LLM_API_KEY_FILE` through the route's `credential_ref`.

Restart Agent defaults supply history bounds, retry policy, tool advertisement,
provider retries, and other fields not represented by attrsvc settings.
Progressive polling, idle, active-state, completed-result, and terminal-drain
settings remain attrsvc service configuration; they are traced but are not
merged into `RestartAgentConfig`.

`NVRX_ATTRSVC_RESTART_AGENT_PROGRESSIVE_ENABLED` defaults to `false`. Setting it
to `true` enables pre-end Restart Agent polling only when the existing
`NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS` policy also permits explicit progressive
requests. This restart-agent-specific switch does not alter the legacy
attribution backend.

A supplied file is authoritative. Attrsvc MUST NOT overlay model or behavioral
settings onto it. Credential references in the file still resolve through the
process environment. Attrsvc MUST reject a restart-agent config path when the
selected backend is `mcp`.

The first attrsvc integration supports exactly one model route so that the
client-facing recommendation is unambiguous. CLI and library callers retain
multi-route `collect_all` support. Attrsvc route priority/arbitration is a later
extension.

## State Ownership

`RestartAgentRuntime` owns the bounded current-process `AttemptRecord` history
used by L3. Attrsvc does not transform or persist those records.

`RestartAgentServiceBackend` owns a separate bounded attempt execution registry
needed by the HTTP lifecycle:

```text
registered -> precomputing <-> idle -> finalizing -> analyzing -> completed
                                                           +-> failed
```

The registry contains active futures, the best currently available candidate,
the final result, request identity, and timing/error metadata. This is
operational workflow state, not the legacy attrsvc result cache. It MUST NOT use
`RequestCoalescer`, cache-file persistence, mtime cache validation, or legacy
LogSage result envelopes internally.

Completed service results and retained active L0A states have separate attrsvc
bounds. Old completed entries are evicted before accepting replacement
retention; progressive-state eviction never removes attempt registration.
Attrsvc process restart clears both execution state and current-lifetime
history in the MVP.

Each progressive entry uses the state machine and `ProgressiveL0State` defined
by `PROGRESSIVE.md`. One coordinator scheduler
tracks next-poll times and wakes on terminal submission; it MUST NOT dedicate a
sleeping thread to every attempt. Due reads use a bounded executor and are
serialized per attempt. Progressive-state eviction drops only the precomputed
accumulator, never registration or the ability to run terminal analysis.

## Request Semantics

### Progressive POST

The backend validates the path boundary, resolves explicit-or-inferred
identity, and registers the attempt. The parent directory must exist under
`ALLOWED_ROOT`; the file itself need not exist. It schedules an immediate
metadata check and L0A-only precomputation when bytes are available. L0B-L4 do
not run before terminal submission.

### Terminal POST

The backend resolves the same attempt and starts at most one background
analysis. The POST returns after scheduling. Duplicate terminal requests return
the existing state without starting another model call.

Before L0 finalizes the captured source boundary, the backend performs a bounded
convergence drain so log-funnel writes can finish. Growth notifications ingest
and classify only newly visible complete lines. A quiet interval may start a
speculative L0A reduction while metadata observation continues. Later growth
invalidates that checkpoint; only an exact final source-boundary match permits
reuse. Quiet convergence flushes a final unterminated line. If maximum wait
expires before convergence, an incomplete actively written tail is excluded
and its byte count is traced rather than treated as authoritative evidence.

The existing progressive accumulator reads only its unread final tail;
terminal-only execution feeds the same accumulator from byte zero. Quiet and
maximum-wait values belong to attrsvc integration settings. The drain bound is
independent of the external NVRx action deadline and does not consume the
Restart Agent model-route timeout.

### GET

`GET /logs?wait=false` never starts analysis. It returns the current state:

- `pending`: registered but terminal analysis has not started;
- `in_flight`: analysis is running, optionally with a deterministic candidate;
- `completed`: the configured route has completed, or no better result can be
  produced within the Restart Agent analysis deadline;
- `failed`: service execution failed without a valid candidate.

The top-level response retains `status`, `result`, `recommendation`, and
`candidate_recommendation`. The full Restart Agent response is stored under
`result`.

`recommendation` exposes attrsvc's currently selected result;
`candidate_recommendation` explicitly exposes its best provisional candidate.
Attrsvc publishes both fields, but the current NVRx client consumes
`recommendation` only when `status=completed` and intentionally ignores
`candidate_recommendation`. Consuming the latter requires a separate, explicit
NVRx behavior change.

## Candidate Timing

The deterministic recommendation callback publishes the first usable candidate
after L0, L3, and L4. When enrichment is enabled, L1 continues in the
background and a completed usable route publishes its own enriched candidate.

NVRx waits for `completed` and consumes only `recommendation`. If its action
deadline expires first, it preserves the existing fail-open restart behavior.
The independently published `candidate_recommendation` remains observable but
is not an NVRx decision input in this integration.

A candidate whose `result_provenance.nvrx_use` is
`fallback_to_nvrx_default` is exposed with an `UNKNOWN` action.

The NVRx action deadline is external to Restart Agent analysis. NVRx moving on
does not close the attempt inside the analyzer. An L1 result that completes
within the configured Restart Agent analysis timeout, 240 seconds by default,
still updates the current attempt, runtime history, and service-visible result,
and remains observable from attrsvc until bounded registry eviction. It does
not retroactively alter the action NVRx already took.

## NVRx Changes

NVRx keeps its current endpoint, POST body, GET query, model, base URL, key-file,
user, job-ID, and cycle-log behavior. The managed attrsvc launcher changes only
to:

- allow `lib` and `mcp` backend values;
- let an unspecified backend use attrsvc's `lib` default.

The current client parses and consumes only a completed top-level
`recommendation`. Adding `candidate_recommendation` to the client contract and
using it at the action deadline is intentionally deferred to a separate PR.

No Restart Agent configuration file is required for the managed NVRx path.

## Observability

Attrsvc health and stats for `lib` report:

- backend name and effective config fingerprint;
- registered, in-flight, completed, and failed attempt counts;
- deterministic-ready and route-complete counts;
- background execution errors;
- Restart Agent history-record count; and
- active attempt paths without model credentials or raw prompt content.

Legacy `/cache` persistence is unavailable for `lib`. `/inflight` and `/jobs`
project the execution registry. Detailed traces remain owned by Restart Agent
callbacks and optional artifact publishers rather than attrsvc reconstructing
stage behavior. After L0A finalization attrsvc drops the progressive
accumulator. Each route publishes and releases its route-local transcript/tool
state independently. A completed execution-registry entry retains only the
compact selected result and operational counters.

## Verification

The integration requires tests for:

- file and environment configuration resolution;
- `lib` default and `mcp` legacy selection;
- cycle zero, positive cycle, and missing-cycle identity;
- progressive submission before file creation;
- duplicate progressive and terminal requests;
- nonblocking pending, deterministic, completed, and failed GET responses;
- attrsvc publishing completed and candidate recommendations independently;
- NVRx ignoring in-flight candidates and consuming only a completed
  recommendation;
- missing/empty logs and log-convergence bounds;
- late route completion updating history;
- execution-registry eviction and clean shutdown; and
- proof that the `lib` path does not instantiate legacy Analyzer, MCP, LogSage,
  Flight Recorder, or RequestCoalescer objects.
