# Restart Agent Requirements

This document captures product use cases and operational requirements. It is
the readable product contract; `DESIGN.md` remains the architecture map.
Focused normative details live in `RUNTIME.md`, `POLICY.md`, `SCHEMA.md`,
`EVIDENCE_BUNDLE.md`, `TOOLS.md`, `PROFILE.md`, `PROGRESSIVE.md`, and
`TAXONOMY.md`.

## Goal

Decide whether a failed distributed training attempt should restart immediately
or be held for user/human intervention.

The production target is NVRx at large LLM-training scale. Wasting restart
decision time wastes GPU allocation time, and the cost is material at current
target scales such as 12k GPUs.

## Use Cases

1. Progressive failed-cycle analysis for NVRx.
   - This is the primary production path.
   - Cycle start uses the progressive service signal; cycle end/failure uses
     the terminal service signal; result probes use the service contract below.
   - After cycle end/failure, NVRx drives repeated result probes until a result
     is available or the NVRx-owned decision window expires.
   - The analyzer should return `STOP` or `RESTART` within the configured
     post-failure decision window. The default value belongs to NVRx/service
     configuration, not the analyzer spec.
   - If the analyzer misses that window, NVRx may decide without analyzer input;
     from the analyzer contract this is a fail-open/no-STOP result.

2. Terminal-only failed-cycle analysis.
   - Required for local use, eval, replay, debugging, and fallback.
   - Cycle start may register the log using the track-only service signal.
   - Cycle end/failure uses the terminal service signal.
   - Terminal-only mode does not shift analyzer cost into the running cycle, so
     it is not sufficient by itself for the production latency goal.

3. Same-job retry history comparison.
   - Compare prior `AttemptRecord` values for the same `job_id`.
   - Use exact root-fingerprint match plus progress/checkpoint comparison.
   - The same neutral `AttemptRecord` type MUST represent the current attempt
     during analysis and that attempt as prior state in later cycles; no
     current-to-history conversion is allowed.
   - Every stored record MUST retain a typed progress summary that distinguishes
     observed training/checkpoint progress, no observed progress, and unknown
     progress. It MUST preserve comparable marker values/counts and whether the
     failure occurred before or after observed training progress.
   - Every record MUST contain one required deterministic failure-facts block
     derived from L0 and MAY contain one L2-grounded enriched block per route.
     Repeating a route replaces its prior enriched block for that record.
   - MVP L3 comparisons MUST use the selected current branch's facts against the
     deterministic block of each prior record. Enriched prior-record comparison
     and route selection remain deferred.
   - L3 MUST expose both relative cross-attempt advancement and the absolute
     progress summary of each comparable attempt to L4. It MUST NOT collapse
     these facts into only a recurrence count.
   - Do not use an LLM to decide whether history means "same issue" in MVP.
   - `RestartAgentRuntime` MUST own bounded current-process history outside the
     stateless analysis pipeline.
   - History MUST be enabled by default with `max_attempts_per_job=10` and
     `max_total_records=3000`. Configuration MAY disable history or override
     either positive integer bound.
   - The runtime `PriorAttemptView` MUST select the same `job_id`, exclude current
     and future cycles, order by integer `cycle_id`, and return at most the
     configured last `N` prior attempts.
   - Record upserts MUST be idempotent by `(job_id, cycle_id)` so replaying a
     cycle replaces its record rather than manufacturing recurrence.
   - Per-job retention MUST behave as a cycle-ordered bounded queue. Total
     retention MUST also be bounded across jobs; exceeding the total bound
     evicts the oldest internally inserted record.
   - Library/unit tests MUST be able to seed, inspect, and clear the same
     `AttemptRecordStore` used by runtime analysis.
   - MVP MUST NOT require production history persistence or restart-surviving
     state. The CLI MAY explicitly import and export JSON-array fixtures for
     manual testing; it MUST NOT treat them as automatically maintained history
     files.
   - Attempt records and prior-attempt views MUST remain outside the public
     per-analysis request.
   - Missing history means no recurrence evidence.
   - Missing `job_id` or `cycle_id` MUST disable prior-view selection and record
     upsert for that invocation without failing analysis. A prior view may be
     selected before L0 computes the current fingerprint; L3 additionally
     requires the selected branch's root fingerprint. Record upsert requires
     the deterministic L0 root fingerprint. The runtime MUST trace ineligibility
     and MUST NOT manufacture a shared `"unknown"` or zero-valued key.

4. Shadow-mode STOP evaluation.
   - Allow offline calibration by restarting despite a shadow `STOP`.
   - Measure whether later progress, checkpoint, or same-root no-progress
     outcomes support or refute the shadow decision.
   - Shadow outcomes are offline calibration data; they must not feed back into
     the same runtime decision.

## NVRx Service Contract

Exact payloads, statuses, and schema fields are specified in `SCHEMA.md`. The
requirements-level semantics are below. `PROGRESSIVE.md` owns the detailed,
currently unimplemented lifecycle and retained-state contract.

| Call | Meaning | Analyzer decision work |
| --- | --- | --- |
| `POST /logs` with `analysis_intent=track_only` | Register/track the log and optionally validate accessibility. | No `STOP`/`RESTART` work. |
| `POST /logs` with `analysis_intent=progressive` | Cycle-start signal for bounded, non-authoritative precompute. | May start progressive analysis, but must not emit a final verdict. |
| `POST /logs` with `analysis_intent=terminal` | Cycle-end/failure signal. | Uses progressive state when present; otherwise schedules terminal-only verdict-producing analysis asynchronously. |
| `GET /logs` with omitted `wait` or `wait=true` | Blocking result fetch. | May start or join analysis and wait for result or timeout. |
| `GET /logs?wait=false` | Immediate readiness/result probe. | Must not start or await analysis; returns current completed/in-flight/pending state. |

Production NVRx should use `wait=false` probes when protecting the restart
decision window. Blocking fetch remains useful for local, debugging, and
non-critical callers.

The log path is unique per NVRx cycle in MVP and is the service result identity.
`job_id` and integer `cycle_id` may be recorded as metadata when available.
`cycle_id` is useful for history and trace readability, but the service must
not require it or parse it from `log_path` as source-of-truth metadata.
Terminal/end signals use normalized `log_path` as identity and should not
repeat `job_id` or `cycle_id`; the service looks up stored metadata by
`log_path`. Accepted terminal POSTs must make the result state visible as
in-flight before returning.

Missing `log_path`, relative paths, paths outside allowed roots, and
missing/inaccessible parent directories are request validation errors. They do
not produce a completed `STOP` or `RESTART` result. During `track_only` or
`progressive` cycle-start registration, the target log file may be absent or
empty; the path minus the final file component must exist and be accessible. If
the cycle later ends without creating a readable log file, terminal analysis
returns the restart-biased `log_unavailable` fallback.

The stateful `RestartAgentRuntime` MUST own the current-lifetime attempt-record
cache.
Configuration parsing and component construction MUST remain outside the
runtime: a loader produces typed immutable configuration and a composition root
constructs routes, clients, the `AttemptRecordStore`, `AttemptRecordAssembler`,
and runtime.

The library MUST permit dependency injection and in-memory seeding of an
`AttemptRecordStore` for unit and state-based tests. The MVP CLI does not import
or export production state. It MAY seed one manual test invocation from
`--attempt-records-json-in` and write the post-analysis fixture only when
`--attempt-records-json-out` is supplied. Repeated library invocations against
one runtime MUST exercise the same stateful behavior that a future service
adapter invokes.

The future MCP layer is a thin transport adapter over the runtime. The product
MCP API need not expose record-control operations. If attrsvc later hydrates
runtime history, that design MUST reuse the typed attempt-record boundary and MUST NOT
introduce separate recurrence semantics. Persistent or distributed history and
attrsvc hydration are outside MVP scope.

Progressive state may be local in memory for the first service implementation.
It must be bounded, preserve structured evidence ahead of raw windows, and
degrade visibly to terminal analysis after state loss. The exact retention rules
and knobs are specified in `PROGRESSIVE.md` rather than repeated here.

## Latency

- NVRx policy/config owns the exact post-failure decision window and drives the
  result-probe loop. The analyzer consumes, honors, and records the configured
  value. Profiles may carry local or eval defaults, but production
  NVRx-provided values take precedence.
- Progressive start must shift expensive work into the running cycle.
- Progressive end must do only bounded finalization: read unread tail, merge
  progressive state, validate terminality/progress/checkpoint facts, apply
  history, and emit the final analyzer schema.
- The NVRx decision window starts when NVRx observes cycle end/failure and
  issues the terminal signal. The analyzer SHOULD consume a caller-provided
  deadline or remaining budget when available; otherwise, terminal request
  receipt is the local accounting anchor. `progressive_start` uses separate
  profile caps because it runs while training is still alive.
- Tool loops and model calls are optional profile components. When enabled,
  post-failure calls must be timed, traced, and accounted for against the
  caller-provided decision window. Numeric tool budgets are profile/eval
  tuning outputs, not fixed requirements in this spec.
- Whenever L1 is scheduled, the analyzer MUST start a deterministic fallback
  branch from the same immutable L0 bundle and runtime-selected
  `PriorAttemptView`. That branch runs L3 and L4 without waiting for L1 and
  becomes available as a deadline-usable candidate.
- The deterministic fallback MUST be published before the first L1 model route
  starts. Publication failure is observable but MUST NOT prevent model-route
  execution or change the fallback result.
- When record identity is eligible, the runtime MUST assemble and upsert the
  current `AttemptRecord` from L0 progress and deterministic failure facts before
  publishing the fallback. A caller can therefore consume the fallback without
  racing record creation.
- The fallback candidate MUST identify L1 as pending, not failed. If a usable
  L1/L2 result arrives in time, the runtime adds or replaces that route's
  enriched fact block and L3/L4 are recomputed as an enriched candidate. The
  immutable `PriorAttemptView` MUST NOT change between branches.
- The caller or service MUST expose/select the best candidate available at the
  caller-owned deadline: enriched when ready, otherwise deterministic fallback.
  A late enriched result MUST NOT revise an NVRx decision for a closed cycle.
- Missed-window decisions must be observable as analyzer misses.
- Direct and `collect_all` analysis MUST enforce one absolute analysis deadline
  across L0, fallback publication, model routes, retries, tool rounds, and
  provider requests. `routing.timeout_seconds` supplies the configurable local
  default (600 seconds). A provider-request timeout is a subordinate cap and
  MUST be clamped to the remaining analysis budget.
- When the analysis deadline expires, the analyzer MUST return without waiting
  for unfinished routes. It returns completed route results and the already
  published deterministic fallback for each unfinished route, marked
  `execution_status=deadline_exceeded`. No retry, model call, or tool call may
  start after the deadline.
- Late analyzer results must not change a decision for a cycle after NVRx has
  moved on. At the analysis deadline, unfinished route output is abandoned and
  MUST NOT mutate trace publication or the closed `AttemptRecord`. An
  already-running worker may
  unwind after cancellation, but no consumer accepts its output.
- Eval and load tests should report p50/p90/p99 latency, timeout rate, and
  missed-window rate by model, profile, phase, and hosting path.

## Evidence and Policy Inputs

- The analyzer MUST build a deterministic evidence bundle before any model call.
- The bundle MUST be traceable to original log lines and MUST record
  selection/lossiness metadata for filtering, dedupe, sampling, truncation,
  context windows, progressive eviction, and configured caps.
- The bundle MUST include candidate failures, candidate anchors/event timeline
  entries, normalized occurrence groups, bounded original-log excerpts for top
  candidate windows, forward-progress facts, checkpoint-save facts, setup
  milestones such as successful checkpoint load and completed graph build,
  operation/artifact comparisons, recovery/terminality hints, cascade groups,
  root fingerprint candidates, and progressive
  candidate summaries when progressive mode is used.
- Distributed incident records MUST distinguish `distributed_mechanism`, which
  may have one observed event/reporter, from `distributed_fanout`, which
  requires at least two distinct observed ranks. An ordinary failure observed
  on only one rank MUST remain a failure episode without a distributed
  incident. Episodes and incidents are parallel and may both describe one
  terminal sequence.
- Operation/artifact comparison evidence MUST distinguish prior completed observations from
  the latest started, completed, failed, or unresolved attempt. When parseable,
  it MUST retain the operation, logical artifact, physical unit or shard, data
  region, integrity marker, observer locality, progress value, source lines,
  and associated terminal incident. It MUST distinguish `exact_physical_unit`,
  `same_logical_artifact_other_or_unknown_unit`,
  `same_operation_different_artifact`, and `unknown_comparability`. Prior
  success is evidence at that declared identity strength; it is not proof that
  every sub-operation succeeded or that the current failure is transient.
- L0 MUST use generic exception/assertion structure for failure anchors and
  MUST label common CUDA/PyTorch debugging advice as diagnostic context that
  cannot become a primary candidate or root fingerprint.
- L0 MUST represent a bare process-kill record as termination with unknown
  cause. When the terminal log later contains an explicit scheduler, kernel,
  or runtime cause record, L0 MUST preserve bounded representatives as
  `cause_confirmation` evidence and associate them with the preceding terminal
  episode when no compatible progress intervenes.
- L0 MUST extract bounded path-access facts for configured read, write, and
  cache paths and for failed accesses. Path namespaces are string evidence
  only; they MUST NOT be presented as proof of effective UID, file ownership,
  mode, or ACL. Repeated copies of one exception across ranks MUST be grouped
  as one same-attempt distributed incident rather than recurrence. Repeated
  structurally classified cascade variants with volatile values MUST share one
  model-facing occurrence group while retaining aggregate counts and bounded raw
  representatives. Same-incident rank copies and structural teardown wrappers
  MUST be consolidated before excerpt selection.
- L0 context assembly MUST expose three deterministic boundaries: L0A produces
  the complete structured `L0Bundle`; a versioned `DecisionEvidence` object
  selects canonical policy-relevant facts and L0A references once for both
  product branches; and L0B produces the versioned,
  attention-efficient `L0ModelFacingView`, the Initial Model Evidence View
  supplied to L1. L1 MUST NOT
  rebuild or independently select its initial evidence from L0A.
- Optional L1 tools are only for additional raw-context inspection when L0B is
  ambiguous. They may use L0A and the source log as deterministic backing data.
- L0A, Decision Evidence, and L0B MUST NOT emit `decision_basis`, `STOP`, or
  `RESTART`; L4 owns those outputs. The public contract MUST NOT expose
  user/not-user policy scores.
- The public result MUST preserve downstream causal roles. Structurally
  identified and L2-grounded downstream events MUST appear under `cascades`
  with `causal_role=cascade` or `causal_role=teardown`; they MUST NOT be
  flattened into independent secondary failures. Grounded L1 relationship
  rationales MUST remain observable without changing primary selection or L4
  policy.
- L1 MUST report exactly two current-attempt recovery claims:
  `failure_domain` and `retry_outlook_without_workload_change`. Each claim
  MUST contain a closed value, an independent evidence status, and confidence
  from 1 to 99. L1 MUST also report one recovery rationale and claim-tagged
  evidence citations. These are semantic assessments, not a canonical policy
  class or action.
  `workload` includes application, model/data/configuration, and
  workload-selected framework/library behavior. Durable remediation and
  preventive advice are outside the current-attempt recovery contract.
- L0B MUST supply a deterministic `restart_environment_context` to L1. Its
  default states that the workload remains unchanged, the process is recreated
  after normal teardown and restart delay, and hardware allocation and mutable
  external-service state may change. These are transition capabilities, not
  evidence that recovery will occur. Callers MAY override the booleans when the
  deployment provides stronger guarantees.
- L4 MUST select a versioned retry rule and budget from the grounded L1 recovery
  claims and L3 observations. Immediate `STOP` requires both
  `failure_domain.value=workload` and
  `retry_outlook_without_workload_change.value=cannot_recover`, with each
  claim's status equal to `established_by_current_log`. L2 MUST resolve the
  primary and claim-tagged support to the source log, and root-cause status
  MUST be `established_by_current_log` or `supported_but_unconfirmed`, before
  this predicate is policy-active. A grounded `may_recover` outlook with
  `established_by_current_log` or `supported_but_unconfirmed` status selects
  the bounded retry budget. All other combinations select the general retry
  budget. Domain alone MUST NOT select `STOP`, and confidence MUST NOT act as a
  policy threshold.
- L4 MUST exhaust a retry budget only from consecutive prior attempts with the
  same job and exact root fingerprint and no observed compatible advancement.
  Unknown progress MUST NOT count as proven no advance. Observed advancement
  MUST prevent exhaustion for the current decision.
- When a model-selected primary is a wrapper summary or traceback line within
  an L0 failure episode, L2 MUST derive the policy-active `root_fingerprint`
  and experimental `client_concrete` identity from that episode's canonical
  terminal exception. The model-selected line remains the primary provenance
  anchor and MUST remain visible in the trace.
- L0 MUST derive the policy-active `root_fingerprint` for the deterministic
  fallback path. L2 MUST derive it for each enriched model path. L1 MUST NOT
  emit a fingerprint or own the history key. L3 MUST consume
  the selected path's fingerprint without creating or rewriting it.
- L0 and L2 MUST each report fingerprint availability, derivation source,
  history-readiness, and derivation latency in their trace KPIs. Eval MUST score
  fingerprint accuracy, false-merge rate, and false-split rate separately for
  L0 fallback identities and L2 enriched identities against reviewed corpus
  labels. Cross-model L2 agreement is a diagnostic, not a substitute for gold.
- L1 operational fields are policy inputs, not recommendations. L2 MUST audit
  their grounding without replacing the model's semantic opinion. It MAY emit
  a trace-visible suggested interpretation when support is weak, including
  persistence attributed only to same-attempt rank fanout or only to the latest
  failed observation when L0 shows success of the exact physical unit or same
  logical artifact. Such suggestions MUST carry `applied=false` and MUST NOT
  alter L4 inputs. Success of a different checkpoint, dataset file, or shard
  MUST be identified as weaker parent-pipeline evidence. Deterministic checker behavior and
  execution position or replay distance alone MUST NOT become affirmative
  restart-surviving persistence. L3 MUST add compatible-history facts; L4 MUST
  be the only stage that combines those inputs into a retry rule, budget state,
  `decision_basis`, and `STOP` / `RESTART`.
- Recovered model-call failures, retries, gateway timeouts, and provider errors
  MUST be exposed as degraded L1 execution with issue details. This transport
  status MUST NOT overwrite the model's semantic confidence.
- Model-route qualification MUST expose four separate views: semantic quality,
  behavioral efficiency, endpoint reliability, and their combined route
  outcome. Semantic quality is conditional on a delivered response;
  endpoint-only absence is `not_observed`. Behavioral efficiency excludes
  provider retries. Route outcome records enriched-versus-fallback
  contribution, final usability, latency basis, and deadline behavior without
  replacing the independent views with an unexplained aggregate score.
- An accelerator/runtime assertion, bounds failure, memory failure,
  communication timeout, numeric instability, or framework crash string alone
  must not determine policy without causal context or deterministic history.
- A deterministic assertion for the current observed state MUST NOT by itself
  establish that the condition persists across process teardown and restart.
  Prior success of an exact physical unit or the same logical artifact is a
  stronger audit observation than success of a different artifact. Neither
  observation changes L1 automatically; direct evidence may still establish
  that the unchanged workload cannot recover through the declared transition.
- L2 MUST record same-attempt rank fanout used as cross-attempt recurrence
  evidence as an advisory, but MUST NOT alter either policy-active L1 claim
  solely because model prose mentions fanout. Cross-cycle recurrence remains
  exclusively an L3 fact.
- Workload-managed retry/skip behavior for bad-token or token-window failures is
  a deferred extension. The MVP MUST NOT hide a workload-specific grace rule in
  the generic policy.
- `POLICY.md` owns the normative retry-rule selection, retry-budget accounting,
  no-progress comparison, and decision mapping.

## Analysis Profiles

An analysis profile is the versioned unit of deployment and comparison. It binds
the model route, prompt/schema versions, evidence-bundle config, declared tool
interface and budget policy, reasoning/thinking mode, decision-window
defaults/caps for local or eval runs, policy version, and retry budgets.
`PROFILE.md` owns the exact profile identity, fingerprinting, default
resolution, runtime override precedence, delta, and promotion contract.

- Production and eval MUST use the same profile when comparing effectiveness.
- Terminal analysis and progressive analysis are execution schedules for
  filling the same evidence state; they are not separate semantic schemes.
- Existing production logs SHOULD be evaluated with progressive replay when the
  goal is production-fidelity measurement. Terminal static replay remains useful
  for corpus coverage, local debugging, and fallback validation.
- The profile MUST declare whether tools are enabled and MUST support a
  per-tool advertisement boolean. The default advertises `overview`,
  `grep_log`, and `read_window`, while implemented object lookup
  `get_evidence_objects` defaults to not advertised. The profile also declares
  tool budget policy when configured, reasoning/thinking mode, and model routes.
  Priority and winner selection belong only to the future `priority_select`
  mode.
- Production MUST NOT dynamically create arbitrary executable tools from model
  requests.
- Every production trace and eval result MUST record the analysis profile
  fingerprint and SHOULD record the human-readable profile id/version.

## Export Compliance Deployment Requirement

Workloads handling export-controlled data, including chip-design data, IP, or
regulated research, MUST use the Regulated Inference Hub with an approved
ECCN-compliant inference route and credentials authorized for that workload.
This requirement applies to production analysis, local review, replay, and eval
runs whenever model-facing context may contain export-controlled data.

The workload owner or deployment operator owns this configuration. The model
id, inference-hub base URL, and API-key source are configurable, so the analyzer
does not determine whether log content is export-controlled, validate user
entitlement, or certify that a route is compliant. It MUST NOT infer compliance
from an `eccn` substring in a model name or endpoint URL.

For an export-compliance workload, every configured model candidate and network
fallback MUST be approved for the Regulated Inference Hub. If those routes are
unavailable, the analyzer may use an approved no-model deterministic fallback,
but it MUST NOT send the workload context to an unapproved inference endpoint.

Traces and eval reports SHOULD record non-secret route/hub identity and any
operator-supplied compliance classification needed for audit. API keys, auth
headers, bearer tokens, and other credentials MUST NOT appear in profiles,
prompts, traces, or reports. Recorded route metadata is observability evidence;
it is not an analyzer-issued compliance certification.

## Model, Tool, and Routing Requirements

- The model should primarily consume the L0 evidence bundle and bounded
  original-log excerpt windows.
- The model must return structured current-log evidence only, cite valid
  evidence lines from the supplied context, and distinguish root cause from
  cascade.
- Malformed, unsupported, schema-invalid, or mechanically contradicted model
  output MUST NOT become policy-active. A future selection mode MUST NOT choose
  such output as its winner.
- Tool calling is an optional context-access mechanism inside a profile, not a
  separate decision scheme.
- Tool calls are an optimization signal. A tool call that retrieves context
  already available in the L0 bundle should be treated as evidence that the
  bundle or prompt view can be improved, not as a model failure by itself.
- The MVP production generic log-inspection tool set is `overview`, `grep_log`,
  and `read_window`. Tools should remain read-only, generic, and
  failure-agnostic.
- Unsupported, unknown, or invalid tool-call requests MUST be rejected and
  recorded.
- Reasoning/thinking mode is profile-owned. Production and eval must request
  comparable modes for comparable runs, and traces must record requested and
  resolved behavior.
- Provider-specific reasoning controls, such as Qwen `enable_thinking`, are
  eval-harness owned profile-generation details. Production applies the selected
  profile; it does not discover provider capability mappings at runtime.
- A production-critical model route must have a predictable latency envelope
  under representative load.
- Prefer endpoints where NVIDIA controls capacity or has an operational path to
  add capacity. Qwen and Nemotron are locally hosted through the company
  inference hub, and capacity can be added on request.
- Frontier models may be used for eval, offline review, optional slow-path
  analysis, or bounded opportunistic production routing. They are not the
  preferred sole critical-path dependency when hosting or capacity ownership is
  unknown.
- A `collect_all` multi-model profile MUST accept N model/endpoint routes,
  construct L0A, Decision Evidence, and L0B once, query the routes concurrently,
  and return one independent L1-L4 result per route. Routes receive the same
  immutable evidence and `PriorAttemptView`. One failed route MUST NOT fail the
  other routes or the batch. `collect_all` performs no preference, voting,
  semantic merging, or winner selection.
- Each route MUST declare a unique route id, model, endpoint, and external
  credential reference. The profile and trace MUST NOT contain key values.
- The implemented model-route config MUST be versioned and MUST allow shared
  defaults plus per-route request, reasoning, tool, and reliability settings.
  Loading MUST resolve and trace the complete non-secret effective profile and
  a stable fingerprint. A supplied profile is authoritative; single-model CLI
  defaults MUST NOT silently mutate it.
- A future `priority_select` profile may add an ordered list and select the
  highest-priority valid response received before the configured cutoff. Lower
  priority responses cannot win before the cutoff unless all higher-priority
  candidates are already resolved invalid, failed, or timed out.
- `priority_select` MUST support a fast-plus-enriched production profile. A
  fast route is configured to produce a usable result with a predictable low
  latency, while a higher-priority heavyweight route may use deeper reasoning
  or tools. If the heavyweight route is valid before the caller deadline it is
  selected; otherwise the best valid fast result available at the deadline is
  selected. The deterministic fallback remains available when neither route is
  usable.
- Route-selected canonical history and use of enriched prior-record blocks are
  deferred with `priority_select`. The MVP `collect_all` runtime stores the
  required deterministic block and each completed route's enriched block, but
  prior comparisons use only deterministic blocks; model completion order
  therefore cannot change recurrence policy.
- If no valid model response arrives in time, the analyzer uses the profile's
  deterministic or hosted fallback behavior and records the miss.
- The terminal MVP MUST enforce per-analysis route concurrency, per-request
  timeouts/retries, and an absolute whole-analysis deadline with a ready
  deterministic fallback. Production qualification additionally requires a
  shared provider concurrency limiter and stateful circuit-breaker behavior so
  saturation across invocations does not consume the NVRx decision window.
- Production model-routing metadata SHOULD record whether a response came from
  owned hosted capacity, a company-managed inference-hub deployment, or a
  forwarded external provider. Capacity ownership is not inferred by the
  terminal MVP.

## Observability

Production traces and eval reports MUST make these outcomes explainable:

- final decision, retry-policy rule/budget state, decision basis, cited
  evidence, and evidence coverage;
- result provenance: whether the final recommendation was driven by L0
  deterministic evidence, source-grounded L1 model evidence with a separate
  credibility audit, history recurrence, or fallback behavior;
- result usability for NVRx: whether the completed result is a normal
  verdict-producing result, a degraded restart-biased result, or fallback-only
  service/default guidance;
- configured decision window, profile fingerprint, profile id/version when
  available, policy version, and schema versions;
- progressive-vs-terminal execution path and missed-window outcomes;
- deterministic-fallback and enriched candidate readiness times, candidate
  provenance, selected candidate/reason, and L1 state at fallback publication;
- model route, latency, timeout/error status, malformed-output status, and
  response-used status; future selection modes additionally record priority and
  arbitration outcome;
- optional provider-reported downstream LLM API and proxy pre/post/message-copy
  timing spans for each L1 call when response headers supply them. The
  downstream span MUST NOT be labeled model compute, missing spans MUST NOT be
  inferred, and these timings MUST NOT be attributed to L2;
- requested/resolved reasoning mode when a model call is made;
- per-tool-call latency, cap hits, unsupported requests, and errors for
  tool-enabled profiles;
- artifact references for the evidence bundle and detailed LLM/tool interaction
  transcript;
- terminal-vs-progressive or production-vs-eval divergence.

Service deployments MUST expose trace `summary` and `detail` views for retained
completed decisions by `decision_id`. These views are for operators and
debugging; they must not start or recompute analysis. CLI/local runs have no
service view endpoint, so they MUST support writing a local trace artifact,
return a local `trace_uri` or path in the external output when one is written,
and provide a local way to print summary/detail views from that artifact.

When a model or tool is used, the analyzer MUST write a detailed interaction
transcript artifact unless explicitly disabled in a non-production debug
profile. The transcript records the evidence bundle snapshot/reference,
rendered prompts/messages, advertised tools, model responses, tool requests,
tool results, retries, provider errors, token/finish reasons, malformed output,
JSON repair, schema validation, and arbitration. It is stored out-of-band from
the compact trace because it can be large.

## Non-Goals

- Node drain, GPU quarantine, or scheduler mutation.
- Full repair advice for the user workload.
- Depending on a model to compare prior `AttemptRecord` values.
- Depending on unbounded or mandatory model-driven tool loops after cycle
  failure.

## Acceptance Criteria

- Explicit infrastructure-domain evidence uses the general or bounded retry
  rule; it does not stop solely because of domain.
- A source-grounded current-attempt failure maps to immediate `STOP` only when
  both `failure_domain=workload` and
  `retry_outlook_without_workload_change=cannot_recover` are
  `established_by_current_log`, and the L2 root-cause/support eligibility checks
  in `POLICY.md` pass. Cross-attempt persistence is an L3 fact, not an L1 field.
- Ambiguous accelerator/runtime assertion, bounds, memory, communication,
  numeric-instability, or framework-crash symptoms do not produce `STOP` by
  themselves.
- L3 reports same-root history and typed observed-progress relations without
  choosing an action. L4 produces `STOP` when the selected versioned retry
  budget is exhausted by consecutive same-root, no-advance prior attempts.
- When L1 is blocked, a deterministic L0/history/L4 candidate becomes available
  without waiting for the model; an L4-qualified history rule may make that
  fallback a `STOP` candidate.
- Workload-managed bad-token retry/skip policy remains explicitly deferred and
  is not treated as a generic MVP exception.
- Evidence lines cited in analyzer output exist in the supplied log/context
  view.
- Progressive end meets the configured decision window on eval cases that
  include progressive state; late analyzer results are reported as missed-window
  outcomes.
- Contract tests verify the service semantics in `NVRx Service Contract`,
  including `track_only`, `progressive`, `terminal`, blocking fetch, and
  non-blocking readiness probe behavior, in-flight transition, cycle-unique log
  path identity, and late-result handling.
- Production-critical model candidates demonstrate acceptable p50/p90/p99
  latency and timeout behavior under representative load, as determined by the
  eval/load harness for that route, profile, phase, and hosting path.
- Profiles that use frontier models opportunistically preserve the configured
  decision window with bounded deadlines and hosted or deterministic fallback.
- Tool-enabled profiles emit per-tool-call metrics sufficient to compare
  production and eval behavior.

## Tracked Follow-Up Specs

These items do not block closing this requirements document, but they must be
resolved before the corresponding implementation or production promotion work.

- Whether each frontier-model inference-hub route is hosted capacity or a
  forwarding layer before the route is used as a production-critical candidate.
- Shared progressive-state store requirements for multi-instance deployments.
- Progressive replay mechanics for already-completed production logs.
