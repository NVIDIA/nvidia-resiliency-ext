# Restart Agent Design Spec

This document is the canonical index and global system contract for the
restart agent. The detailed normative rules live in the focused specs
listed below. If two docs appear to conflict, treat this file as the routing
source and the named focused spec as authoritative for that topic.

## Canonical Docs

- `README.md`: human-readable system narrative and learning path.
- `REQUIREMENTS.md`: product use cases, latency requirements, model-selection
  requirements, non-goals, and acceptance criteria.
- `POLICY.md`: STOP/RESTART rule selection, retry budgets, evidence rules, and
  restart-history accounting.
- `RUNTIME.md`: configuration/bootstrap boundaries, stateful runtime ownership,
  history injection, and CLI/library execution.
- `SCHEMA.md`: public request/response, internal execution context, model evidence
  output, attempt records, trace records, and shared eval-boundary data shapes.
- `EVIDENCE_BUNDLE.md`: deterministic bundle generation, required bundle
  content, and LogSage-derived bundle lessons.
- `PROGRESSIVE.md`: the target, currently unimplemented, progressive lifecycle,
  retained state, finalization, latency, and validation contract.
- `PROFILE.md`: analysis-profile identity, fingerprinting, default resolution,
  runtime override precedence, deltas, and promotion.
- `TOOLS.md`: L1 read-only tool loop, tool contracts, prompt/client behavior,
  and model/provider fallback.
- `TAXONOMY.md`: canonical semantic vocabulary and structural-role meanings.
- `PATTERN_REGISTRY.md`: executable deterministic progress, failure, cascade,
  diagnostic, lifecycle, and noise detectors used by L0.
- `STATUS.md`: implementation coverage, maturity, and tracked follow-ups.
- `DECISIONS.md`: active architectural choices and consequences.
- `EVALUATION.md`: product/harness ownership, packaging, secrets, and parity
  boundary.

The eval harness is a companion developer product under
`tools/restart_agent_eval/` in a separate NVRx change chain. Its requirements,
corpus, review-panel process, and model-comparison tooling are not canonical
runtime specifications.

## Goal

The system decides whether a failed distributed training job should be restarted
immediately or held for user/human intervention.

The public action is binary:

- `STOP`: do not restart immediately; hold for user/human intervention.
- `RESTART`: restart immediately.

Default bias is `RESTART`. L1 reports semantic recovery facts; L3 reports
observational history; L4 selects a versioned retry rule and emits `STOP` only
for grounded unrecoverable workload evidence or an exhausted retry budget.
The public response contract has no user/not-user policy scores.

## Scope And Boundary

### Public Request

The caller supplies a `restart_agent_request.v1` object containing:

- `log_path`: path to one single interleaved multi-rank training log.
- `job_id`: optional job identifier.
- `cycle_id`: optional integer NVRx restart/cycle identifier. When present, it orders
  restart attempts within the same `job_id`; it is not application progress.
- `analysis_mode`: optional mode. Default is `terminal`; progressive cycle
  integrations use `progressive_start` and `progressive_end`.

Configuration, history, and evaluation metadata are deliberately absent from
the public request. `SCHEMA.md` is canonical for the exact shape.

### Internal Execution Context

After request validation, the agent combines the request with:

- an immutable `PriorAttemptView` selected by the restart-agent runtime;
- configured restart-environment assumptions; and
- configured L4 retry budgets.

The resulting immutable `AnalysisExecutionContext` is the internal input to
L0-L4. The stateful `RestartAgentRuntime` supplies history through the same
boundary for library, CLI, and future MCP entrypoints. Configuration is loaded
from `restart_agent_config.v1`; history data and evaluation labels are never
configuration fields.

The initial implementation targets one log file. Multi-file one-rank-per-file
inputs are out of scope unless they are pre-concatenated into one interleaved
file.

Missing or invalid `log_path` is a request validation error, not an analyzer
decision. `REQUIREMENTS.md` owns request-validation and unavailable-log
behavior; `PROGRESSIVE.md` owns the cycle-start exception where the parent path
exists but the target file may not yet exist.

### Runtime Boundary

`RestartAgentRuntime` is stateful orchestration around the stateless analysis
pipeline. It owns current-process history, request orchestration, model-route
concurrency, deadlines, and candidate lifecycle. Configuration parsing,
credential resolution, and concrete dependency construction happen outside it
in a configuration loader and composition root.

The composition root injects model routes, provider clients, an
`AttemptRecordStore`, and an `AttemptRecordAssembler`. The runtime receives
ready-to-use dependencies, not a configuration filename, raw JSON,
environment-variable names, or secret paths. `RUNTIME.md` is canonical for
these boundaries.

History is enabled by default and retains up to 10 attempts per exact `job_id`
and 3000 records across all jobs; product configuration may disable it or
override either bound. The MVP store is local in-memory state for one runtime
lifetime. It orders attempts by integer `cycle_id`, excludes current/future
cycles, and upserts idempotently by `(job_id, cycle_id)`.

The runtime exposes an `AttemptRecordControl` seam for library/unit tests to
seed, inspect, and clear state. The CLI may explicitly import and export a
deterministic JSON-array fixture for manual testing and scenario construction.
This is not automatic persistence, a production transfer format, or an MCP
operation. Attempt records are runtime state rather than a per-analysis public
request field.

The implemented terminal facade will be adapted to this runtime through a
follow-up. Once implemented, history state will be exercised through
library/unit tests; the CLI may explicitly import and export manual-test
fixtures. A future MCP
layer is a thin adapter over the same runtime and need not expose history
operations in the product API.
Persistent/distributed history, attrsvc integration, and the MCP transport are
outside MVP scope.

The restart agent does not own job restart execution, node drain, or scheduler
mutation.

### Out Of Scope

This design only emits binary `STOP` or `RESTART`. Non-binary operational
actions such as node drain, GPU quarantine, retry-after-infra, or scheduler
annotation are out of scope.

Provider/model failure fallback is in scope because the analyzer must always
return the external output schema. `TOOLS.md` defines the deterministic fallback
for provider timeout, provider unavailability, failed forced final calls, and
failed JSON repair calls. Broader production safe-degradation behavior outside
the analyzer, such as retry scheduling, alerting, or provider failover, is out
of scope.

### Experimental Future: Isolation

Isolation is out of MVP scope and should be treated as experimental future work.
The MVP MUST NOT emit, execute, or require non-binary actions such as node drain,
GPU quarantine, temporal isolation, or scheduler mutation.

The MVP SHOULD still preserve rank, node, and GPU evidence as structured fields
when available. That keeps the current `STOP`/`RESTART` analyzer useful for a
future isolation layer without making isolation part of the decision contract.

A future isolation extension MAY combine log-derived locality evidence with
external health signals. It MUST define its own schema, confidence threshold,
action ownership, propagation path, and eval criteria before any isolation
recommendation is acted on.

## Global Invariants

- L0 deterministic analysis builds the first evidence bundle before any LLM
  call. `EVIDENCE_BUNDLE.md` owns the detailed bundle contract.
- L0 groups repeated log shapes by normalized template, summarizes routine log
  output as structured facts, then preserves bounded original-log excerpts around
  top candidate lines. Structurally classified cascades use their stable L0
  identity for grouping so volatile temporary paths do not create hundreds of
  model-facing patterns; exact counts and bounded representatives remain visible.
- L0 parses complete traceback episodes and exposes both traceback starts and
  terminal exception lines. Cleanup/finalizer stack frames may produce a
  deterministic `teardown` cascade classification; a temporary path or
  exception type without the cleanup stack is insufficient. All other causal
  roles remain unasserted until L1 evaluates the episode.
- L0 uses generic exception/assertion structure for observed failure anchors.
  CUDA/PyTorch debugging advice is retained as `diagnostic_context` but MUST
  NOT become a failure anchor, root fingerprint, or semantic taxonomy match.
- L0 keeps terminal episodes open for later explicit scheduler, kernel, or
  runtime cause confirmation. Bare process-kill records remain cause-unknown;
  bounded confirmation representatives and excerpts are linked only when the
  log directly names the cause and no compatible progress intervenes.
- L0 represents an inherently distributed terminal mechanism, such as a
  collective timeout, as a `distributed_mechanism` incident even when only one
  reporter is observed. Additional same-epoch reports across ranks, operations,
  and process groups are grouped into that incident. A separate
  `distributed_fanout` incident requires observations from at least two
  distinct ranks. The earliest terminal report is the observed mechanism;
  later reports are fanout. L0 records progress-to-detection timing but leaves
  the initiating cause unknown unless separate direct evidence establishes it.
- L0 extracts bounded path-access facts from configured read/write/cache paths
  and failed accesses. It may report distinct `/users/<name>` namespaces and a
  failed-vs-configured-write namespace mismatch, but it records effective user
  and ownership as unverified unless explicit evidence is present. Repeated
  rank copies of one normalized terminal exception form a same-attempt
  distributed exception-fanout incident only when at least two distinct ranks
  are observed; a single-rank ordinary exception remains a failure episode
  without a distributed incident. Incident membership
  also consolidates those rank copies and subsequent structural teardown
  exceptions into one failure episode before excerpt selection.
- Model-facing registry evidence is deduplicated and explicitly provisional.
  Internal registry policy/role fields remain available for fallback and trace
  diagnostics, but MUST NOT be presented to L1 as an authoritative primary or
  policy classification.
- L0 separates observed accelerator-access mechanisms from root-cause priors.
  In particular, invalid peer-GPU memory access over NVLink is an ambiguous
  observed mechanism: fabric or remote-GPU failure is common, but invalid client
  peer access remains possible. Only corroborating hardware diagnostics promote
  it to a direct `gpu_hardware_fault` observation.
- L0 candidate-anchor selection is progress-aware. High-signal-looking warning
  patterns that are followed by compatible training/checkpoint progress are
  summarized as background normalized occurrence groups or `progressed_after` context; they
  should not be promoted to primary L1 evidence just because they appeared
  early or repeated often.
- L0 exposes deterministic job/run metadata such as explicit `world_size`,
  observed-rank lower bounds, iteration deltas, consumed-sample deltas, and
  checkpoint counts. An iteration explicitly attached to a terminal failure is
  recorded as observed position, not completed progress; it may derive phase
  and distance from a checkpoint load. L0 also records successful-runtime
  duration, replay distance from the latest checkpoint, and later-progress
  observations after fault-like events in the current log. These summarize scale, progress
  depth, and execution ordering; they do not directly decide policy or prove
  component recovery.
- L0 also builds operation/artifact comparisons from explicit start/completion
  markers. It records prior completed observations and the latest attempt
  outcome at a declared identity strength: exact physical unit, same logical
  artifact with another or unknown unit/shard, different artifact under the
  same operation, or unknown comparability. This is current-log execution
  context, distinct from L3 cross-cycle recurrence, and does not infer that an
  inner write or append succeeded merely because the parent operation once
  completed.
- L0 does not make semantic `STOP` decisions in MVP. It may return fallback
  results for nonsemantic availability cases such as a missing, unreadable, or
  empty log, but semantic `STOP` requires source-grounded L1 recovery evidence
  or an exhausted L3/L4 retry budget.
- Evidence extraction, candidate terminality, and final policy/action are
  separate stages. Extraction may nominate candidates, but it does not decide
  terminality or `STOP` / `RESTART`.
- L1 may use read-only tools for ambiguous current-log evidence, but it does not
  see attempt history in MVP.
- The L1 tool interface is declared by the analysis profile. The model can call
  only tools advertised by the analyzer client; production MUST NOT dynamically
  create arbitrary executable tools from model requests.
- The model returns structured current-log evidence only. It must not emit
  `decision_basis`, `decision`, or `evidence_coverage`. Its response separates
  the observed primary mechanism, root-cause assessment, and model recovery
  assessment; the client computes the action.
- A non-null model primary MUST include a causal role and canonical evidence
  citations. Missing required structural fields may trigger one
  bounded L1 contract-repair response; repeated structural failure is invalid L1
  output. A primary labeled `cascade` or `teardown`, or imperfect citation
  grounding, is an L2 credibility finding rather than an L1 contract failure.
- L1 evidence uses a compact closed contract: one observed primary failure,
  one root-cause assessment, one model recovery assessment, optional minimal
  related-failure role annotations, and cited evidence. L0/L2 derive
  fine class, fault outcome, locality, data-position identity, and the stable
  history fingerprint. L1 MUST NOT emit those client-owned fields or the final
  action.
- The model-visible L1 response schema and the client validator are generated
  from one executable response contract. It owns fields, enums, limits,
  confidence bounds, exact evidence support tags, and canonical
  `no_failure_observed` / `insufficient_evidence` values, including their fixed
  summary and rationale strings.
- L0B selected object ids are provenance-only unless the route advertises
  `get_evidence_objects`. Related failures are grounded diagnostic source
  references, not additional policy citations; the canonical evidence array is
  the only source of claim-support tags.
- L1 recovery assessment reports exactly two independently qualified claims:
  `failure_domain` and `retry_outlook_without_workload_change`. Each claim has
  a value, evidence status, and confidence. Root-cause
  assessment separately reports
  whether the proposed cause is established, supported but unconfirmed,
  hypothesis-only, or unknown, plus missing evidence. `workload`
  includes application, model/data/configuration, and workload-selected
  framework/library behavior. Uncertain ownership within the
  workload stack MUST NOT by itself produce an ambiguous assessment.
- External mutable resource state is different from ownership inside the
  workload stack. A workload callsite does not prove who owns a port, lock,
  path lease, or similar shared resource, nor that the state persists across a
  restart. L1 requires evidence for both domain and retry-outlook claims;
  absent that, it preserves supported, hypothetical, or unknown semantics for
  L4 while L3 independently evaluates prior-attempt recurrence.
- L0B supplies `restart_environment_context` beside the log-derived execution
  context. The default says that the workload is unchanged, the process is
  recreated after normal teardown and restart delay, and hardware allocation
  and mutable external-service state may change. L1 must reason about the next
  attempt under those transition semantics rather than assuming the failed
  process, allocation, port ownership, or service state is preserved.
- `retry_outlook_without_workload_change` asks whether the same workload may
  recover after the declared restart transition. Cross-attempt persistence is
  not an L1 claim; L3 derives it from exact job and root-fingerprint matches.
  Long-term remediation and
  preventive advice are outside this contract. A deterministic resource
  request proves repeated selection, not persistence of conflicting state.
- Generic
  CUDA asynchronous-reporting, `CUDA_LAUNCH_BLOCKING`, and
  `TORCH_USE_CUDA_DSA` advice is not evidence of a transient fault or of the
  condition named by the advice.
- L1-selected fields used by policy receive a grounding audit and
  stable client-derived identity. L0 observations may supply deterministic
  fingerprint inputs. L2 preserves the raw L1 root-cause and recovery
  assessment; it may emit audit findings, but has no separate policy-active
  semantic view.
  History identity is client-derived from observed log evidence rather than
  model vocabulary.
- L2 grounds the evidence tagged for each recovery claim. It does not infer
  cross-attempt persistence from same-attempt fanout, deterministic exception
  handling, or execution position.
- L2 derives model visibility from the exact complete initial
  `model_visible_payload` retained in the conversation trace and from returned
  tool payloads. It does not reconstruct visibility from the compact evidence
  subsection. Invalid related-failure references are findings and are omitted
  from the audited projection without altering raw L1 output.
- Canonical L1 citations are grounded only when their line/quote text was
  model-visible. A quote that merely matches an unseen source-log line is an
  audit finding and cannot support policy. Nearby line correction remains
  available when the quoted text was visible at one unique nearby source line.
- When L1 claims established infrastructure ownership or unrecoverability while
  the declared restart environment may replace the allocation or mutable
  service state, L2 records a policy-material audit finding and an unapplied
  suggestion. The raw L1 claim remains unchanged.
- L2 records same-attempt rank fanout used as cross-attempt support as an
  advisory because fanout is not recurrence by itself. The advisory does not
  rewrite either L1 claim. Current-attempt deterministic checker behavior and
  execution position/replay distance establish the current event, not survival
  of its triggering state in the next attempt.
- When L1 changes the primary anchor, the client MUST rebuild secondary and
  cascade relationships relative to that grounded primary. It MUST NOT combine
  an L1 primary with stale L0 relationship text or contradictory secondary
  policy/causal labels.
- L4 selects a versioned retry rule and budget, then computes
  `decision_basis` and final `STOP` / `RESTART` from grounded recovery facts and
  L3 observations.
- Workload-managed retry/skip behavior for bad-token or token-window failures
  is a deferred extension, not a hidden generic policy rule.
- Progressive execution follows the separate target contract in
  `PROGRESSIVE.md`: start is non-authoritative and end may produce the final
  action after combining retained state with the final log tail.
- Ordering, progress-before-fault, coverage, and history inputs used by L3/L4
  come from L0, tool-call accounting, fallback context assembly, the validated
  request, effective configuration, and runtime-selected history. They do not
  come from model-authored fields.
- Observability is a design requirement for each layer. L0A reports assembly,
  coverage, selection, progress, and fingerprint facts; Decision Evidence and
  L0B report selection/projection integrity and size; L1 reports semantic
  output, model/tool latency, retries/timeouts, tokens, contract status, and
  tool-use efficiency; L2 reports grounding, identity, citation, and advisory
  audit outcomes; L3 reports availability, per-dimension progress relations,
  and recurrence counts; L4 reports the selected rule/budget, action, basis,
  and latency.
- Error-only candidate extraction can never prove terminality by itself; terminal
  decisions require original-log context, explicit terminal evidence, or
  qualifying recurrence.
- Any filtering, deduplication, sampling, truncation, or summarization that
  affects candidate evidence must be recorded as selection/lossiness metadata in
  the trace.
- Prompt-only policy rules are not authoritative. Any rule that can change the
  final action belongs in `POLICY.md`, `TAXONOMY.md`, or deterministic client
  code, and must be covered by eval.
- L1 behavioral and policy semantics are single-sourced in the versioned system
  prompt. The dynamic user message contains only the response schema and typed,
  request-specific L0B evidence/context; it does not repeat the task, system
  policy text, advertised tool schemas, or client tool-loop limits. Provider
  tool definitions travel in the request's `tools` field. The eval harness
  compares prompt revisions using
  contract compliance, semantic accuracy, tool use, latency, and token cost.
- The static L1 prompt is generic-first: it defines causal reasoning,
  restart-transition semantics, the two recovery concepts, and
  grounding requirements. Failure-family examples belong in typed L0 evidence,
  taxonomy/registries, or separately versioned prompt experiments. A rule
  observed in one log MUST NOT be promoted into the core prompt without corpus
  evidence and an A/B evaluation showing improvement without regressions.
- A versioned analysis profile binds model, prompt version, schema version,
  policy version, retry budgets, context assembly config, declared tool
  interface, tool budget policy, reasoning/thinking mode, model-routing behavior,
  fallback behavior, and progressive decision-window settings.
- `PROFILE.md` owns profile identity, fingerprinting, default resolution, and
  profile delta semantics.
- Runtime NVRx/service decision-window configuration takes precedence over any
  profile local/eval decision-window default.
- Multi-model execution has an implemented, non-arbitrating `collect_all` mode
  and a future `priority_select` production mode. Both reuse one immutable L0
  evidence state and the same per-route result contract. `collect_all` publishes
  the deterministic fallback and returns every independently computed route
  result; it performs no preference, voting, or merging. `priority_select` adds
  deadline-aware fast-versus-enriched selection without changing a closed
  cycle's action. `REQUIREMENTS.md` owns the behavioral contract and
  `PROFILE.md` owns route/configuration semantics.
- The implemented `restart_agent_config.v1` JSON contract makes a
  route the complete `(model, endpoint, request sampling/budgets, reasoning,
  tools, reliability)` configuration rather than a model name. Shared defaults
  are resolved before per-route overrides. Credentials remain external
  environment references, while the resolved non-secret config and fingerprint
  are traced. `restart_agent.json` is the canonical file and currently
  implements the model-routing subset of the future complete L0-L4 config.
- Production and eval results are comparable only when they use the same
  analysis profile or an explicitly declared profile delta.
- Terminal analysis and progressive analysis are execution schedules for filling
  the same evidence state; they MUST feed the same schema, fingerprinting,
  history comparison, retry-budget mapping, and `STOP` / `RESTART` policy.
- The production service shape is
  `attrsvc/service adapter -> thin MCP adapter -> RestartAgentRuntime`. The
  runtime owns current-lifetime history and orchestration; the MCP adapter owns
  transport translation only. CLI and library entrypoints exercise the same
  runtime without MCP.
- Only the selected `AttemptFailureFacts.root_fingerprint` participates in
  recurrence policy. Secondary and cascade fingerprints are not recurrence keys.
- Fingerprint ownership is path-specific and deterministic. L0 creates the
  fallback-path root fingerprint. L2 creates the enriched-path root fingerprint
  after auditing the L1-selected primary against L0 evidence. L1 proposals are
  trace-only, L3 performs exact history comparison, and L4 does not construct
  identity.
- If L1 selects a wrapper summary or traceback line belonging to an L0 failure
  episode, L2 derives `root_fingerprint` from the episode's canonical causal
  terminal exception. L0 consolidates duplicate serialization, inner-cause,
  and outer-wrapper lines into that episode. This keeps history identity
  independent of which equivalent line a model selected while preserving that
  selected line as provenance.
- If the stable anchor belongs to a distributed timeout incident, L2 uses the
  incident history key. That key is invariant to rank, sequence number,
  operation type, tensor size, and which member report appeared first; those
  details remain diagnostic incident fields.
- Rank, node, and GPU locality are structured evidence fields, not recurrence
  keys. Same rank does not imply same GPU unless a rank-to-GPU mapping is
  present; cross-node recurrence is recorded for calibration but does not
  independently change the MVP retry budget.
- Previously accepted paths that become unreadable/empty, malformed model
  outputs, and provider-failed analysis paths must still return the external
  analyzer schema with a restart-biased fallback. Invalid requests are rejected
  before analysis.

## Pipeline

```text
restart_agent_request.v1 + restart_agent_config.v1 + PriorAttemptView
  -> terminal or progressive context assembly into the same evidence state
  -> L0A complete evidence assembly -> immutable L0A bundle
  -> shared DecisionEvidence
  -> AttemptRecord(progress + deterministic failure facts)
       +-> L3(current deterministic, prior deterministic) -> L4
       |     -> publish deterministic fallback
       +-> L0B initial model evidence view -> L1 semantic analysis
             -> L2 grounding, identity, and advisory audit
             -> add/replace AttemptRecord.enriched[route_id]
             -> L3(current enriched, prior deterministic) -> L4
  -> best available candidate selected before the caller-owned deadline
  -> external analyzer output: retry-policy state + decision_basis + STOP/RESTART
```

### Layer Model

The five layers are explicit trust and observability boundaries:

- **L0: Evidence Assembly And Projection.** L0 has two deterministic
  sub-stages. **L0A** reads the log and builds the complete structured evidence
  bundle: progress/checkpoint facts, normalized occurrence groups, candidate anchors,
  failure episodes, distributed failure incidents, bounded raw excerpts, job
  metadata, operation/artifact comparisons, cascades, and lossiness accounting.
  **L0B** projects that immutable bundle into the bounded, attention-efficient
  `L0ModelFacingView`, the Initial Model Evidence View supplied in the first L1
  request. L0B owns deterministic
  section selection, excerpt rendering, deduplication, truncation, and
  projection metrics; none of that work belongs to L1. Current-attempt
  execution facts include successful runtime, replay distance from the latest
  checkpoint, progress-to-terminal-detection timing, and later-progress
  observations after fault-like events in the current log. L0 does not make a semantic user-failure
  decision or infer component recovery from interleaved ordering.
  Between L0A and L0B, L0 constructs one immutable, versioned
  `DecisionEvidence` object containing the deterministic primary candidate,
  canonical observed identity, policy-relevant progress/recovery facts, and
  references back into L0A. The fallback and model branches share this exact
  object; it is not independently reconstructed by either branch.
- **L1: Semantic Analysis.** Sends the L0B model-facing view and optional
  read-only tool results to the configured model. Read-only tools may inspect
  the retained L0A evidence and source log when L0B is insufficient. L1 emits
  the model's structured primary,
  root-cause assessment, and operational claims: `failure_domain` and
  `retry_outlook_without_workload_change`, each with its own evidence status
  and confidence, plus related-failure roles and canonical citations. The exact
  raw response and interaction transcript are
  immutable observability artifacts. L1 does not produce the final action.
- **L2: Evidence Grounding and Identity.** Minimally grounds the selected L1
  failure in source evidence and derives the enriched-path history identity.
  Its functional output is an enriched `AttemptFailureFacts` block, the same
  fact shape used by the deterministic block created from Decision Evidence.
  It also audits
  citations, causal-role claims, and prohibited fields without judging whether
  the semantic conclusion is correct.
  It resolves a uniquely matching quote within a small bounded line range,
  records unresolved findings, rejects prohibited action fields at the closed
  L1 contract boundary, and derives
  stable identity fields for deterministic history comparison. Raw L1
  semantics remain immutable and available to L4 whenever the L1 response is
  structurally usable. The runtime adds or replaces the route-keyed enriched
  block in the current `AttemptRecord`; L2 does not own record storage.
  Findings carry `severity` and `policy_material`. Advisory or observational
  credibility findings remain visible without degrading the result; only
  material credibility findings degrade result quality without changing the
  L1 values used by policy.
- **L3: History Enrichment.** Accepts the current `AttemptRecord`, a selector
  for its deterministic or route-keyed enriched fact block, and the immutable
  `PriorAttemptView` selected before analysis. L3 compares the selected current
  identity against ordered prior attempts in the same job. MVP comparisons use
  the deterministic block of every prior record, even on an enriched current
  route. It emits exact-root matches,
  marker-compatible progress relations and deltas, exact failure/data/artifact
  matches, per-attempt absolute progress summaries, and consecutive
  no-observed-advance counts. The absolute summary distinguishes early failure,
  observed training/checkpoint progress, and unknown progress so L4 does not
  have to reconstruct that information from marker deltas. L3 and L4 consume
  but never mutate the current record. L3 does not choose a recurrence
  threshold, map policy, or emit an action.
- **L4: Policy Decision.** Consumes grounded L1 current-cycle semantics and L3
  history facts. It selects a versioned retry rule, applies its configured
  retry budget, and emits retry-policy state, provenance, `decision_basis`, and
  `STOP` / `RESTART`. L4 is the only layer that owns the final recommendation.
  Failure domain remains attribution evidence rather than an unconditional
  action.

The L1 operational fields are proposals for downstream policy, not an action:

| L1 output | Downstream treatment |
| --- | --- |
| Primary, root cause, citations | L2 minimally grounds the selected failure, derives stable identity, and separately audits citations without rewriting the model's semantic opinion. |
| Failure-domain claim plus retry-outlook-without-workload-change claim | Each claim has an independent value, evidence status, confidence, and grounded citation support. L4 applies the canonical rule selection in `POLICY.md`; L2 may audit the claims but cannot substitute values. |
| Per-claim confidence | Retained for observability and calibration; it is not an L4 threshold. |

Separately, L4 combines the grounded L1 claims with L3 compatible-history facts
before emitting retry-budget state and action.

L2 exposes `recovery_assessment_policy_grounded` for L4 retry-rule selection.
It is true only when the primary and recovery-supporting
citations resolve to model-visible or cited source evidence and root-cause status is
`established_by_current_log` or `supported_but_unconfirmed`. This is a
mechanical policy-eligibility check, not an L2 rewrite or endorsement of L1's
semantic conclusion. A `hypothesis_only` assessment remains observable and may
use history, but cannot select either the zero-retry or bounded-retry rule.

The functional boundaries use explicit typed inputs and outputs. L0 produces
`L0Bundle`, `DecisionEvidence`, and `L0ModelFacingView`; L1 consumes
`L1EvidenceContext` and emits provider-neutral `L1EvidenceResult`; L2 maps
`L2GroundingInput` to `L2Result`; L3 maps `HistoryEvaluationInput` to
`HistorySummary`; and L4 maps `L4PolicyInput` to `L4PolicyOutcome`.
`AttemptRecord` is the neutral runtime-owned aggregate for the current attempt
and, without conversion, for later prior-attempt views. It contains shared
deterministic progress, one required deterministic `AttemptFailureFacts` block
derived from Decision Evidence, and a route-keyed list of enriched
`AttemptFailureFacts` blocks produced through L2. It contains no L3 history
judgment or L4 policy outcome. `SCHEMA.md` owns the exact contract.
L1 receives an `L1EvidenceContext`, not the complete bundle: it contains the
immutable L0B view plus a controlled read-only tool provider bound to L0A and
the source log. A common generic result wrapper MUST NOT erase those
distinctions. Telemetry MUST remain trace data rather than a functional input
to a later stage; L1 currently combines semantic output and provider/tool
telemetry inside one provider-neutral result envelope, which may be separated
after behavior is stable.

Module ownership follows the stage boundary. `l0/` owns deterministic assembly,
DecisionEvidence, bounded projection, replay codec, and the registry. `l1/`
owns provider-neutral contracts, schema validation, prompt, read-only tools,
invocation health, and the OpenAI-compatible adapter. `l2/` owns minimal source
grounding, enriched failure-fact construction, identity, and advisory audit. `l3/`
and `l4/` own history and action policy respectively. `infrastructure/` owns log
and artifact I/O; `observability/` owns trace construction and envelope schema
identifiers. Shared immutable contracts live in `models.py`, cross-stage
identity in `identity.py`, public downstream-role assembly in `causality.py`,
and invocation envelopes in `execution.py`. `pipeline.py` is the public facade;
`preparation.py`, `single_run.py`, and `multi_route.py` own preparation,
deadline-aware orchestration, and candidate publication. `decision_pipeline.py`
composes L2/L3/L4 without reimplementing their policy.
The target runtime layer owns `AttemptRecordAssembler`, store generations,
record closure, and immutable update commits; those responsibilities do not
belong to any L0-L4 stage.

#### Runtime Dependency Boundaries

Preparation creates one immutable `LogSnapshot` through an injected
`LogSource`. L0A, L1 evidence tools, and L2 grounding reuse that snapshot.
Stages do not reopen `L0Bundle.log_path` or require local-file storage.
`LocalLogSource` is the default adapter; a stream or object-store adapter may
implement the same contract.

Configuration loading produces immutable `ModelRouteSpec` values and does not
create provider clients. The CLI composition root converts route specs into
runtime `ModelRoute` values through an `EvidenceExtractorFactory`.
The file loader delegates to the pure `parse_restart_agent_config` validator,
which receives an explicit environment mapping. `LlmEvidenceExtractor` accepts
a `CredentialProvider`, `ChatTransport`, clock, sleeper, and optional retry
transport factory. `OpenAICompatibleTransport` receives an `HttpClient`; its
default adapter alone owns `urllib`. The read-only evidence-tool factory,
executor factory, and multi-route prepared-runner factory are also injectable.
Direct construction in the CLI and public facade is composition, not a stage
dependency.

The target composition root also constructs `RestartAgentRuntime`, its
`AttemptRecordStore`, and its `AttemptRecordAssembler`. History is enabled by
default with `max_attempts_per_job=10` and `max_total_records=3000`. The
runtime owns the live state but does not parse configuration or construct the
store. Library/unit tests may
inject or seed the store directly through `AttemptRecordControl`; that
in-memory test seam is not a serialized artifact or transport contract.

Orchestration receives a `Clock` and `ExecutorFactory`. Deadline checks,
candidate timings, and route scheduling therefore have deterministic unit-test
seams without changing the production thread-based implementation. The
asynchronous L0 artifact publisher receives the same executor-factory contract,
and the live artifact writer receives a clock for deterministic lifecycle
timestamps and elapsed time. L0A itself
is decomposed into typed detection, contextualization, and bundle-assembly
steps. DecisionEvidence selection, L2 citation grounding, attempt-record
assembly, history/policy execution, and final result assembly are separately
testable transformations.

The library contract is `RestartAgent.run()` / `run_many()`. Each returns an
immutable run envelope containing the public result and the exact bundle,
model view, trace, and fallback artifacts for that invocation. The core
orchestrator stores no caller-visible last-run state. There is no mutable
compatibility adapter or alternate legacy execution path.

L1 owns provider health, timeout, truncation, parsing, required output-contract
checks, and its bounded contract-repair turn. L2 runs only when L1 produced a
structurally usable semantic response; otherwise L2 reports `not_run`, L3 uses
the available fallback identity/history path, and L4 emits the fallback policy.

#### Concurrent Deadline Candidate

Whenever L1 is scheduled, the analyzer first computes and publishes a
`deterministic_fallback` candidate from the same immutable Decision Evidence and
immutable `PriorAttemptView`. L0 supplies shared progress and deterministic
failure facts; the runtime assembles the initial `AttemptRecord` and upserts it
when eligible before running deterministic L3 and L4. The fallback branch skips L1/L2. It is
a real deadline-usable result, not a claim that L1 failed. Its provenance
records `model_contribution=pending_not_used` and
`l1_execution_status=in_flight` while the enriched branch is pending.

If structurally usable L1 output becomes available in time, the analyzer runs
L2, adds or replaces that route's enriched fact block in the same-key record,
and recomputes L3/L4 as an `l1_enriched` candidate. Both branches use the same
immutable runtime-selected `PriorAttemptView`; prior-record comparisons remain
deterministic in MVP. The service or caller selects the best available
candidate at its deadline: enriched when ready, otherwise deterministic
fallback. Unfinished route output is abandoned at the deadline and cannot
revise the closed-cycle action, published artifacts, or attempt record. Route
priority and canonical enriched-history selection remain deferred.

This is two executions of existing L3/L4 over different current-cycle evidence,
not additional semantic layers. After fallback publication, the analyzer starts
L1. The synchronous library/CLI waits only until the configured absolute
analysis deadline and returns the selected final result; `on_fallback_ready`
exposes the earlier typed `DecisionCandidate` for a progressive service. The
library callback executes synchronously, is failure-isolated, and has its
latency traced separately from L1. It MUST perform only bounded work or hand
off immediately; otherwise it delays L1 and consumes the analysis deadline.

`run()` and `run_many()` also expose `on_l0_ready`. For model-backed execution,
it fires once after L0A, Decision Evidence, and L0B are complete and before
model-route fanout. For deterministic-only execution it fires after L0A and
Decision Evidence, with `model_view=null`. The
callback receives one immutable `L0Artifacts` object; unavailable or empty logs
do not produce it. The CLI binds this boundary to one background artifact
writer, so serialization does not delay route start. When requested, the
writer atomically publishes canonical `l0_bundle.json`,
`decision_evidence.json`, and `l0_model_view.json` files while L1 is still
running. Callback/persistence timing is observability data, not L0 build time.

`collect_all` additionally exposes `on_route_complete`. The callback runs once
for each route as that route reaches a terminal execution status; it does not
wait for slower routes and callback failure cannot change route semantics. The
caller may provide exact canonical result/trace paths for every configured route
through a route-artifact manifest. The CLI writes each route trace and then its
result directly to those final paths; result existence is the completion marker
that guarantees the companion trace is already complete. The deterministic
fallback and shared L0 products likewise go directly to caller-declared paths.

`--incremental-artifact-dir` contains lifecycle control data only: an atomically
replaced status snapshot and an append-only event stream. Events reference the
canonical artifacts rather than duplicating payloads below `live/`. The
canonical batch trace is written before its result after all routes finish or
the analysis deadline expires. This is incremental publication by completed
logical artifact, not fragment streaming into one invalid partial JSON object.

The implemented local deadline is configured by `routing.timeout_seconds` and
defaults to 600 seconds from analysis start. Route-level provider timeouts are
subordinate: each HTTP request is clamped to the remaining analysis budget, and
retries, model turns, tool calls, and forced final responses cannot start after
the deadline. Worker cancellation is cooperative; orchestration returns without
waiting for an unfinished worker, while the built-in provider client unwinds at
its clamped request timeout.

#### Audit And Policy Ownership

- L2 may make **mechanical reference repairs** only. It may resolve a cited line to a
  unique nearby quote, derive a deterministic root fingerprint, and rebuild
  related-failure references. Semantic concerns are emitted separately as
  `recovery_field_audits` with the L1 value, audit support, suggested
  interpretation, reason, and `applied=false`. This includes persistence based
  only on a latest failure despite success of the exact physical unit or same
  logical artifact, while retaining shard, region, and observer uncertainty.
  Same-attempt rank fanout is recorded as an advisory. Raw L1 remains unchanged
  in the trace and remains L4's semantic input. L2 does not replace L1 with a
  competing root-cause opinion, alter model confidence, or apply its suggested
  interpretation.
  When L1 describes a checkpoint-to-failure position as completed progress but
  L0 has no completed progress marker for that interval, L2 emits the
  non-material credibility finding
  `observed_failure_position_treated_as_completed_progress`. It does not rewrite
  the L1 assessment or L4 inputs.
- L4 exclusively maps L1 recovery facts and L3 history to action. It records
  the selected rule, configured and consumed retry budget, semantic inputs,
  observed advancement, decision basis, and final action under
  `l4_policy.retry_policy`. Semantic confidence is preserved for calibration
  and display but MUST NOT become a retry threshold.

The public result assembles downstream evidence after L2 without changing any
stage's judgment. L0 structural groups retain explicit `cascade` or `teardown`
roles. L2-grounded L1 relationship rationales annotate matching groups, and a
grounded downstream line that has no L0 registry match becomes a bounded
single-event group. These events are excluded from `secondary_failures` so a
consumer cannot mistake cleanup or fanout for another root candidate. This
assembly is observational and is not an L4 input.

L2 outputs `l2_grounded_semantics`, enriched `AttemptFailureFacts`, and
`l2_audit`. Minimal source grounding and identity construction are functional:
without a grounded source line, L2 may preserve usable current-cycle semantics
but MUST leave the enriched root fingerprint null, preventing an invented line
from matching history. The audit payload contains citation findings/resolutions,
operation/artifact observations, and unapplied semantic suggestions. L3 consumes
the current `AttemptRecord`, selected fact block, and separately supplied
`PriorAttemptView`; L4 consumes the unchanged L1 semantics and L3 history output
and remains the only stage allowed to derive canonical policy or emit `STOP` /
`RESTART`.

L2 is non-blocking in MVP. Raw-source exact citations are marked `exact`.
Quotes that exactly match a model-visible bundle/tool rendering, including a
client-added truncation marker, are marked `rendered_exact`. Unique raw quote
matches within the configured nearby range are marked `nearby_resolved`;
unresolved or ungrounded citations become findings. A material finding may
degrade result quality, but a non-material advisory or credibility finding does
not. Neither kind erases the primary, root-cause assessment, or policy
assessment. Only L1 can classify output as structurally unusable. Tightening
this boundary later requires corpus evidence, an
explicit profile/version change, and eval review.

No layer may silently rewrite an earlier layer. L1 values remain visible when
L2 resolves their evidence references or derives stable identity, the L3 trace
records history inputs and matches, and the L4 trace identifies the semantic
and history inputs used for policy.

#### Experimental Family and Concrete Identity

The analyzer additionally emits a versioned, experimental `failure_identity`
for corpus review. It is observational only: MVP history and policy continue to
use `root_fingerprint`, and `failure_identity.policy_active` MUST be `false`.

- The existing family identity combines best-effort L1 `operation` and
  `mechanism` with the exception type observed at the grounded primary line.
- The existing concrete identity retains the model-associated component and
  grounded artifact-path proposal plus observed callsite, failure position, and
  stack details. Its fingerprint is preserved for comparison while vocabulary
  and model dependence are evaluated.
- The additive `client_concrete` identity is derived only from the stable
  observed identity anchor of the grounded primary's L0 failure episode and
  deterministic source-log context. If no such episode exists, it falls back
  to the grounded primary line. It contains exception
  type, normalized terminal message, traceback source file and callsite,
  bounded rank-aware stack path, conservatively extracted artifact path, and
  named failure position. L1 wording and model-specific evidence lists cannot
  alter this fingerprint.
- The L2 trace records both the model-selected primary line and the stable
  identity anchor line/reason. Equivalent precursor, wrapper-summary, and
  terminal-exception selections within one episode MUST produce the same policy-active
  `root_fingerprint` and experimental `client_concrete` fingerprint.
- Rank, node, GPU, timestamp, cycle ID, and source-log line number remain
  occurrence metadata and MUST NOT enter either identity.
- The source log path passed to the analyzer is never an artifact identity. The
  model-associated concrete identity retains a model path only when it is
  present in cited context; `client_concrete` retains a path only when one
  unique non-traceback path is tied to an explicit operation inside the
  active failure episode. Stale setup/checkpoint context is excluded once
  later workload progress establishes a different execution phase.
- Unknown fields remain null. Their absence does not trigger L1 contract repair.

All identities retain their structured fields, a readable label, completeness,
and a SHA-256 fingerprint over canonical sorted-key JSON. Eval surfaces the
model-associated identities alongside `client_concrete` agreement so the
vocabulary and matching behavior can be iterated before any production history
policy adopts them.

##### Status and Activation Criteria

`client_concrete` is the preferred experimental candidate for future exact
failure-recurrence matching. Early manual evaluation is encouraging:

- five different L1 models that selected the same grounded terminal exception
  produced divergent model-associated identities but one `client_concrete`
  fingerprint; and
- two distinct restart cycles for the same job failed at the same application
  iteration on different nodes and produced the same `client_concrete`
  fingerprint.

These observations validate model independence and locality normalization for
the tested numeric-instability case; they do not establish general accuracy.
Before `client_concrete` may affect history or policy, corpus evaluation MUST
measure false merges and false splits across Python, C++/CUDA, NCCL,
checkpoint/filesystem, numeric, configuration, and incomplete-stack failures.
It MUST also verify artifact-path and named-position behavior across software
versions and deployment roots.

A future history policy MUST NOT treat a matching client fingerprint alone as
recurrence. It must additionally require distinct cycle/attempt identity,
exact job identity, typed progress comparison, and any
available application/data position comparison. Multiple captures of one cycle
must not count as repeated attempts. Activation requires an explicit
schema/profile and policy version change; `policy_active=false` remains the MVP
default.

L1 may make one bounded contract-repair turn before emitting its stage output.
The repair request, model-call latency, tokens, and final structural status are
all L1 work. L2 emits `clean`, `resolved`, `findings`, or `not_run`.

Provider retries and contract repair are different mechanisms. Provider retries
repeat the same request after transport/service failure according to provider
configuration. Contract repair is at most one additional L1 turn triggered by
L1 output-contract parsing. The shared call ledger records a unique call id,
latency, tokens, and outcome so aggregates do not count a repair twice.

Before every model turn, L1 MUST apply the effective profile's model context
window to the complete replayed request: system/user messages, prior assistant
turns, tool results, and advertised tool schemas. The client estimates current
input use conservatively, reserves a configured safety margin, and lowers only
that turn's requested output cap when necessary. It MUST trace the declared
context window, input estimate, configured and effective output caps, safety
margin, and whether adjustment occurred. Context budgeting is transport safety;
it does not alter the L1 semantic contract or silently truncate evidence.

### Layer Runtime Metrics

The product emits measurements; the companion eval harness owns gold comparison,
quality scoring, aggregation, and promotion gates. Required runtime measurement
families are:

| Boundary | Product-emitted measurements |
| --- | --- |
| L0A | Assembly time, source bytes/lines, object counts, caps/lossiness, fallback-fingerprint availability/source/readiness. |
| Decision Evidence | Selection time, schema, primary and identity availability, referenced source lines, exact branch reuse. |
| L0B | Projection time, characters/estimated tokens, selected context, compaction/truncation, payload integrity. |
| L1 | Semantic output, first-turn usability, model/tool/repair turns, tool yield, tokens, parsing/truncation, and contract status. |
| Endpoint | Provider attempts, successes/failures, retries, timeouts, HTTP/provider errors, and failed-call time. |
| L2 | Grounding/audit time and status, citation outcomes, findings/materiality, repairs, and enriched-fingerprint readiness/source. |
| L3 | Comparable attempt counts, progress markers/relations, exact position matches, no-observed-advance streak, and unknowns. |
| L4 | Policy version, selected rule, retry budget/count/exhaustion, observed-advance handling, action/basis, and latency. |
| Candidate readiness | Fallback/enriched readiness, result provenance/usability, selection reason, deadline outcome, and terminal request-to-result latency. |

The eval harness specifications define how these measurements combine with human
gold into per-stage quality KPIs. Product documentation does not duplicate those
scoring formulas.

End-to-end latency is measured once from the relevant external start event to
the usable result; stage wall times are diagnostics and need not sum to it. For
production progressive analysis, the qualification gate is latency from
`progressive_end` to a usable L4 result plus decision-window hit rate. Total
precompute work remains a capacity/cost metric. The current manual one-log bench
does not exercise this progressive gate and reports terminal request-to-result
latency instead.

## Observability Contract

Observability is part of the analyzer contract, not an optional feature. Every
verdict-producing path MUST emit the external analyzer output and a decision
trace, including fallback paths such as log unavailable, provider failure,
malformed model output, timeout, and missed decision window.

The trace must be sufficient to explain which evidence view produced the final
action and why production/eval runs may diverge. At architecture level that
means preserving:

- analysis profile fingerprint, effective decision budget, model-routing
  outcome, and fallback path;
- result provenance and usability: whether the final result used L0
  deterministic evidence, source-grounded L1 model evidence with a separate
  credibility audit, history recurrence, or fallback behavior, and whether NVRx
  should treat the result as normal, degraded, or fallback-only;
- L1 raw semantic output and interaction transcript, preserved without client
  rewriting;
- L2 functional grounding/identity status and enriched `AttemptFailureFacts`,
  plus separate per-field audit findings, citation resolutions,
  unresolved-grounding reasons, and L1 usability;
- L0A bundle reference or content, selected candidates, candidate anchors/event
  timeline, evidence coverage, progress/checkpoint facts, and assembly
  selection/lossiness metadata;
- exact versioned Decision Evidence, including the deterministic primary and
  references back to L0A;
- exact typed L0B model-facing view, its schema version, projection metrics,
  and projection selection/truncation metadata;
- per-model and per-tool-call latency, configured budget/cap events,
  truncation, unsupported requests, errors, and whether each result affected the
  final evidence;
- optional provider-reported L1 downstream-API and proxy processing spans when
  response headers supply them. These remain distinct from client wall time,
  are not labeled model compute, and are omitted when unavailable;
- attempt-record lifecycle: deterministic creation, route-keyed enriched
  updates, same-key generation, close/deadline state, and rejected stale or
  late updates;
- L3 history inputs and outputs: current `AttemptRecord`, selected deterministic
  or enriched fact block, immutable ordered `PriorAttemptView`, typed progress
  comparisons and deltas, stronger exact-position observations, and
  streak/count facts;
- L4 policy inputs and outputs: grounded L1 recovery assessment, L3 history
  facts, selected retry rule, allowed retries, matching prior failures,
  exhaustion and observed-advance state, `decision_basis`, and final
  `STOP` / `RESTART`;
- progressive state hit/fallback behavior, retained candidate summaries, and
  missed-window outcomes.

The trace preserves primary selection by stage: L0 deterministic candidate,
raw L1 semantic primary, L2 grounded primary, and final L4 result. These fields
MUST remain distinct even when their lines or classes agree.
- references to bulky artifacts, especially the L0 evidence bundle and the
  LLM/tool interaction transcript.

`SCHEMA.md` owns the trace schema and required metrics. `EVIDENCE_BUNDLE.md`
owns bundle traceability and selection/lossiness accounting. `TOOLS.md` owns
tool-call observability. `PROFILE.md` owns profile fingerprint and profile
runtime override traceability.

External trace sink failures MUST NOT change the restart decision, but the
analyzer MUST still return the external output schema and SHOULD record the
sink failure wherever a local trace or response anomaly can preserve it.

Service mode exposes trace inspection as non-critical observability views:
`summary` for quick operator inspection and `detail` for the full trace record.
CLI mode cannot rely on service endpoints, so CLI verdict-producing runs should
support writing a local trace artifact, include a local `trace_uri`/path in the
output when one is written, and support local rendering of the same
summary/detail projections from that artifact.

The optional CLI `--summary` is a human-readable projection of result and
metric fields for interactive use; it is not a diagnostic artifact and review
harnesses SHOULD NOT persist it. Decisions and primary evidence belong to the
result, while timing, tokens, calls, retries, and handled model/provider errors
belong to structured trace telemetry. Process stderr is reserved for unexpected
warnings and failures and MAY be retained as a failure-only diagnostic log.

Model/tool interaction debugging requires a separate artifact from the compact
trace. The analyzer SHOULD write an interaction transcript file containing the
evidence bundle snapshot or reference, rendered prompts/messages, advertised
tools, raw visible model responses, parsed tool requests, tool results,
provider retries, provider errors, token exhaustion, malformed output, JSON
repair attempts, and any selection/arbitration outcomes when that future mode is
enabled. The compact trace records summary
counts, hashes, and artifact URIs; the transcript file carries the bulky
payloads after required secret redaction.

## Progressive Cycle Mode

Progressive execution is required for the production latency goal but remains a
separate, unimplemented change chain. It changes when L0 and optional model work
run, not the evidence, history, or policy semantics. `PROGRESSIVE.md` owns the
target lifecycle, retained-state, finalization, latency, and validation contract.

## Offline Calibration Boundary

Production feedback such as shadow-mode STOP outcomes is calibration data, not
runtime evidence for the decision that generated it. The companion eval harness
owns shadow-outcome labels, ingestion, measurement, and translation into
profile or policy recommendations. Product `SCHEMA.md` owns only the decision
and trace artifacts consumed by that evaluation.

## Configuration Ownership

This architecture document does not own concrete defaults. Focused specs own
defaults close to the behavior they configure:

- `POLICY.md` owns retry-rule selection, retry budgets, history counting, and
  action mapping.
- `PROFILE.md` owns analysis-profile identity, fingerprinting, default
  resolution, runtime override precedence, and profile deltas.
- `RUNTIME.md` owns configuration/bootstrap boundaries and runtime-history
  defaults, lifecycle, injection, and replay.
- `TOOLS.md` owns L1 tool, fallback, large-log, and model-call defaults.
- `REQUIREMENTS.md` and `SCHEMA.md` own NVRx decision-window semantics and
  runtime schema fields.
- The companion eval harness owns generated profile recommendations and
  measured latency gates.
- Provider deployment defaults are explicit profiles. Credential values and
  credential-file locations come from route configuration or environment; the
  library does not assume a per-user key path.

`STATUS.md` tracks work needed before promotion, including profile
qualification and progressive replay.
