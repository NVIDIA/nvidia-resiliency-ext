# Restart Agent Design Spec

This document is the canonical index and global system contract for the
restart agent. The detailed normative rules live in the focused specs
listed below. If two docs appear to conflict, treat this file as the routing
source and the named focused spec as authoritative for that topic.

## Canonical Docs

- `README.md`: human-readable system narrative and learning path.
- `REQUIREMENTS.md`: product use cases, latency requirements, model-selection
  requirements, non-goals, and acceptance criteria.
- `L0A.md`: complete deterministic evidence assembly, Decision Evidence,
  progress, deterministic identity, and L0A KPIs.
- `L0B.md`: bounded attention-efficient model projection, selection accounting,
  and L0B KPIs.
- `L1.md`: semantic model task, prompts, route execution, usability, and L1
  measurements.
- `L2.md`: source grounding, enriched identity, non-overriding audit, and L2
  measurements.
- `L3.md`: exact-job history comparison, affected-entity relation, progress,
  and recurrence.
- `L4.md`: ordered STOP/RESTART rule selection, retry budgets, declared
  recovery capabilities, and action.
- `RUNTIME.md`: configuration/bootstrap boundaries, stateful runtime ownership,
  artifact lifetimes, history injection, and CLI/library execution.
- `SCHEMA.md`: public request/response, internal execution context, model evidence
  output, attempt records, trace records, and shared eval-boundary data shapes.
- `PROGRESSIVE.md`: the optional progressive L0 precompute lifecycle, retained
  state, finalization, latency, and parity validation. Terminal execution is the
  production default.
- `CONFIGURATION.md`: complete product configuration, credentials, resolution, validation, and
  configuration identity.
- `TOOLS.md`: L1 read-only tool interfaces, limits, advertisement, rejection
  behavior, and call observability.
- `TAXONOMY.md`: canonical distinctions among structural, semantic, history,
  and policy vocabulary.
- `PATTERN_REGISTRY.md`: active deterministic detector catalog, emitted L0
  observations, and safety constraints.
- `STATUS.md`: implementation coverage, maturity, and tracked follow-ups.
- `DECISIONS.md`: rationale, rejected alternatives, consequences, and revisit
  conditions for active architectural choices.
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
concurrency, analysis timeouts, and candidate lifecycle. Configuration parsing,
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

The CLI, library, and direct attrsvc integration use this same runtime. History
state is exercised through library/unit tests, and the CLI may explicitly
import and export manual-test fixtures. A future Restart Agent MCP layer, if
needed, is a thin adapter over the runtime and need not expose history
operations in the product API. Persistent/distributed history and
restart-surviving hydration remain outside MVP scope.

The restart agent does not own job restart execution, node drain, or scheduler
mutation.

### Out Of Scope

This design only emits binary `STOP` or `RESTART`. Non-binary operational
actions such as node drain, GPU quarantine, retry-after-infra, or scheduler
annotation are out of scope.

Provider/model failure handling is in scope because the analyzer must always
return the external output schema. `L1.md`, `L4.md`, and `RUNTIME.md` define
provider failure handling and deterministic recommendation publication. Broader
production safe-degradation behavior outside the analyzer, such as retry
scheduling, alerting, or provider failover, is out of scope.

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
  call. `L0A.md` and `L0B.md` own the detailed assembly and projection
  contracts.
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
  Internal registry policy/role fields remain available for deterministic analysis and trace
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
- L0 does not make semantic `STOP` decisions in MVP. The analyzer may return
  restart-biased results for nonsemantic availability cases such as a missing,
  unreadable, or
  empty log, but semantic `STOP` requires source-grounded L1 recovery evidence
  or an exhausted L3/L4 retry budget.
- Evidence extraction, candidate terminality, and final policy/action are
  separate stages. Extraction may nominate candidates, but it does not decide
  terminality or `STOP` / `RESTART`.
- L1 may use read-only tools for ambiguous current-log evidence, but it does not
  see attempt history in MVP.
- The L1 tool interface is declared by each model-route configuration. The
  model can call only tools advertised by the analyzer client; production MUST
  NOT dynamically create arbitrary executable tools from model requests.
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
- The static L1 system prompt declares immutable product guarantees: the
  workload is unchanged, process state is recreated, normal restart delay
  applies, and hardware allocation and mutable external-service state may
  change. L1 must reason about the next attempt under those transition
  semantics rather than assuming the failed process, allocation, port
  ownership, or service state is preserved. L0B does not repeat them.
- `retry_outlook_without_workload_change` asks whether the same workload may
  recover after the product restart transition. Cross-attempt persistence is
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
  the product restart guarantees may replace the allocation or mutable
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
- A deterministic exact artifact or data-position identity selects
  `confirmation_retry`. Its default one-retry ledger consumes only exact
  root-and-entity recurrence without observed advance; a changed entity does
  not reset the concurrent general same-root ceiling.
- Trusted product configuration may declare workload-managed recovery
  capabilities consumed only by L4. The closed MVP supports
  `bad_token_retry_then_skip`: L4 applies its two-retry budget only when the
  grounded primary has classifier `bad_token_or_window` and deterministic
  `recovery_behavior=retry_then_skip`, and the selected facts contain a
  `data_position` affected entity. Its retry accounting uses exact root and
  entity matching while the general same-root ceiling remains active. Model
  prose cannot activate it, and absence of the declaration leaves generic
  policy unchanged.
- Progressive execution follows `ATTRSVC_INTEGRATION.md` for the implemented
  registration/terminal first cut and `PROGRESSIVE.md` for later L0
  precompute: start is non-authoritative and end may produce the final action
  after combining retained state with the final log tail.
- Ordering, progress-before-fault, coverage, and history inputs used by L3/L4
  come from L0, tool-call accounting, deterministic context assembly, the validated
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
  final action belongs in `L4.md`, `TAXONOMY.md`, or deterministic client
  code, and must be covered by eval.
- L1 behavioral and policy semantics are single-sourced in the versioned system
  prompt. The user message is a machine-readable invocation envelope containing
  the static generated response schema and typed, request-specific L0B
  evidence/context. It does not repeat the task, system policy text, advertised
  tool schemas, or client tool-loop limits. Provider tool definitions travel in
  the request's `tools` field. The eval harness compares prompt revisions using
  contract compliance, semantic accuracy, tool use, latency, and token cost.
- The static L1 prompt is generic-first: it defines causal reasoning,
  restart-transition semantics, the two recovery concepts, and
  grounding requirements. Failure-family examples belong in typed L0 evidence,
  taxonomy/registries, or separately versioned prompt experiments. A rule
  observed in one log MUST NOT be promoted into the core prompt without corpus
  evidence and an A/B evaluation showing improvement without regressions.
- The versioned `restart_agent_config.v1` contract binds runtime history, retry
  policy, declared recovery capabilities, routing, and per-route request,
  reasoning, tool, and reliability settings. Prompt, schema, detector, and
  stage-algorithm versions belong to the product build rather than this config.
- `CONFIGURATION.md` owns configuration resolution, credential handling, effective
  identity, fingerprinting, and production/eval comparison semantics.
- The configured whole-analysis timeout bounds Restart Agent execution. An
  external NVRx action deadline may determine when NVRx consumes an available
  candidate, but it does not close Restart Agent analysis or history updates.
- Multi-model execution has one implemented, non-arbitrating `collect_all`
  mode. It is enabled by default and may be explicitly disabled. When enabled, it reuses one
  immutable L0 evidence state, publishes the deterministic recommendation, and
  returns every independently computed route result without
  preference, voting, or merging. Future arbitration is outside the current
  configuration contract.
- The `restart_agent_config.v1` JSON contract makes a
  route the complete `(model, endpoint, request sampling/budgets, reasoning,
  tools, reliability)` configuration rather than a model name. Shared defaults
  are resolved before per-route overrides. Credentials remain external
  environment references, while the resolved non-secret config and fingerprint
  are traced. `restart_agent.json` is the conventional filename.
- Production and eval results are comparable only when they use the same
  product revision and resolved configuration fingerprint, or explicitly
  report the differences.
- Terminal analysis and progressive analysis are execution schedules for filling
  the same evidence state; they MUST feed the same schema, fingerprinting,
  history comparison, retry-budget mapping, and `STOP` / `RESTART` policy.
- The production service shape is
  `attrsvc/service adapter -> RestartAgentRuntime`. The runtime owns
  current-lifetime history and orchestration. CLI and library entrypoints
  exercise the same runtime. `ATTRSVC_INTEGRATION.md` owns this service
  contract; any future Restart Agent MCP adapter remains optional and thin.
- Only the selected `AttemptFailureFacts` identity participates in recurrence
  policy. It contains a mechanism-level `root_fingerprint` and may contain one
  exact-object `affected_entity`; secondary and cascade identities are not
  recurrence keys.
- Identity ownership is path-specific and deterministic. L0 creates the
  deterministic root fingerprint and affected entity. L2 creates the enriched
  equivalents after auditing the L1-selected primary against L0 evidence. L1
  proposals are trace-only, L3 performs exact history comparison, and L4 does
  not construct identity.
- L3 computes root-only and root-plus-entity no-advance recurrence as
  independent observations. L4 always evaluates the general same-root safety
  ceiling and concurrently evaluates any narrower budget selected by grounded
  policy or a declared recovery capability. A narrower budget may stop earlier
  but cannot extend the general ceiling. Root identity answers whether the
  mechanism recurred; entity identity distinguishes exact
  token/sample/window positions and checkpoint/artifact paths. Entity identity
  alone never selects a stricter rule.
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
  analyzer schema with a restart-biased result. Invalid requests are rejected
  before analysis.

## Pipeline

```text
restart_agent_request.v1 + restart_agent_config.v1 + PriorAttemptView
  -> terminal or progressive context assembly into the same evidence state
  -> L0A complete evidence assembly -> immutable L0A bundle
  -> shared DecisionEvidence
  -> AttemptRecord(progress + deterministic failure facts)
       +-> L3(current deterministic, prior deterministic) -> L4
       |     -> publish deterministic recommendation
       +-> L0B initial model evidence view -> L1 semantic analysis
             -> L2 grounding, identity, and advisory audit
             -> add/replace AttemptRecord.enriched[route_id]
             -> L3(current enriched, prior deterministic) -> L4
  -> publish deterministic and enriched candidates as they become ready
  -> complete or stop at the configured Restart Agent analysis timeout
  -> external analyzer output: retry-policy state + decision_basis + STOP/RESTART
```

### Layer Model

The stages are explicit trust and observability boundaries:

| Stage | Typed transformation | Authority |
| --- | --- | --- |
| L0A | `LogSnapshot -> L0Bundle + DecisionEvidence + deterministic attempt facts` | Complete deterministic evidence, progress, primary, and deterministic identity. See `L0A.md`. |
| L0B | `L0Bundle + DecisionEvidence -> L0ModelFacingView` | Bounded attention projection and selection/lossiness accounting. See `L0B.md`. |
| L1 | `L1EvidenceContext -> L1EvidenceResult` | Model semantic assessment and provider/tool execution record. See `L1.md`. |
| L2 | `L2GroundingInput -> L2Result` | Source grounding, enriched identity, and non-overriding audit. See `L2.md`. |
| L3 | `HistoryEvaluationInput -> HistorySummary` | Exact-job recurrence, affected-entity relation, and progress comparison. See `L3.md`. |
| L4 | `L4PolicyInput -> L4PolicyOutcome` | Ordered retry-rule selection, budget accounting, and final action. See `L4.md`. |

No layer silently rewrites an earlier layer. L1 semantics remain visible after
L2 grounding, L3 reports observations without selecting thresholds, and L4
records the exact semantic and history inputs used for policy. L1 confidence is
calibration data, not an L4 threshold.

`AttemptRecord` is the neutral runtime-owned aggregate for the current attempt
and later immutable prior views. It contains shared L0 progress, one required
deterministic `AttemptFailureFacts` block, and route-keyed L2 enriched blocks.
It contains no L3 judgment or L4 outcome. `SCHEMA.md` owns exact contracts.

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
The runtime layer owns `AttemptRecordAssembler`, store generations,
record closure, and immutable update commits; those responsibilities do not
belong to any L0-L4 stage.

#### Runtime Dependency Boundaries

Preparation feeds one captured source boundary through the shared byte-chunk
reader, incremental line decoder, and L0 observation accumulator, then freezes
one immutable `LogSnapshot`. Terminal execution supplies every available chunk
without waiting; progressive execution supplies available chunks, waits for
growth, and drains the final tail at end. L0A, L1 evidence tools, and L2
grounding reuse the finalized snapshot. Stages do not reopen
`L0Bundle.log_path`. `LocalLogSource` is the default adapter; another source
adapter may implement the same captured-boundary contract.

Configuration loading produces immutable `ModelRouteSpec` values and does not
create provider clients. The CLI composition root converts route specs into
runtime `ModelRoute` values through an `EvidenceExtractorFactory`.
The file loader delegates to the pure `parse_restart_agent_config` validator,
which receives an explicit environment mapping. `LlmEvidenceExtractor` accepts
a `CredentialProvider`, `ChatTransport`, clock, sleeper, and optional retry
transport factory. `OpenAICompatibleTransport` receives an `HttpClient`; its
default adapter owns one persistent synchronous `httpx.Client` connection pool
per route. Restart Agent retains responsibility for provider retries,
deadline clamping, error classification, and trace records rather than
delegating those policies to the HTTP library. The read-only evidence-tool
factory, executor factory, and multi-route prepared-runner factory are also
injectable. Direct construction in the CLI and public facade is composition,
not a stage dependency.

The composition root also constructs `RestartAgentRuntime`, its
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
model view, trace, and deterministic artifacts for that invocation. The core
orchestrator stores no caller-visible last-run state. There is no mutable
compatibility adapter or alternate legacy execution path.

L1 owns provider health, timeout, truncation, parsing, required output-contract
checks, and its bounded contract-repair turn. L2 runs only when L1 produced a
structurally usable semantic response; otherwise L2 reports `not_run`, L3 uses
the available deterministic identity/history path, and L4 emits the deterministic policy.

#### Concurrent Candidate Publication

Every product run computes a `deterministic` candidate from immutable
Decision Evidence and
immutable `PriorAttemptView`. L0 supplies shared progress and deterministic
failure facts; the runtime assembles the initial `AttemptRecord` and upserts it
when eligible before running deterministic L3 and L4. This path does not build
L0B or run L1/L2. Its provenance records `model_contribution=not_enabled` and
`l1_execution_status=not_run`.

With the default enriched execution, the analyzer publishes that same
deterministic candidate before model routes start. While enrichment is pending,
its provenance records `model_contribution=pending_not_used` and
`l1_execution_status=in_flight`.

If structurally usable L1 output becomes available before the configured
Restart Agent analysis timeout, the analyzer runs L2, adds or replaces that
route's enriched fact block in the same-key record, and recomputes L3/L4 as an
`l1_enriched` candidate. Both branches use the same immutable runtime-selected
`PriorAttemptView`; prior-record comparisons remain deterministic in MVP.
Candidates are published as they become ready. An external caller may stop
waiting and act on an earlier candidate without closing analysis. Route priority
and canonical enriched-history selection remain deferred.

Enriched execution is a second execution of existing L3/L4 over different
current-cycle evidence, not an additional semantic layer. After deterministic
publication, the analyzer starts L1. The synchronous library/CLI waits only
until the configured analysis
timeout and returns the selected final result; `on_deterministic_ready`
exposes the earlier typed `DecisionCandidate` for a progressive service. The
library callback executes synchronously, is failure-isolated, and has its
latency traced separately from L1. It MUST perform only bounded work or hand
off immediately; otherwise it delays L1 and consumes the analysis budget.

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
recommendation and shared L0 products likewise go directly to
caller-declared paths.

`--incremental-artifact-dir` contains lifecycle control data only: an atomically
replaced status snapshot and an append-only event stream. Events reference the
canonical artifacts rather than duplicating payloads below `live/`. The
canonical batch trace is written before its result after all routes finish or
the analysis timeout expires. This is incremental publication by completed
logical artifact, not fragment streaming into one invalid partial JSON object.

The implemented Restart Agent analysis timeout is configured by
`routing.timeout_seconds` and defaults to 240 seconds from analysis start.
Internally it is represented as an absolute monotonic deadline. Route-level
provider timeouts are subordinate: each HTTP request is clamped to the remaining
analysis budget, and retries, model turns, tool calls, and forced final
responses cannot start after that boundary. Worker cancellation is cooperative;
orchestration returns without waiting for an unfinished worker, while the
built-in provider client unwinds at its clamped request timeout. This boundary
is independent of any external NVRx action deadline.

#### Audit And Policy Ownership

L2 may make narrow mechanical reference repairs and derive client-owned
identity, but its semantic suggestions are advisory and marked unapplied. Raw
L1 remains visible. L4 alone maps eligible L1 semantics and L3 observations to
retry policy and action. See `L2.md` and `L4.md` for the normative boundaries.

Public downstream evidence assembly may annotate L0 cascade/teardown groups
with grounded L1 relationships, but that observational rendering is not an L4
input.

#### Grounded History Identity

History uses `root_fingerprint` for failure mechanism and optional
`affected_entity` for the exact artifact or data position. L0A derives the
deterministic identity; L2 derives route-enriched identity after grounding. L3
consumes those typed values without reopening source logs or model prose.
`L0A.md`, `L2.md`, and `L3.md` own the detailed rules.

### End-to-End Example

For a log whose first terminal episode contains a checkpoint metadata decode
failure:

1. L0A records the normalized occurrence group, bounded source window, failure
   episode, progress/checkpoint facts, deterministic root fingerprint, and
   affected checkpoint identity. The identity uses checkpoint path plus
   iteration and an explicit shard/file/object when available; decoder-local
   buffer positions remain diagnostic only.
2. Decision Evidence selects that episode as the deterministic primary and
   references the supporting L0A objects. L0B renders the bounded model-visible
   view without changing those facts.
3. The deterministic branch immediately compares the deterministic root/entity and
   progress facts with prior exact-job attempts. In parallel, L1 may describe
   the mechanism and recovery outlook; L2 grounds its cited line and derives
   the enriched root/entity without rewriting the model assessment.
4. L3 compares the selected current branch directly with each earlier
   same-root attempt. A first occurrence has no consumed history. Repeated
   terminal/unresolved occurrences with no observed advance consume the
   selected rule's retry budget; an unknown progress relation does not. One
   prior exact root-and-checkpoint recurrence exhausts the default
   confirmation budget.
5. L4 alone maps the selected rule and consumed budget to `RESTART` or `STOP`.
   The result and trace keep deterministic recommendation, raw L1 assessment, L2
   grounding, L3 comparisons, and L4 policy provenance distinct.

### Layer Runtime Metrics

The product emits measurements; the companion eval harness owns gold comparison,
quality scoring, aggregation, and promotion gates. Required runtime measurement
families are:

| Boundary | Product-emitted measurements |
| --- | --- |
| L0A | Assembly time, source bytes/lines, object counts, caps/lossiness, deterministic root/entity availability/source/readiness. |
| Decision Evidence | Selection time, schema, primary and identity availability, referenced source lines, exact branch reuse. |
| L0B | Projection time, characters/estimated tokens, selected context, compaction/truncation, payload integrity. |
| L1 | Semantic output, first-turn usability, model/tool/repair turns, tool yield, tokens, parsing/truncation, and contract status. |
| Endpoint | Provider attempts, successes/failures, retries, timeouts, HTTP/provider errors, and failed-call time. |
| L2 | Grounding/audit time and status, citation outcomes, findings/materiality, repairs, and enriched root/entity readiness/source. |
| L3 | Comparable attempt counts, progress markers/relations, entity relations, root-only and root-plus-entity no-advance streaks, and unknowns. |
| L4 | Policy version, selected rule and history scope, retry budget/count/exhaustion, observed-advance handling, action/basis, and latency. |
| Candidate readiness | Deterministic/enriched readiness, result provenance/usability, selection reason, analysis-timeout outcome, and terminal request-to-result latency. |

The eval harness specifications define how these measurements combine with human
gold into per-stage quality KPIs. Product documentation does not duplicate those
scoring formulas.

End-to-end latency is measured once from the relevant analysis event to each
usable candidate; stage wall times are diagnostics and need not sum to it.
Progressive qualification measures `progressive_end` to deterministic recommendation
and enriched L4 results separately. Total precompute work remains a
capacity/cost metric. The current manual one-log bench does not exercise this
progressive lifecycle and reports terminal request-to-result latency instead.
Whether NVRx consumes a candidate within its own action window is an integration
measurement, not a Restart Agent correctness requirement.

## Observability Contract

Observability is part of the analyzer contract, not an optional feature. Every
verdict-producing path MUST emit the external analyzer output and a decision
trace, including deterministic paths such as log unavailable, provider failure,
malformed model output, and analysis timeout.

The trace must be sufficient to explain which evidence view produced the final
action and why production/eval runs may diverge. At architecture level that
means preserving:

- resolved configuration fingerprint, product contract versions, effective
  analysis timeout, model-routing outcome, and deterministic path;
- result provenance and usability: whether the final result used L0
  deterministic evidence, source-grounded L1 model evidence with a separate
  credibility audit, history recurrence, or deterministic behavior, and whether
  NVRx should treat the result as normal, degraded, or unusable;
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
  updates, same-key generation, close/timeout state, and rejected stale or
  late updates;
- L3 history inputs and outputs: current `AttemptRecord`, selected deterministic
  or enriched fact block, immutable ordered `PriorAttemptView`, typed progress
  comparisons and deltas, stronger exact-position observations, and
  streak/count facts;
- L4 policy inputs and outputs: grounded L1 recovery assessment, L3 history
  facts, selected retry rule, allowed retries, matching prior failures,
  exhaustion and observed-advance state, `decision_basis`, and final
  `STOP` / `RESTART`;
- progressive state hit/fallback behavior, retained candidate summaries,
  terminal-equivalence outcome, and deterministic/enriched publication latency.

The trace preserves primary selection by stage: L0 deterministic candidate,
raw L1 semantic primary, L2 grounded primary, and final L4 result. These fields
MUST remain distinct even when their lines or classes agree.
- references to bulky artifacts, especially the L0 evidence bundle and the
  LLM/tool interaction transcript.

`SCHEMA.md` owns the trace schema and required metrics. `L0A.md` and `L0B.md`
own bundle traceability and selection/lossiness accounting. `TOOLS.md` owns
tool-call observability. `CONFIGURATION.md` owns resolved configuration identity and
route-setting traceability.

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

Attrsvc implements progressive registration, periodic L0A precomputation,
terminal background scheduling, log-drain convergence, finalized-L0A handoff,
and nonblocking result probes. Pre-end polling is opt-in; terminal-first
execution is the default. The switch changes when L0A runs, not the evidence,
history, or policy semantics. `PROGRESSIVE.md` owns retained state,
finalization, latency, and validation.

## Offline Calibration Boundary

Production feedback such as shadow-mode STOP outcomes is calibration data, not
runtime evidence for the decision that generated it. The companion eval harness
owns shadow-outcome labels, ingestion, measurement, and translation into
configuration or policy recommendations. Product `SCHEMA.md` owns only the decision
and trace artifacts consumed by that evaluation.

## Configuration Ownership

This architecture document does not own concrete defaults. Focused specs own
defaults close to the behavior they configure:

- `L3.md` owns history comparison and recurrence counting.
- `L4.md` owns retry-rule selection, retry budgets, capabilities, and action
  mapping.
- `CONFIGURATION.md` owns restart-agent configuration resolution, credential
  references, effective identity, fingerprinting, and comparison semantics.
- `RUNTIME.md` owns configuration/bootstrap boundaries and runtime-history
  defaults, lifecycle, injection, and replay.
- `TOOLS.md` owns fixed tool interfaces and response limits. `CONFIGURATION.md` owns
  tool advertisement, model-call settings, and route overrides.
- `REQUIREMENTS.md` and `SCHEMA.md` own Restart Agent execution semantics and
  runtime schema fields. `ATTRSVC_INTEGRATION.md` owns how NVRx consumes
  available candidates.
- The companion eval harness owns generated configuration recommendations and
  measured latency gates.
- Provider deployment defaults are explicit route configurations. Credential values and
  credential-file locations come from route configuration or environment; the
  library does not assume a per-user key path.

`STATUS.md` tracks work needed before promotion, including route
qualification and progressive replay.
