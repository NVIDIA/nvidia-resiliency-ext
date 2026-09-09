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
- `L0B.md`: deterministic failure narrative, compact Decision Evidence view,
  bounded supporting evidence, selection accounting, and L0B KPIs.
- `L1.md`: semantic model task, prompts, route execution, usability, and L1
  measurements.
- `L2.md`: source grounding, enriched identity, non-overriding audit, and L2
  measurements.
- `L3.md`: exact-job history comparison, affected-entity relation, progress,
  and recurrence.
- `L4.md`: ordered STOP/RESTART rule selection, retry budgets, declared policy
  context, and action.
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
when a mechanically grounded L1 primary carries established
workload/nonrecoverability semantics or an applicable retry budget or job guard
is exhausted. A terminal observation may support root-independent general
retry when the initiating primary is absent, but it cannot create root identity.
The public response contract has no user/not-user policy scores.

## Scope And Boundary

### Public Request

The caller supplies a `restart_agent_request.v1` object containing:

- `log_path`: path to one single interleaved multi-rank training log.
- `job_id`: optional job identifier.
- `cycle_id`: optional integer NVRx restart/cycle identifier. When present, it orders
  restart attempts within the same `job_id`; it is not application progress.

Configuration, history, and evaluation metadata are deliberately absent from
the public request. Terminal/progressive lifecycle intent is also absent: the
attrsvc orchestration boundary owns polling and finalization, then invokes the
same mode-neutral Restart Agent core with finalized L0A evidence. `SCHEMA.md`
is canonical for the exact request shape.

### Internal Execution Context

After request validation, the agent combines the request with:

- an immutable `PriorAttemptView` selected by the restart-agent runtime;
- configured L4 retry budgets.

The resulting immutable `AnalysisExecutionContext` is the internal input to
L0-L4. The stateful `RestartAgentRuntime` supplies history through the same
boundary for library, CLI, and future MCP entrypoints. Configuration is loaded
from `restart_agent_config.v1`; history data and evaluation labels are never
configuration fields.

Restart-transition guarantees are not execution-context fields or deployment
configuration. They are immutable product semantics encoded in the static L1
prompt contract described under `L1 Semantic Analysis`.

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

History is enabled by default and bounded per exact `job_id` and across all
jobs. `CONFIGURATION.md` owns the concrete limits. The MVP store is local
in-memory state for one runtime lifetime. It orders attempts by integer
`cycle_id`, excludes current/future cycles, and upserts idempotently by
`(job_id, cycle_id)`.

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

## Cross-Stage Safety Invariants

These are non-substitution and authority rules that cross stage boundaries.
Focused L0A-L4 specs own the algorithms, examples, and exact fields; new
stage-local detail should be added there rather than extending this summary.

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
  log directly names the cause and no compatible progress intervenes. A linked
  confirmation becomes the episode identity only when the prior identity is a
  cause-unknown process termination; it remains supporting evidence when the
  episode already has a concrete initiating failure. Recovered and
  progressed-after episodes cannot supply the deterministic primary, and exact
  ties use stable evidence fields rather than collection order.
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
- L0 preserves failure observations even when the initiating event is absent.
  It may select one canonical terminal failure surface after the last durable
  progress marker, but records it as `observation_only`, not as a primary or
  root. Retry-pending, recovered, progressed-after, diagnostic, and ambiguous
  tied candidates remain in evidence without becoming the selected surface.
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
  empty log, but semantic `STOP` requires a mechanically grounded L1 primary
  with qualifying typed recovery semantics
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
  citations. Missing required structural fields may trigger one bounded,
  tools-disabled L1 final-evidence response with reason `contract_repair`;
  repeated structural failure is invalid L1 output. The same one-turn mechanism
  uses distinct reasons for tool-round exhaustion and an output-limited prior
  response. A primary labeled `cascade` or `teardown` is an L1 contract failure.
  Canonical evidence must minimally support the selected primary and root-cause
  assessment. Missing recovery support tags remain L2 credibility findings and
  do not discard an otherwise grounded primary or entity; unknown recovery
  claims are abstentions and require no positive support citation.
- L1 evidence uses a compact closed contract: one optional primary failure,
  optional non-primary observed failures with at most one selected observation,
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
- The static L1 system prompt canonically renders the typed, immutable
  `ClusterExecutionContext`. The workload has an exclusive allocation drawn
  from a homogeneous node pool; replacement preserves hardware and software
  BOM, resource capacity and limits, and storage access. It also declares
  generic compute-node, scale-up-fabric, scale-out-fabric,
  distributed-storage, and service/control dependency paths. Workload code,
  data, configuration, and workload-selected software remain unchanged;
  process state is recreated; and normal restart delay applies. A separate
  health mechanism may quarantine malfunctioning resources. A competing cause
  is relevant only when the failed operation depends on its path and the cause
  can produce the observed mechanism; exact physical-component identity is not
  required, while generic component fallibility is not evidence. This context
  describes the restart transition rather than proving cause, ownership, or
  recovery. L0B does not repeat the context.
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
  assessment; it may emit observational audit findings, but has no separate
  policy-active semantic view and cannot degrade the route.
  History identity is client-derived from observed log evidence rather than
  model vocabulary.
- L2 audits the evidence tagged for each recovery claim. Its support findings
  do not gate the typed L1 assessment passed to L4. It does not infer
  cross-attempt persistence from same-attempt fanout, deterministic exception
  handling, or execution position.
- L2 derives model visibility from the exact complete initial
  `model_visible_payload` retained in the conversation trace and from returned
  tool payloads. It does not reconstruct visibility from the compact evidence
  subsection. Invalid related-failure references are findings and are omitted
  from the audited projection without altering raw L1 output.
- Canonical L1 primary and selected-observation citations are mechanically grounded only when their line/quote text was
  model-visible. A quote that merely matches an unseen source-log line is an
  audit finding and cannot create an enriched branch. Nearby line correction remains
  available when the quoted text was visible at one unique nearby source line.
- When L1 claims established infrastructure ownership or unrecoverability while
  the product restart guarantees may replace the allocation or mutable
  service state, L2 records an observational audit finding and an unapplied
  suggestion. The raw L1 claim and L4 input remain unchanged.
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
  `decision_basis` and final `STOP` / `RESTART` from the typed L1 recovery
  assessment attached to a mechanically grounded primary or selected
  observation and L3 observations. Observation-only assessment is restricted
  to root-independent general policy unless a declared policy context matches.
- A deterministic exact artifact identity selects
  `concrete_confirmation_retry`. Its default one-retry ledger consumes only
  exact root-and-non-null-entity recurrence without observed advance. A
  qualifying L1 workload failure without full identity instead selects
  `workload_confirmation_retry`, whose one-retry ledger uses root-only
  recurrence. Neither rule treats null entities as equal, and a changed entity
  does not reset the concurrent general same-root ceiling.
- Trusted policy context supplied outside the log belongs in L4. The active
  `cuda_oom_no_retry` context applies a zero-retry product policy to a selected
  terminal CUDA OOM. The active `port_bind_confirmation_retry` context gives a
  selected terminal address-in-use bind failure one same-root confirmation
  retry without depending on its L1 domain label. The active
  `rejected_iteration_retry_then_skip` context
  matches typed current facts and uses training iteration only as a
  context-specific history key, not as an affected entity.
- Progressive execution follows `ATTRSVC_INTEGRATION.md` for service
  registration and terminal submission, and `PROGRESSIVE.md` for optional L0A
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
  policy, routing, and per-route request,
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
- Each `AttemptFailureFacts` block has exactly one identity kind: root identity
  with an optional exact-object `affected_entity`, observation-only identity,
  or no identity. A cycle preserves a shared deterministic block plus
  independent route-primary and route-observation blocks. Observation identity
  describes the visible failure surface and never enters root or concrete
  ledgers; secondary and cascade identities are not recurrence keys.
- Identity ownership is path-specific and deterministic. L0 creates the
  deterministic root/entity or observation-only identity. L2 creates the
  enriched primary and observation equivalents after grounding L1 evidence
  against model-visible evidence. L1 proposals are trace-only, L3 performs
  exact like-kind history comparison, and L4 does not construct identity.
- L1's `failure_identity` is a semantic description, not a third history
  identity. When L1 selects the same canonical incident as L0, L2 reuses the L0
  observed root identity and may add only source-grounded entity detail. When
  L1 selects a different grounded incident, L2 derives that track's observed
  identity from source evidence. L3 compares deterministic, route-primary, and
  route-observation tracks independently; it does not select among them or
  silently substitute one for another.
- L3 computes root-only, root-plus-entity, and separately labeled
  observation-only no-advance recurrence. L4 always evaluates the general same-root safety
  ceiling and concurrently evaluates any narrower budget selected by grounded
  policy or a declared policy context. A narrower budget may stop earlier
  but cannot extend the general ceiling. Root identity answers whether the
  mechanism recurred; entity identity distinguishes exact
  token/sample/window positions and checkpoint/artifact paths. Entity identity
  alone never selects a stricter rule.
- L4 composes tracks with explicit precedence: eligible grounded primary,
  grounded selected observation, deterministic facts, then none. The selected
  track consumes only its matching L3 history. Observation fallback uses
  root-independent `general_retry` and same-job progress; its fingerprint is
  never a surrogate root. Unconditional job guards apply independently.
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
       +-> L3(compare deterministic) -> L4
       |     -> publish deterministic recommendation
       +-> L0B context management
             -> deterministic failure narrative
             -> compact Decision Evidence view + selected support
             -> Initial Model Evidence View
             -> L1 semantic analysis
             -> L2 grounding, identity, and advisory audit
             -> add/replace route primary and observation tracks
             -> L3 compare all tracks with like-kind prior tracks
             -> L4 select the policy-active path
  -> publish deterministic and enriched candidates as they become ready
  -> complete or stop at the configured Restart Agent analysis timeout
  -> external analyzer output: retry-policy state + decision_basis + STOP/RESTART
```

### Layer Model

The stages are explicit trust and observability boundaries:

| Stage | Typed transformation | Authority |
| --- | --- | --- |
| L0A | `LogSnapshot -> L0Bundle + DecisionEvidence` | Complete deterministic evidence, progress observations, strict optional primary, optional selected observation, and deterministic identity. Runtime assembly derives compact attempt facts from these outputs. See `L0A.md`. |
| L0B | `L0Bundle + DecisionEvidence -> L0ModelFacingView` | Context management through a deterministic failure narrative, compact Decision Evidence projection, bounded supporting evidence, and selection/lossiness accounting. See `L0B.md`. |
| L1 | `L1EvidenceContext -> L1EvidenceResult` | Model semantic assessment, strict optional primary, optional observed failure surfaces, and provider/tool execution record. See `L1.md`. |
| L2 | `L2GroundingInput -> L2Result` | Independent grounding of route-primary/root and route-observation identities plus non-overriding audit. See `L2.md`. |
| L3 | `HistoryEvaluationInput -> CycleHistoryComparison` | Like-kind deterministic/primary/observation recurrence plus shared progress comparison. See `L3.md`. |
| L4 | `L4PolicyInput -> L4PolicyOutcome` | Evidence-path precedence, ordered retry-rule selection, budget accounting, and final action. See `L4.md`. |

The three-track history and L4 path-composition contract in this table is
implemented. `STATUS.md` owns the remaining production-qualification gaps.

No layer silently rewrites an earlier layer. L1 semantics remain visible after
L2 grounding, L3 reports observations without selecting thresholds, and L4
records the exact semantic and history inputs used for policy. L1 confidence is
calibration data, not an L4 threshold.

`AttemptRecord` is the neutral runtime-owned aggregate for the current attempt
and later immutable prior views. It contains shared L0 progress, one required
deterministic `AttemptFailureFacts` block with root, observation-only, or no
identity, and route-keyed L2 primary/observation blocks.
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

Preparation freezes one captured source boundary as an immutable
`LogSnapshot`. L0A, L1 evidence tools, and L2 grounding reuse that snapshot;
stages do not reopen the log path or construct infrastructure dependencies.

The composition root resolves configuration, credentials, model routes,
transports, clocks, executors, and history storage, then injects typed
dependencies into the runtime. `RUNTIME.md` owns construction, lifetime,
callback, and test-seam details. `CONFIGURATION.md` owns concrete fields and
defaults, while `L1.md` owns provider execution and deadline behavior.

#### Concurrent Candidate Publication

Every run publishes the deterministic L0A-to-L3/L4 candidate before model-route
enrichment. Each structurally usable route may later publish an independent
L1-enriched candidate after L2, L3, and L4. All branches use the same immutable
`PriorAttemptView`; an external caller may act on an earlier candidate without
closing the internal analysis lifecycle.

`RUNTIME.md` owns callback ordering, deadline handling, and attempt-record
updates. `SCHEMA.md` owns artifact paths, completion markers, result provenance,
and trace contracts. `CONFIGURATION.md` owns route and timeout defaults.

#### Audit And Policy Ownership

L2 may make narrow mechanical reference repairs and derive client-owned
identity, but every semantic or chronology finding is observational only. Raw
L1 remains visible and audit findings cannot degrade the route or alter policy.
Once L1 evidence is mechanically grounded, L2 publishes primary and observation
tracks independently. L3 compares both without selection. L4 uses the L1
recovery assessment with the primary track when one exists, or with the
observation track only when L1 had no primary. Selected-observation input is
limited to root-independent general policy unless a declared context matches.
See `L2.md` and `L4.md` for the normative boundaries.

Public downstream evidence assembly may annotate L0 cascade/teardown groups
with grounded L1 relationships, but that observational rendering is not an L4
input.

#### Grounded Root And Observation Identity

History uses `root_fingerprint` for an initiating failure mechanism and optional
`affected_entity` for the exact artifact or data position.
`observation_fingerprint` separately identifies one selected visible failure
surface and may coexist in another track. The two identity kinds are mutually exclusive within
one fact block but may coexist as separate tracks in one cycle. Null roots never
compare, and observation identity cannot enter root-scoped policy. L0A derives
deterministic identity; L2 derives route-primary and route-observation identity
after grounding. L3 consumes those typed values without reopening source logs
or model prose. `L0A.md`, `L2.md`, and `L3.md` own the detailed rules.

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
4. L3 compares deterministic and route-primary tracks independently with
   like-kind earlier same-root attempts; any observation track is compared in
   its separate scope. A first occurrence has no consumed history. Repeated
   terminal/unresolved occurrences with no observed advance consume the
   eventual selected path's retry budget; an unknown progress relation does not. One
   prior exact root-and-checkpoint recurrence exhausts the default
   concrete-confirmation budget. Root-independent same-job guards separately
   bound repeated no-advance and unverifiable-progress attempts.
5. L4 selects primary, observation, or deterministic evidence, pairs it with
   the matching L3 history, and maps the selected rule and consumed budget to
   `RESTART` or `STOP`.
   The result and trace keep deterministic recommendation, raw L1 assessment, L2
   grounding, L3 comparisons, and L4 policy provenance distinct.

If the initiating process termination is absent but the terminal episode shows
repeated TCPStore peer disconnects, both L0A and L1 leave the primary null. They
may select the canonical disconnect as an observation-only anchor. L2 grounds
that observation and derives an observation fingerprint, L3 keeps its
recurrence separate from root history, and L4 applies root-independent general
retry using same-job progress. The result says what was observed without
claiming why the store owner disappeared.

### Layer Runtime Metrics

The product emits measurements; the companion eval harness owns gold comparison,
quality scoring, aggregation, and promotion gates. Required runtime measurement
families are:

| Boundary | Product-emitted measurements |
| --- | --- |
| L0A | Assembly time, source bytes/lines, object counts, caps/lossiness, primary/selected-observation state, and deterministic root/observation/entity availability. |
| Decision Evidence | Selection time, schema, primary and selected-observation availability, identity kind, referenced source lines, exact branch reuse. |
| L0B | Projection time, per-section characters/estimated tokens, narrative event/reference coverage, selected context, fanout/overlap compaction, truncation, and payload integrity. |
| L1 | Semantic output, first-turn usability, model/tool/repair turns, tool yield, tokens, parsing/truncation, and contract status. |
| Endpoint | Provider attempts, successes/failures, retries, timeouts, HTTP/provider errors, and failed-call time. |
| L2 | Grounding/audit time and status, citation outcomes, observational findings, repairs, and enriched root/observation/entity readiness/source. |
| L3 | Comparable attempt counts, progress markers/relations, entity relations, root-only, root-plus-entity, observation-only, and same-job progress streaks. |
| L4 | Policy version, base and effective policy, concurrent ledger/guard counts and exhaustion, observed-advance handling, action/basis, and latency. |
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

The trace must preserve the configuration and contract identity, exact stage
inputs and outputs used by policy, deterministic and enriched provenance,
candidate readiness, timing, endpoint/tool outcomes, history comparisons, and
the final retry rule and budget state. Raw L1 output remains distinct from L2
grounding and audit; root and observation identities remain distinct at every
stage.

`SCHEMA.md` owns exact result, trace, artifact, and metric shapes. The L0A-L4
specs own stage-specific functionality, tracing, and measurements. `TOOLS.md`
owns tool-call observability, `CONFIGURATION.md` owns resolved configuration
identity, and `RUNTIME.md` owns artifact publication and lifetime.

External trace sink failures MUST NOT change the restart decision, but the
analyzer MUST still return the external output schema and SHOULD record the
sink failure wherever a local trace or response anomaly can preserve it.

Bulky model/tool transcripts remain separate from compact trace telemetry and
are referenced by artifact path after required secret redaction. Human-readable
summaries are projections, not sources of truth.

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

This architecture document does not own concrete configuration defaults.
`CONFIGURATION.md` is the complete field/default catalog; focused specs explain
the behavior those settings control:

- `L3.md` owns history comparison and recurrence counting.
- `L4.md` owns retry-rule selection, retry-budget semantics, declared policy
  context, and action mapping.
- `CONFIGURATION.md` owns fields, defaults, validation, resolution, credential
  references, effective identity, fingerprinting, and comparison semantics.
- `RUNTIME.md` owns configuration/bootstrap boundaries and runtime-history
  lifecycle, injection, and replay.
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
