# Restart Agent Requirements

This document defines externally meaningful product obligations and production
qualification gates. `DESIGN.md` routes detailed behavior to the L0A-L4,
runtime, schema, configuration, tool, progressive, and integration
specifications.

## Goal

Given the application log for one failed distributed-training attempt, produce
an auditable recommendation to:

- `RESTART`: start another NVRx cycle; or
- `STOP`: hold the workload for user or operator intervention.

The default bias is `RESTART`. Unavailable, late, or ambiguous model analysis
must not create an unsafe `STOP`.

## Core Behavior

- **RA-01: Input and invocation.** The analyzer MUST accept one attempt log and
  MAY accept `job_id` and integer `cycle_id`. It MUST support library, CLI, and
  NVRx/attrsvc invocation. Missing attempt identity disables history without
  preventing analysis.
  **Verify:** request-validation, missing-identity, CLI/library, and attrsvc
  contract tests.

- **RA-02: Decision contract.** A completed analysis MUST return one `STOP` or
  `RESTART`, its decision basis, retry rule and budget state, primary evidence
  when available, separately labeled observed failure surfaces, provenance,
  and degradation status. A selected observation MUST NOT be represented as a
  primary or root cause.
  **Verify:** public-schema and decision-composition tests.

- **RA-03: Terminal-first production path.** Production MUST default to
  terminal analysis. Attrsvc MUST overlap incremental L0A ingestion and
  boundary-safe precomputation with its bounded post-terminal log-drain wait.
  Progressive pre-end L0A polling remains an explicit optimization that MAY be
  enabled when measured terminal latency at target scale justifies its retained
  state and polling cost. For identical finalized source bytes, enabled
  progressive and terminal execution MUST produce structurally equal canonical
  `L0Bundle`, `DecisionEvidence`, progress facts, and deterministic failure
  facts after designated timing and operational metadata are removed. Both
  schedules MUST use the same byte decoder, observation accumulator, reducers,
  and finalizer. Chunk boundaries and incomplete-line tails MUST neither
  duplicate nor omit evidence. A `single_snapshot` parity/regression mode MUST
  exercise the same path as production `chunked` ingestion.
  **Verify:** terminal-default and progressive-opt-in configuration,
  polling/state-race, canonical terminal/progressive equivalence, byte-chunk
  boundary, source-reset, log-drain, state-loss, and post-end
  deterministic/enriched latency tests.

- **RA-04: Deterministic-first and bounded enrichment.** The analyzer MUST
  produce a deterministic L0A/L3/L4 recommendation without requiring a model
  route. Model enrichment MUST be enabled by default, and configuration MUST
  allow it to be explicitly disabled. Whenever enrichment is enabled, the analyzer MUST
  publish the deterministic recommendation before model routes start. Model
  routes MUST honor the configured Restart Agent analysis timeout. An external
  caller acting on the deterministic recommendation MUST NOT
  close internal analysis; a later usable enriched result may update the current
  attempt record, history, and service-visible completed result. Runtime
  artifacts MUST be released after their last consumer: byte chunks after
  decoding, terminal drain state after L0A finalization, route-local model/tool
  state after route publication, and source access after all routes finish or
  the analysis timeout expires. A completed service entry MUST retain only its
  compact result, history, and operational metadata.
  **Verify:** blocked-provider, analysis-timeout, cancellation, late-enrichment,
  superseding-generation, deterministic-readiness, artifact-retention, and repeated
  cycle memory tests.

## Analysis And Policy

- **RA-05: Deterministic evidence.** Before any model call, L0A MUST construct
  source-traceable structured evidence for candidate failures, bounded source
  context, progress, checkpoints, recovery, termination, cascades, teardown,
  and stable client identity when available. When the initiating failure is
  absent, L0A MAY select one canonical terminal observation, but MUST preserve
  a null primary and a separate observation-only identity. L0B MUST create a
  bounded, attention-efficient initial model view without changing those facts.
  The view MUST lead with a deterministic, source-referenced failure narrative,
  followed by compact policy-relevant facts and selected support. Exhaustive
  fanout MUST be represented by exact counts and bounded samples rather than
  one model-visible member per rank. Narrative text MUST use typed facts and
  fixed templates and MUST NOT introduce semantic cause, ownership, recovery,
  history, or action claims. Neither stage may choose the action.
  **Verify:** reviewed L0 gold facts, source-reference integrity, projection
  and narrative coverage, semantic-restraint checks, fanout-compaction
  correctness, first-turn usability, and model-conditioned reread/no-new-tool
  measurements across controlled L0B profile ablations.

- **RA-06: Model semantic assessment.** L1 MUST identify the primary failure,
  semantic identity, and root-cause support when the initiating failure is
  visible. It MUST preserve visible non-primary failure surfaces independently
  and MAY select at most one non-causal observation whether or not a primary is
  present. Their presence MUST NOT invalidate an otherwise valid primary. It then independently assesses
  failure domain and whether the next normal cycle may recover without workload
  changes. It MUST return structured citations and calibration confidence. It
  MUST minimally cite the selected primary or observation and root-cause
  assessment. Missing
  recovery support tags MUST NOT invalidate an otherwise usable response, and
  an unknown recovery claim MUST be treated as an abstention rather than a
  positive claim requiring evidence. It MUST NOT compare history or choose the
  action.
  The static prompt MUST canonically render the immutable typed cluster
  execution context: exclusive allocation from a homogeneous node pool,
  invariant replacement resource envelope, generic compute/fabric/storage/
  service dependency paths, and independent faulty-resource detection and
  quarantine. A competing cause MUST be treated as relevant only when the
  failed operation depends on its path and the cause can produce the observed
  mechanism; exact physical-component identity MUST NOT be required and
  generic component fallibility MUST NOT count as evidence. Failure domain and
  retry outlook MUST remain independent, including the valid combination of
  unknown domain and `may_recover` outlook.
  **Verify:** response-contract, semantic-gold, calibration, stability, and
  malformed-output tests.

- **RA-07: Grounding and audit.** L2 MUST mechanically ground the model-selected
  primary and selected observation independently before deriving either
  enriched identity. A grounded primary may produce root identity; a grounded
  observation may produce only a separately typed observation fingerprint.
  Either or both tracks MAY be present for one route.
  Null roots MUST never compare equal. L2 MUST audit supporting evidence separately. Audit findings MUST
  remain visible and MUST NOT rewrite model semantics, degrade route usability,
  or influence L3 identity, L4 policy, or action.
  **Verify:** exact, nearby, unresolved, invented-line, and non-overriding audit
  tests.

- **RA-08: Same-job history and progress.** Runtime history MUST be bounded,
  exact-job, cycle ordered, and idempotent by `(job_id, cycle_id)`. L3 MUST
  compare deterministic, route-primary, and route-observation identities only
  with the same kind and route in prior records. It MUST report independent
  root-only and root-plus-entity no-advance recurrence counts, separately
  labeled observation-only recurrence, plus root-independent same-job
  no-advance and unknown-progress streaks. It MUST NOT choose a track or
  silently substitute deterministic history for missing enriched history.
  Observation recurrence MUST NOT be promoted to root recurrence. Multiple
  agreeing routes MUST NOT multiply one prior cycle. Entity changes MUST NOT reset the root-only count. Missing or
  unknown progress MUST NOT count as proven no progress.
  **Verify:** ordering, replacement, eviction, identity-match, and
  progress-comparison tests.

- **RA-09: Deterministic retry policy.** L4 alone MUST select the evidence path
  and retry rule,
  evaluate the general same-root ceiling and any applicable narrower retry
  budget plus both same-job guards concurrently, and emit the action. Immediate `STOP` requires
  a mechanically grounded primary with L1 values and statuses satisfying the configured nonrecoverability
  rule; otherwise `STOP` requires exhaustion of an applicable ledger or guard
  without compatible progress. A narrower rule MAY stop earlier but MUST NOT
  extend or reset the general ceiling. Domain, confidence, rank fanout, exact
  entity availability, an L2 audit finding, or an ambiguous symptom alone MUST
  NOT produce `STOP`.
  An exact replay-stable entity MUST select a concrete-confirmation ledger
  scoped to exact root and non-null entity. A qualifying L1 workload failure without
  full identity MUST instead select a workload-confirmation ledger scoped to
  the root family. Each defaults to one retry and stops only on its first
  qualifying recurrence without observed advance. Missing entities MUST NOT
  compare equal or form a concrete identity.
  Path precedence MUST be eligible grounded primary, grounded selected
  observation, deterministic facts, then none. L4 MUST use only the matching L3
  history for the selected path and MUST preserve all other tracks in the
  attempt record. When no eligible primary exists but one source-grounded terminal observation is selected, L4
  MUST use root-independent general retry and same-job progress accounting. It
  MUST NOT consume root or concrete ledgers from an observation fingerprint.
  When neither a primary nor selected observation exists, failure-dependent
  policy is unavailable and the result remains restart-biased while global job
  guards continue to apply.
  **Verify:** ordered-rule, immediate-STOP, retry-exhaustion,
  progress-protection, and ambiguity tests.

- **RA-10: Declared policy context.** Policy information supplied outside the
  log MUST be explicit rather than attributed to model inference. L4 MUST apply
  it only when its complete typed signature matches, while preserving the base
  rule and applied context in the result. The rejected-iteration retry-then-skip
  context is selected from current facts; same-root, same-iteration, observer,
  and progress history only consume its retry budget. The port-bind
  confirmation context MUST match only an explicit typed address-in-use
  primary and MUST preserve L1 domain/recovery semantics while applying its
  independent same-root budget.
  **Verify:** context-present, context-absent,
  changed-iteration, observer-mismatch, port-bind recurrence, and
  policy-override tests.

## Operations And Qualification

- **RA-11: Configuration, tools, and parallel routes.** A versioned
  configuration MUST bind runtime history, L0 source read mode/chunk size,
  retry policy, declared policy contexts, routing, and per-route request,
  reasoning, tool, and reliability settings. N routes MUST execute independently
  and concurrently from identical L0 evidence and prior history. MVP collection
  MUST NOT silently vote, merge, or select a winner. Tools MUST be optional,
  read-only, advertised, bounded, and traced.
  **Verify:** configuration-resolution, shared-L0, route-isolation, concurrency,
  analysis-timeout, and tool-contract tests.

- **RA-12: Observability.** Results and traces MUST explain evidence selection,
  model/tool interactions, grounding, history comparison, policy, provenance,
  latency, analysis-timeout behavior, endpoint errors, and degradation. Large
  transcripts may be out of band but MUST remain discoverable. Secrets MUST
  never be recorded.
  **Verify:** trace-schema, redaction, artifact-publication, and summary/detail
  tests.

- **RA-13: Production/eval parity.** Production and evaluation MUST use the
  same product revision and equivalent resolved configuration. Qualification
  MUST measure decision and root-cause accuracy, false-STOP safety, shadow
  STOP, repeated-run stability, fingerprint false merges/splits, model
  efficiency, endpoint reliability, latency percentiles, and timeout rate.
  **Verify:** reviewed corpus and representative load tests.

- **RA-14: Export compliance and secrets.** Export-controlled model context MUST
  use approved regulated routes and credentials, including network fallbacks.
  The operator owns workload classification and authorization. The analyzer
  MUST NOT infer compliance from route names or expose credentials in
  configuration artifacts, prompts, traces, or reports.
  **Verify:** approved route/fallback and secret-redaction tests.

## Non-Goals

- Selecting nodes or devices for drain, quarantine, or scheduler mutation.
- Providing complete workload repair instructions.
- Using a model to compare prior attempts or consume retry budgets.
- Depending on unbounded or mandatory model-driven tool loops.
- Persistent or distributed history in the MVP.
- Model voting, semantic merging, or priority arbitration in the MVP.
- Owning or enforcing the external NVRx action deadline.
- Treating current-process feasibility as production qualification.

Exact contracts and algorithms remain canonical in `SCHEMA.md`, `L0A.md`
through `L4.md`, `RUNTIME.md`, `CONFIGURATION.md`, `TOOLS.md`,
`ATTRSVC_INTEGRATION.md`, and `PROGRESSIVE.md`.
