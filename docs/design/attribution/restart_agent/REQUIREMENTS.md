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
  `RESTART`, its decision basis, retry rule and budget state, primary evidence,
  provenance, and degradation status.
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
  and stable client identity when available. L0B MUST create a bounded,
  attention-efficient initial model view without changing those facts. Neither
  stage may choose the action.
  **Verify:** reviewed L0 gold facts, source-reference integrity, projection
  coverage, and attention-efficiency measurements.

- **RA-06: Model semantic assessment.** L1 MUST identify the primary failure,
  semantic identity, and root-cause support, then independently assess failure
  domain and whether the next normal cycle may recover without workload
  changes. It MUST return structured citations and calibration confidence. It
  MUST NOT compare history or choose the action.
  **Verify:** response-contract, semantic-gold, calibration, stability, and
  malformed-output tests.

- **RA-07: Grounding and audit.** L2 MUST ground model-selected primary and
  supporting evidence before deriving enriched history identity or enabling
  narrow policy. Audit findings MUST remain visible and MUST NOT silently
  rewrite model semantics.
  **Verify:** exact, nearby, unresolved, invented-line, and non-overriding audit
  tests.

- **RA-08: Same-job history and progress.** Runtime history MUST be bounded,
  exact-job, cycle ordered, and idempotent by `(job_id, cycle_id)`. L3 MUST
  compare stable root identity and optional affected entity without an LLM and
  report independent root-only and root-plus-entity no-advance recurrence
  counts plus compatible training/checkpoint advancement. Entity changes MUST
  NOT reset the root-only count. Missing or unknown progress MUST NOT count as
  proven no progress.
  **Verify:** ordering, replacement, eviction, identity-match, and
  progress-comparison tests.

- **RA-09: Deterministic retry policy.** L4 alone MUST select the retry rule,
  evaluate the general same-root ceiling and any applicable narrower retry
  budget concurrently, and emit the action. Immediate `STOP` requires
  grounded current-log evidence satisfying the configured nonrecoverability
  rule; otherwise `STOP` requires exhaustion of either applicable ledger
  without compatible progress. A narrower rule MAY stop earlier but MUST NOT
  extend or reset the general ceiling. Domain, confidence, rank fanout, exact
  entity availability, or an ambiguous symptom alone MUST NOT produce `STOP`.
  An exact replay-stable entity MUST select a confirmation ledger scoped to
  exact root and entity; its default one retry stops only on the first
  qualifying recurrence without observed advance.
  **Verify:** ordered-rule, immediate-STOP, retry-exhaustion,
  progress-protection, and ambiguity tests.

- **RA-10: Declared recovery capabilities.** Workload-managed recovery MUST be
  explicit configuration rather than model inference. The MVP MUST support
  bad-token retry-then-skip using same-root-and-data-position history; without
  a matching declaration and grounded entity, generic policy applies.
  **Verify:** capability-present, capability-absent, changed-position, and
  missing-entity tests.

## Operations And Qualification

- **RA-11: Configuration, tools, and parallel routes.** A versioned
  configuration MUST bind runtime history, L0 source read mode/chunk size,
  retry policy, declared recovery capabilities, routing, and per-route request,
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
