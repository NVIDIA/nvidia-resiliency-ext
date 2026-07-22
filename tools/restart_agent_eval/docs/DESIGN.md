# Restart Agent Eval Design

## Purpose

The Restart Agent evaluation harness qualifies product versions, evidence
profiles, prompts, model routes, tool profiles, and retry-policy configurations
before production promotion. It is a non-installable developer/QA tool. Failure
analysis remains implemented by the NVRx product checkout under test.

The harness supports two workflows:

- review a loose log against one or more routes and produce inspectable
  artifacts without claiming semantic accuracy;
- run explicitly human-reviewed cases and score each product stage against
  immutable gold bound to exact source bytes.

## Authority Boundary

The canonical ownership and packaging boundary is the product document
[EVALUATION.md](../../../docs/design/attribution/restart_agent/EVALUATION.md).
In summary, the product owns runtime analysis and every L0-L4 result.

The harness owns case metadata, panel route selection/configuration, artifact
review, gold comparison, corpus aggregation, model/profile qualification, and
promotion recommendations. Product `collect_all` owns parallel route execution.
Gold labels and source paths are never included in model-visible content.

## Common Execution Path

```text
loose log or scored case
        |
        v
restart_agent_eval.review
        |
        v
product CLI: collect_all
        |
        +--> shared L0A --> DecisionEvidence --> L0B
        |
        +--> parallel independent model routes --> L1 --> L2 --> L3 --> L4
        |
        +--> shared deterministic fallback ---------> L3 --> L4
        |
        v
per-route result + full trace
        |
        +--> per-route review
        +--> panel summary
        +--> optional gold comparison
```

`restart_agent_eval.evaluate` invokes the same product-review path used for
manual review.
The analyzer is therefore not reimplemented in the harness.

Product execution is isolated behind `product_process.ProcessExecutor`; the
versioned payload builders in `product_contract` own the harness side of the
CLI contract. One-log review and corpus evaluation do not construct operating-
system processes directly or consume `subprocess` result types. Deployment
endpoints, model IDs, credential
references, and tool advertisement are immutable values in `profiles`.

Deterministic behavior capture imports the product only in a short-lived worker
interpreter. A product worktree therefore cannot persist through parent-process
`sys.path` or `sys.modules` state and contaminate a later fixture capture.

## Modes

### Review Mode

Review mode accepts an arbitrary log and one or more targets. It reports model
semantics, route efficiency, endpoint behavior, and product policy output as
observations. Without human-approved gold, all accuracy fields are `unscored`.

The log path relative to the declared source root selects a durable
`<gold-root>/<relative-log-path>/gold.json` label. This enables local gold
comparison without placing labels beside logs or exposing them to the product
or model. Scoring begins only after the harness verifies the label's human
review state and `source_sha256` against the selected log.

### Artifact Roots

`REQUIREMENTS.md` owns the three-root isolation invariant. `SCHEMA.md` owns the
concrete mirrored path layout and manifest fields. Review and scored modes use
that same layout; this design does not define another artifact scheme.

### Scored Mode

Scored mode discovers mirrored `gold.json` files using
`restart_agent_eval.v1`. A missing mirrored source log makes the case
unavailable; it does not remove the label or count as an analyzer failure.

### Replay Modes

Terminal replay is implemented. Progressive replay is required later to
measure the production gate from `progressive_end` to a usable result. Terminal
latency must not be presented as post-progressive-end latency.

## Components

### `restart_agent_eval.review`

Runs one product analysis, captures product result/trace artifacts, derives
per-route KPIs, writes per-route reviews, and invokes the panel summarizer.
Multi-model review uses product `collect_all`, so L0 is built once and every
route receives byte-equivalent shared evidence.

The review index is the run navigation boundary, not a duplicate metrics
report. It links to the panel comparison and each route's human review, final
result, and deep trace. A route review reproduces the complete parsed L1 object
before showing gold scores or downstream stage diagnostics, so reviewers do not
need to navigate trace internals to inspect what the model actually returned.

The runner creates and prints the run directory before invoking the product. It
passes a route-artifact manifest containing each route's exact final result and
trace paths, streams the product's append-only lifecycle events to the console,
and retains only lifecycle status and events under `live/`.
As soon as shared L0 completes, the product atomically publishes canonical
`l0_bundle.json`, `decision_evidence.json`, and `l0_model_view.json` at the run
root and emits their readiness event. These are complete shared stage outputs,
not per-route or partial-live copies.
When each route completes, the product writes its trace and then result directly
at those canonical paths and emits a completion event. The harness reads those
files and atomically writes that route's review JSON and review Markdown without
waiting for slower routes. Only batch result/trace, review index, and panel
artifacts wait for every route or the product deadline. There is no duplicate
`live/routes` payload tree.

Product trace adaptation is isolated in `product_trace` and `review_context`.
Review construction consumes that normalized context; gold scoring and runtime
KPI derivation live in `scoring`; Markdown publication is a separate renderer.
The runner does not reinterpret product stage semantics.

`ReviewApplication` is the one-log composition root. It receives an immutable
environment snapshot, process executor, clock, and polling sleeper. Artifact
layout accepts an injected clock or run ID, and `ReviewContext.from_payloads`
normalizes already-loaded product data without filesystem access. The default
runtime still uses subprocesses, the system clock, and atomic local files.

Product command/process control is owned by `product_process`. Product config
and route-artifact payload shapes are owned by `product_contract`; product
validation remains the final authority. Repository provenance is isolated in
`repository_identity` rather than duplicated in review and corpus workflows.

### `restart_agent_eval.panel`

Builds `panel_summary.json`, compact reviewer-facing `panel_summary.md`, and
exhaustive `panel_diagnostics.md` from existing review JSON. It never calls a
model. The compact report leads with run identity and actionable concerns, then
separates semantic quality, behavioral efficiency, endpoint reliability, and
route outcome. Full stage data remains in JSON and diagnostics.

Panel normalization and concern derivation are separate from compact and
diagnostic Markdown renderers. Presentation changes therefore do not alter
scoring or route execution. `PanelInput` separates run-file loading from the
pure panel payload calculation.

Gold scoring keeps L1 RCA, L1 recovery semantics, L2 history identity, final
cascade grouping, and L4 policy/action as independent outcomes. L2 identity
gold describes canonical anchor, operation, mechanism, and expected cross-route
cardinality; it does not freeze an implementation hash. Replayed L0 artifacts
are labeled as replay rather than counted as zero-cost assembly.

### `restart_agent_eval.stability`

Reads completed one-log run directories and produces a repeated-run stability
report without calling a model. It derives strict comparable cohorts from
source, product, analyzer-config, runtime-input, route-profile, L0A, L0B, and
first-request hashes. It then measures L1 availability, final-action and policy-input flips,
primary and fingerprint agreement, tool-path variability, endpoint health,
latency, tokens, and independently available gold accuracy.

The summarizer is intentionally downstream of ordinary run artifacts. It does
not own expensive repetition, change route settings, vote across models, or
reinterpret product policy. See `STABILITY.md` for the contract and workflow.

### `restart_agent_eval.evaluate`

Discovers scored cases, invokes one product review target per case, compares
product artifacts with gold, and writes per-case JSONL plus aggregate metrics.
`EvaluationApplication` owns its clock and process executor. Case scoring does
not read environment variables or instantiate subprocesses.

### Testability Boundaries

Runtime dependencies are supplied at application boundaries: clocks generate
run IDs and timestamps, process executors launch the product, sleepers poll
incremental events, and `ArtifactStore` owns structured review/index artifact
persistence. Product-published route files and Markdown renderers retain their
explicit local-file contracts. Payload
adapters (`ReviewContext`, `PanelInput`, and `ToolEfficiencyInput`) accept
already-loaded values so scoring and report calculations can be tested without
temporary files. Environment variables are snapshotted by CLI entrypoints and
passed into route validation and process construction.

### `restart_agent_eval.product_trace`

Validates supported product trace envelopes and presents a single typed
boundary to the rest of the harness. It does not reinterpret product-internal
layer schemas.

### `restart_agent_eval.inspect`

Shows L0A, DecisionEvidence, L0B, or their relationship for evidence-profile
review without invoking a model.

## Stage Measurement

`REQUIREMENTS.md` is canonical for KPI definitions, and `SCHEMA.md` is canonical
for their serialized fields. This design owns only the measurement flow:

1. Normalize product stage outputs and runtime telemetry into one review context.
2. Keep operational measurements valid even when no gold exists.
3. Attach human-gold comparisons only after approval and source-hash validation.
4. Preserve L1 semantic quality, behavioral efficiency, endpoint reliability,
   and combined route outcome as separate axes.
5. Score deterministic-fallback and L1-enriched L4 paths independently.
6. Render compact reviewer conclusions separately from exhaustive diagnostics.

Tool behavior remains model-conditioned: final dependence on relevant tool-only
evidence indicates a likely L0B gap, while rereading bundled evidence indicates
model/prompt inefficiency. L2 is grounding, identity, and advisory audit; it does
not rewrite L1 semantics. L3 remains observational, and L4 owns policy/action.

## Model Comparison

Each route is judged through four independent views:

1. semantic quality;
2. behavioral efficiency;
3. endpoint reliability;
4. route outcome, including model-enriched versus fallback contribution,
   result usability, latency, and deadline status.

No opaque aggregate score is required. Promotion first applies correctness,
false-STOP, contract, deadline, and reliability gates; efficiency differentiates
routes that pass those gates.

## Multi-Model Execution

The harness supplies route profiles to product `collect_all`. Routes execute in
parallel after shared L0. This supports panel evaluation and future production
pairing of a predictable fast route with a slower preferred route. The current
product mode returns every route independently and does not vote or select a
winner.

Route identity includes model, endpoint, credential reference, request budget,
reasoning controls, and advertised tools. Secrets remain external. The primary
and secondary key slots are `LLM_API_KEY_FILE` and `LLM_API_KEY_OLD_FILE`.

## Reproducibility And Isolation

Each run records eval and product commits, dirty state, case/label version,
non-secret route configuration, result and trace paths, timings, tokens,
retries, endpoint failures, tool activity, and redaction checks.

The model transcript is audited for source-path leakage. Generic path tokens
such as `logs` do not identify a source and are ignored unless combined with
specific source-path content.

## Profile Optimization Loop

The future automated loop may vary registries, L0 selection and size limits,
prompt/schema versions, model/reasoning/tool profiles, and retry-policy
configuration. Every candidate runs through the same product path and is
compared per stage and end to end. Human-approved labels and holdout cases
remain the quality authority; promotion remains explicit.

Reactive signature additions and prompts that merely forbid observed mistakes
are anti-patterns. Review should ask whether a change is generic, which stage
owns it, and whether corpus and holdout results improve without regression.
