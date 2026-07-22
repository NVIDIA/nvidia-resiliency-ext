# Eval Panel Summary Contract

This document owns report organization and rendering. `REQUIREMENTS.md` owns KPI
meaning and qualification rules; `SCHEMA.md` owns machine-readable fields. The
tables below select and arrange those values rather than redefine them.

Status: compact reviewer summary and exhaustive diagnostic companion are
implemented. Corpus aggregation and qualification gates remain partial, and
fields have no compatibility guarantee yet.

The report header records the restart-agent config id, version, and resolved
config fingerprint when product `collect_all` is used. A route is the model
plus its endpoint, sampling, reasoning, tools, and budgets; this prevents two
different profiles of the same model from being collapsed into one label.

## Report Split

The harness writes two Markdown views from the same `panel_summary.json`:

- `panel_summary.md` is the compact reviewer view. It starts with source/run
  identity and actionable concerns, then compares route outcomes, semantics,
  behavioral efficiency, and endpoint reliability.
- `panel_diagnostics.md` is the exhaustive engineering view. It retains the
  complete layer-oriented tables, identities, fingerprints, paths, token and
  tool detail, and low-level validation observations.

This split keeps the first review readable without discarding diagnostic data.
The compact view uses four separate comparison axes:

1. **Semantic quality:** did the model understand the failure correctly?
2. **Behavioral efficiency:** how much interaction work did the model require?
3. **Endpoint reliability:** did the provider deliver responses reliably?
4. **Route outcome:** did their combination produce a usable result in time?

The run-level `review_index.md` is the entry point. It links to this panel and
to each route's `review.md`, where the complete parsed L1 response is shown
before downstream stage KPIs. The index does not duplicate comparison metrics.

The layer-oriented measurements should remain available to explain those
outcomes, but they should not lead the report.

Token count is not an efficiency result by itself. Endpoint queueing, model
throughput, extra model turns, and tool round trips can produce very different
latency for similar token counts.

## Compact Report Structure

`panel_summary.md` reads from review context to model-selection outcomes:

1. Run Identity
2. Attention Required
3. Gold Scorecard, when gold exists
4. Cross-Route Outcome
5. Semantic Comparison
6. Operational Comparison
7. Shared Deterministic Evidence
8. History And Policy
9. Conditional Diagnostics
10. Artifact links

Route-profile detail remains in `panel_diagnostics.md`, `run_manifest.json`,
and `panel_summary.json`; it is configuration provenance rather than a compact
review outcome.

Shared L0 and policy facts appear once rather than once per route. Conditional
diagnostics appear only when tool use, degraded execution, grounding findings,
context-budget changes, or unstable fingerprints require attention. The full
stage-by-stage view always remains available in `panel_diagnostics.md`.

## Route Outcome

This is the combined production-facing view. It reports whether the route
returned usable model-enriched evidence or required deterministic fallback,
the final result quality and NVRx usability, the measured route latency and its
basis, and the independent semantic and endpoint statuses. It MUST NOT collapse
the three contributing axes into an unexplained numeric score.

## Semantic Quality

This section answers whether the model understood the failure and recommended
the correct recovery semantics. It is conditional on a delivered model
response. When no response is delivered because of an endpoint-only failure,
semantic quality is `not_observed`, not `incorrect`. A malformed delivered
response is a model/contract-quality failure.

Suggested columns:

| target | L1 status | decision | primary | recovery confidence | L4 rule | retry count/budget |
| --- | --- | --- | --- | --- | --- | --- |

Semantics:

- `L1 status` is `ok`, `degraded`, `failed`, or `not_run`. A recovered failed
  call/retry is degraded even when the final structured response is usable.
- `decision result` is `correct`, `incorrect`, or `unscored`.
- `L1 RCA` scores the approved primary, observed operation/mechanism, required
  RCA concepts, uncertainty, and explicit mechanism contradictions.
- `L1 recovery` independently scores the two claim values and their two
  evidence statuses. RCA
  and recovery are not collapsed into one cell.
- `L1 related failures` independently measures whether the model enumerated
  expected cascade/teardown context. This is a model-context KPI, not the
  analyzer's final cascade result. It remains in the scorecard/diagnostics and
  does not create an attention item when final cascades are correct.
- `final cascades` measures the assembled analyzer result after deterministic
  L0 facts are available. It is the product-level cascade-correctness check.
- `Final Downstream Roles` reports per-route `cascade` and `teardown` counts and
  line ranges. Gold `teardown_lines` requires the explicit teardown role, so a
  retained downstream event with the wrong role fails that check.
- `recovery assessment` exposes L1's raw failure domain and
  `retry_outlook_without_workload_change`, including each claim's evidence
  status and confidence.
- `recovery confidence` preserves the two unchanged 1-99 claim confidences.
  They are observable and calibratable across a corpus, but are not action
  scores or L4 thresholds.
- `L4 rule` and `retry count/budget` expose the deterministic rule selected,
  matching prior failures, allowed retries, and exhaustion state.
- `L4 policy/action` scores only the retry-rule fields and final action. It does
  not inherit L1 RCA/recovery failures; the all-stage semantic result remains
  available separately in diagnostics.

The deterministic fallback is not an L1 model route. Its L1 cells are
`not_scored`,
and it is excluded from model fingerprint agreement and model-quality concerns.
Its L0 and L4 policy/action results remain scored.

`Fallback Versus L1-Enriched Policy` scores the product's two actual decision
paths against the same gold label. The shared deterministic fallback is shown
once. Each model route then shows its enriched decision and retry rule plus two
independent effects:

- `action_effect` compares only the final `STOP`/`RESTART` action;
- `policy_action_effect` compares the complete labeled L4 rule, retry budget,
  exhaustion state, and action.

Effect values are `improved`, `regressed`, `unchanged_correct`,
`unchanged_incorrect`, `unscored`, or `not_available`. An unavailable enriched
candidate is not treated as an unchanged fallback. The JSON report retains
per-route values and aggregate effect counts so corpus tooling can calculate
fallback accuracy, enriched accuracy, improvement rate, and regression rate.

Review-mode logs MUST show semantic results as `unscored`. Only a log with a
human-approved mirrored
`<gold-root>/<relative-log-path>/gold.json` may report decision or RCA
accuracy. A gold-scored panel shows L1 core semantics, L1 related-failure
recall, L2 grounding/identity/audit, final cascades, L4 policy/action, unsupported claims,
and confidence separately. Confidence is not called calibration for a single
case.

### Deterministic Fallback Inputs

The report MUST show the shared L0 `DecisionEvidence` once, before per-model
stage comparisons. This section is the directly inspectable input to the
concurrent fallback L3/L4 path when L1 is late or unavailable. It includes:

- the deterministic L0 primary and its registry hint, outcome, phase, and causal
  role;
- the deterministic primary's actual fallback L3 root fingerprint and source;
- the canonical L0 identity anchor, reason, fingerprint, and source, shown
  separately when it differs from the primary fingerprint;
- compact progress/checkpoint facts used by fallback history; and
- counts and source lines for the selected L0A evidence references.

The Markdown report MUST state whether every model artifact carried a
byte-equivalent JSON payload. `panel_summary.json` MUST retain the exact shared
object as `shared_decision_evidence` and consistency metadata as
`decision_evidence_consistency`. If payloads are missing or differ, the panel
MUST emit an `l0_decision_evidence_consistency` concern and retain the available
per-target payloads in `decision_evidence_by_target`.

### Primary Selection By Stage

The report MUST preserve and compare three selections instead of showing only
the final primary:

| Selection | Meaning |
| --- | --- |
| L0 deterministic | `DecisionEvidence.deterministic_primary_candidate`, selected without model semantics. |
| L1 semantic | The model's unchanged `primary_failure` from its raw structured response. |
| L2 grounded | The mechanically grounded primary and client-derived identity. Separate audit findings remain diagnostic; L1 semantics remain the L4 input. |

The Markdown view shows `fine_class@line` for each stage and a separate compact
relationship table. Relationship values are `same_line`,
`same_failure_episode`, `same_distributed_incident`, `different_selection`, or
`not_available`. These are observability facts, not accuracy scores: a model may
legitimately choose a more descriptive line in the same episode than L0.

The JSON artifact retains all three compact primary objects in each row and the
complete versioned shared Decision Evidence at panel level so reviewers can
distinguish L0 coverage, semantic selection, grounding, and final policy
behavior.

## Behavioral Efficiency

This section answers how much interaction work the model required after context
was delivered. It reports first-turn usability, conversational turns,
tool-driven and contract-repair turns, tool calls and redundancy, and tokens.
Provider retries are excluded because they are endpoint behavior.

Suggested columns:

| target | first-turn usable | model turns | tool turns | repair turns | tool calls | duplicate/no-new | total tokens |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |

Semantics:

- `first-turn usable` is scored only when at least one model response was
  delivered. A timeout with no response is `not_observed` here and is counted
  under endpoint reliability.
- `model turns` counts the initial request plus each continuation after tool
  results or output repair. `tool-driven turns` and `contract-repair turns` are
  also shown separately so a tool-efficiency view does not attribute repair to
  tool use. Provider retries are additional call attempts in the same turn and
  MUST NOT increase any conversational-turn count.
- `useful tools` counts calls that added relevant context not already present
  in the initial model-facing bundle. A final primary/supporting citation that
  depends on tool-only lines is the automatic high-confidence signal. Relevance
  otherwise requires a scored rubric or review; new lines alone are only a
  proxy.
- `final-answer dependency` records whether the selected primary or cited
  evidence came only from tool-returned lines. It does not claim policy changed
  when the model emitted no provisional pre-tool assessment.
- tool-only context is split into decision-relevant lines, structured L0 fact
  repeats, incidental cited lines, and unused returned lines. The L0-gap concern
  reports the decision-relevant subset rather than attributing every newly
  returned line to the final answer.
- `total tokens` preserves available input, output, reasoning, cached, and
  aggregate token details in the JSON artifact and internal diagnostics.
- End-to-end elapsed time and deadline outcome belong to Route Outcome because
  they combine model behavior, local tool work, transport, queueing, and
  provider service time.
- `Provider-Reported L1 Timing` appears only when response headers carry timing
  spans. It shows downstream LLM API and proxy pre/post/message-copy totals.
  Downstream LLM API is a proxy-observed downstream-call span, not model compute;
  unknown queue, prefill, decode, transport, and residual spans are not invented.

The report should not initially collapse turns, tools, and tokens into one
numeric efficiency score. Their relative weights depend on deployment
constraints. The raw dimensions allow later profiles to define a gate without
discarding the underlying measurements.

For progressive runs, the product-level metric is post-`progressive_end`
decision latency. L0 work completed before that signal should not be charged to
the post-failure gate. Static one-log review currently reports terminal
request-to-result latency plus stage timings; it must show progressive gate
latency and decision-window hit rate as unavailable rather than infer them.
The compact per-model latency table omits the post-end column when no
progressive measurement exists and uses L1 as the comparable per-model terminal
latency. Shared L0 build latency appears once in the L0 diagnostics rather than
being charged only to the first model that materialized the shared bundle.

## Endpoint Reliability

This section separates provider/service behavior from semantic quality and
behavioral efficiency.

Suggested columns:

| target | endpoint | attempts | retries | timeouts | provider errors | failed-call time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

Semantics:

- `endpoint` is `ok`, `endpoint_issue`, or `deadline_exceeded`.
- `deadline_exceeded` means the route did not publish a result within the
  analysis deadline. It MUST NOT be reported as `ok` merely because the
  completed artifact contains zero provider attempts; an abandoned in-flight
  request may not have returned its partial call telemetry.
- Retries, timeouts, HTTP failures, connection resets, and gateway errors are
  endpoint events and MUST NOT count as semantic errors.
- `attempts` counts HTTP/provider calls. `model turns` counts distinct
  conversational turns. A retry changes attempts and retry counts, not turns.
- Timeout and HTTP-error counts are overlapping classifications of failed
  attempts. Reports MUST display the dimensions separately and MUST NOT sum
  them into a misleading aggregate endpoint-issue count.
- `failed-call time` captures latency consumed by unsuccessful attempts.
- Invalid JSON, schema violations, and unsupported tool requests are model or
  contract diagnostics, not endpoint failures. They remain visible under L1
  Validation and may also make `L1 valid` false in Semantic Quality.

Successful-call timing does not change endpoint reliability from `ok` to
`degraded`. Optional provider-reported timing is displayed in its own L1 timing
detail immediately after this section.

## Internal Diagnostics

The existing layer-oriented detail remains after the headline sections.

### L0A Quality And Operations

Report gold-scored primary coverage/selection, progress/checkpoint detection,
and other declared L0A quality facts separately from shared runtime operations.
Report bundle size, windows, anchors, episodes, incidents, caps, lossiness, scan
latency, and replay consistency once because the byte-identical L0A bundle is
replayed to every model.

### Decision Evidence

Report deterministic primary, canonical identity, referenced L0A objects, and
selection latency once. The fallback and model paths must consume the same
Decision Evidence object.

### L0B Quality And Operations

Report gold initial-view evidence retention separately from projection latency,
view size, budget utilization, selection/compaction counts, and projection
integrity/hash. Retain per-model first-turn completion, tool-only relevant
context, and reread behavior as model-conditioned diagnostics. These signals
require cross-model or profile comparison before being assigned to L0B.

Under current-attempt execution context, show observed failure position in a
separate compact table: checkpoint-load iteration, explicit failure iteration
canonical incident line, latest observed rank/rendering copy, replay distance,
and L0 phase. Keep completed progress in its own
table so an observed failure position is not mistaken for completed work.

### L1 Model And Tool Interaction

Retain per-turn timing, tool names, new-line yield, truncation, errors, and
per-call context details. The compact table should prefer `no-new context
calls`; exact duplicate line re-reads can remain in per-call detail. A duplicate
line read is a strict subset of calls that add no new context for the current
line-oriented tools.

When the product reports a model context limit, show a separate narrow context
budget table with the maximum estimated input, configured output cap, minimum
effective output cap, and adjusted-call count. This distinguishes client/profile
capacity mistakes from endpoint faults and model semantic quality.

### L2 Grounding, Identity, And Audit

Retain raw-source grounding, exact model-visible rendering matches, nearby
resolution, per-field credibility
findings, model causal role, raw model policy fields, and stable client-derived
identity. L2 does not judge semantic correctness. Output
repair, schema/contract status, timeout, truncation, and provider failures remain
in L1 runtime health.
Show total/material finding counts and severity counts. Only material findings
belong in the panel concern list; advisory and non-material credibility findings
remain visible in the L2 table and per-model review.

### Experimental Failure Identity

When the product emits `failure_identity`, show family and concrete identity in
separate narrow tables. Family comparison includes operation, mechanism,
exception, completeness, and a shortened display ID. Concrete comparison
includes component, callsite, failure position, completeness, and a shortened
display ID; grounded artifact paths are listed separately to avoid an overly
wide table. `panel_summary.json` retains full labels, fingerprints, stack paths,
and artifact paths. These values are observational and must not be presented as
active L3 history or L4 policy inputs while `policy_active=false`.

The panel also shows additive `client_concrete` identity separately from the
existing model-associated fingerprints. Its table includes observed exception,
callsite, failure position, completeness, and shortened ID. Full message
signature, source file, stack path, artifact path, and fingerprint remain in
JSON. The panel reports available-model count, unique fingerprint count, and
whether every available client fingerprint agrees. It also reports whether
identities are consistent among models selecting the same primary line and
classifies disagreement as `primary_selection_disagreement` or
`same_primary_identity_disagreement`. Markdown includes the full
shared `client_concrete_label` once when all available labels agree; otherwise
it lists the label per model.

`client_concrete` is the product's preferred experimental candidate for future
exact recurrence, not a current policy key. Agreement is an early stability
signal only. Corpus evaluation must separately measure false merges and false
splits, and any future recurrence score must also require distinct cycle,
exact job/root identity and typed observed-progress evidence.

### Root-Fingerprint KPIs

Show two ownership-specific sections because the exact-match key consumed by L3
can come from either path. `L0 Fallback Root-Fingerprint KPI` reports the shared
deterministic fallback key. `L2 Enriched Root-Fingerprint KPI` reports each
model route's grounded key, cross-model agreement, stable anchor, source,
history-readiness, and relation to L0. L1 does not own either key.
When all models agree, Markdown emits the full shared root fingerprint once;
otherwise it lists each model's full value. Equivalent wrapper-summary and
terminal-exception selections within one L0 failure episode should share the
same terminal identity anchor and root fingerprint.

Every panel assigns the policy-active fingerprint check one of three statuses:
`stable` requires at least two models, complete fingerprint coverage, and one
shared value; `unstable` means models produced different values; and
`not_checkable` means too few models or missing fingerprints. Any status other
than `stable` is emitted as an `l2_root_fingerprint_stability` concern. Runtime
availability and agreement are not accuracy: reviewed gold must independently
score the canonical anchor, operation context, and normalized observed
mechanism without locking the label to an opaque hash. Panel gold also declares
the expected number of cross-route identities. Corpus evaluation must still
measure false merges and false splits.

When L0 was supplied through deterministic replay, compact reporting labels the
L0A build as `replayed`; `0.0s` is not presented as fresh assembly performance.
Routes with no usable model enrichment are displayed as `fallback_only` and
`eligible_fallback`, with reason `no_model_enrichment`.

### Current-Attempt Execution, L1 Assessment, L3 History, And L4 Policy

Show one shared `Current-Attempt Execution Context` section derived from the L0
bundle. It includes successful runtime, first/last iteration, iteration delta,
last completed checkpoint, iterations since that checkpoint, progress after the
terminal episode, and later-progress-after-fault observation/event counts. State that
later job progress does not prove recovery of the same rank, node, path,
network, or component. These fields are execution context, not ground truth for
STOP/RESTART.

Show one shared `L1 Restart Environment` section from L0B and verify that it is
consistent across all available routes. When a failed route does not publish
the shared payload, report `consistent_among_available` with the availability
count; reserve `inconsistent` for differing payloads. It records whether
workload, process state, restart delay, hardware allocation, and mutable
external-service state are preserved or may change before the next attempt.

Retain raw L1 root-cause status and missing evidence, `failure_domain`,
`retry_outlook_without_workload_change`, both claim statuses and confidences;
L2
mechanical reference repairs, stable identity, credibility findings, and
unapplied suggestions; L3 history compatibility and progress relations; and
L4 policy version, retry rule, allowed retries, matching prior failures,
exhaustion state, decision basis, final action, result provenance, and NVRx
usability. The two recovery concepts must have separate columns so recovery
domain and retry outlook are not conflated. L4 output must never be
presented as though it were the raw L1 opinion.

Keep the two claims in one narrow `L1 Recovery Assessment` table: failure
domain and retry outlook without workload change, with value, status, and
confidence for each. The L4 table reports the selected rule, allowed retry
budget, matching prior count, and exhaustion state.

The report MUST show when L4 chooses STOP because current workload evidence
qualifies for immediate stop or because exact-root recurrence without observed
advance has exhausted the selected retry budget.

Headline sections should summarize these facts rather than duplicate all
diagnostic columns. For example, Behavioral Efficiency may show three tool calls
and one useful call; Internal Diagnostics explains which two calls duplicated
existing context.

Semantic-quality headlines show L1 raw semantic correctness and L4/final
product correctness separately. L2 shows citation/reference audit outcomes,
independent policy observations, and whether any suggestion remained unapplied.
Current
manual runs label latency as terminal request-to-result; progressive gate
latency remains unavailable until the bench executes `progressive_end`.

## JSON Artifact

`panel_summary.json` should preserve the same separation so downstream analysis
does not have to reconstruct external outcomes from L0/L1 implementation data:

```text
run_context
comparison_axes.semantic_quality[]
comparison_axes.behavioral_efficiency[]
comparison_axes.endpoint_reliability[]
comparison_axes.route_outcome[]
rows[]  # layer-oriented diagnostic fields
concerns
paths
```

Existing raw measurements should be retained even if the Markdown view becomes
smaller. Field names and migration behavior should be defined before
implementation.

## Review Questions

1. Do Semantic Quality, Behavioral Efficiency, Endpoint Reliability, and Route
   Outcome remain visibly separate in both Markdown and JSON?
2. Should `time to valid decision` exclude all L0 work, with product end-to-end
   latency reported separately?
3. Is `useful tools` sufficiently defined, or should the headline use the more
   objective `tools adding new context` until relevance can be scored?
4. Should malformed structured output remain only a contract diagnostic, or be
   promoted to a fourth headline reliability area?
5. Which existing `panel_summary.json` fields, if any, require compatibility
   during the change?
