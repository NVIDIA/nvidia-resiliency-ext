# Restart Agent Eval Requirements

## Scope

The harness MUST evaluate the NVRx Restart Agent product without duplicating
its analysis logic. It MUST support loose-log review, human-scored corpus runs,
multi-model panel comparison, and reproducible profile qualification.

Progressive replay, automated profile search, production route arbitration,
and external log-source correlation are outside the current implementation.

## Product Boundary

- Every real analysis MUST execute the selected product checkout.
- Manual review and scored eval MUST use the same `review_log.py` product path.
- The harness MUST record product commit and dirty state.
- Product trace envelopes MUST be validated through `product_trace.py`.
- Unsupported trace schemas MUST fail explicitly.
- Gold labels, case ids, source directories, and expected outcomes MUST NOT be
  supplied to the product or model.

## Review Mode

- One-log review MUST accept deterministic, configured, named model, and panel
  targets.
- Multi-model review MUST use product `collect_all` so L0 is built once and
  routes run concurrently over equivalent shared evidence.
- The harness MUST write per-route result, trace, review JSON, and review
  Markdown plus panel JSON, compact panel Markdown, and exhaustive diagnostic
  Markdown.
- Semantic accuracy MUST be `unscored` without an explicitly human-reviewed
  gold label whose `source_sha256` matches the source-log bytes.
- A mirrored `<gold-root>/<relative-log-path>/gold.json` MAY enable scoring.
- Source-path leakage, endpoint failures, retries, timeouts, malformed output,
  tool activity, tokens, and stage timings MUST remain visible.

## Artifact Roots

- Source logs, durable gold, and generated runs MUST use three distinct,
  non-overlapping roots.
- The source path relative to `log-root` MUST be mirrored under `gold-root` and
  `run-root` without flattening or path-derived model hints.
- One-log runs MUST be written to
  `<run-root>/<relative-log-path>/<run-id>/`.
- Scored corpus runs MUST be written below `run-root`; deleting that root MUST
  NOT delete logs or human-approved gold.
- A run manifest MUST record the source relative path, SHA-256 and byte size,
  all artifact roots, product and harness commits, route identities, and the
  expected gold path.

## Scored Cases

Each case MUST use `restart_agent_eval.v1` and include:

- stable `case_id` and `label_version`;
- `action_expectation.accepted` containing `STOP` and/or `RESTART`;
- optional L0A, L0B, primary-anchor, cascade, and L2-audit expectations;
- optional `recovery_assessment_expectation`;
- optional `retry_policy_expectation`;
- optional unsupported-claim text patterns and human rationale.

Missing `input.log` MUST mark a case unavailable. It MUST NOT remove the label,
count as an analyzer error, or silently weaken aggregate denominators.

The harness MUST NOT score policy classes or reconstruct user/not-user scores.

## Recovery Assessment Gold

Gold MAY specify one or more acceptable values for each L1 field:

```yaml
recovery_assessment_expectation:
  failure_domain: [workload]
  failure_domain_status: [established_by_current_log, supported_but_unconfirmed]
  retry_outlook_without_workload_change: [may_recover]
  retry_outlook_status: [supported_but_unconfirmed]
```

Only declared fields are scored. Per-field outcomes and an overall declared-
field result MUST both be retained. Each claim's confidence MUST be preserved;
calibration MUST be reported only over a corpus.

## Retry Policy Gold

Gold MAY specify:

```yaml
retry_policy_expectation:
  accepted_rules: [bounded_retry]
  allowed_retries: 1
  matching_prior_failures: 0
  retry_budget_exhausted: false
```

Rule selection, allowed retries, matching count, exhaustion, decision basis,
and action MUST be evaluated independently when labeled. A correct action MUST
NOT hide an incorrect rule or retry count.

## Stage KPIs

### L0A

Quality KPIs SHOULD include primary evidence coverage, selected-primary
accuracy, progress/checkpoint detection, deterministic fingerprint accuracy,
false merges, and false splits. Operational metrics MUST include assembly
latency, input size, object counts, caps, and lossiness.

### DecisionEvidence And L0B

Quality KPIs SHOULD include required fact/reference selection and model-view
retention. Operational metrics MUST include selection/projection latency, view
size, estimated tokens, budget use, selection/compaction counts, integrity, and
replay consistency.

### L1

Semantic quality MUST cover primary/mechanism, RCA, recovery fields, causal
roles, uncertainty, and unsupported claims when labeled. Behavioral efficiency
MUST cover first-turn usability, model/tool/repair turns, tool yield and
redundancy, tokens, and latency. Endpoint reliability MUST separately cover
attempts, retries, timeouts, HTTP/provider failures, and failed-attempt time.

### L2

Quality SHOULD cover citation/reference correctness, audit correctness, and
enriched fingerprint correctness/merge/split behavior. Runtime metrics MUST
show grounding status, citation outcomes, finding materiality, identity
readiness, and latency. L2 disagreement MUST NOT be reported as an L1 rewrite.

### L3

Quality SHOULD cover exact job/root filtering, progress relations, matching
prior counts, streaks, and exact position/data/artifact observations. Unknown
progress MUST remain distinct from no observed advance.

### L4

Quality MUST report retry-rule accuracy, allowed-retry accuracy, exhaustion
accuracy, decision-basis accuracy when labeled, and action accuracy. False STOP
rate is a release gate. Runtime metrics MUST expose policy version, rule,
allowed retries, matching prior failures, exhaustion, and latency.

The deterministic fallback L4 result and every available L1-enriched L4 result
MUST be scored separately against the same gold expectations. Reports MUST
distinguish action-only improvement/regression from complete policy/action
improvement/regression. A missing or late enriched result MUST remain
`not_available`, not be counted as agreement with the fallback.

## Tool Evaluation

- Tool availability and advertisement MUST be distinguished.
- Production-comparable eval MUST use the same advertised tools and limits as
  the production route profile.
- Model turns, tool-driven turns, repair turns, and provider retries MUST be
  counted separately.
- Duplicate reads and no-new-context calls MUST be visible.
- Tool-returned lines MUST be classified as decision-relevant, structured-fact
  repeats, incidental cited context, or unused exploration when possible.
- Tool calls alone MUST NOT be labeled an L0 gap. Final dependence on missing
  tool-only evidence is the stronger signal.

## Multi-Model And Credentials

- Named routes MUST execute independently and concurrently through product
  `collect_all` after shared L0.
- A route MUST record model, endpoint, request/reasoning/tool profile, and
  logical credential source without recording secret values.
- The harness MUST support primary and secondary external key-file variables.
- Route failure MUST NOT prevent other route artifacts from being produced.
- The current panel MUST NOT vote, merge, prefer, or select a winning route.
- A one-log panel run MUST print its run directory before model execution and
  expose route completion while slower routes remain in flight.

## Decision Stability

- The harness MUST measure repeated-run stability without invoking a model
  while summarizing existing result directories.
- Samples MUST be compared only within cohorts that agree on source, product,
  analyzer configuration, runtime input, route profile, L0A, L0B, and
  first-request identities.
- Input mismatches MUST create separate cohorts rather than reduce one
  stability score. Missing identities and dirty product state MUST be visible.
- Usable L1 rate, final-action agreement and sequential flips, exact
  action-driving L1 field agreement, primary/fingerprint agreement, tool-path
  variability, endpoint reliability, latency, and tokens MUST be reported
  separately.
- Endpoint failures MUST NOT be treated as usable L1 semantic samples.
- Confidence and natural-language rationale MUST NOT be part of the exact
  action-driving L1 tuple.
- Gold accuracy and repeated-run stability MUST remain independent dimensions.
- Stability status MUST be descriptive and MUST NOT promote or reject a route.
- The default minimum for an established observation SHOULD be ten comparable
  samples and MUST be configurable by the harness command.

## Latency

- Terminal review MUST label latency as terminal request-to-result.
- Progressive qualification MUST measure from `progressive_end` to usable
  result and separately report work shifted before that signal.
- Terminal latency MUST NOT be substituted for the production progressive gate.
- p50, p90, and p99 are corpus/profile measurements, not constants in the
  product specification.
- Optional provider-reported downstream timing MUST be shown only when supplied
  by the endpoint and MUST NOT be called pure model compute.

## Artifacts And Reproducibility

Every scored run MUST produce a manifest, per-case JSONL, aggregate JSON, and
the underlying product artifacts. The manifest MUST identify eval/product
commits, label versions, route/profile identity, command options, timestamps,
discovery mode, and all artifact roots.

`panel_summary.md` MUST lead with run identity and actionable concerns, then
separate route outcome, semantic comparison, behavioral efficiency, and
endpoint reliability. Shared deterministic evidence and policy/history facts
MUST be shown once. Exhaustive per-stage tables, identities, fingerprints,
paths, and low-level interaction details MUST remain in `panel_diagnostics.md`
and structured JSON rather than widening the reviewer summary.

Artifact writes MUST use UTF-8 with replacement for source-derived diagnostic
text. Secrets and key-file contents MUST never be persisted.

Long-running `collect_all` review MUST expose an atomic `run_status.json`, an
append-only lifecycle event stream, the deterministic fallback when available,
and each route result/trace as that route completes. The harness MUST declare
each route's canonical result/trace paths before execution. The product MUST
write directly to those paths and emit a completion event; it MUST NOT create a
duplicate per-route payload tree under `live/`. The harness MUST use each event
to atomically write that route's review JSON and review Markdown without waiting
for other routes.
Batch-level results, review indexes, and panel summaries remain final derived
artifacts written after all routes finish or the product deadline expires.
Readers MUST NOT consume partially written JSON.

Shared L0A, Decision Evidence, and L0B artifacts MUST be written once at the run
root as soon as complete, before waiting for model routes. Publication MUST be
atomic and MUST NOT serialize the same shared payload once per model. The live
status/event stream MUST report their readiness and paths; `live/` need not
duplicate their contents. Canonical route and batch traces MUST be written before
their result files so result existence is a reliable completion marker.

## Promotion

Promotion MUST be based on reproducible profile identity and a human-approved
corpus. Correctness, false-STOP, contract, latency, and endpoint-reliability
gates apply before efficiency comparisons. Holdout cases MUST be used for
prompt, registry, evidence-selection, fingerprint, and policy changes.

The harness MAY recommend a profile. It MUST NOT rewrite production
configuration or tune the product online.
