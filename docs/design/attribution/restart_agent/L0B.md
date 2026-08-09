# L0B Context Management

L0B is the deterministic context-management and attention-projection stage. It
converts the complete L0A evidence into the bounded structured payload supplied
in the first L1 model request. That output artifact is the Initial Model
Evidence View (`L0ModelFacingView`). The view leads with a compact deterministic
failure narrative and follows it with selected supporting detail.

L0B is intentionally lossy but explicitly accountable. It prioritizes what the
model should inspect first while preserving trace references to the complete
evidence and allowing route-advertised read-only tools to retrieve additional
context. A trace reference identifies provenance; it does not expose the
referenced evidence unless an advertised tool resolves it.

The failure narrative is a projection, not another interpretation stage. It
orders already typed current-log facts using fixed templates and source
references. It must not introduce a semantic root cause, failure domain, retry
outlook, history conclusion, or action.

`L0A.md` owns source interpretation. `TOOLS.md` owns optional retrieval APIs.
`SCHEMA.md` owns the exact `L0ModelFacingView` shape.

L0B ends at deterministic evidence selection and rendering; model execution,
grounding, history comparison, and action policy belong to L1-L4.

## Purpose And Authority

```text
L0Bundle + DecisionEvidence + L0B selection limits
  -> construct deterministic failure narrative
  -> project policy-relevant Decision Evidence
  -> prioritize supporting evidence
  -> deduplicate
  -> compact
  -> render
  -> L0ModelFacingView
  -> L1 initial request
```

L0B is authoritative for:

- which L0A objects and source excerpts appear in the initial model payload;
- the deterministic ordering and rendering of the current-log failure
  narrative;
- the model-facing projection of exact internal Decision Evidence;
- deduplication and merging of overlapping windows;
- per-section projection limits and size accounting supplied to L1 preflight;
- truncation and omission metadata;
- projection integrity and deterministic payload hash.

L0B does not reinterpret the log, choose another deterministic primary or
selected observation, or change the exact internal Decision Evidence. Its
model-facing Decision Evidence projection may compact exhaustive collections
into exact counts, bounded samples, and references.

## Inputs And Output

| Direction | Contract | Meaning |
| --- | --- | --- |
| Input | `L0Bundle` | Complete typed evidence retained by the analyzer. |
| Input | `DecisionEvidence` | Canonical deterministic facts shared with the deterministic path. |
| Input | L0B evidence-selection rules and size limits | Deterministic per-section limits for collections, excerpts, text, and references. |
| Output | `L0ModelFacingView` | Exact model-facing current-attempt payload: deterministic narrative, compact Decision Evidence view, execution context, and selected support. It excludes the L1 response schema and provider tool schemas. |

The output contains:

```json
{
  "schema_version": "restart_agent_l0_model_view.v1",
  "failure_narrative": {
    "status": "available",
    "identity_kind": "primary",
    "events": [],
    "known_unknowns": []
  },
  "decision_evidence_view": {},
  "attempt_execution_context": {},
  "evidence_bundle": {
    "selection_coverage": {
      "status": "bounded",
      "semantics": "initial_model_view_selection",
      "collections": {
        "candidate_anchors": {"available": 14, "included": 6, "omitted": 8}
      }
    }
  },
  "projection_metrics": {}
}
```

`L0ModelFacingView.prompt_payload()` supplies the four evidence sections to L1
in the order shown above. L1 constructs the complete user-message envelope by
adding its generated `response_schema`. The L0B `schema_version` and
`projection_metrics` are not sent to the model; `projection_metrics` remains
trace/eval metadata. The exact internal `DecisionEvidence` remains a separate
trace artifact and is not duplicated verbatim in the initial model payload.

## Deterministic Failure Narrative

The narrative gives the model a compact causal reading order without claiming
more causality than L0A established. It selects applicable events from this
ordered vocabulary:

1. prior success of the same operation or artifact;
2. last applicable progress or checkpoint;
3. current operation start;
4. fault-like observation and any same-attempt recovery;
5. deterministic primary, otherwise selected terminal observation;
6. distributed fanout, cascade, and teardown summaries; and
7. later-progress outcome.

Each event contains a one-based `sequence`, typed `kind`, fixed-template
`summary`, source line when one canonical line exists, and resolvable evidence
references. Count and sample fields have one closed shape across event kinds.
Repeated rank fanout is one event with an exact occurrence/observer count and
bounded rank and line samples. The narrative never renders one event per rank.
`SCHEMA.md` owns the closed event and known-unknown shapes and vocabulary.

`status` is `available`, `partial`, or `not_available`:

| Status | Meaning |
| --- | --- |
| `available` | A primary or selected observation is represented and every included narrative reference resolves. |
| `partial` | Current-log progress or failure-surface facts are represented, but neither a primary nor selected observation is available. |
| `not_available` | No failure narrative can be constructed from typed current-log facts. |

`known_unknowns` states only mechanically established evidence boundaries, for
example `no_typed_cause_confirmation_selected` or
`direct_failure_object_not_observed`. It must not convert a failed free-text
search into a claim that a cause is absent. Unknowns use fixed identifiers,
bounded descriptions, and references to the coverage facts that justify them.

For the progress-log failure example, the narrative may read:

```text
1. checkpoint saves at iterations 656125 and 656250 completed
2. training progressed through iteration 656370
3. checkpoint save 656375 started
4. a progress-log traceback began
5. the terminal assertion was observed at line 40284
6. the incident fanned out to 1,542 ranks; repeated copies were compacted
7. no later application progress was observed
known unknown: no typed evidence establishes why progress-log state was unavailable
```

Every sentence above is rendered from typed L0A/Decision Evidence fields. The
last sentence describes an evidence boundary; it is not an inferred root cause.

## Selection Algorithm

L0B has three selection classes:

| Class | Material | Limit behavior |
| --- | --- | --- |
| Unconditional | Failure narrative; compact `decision_evidence_view`; `attempt_execution_context` containing only current-log scope and terminal timing | Outside bounded supporting-evidence limits. Narrative and Decision Evidence projection are deterministic and may summarize exhaustive L0A collections by exact counts, bounded samples, and references. |
| Required decision-anchor support, when available | The deterministic primary line, otherwise the selected-observation line, in an emitted context excerpt | Consumes a context-window slot before optional support and must survive merging and excerpt truncation. |
| Optional support | Additional occurrence groups, candidate anchors, earlier high-signal unrecovered candidates, and compact job/run orientation | Uses remaining collection slots in deterministic priority order. |

The minimum decision-anchor support predicate is:

- deterministic primary available: `kind=primary` and its line is required;
- no primary but a selected observation is available:
  `kind=observation_only` and its line is required;
- neither line available: `kind=none`, `status=not_applicable`;
- a required line without an L0A covering window: `status=unavailable`;
- a covering window exists and the emitted excerpt contains the exact line:
  `status=included`.

When a covering L0A window exists, L0B reserves it before optional windows and
reserves the decision-anchor line before excerpt compaction. If projection
limits still remove that line, L0B is unusable rather than silently presenting
incomplete decision context.
`selection_coverage.required_decision_anchor_support` records the identity kind,
predicate result, and required line.

Primary episode/incident details, nearest compatible progress or checkpoint
context, primary-linked cause confirmations, and cascade or teardown evidence
remain prioritized support. They are not part of the minimum predicate.

Within a collection, selection prefers:

- primary support, otherwise selected-observation support;
- root and observation identity-anchor support;
- terminal or unresolved evidence over retry-pending, recovered, or noisy observations;
- explicit cause confirmation over diagnostic suggestion;
- earlier initiating evidence over later cascade;
- progress boundaries and complete traceback summaries;
- representative evidence with wider causal/locality coverage.

Overlapping windows are merged. Repeated occurrence groups contribute counts and
bounded samples, not one copy per rank.

`attempt_execution_context` has this closed shape:

```json
{
  "scope": "current_log_only",
  "terminal_timing": {
    "coverage_status": "not_applicable",
    "incident_configured_timeout_seconds": null,
    "seconds_from_last_progress_to_terminal_incident": null,
    "terminal_detection_lag_seconds": null
  }
}
```

All three `terminal_timing` values are current-log observations derived from
the first terminal distributed incident. `incident_configured_timeout_seconds`
is the timeout reported by that operation, such as an NCCL `Timeout(ms)` value;
it is not an attrsvc or Restart Agent deadline. The other fields compare
log-observed progress and incident timestamps. Missing or unparsable inputs
produce `null` rather than an inferred duration. `coverage_status` is:

| Status | Meaning |
| --- | --- |
| `not_applicable` | No terminal distributed incident was observed. |
| `unavailable` | An incident was observed, but none of the three timing values could be derived. |
| `partial` | An incident was observed and only some timing values could be derived. |
| `complete` | All three timing values were derived. |

The timing remains incident-specific. L0B does not reinterpret an ordinary
single-rank terminal episode as a timed distributed incident.

Progress, checkpoint, operation, artifact, and later-progress/recovery facts
remain authoritative in internal `DecisionEvidence`. The narrative summarizes
the applicable sequence and `decision_evidence_view` carries their compact
model-visible values; neither is repeated in `attempt_execution_context`.

If declared per-section limits cannot include every candidate, L0B omits the
lowest-priority objects and records available, selected, omitted, and truncated
counts. It never silently drops or truncates required Decision Evidence facts.

L0B currently has no overall hard payload-size cap. Narrative and compact
Decision Evidence facts are mandatory, while supporting collections are
controlled by declared section limits. Exhaustive member lists, such as every
rank in a large fanout, remain in L0A/trace and are represented initially by
counts, bounded samples, and references. The L1 route owns total model-context
preflight. If estimated input plus the route's safety reserve leaves no response
capacity, L1 does not call the provider and that route is unusable; the
deterministic recommendation remains available.

`evidence_bundle.selection_coverage` is the compact model-visible accounting
surface. For each bounded collection it reports `available`, `included`, and
`omitted`. Context windows additionally report `selected_before_merge`,
`merged`, `truncated`, and `truncated_lines`. Context-window accounting MUST
satisfy:

| Boundary | Available population | Omitted meaning |
| --- | --- | --- |
| L0A context construction | Unique eligible seed lines | Seeds not emitted as `L0Bundle.context_windows` because of the L0A cap. |
| L0B model-view selection | `len(L0Bundle.context_windows)`, equal to L0A `selected_seed_count` | L0A-emitted source windows not selected for the initial model view. |

L0B does not recount seeds already omitted by L0A. Window merging is L0B
compaction, not omission.

Context-window operations have this fixed order:

1. Rank the L0A-emitted source windows and select at most the L0B window limit;
   every unselected source window is `omitted`.
2. Merge overlaps only within that selected set; each selected source window
   absorbed into another increments `merged`.
3. Render the post-merge windows; excerpt clipping increments `truncated` or
   `truncated_lines` without changing window membership.

A source window cannot be both omitted and merged. `included` is the number of
post-merge windows emitted to the model, while truncation is an orthogonal
property of those included windows. The counters therefore form this partition:

```text
available = selected_before_merge + omitted
included = selected_before_merge - merged
available = included + merged + omitted
```

Merging overlapping windows is neither omission nor truncation. `truncated`
counts rendered windows clipped by a window-level character boundary;
`truncated_lines` counts individually clipped source lines, so a truncated
window may have zero truncated lines. Status is `complete` only when every
collection has `omitted=0`, every context-window collection has `truncated=0`
and `truncated_lines=0`, and all accounting equations reconcile. Otherwise it
is `bounded`.

The detailed limits, utilization, source-line counts, integrity checks, and
payload size remain in trace-only `projection_metrics`. Both surfaces are
derived from the same accounting pass.

## References And Retrieval

L1 may expand beyond the initial view through route-advertised read-only tools.
Retrieved content is recorded separately and does not modify L0B.

## Attention Efficiency

The goal is not the smallest payload. The goal is enough relevant evidence for
a correct first-turn semantic assessment without burying the primary causal
story in noise.

The first objective is narrative closure: a model should be able to read the
current-log sequence, evidence boundary, and selected identity before it sees
large supporting collections. Bundle size alone is neither success nor failure:

- a larger view is justified when it carries distinct decision-relevant
  evidence;
- a smaller view is harmful if every model retrieves the same omitted context;
- an isolated route rereading lines already present in L0B is model/route
  behavior;
- tool retrieval of new decision-relevant lines is evidence of an L0B coverage
  gap;
- disagreement across models must be separated from a shared projection defect.

Repeated broad searches across routes for chronology, already bundled progress, the selected
terminal line, or full fanout are evidence that the initial view did not present
its existing facts effectively, even when those searches return no new lines.
That is an L0B attention-efficiency concern before it is labeled model
inefficiency.

## Invariants

- Exact internal `DecisionEvidence` is byte-equivalent across deterministic and
  enriched runtime branches. L0B does not mutate it. The model-visible
  `decision_evidence_view` is a deterministic compact projection whose scalar
  facts and references must agree with that exact object.
- The model-visible payload excludes the request's source `log_path`, its
  basename and parent-directory components, eval labels, case ids, and metadata
  inferred from that source location.
- Workload artifact paths observed in source content, such as checkpoint,
  dataset, configuration, or socket paths, remain eligible evidence. They are
  not source-location leakage and may appear in `DecisionEvidence`, excerpts,
  and affected-entity identity.
- Every source line has its original line number.
- A cited line is considered initially visible only when the paired source text
  or quote is present, not merely its reference number.
- L0B does not include `AttemptRecord`, `PriorAttemptView`, retry budgets,
  history conclusions, or final policy.
- Advertised tool schemas are provider request metadata; they are not duplicated
  in `L0ModelFacingView`.
- The response schema is generated and injected by L1; it is not an L0B output
  or evidence field.
- Projection is deterministic for the same L0A bundle and effective limits.
- Narrative event order, summaries, counts, and known-unknown identifiers are
  deterministic for the same inputs.

## Degraded Behavior

- If serialization fails, references do not resolve, line ranges are invalid,
  or accounting does not reconcile, L0B is unusable and the deterministic
  recommendation remains available.
- Deterministic-only execution need not build L0B.
- Truncation is allowed only when represented in projection metrics and the
  rendered section. Collection-level omission is represented in model-visible
  `selection_coverage`.
- Missing optional context remains an explicit omission, never fabricated text.

## Tracing

The trace must preserve the exact versioned L0B snapshot supplied to L1, or a
lossless artifact reference and payload hash. It also records:

- selection rules and resolved limits;
- available, selected, omitted, and truncated counts by collection;
- merged windows and source/model-facing line counts;
- characters and estimated tokens;
- budget utilization;
- integrity checks and anomalies.
- narrative event/reference coverage, fanout compaction, and per-section size.

## Service Logging

L0B emits one INFO completion event when context management succeeds or
degrades:

```text
event=restart_agent.l0b.completed status=<status> wall_clock_s=<seconds>
compact_json_chars=<count> estimated_tokens=<count>
selected_windows=<count> omitted_objects=<count>
truncated_objects=<count> projection_integrity=<status>
```

DEBUG emits `restart_agent.l0b.detail` with per-collection
available/selected/omitted counts, budget utilization, compaction and merged
window counts, truncation reasons, and deterministic payload hash. It does not
emit the model-facing payload itself.

## KPIs

Quality KPIs require gold evidence or controlled ablation:

| Quality KPI | Example |
| --- | --- |
| Required-evidence coverage | Gold requires the primary line, prior progress, and teardown; all three appear in the initial view. |
| Primary retention | The L0A deterministic primary is represented with a covering excerpt. |
| Supporting-context coverage | The checkpoint-load start and terminal decode error are visible together. |
| Compaction safety | Ten duplicate rank errors become one group with exact count and samples; no distinct cause is lost. |
| Narrative fact coverage | Gold requires prior success, last progress, operation start, terminal failure, fanout, and no later progress; all six appear in order with valid references. |
| Narrative semantic restraint | The narrative says no typed cause confirmation was selected; it does not invent storage, network, or workload ownership. |
| First-turn usability | L1 returns valid structured evidence without a retrieval tool. |
| Missing-context tool rate | `read_window` discovers a necessary line absent from L0B. |
| Bundled-evidence reread rate | A model retrieves lines already visible in L0B. Across several routes, repeated narrative reconstruction is an L0B presentation signal; isolated rereads remain model/route behavior. |
| No-new exploration rate | Broad tool calls return no decision-relevant context and do not change the assessment. Compare this across routes and L0B profile ablations. |

Operational metrics include:

- L0B projection wall time;
- JSON characters and estimated input tokens;
- narrative, Decision Evidence view, and supporting-evidence characters/tokens;
- narrative event count and fanout compression ratio;
- section budget used and limit;
- available/selected/omitted counts;
- merged window and truncation counts;
- payload serialization and reference-resolution status;
- deterministic hash consistency.

Downstream first-turn and tool metrics are attribution signals. They do not make
an individual projection automatically good or bad.

## Example

Assume L0A found:

- 400 repeated warnings followed by progress;
- one terminal failure episode around lines 1000-1030;
- 14 candidate anchors;
- 8 partially overlapping windows, the L0A maximum;
- a deterministic primary at line 1012.

L0B may render:

- a seven-event deterministic failure narrative with one explicit known
  unknown;
- a compact Decision Evidence view preserving the primary, progress, operation,
  outcome, exact fanout count, bounded fanout samples, and references;
- one merged primary episode window;
- nearest progress before the fault;
- one representative recovered-warning group with count 400;
- bounded supporting anchors and the first cascade;
- `selection_coverage` stating that 5 lower-priority anchors were omitted, 4
  source windows were selected, 2 selected windows were absorbed by merging,
  and 4 source windows were omitted.

The context-window accounting is therefore:

```text
available=8
selected_before_merge=4
merged=2
omitted=4
included=2
```

The model begins with the ordered failure story, then receives the compact facts
and supporting excerpts. Exact Decision Evidence, complete L0A objects, and the
raw snapshot remain retained for runtime, trace, L2, and optional tools.
