# L0B Initial Model Evidence View

L0B is the deterministic attention-projection stage. It converts the complete
L0A evidence into the bounded structured payload supplied in the first L1 model
request.

L0B is intentionally lossy but explicitly accountable. It prioritizes what the
model should inspect first while preserving references to the complete evidence
and allowing route-advertised read-only tools to retrieve additional context.

`L0A.md` owns source interpretation. `TOOLS.md` owns optional retrieval APIs.
`SCHEMA.md` owns the exact `L0ModelFacingView` shape.

L0B ends at deterministic evidence selection and rendering; model execution,
grounding, history comparison, and action policy belong to L1-L4.

## Purpose And Authority

```text
L0Bundle + DecisionEvidence + L0B selection limits
  -> prioritize
  -> deduplicate
  -> compact
  -> render
  -> L0ModelFacingView
  -> L1 initial request
```

L0B is authoritative for:

- which L0A objects and source excerpts appear in the initial model payload;
- deduplication and merging of overlapping windows;
- per-section and total projection limits;
- truncation and omission metadata;
- projection integrity and deterministic payload hash.

L0B does not reinterpret the log, choose another deterministic primary, or
change Decision Evidence.

## Inputs And Output

| Direction | Contract | Meaning |
| --- | --- | --- |
| Input | `L0Bundle` | Complete typed evidence retained by the analyzer. |
| Input | `DecisionEvidence` | Canonical deterministic facts shared with the deterministic path. |
| Input | L0B evidence-selection rules and size limits | Deterministic limits for sections, excerpts, text, and references. |
| Output | `L0ModelFacingView` | Exact model-visible current-attempt evidence, excluding provider tool schemas. |

The output contains:

```json
{
  "schema_version": "restart_agent_l0_model_view.v1",
  "decision_evidence": {},
  "attempt_execution_context": {},
  "evidence_bundle": {},
  "projection_metrics": {}
}
```

Only the first three data sections are sent in the model user message.
`projection_metrics` remains trace/eval metadata.

## Selection Algorithm

L0B starts with mandatory material:

1. include exact `DecisionEvidence`;
2. include current-attempt execution context that is not already authoritative
   in Decision Evidence;
3. include the deterministic primary's episode/incident and covering windows;
4. include the nearest prior and next compatible progress/checkpoint facts;
5. include explicit cause confirmations and first downstream cascade/teardown;
6. include bounded supporting occurrence groups and candidate anchors;
7. use remaining budget for earlier high-signal unrecovered candidates and
   compact job/run orientation.

Within a collection, selection prefers:

- primary and identity-anchor support;
- terminal or unresolved evidence over recovered/noisy observations;
- explicit cause confirmation over diagnostic suggestion;
- earlier initiating evidence over later cascade;
- progress boundaries and complete traceback summaries;
- representative evidence with wider causal/locality coverage.

Overlapping windows are merged. Repeated occurrence groups contribute counts and
bounded samples, not one copy per rank.

If the configured budget cannot include every candidate, L0B omits the lowest
priority objects and records available, selected, omitted, and truncated counts.
It never silently drops required Decision Evidence.

## References And Retrieval

L1 may expand beyond the initial view through route-advertised read-only tools.
Retrieved content is recorded separately and does not modify L0B.

## Attention Efficiency

The goal is not the smallest payload. The goal is enough relevant evidence for
a correct first-turn semantic assessment without burying the primary causal
story in noise.

Bundle size alone is neither success nor failure:

- a larger view is justified when it carries distinct decision-relevant
  evidence;
- a smaller view is harmful if every model retrieves the same omitted context;
- rereading lines already present in L0B is model/route inefficiency;
- tool retrieval of new decision-relevant lines is evidence of an L0B coverage
  gap;
- disagreement across models must be separated from a shared projection defect.

## Invariants

- `DecisionEvidence` is byte-equivalent to the object used by the deterministic
  branch.
- The model-visible payload contains no absolute log path, basename, parent
  directory, eval label, case id, or path-derived hint.
- Every source line has its original line number.
- A cited line is considered initially visible only when the paired source text
  or quote is present, not merely its reference number.
- L0B does not include `AttemptRecord`, `PriorAttemptView`, retry budgets,
  history conclusions, or final policy.
- Advertised tool schemas are provider request metadata; they are not duplicated
  in `L0ModelFacingView`.
- Projection is deterministic for the same L0A bundle and effective limits.

## Degraded Behavior

- If serialization fails, references do not resolve, line ranges are invalid,
  or accounting does not reconcile, L0B is unusable and the deterministic
  recommendation remains available.
- Deterministic-only execution need not build L0B.
- Truncation is allowed only when represented in projection metrics and the
  rendered section.
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

## KPIs

Quality KPIs require gold evidence or controlled ablation:

| Quality KPI | Example |
| --- | --- |
| Required-evidence coverage | Gold requires the primary line, prior progress, and teardown; all three appear in the initial view. |
| Primary retention | The L0A deterministic primary is represented with a covering excerpt. |
| Supporting-context coverage | The checkpoint-load start and terminal decode error are visible together. |
| Compaction safety | Ten duplicate rank errors become one group with exact count and samples; no distinct cause is lost. |
| First-turn usability | L1 returns valid structured evidence without a retrieval tool. |
| Missing-context tool rate | `read_window` discovers a necessary line absent from L0B. |
| Bundled-evidence reread rate | A model retrieves lines already visible in L0B. This is an L1 efficiency issue, not an L0A miss. |

Operational metrics include:

- L0B projection wall time;
- JSON characters and estimated input tokens;
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
- 9 partially overlapping windows;
- a deterministic primary at line 1012.

L0B may render:

- exact Decision Evidence;
- one merged primary episode window;
- nearest progress before the fault;
- one representative recovered-warning group with count 400;
- bounded supporting anchors and the first cascade;
- metadata stating that 5 lower-priority anchors and 7 redundant windows were
  omitted or merged.

The model begins with the terminal story, while the complete L0A objects and raw
snapshot remain available through optional tools.
