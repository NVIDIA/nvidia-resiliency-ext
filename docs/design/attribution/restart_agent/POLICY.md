# Restart Agent Policy Spec

This file is canonical for the deterministic L4 `STOP` / `RESTART` policy.
`SCHEMA.md` owns field shapes; `DESIGN.md` owns stage architecture.

## Policy Boundary

The model does not choose an action and does not emit policy scores. Stage
ownership is:

| Stage | Policy-relevant output |
| --- | --- |
| L0 | Shared progress plus deterministic current-log `AttemptFailureFacts`. |
| L1 | Current-attempt semantic recovery assessment. |
| L2 | Grounded L1 evidence and one route-keyed enriched `AttemptFailureFacts` entry. L2 may report audit findings but does not rewrite L1 semantics. |
| L3 | Observational comparison with prior attempts. L3 does not choose thresholds or actions. |
| L4 | Versioned retry-rule selection, retry-budget accounting, `decision_basis`, and final action. |

Default bias is `RESTART`. `STOP` requires either grounded current-attempt
evidence that the unchanged workload cannot recover through the declared
restart transition, or exhaustion of a selected retry budget.

## L1 Recovery Assessment

L1 emits the `model_recovery_assessment` vocabulary defined by `TAXONOMY.md`
and serialized by `SCHEMA.md`. L4 consumes two independently qualified claims:
failure domain and retry outlook without workload change. Each claim has a
value, evidence status, and confidence. Confidence remains calibration data and
is not a policy threshold.

The declared restart transition may recreate process state, apply normal delay,
change hardware allocation, and encounter changed external-service state while
leaving the workload unchanged. Therefore:

- deterministic behavior inside the failed process does not by itself prove
  `cannot_recover`;
- repeated rank copies in one attempt do not prove persistence across restart;
- a possible transient explanation does not prove `may_recover`;
- `cannot_recover` with `established_by_current_log` requires direct evidence
  that normal restart effects are insufficient;
- cross-attempt persistence is not an L1 claim. L3 establishes observed
  recurrence from prior attempts with the same client-owned fingerprint.

## Configured Retry Policy

MVP uses the configuration below:

```json
{
  "bounded_retry_allowed_retries": 1,
  "general_retry_allowed_retries": 3
}
```

L4 emits `policy_version: retry_budget.v1` as behavior provenance. Callers do
not configure that identifier inside the retry-budget object.

`allowed_retries` counts retries after the first observed failure. A bounded
budget of one means: first failure `RESTART`; first matching prior failure plus
the current failure `STOP`.

## Rule Selection

L4 selects exactly one rule in this order:

1. `no_primary`: no usable primary failure exists.
2. `time_limit`: the selected primary is an expected time-limit failure.
3. `workload_unrecoverable`: all of the following grounded current-attempt
   conditions hold:
   - L2 resolved the primary and claim-tagged recovery support to the source log,
     and root-cause status is `established_by_current_log` or
     `supported_but_unconfirmed`;
   - `failure_domain.value=workload` and
     `failure_domain.status=established_by_current_log`;
   - `retry_outlook_without_workload_change.value=cannot_recover` and
     `retry_outlook_without_workload_change.status=established_by_current_log`.
4. `bounded_retry`: the L1 assessment is grounded, retry outlook is
   `may_recover`, and its status is `established_by_current_log` or
   `supported_but_unconfirmed`.
5. `general_retry`: all other combinations, including unknown semantics,
   infrastructure attribution, and ungrounded L1 output. Ungrounded L1 fields
   remain observable but cannot tighten the retry budget.

Rule table:

| Rule | Allowed retries | Initial action |
| --- | ---: | --- |
| `no_primary` | general budget | `RESTART` |
| `time_limit` | general budget, not exhausted by this failure | `RESTART` |
| `workload_unrecoverable` | `0` | `STOP` |
| `bounded_retry` | `bounded_retry_allowed_retries` | `RESTART` until exhausted |
| `general_retry` | `general_retry_allowed_retries` | `RESTART` until exhausted |

No individual field selects `STOP`. In particular, workload ownership does not
imply unrecoverability, a supported-but-unconfirmed `cannot_recover` claim does
not justify immediate STOP, and infrastructure attribution does not grant
unlimited retries.

## L3 History Input

L3 receives one immutable bounded `PriorAttemptView` selected by
`RestartAgentRuntime` from its current-lifetime `AttemptRecordStore`. The view
is fixed before fallback and model branches run. L0 supplies shared progress and
deterministic facts; the runtime assembler creates the current `AttemptRecord`
with an empty enriched list. Each usable L2 route supplies one enriched fact
block that the runtime may add or replace while the invocation is open. L3 and
L4 never mutate the record. Library/unit tests may
seed the same store through `AttemptRecordControl`; explicit CLI JSON fixtures
are test artifacts, not automatic persistence.

When history is disabled or the request lacks a non-empty `job_id` or integer
`cycle_id`, no prior view is selected. A prior view may be selected before L0
computes identity, but L3 reports history unavailable if the selected current
fact block lacks a root fingerprint. It never invents an `"unknown"` job or
cycle identity.

When all eligibility inputs exist but there are no prior records, history is
available with an empty view and zero recurrence. This is observably different
from disabled history or missing identity.

L3 compares only prior records satisfying the MVP history boundary:

- exact `job_id` match;
- distinct earlier integer `cycle_id`;
- exact selected `root_fingerprint` match.

For MVP, the current fallback branch selects `current_record.deterministic`, an
enriched branch selects its matching `current_record.enriched[route_id]`, and
every prior record is compared through `prior_record.deterministic`. Stored
enriched entries are observable but cannot change history counting.

For each comparable prior attempt L3 reports whether progress was `advanced`,
`same`, `regressed`, or `unknown`. L4 consumes
`consecutive_same_root_no_advance_attempts`.

### Progress Comparison

L3 compares `current_record.progress` with each `prior_record.progress` without
re-reading the log:

1. Compare `last_completed_step` when both attempts have observed, comparable
   completed training progress.
2. Compare `last_checkpoint_step` when both attempts have observed, comparable
   completed checkpoint saves.
3. When neither positive-progress dimension is comparable, compare
   `failure_iteration` as the weaker fallback position.
4. A positive delta is `advanced`, zero is `same`, and a negative delta is
   `regressed`. MVP uses any positive delta; it does not invent a minimum
   "substantial progress" threshold.
5. Combine available positive-progress dimensions as follows: any `advanced`
   and no `regressed` yields `advanced`; any `regressed` and no `advanced`
   yields `regressed`; all `same` yields `same`; both `advanced` and `regressed`
   yields `unknown`. The per-dimension deltas and conflict remain observable.
6. If the current attempt is `observed` and a reliably observable prior attempt
   is `not_observed`, the relation is `advanced`; the inverse is `regressed`.
   Any comparison involving `unknown` remains `unknown` unless another stronger
   comparable dimension establishes the relation.
7. Marker counts and first-to-last deltas describe the absolute attempt; they do
   not independently select the relative relation.

For example, checkpoints saved at `100`, `200`, and `400` have
`checkpoint_step_delta=300` and `checkpoint_marker_count=3`. Comparing a prior
last save of `400` with a current last save of `500` establishes advancement;
the count alone does not.

L3 also carries each comparable attempt's absolute `progress` summary:
whether training/checkpoint progress was observed, the first and last comparable
iteration and checkpoint markers, their deltas/counts, whether the failure was
before or after observed training progress, and whether progress followed the
failure. This is distinct from relative advancement. An attempt may have made
substantial observable progress while still reaching the same marker as a prior
attempt.

History counting rules:

- `same` and `regressed` count as no observed advance;
- `unknown` does not count as no advance;
- a current comparison that advanced beyond all comparable attempts prevents
  retry-budget exhaustion for this decision;
- rank, node, GPU, exact failure position, data position, and artifact identity
  remain observable comparison facts but do not independently select an MVP
  threshold;
- history never changes the semantic `failure_domain`.

The MVP L4 rule directly uses observed advancement and the consecutive
same-root no-observed-advance count. Absolute early-versus-progressed attempt
facts are required inputs and trace fields, but they do not select a different
retry budget until a versioned policy rule is calibrated. This preserves the
signal now without silently inventing an unreviewed threshold for "enough"
progress.

## Decision Algorithm

```text
assessment = grounded L1 recovery assessment, otherwise unavailable
rule, allowed_retries = select_rule(primary, assessment, retry_policy)
matching_prior_failures = L3.consecutive_same_root_no_advance_attempts

if no primary:
    RESTART / no_primary_failure
else if time limit:
    RESTART / time_limit
else if rule == workload_unrecoverable:
    STOP / workload_unrecoverable
else if L3 observed advancement:
    RESTART / observed_advance
else if matching_prior_failures >= allowed_retries:
    STOP / retry_budget_exhausted
else if rule == bounded_retry:
    RESTART / retry_recovery_available
else:
    RESTART / general_retry_available
```

Valid `decision_basis` values are:

- `log_unavailable`
- `workload_unrecoverable`
- `retry_budget_exhausted`
- `retry_recovery_available`
- `general_retry_available`
- `observed_advance`
- `no_primary_failure`
- `time_limit`
- `malformed_model_output`

## Examples

### Grounded Workload Code Failure

```text
domain: workload
domain status: established_by_current_log
retry outlook without workload change: cannot_recover
retry outlook status: established_by_current_log
history: unavailable
```

L4 selects `workload_unrecoverable`, `allowed_retries=0`, and `STOP`.

### Supported Permission Or Access Failure

```text
domain: workload
domain status: supported_but_unconfirmed
retry outlook without workload change: cannot_recover
retry outlook status: supported_but_unconfirmed
history: unavailable
```

The log directly grounds the failed access, while ownership, mode, or ACL state
may remain unverified. The claims are not directly established, so immediate
`STOP` is unavailable; L4 selects `general_retry` and preserves the first-cycle
retry. Matching history without observed advancement can exhaust that retry
budget later.

### Port Already In Use

```text
domain: workload
domain status: supported_but_unconfirmed
retry outlook without workload change: may_recover
retry outlook status: supported_but_unconfirmed
```

L4 selects `bounded_retry`. The first failure restarts. If the same fingerprint
recurs in the same job without observed advancement, the one-retry budget is
exhausted and the second failure stops.

### Unknown or Infrastructure Failure

```text
domain: infrastructure
domain status: supported_but_unconfirmed
retry outlook without workload change: unknown
retry outlook status: unknown
```

L4 selects `general_retry`. It does not stop on the first occurrence, but
repeated same-fingerprint failures without observed advancement eventually
exhaust the general budget.

### Progress Across Attempts

The same fingerprint recurs, but the current attempt reaches a later compatible
iteration or checkpoint than all comparable prior attempts. L4 emits
`RESTART / observed_advance`; sparse or incomparable progress is `unknown`, not
advance and not proven stalling.

## Failure And Degraded Paths

- A previously accepted log that is unavailable or empty at verdict time
  returns `RESTART / log_unavailable` with no primary.
- Provider timeout, truncation, malformed output, or unusable L1 evidence never
  makes model semantics policy-active.
- A grounded primary line is not sufficient by itself. Missing, unseen, or
  nonexistent recovery-supporting lines, and `hypothesis_only` or `unknown`
  root-cause status, keep `recovery_assessment_policy_grounded=false` and
  prevent the zero-retry rule.
- If L0 has a deterministic primary, L3/L4 still produce the generic fallback
  concurrently with L1.
- If no primary exists after malformed L1 output, return
  `RESTART / malformed_model_output`.
- L2 audit findings remain visible. Only source grounding determines whether
  L1 recovery facts can qualify for immediate `workload_unrecoverable`; audit
  suggestions do not silently replace model values.

## Deferred Workload-Managed Retry/Skip Policy

Some workloads deliberately replay the same data position several times before
skipping or quarantining it. The generic MVP does not encode a special
`workload_managed_retry_skip` field or rule. A later design must define:

- how the capability is declared and validated;
- whether iteration is a sufficient replay-position proxy;
- the workload-specific retry threshold;
- how that threshold composes with generic retry budgets;
- evaluation cases proving that the exception does not terminate a workload
  before its own recovery behavior can take effect.

Until then these cases use `bounded_retry` or `general_retry` from the grounded
generic assessment.

## L4 Output And KPIs

`retry_policy` records:

- `policy_version`;
- selected `rule`;
- `allowed_retries`;
- `matching_prior_failures`;
- `retry_budget_exhausted`;
- semantic inputs used by rule selection;
- `recovery_assessment_policy_grounded`;
- `current_evidence_qualified`;
- `observed_advance`;
- match requirements;
- final decision and basis.

Eval scores L4 separately on rule-selection accuracy, retry-count accuracy,
budget-exhaustion accuracy, decision/basis accuracy, false-STOP rate,
deterministic replay, and L4 latency. Model confidence is evaluated under L1
semantic calibration, not as an L4 score.
