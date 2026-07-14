# Restart Agent Tuning Guide

This guide defines the repeatable process for improving the NVRx Restart Agent.
It is intentionally separate from `DESIGN.md`: design states how the eval
product works; this document states how to use it without overfitting.

## Goal

Tune a versioned product and analysis profile toward four independent outcomes:

1. correct root-cause and action behavior;
2. efficient model interaction;
3. reliable endpoint operation;
4. production-compatible progressive latency.

A change is not successful merely because one log produces the desired action.

## Unit Of Work

Start with one log and a human review note. The note should become structured
gold before the case is used to claim quality. At minimum, capture:

- initiating failure and acceptable nearby line range;
- expected ambiguity;
- acceptable final action or action set;
- downstream cascades;
- claims the log does not support;
- any history facts needed to disambiguate the case.

The gold artifact is eval-only. It must never be passed to the product or model.

## Baseline Run

Run all declared targets through the same product checkout:

```bash
export RESTART_AGENT_EVAL_LOG_ROOT=/abs/path/to/logs
export RESTART_AGENT_EVAL_GOLD_ROOT=/abs/path/to/restart_agent_gold
export RESTART_AGENT_EVAL_RUN_ROOT=/abs/path/to/restart_agent_runs
./examples/review_one_log.sh "$RESTART_AGENT_EVAL_LOG_ROOT/path/to/input.log" models
```

The full structured trace is required for review. Successful runs do not create
a debug artifact. Process-level stderr is retained only when the product emits
a warning or fails; handled model and endpoint errors are recorded in the
structured trace and panel metrics.

The containing NVRx checkout is the default. Set
`NVRX_RESTART_AGENT_PRODUCT_REPO` only for an explicit cross-version
comparison.

The harness invokes product `collect_all` once. Product builds L0A, Decision
Evidence, and L0B once and runs all selected model routes concurrently over the
same immutable objects. The harness then materializes the familiar per-model
artifacts and panel summary from that batch. Preserve the complete run
directory; do not compare models from different bundle or product versions as
if they were one panel. Generated runs are disposable under the run root;
source logs and approved gold remain under their independent roots.

The run directory is printed immediately. During execution, `live/events.jsonl`
drives console updates for analysis start, shared L0 artifact readiness,
deterministic fallback readiness, route completion, and final completion.
Inspect run-root `l0_bundle.json`, `decision_evidence.json`, and
`l0_model_view.json` once L0 is ready; inspect `live/run_status.json` for the
current snapshot. The canonical `model.*.result.json`, `model.*.trace.json`, `model.*.review.json`,
and `model.*.review.md` files appear independently as each route completes.
`panel_summary.md` remains batch-final because it compares all routes.

Parallel batch latency is throughput-oriented panel data. Shared endpoint or
credential contention can change per-route latency, so isolated endpoint
benchmarking must use controlled concurrency and record that execution context.

## Review Order

Review the artifacts in causal order.

### 1. Human Expectation

Confirm the human note or gold label before interpreting model agreement. A
panel consensus is not ground truth.

### 2. L0 Evidence

Check:

- progress before and after the candidate fault;
- setup milestones such as successful checkpoint load or CUDA graph completion,
  without treating them as forward progress;
- the first coherent failure episode;
- distributed terminal incidents: whether same-epoch rank/process-group fanout
  collapsed into one incident, timing was derived from the last progress
  marker, root cause stayed unknown without confirmation, and the history key
  is invariant to rank/collective ordering;
- initiating anchors and complete bounded excerpts;
- cascades and teardown markers;
- diagnostic-only lines;
- selection, coverage, and lossiness metadata.

Inspect the typed L0B view exactly as supplied to the model, or compare it with
the complete L0A bundle:

```bash
python3 src/inspect_trace.py /path/to/model.trace.json \
  --view model-l0

python3 src/inspect_trace.py /path/to/model.trace.json \
  --view decision-evidence

python3 src/inspect_trace.py /path/to/model.trace.json \
  --view comparison
```

### 3. L1 Raw Analysis

Evaluate the model's answer before L2, L3, or L4:

- initiating mechanism and root-cause summary;
- root-cause status and missing evidence;
- primary and related evidence;
- failure domain;
- retry outlook without workload change;
- current-attempt persistence evidence;
- recovery requirement;
- confidence and unsupported claims.

Also inspect L1 execution status independently from semantics. A usable answer
after a timeout, provider error, or retry is `degraded`; this is endpoint/run
health and does not reduce or replace the confidence emitted by the model.

Do not infer model quality from the final `STOP` or `RESTART` alone.

### 4. L2 Grounding, Identity, And Audit

Check whether cited evidence exists and whether model claims line up credibly
with the source. L2 does not decide whether the model's semantic opinion is
correct and should be forgiving of harmless nearby-line differences.
When deterministic L0 or restart-transition facts contradict affirmative
current-attempt persistence evidence, inspect both the raw L1 value and L2's
separate audit finding. L2 does not rewrite the L1 semantics.

### 5. L3 History

Inspect same-job/exact-root attempts, typed progress relations and deltas,
exact-position observations, unknown comparisons, and no-observed-advance
streaks. L3 does not choose a threshold, map policy, or emit an action.

### 6. L4 Policy

Compare the raw L1 recovery assessment and L3 history facts with L4's policy
version, selected retry rule, allowed retries, matching prior failures,
exhaustion state, decision basis, and final action. Any deterministic policy
change must be visible as L4 behavior. Confirm that a first-cycle workload STOP
requires grounded `cannot_recover`, affirmative persistence, and a workload
change or external intervention requirement. Confirm that unknown/degraded
semantics receive the general retry budget rather than a fabricated certainty.

### 7. Behavioral Efficiency, Endpoint Reliability, And Route Outcome

Review model turns, tool calls, new context, duplicate/no-new-context calls,
and tokens as behavioral efficiency. Review provider attempts, retries,
timeouts, provider errors, and failed-call time as endpoint reliability.
Review end-to-end latency, deadline outcome, model-enriched versus fallback
contribution, and final NVRx usability as route outcome. Keep all four views,
including semantic quality, separate before applying promotion gates.

Count distinct conversational turns separately from provider attempts. A retry
of turn one is another attempt, not turn two. Timeout, HTTP-error, and generic
provider-error counts may describe the same failed attempt, so compare the raw
dimensions rather than summing them into one endpoint-issue total.

Inspect the traced per-turn context budget before classifying a context-window
rejection. If configured input plus requested output exceeded a declared model
limit, the defect belongs to the client/profile rather than model semantics or
endpoint reliability. Dynamic output-cap adjustment prevents the invalid
request and is reported separately; qualification must still verify that the
remaining output budget produces a complete answer.

Use `--llm-context-window-tokens` when a route's limit is known but the product
does not have a built-in model capability entry. Keep that value in the same
declared profile for production and eval.

For Qwen 235B tool-loop experiments, the default eval target uses
`qwen235b.experimental.one_tool_round.v1`: one tool-enabled round followed by
one tools-disabled final turn. Compare this profile with a tools-disabled
counterfactual before promotion. Raising the round limit is a profile delta;
judge it by semantic gain, final-answer dependency on newly retrieved context,
latency, and token growth rather than by successful completion alone.

The default model panel also includes the separate
`qwen397b.tools_supported.v1` candidate for
`nvidia/qwen/eccn-qwen3-5-397b-a17b`. It uses `LLM_API_KEY_OLD_FILE`, declares
a 262,144-token context window, and has no Qwen-specific tool-round cap. It
inherits the product's general tool-loop safety policy. Keep its semantic
quality, latency, endpoint reliability, and tool efficiency separate from the
Qwen 235B baseline; inclusion in the panel is evaluation, not promotion.

## Assign Ownership Before Changing Code

| Observation | Likely owner |
| --- | --- |
| Most models request the same relevant missing lines | L0 bundle coverage |
| One model rereads already supplied lines | Model/profile/tool efficiency |
| Models select a diagnostic advice line as primary | L0 role/anchor construction or prompt |
| Root cause is right but domain/recurrence/persistence/precondition is wrong | L1 prompt/model/profile |
| Conditional diagnostic is promoted to established cause or affirmative persistence | L1 prompt/model/profile |
| Common hardware-associated mechanism is promoted to confirmed hardware without corroboration | L0 taxonomy prior versus fact, then L1 uncertainty/calibration |
| Same-attempt rank fanout or the configured timeout duration is treated as recurrence/persistence | L1 prompt/model/profile |
| Latest-attempt assertion is treated as cross-restart persistence despite prior success of the same operation and artifact | L1 prompt/model/profile, with L2 advisory-audit visibility |
| Current-attempt continuation is treated as proof that the same component recovered | L0 wording or L1 prompt/model/profile |
| Model cites nonexistent evidence | L1 credibility plus L2 grounding/advisory-audit visibility |
| History match or recurrence effect is wrong | L3 history |
| Raw L1 assessment and L3 facts are reasonable but final action is wrong | L4 policy |
| Retry, timeout, gateway, or capacity failure | Endpoint/provider |
| Input plus requested output exceeds declared context window | Client/profile budget |
| Model output is malformed or omits required structure | L1 contract/model profile |
| Eval and production differ with the same declared profile | Harness/profile parity |

Tool use alone does not prove an L0 gap. A final primary or decision-relevant
support citation that depends on tool-only context is a strong bundle signal.
Tool-only teardown, scheduler cancellation, cleanup, or cascade context is
incidental unless it supplies a causal or policy premise. New lines that the
final answer does not use are model exploration. A raw line not present in an
excerpt may still add no semantic context when L0 already supplied the same
operation-history, progress, checkpoint, or incident fact structurally;
classify that as structured-fact redundancy. Duplicate context is usually a
model/profile signal. Count tool-driven turns separately from contract repair
and provider retry attempts.

## Change Hierarchy

Prefer changes in this order:

1. Generic L0 structure: progress, failure episodes, complete excerpts,
   diagnostic roles, cascades, or lossiness.
2. Generic prompt semantics: mechanism versus cause, workload versus
   infrastructure, retry outlook without workload change, current-attempt
   persistence evidence, recovery requirement, confidence, and uncertainty.
3. Model/profile controls: model route, thinking mode, temperature, token
   limits, tool advertisement, retries, and timeouts.
4. Generic taxonomy or pattern entries demonstrated across a failure family.
5. Deterministic L4 policy only for explicit, reviewable operational behavior.

Do not add a signature solely because it makes one known log pass. Do not turn
the system into lookup-based retrieval of past answers. Preserve ambiguity when
the current log cannot distinguish transient and persistent causes.

## Anti-Patterns And Safeguards

Instructions such as “do not decide `STOP`” or “treat registry hints as
provisional” do not guarantee model compliance. Preserve the raw response and
use deterministic validation/audit for anything important to downstream policy.

| Anti-pattern | Primary safeguard |
| --- | --- |
| Exact case/test/path/rank/timestamp encoded as a registry pattern | Static lint and PR review |
| Literal error string mapped directly to an action in the prompt | Prompt diff review and profile lint |
| L0 registry hint treated as causal truth | Raw L1 review, L2 credibility audit, and no-registry ablation |
| Bare component/transport token such as NVLink is treated as direct hardware proof | Split observed mechanism from corroborated hardware signature; score unsupported certainty |
| Diagnostic CUDA/framework advice selected as primary | Deterministic line role plus L2 finding |
| Generic downstream exception selected over the initiating operation | Gold RCA/cascade scoring and complete episode excerpts |
| Model emits prohibited `decision` or user/not-user fields | Contract validation; ignore/reject and report raw violation |
| Model cites unavailable evidence | L2 grounding and advisory audit |
| Model rereads bundled evidence | Tool-efficiency diagnostics |
| One-case prompt/registry improvement presented as general progress | Affected-family and unrelated holdout regressions |
| Cross-model consensus treated as ground truth | Human-approved gold |
| Silent client normalization makes a model look correct | Preserve raw L1 and report every L2/L3/L4 transformation |

Every prompt, registry, bundle-profile, or policy change should state:

- the generic hypothesis;
- why the change belongs to that layer;
- motivating positive cases;
- negative and ambiguous cases that must remain unchanged;
- holdout families;
- expected metric movement;
- rollback criteria.

When feasible, run an otherwise identical baseline/candidate ablation. A change
that only fixes its motivating example remains a hypothesis, not a promoted
improvement.

## Re-Run Discipline

For each change:

1. Record the product commit, eval commit, model/profile, and hypothesis.
2. Re-run the original case.
3. Re-run nearby cases from the same failure family.
4. Re-run a holdout set from unrelated families.
5. Compare L1 raw semantics, L3 history behavior, and L4 final behavior separately.
6. Compare latency, tokens, tool behavior, and endpoint events.
7. Reject the change if it fixes the target by degrading unrelated cases or by
   encoding path/case-specific information.

Existing artifacts can be re-summarized without another model call:

```bash
python3 src/summarize_review_panel.py /path/to/restart_agent_runs/path/to/log/<run-id>
```

## Repeated-Run Stability

After the corpus has enough human-approved gold, repeat the same log and route
profile and summarize the completed run directories with
`summarize_decision_stability.py`. Use at least ten comparable samples for an
initial observation. Do not mix changed source, product, profile, evidence,
prompt, or first-request identities in one score; the summarizer creates a new
cohort for each such change.

Use the stability report alongside gold accuracy. A stable wrong answer is not
qualified, and an accurate but frequently flipping answer is not operationally
predictable. Endpoint failures, semantic tuple changes, action flips, and tool
path variation have different owners and remain separate in the report.

## Model And Profile Selection

Do not choose a model route from a single panel. Aggregate the four views:

- **Semantic quality:** primary/root-cause and recovery-field correctness plus
  unsupported claims against human-approved gold. Model-side related-failure
  recall and final analyzer cascade correctness are reported separately so an
  omitted downstream failure does not masquerade as an RCA failure. Corpus
  analysis also measures ambiguity, confidence calibration, and false-`STOP`
  behavior.
- **Behavioral efficiency:** first-turn usability, model/tool/repair turns,
  duplicate/no-new-context calls, final-answer tool dependency, and tokens.
- **Endpoint reliability:** success rate, retries, timeouts, provider failures,
  and failed-call latency.
- **Route outcome:** model-enriched versus fallback use, terminal p50/p90/p99,
  post-`progressive_end` p50/p90/p99, decision-window hit rate, and final NVRx
  usability. Score the shared deterministic fallback and each available
  L1-enriched result separately, then report L1 action and full policy/action
  improvement and regression rates.

Locally capacity-controlled model routes and forwarded frontier routes should
be evaluated separately for capacity ownership and latency risk.

## Promotion Gate

A profile is eligible for promotion only when:

- the product and profile versions are immutable and reproducible;
- a representative human-approved corpus passes declared semantic thresholds;
- false-`STOP` and ambiguous-case behavior are acceptable;
- endpoint and token-limit behavior are acceptable;
- progressive replay meets the NVRx-owned decision window;
- no path, label, or expected-answer leakage is present;
- holdout regressions are reviewed;
- rollback to the prior profile is defined.

Until these gates exist and pass, results are feasibility or tuning evidence,
not production qualification.

## Automation Target

The long-term tuning loop should automate candidate generation, execution,
scoring, regression checks, and profile recommendation across:

- L0 bundle selection and prompt-facing budgets;
- L1 prompt/semantic-contract versions;
- model route, thinking mode, tools, sampling, timeout, and retry controls;
- L3 history filtering, progress comparison, and observation accuracy;
- L4 retry-rule selection and retry-budget behavior;
- L4 action mapping and candidate priority.

Automated tuning must hold unrelated dimensions fixed when measuring causality
and must apply quality/false-`STOP`/deadline/reliability gates before optimizing
efficiency. Human-approved gold remains the semantic authority. The harness may
recommend a versioned profile; it must not mutate or promote production policy
without explicit external approval.
