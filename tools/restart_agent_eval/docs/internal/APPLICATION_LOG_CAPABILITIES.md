# Application-Log Attribution Capabilities

> **Noncanonical exploration.** This note preserves broader application-log
> policy-context framing and earlier policy exploration for evaluation and
> presentation work. The product architecture and policy sources of truth are
> `DESIGN.md` and `L0A.md` through `L4.md` in the Restart Agent product
> documentation.

Application logs are one evidence source. They can describe what the workload
observed, when progress stopped, and whether execution continued, but they do
not always establish the underlying physical cause. The architecture therefore
separates **attribution evidence** from the **policy or operational action** that
consumes it.

## What Application Logs Can Support

Application-log analysis can produce typed evidence about:

- the primary observed failure mechanism and downstream cascades;
- the likely failure domain: workload, infrastructure/external, or unknown;
- affected ranks, nodes, operations, artifacts, and apparent blast radius;
- completed iteration/checkpoint progress before and after a fault;
- same-attempt recovery, terminality, and retry outlook;
- stable failure fingerprints for comparison across restart attempts; and
- confidence, supporting source lines, missing evidence, and ambiguity.

That evidence can support a larger operational architecture:

```text
application logs ---------> typed application attribution ----+
node / OS signals --------> future cross-source correlation --+--> policy consumers
scheduler signals -------->                                   |      - restart or stop
fabric / storage signals -->                                   |      - retry shaping
attempt history ----------------------------------------------+      - incident routing
                                                                    - isolation candidates
                                                                    - fleet analysis
```

The resulting uses extend beyond a binary restart decision:

| Use | What attribution contributes |
| --- | --- |
| Restart recommendation | Recovery outlook, progress, recurrence, and uncertainty. |
| Retry shaping | Whether a retry is general, bounded, conditional, or unlikely to help. |
| Progress protection | Evidence that useful work continues despite noisy warnings or recoverable faults. |
| Recurrence control | Same-mechanism roots, optional exact affected entities, and progress comparison across attempts. |
| Failure isolation | Rank/node/device locality hints for a future correlator; not proof by themselves. |
| Incident routing | Likely workload, storage, network, scheduler, or hardware ownership. |
| Fleet learning | Repeated mechanisms, regressions, endpoint behavior, and policy outcomes across jobs. |
| Explanation and evaluation | Auditable evidence, model interpretation, policy rationale, and later outcomes. |

Application logs alone generally cannot prove that a GPU, node, network link,
or storage service is faulty. Such conclusions require correlation with sources
such as `dmesg`, scheduler state, rank-to-device placement, fabric telemetry,
and storage health. The attribution result should preserve the observations and
uncertainty needed by that future correlation layer.

## What The Current MVP Can Do

The Restart Agent produces a progress-aware `STOP` or `RESTART`
recommendation by assessing failure domain and recoverability, then applying a
retry policy informed by same-job recurrence and progress across restart
cycles.

```text
domain + recoverability + evidence quality
    -> base retry rule and budget

same-job root recurrence + progress comparison
    -> consumed retry budget

base rule + budget state
    -> STOP or RESTART
```

### Target MVP Policy Matrix

`Allowed retries` counts retries after the first observed failure. The target
MVP current-attempt assessment is:

- **Failure domain** is `workload`, `infrastructure/external`, or `unknown`.
  Workload includes application code, data, configuration, and
  workload-selected framework behavior.
- **Restart recovery outlook** asks whether the same workload can recover in
  the next NVRx cycle after the declared process teardown, restart delay,
  allocation behavior, and external-state changes.
- **Evidence quality** is claim-specific. `Grounded` means L2 resolved the
  primary and policy-supporting evidence to the source log. `Established`
  means the current log directly establishes the claim; supported,
  hypothesis-only, unknown, or ungrounded claims cannot justify immediate
  `STOP`.

| Failure domain | Restart recovery outlook | Evidence or policy context | Retry rule | Initial action | Allowed retries |
| --- | --- | --- | --- | --- | ---: |
| Workload | Cannot recover | Grounded and established | `workload_unrecoverable` | `STOP` | 0 |
| Workload | Cannot recover or may recover | Grounded, but insufficient for immediate `STOP` | `workload_confirmation_retry` | `RESTART` | 1 |
| Workload | Unknown | Workload domain grounded; recovery uncertain | `workload_recovery_window` | `RESTART` | 2 |
| Workload | May recover or unknown | Planned rejected-iteration policy-context signature | Context-provided retry policy | `RESTART` | Context-defined; target uses 2 |
| Infrastructure/external | Any | Domain grounded | `external_recovery_window` | `RESTART` | Policy-configured; default requires evaluation |
| Unknown | Any | Domain unknown or ungrounded, or L1 unavailable/unusable | `unclassified_safety_retry` | `RESTART` | Policy-configured; default requires evaluation |

Managed recovery is policy outside the model. No declared policy context is
active in the current contract. The planned `rejected_iteration_retry_then_skip`
handler will match typed nonfinite-result, failure-iteration, observer, and L3
progress facts in L4; model prose alone will not activate it.

An unavailable L1 result is not a semantic domain, but it has the same L4
behavior as an unknown domain. The result provenance still distinguishes
`semantic_unknown`, `l1_unavailable`, and `l1_ungrounded` for model and endpoint
evaluation.

For every retryable rule, history is eligible only for an exact `job_id`, a
distinct earlier `cycle_id`, and an exact deterministic root-fingerprint
match. Eligible history determines whether a retry budget is consumed:

| History observation | Budget effect |
| --- | --- |
| Same root, progress advanced | Protect the retry; do not exhaust the budget. |
| Same root, progress same or regressed | Consume one retry. |
| Same root, progress unknown | Do not count it as proven no progress. |
| Different root | Do not consume this root's budget. |

Decision precedence is:

```text
grounded, established workload-unrecoverable evidence -> STOP
otherwise observed progress advance                   -> RESTART
otherwise exhausted retry budget                      -> STOP
otherwise                                              -> RESTART
```

The semantic assessment remains observable even when two combinations select
the same rule. Different semantic assessments need different L4 rules only
when they produce different behavior. A missing primary remains outside this
matrix: it restarts but cannot establish same-root recurrence. An explicit
scheduler time-limit event is ordinary observed evidence; scheduler time policy
and enforcement remain outside Restart Agent authority. Missing history
identity also prevents recurrence-based exhaustion.

Examples anchor the policy intent:

| Example | Policy interpretation |
| --- | --- |
| Grounded workload code defect that cannot recover unchanged | Immediate `STOP`; zero retries. |
| Permission/access failure that looks persistent but is not established | One confirmation retry. |
| Rejected nonfinite iteration matching the planned retry-then-skip signature | Context-defined two-retry budget counted by the handler's same-root, same-iteration history rule. |
| Storage, network, or hardware-local symptom | External recovery window; numeric default remains an evaluation question. |
| Unknown domain or unavailable L1 | Unclassified safety retry; numeric default remains an evaluation question. |

Retry count is an imperfect proxy for infrastructure recovery duration. An
external outage may require elapsed time, and a hardware-local fault depends on
whether the restart preserves or replaces the allocation. The target keeps
these budgets configurable rather than asserting an unsupported cycle count.

For one failed training attempt, the pipeline:

1. Converts the application log into complete typed L0A evidence, including
   progress, checkpoints, candidate failures, episodes, cascades, and locality.
2. Selects deterministic `DecisionEvidence` and publishes a model-independent
   recommendation.
3. Builds an attention-efficient L0B model view and optionally lets L1 inspect
   bounded raw context through read-only tools.
4. Obtains a structured semantic failure and recovery assessment from L1.
5. Grounds the model-selected evidence in L2 without silently replacing the
   model's semantic answer.
6. Compares the current deterministic fingerprint and progress with bounded,
   current-process attempt history in L3.
7. Applies deterministic L4 retry policy and returns an auditable `RESTART` or
   `STOP` recommendation with provenance, stage metrics, and traces.
8. Serves the first direct NVRx-to-attrsvc integration: progressive cycle
   registration, terminal background analysis, and nonblocking result probes.

The MVP is intentionally limited to application logs and restart
recommendation. It does not yet provide cross-source correlation, component
isolation or quarantine, persistent/distributed history, full progressive L0
tailing, or production-qualified model and policy accuracy. Those capabilities
can consume the same typed attribution contract without moving their policy
into log parsing or model prompting.
