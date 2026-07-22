# Progressive Analysis Target Contract

This document is canonical for the future Restart Agent progressive lifecycle.
Progressive execution is a primary production requirement, but it is not
implemented by the current terminal core. `STATUS.md` records implementation
state; this file records the target contract for the separate progressive change
chain.

Progressive execution changes when evidence work runs. It does not introduce a
second evidence schema, semantic analysis, history comparison, or retry policy.

## Lifecycle

The intended NVRx service sequence is:

```text
cycle start
  POST /logs analysis_intent=progressive
    -> validate/register the cycle-unique log path
    -> start bounded, non-authoritative evidence precomputation

cycle end or failure
  POST /logs analysis_intent=terminal
    -> combine retained state with the unread final log tail
    -> schedule authoritative L0-L4 finalization

result probes
  GET /logs?wait=false
    -> return completed, in_flight, or pending immediately
```

`GET /logs` with omitted `wait` or `wait=true` may join analysis and block until
a result or service timeout. `track_only` is service-local registration; it does
not invoke verdict-producing Restart Agent work.

The normalized, cycle-unique `log_path` is the service correlation key. The
service records `job_id` and integer `cycle_id` when available, but terminal
signals need not repeat them.

## Start And End Authority

Progressive start may inspect newly appended bytes, update deterministic
occurrence groups, collect progress/checkpoint facts, build candidate summaries,
and retain bounded context windows. It must not emit `STOP`, `RESTART`,
`decision_basis`, or another final policy result.

Progressive end is authoritative. It must:

- merge retained state with the unread final log tail;
- allow for bounded post-end log drain and rank interleaving;
- re-evaluate terminality, progress-after-fault, checkpoint, fingerprint,
  history, and retry-policy facts over the combined evidence;
- return the same external result schema as terminal execution; and
- trace missing, stale, evicted, or unusable progressive state before falling
  back to terminal analysis.

## Retained State

Progressive state is compact analyzer state, not a raw-log cache. It should
retain:

- source offset and observation metadata;
- normalized occurrence groups and deterministic candidate summaries;
- progress, checkpoint, failure-episode, and incident facts;
- bounded raw windows around recent tail, progress, checkpoint, first-fault, and
  top-candidate anchors;
- selection, truncation, eviction, and lossiness metadata; and
- request identity and anomalies required for finalization.

If raw lines age out, the structured candidate remains, including original line
numbers and context availability. A structured summary never substitutes for a
raw quote when the public result cites source evidence.

MVP service retention may be local in memory and is bounded by:

- `active_idle_seconds`: stop active tailing after no observed growth; becoming
  idle does not emit a final action;
- `max_active_states`: cap active progressive records; and
- `max_completed_results`: cap retained completed decisions.

An optional compact-state byte guard may evict lower-value raw windows first.
Structured summaries, anchors, offsets, progress/checkpoint facts, and lossiness
metadata have higher retention priority.

## Latency And Candidate Selection

NVRx owns the post-failure decision window. The analyzer consumes and records a
caller or service deadline; it does not define one universal numeric window.
Qualification measures latency from `progressive_end` to a usable L4 result,
decision-window hit rate, and work shifted before cycle end.

The deterministic fallback and every configured model route use the same L0A
state, Decision Evidence, immutable `PriorAttemptView`, schemas, and policy.
Model output arriving after cycle closure is abandoned and cannot revise the
action, published artifacts, or closed `AttemptRecord`. Future priority
selection and enriched prior-record selection are specified separately in
`PROFILE.md` and are not part of the implemented `collect_all` mode.

## Validation

Progressive qualification must verify:

- terminal and progressive executions converge on equivalent complete evidence
  for the same source bytes;
- post-end drain does not omit the initiating failure;
- retained summaries survive raw-window eviction without fabricating quotes;
- state loss degrades to terminal execution without changing policy semantics;
- terminal-versus-progressive divergence is reported; and
- post-`progressive_end` p50, p90, p99, and deadline-hit measurements use the
  production lifecycle rather than terminal request-to-result timing.
