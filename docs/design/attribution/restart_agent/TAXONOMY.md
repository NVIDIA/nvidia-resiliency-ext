# Failure Vocabulary

Restart Agent labels describe different aspects of a failure. They are not
interchangeable, and no single label directly determines `STOP` or `RESTART`.
This document is the canonical glossary for those distinctions. Exact field
shapes live in `SCHEMA.md`; stage behavior lives in `L0A.md` through `L4.md`.

## Vocabulary Map

| Stage | Vocabulary | Question answered | Example |
| --- | --- | --- | --- |
| L0A | Registry role | How may a deterministic match contribute to evidence? | `root_candidate`, `cascade_candidate`, `cause_confirmation` |
| L0A/L1 | Causal role | Where does an event sit in the observed failure chain? | `initiating`, `cascade`, `teardown`, `unknown` |
| L0A/L2 | Failure class | What stable mechanism was observed by the client? | `observed_exception`, `cuda_oom`, `nccl_cascade` |
| L1 | Failure identity | What operation, mechanism, component, and artifact does the model infer? | checkpoint load, metadata deserialization, checkpoint component, exact path |
| L1 | Root-cause status | How strongly does the current log support the explanation? | `supported_but_unconfirmed` |
| L1 | Failure domain | Where does the failure originate? | `workload`, `infrastructure`, `unknown` |
| L1 | Retry outlook | Can the next cycle recover without changing workload code, data, or configuration? | `cannot_recover`, `may_recover`, `unknown` |
| L2 | History identity | Which stable root and optional affected entity may be compared across cycles? | exception fingerprint plus checkpoint path |
| L3 | Recurrence and progress | Has the same root recurred, and did the job advance between occurrences? | same root and artifact, checkpoint step advanced |
| L4 | Retry rule and action | Which rule applies, and are the general ceiling or a narrower selected-rule budget exhausted? | `general_retry` leading to `RESTART` |

## Structural Roles

### Registry Role

Registry roles are deterministic candidate-selection hints:

| Value | Meaning |
| --- | --- |
| `root_candidate` | May initiate a failure episode. |
| `cascade_candidate` | Usually follows another failure and should be checked against earlier evidence. |
| `cause_confirmation` | Corroborates a cause but need not be the first observed symptom. |
| `either` | Requires surrounding chronology to determine its role. |

A registry match is observed structure, not semantic root cause or policy.
`PATTERN_REGISTRY.md` owns the executable detectors and emitted failure classes.

### Causal Role

| Value | Meaning |
| --- | --- |
| `initiating` | Best-supported start of the selected failure chain. |
| `cascade` | Downstream consequence of an earlier failure. |
| `teardown` | Shutdown or cleanup activity after failure. |
| `unknown` | The available evidence cannot establish the relationship. |

Diagnostic advice, such as CUDA asynchronous-reporting warnings or debugging
suggestions, remains visible as context but is not a failure anchor.

## Semantic Assessment

L1 produces two independent recovery claims:

| Claim | Values | Meaning |
| --- | --- | --- |
| `failure_domain` | `workload`, `infrastructure`, `unknown` | Where the failure is attributed. Workload includes application code, model, data, configuration, and workload-selected framework behavior. |
| `retry_outlook_without_workload_change` | `cannot_recover`, `may_recover`, `unknown` | Whether the next normal NVRx cycle may recover while workload code, data, and configuration remain unchanged. |

Each claim has an independent evidence status:

| Status | Meaning |
| --- | --- |
| `established_by_current_log` | Direct current-log evidence establishes the claim. |
| `supported_but_unconfirmed` | Evidence favors the claim but material alternatives remain. |
| `hypothesis_only` | Plausible explanation without sufficient supporting evidence. |
| `unknown` | The log does not support a useful conclusion. |

Confidence from 1 to 99 records model self-confidence for calibration. It is
not an action score or an L4 threshold.

The retry outlook assumes failed processes are recreated, normal restart delay
occurs, hardware allocation may change, and mutable external-service state may
change. Same-attempt fanout cannot establish cross-cycle persistence; L3 owns
observed recurrence.

## Worked Example

```text
12083 [rank 4175] UnicodeDecodeError loading /checkpoints/step_600000/metadata
12135 [rank 4175] worker exited with exception
12150 scheduler: cancelling remaining tasks
```

The labels can coexist without collapsing into one conclusion:

```text
L0A registry role:       root_candidate
L1 causal role:          initiating
client failure class:    observed_exception
L1 failure identity:     checkpoint_load / metadata_deserialization
L1 root-cause status:    supported_but_unconfirmed
L1 failure domain:       unknown
L1 retry outlook:        may_recover
L2 history identity:     UnicodeDecodeError mechanism + exact checkpoint path
L3 observation:          no prior matching cycle yet
L4 result:               select the applicable retry rule and budget
```

The worker exit and scheduler cancellation are teardown consequences, not
competing root causes. A decode error may indicate a bad checkpoint or a
transient read failure; the current log need not prove which one occurred.

## Non-Substitution Rules

- An exception does not by itself imply workload ownership.
- A registry match does not establish semantic domain or recovery outlook.
- Diagnostic text cannot become a primary failure or history fingerprint.
- Cascade and teardown events cannot replace an earlier initiating failure.
- Same-attempt repetition across ranks is not cross-cycle persistence.
- L1 semantics do not directly choose an action; L3 supplies history and L4
  applies retry policy.
