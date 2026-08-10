# Restart Agent Schema Spec

This file is canonical for product configuration, public request/response,
internal stage, and trace data shapes. `L3.md` and `L4.md` own interpretation and
action rules.

## Contract Classes

### Public Product Contracts

| Object | Version |
| --- | --- |
| Product configuration | `restart_agent_config.v1` |
| Public analysis request | `restart_agent_request.v1` |
| Public analysis response | `restart_agent_response.v1` |
| Collect-all result | `restart_agent_collect_all.v1` |

These are caller-visible wire contracts. Unknown fields and unsupported schema
versions are rejected at the product boundary.

### Serialized Internal Contracts

| Object | Version |
| --- | --- |
| Persisted L0A evidence bundle | `restart_agent_l0_bundle.v1` |
| Decision Evidence | `restart_agent_decision_evidence.v1` |
| L0B model view | `restart_agent_l0_model_view.v1` |
| L1 model evidence | `restart_agent_evidence.v1` |

These objects are not general caller inputs. They are serialized because they
cross a provider, trace, replay, or evaluation boundary. Schema versions are
exact contracts; producers must not silently emit a new shape under an old
version.

### Operational Artifact Contracts

| Object | Version |
| --- | --- |
| Single-route CLI trace | `restart_agent_cli_trace.v1` |
| Collect-all CLI trace | `restart_agent_cli_collect_all_trace.v1` |
| Route-artifact manifest | `restart_agent_route_artifacts.v1` |
| Incremental status snapshot | `restart_agent_live_status.v1` |
| Incremental lifecycle event | `restart_agent_live_event.v1` |
| Deterministic recommendation artifact | `restart_agent_deterministic_recommendation.v1` |
| Optional evidence-object tool response | `restart_agent_evidence_objects.v1` |

These versions identify product-generated operational artifacts, not additional
analysis stages. Their lifecycle and commit-marker semantics are defined under
`Incremental Collect-All Artifacts`; tool response semantics remain in
`TOOLS.md`.

### Internal Stage Contracts

Internal stage inputs and outputs are immutable Python types. They are explicit
contracts even when they have no independent wire-schema version.

| Stage | Input | Output |
| --- | --- | --- |
| Request assembly | `RestartAgentRequest`, effective config, runtime-selected `PriorAttemptView` | `AnalysisExecutionContext` |
| Attempt-record assembly | L0 progress/deterministic facts or one L2 enriched fact update | immutable `AttemptRecord` replacement |
| L0A log interpretation | validated log path, `LogSnapshot` or replayed `L0Bundle` | `L0Bundle` |
| L0A decision selection | `L0Bundle` | `DecisionEvidence` |
| L0B attention projection | `L0Bundle`, `DecisionEvidence` | `L0ModelFacingView` |
| L1 semantic extraction | `L1EvidenceContext`, route settings, deadline | `L1EvidenceResult` containing `restart_agent_evidence.v1` when usable |
| L2 grounding and audit | `L2GroundingInput` | `L2Result` |
| L3 history comparison | `HistoryEvaluationInput` | `HistorySummary` |
| L4 deterministic policy | `L4PolicyInput` | `L4PolicyOutcome` |
| Response assembly | L0-L4 outputs and execution health | `AnalysisResult` / `restart_agent_response.v1` |

An internal contract receives its own version only if it later becomes a
persisted, replayed, provider-facing, or independently consumed wire object.

`retry_budget.v1` is an L4 behavior identifier emitted in policy results and
traces. It is not a nested request or configuration schema.

Implementation status matters when reading the contracts below. The terminal
L0-L4 pipeline and current per-invocation history input are implemented. The
stateful `RestartAgentRuntime`, `AttemptProgressSummary`, `AttemptRecord`,
runtime-selected `PriorAttemptView`, record control/export, and multi-dimension
L3 comparison are implemented contracts. `STATUS.md` records qualification
gaps rather than schema availability.

## Public Analysis Request

`RestartAgent.run()` accepts `RestartAgentRequest` or its exact JSON mapping and
returns an `AnalysisRun` containing the public result plus the exact trace,
L0A bundle, Decision Evidence, optional L0B view, and deterministic candidate owned
by that invocation. L0B is present when an L1 route is scheduled and may be
null for deterministic-only or log-unavailable execution. The same ownership
applies to `run_many()`.

```json
{
  "schema_version": "restart_agent_request.v1",
  "log_path": "/logs/job-123/cycle-2.log",
  "job_id": "job-123",
  "cycle_id": 2,
  "analysis_mode": "terminal"
}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | exact string | yes | `restart_agent_request.v1`. |
| `log_path` | absolute string | yes | One interleaved current-attempt log. |
| `job_id` | string or null | no | Exact MVP history boundary. |
| `cycle_id` | integer or null | no | NVRx restart-attempt order within the job. |
| `analysis_mode` | enum | no | `terminal` by default. Attrsvc owns progressive start/end orchestration and hands finalized L0A to the same runtime path. |

Validation:

- `schema_version` must exactly match `restart_agent_request.v1`;
- unknown request fields are rejected;
- `log_path` must be present and absolute;
- `cycle_id` must be an integer, not a numeric string or Boolean;
- a path unavailable at analysis time produces the documented
  `log_unavailable` result after request validation.

Literal zero is accepted for `cycle_id` only when it is the real caller-supplied
cycle number. The runtime does not substitute `0` or `"unknown"` for absent
`job_id`/`cycle_id`. If either is absent, analysis remains valid but history
lookup and current-attempt upsert are disabled and the reason is traced.

Attempt records, prior-attempt views, `retry_policy`, eval labels, and case
metadata are not public request fields.

## Internal Analysis Execution Context

**Boundary contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | `RestartAgentRequest` | Validated caller-owned request. |
| Input | selected `PriorAttemptView` | Immutable in-memory prior-attempt view from `RestartAgentRuntime`. |
| Input | effective product configuration | Retry counts and declared recovery capabilities. |
| Output | `AnalysisExecutionContext` | Immutable invocation context consumed by pipeline orchestration. |

The agent validates the public request and then builds one immutable
`AnalysisExecutionContext`:

```text
RestartAgentRequest
+ RestartAgentRuntime PriorAttemptView
+ effective restart-agent configuration
-> AnalysisExecutionContext
```

The context contains:

- the validated request;
- compact `prior_attempts` selected by the runtime attempt-record store;
- effective retry-budget counts from product configuration.
- effective declared recovery capabilities from product configuration.

This object is an internal stage boundary, not a caller-controlled wire shape.
Duplicate `(job_id, cycle_id)` records are resolved by explicit idempotent
seed/upsert semantics in library/unit tests; current and future cycle records
are excluded from the immutable in-memory view supplied to L3.

### Restart Environment Guarantees

The workload is unchanged, process state is recreated, normal restart delay
applies, hardware allocation may change, and external-service state may change.
These are immutable product guarantees encoded in the static L1 system prompt.
They are not configuration, request, execution-context, or L0B fields. They
describe possible restart transitions, not evidence that recovery will happen.

### Retry-Budget Configuration

```json
{
  "confirmation_retry_allowed_retries": 1,
  "bounded_retry_allowed_retries": 1,
  "general_retry_allowed_retries": 3
}
```

All retry counts are non-negative integers. The confirmation and bounded
budgets must not exceed `general_retry_allowed_retries`, and unknown fields are
rejected. L4 emits
`policy_version: retry_budget.v1` to identify the behavior that interpreted
these values.

### Declared Recovery Capabilities

`declared_recovery_capabilities` is a closed array in product configuration.
It is empty when omitted. The MVP accepts only:

```json
[
  {
    "capability_id": "bad_token_retry_then_skip",
    "behavior": "retry_then_skip",
    "applies_to": ["bad_token_or_window"],
    "required_entity_kind": "data_position",
    "history_match_scope": "root_and_entity",
    "allowed_retries": 2
  }
]
```

Every item requires exactly those fields. `allowed_retries` is a positive
integer that must not exceed `general_retry_allowed_retries`, duplicate
capability identifiers are rejected, and the MVP values for `behavior` and
`applies_to` are exact. The array is copied into `AnalysisExecutionContext` and
then into `L4PolicyInput`; L0-L3 do not consume it.

### Attempt Progress Summary

`AttemptProgressSummary` is the shared immutable L0 progress type stored once
at the top of every `AttemptRecord`:

```json
{
  "training_progress": "observed",
  "first_completed_step": 1,
  "last_completed_step": 418,
  "completed_step_delta": 417,
  "progress_marker_count": 42,
  "checkpoint_progress": "observed",
  "checkpoint_load_step": 0,
  "first_checkpoint_step": 100,
  "last_checkpoint_step": 400,
  "checkpoint_step_delta": 300,
  "checkpoint_marker_count": 4,
  "failure_position": "after_observed_training_progress",
  "progress_after_failure": "not_observed"
}
```

All numeric fields are integer or null. Counts are non-negative integers.
`training_progress`, `checkpoint_progress`, and `progress_after_failure` are
`observed`, `not_observed`, or `unknown`. `failure_position` is
`before_observed_training_progress`, `after_observed_training_progress`, or
`unknown`. `L0A.md` is canonical for progress observation semantics,
`RUNTIME.md` for attempt-record construction/deduplication, and `L3.md` for
cross-attempt comparison.

### Attempt Failure Facts

`AttemptFailureFacts` is the compact branch-specific observation stored in the
deterministic block or one enriched entry:

```json
{
  "source": "l0_deterministic",
  "root_fingerprint": "observed:runtimeerror:cuda_device_assert",
  "root_fingerprint_source": "observed_exception",
  "fault_outcome": "terminal",
  "primary_line": 1012,
  "identity_anchor_line": 1012,
  "identity_anchor_reason": "canonical_episode_terminal",
  "failure_iteration": 419,
  "affected_entity": null,
  "faulting_rank": "7",
  "faulting_node": "node-2",
  "faulting_gpu": "3",
  "rank_to_gpu_map": {"7": "3"}
}
```

The required fields are `source`, `root_fingerprint`,
`root_fingerprint_source`, and `fault_outcome`; other fields may be null or
empty. The deterministic block requires a non-empty `root_fingerprint` and
source before the record is eligible for storage. An enriched block may retain
null identity when L2 could not ground one; that branch is ineligible for L3
comparison. `source` is `l0_deterministic` for the deterministic block and
`l2_grounded` for enriched entries. Progress is deliberately absent because the
enclosing `AttemptRecord.progress` is shared across every route.
The semantic `failure_class` remains on `FailureEvidence` and analysis output; it
is not duplicated in this L3-facing recurrence contract.

`root_fingerprint` identifies the failure mechanism. Optional
`affected_entity` identifies the exact observed object:

```json
{
  "kind": "data_position",
  "identity": "token:42",
  "fingerprint": "affected_entity:data_position:3d97...",
  "evidence_line": 1012
}
```

Allowed kinds are `data_position` and `artifact`. The identity remains visible
for review; the fingerprint is an exact stable hash over kind and identity.
An artifact identity is replay-stable evidence from the failing operation: for
checkpoint load this is normally normalized checkpoint path plus iteration,
refined by an explicit shard/file/object when available. A decoder's
buffer-local offset, offending byte, rank, timestamp, and traceback location
remain observations and are excluded from exact entity matching.

### Attempt Record

`AttemptRecord` is temporally neutral: immutable replacements represent the
current cycle while it is open, and the final value appears under the same
contract as a prior attempt in a later cycle.

```json
{
  "job_id": "job-123",
  "cycle_id": 2,
  "progress": {
    "training_progress": "observed",
    "first_completed_step": 1,
    "last_completed_step": 418,
    "completed_step_delta": 417,
    "progress_marker_count": 42,
    "checkpoint_progress": "observed",
    "checkpoint_load_step": 0,
    "first_checkpoint_step": 100,
    "last_checkpoint_step": 400,
    "checkpoint_step_delta": 300,
    "checkpoint_marker_count": 4,
    "failure_position": "after_observed_training_progress",
    "progress_after_failure": "not_observed"
  },
  "deterministic": {
    "source": "l0_deterministic",
    "root_fingerprint": "observed:runtimeerror:cuda_device_assert",
    "root_fingerprint_source": "observed_exception",
    "fault_outcome": "terminal",
    "primary_line": 1012,
    "identity_anchor_line": 1012,
    "identity_anchor_reason": "canonical_episode_terminal",
    "failure_iteration": 419,
    "affected_entity": null,
    "faulting_rank": "7",
    "faulting_node": "node-2",
    "faulting_gpu": "3",
    "rank_to_gpu_map": {"7": "3"}
  },
  "enriched": [
    {
      "route_id": "gpt",
      "facts": {
        "source": "l2_grounded",
        "root_fingerprint": "observed:runtimeerror:cuda_device_assert",
        "root_fingerprint_source": "observed_exception",
        "fault_outcome": "terminal",
        "primary_line": 1012,
        "identity_anchor_line": 1012,
        "identity_anchor_reason": "l2_grounded_primary",
        "failure_iteration": 419,
        "affected_entity": null,
        "faulting_rank": "7",
        "faulting_node": "node-2",
        "faulting_gpu": "3",
        "rank_to_gpu_map": {"7": "3"}
      }
    }
  ]
}
```

Required fields are non-empty `job_id`, integer `cycle_id`, `progress`,
`deterministic`, and `enriched`. `enriched` is an array for serialization but
has unique `route_id` keys. Adding the same route again replaces its entry. An
initial L0 record has `enriched=[]`; completed L2 routes may add compact entries
before the invocation closes.

The record contains no raw logs, L1 transcript, citations, tool payloads,
`HistorySummary`, `L4PolicyOutcome`, token/latency metrics, or final decision.
Those remain in result and trace artifacts. MVP L3 compares only the
deterministic block; enriched entries are retained but policy-inactive.

`not_observed` means the relevant marker was absent from a fully scanned,
readable current-attempt log for which L0 recognized a compatible marker
dialect; it is not a claim that no unlogged work occurred. `unknown` is required
when log coverage, marker-dialect applicability, or comparability is
insufficient. L0A preserves this prerequisite as
`ProgressFacts.training_progress_dialect_recognized` and
`checkpoint_progress_dialect_recognized`. `progress_after_failure` additionally
requires readable source lines after the primary before absence can become
`not_observed`.

### Prior Attempt View

`PriorAttemptView` is an immutable ordered tuple of `AttemptRecord` objects
selected from the runtime-owned store for one invocation. It contains exact-job
records with `cycle_id` less than the current cycle and therefore excludes the
current and future attempts. It is an internal typed object, not a versioned
JSON artifact, disk format, or public request field.

### Runtime Attempt Record Control

The internal library control contract is:

```text
seed(AttemptRecord[], mode=replace|merge) -> status
records(job_id=None) -> immutable AttemptRecord[]
clear(job_id=None) -> status
```

This is an in-memory library/unit-test control surface, not part of
`restart_agent_request.v1`.

### Manual Attempt Record Fixture

For manual testing, `--attempt-records-json-in` reads a plain JSON array of
`AttemptRecord` objects and seeds the store before analysis.
`--attempt-records-json-out` atomically writes the complete post-upsert store as
the same array shape, ordered by `job_id` and integer `cycle_id`. The output can
be edited or copied to construct alternate L3/L4 scenarios and then reused as a
later input fixture.

The fixture has no wrapper, schema-version field, implicit location, or
automatic lifecycle. It is an explicit test artifact, not a public request
field, production persistence format, automatic checkpoint, or MCP history
operation.

The runtime selects prior attempts as:

```text
get_prior_attempts(job_id, before_cycle_id)
  -> select exact job_id
  -> select cycle_id < before_cycle_id
  -> order by integer cycle_id
  -> return the configured last N records as PriorAttemptView
```

`AttemptRecordAssembler` creates the initial deterministic record from L0 and
produces immutable same-key replacements when L2 adds or replaces one enriched
entry. Reanalysis of the same `(job_id, cycle_id)` replaces the record and
starts with an empty enriched list. L3 and L4 never mutate the record.

## L0A Complete Evidence Bundle

**Stage contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | validated log path | Current-attempt path selected from `AnalysisExecutionContext`. |
| Input | `LogSnapshot` | Immutable captured byte boundary, decoding choice, read mode, storage mode, stable line index, and logical-line access for the current log. |
| Optional input | replayed `L0Bundle` | Reuses a previously built bundle after log-path validation. |
| Output | `L0Bundle` | Complete immutable structured evidence for this log snapshot. |

`L0Bundle` is immutable structured evidence derived from the complete log. Its
top-level collections are:

- source identity: `log_path`, `byte_size`, `line_count`;
- path/access facts and namespace summary;
- `occurrence_groups`;
- `context_windows`;
- `candidate_anchors`;
- registry matches and cause confirmations;
- `deterministic_primary_candidate`;
- cascades, failure episodes, and distributed incidents;
- post-fault summaries;
- progress, checkpoint, setup, and run-progress summaries;
- operation/artifact comparisons;
- later-progress-after-fault observations;
- job metadata;
- evidence coverage, selection/lossiness summary, and anomalies.

The detailed collection semantics are canonical in `L0A.md`. L0A
does not emit an action.

`LogSnapshot` is an internal source-view contract, not a wire payload.
Production `storage_mode=indexed_file` retains compact line-start offsets and a
boundary-limited source handle; `memory` is available for provided snapshots
and tests. `line()`, `log_lines()`, and `context_before()` have identical
logical behavior in either mode. Progressive metrics record pending and
discarded incomplete-tail bytes.

### Failure Evidence

The primary and registry candidate payload shape is:

```json
{
  "failure_class": "checkpoint_metadata_decode_error",
  "signature": "UnicodeDecodeError while decoding metadata",
  "root_fingerprint": "observed:unicodedecodeerror:checkpoint_metadata",
  "root_fingerprint_source": "observed_exception",
  "fault_outcome": "terminal",
  "causal_role": "initiating",
  "failure_iteration": null,
  "data_position_fingerprint": null,
  "line": 12083,
  "rank": "4175",
  "phase": "setup",
  "node": null,
  "gpu": null,
  "affected_entity": {
    "kind": "artifact",
    "identity": "/checkpoints/job-1#checkpoint_iteration=622125",
    "fingerprint": "affected_entity:artifact:...",
    "evidence_line": 12050
  }
}
```

`failure_class` names the observed failure mechanism; it does not prescribe an
action. `causal_role` records structural position such as initiating, cascade,
or teardown. `affected_entity` is optional and identifies an exact grounded
artifact or data position involved in the failure. L0 derives it from
deterministic evidence; L2 may derive it from a model-proposed artifact only
after the path is found in source evidence.

## Decision Evidence

**Stage contract:** `L0Bundle -> DecisionEvidence`.

`DecisionEvidence` is the canonical deterministic subset selected once from
L0A and shared by deterministic and model branches:

```json
{
  "schema_version": "restart_agent_decision_evidence.v1",
  "deterministic_primary_candidate": {},
  "canonical_observed_identity": {},
  "selected_evidence_references": {},
  "failure_position": {},
  "progress_checkpoint_state": {},
  "operation_artifact_facts": [],
  "later_progress_recovery": {},
  "locality": {},
  "coverage_lossiness": {},
  "provenance": {}
}
```

References retain L0A object/line identity. They are provenance-only: an object
id is resolvable by L1 only when `get_evidence_objects` is advertised for the
route. Decision Evidence does not inline every raw excerpt and does not choose
an action.

## L0B Initial Model Evidence View

**Stage contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | `L0Bundle` | Complete structured source evidence. |
| Input | `DecisionEvidence` | Canonical policy-relevant facts and references. |
| Output | `L0ModelFacingView` | Bounded initial L1 payload plus projection metrics. |

```json
{
  "schema_version": "restart_agent_l0_model_view.v1",
  "decision_evidence": {},
  "attempt_execution_context": {},
  "evidence_bundle": {},
  "projection_metrics": {}
}
```

The L1 user message is one machine-readable per-invocation envelope. It carries
the static generated `response_schema` alongside request-specific
`decision_evidence`, `attempt_execution_context`, and `evidence_bundle`.
Transporting the schema in this envelope does not make it dynamic evidence.
Immutable restart guarantees remain in the static system prompt and are not
repeated here.
`attempt_execution_context` contains only current-log scope and terminal timing;
progress, checkpoint, operation, artifact, and later-progress facts remain
single-sourced in Decision Evidence. `projection_metrics` remain client trace
data. L0B is bounded and lossy by design; it must record selection, compaction,
truncation, size, and estimated-token facts.

The initial conversation trace stores `model_visible_payload`, the exact parsed
JSON object serialized into the user message. L2 visibility grounding consumes
that complete payload, plus subsequent tool results. It does not reconstruct a
smaller approximation from `evidence_bundle` alone.

## L1 Model Evidence Contract

**Stage contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | `L1EvidenceContext` | L0B model view plus the read-only tools advertised by the route. |
| Input | route settings and deadline | Model, endpoint, generation controls, tool policy, and remaining time. |
| Output | `L1EvidenceResult` | Provider-neutral execution result, transcript, calls, errors, and parsed evidence. |
| Usable semantic payload | `restart_agent_evidence.v1` | Validated model assessment inside `L1EvidenceResult.evidence`. |

L1 returns exactly one `restart_agent_evidence.v1` JSON object. Required
top-level fields are:

```json
{
  "schema_version": "restart_agent_evidence.v1",
  "analysis_status": "primary_identified",
  "primary_failure": {},
  "root_cause_assessment": {},
  "model_recovery_assessment": {},
  "related_failures": [],
  "evidence": []
}
```

`analysis_status` is `primary_identified`, `no_failure_observed`, or
`insufficient_evidence`. `primary_failure` is null unless a primary was
identified.

The model-visible `response_schema` and client validator are generated from the
same `L1ResponseContract`. It owns closed field sets, required fields, enums,
array limits, confidence bounds, evidence support tags, and non-primary
semantics. A contract change must therefore update one executable source and
its tests rather than separate prompt and parser descriptions.

An L1 result is **usable** when the route completes before the whole-analysis
deadline without provider timeout or output truncation, the extractor reports
success with a parsed evidence object, and that object passes
`L1ResponseContract`. Provider retries, earlier failed HTTP attempts, or
unsupported tool requests may make an otherwise usable route `degraded`; they
do not erase its valid semantic payload. L2 advisory findings likewise do not
make L1 unusable. Provider timeout, truncation, missing/unparseable output, or a
contract-invalid object is unusable and leaves the deterministic branch
authoritative.

Non-primary results use canonical, deliberately uninformative values:

| Status | Primary | Root cause | Missing evidence | Recovery claims | Related/evidence |
| --- | --- | --- | --- | --- | --- |
| `no_failure_observed` | `null` | `No failure was observed in the supplied evidence.`; `unknown`; no plausible causes | empty | `unknown` value/status, confidence `1`; canonical no-failure rationale | empty |
| `insufficient_evidence` | `null` | `Insufficient evidence to identify a primary failure.`; `unknown`; no plausible causes | one or more gaps | `unknown` value/status, confidence `1`; canonical insufficient-evidence rationale | empty |

This prevents a non-primary response from smuggling an unauditable semantic or
policy conclusion through fields that look authoritative.

### L1 Primary Failure

```json
{
  "line": 12083,
  "causal_role": "initiating",
  "failure_identity": {
    "operation": "checkpoint_load",
    "mechanism": "metadata_deserialization",
    "component": "torch_distributed_checkpoint",
    "artifact_path": "/path/from/log/or/null"
  }
}
```

Required `causal_role` values are `initiating`, `cascade`, `teardown`, or
`unknown`. The nested L1 `failure_identity` contains semantic claims used during
grounding; it is not a history identity and is not copied into the product
result. L0/L2 derive `failure_class`, signature, fault outcome, locality,
`root_fingerprint`, and optional `affected_entity` from grounded client
evidence.

### Root-Cause Assessment

```json
{
  "summary": "Checkpoint metadata could not be decoded.",
  "status": "supported_but_unconfirmed",
  "plausible_causes": ["corrupt metadata", "transient read corruption"],
  "missing_evidence": ["same offset failure on another attempt"]
}
```

`status` is one of `established_by_current_log`,
`supported_but_unconfirmed`, `hypothesis_only`, or `unknown`.

### Model Recovery Assessment

```json
{
  "failure_domain": {
    "value": "workload",
    "status": "supported_but_unconfirmed",
    "confidence": 74
  },
  "retry_outlook_without_workload_change": {
    "value": "unknown",
    "status": "unknown",
    "confidence": 68
  },
  "rationale": "One failed read cannot distinguish persistent corruption from a transient read failure."
}
```

Allowed values:

| Field | Values |
| --- | --- |
| `failure_domain.value` | `workload`, `infrastructure`, `unknown` |
| `retry_outlook_without_workload_change.value` | `cannot_recover`, `may_recover`, `unknown` |
| either claim's `status` | `established_by_current_log`, `supported_but_unconfirmed`, `hypothesis_only`, `unknown` |
| either claim's `confidence` | integer `1..99` |

This object contains no action, user/not-user score, retry count, or history
assessment. Each claim has its own evidence status and confidence. Confidence
is retained for corpus calibration and is not an L4 threshold.

### Related Failure And Evidence Items

Related failures contain exactly:

```json
{
  "line": 12135,
  "causal_role": "cascade",
  "rationale": "Wrapper exception after the checkpoint decode failure."
}
```

These are diagnostic source references describing cascade, teardown, or unknown
relationships to the selected primary. Their line must be visible to the model.
They are not additional policy-claim citations and do not substitute for the
canonical `evidence` array.

Evidence entries contain:

```json
{
  "id": "e1",
  "line": 12083,
  "quote": "UnicodeDecodeError: ...",
  "supports": [
    "primary_failure",
    "root_cause_assessment",
    "failure_domain",
    "retry_outlook_without_workload_change"
  ]
}
```

The model may cite only line/quote text visible in L0B or returned by advertised
tools. A line number present only as provenance does not authorize a model-made
quote, even when that quote happens to match the source log. L2 may correct a
nearby line-number error only when the quoted text was visible at the resolved
line. Evidence IDs are unique and `supports` uses only the four closed claim tags
shown above. This array is the canonical citation source; the contract has no
second supporting-line list. The tags identify which claim a citation supports;
they do not encode claim strength or policy. L2 records unavailable or
ungrounded citations.

## L2 Result

**Stage contract**

`L2GroundingInput` is an immutable aggregate rather than a wire payload:

| Direction | Field | Type | Meaning |
| --- | --- | --- | --- |
| Input | `bundle` | `L0Bundle` | Complete structured evidence used to resolve source facts. |
| Input | `model_view` | `L0ModelFacingView` | Exact initial model-visible evidence and Decision Evidence. |
| Input | `l1_result` | `L1EvidenceResult` | Raw and parsed L1 output, transcript, and tool results. |
| Input | `source_log` | `LogSnapshot` | Immutable source used for citation and line grounding. |
| Output | - | `L2Result` | Grounded L1 semantics, enriched `AttemptFailureFacts`, and non-overriding audit diagnostics. |

L2 emits a typed result and a trace payload:

```json
{
  "used": true,
  "grounding_status": "grounded",
  "grounding_method": "exact_source_line",
  "audit_status": "clean",
  "primary_used": true,
  "enriched_failure_facts": {},
  "recovery_assessment_used": true,
  "recovery_assessment_policy_grounded": true,
  "root_cause_assessment": {},
  "model_recovery_assessment": {},
  "field_findings": {},
  "field_finding_codes": {},
  "findings": [],
  "citation_audits": [],
  "grounding_adjustments": [],
  "recovery_field_audits": []
}
```

Grounding status is `grounded`, `unavailable`, or `not_run`. Audit status is
`clean`, `resolved`, `findings`, or `not_run`. Recovery-field audits are
non-overriding and include `applied=false` when suggesting another
interpretation. `recovery_assessment_policy_grounded` is the mechanical
eligibility bit that lets L4 use L1 claims for a rule narrower than
`general_retry`: the primary plus domain and retry-outlook support must resolve
to source evidence, and root-cause status must be
`established_by_current_log` or `supported_but_unconfirmed`. `L4.md` then
applies the distinct status/value predicates for `workload_unrecoverable` and
`bounded_retry`.

L2 visibility is evaluated against the exact full `model_visible_payload`
recorded for the initial request and every returned tool payload. A source line
present in Decision Evidence or another initial-payload
section is therefore visible even when it is absent from the compact
`evidence_bundle` subsection. Canonical evidence requires visible line/quote
text; exact source-log content that the model never saw is not grounded support.
A related-failure line outside the line-reference visibility set is retained in
raw L1 trace but omitted from L2's audited related-failure view.

## L2 Enriched Attempt Facts

L0 supplies `AttemptRecord.deterministic`; each usable L2 route supplies one
entry with the same `AttemptFailureFacts` shape for
`AttemptRecord.enriched`. The runtime assembler applies both updates:

```json
{
  "route_id": "gpt",
  "facts": {
    "source": "l2_grounded",
    "root_fingerprint": "observed:unicodedecodeerror:checkpoint_metadata",
    "root_fingerprint_source": "observed_exception",
    "fault_outcome": "terminal",
    "primary_line": 12083,
    "identity_anchor_line": 12083,
    "identity_anchor_reason": "canonical_episode_terminal",
    "failure_iteration": null,
    "affected_entity": {
      "kind": "artifact",
      "identity": "/path/to/checkpoint",
      "fingerprint": "affected_entity:artifact:7f28...",
      "evidence_line": 12083
    },
    "faulting_rank": "4175",
    "faulting_node": null,
    "faulting_gpu": null,
    "rank_to_gpu_map": {}
  }
}
```

The route entry does not duplicate shared progress. It is inserted only while
the invocation remains open. L2 audit output remains separate and is not copied
into the compact attempt record.

## L3 History Summary

**Stage contract**

| Direction | Field | Type | Meaning |
| --- | --- | --- | --- |
| Input | `current_record` | `AttemptRecord` | Current attempt with shared progress and deterministic/enriched fact blocks. |
| Input | `fact_selector` | deterministic or enriched `route_id` | Chooses the current facts evaluated by this branch. |
| Input | `prior_attempts` | `PriorAttemptView` | Runtime-selected bounded exact-job earlier records. |
| Output | - | `HistorySummary` | Deterministic recurrence, locality, and observed-progress comparisons. |

These input fields are carried together as immutable
`HistoryEvaluationInput`. L3 receives no raw log, L1 transcript, model
confidence, or retry policy. MVP prior comparison always reads each prior
record's deterministic block even when enriched entries are present.

```json
{
  "available": true,
  "availability_reason": "ready",
  "same_job_attempts": 2,
  "matching_root_attempts": 1,
  "comparisons": [
    {
      "prior_cycle_id": 1,
      "selected_basis": "completed_step_and_checkpoint_step",
      "dimension_comparisons": [
        {
          "dimension": "completed_step",
          "prior_observation_status": "observed",
          "current_observation_status": "observed",
          "prior_value": 418,
          "current_value": 418,
          "delta": 0,
          "relation": "same"
        },
        {
          "dimension": "checkpoint_step",
          "prior_observation_status": "observed",
          "current_observation_status": "observed",
          "prior_value": 400,
          "current_value": 400,
          "delta": 0,
          "relation": "same"
        }
      ],
      "positive_progress_conflict": false,
      "relation": "same",
      "prior_attempt_progress": {
        "training_progress": "observed",
        "progress_marker_count": 42,
        "checkpoint_progress": "observed",
        "failure_position": "after_observed_training_progress"
      },
      "prior_fault_outcome": "terminal",
      "same_failure_iteration": true,
      "same_rank": false,
      "affected_entity_relation": "same"
    }
  ],
  "observed_advance_attempts": 0,
  "same_progress_attempts": 1,
  "regressed_progress_attempts": 0,
  "unknown_progress_attempts": 0,
  "no_observed_advance_attempts": 1,
  "matching_root_attempts_with_observed_training_progress": 1,
  "matching_root_attempts_before_observed_training_progress": 0,
  "matching_root_attempts_with_unknown_training_progress": 0,
  "exact_failure_position_attempts": 1,
  "same_rank_iteration_attempts": 0,
  "same_entity_attempts": 1,
  "different_entity_attempts": 0,
  "unknown_entity_attempts": 0,
  "consecutive_same_root_no_advance_attempts": 1,
  "consecutive_same_root_and_entity_no_advance_attempts": 1,
  "advanced_beyond_all_comparable_attempts": false,
  "advanced_beyond_all_same_entity_comparable_attempts": false,
  "cross_node_recurrence": false,
  "same_node_recurrence": false,
  "same_gpu_recurrence": false,
  "same_rank_only_recurrence": false,
  "rank_to_gpu_mapping_available": false
}
```

`HistorySummary` preserves both relative progress (`advanced`, `same`,
`regressed`, or `unknown`) and each comparable attempt's absolute progress
summary. Relative progress answers whether the current attempt advanced beyond
an earlier one. Absolute progress answers whether that earlier attempt failed
before training progress or after doing observable work. L3 reports both and
does not decide how much progress changes the retry policy. Entity relation is
reported independently as `same`, `different`, or `unknown`, and L3 exposes
both root-only and root-plus-entity aggregates.

The two consecutive fields are independent observations, not alternate views
selected by L3:

- `consecutive_same_root_no_advance_attempts` continues across entity changes;
- `consecutive_same_root_and_entity_no_advance_attempts` requires exact entity
  kind and fingerprint equality at every newest-first step.

A different or unknown entity stops only the second count. Root mismatch,
nonqualifying prior outcome, observed advance, or unknown progress stops both
as applicable. The current failure is not included in either prior-attempt
count.

`available=true` means history lookup was eligible and completed, including when
the selected view contains zero prior records; its `availability_reason` is
`ready`. `available=false` means lookup was disabled or identity was incomplete;
its reason is one of `history_disabled`, `missing_job_id`, `missing_cycle_id`,
or `missing_root_fingerprint`. Current/future-cycle filtering and its counts
belong to the runtime-history trace because the view supplied to L3 already
contains only prior records.

Each dimension relation is `advanced`, `same`, `regressed`, or `unknown`.
`selected_basis` records whether L3 used completed steps, checkpoint-save steps,
both positive-progress dimensions, or fallback failure position. If completed
step and checkpoint directions conflict, `positive_progress_conflict=true` and
the overall relation is `unknown`; the dimension results remain visible. L3
reports facts and does not report `STOP`, `RESTART`, or budget exhaustion.

### Multi-Attempt Comparison Example

For current cycle 4 with root `R` and completed step 100, L3 compares cycle 4
directly with every earlier same-root record. It does not build an adjacent
trajectory such as 4-to-3, then 3-to-2:

| Prior cycle | Root | Outcome | Completed step | Current-to-prior relation | Policy-qualifying |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `R` | `terminal` | 80 | `advanced` | yes |
| 2 | `R` | `progressed_after` | 90 | `advanced` | no |
| 3 | `R` | `terminal` | 100 | `same` | yes |

This produces `matching_root_attempts=3`, but retry accounting considers only
cycles 1 and 3 because `progressed_after` is not a terminal/unresolved prior
failure. The contiguous newest-to-oldest count is
`consecutive_same_root_no_advance_attempts=1`: cycle 3 is the same, then cycle
2 stops the scan because its outcome is nonqualifying. Counts describing the
shape of all matching history and counts consumed by policy are therefore
deliberately different.

## L4 Retry Policy Evaluation

**Stage contract**

| Direction | Field | Type | Meaning |
| --- | --- | --- | --- |
| Input | `primary` | `FailureEvidence` or null | Current primary selected from L0 deterministic or grounded L2 evidence. |
| Input | `history` | `HistorySummary` | L3 facts; L4 does not recompute history. |
| Input | `current_affected_entity` | `AffectedEntity` or null | Selected branch's exact-object identity; L4 does not derive it. |
| Input | `model_recovery_assessment` | `ModelRecoveryAssessment` or null | Grounded L1 domain and unchanged-workload retry outlook. |
| Input | `assessment_grounded` | Boolean | Whether L2 established policy-eligible support. |
| Input | `retry_policy` | `RetryPolicyConfig` | Effective confirmation, bounded, and general retry counts. |
| Input | `declared_recovery_capabilities` | tuple of `DeclaredRecoveryCapability` | Trusted configured recovery behaviors matched only in L4. |
| Output | `primary` | `FailureEvidence` or null | Primary propagated into response assembly. |
| Output | `retry_policy` | `RetryPolicyEvaluation` | Deterministic decision, basis, selected rule, and concurrent ledger accounting. |

The inputs are carried as `L4PolicyInput`; the outputs are carried as
`L4PolicyOutcome`.

`matching_prior_failures` counts prior matching attempts only. The total
matching occurrences observed at the current decision are
`matching_prior_failures + 1`.

```json
{
  "policy_version": "retry_budget.v1",
  "rule": "workload_managed_recovery",
  "decision": "RESTART",
  "decision_basis": "workload_managed_recovery_available",
  "retry_budget_exhausted": false,
  "exhausted_by": [],
  "general_root_ceiling": {
    "ledger_id": "general_root_ceiling",
    "applicable": true,
    "rule": "general_retry",
    "history_match_scope": "root_only",
    "allowed_retries": 3,
    "matching_prior_failures": 1,
    "observed_advance": false,
    "exhausted": false,
    "inapplicable_reason": null
  },
  "selected_rule_budget": {
    "ledger_id": "selected_rule_budget",
    "applicable": true,
    "rule": "workload_managed_recovery",
    "history_match_scope": "root_and_entity",
    "allowed_retries": 2,
    "matching_prior_failures": 1,
    "observed_advance": false,
    "exhausted": false,
    "inapplicable_reason": null
  },
  "failure_domain": "workload",
  "failure_domain_status": "supported_but_unconfirmed",
  "failure_domain_confidence": 74,
  "retry_outlook_without_workload_change": "may_recover",
  "retry_outlook_status": "supported_but_unconfirmed",
  "retry_outlook_confidence": 68,
  "recovery_assessment_policy_grounded": true,
  "current_evidence_qualified": false,
  "current_affected_entity": {
    "kind": "data_position",
    "identity": "token:42",
    "fingerprint": "affected_entity:data_position:3d97...",
    "evidence_line": 1012
  },
  "declared_recovery_capability_ids": [
    "bad_token_retry_then_skip"
  ],
  "applied_recovery_capability": {
    "capability_id": "bad_token_retry_then_skip",
    "behavior": "retry_then_skip",
    "applies_to": ["bad_token_or_window"],
    "required_entity_kind": "data_position",
    "history_match_scope": "root_and_entity",
    "allowed_retries": 2
  },
  "match_requirements": {
    "job_id": "exact",
    "root_fingerprint": "exact",
    "affected_entity": "exact",
    "progress": "no_observed_advance"
  }
}
```

`general_root_ceiling` is present for every history-eligible primary and always
uses `consecutive_same_root_no_advance_attempts`. It is inapplicable for
missing-primary and immediate `workload_unrecoverable` outcomes. A
missing-primary result has `rule=null`, `decision=RESTART`, and
`decision_basis=no_primary_failure`; it is a degraded-analysis result rather
than a retry rule.

`selected_rule_budget` is nullable. It is present only when the selected rule
declares a budget narrower than the general ceiling. Its
`history_match_scope` selects the corresponding L3 count:

- `root_only` uses `consecutive_same_root_no_advance_attempts`;
- `root_and_entity` uses
  `consecutive_same_root_and_entity_no_advance_attempts`.

`retry_budget_exhausted` is the aggregate of both ledgers. `exhausted_by`
contains zero or more of `general_root_ceiling` and `selected_rule_budget`; both
may be present. An immediate `workload_unrecoverable` STOP has
`retry_budget_exhausted=false` and is identified by `decision_basis`. For MVP,
a selected-rule `allowed_retries` value must not exceed the configured general
ceiling.

Each ledger is a `RetryLedgerEvaluation` with the exact fields shown above.
`inapplicable_reason` is null when active; otherwise it is one of
`missing_primary`, `immediate_unrecoverable`,
`rule_has_no_narrower_budget`, or `history_unavailable`.

Allowed non-null rules are `workload_managed_recovery`,
`workload_unrecoverable`, `confirmation_retry`, `bounded_retry`, and
`general_retry`. See `L4.md` for selection and action semantics.

## Public Analysis Response

**Boundary contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | L0-L4 outputs and execution health | Stage results selected by the deterministic or enriched branch. |
| Output | `AnalysisResult` | Typed public result assembled by the client. |
| Wire output | `restart_agent_response.v1` | Exact serialized caller-visible response. |

`restart_agent_response.v1`:

```json
{
  "schema_version": "restart_agent_response.v1",
  "decision": "RESTART",
  "decision_basis": "general_retry_available",
  "retry_policy": {},
  "failure_domain": "unknown",
  "result_provenance": {},
  "primary_failure": {},
  "root_cause_assessment": {},
  "model_recovery_assessment": {},
  "secondary_failures": [],
  "cascades": [],
  "evidence_coverage": {},
  "evidence": [],
  "justification": "..."
}
```

Required top-level fields are `schema_version`, `decision`, `decision_basis`,
`retry_policy`, `failure_domain`, `result_provenance`, `primary_failure`,
`root_cause_assessment`, `model_recovery_assessment`, `secondary_failures`,
`cascades`, `evidence_coverage`, `evidence`, and `justification`. Nullable
semantic fields remain present with `null` when unavailable.

`cascades` is the public downstream-effect collection. Every entry preserves
whether the effect is an ordinary cascade or teardown:

```json
{
  "failure_class": "observed_exception",
  "cascade_fingerprint": "teardown_cleanup:filenotfounderror",
  "causal_role": "teardown",
  "first_line": 12135,
  "last_line": 12135,
  "count": 1,
  "sample_lines": [12135],
  "rank_spread": ["4175"],
  "node_spread": [],
  "gpu_spread": [],
  "reason": "appears after primary candidate at line 12083",
  "relationship_rationales": [
    "Finalizer cleanup occurred after the checkpoint-load failure."
  ]
}
```

L0 owns structural `cascade` and `teardown` roles. When L2 grounds an L1
related-failure rationale for the same event, the rationale annotates the
deterministic group. A grounded L1-only downstream event is retained as a
single-entry group. Downstream events do not remain in `secondary_failures`.

### Result Provenance

The compact result provenance includes:

- `candidate_kind`: `deterministic` or `l1_enriched`;
- `evidence_source`;
- `model_contribution`;
- `history_contribution`;
- `result_quality`: `normal`, `degraded`, or `unusable`;
- `nvrx_use`: `eligible`, `eligible_degraded`, or
  `fallback_to_nvrx_default`;
- L1 execution status/issues;
- concise notes.

`result_quality` measures result usability, not failure ownership.

### Deterministic And Degraded Results

| Condition | Decision / basis | Primary | Quality |
| --- | --- | --- | --- |
| Accepted path unavailable or empty | `RESTART / log_unavailable` | null | `unusable` |
| Provider timeout/truncation with L0 primary | Deterministic L4 result | L0 candidate | normally `degraded` |
| Malformed L1 and no L0 primary | `RESTART / malformed_model_output` | null | `unusable` |
| L1 pending | Published deterministic L0/L3/L4 candidate | L0 candidate or null | normally `degraded` |

## Collect-All Result

`RestartAgent.run_many().result`
returns:

```json
{
  "schema_version": "restart_agent_collect_all.v1",
  "deterministic_result": {},
  "model_results": [
    {
      "route_id": "qwen_fast",
      "model": "nvidia/qwen/eccn-qwen-235b",
      "endpoint": "https://inference-api.nvidia.com/v1",
      "credential_ref": "LLM_API_KEY_FILE",
      "execution_status": "completed",
      "l1_usable": true,
      "analysis_result": {},
      "error": null
    }
  ],
  "shared_analysis": {}
}
```

All routes receive the same immutable L0A, Decision Evidence, L0B, public
request, `PriorAttemptView`, and deadline. Results are independent; collect-all
does not vote, merge, or select a winner.

`analysis_result` is the route's final product response after L2, L3, and L4;
it is not the raw L1 model answer. The exact raw/parsed model response and tool
conversation remain in that route's trace/transcript.

## Product Configuration

Product configuration uses `restart_agent_config.v1`.
`CONFIGURATION.md` owns its complete field catalog, defaults, constraints,
inheritance, credential handling, and fingerprint semantics. The maintained
example is `examples/attribution/restart_agent.json`.

Configuration is supplied independently from `restart_agent_request.v1` and
attempt history. Parsing emits immutable route/runtime settings plus a
credential-free `effective_config` and stable `config_fingerprint`; those
identity fields are recorded in results and traces.

## Trace Contract

The detailed trace is local/out-of-band from the compact public result. It must
preserve raw stage behavior before downstream interpretation.

| Layer | Required trace content |
| --- | --- |
| Runtime attempt records | Availability/reason, configured per-job and total bounds, records before/after, eviction, deterministic creation, route-keyed enriched updates, same-key generation/replacement, close/timeout state, rejected late updates, and operation timing. |
| Progressive service | State transitions, generation, poll/growth counts, source identity and offsets, bytes ingested/reread, malformed-byte replacements, resets/eviction, drain outcome, terminal-to-L0A/deterministic/route/completion timing, finalized L0A hash, pre-end work, and terminal-degradation reason. |
| L0A | Bundle schema, source size, source decode time, source index/classification time, canonical reduction time, coverage/lossiness, primary, and fingerprint provenance. |
| Decision Evidence | Selection timing, deterministic primary, canonical identity, references, and shared-object provenance. |
| L0B | View schema, projection timing, selected/compacted/truncated counts, characters, estimated tokens, and payload hash. |
| L1 | Exact credential-free requests, prompts/messages, advertised tools, raw and parsed responses, model/tool calls, retries, provider errors, finish reasons, token usage, and timing. |
| L2 | Grounding status/method, citation audits, adjustments, findings, recovery-field audits, enriched identity, and timing. |
| L3 | Current facts source, history input summary, per-attempt comparisons, aggregates, and timing. |
| L4 | Full `retry_policy` evaluation, result provenance, decision/basis, and timing. |

The trace also records candidate readiness, analysis timeout, deterministic publication,
the candidate used by each result, route-selection state when applicable, and
anomalies. A debug/summary stream is optional and must not duplicate the
reconstructable trace.

## Incremental Collect-All Artifacts

The CLI may publish the following canonical stage-complete L0 files before
model routes finish:

```text
--l0-bundle-json-out          -> l0_bundle.json
--decision-evidence-json-out -> decision_evidence.json
--l0-model-view-json-out     -> l0_model_view.json
```

The first file is the replay envelope described by
`restart_agent_l0_bundle.v1`; the other two are the exact
`DecisionEvidence.to_payload()` and `L0ModelFacingView.to_payload()` objects.
All use same-directory temporary files and atomic replacement. They are absent
until complete, then immutable for the invocation.

The caller may declare final per-route paths with
`--route-artifact-manifest`:

```json
{
  "schema_version": "restart_agent_route_artifacts.v1",
  "routes": {
    "qwen-fast": {
      "result_json": "model.qwen-fast.result.json",
      "trace_json": "model.qwen-fast.trace.json"
    }
  }
}
```

Relative paths resolve from the manifest directory. Route IDs MUST exactly
match the configured routes. On route completion the CLI writes the complete
`restart_agent_cli_trace.v1` trace first and the `AnalysisResult` second. The
result file is the route commit marker: if it exists, its trace is already
durable. `--deterministic-json-out`, `--trace-json`, and `--result-json` similarly
select canonical deterministic-recommendation and final batch paths. The final batch
trace is written before its result.

When the CLI receives `--incremental-artifact-dir`, it publishes only this
local lifecycle projection while `collect_all` is running:

```text
<dir>/run_status.json
<dir>/events.jsonl
```

`run_status.json` uses `restart_agent_live_status.v1` and is a complete atomic
snapshot. `events.jsonl` is append-only; every line is a complete
`restart_agent_live_event.v1` object with a sequence number, UTC timestamp, and
elapsed time. Its `l0.status` is `pending`, `ready`, or `not_published`; a
`l0_artifacts_ready` event includes stage timings and canonical artifact paths.
The event stream reports canonical L0, deterministic recommendation, per-route, and
final batch paths as they become ready. The deterministic recommendation uses
`restart_agent_deterministic_recommendation.v1`. A route trace carries the normalized
public request, its complete analyzer trace, and shared L0, so a downstream
harness can review the route before the final batch trace exists.

Lifecycle artifacts are observational. They do not replace the canonical batch
result/trace, select a route, or modify an analysis result. Canonical JSON files
and status snapshots use temporary files plus atomic replacement. Readers see
each completed artifact as it becomes ready, never a partially serialized JSON
object. `events.jsonl` is the intentionally append-as-you-go interface.

## Progressive Service Shape

Attrsvc's implemented first-cut `POST /logs` request is:

```json
{
  "log_path": "/logs/job-123/train_cycle2.log",
  "user": "alice",
  "job_id": "job-123",
  "cycle_id": 2,
  "analysis_intent": "progressive"
}
```

`cycle_id` is an optional strict integer. When absent, the direct Restart Agent
backend infers it from `_cycle<N>.log`; an explicit value takes precedence.
`analysis_intent` is `track_only`, `progressive`, or `terminal`. Repeating the
same normalized path is idempotent when its resolved attempt identity is
unchanged.

The target progressive implementation adds service-local settings. They do not
belong to `restart_agent_config.v1` because attrsvc owns polling and result
retention.

| Attrsvc setting | Default | Validation |
| --- | ---: | --- |
| `NVRX_ATTRSVC_RESTART_AGENT_PROGRESSIVE_ENABLED` | `false` | boolean |
| `NVRX_ATTRSVC_RESTART_AGENT_PRE_END_POLL_SECONDS` | `180` | finite and greater than zero |
| `NVRX_ATTRSVC_RESTART_AGENT_ACTIVE_IDLE_SECONDS` | `900` | finite and at least the poll interval |
| `NVRX_ATTRSVC_RESTART_AGENT_MAX_ACTIVE_STATES` | `64` | positive integer |
| `NVRX_ATTRSVC_RESTART_AGENT_MAX_COMPLETED_RESULTS` | `3000` | positive integer |

The existing terminal-drain settings remain
`NVRX_ATTRSVC_RESTART_AGENT_LOG_POLL_SECONDS`,
`NVRX_ATTRSVC_RESTART_AGENT_LOG_QUIET_SECONDS`, and
`NVRX_ATTRSVC_RESTART_AGENT_LOG_MAX_WAIT_SECONDS`, with defaults of `0.25`, `5`,
and `40` seconds respectively. Live attrsvc analysis also requires an internal
10-second minimum observation period before quiet can establish convergence.
Direct CLI/library analysis treats an existing input as a known-complete
snapshot and does not wait for source convergence.

The progressive implementation introduces two internal typed contracts:

| Contract | Required content |
| --- | --- |
| `ProgressiveL0State` | Phase, source identity/boundary, read mode/chunk size, decoding choice, decoded-line and pending-tail counts, read offset, poll/chunk/growth/reset/replay/build counters, precompute timing, last activity, and last error. |
| `FinalizedL0A` | Canonical `L0Bundle`, `DecisionEvidence`, immutable indexed source `LogSnapshot`, source identity/byte/line boundary, canonical hash, and separate operational timings. |

These objects are not public wire schemas. `RestartAgentRuntime` accepts
`FinalizedL0A` directly; L0B and model tools must honor its immutable source
boundary.

`GET /logs?wait=false` returns `pending`, `in_flight`, `completed`, or `failed`
without starting or awaiting analysis. The response retains top-level
`status`, `result`, `wl_restart`, and `recommendation`; the full
`restart_agent_response.v1` payload is under `result` once available.
`ATTRSVC_INTEGRATION.md` owns this implemented HTTP lifecycle.
`PROGRESSIVE.md` owns the service sequence, L0 precompute state, finalization,
and late-result behavior.

## Eval Boundary

`EVALUATION.md` owns the product/harness boundary. This file defines only the
product artifacts and measurements consumed by evaluation; harness gold,
scores, panel reports, and aggregate schemas are not repeated here.
