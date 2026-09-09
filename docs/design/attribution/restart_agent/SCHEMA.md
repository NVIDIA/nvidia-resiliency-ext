# Restart Agent Schema Spec

This file is canonical for public request/response, serialized internal
contracts, stage contracts, and trace/artifact data shapes. `CONFIGURATION.md`
owns product configuration fields, defaults, resolution, and validation.
`L3.md` and `L4.md` own interpretation and action rules.

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
| L1 evidence-tool result | `restart_agent_tool_result.v1` |
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
| L3 history comparison | current `AttemptRecord` plus `PriorAttemptView` | `CycleHistoryComparison` containing shared progress and like-kind track histories |
| L4 deterministic policy | `L4CyclePolicyInput` | `L4PolicyOutcome` containing path selection and the selected track policy |
| Response assembly | L0-L4 outputs and execution health | `AnalysisResult` / `restart_agent_response.v1` |

An internal contract receives its own version only if it later becomes a
persisted, replayed, provider-facing, or independently consumed wire object.

`retry_budget.v1` is an L4 behavior identifier emitted in policy results and
traces. It is not a nested request or configuration schema.

Implementation status matters when reading the contracts below. The terminal
L0-L4 pipeline, stateful runtime, attempt records, prior-attempt selection,
record control/export, and current multi-dimension L3 comparison are
implemented. The observation-only fields and three-track policy path are part
of the development-stage v1 contracts and implementation. `STATUS.md` owns the
remaining production-qualification state.

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
  "cycle_id": 2
}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | exact string | yes | `restart_agent_request.v1`. |
| `log_path` | absolute string | yes | One interleaved current-attempt log. |
| `job_id` | string or null | no | Exact MVP history boundary. |
| `cycle_id` | integer or null | no | NVRx restart-attempt order within the job. |

Terminal/progressive lifecycle intent is not a request field. Attrsvc owns
progressive polling and finalization. It either invokes `analyze()` for a
terminal source or invokes `analyze_prepared()` with caller-finalized L0A
evidence; both enter the same mode-neutral analysis core.

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

Attempt records, prior-attempt views, `retry_policy`, `policy_contexts`, eval
labels, and case metadata are not public request fields.

## Internal Analysis Execution Context

**Boundary contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | `RestartAgentRequest` | Validated caller-owned request. |
| Input | selected `PriorAttemptView` | Immutable in-memory prior-attempt view from `RestartAgentRuntime`. |
| Input | effective product configuration | Retry counts and trusted policy contexts. |
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
- effective retry-budget counts from product configuration; and
- immutable trusted policy-context settings.

This object is an internal stage boundary, not a caller-controlled wire shape.
Duplicate `(job_id, cycle_id)` records are resolved by explicit idempotent
seed/upsert semantics in library/unit tests; current and future cycle records
are excluded from the immutable in-memory view supplied to L3.

### Cluster Execution Context

`ClusterExecutionContext` is a typed, immutable product contract with this
canonical payload:

```json
{
  "allocation_model": "homogeneous_node_pool",
  "workload_isolation": "exclusive",
  "replacement_hardware_bom": "equivalent",
  "replacement_software_bom": "equivalent",
  "replacement_resource_capacity": "equivalent",
  "replacement_resource_limits": "unchanged",
  "replacement_storage_access": "equivalent",
  "dependency_paths": [
    "compute_node",
    "scale_up_fabric",
    "scale_out_fabric",
    "distributed_storage",
    "service_control"
  ],
  "faulty_resource_handling": "independent_detection_and_quarantine"
}
```

One canonical renderer includes this contract in the static L1 system prompt.
It is not a configuration, request, `AnalysisExecutionContext`, or L0B field.
The homogeneous pool preserves hardware and software BOM, resource capacity
and limits, workload isolation, and storage access across eligible replacement
nodes. Workload code, data, configuration, and workload-selected software also
remain unchanged. Failed process state is recreated, normal restart delay
applies, and ephemeral process, node-local runtime, or external-service state
may change.

The dependency-path values are generic functional categories. Deployment
implementations such as NVLink/NVSwitch, InfiniBand or RoCE, and Lustre are
documentation/evaluation examples rather than prompt literals.

The independent health mechanism may quarantine malfunctioning resources, but
the contract does not prove that a malfunction occurred. A competing cause is
relevant when the failed operation depends on its path and the cause can
produce the observed failure mechanism; exact physical-component identity is
not required. Generic component fallibility is not evidence. Supported
workload and infrastructure alternatives may therefore produce
`failure_domain=unknown` while a restart-addressable mechanism independently
produces `retry_outlook_without_workload_change=may_recover`. Physical
replacement does not imply changed capacity, resource limits, workload demand,
data, configuration, or software behavior, and node replacement does not imply
repair of a persistent fabric, storage, or service fault.

### Retry-Budget Configuration

```json
{
  "concrete_confirmation_retry_allowed_retries": 1,
  "workload_confirmation_retry_allowed_retries": 1,
  "general_retry_allowed_retries": 2,
  "job_no_progress_allowed_retries": 3,
  "job_unknown_progress_allowed_retries": 3
}
```

All retry counts are non-negative integers. The two confirmation budgets must
not exceed `general_retry_allowed_retries`, and unknown fields are
rejected. L4 emits
`policy_version: retry_budget.v1` to identify the behavior that interpreted
these values. `general_retry_allowed_retries` is root-scoped when a root exists
and same-job-no-progress-scoped for the specified observation-only path.

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

`AttemptFailureFacts` is the compact track-specific observation stored in the
deterministic block or one enriched route track:

```json
{
  "source": "l0_deterministic",
  "identity_kind": "root",
  "root_fingerprint": "observed:runtimeerror:cuda_device_assert",
  "root_fingerprint_source": "observed_exception",
  "observation_fingerprint": null,
  "observation_fingerprint_source": null,
  "fault_outcome": "terminal",
  "primary_line": 1012,
  "selected_observation_line": null,
  "selected_observation_causal_role": null,
  "identity_anchor_line": 1012,
  "identity_anchor_reason": "canonical_episode_terminal",
  "failure_iteration": 419,
  "classifiers": ["nan_or_inf", "rejected_nonfinite_iteration"],
  "affected_entity": null,
  "root_observer_ranks": ["7"],
  "unattributed_root_occurrence_count": 0,
  "faulting_rank": "7",
  "faulting_node": "node-2",
  "faulting_gpu": "3",
  "rank_to_gpu_map": {"7": "3"}
}
```

`source` and `identity_kind` are required. `identity_kind` is `root`,
`observation_only`, or `none`:

| Identity kind | Required identity | Forbidden identity | Policy meaning |
| --- | --- | --- | --- |
| `root` | Non-null `root_fingerprint`, `primary_line`, and root identity anchor | Observation-only fields | Root/entity policy and history are available. |
| `observation_only` | Non-null `observation_fingerprint` and `selected_observation_line` | Root fingerprint, primary line, root anchor, root-observer facts, and affected entity | Root-independent general retry and same-job progress accounting are available. |
| `none` | None | Root and observation identities | Only root-independent same-job guards are available. |

Root and observation identities are mutually exclusive **within one
`AttemptFailureFacts` block**. They are not mutually exclusive across a whole
cycle: a route may publish one primary/root block and one observation-only
block. Null means unavailable and is never a comparable value. A record remains
eligible for storage and root-independent same-job progress guards even when
`identity_kind=none`.
`source` is `l0_deterministic` for the deterministic block and `l2_grounded`
for enriched entries. `root_observer_ranks` and
`unattributed_root_occurrence_count` preserve generic L0 observations about the
selected root. They are a paired tri-state contract: `null`/`null` means root
group association was unavailable; `[]` plus an integer means the group was
known but no attributed rank was parsed; a non-empty array plus an integer
means attributed root observers were parsed. Mixed availability is invalid.
`faulting_rank`, `faulting_node`, and `faulting_gpu` are parsed from the single
selected primary evidence line. They are not selected from
`root_observer_ranks`, which may contain several direct observers from the
complete associated root occurrence group.
`classifiers` contains policy-neutral typed observations associated with the
selected primary or observation line. It is empty when no classifier is
established; it does not replace either fingerprint or select policy before L4.
An L2 `same_canonical_incident` may reuse that L0 root occurrence group. A
`different_grounded_incident` must emit the observer pair as `null`/`null`
unless L2 has independently associated an occurrence group with the selected
root; it must not copy locality from the different L0 root.
Progress is deliberately absent because the enclosing `AttemptRecord.progress`
is shared across every route.

`failure_iteration` is the iteration position reported by the selected failure;
it does not claim that the iteration completed. For example,
`last_completed_step=418` and `failure_iteration=419` are consistent. L3 may
compare this position only as a weaker fallback when completed training-step
and completed-checkpoint dimensions are not comparable. It never updates or
substitutes for those positive-progress fields.

The semantic `failure_class` remains on `FailureEvidence` and analysis output; it
is not duplicated in this L3-facing recurrence contract.

`root_fingerprint` identifies the selected initiating failure mechanism.
`observation_fingerprint` identifies only a selected visible failure surface.
It may occupy a separate route track beside a primary. Matching observation
fingerprints do not establish matching roots. Optional `affected_entity`
identifies the exact grounded operation-associated object only for
`identity_kind=root`:

```json
{
  "kind": "artifact",
  "identity": "/checkpoints/job-1#checkpoint_iteration=622125",
  "fingerprint": "affected_entity:artifact:3d97...",
  "evidence_line": 1012
}
```

The sole active kind is `artifact`. The identity remains visible for review;
the fingerprint is an exact stable hash over kind and identity. An artifact
identity is replay-stable evidence from the failing operation: for checkpoint
load this is normally normalized checkpoint path plus iteration, refined by an
explicit shard/file/object when available. A decoder's
buffer-local offset, offending byte, rank, timestamp, and traceback location
remain observations and are excluded from exact entity matching.

### Cycle Failure Entry And Attempt Record

`AttemptRecord` is temporally neutral: immutable replacements represent the
current cycle while it is open, and the final value appears under the same
contract as a prior attempt in a later cycle.

Its failure portion is a `CycleFailureEntry`. The entry preserves independent
evidence tracks instead of selecting one before history comparison:

```text
CycleFailureEntry
  deterministic                         # shared L0 facts
  enriched[route_id].primary            # grounded L1 primary, optional
  enriched[route_id].observation        # grounded selected observation, optional
```

Each non-null track is an ordinary `AttemptFailureFacts` block and therefore
has exactly one identity kind. A route may have both `primary` and
`observation`; this does not create a combined identity. Progress is stored once
on the enclosing `AttemptRecord` and is shared by every track.

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
    "identity_kind": "root",
    "root_fingerprint": "observed:runtimeerror:cuda_device_assert",
    "root_fingerprint_source": "observed_exception",
    "observation_fingerprint": null,
    "observation_fingerprint_source": null,
    "fault_outcome": "terminal",
    "primary_line": 1012,
    "selected_observation_line": null,
    "selected_observation_causal_role": null,
    "identity_anchor_line": 1012,
    "identity_anchor_reason": "canonical_episode_terminal",
    "failure_iteration": 419,
    "affected_entity": null,
    "root_observer_ranks": ["7"],
    "unattributed_root_occurrence_count": 0,
    "faulting_rank": "7",
    "faulting_node": "node-2",
    "faulting_gpu": "3",
    "rank_to_gpu_map": {"7": "3"}
  },
  "enriched": [
    {
      "route_id": "gpt",
      "primary": {
        "source": "l2_grounded",
        "identity_kind": "root",
        "root_fingerprint": "observed:runtimeerror:cuda_device_assert",
        "root_fingerprint_source": "observed_exception",
        "observation_fingerprint": null,
        "observation_fingerprint_source": null,
        "fault_outcome": "terminal",
        "primary_line": 1012,
        "selected_observation_line": null,
        "selected_observation_causal_role": null,
        "identity_anchor_line": 1012,
        "identity_anchor_reason": "l2_grounded_primary",
        "failure_iteration": 419,
        "affected_entity": null,
        "root_observer_ranks": ["7"],
        "unattributed_root_occurrence_count": 0,
        "faulting_rank": "7",
        "faulting_node": "node-2",
        "faulting_gpu": "3",
        "rank_to_gpu_map": {"7": "3"}
      },
      "observation": null
    }
  ]
}
```

Required fields are non-empty `job_id`, integer `cycle_id`, `progress`,
`deterministic`, and `enriched`. `enriched` is an array for serialization but
has unique `route_id` keys. Each entry requires `primary` and `observation`,
either of which may be null but not both for a usable enriched route. Adding the
same route again replaces its entry. An initial L0 record has `enriched=[]`;
completed L2 routes may add compact entries before the invocation closes.

The record contains no raw logs, L1 transcript, citations, tool payloads,
`HistorySummary`, `L4PolicyOutcome`, token/latency metrics, or final decision.
Those remain in result and trace artifacts. L3 compares deterministic,
route-primary, and route-observation tracks independently. L4 later selects the
policy-active path; storing one path never deletes the others.

`not_observed` means the relevant marker was absent from a fully scanned,
readable current-attempt log for which L0 recognized a compatible marker
dialect; it is not a claim that no unlogged work occurred. `unknown` is required
when log coverage, marker-dialect applicability, or comparability is
insufficient. L0A preserves this prerequisite as
`ProgressFacts.training_progress_dialect_recognized` and
`checkpoint_progress_dialect_recognized`. `progress_after_failure` additionally
requires a finalized, fully scanned source boundary and a recognized
training-progress dialect before absence can become `not_observed`. The primary
may be the final source line; no trailing line is required once the captured
boundary is final.

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
route's primary/observation tracks. Reanalysis of the same
`(job_id, cycle_id)` replaces the record and starts with an empty enriched
list. L3 and L4 never mutate the record.

## L0A Complete Evidence Bundle

**Stage contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | validated log path | Current-attempt path selected from `AnalysisExecutionContext`. |
| Input | `LogSnapshot` | Immutable captured byte boundary, decoding choice, read mode, storage mode, stable line index, and logical-line access for the current log. |
| Optional input | replayed `L0Bundle` | Reuses a previously built bundle only for the same captured source boundary. |
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
- `selected_observed_failure`;
- cascades, failure episodes, and distributed incidents;
- post-fault summaries;
- progress, checkpoint, setup, and run-progress summaries;
- operation/artifact comparisons;
- later-progress-after-fault observations;
- job metadata;
- evidence coverage, selection/lossiness summary, and anomalies.

`selection_summary.context_window_selection` accounts for the bounded
context-window reduction:

```json
{
  "rule": "episode_cause_signal_registry",
  "eligible_seed_count": 12,
  "selected_seed_count": 8,
  "omitted_seed_count": 4,
  "limit": 8,
  "cap_hit": true
}
```

`rule=episode_cause_signal_registry` orders failure-episode boundaries, explicit
cause confirmations, deterministically sampled high-signal lines, and stable
registry-group representatives, in that order. `L0A.md` owns the exact sampler
and tie-breaking algorithm. `selection_summary.caps_hit` includes
`context_window_seeds` when eligible seeds are omitted. The counts must
reconcile as `eligible = selected + omitted`, and `selected` must equal the
emitted `context_windows` count. L0A emits no bundle when the rule is unknown or
the accounting is inconsistent.

The detailed collection semantics are canonical in `L0A.md`. L0A
does not emit an action.

Failure episodes may additionally carry a typed fault lifecycle:

| Field | Meaning |
| --- | --- |
| `lifecycle_family` | Stable normalized parser family, such as `rdma_port`; null for ordinary exception episodes. |
| `lifecycle_source_dialects` | Exact source dialects that supplied the lifecycle facts, such as `nccl_net_ib`; these do not assert the physical network protocol. |
| `lifecycle_entities` | Correlation identities retained by the lifecycle parser. |
| `lifecycle_fault_lines` | Initiating fault observations, including compacted fanout members. |
| `recovery_attempt_lines` | Recovery actions that do not by themselves prove recovery. |
| `recovery_confirmation_lines` | Direct same-entity recovery confirmations. |

A recovered lifecycle has `status=recovered` and no terminal exception. Later
progress remains in `first_progress_after`; it proves job continuation but is
not a substitute for direct component-recovery evidence. An unresolved
lifecycle may attach to a compatible terminal episode, which retains the
downstream terminal exception separately from the initiating lifecycle fault.

Replay has two validation levels:

- A persisted replay envelope validates its schema version, exact log path,
  source byte size, and source mtime. Its embedded bundle must agree with the
  envelope's path and byte size.
- A directly injected `L0Bundle` validates its log path, byte size, and logical
  line count against the captured `LogSnapshot` before reuse. Because a bare
  bundle does not retain complete filesystem provenance, its caller remains
  responsible for provenance that cannot be reconstructed from those fields.

The versioned replay envelope is the preferred boundary-safe API. Neither form
permits reuse merely because the current file occupies the same path.

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
  "observation_fingerprint": null,
  "observation_fingerprint_source": null,
  "fault_outcome": "terminal",
  "retry_lifecycle": null,
  "causal_role": "initiating",
  "failure_iteration": null,
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
artifact, direct failed object, or data position involved in the failure. L0
derives it from deterministic evidence. L2 prefers a grounded model-proposed
enclosing artifact and otherwise uses the grounded direct failure object.

`retry_lifecycle` is null when no explicit retry transition is recognized.
Otherwise it contains `state` (`pending`, `succeeded`, or `exhausted`) and
optional integer `attempt` and `max_attempts` values. The lifecycle describes
the observed operation retry, not the Restart Agent's cross-cycle retry budget.
`pending` maps the failure outcome to `retry_pending`; a later explicit success
maps it to `recovered`. An exhausted retry retains its ordinary terminality
assessment.

`deterministic_primary_candidate` is either null or one eligible `FailureEvidence`.
Only terminal or terminal-linked unresolved observations are eligible.
Retry-pending, recovered, and progressed-after observations remain in the
bundle but are never selected. Known cascade-only and teardown-only observations
are also retained but are never promoted to deterministic primary. L0 preserves a concrete
initiating identity over later confirmation,
promotes an explicit cause confirmation only for a linked cause-unknown process
termination, and otherwise may derive a normalized root identity only from a
structurally specific terminal exception supported as initiating. A visible
terminal symptom whose upstream cause is absent may become the selected
observation but not the primary. Bare termination without supporting cause
evidence yields null. `selection_summary.primary_selection_line` and
`primary_selection_basis` make the identity choice auditable.
`selection_summary.primary_episode_id` identifies the selected episode, and
`primary_episode_selection_basis` is
`earliest_eligible_initiating_episode`,
`eligible_registry_root_without_episode`, or `not_available`. Eligible episodes
are ordered by initiating identity line, episode start line, terminal line, and
stable episode id before identity precedence is applied within the selected
episode.

`selected_observed_failure` is null or one `FailureEvidence`. It is null when a
primary exists. When no primary exists, its root fields and affected entity are
null, its observation fingerprint fields are non-null, and its causal role is
`unknown`, `cascade`, or `teardown`. L0A selects it only after retry-pending,
recovered, progressed-after, diagnostic-only, and generic-boilerplate groups
are excluded. Independent tied terminal groups produce no selected
observation. Observation identity is descriptive and not root-policy-active.

## Decision Evidence

**Stage contract:** `L0Bundle -> DecisionEvidence`.

`DecisionEvidence` is the canonical deterministic subset selected once from
L0A and shared by deterministic and model branches:

```json
{
  "schema_version": "restart_agent_decision_evidence.v1",
  "deterministic_primary_candidate": {},
  "selected_observed_failure": null,
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

`deterministic_primary_candidate` and `selected_observed_failure` are mutually
exclusive. The selected observation is null when a primary exists or when L0A
cannot select one without forcing an ambiguous choice.
`canonical_observed_identity` records `identity_kind`, the
mutually exclusive root or observation fingerprint, selected line, and
selection basis. Observation-only identity is not a root-history key.

Trace references retain L0A object/line identity. An object id is resolvable by
L1 only when `get_evidence_objects` is advertised for the route. Without
resolved content, a trace reference identifies provenance but exposes no
evidence and cannot ground an L1 claim or L2 finding. Decision Evidence does
not inline every raw excerpt and does not choose an action.

The exact `DecisionEvidence` object is shared by deterministic and model
branches, so it is model-safe by construction. Its `provenance` object is
closed to `source`, `log_line_count`, `log_byte_size`, `log_rescanned`, and
`model_used`. The request source path, basename, parent-directory components,
eval labels, case ids, and source-location-derived identity are forbidden.
Private source identity remains in `L0Bundle` and trace. Workload artifact paths
observed in log content remain permitted evidence.

`locality.root_observer_ranks` and
`locality.unattributed_root_occurrence_count` use the paired tri-state contract
defined for `AttemptFailureFacts`. `coverage_lossiness.root_observer_facts`
contains:

```json
{
  "status": "complete",
  "reason": "root_occurrence_group_associated",
  "source": "complete_l0_occurrence_groups",
  "projection_caps_applied": false
}
```

`status` is `complete` or `unavailable`. An unavailable reason is
`no_deterministic_primary` or `root_occurrence_group_not_associated`. Current
L0A aggregation computes these facts before representative sampling, so
projection caps are never applied to them.
`status=complete` covers both a non-empty `root_observer_ranks` array and an
empty array: it means root-group association and counting completed, while the
array itself states whether any attributed observer rank was parsed.

## L0B Initial Model Evidence View

**Stage contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | `L0Bundle` | Complete structured source evidence. |
| Input | `DecisionEvidence` | Canonical policy-relevant facts and references. |
| Output | `L0ModelFacingView` | Deterministic failure narrative, compact Decision Evidence projection, bounded supporting evidence, and trace-only projection metrics. |

```json
{
  "schema_version": "restart_agent_l0_model_view.v1",
  "failure_narrative": {
    "status": "available",
    "identity_kind": "primary",
    "events": [
      {
        "sequence": 1,
        "kind": "last_progress",
        "line": 1000,
        "summary": "iteration 418 completed",
        "evidence_references": ["line-1000"],
        "occurrence_count": null,
        "rank_count": null,
        "unattributed_occurrence_count": null,
        "rank_samples": [],
        "line_samples": []
      },
      {
        "sequence": 2,
        "kind": "primary_failure",
        "line": 1012,
        "summary": "CUDA out of memory was observed",
        "evidence_references": ["line-1012"],
        "occurrence_count": null,
        "rank_count": null,
        "unattributed_occurrence_count": null,
        "rank_samples": [],
        "line_samples": []
      }
    ],
    "known_unknowns": []
  },
  "decision_evidence_view": {
    "canonical_observed_identity": {},
    "failure_position": {},
    "progress_checkpoint_state": {},
    "operation_artifact_facts": [],
    "later_progress_recovery": {},
    "locality": {
      "root_observer_count": 10,
      "root_observer_rank_samples": ["7", "9"],
      "unattributed_root_occurrence_count": 0
    },
    "coverage_lossiness": {},
    "selected_evidence_references": {}
  },
  "attempt_execution_context": {},
  "evidence_bundle": {
    "selection_coverage": {
      "status": "bounded",
      "semantics": "initial_model_view_selection",
      "required_decision_anchor_support": {
        "kind": "primary",
        "status": "included",
        "line": 1012
      },
      "collections": {
        "context_windows": {
          "available": 8,
          "included": 3,
          "omitted": 3,
          "selected_before_merge": 5,
          "merged": 2,
          "truncated": 0,
          "truncated_lines": 0
        },
        "candidate_anchors": {
          "available": 14,
          "included": 6,
          "omitted": 8
        }
      }
    }
  },
  "projection_metrics": {}
}
```

`L0ModelFacingView.prompt_payload()` contains `failure_narrative`,
`decision_evidence_view`, `attempt_execution_context`, and `evidence_bundle`,
serialized in that order. L1 owns the complete machine-readable user-message
envelope and adds its generated `response_schema` to those four sections. The
L0B `schema_version` and `projection_metrics` are not model-visible.
Transporting the response schema in the L1 envelope does not make it dynamic
evidence. Immutable restart guarantees remain in the static system prompt and
are not repeated here.

`failure_narrative` is a deterministic orientation projection over typed L0A
and Decision Evidence facts. Its event `kind` vocabulary, fixed-template
rendering, status values, ordering, fanout compaction, and permitted
`known_unknowns` are defined in `L0B.md`. It is not a semantic root-cause result.
Every event reference must resolve to the exact internal evidence used to
render it.

Each event has the closed shape shown above. `sequence` is contiguous and
one-based. `line` is the canonical source line when one exists and otherwise
`null`. Counts are exact integers or `null` when inapplicable/unavailable;
samples are bounded arrays and never substitute for the exact counts.
`evidence_references` contains every typed object or line required by the fixed
summary. The event `kind` vocabulary is:

```text
prior_operation_success
last_progress
last_checkpoint
current_operation_start
fault_observation
same_attempt_recovery
primary_failure
selected_observation
distributed_fanout
cascade_summary
teardown_summary
later_progress_outcome
```

Each `known_unknowns` entry has the closed shape
`{id, summary, coverage_references}`. Its `id` is one of
`no_typed_cause_confirmation_selected`,
`direct_failure_object_not_observed`, or
`failure_identity_not_selected`. A known-unknown entry is emitted only when its
typed coverage predicate was evaluated; it is not derived from unrestricted
free-text search.

`decision_evidence_view` preserves policy-relevant scalar facts and references
from exact internal `DecisionEvidence` but may replace exhaustive collections
with exact counts and bounded samples. In particular, a large
`locality.root_observer_ranks` list is represented by
`root_observer_count`, bounded `root_observer_rank_samples`, and the exact
`unattributed_root_occurrence_count`. The complete rank set remains in exact
internal Decision Evidence and L0A trace data. This compaction must not alter
identity kind, selected line, progress, operation/artifact, outcome, count, or
coverage semantics.
`attempt_execution_context` contains only current-log scope and terminal timing;
progress, checkpoint, operation, artifact, and later-progress facts remain
authoritative in exact internal Decision Evidence. Their narrative and compact
model-facing representations must agree with that source. `projection_metrics`
remain client trace data. L0B is bounded and lossy by design; it must record
selection, compaction, truncation, per-section size, and estimated-token facts.

The closed `terminal_timing` shape is defined in `L0B.md`. Its
`incident_configured_timeout_seconds`,
`seconds_from_last_progress_to_terminal_incident`, and
`terminal_detection_lag_seconds` values are derived only from the first
terminal distributed incident and timestamps observed in the current log.
They do not contain attrsvc convergence settings, Restart Agent deadlines, or
provider timeouts. `coverage_status` distinguishes no applicable incident from
an observed incident with unavailable, partial, or complete timing inputs.

`evidence_bundle.selection_coverage` is model-visible semantic accounting for
the initial projection. `status` is `complete` or `bounded`; `semantics` is
`initial_model_view_selection`.

`required_decision_anchor_support` has this closed contract:

| `kind` | Status | `line` | Meaning |
| --- | --- | --- | --- |
| `none` | `not_applicable` | `null` | Neither deterministic primary nor selected observation exists. |
| `primary` or `observation_only` | `unavailable` | integer | The required anchor exists, but L0A emitted no context window covering it. |
| `primary` or `observation_only` | `included` | integer | At least one emitted L0B context excerpt contains the exact required anchor line. |

When L0A has a covering window, removal of the required anchor line by L0B
projection limits makes L0B unusable; it is not serialized as `unavailable`.

Every collection reports non-negative integer `available`, `included`, and
`omitted` counts. Context windows additionally
report non-negative integer `selected_before_merge`, `merged`, `truncated`, and
`truncated_lines` counts. For context windows, `available` is
`len(L0Bundle.context_windows)`, equal to L0A `selected_seed_count`; it is not
L0A `eligible_seed_count`. Seeds omitted during L0A construction are reported
only in `L0Bundle.selection_summary`.

L0B first selects source windows up to its limit and marks the rest omitted. It
then merges overlaps within the selected set and finally renders or truncates
the merged windows. A source window cannot be both omitted and merged;
truncation does not change membership. Context windows satisfy
`available = selected_before_merge + omitted` and
`included = selected_before_merge - merged`, equivalently
`available = included + merged + omitted`. `omitted` means an L0A-emitted
window was not selected for the initial model view. `merged` means overlapping
selected source windows were represented by fewer model-facing windows and is
neither omission nor truncation. `truncated` counts windows clipped by a
window-level character boundary; `truncated_lines` counts individually clipped
source lines, so a truncated window may have zero truncated lines. Status is
`complete` only when all collections have zero omissions, context windows have
zero truncation counts, and the accounting reconciles; otherwise it is
`bounded`.
Detailed configured limits and integrity accounting remain in trace-only
`projection_metrics`; both surfaces are generated from the same counts.

The initial conversation trace stores `model_visible_payload`, the exact parsed
JSON object serialized into the user message. L2 visibility grounding consumes
that complete payload, plus subsequent tool results. It does not reconstruct a
smaller approximation from `evidence_bundle` alone. L2 also receives exact
internal Decision Evidence and L0A independently; model visibility is
nevertheless limited to the compact L0B payload and later tool results.

## L1 Model Evidence Contract

**Stage contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | `L1EvidenceContext` | L0B model view plus the read-only tools advertised by the route. |
| Input | route settings and deadline | Model, endpoint, generation controls, tool policy, and remaining time. |
| Output | `L1EvidenceResult` | Provider-neutral execution result, transcript, calls, errors, and parsed semantic payload. |
| Usable semantic payload | `restart_agent_evidence.v1` | Validated model assessment inside `L1EvidenceResult.semantic_payload`. Its nested `evidence` field contains claim citations. |

The runtime classifies every `L1EvidenceResult` exactly once with the typed
`L1ExecutionAssessment`. Traces and collect-all route results carry its complete
payload:

```json
{
  "execution_status": "completed",
  "result_quality": "degraded",
  "parse_status": "valid",
  "usable": true,
  "degraded": true,
  "evidence_present": true,
  "final_evidence_reason": null,
  "reason_codes": ["model_call_failed", "retry_used", "provider_http_error"],
  "unusable_reason": null,
  "errors": []
}
```

| Field | Closed values or rule |
| --- | --- |
| `execution_status` | `not_run`, `in_flight`, `completed`, `failed` |
| `result_quality` | `not_applicable`, `usable`, `degraded`, `unusable` |
| `parse_status` | `not_run`, `not_available`, `valid`, `malformed`, `contract_invalid` |
| `usable` | true only for `usable` or `degraded` quality |
| `degraded` | true only for `degraded` quality |
| `evidence_present` | whether the raw result contains a parsed semantic object; presence alone does not establish usability |
| `final_evidence_reason` | null when no additional final turn was selected; otherwise `contract_repair`, `forced_final_after_tool_exhaustion`, or `forced_final_after_output_limit`, including when the deadline prevents its provider dispatch |
| `reason_codes` | ordered, de-duplicated closed execution reasons |
| `unusable_reason` | first reason code for `unusable`; otherwise null |
| `errors` | bounded diagnostic text; never used as a status discriminator |

Closed reason codes are `analysis_deadline_exceeded`,
`context_budget_exceeded`, `context_window_exceeded`, `provider_timeout`,
`provider_error`, `provider_http_error`, `output_truncated`,
`malformed_output`, `contract_invalid`, `no_valid_evidence`,
`model_call_failed`, `retry_used`, `unsupported_tool_request`,
`tool_round_exhausted`, `contract_repair`, and `orchestration_error`.

Final deadline, context, timeout, truncation, provider, parse, and contract
failures are unusable. Earlier failed calls, retries, unsupported tool requests,
contract repair, or tool-round/output-limit exhaustion followed by a contract-valid final
response are degradation reasons. They do not erase or replace the valid
semantic payload. `final_evidence_reason` distinguishes `contract_repair`,
`forced_final_after_tool_exhaustion`, and
`forced_final_after_output_limit`; only one tools-disabled final-evidence turn
is permitted.
The final-turn reason records why the additional turn ran; `reason_codes`
records its quality effect. `unusable_reason` describes only the final unusable
condition and never a recovered prior response.
`assess_execution()` is the executable source of this classification; route
publishing and tracing must not rederive it from raw booleans or anomaly text.

### L1 Provider Call Record

Every provider attempt, plus any client preflight refusal before dispatch, is
represented once in `L1EvidenceResult.model_calls`. Provider retries therefore
add call records without adding model turns; subsequent logical turns use the
maximum recorded `model_turn` plus one rather than the number of call records.
The record is diagnostic input to
execution assessment and evaluation; it is not semantic failure evidence.

| Field | Contract |
| --- | --- |
| `model_turn`, `attempt`, `max_retries` | Logical conversation turn and provider-attempt position. |
| `success`, `latency_s`, `finish_reason`, `usage` | Provider outcome, wall time, completion reason, and provider token accounting. |
| `tools_advertised` | Whether tools were present on that logical turn. |
| `context_budget` | Estimated input, requested/effective output cap, configured context limit, and safety reserve for that turn. |
| `configured_request_timeout_seconds` | Route-level request timeout before deadline clamping. |
| `effective_request_timeout_seconds` | Timeout actually supplied to HTTPX after clamping to remaining analysis time. |
| `remaining_analysis_budget_before_call_s` | Whole-analysis budget immediately before the attempt. |
| `provider_reported_timing` | Optional timing components copied from provider response headers; absent when the endpoint does not report them. |
| `error_type`, `error`, `http_status`, `response_body` | Bounded failure diagnostics on unsuccessful attempts. |
| `retryable`, `retry_scheduled` | Classification and whether another provider attempt was actually scheduled. |
| `timeout` | Whether the failed transport wait ended as a timeout. It is false when the deadline is observed before dispatch or after a non-timeout response, and may remain true on a deadline-clamped HTTPX timeout for diagnosis. |
| `timeout_kind` | When available, `connect`, `pool`, `write`, `read`, or `unknown`. This diagnoses the blocked HTTPX operation and does not replace `error_type`. |
| `deadline_exceeded`, `deadline_reason` | Present on whole-analysis deadline failure; identify the authoritative route-budget outcome and its boundary. |

`analysis_deadline_exceeded`, `context_budget_exceeded`, and
`context_window_exceeded` are non-endpoint outcomes. A deadline-clamped HTTPX
timeout may retain `timeout_kind`, but its authoritative `error_type` is
`analysis_deadline_exceeded`; neither the product assessment nor harness counts
it as `provider_timeout`. A recognized provider
context-window rejection is `context_window_exceeded`, non-retryable, and not a
timeout whether it arrives as an HTTP error or a 2xx provider error envelope.
This precedence prevents a `429`, `503`, or `504` context rejection from being
counted as endpoint instability and prevents a 2xx error envelope from entering
assistant-content parsing.

Assistant-content parsing has two distinct unusable outcomes:

| Outcome | Parsed semantic payload | Diagnostics |
| --- | --- | --- |
| `malformed` | null | Strict JSON location plus the safe fallback-parser error. |
| `contract_invalid` | Retained for inspection only | Individual `L1ResponseContract` violations. |

The retained contract-invalid object is never passed to L2 or used by L3/L4.
The result anomaly is `malformed_model_evidence` or
`contract_invalid_model_evidence`, respectively.
When a contract-invalid response is followed by successful repair, the rejected
parsed object and its individual validation errors remain in the trace-only
`model_response_validation` event.

L1 returns exactly one `restart_agent_evidence.v1` JSON object. Required
top-level fields are:

```json
{
  "schema_version": "restart_agent_evidence.v1",
  "analysis_status": "primary_identified",
  "primary_failure": {},
  "observed_failures": [],
  "selected_observed_failure_id": null,
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
confidence bounds, evidence support tags, non-primary semantics, and
non-blocking response-shape and identifier recommendations.
Citation completeness is observational: L2 independently grounds selected
lines, so an omitted primary, root-cause, or recovery support tag cannot make
an otherwise structured model assessment unusable. A contract change must
therefore update one executable source and its tests rather than separate
prompt and parser descriptions.

The advertised schema remains closed with `additionalProperties=false`, but
the receiving boundary distinguishes extensions from structural damage.
Unknown fields are removed before publishing `L1EvidenceResult.semantic_payload`
and reported together as `unknown_response_fields_ignored`; they do not cause
contract repair or route degradation. Missing required fields, malformed known
fields, invalid enum values, and semantic contradictions remain hard contract
errors. Consequently, an unsolicited `decision` is ignored, while a misspelled
`rationale_extra` still fails if the required `rationale` is absent.

The following are recommendations rather than hard validity boundaries:

| Recommendation | Advisory behavior |
| --- | --- |
| At most 3 plausible causes, 5 missing-evidence entries, 3 observed failures, 3 related failures, and 12 citations | Additional well-formed entries are preserved and reported. |
| Evidence and observed-failure IDs at most 64 characters | Longer non-empty IDs remain usable. |
| Unique object IDs and unique values within `observed_failures[].evidence_ids` and `evidence[].supports` | Duplicates remain usable and are reported. A selected observation ID resolving to multiple objects disables only that track. |
| `selected_observed_failure_id` resolves to exactly one observation | Zero or multiple matches are reported and make only the observation track unavailable. |
| Primary and root-cause support tags are present | Missing tags are reported; L2 grounds the selected line independently. |
| Evidence `supports` values use the advertised claim-tag vocabulary | Unknown non-empty strings are retained and reported, but ignored for claim-support accounting. A missing, empty, or non-string `supports` value remains invalid. |
| No-primary placeholder prose and unknown confidence use canonical values | Alternate non-empty prose and in-range confidence remain usable while typed unknown values/statuses remain mandatory. |
| No unknown response fields | Unknown fields are stripped from the semantic payload and their paths are reported in one advisory. |

For example, an extra plausible cause records this advisory without causing
contract repair, route degradation, or semantic rejection:

```json
{
  "code": "plausible_causes_exceeds_recommended_limit",
  "field": "root_cause_assessment.plausible_causes",
  "observed_count": 4,
  "recommended_maximum": 3,
  "observational_only": true
}
```

The complete advisory appears in `L1EvidenceResult.anomalies.contract_advisories`
and the `model_response_validation` trace event. Its code is also copied to
public `result_provenance.notes`; the exact model output remains unchanged.

An L1 result is **usable** when the route completes before the whole-analysis
deadline, the final provider turn is not timed out or truncated, the extractor
reports success with a parsed semantic payload, and that object passes
`L1ResponseContract`. Provider retries, earlier failed HTTP attempts,
unsupported tool requests, or a prior output limit followed by valid final
evidence may make an otherwise usable route `degraded`; they do not erase its
valid semantic payload. L2 advisory findings likewise do not make L1 unusable.
Final provider timeout or truncation, missing/unparseable output, or a
contract-invalid object is unusable and leaves the deterministic branch
authoritative. Before HTTP, L1 estimates the complete request against the
route's configured context window. If input plus the safety reserve leaves no
response capacity, the route is unusable with non-retryable
`context_budget_exceeded`; an endpoint-side context rejection remains the
distinct `context_window_exceeded` error.

Primary absence and evidence absence are distinct. `observed_failures` is an
independent non-primary collection and may also be populated when a primary is
identified:

| Status | Primary | Observations | Root cause | Recovery claims | Related failures |
| --- | --- | --- | --- | --- | --- |
| `primary_identified` | Required initiating or unknown-role primary | Optional non-primary surfaces; selected id preferably references one canonical terminal surface | Substantive assessment of the primary | Assesses the primary | Relationships to the primary are allowed |
| `no_failure_observed` | `null` | Empty; selected id `null` | `No failure was observed in the supplied evidence.`; `unknown`; no plausible causes or missing evidence | `unknown` value/status, confidence `1`; rationale exactly `Recovery is not assessed because no failure was observed.` | Empty |
| `insufficient_evidence` | `null` | Zero or more grounded candidates; selected id preferably references exactly one item or is `null` when selection is ambiguous | Unknown initiating cause with one or more explicit evidence gaps | May contain substantive current-attempt claims only when a selected observation is grounded; otherwise both claims are `unknown` | Empty because no relationship to a primary can be established |

An `insufficient_evidence` response is not an empty placeholder. It preserves
what visibly failed while abstaining on the initiating primary. Evidence may
support observed failures, root-cause uncertainty, and recovery claims. It
cannot activate primary-dependent workload, root, entity, or concrete policy.
The required `confidence=1` values in `no_failure_observed` are schema
placeholders and MUST be excluded from calibration.

### L1 Primary Failure

```json
{
  "line": 12083,
  "causal_role": "initiating",
  "failure_identity": {
    "operation": "checkpoint_load",
    "mechanism": "metadata_deserialization",
    "component": "torch_distributed_checkpoint",
    "direct_failure_object_path": "/path/to/direct/object/or/null",
    "affected_artifact_path": "/path/from/log/or/null"
  }
}
```

Required primary `causal_role` values are `initiating` or `unknown`. Known
`cascade` and `teardown` events belong in `related_failures` after a primary is
identified. When only downstream evidence is available, L1 returns
`insufficient_evidence` with a null primary. The nested L1 `failure_identity`
contains semantic claims used during grounding; it is not a history identity
and is not copied into the product result. L0/L2 derive `failure_class`,
signature, fault outcome, locality, `root_fingerprint`, and optional
`affected_entity` from grounded client evidence.

`direct_failure_object_path` is the exact path directly acted on by the failing
operation. `affected_artifact_path` is the separate runtime artifact targeted
by the enclosing workload operation. For example, a permission denial may act
directly on a cache lock while the enclosing operation loads a dataset. Either
field may be null independently. A source-code location, traceback frame,
component installation path, log path, or diagnostic callsite is provenance
and MUST NOT populate either field; for example,
`file=/src/permute.cu, line=535, call=cudaGetLastError()` has both fields null.

### L1 Observed Failures

`observed_failures` is a non-primary evidence track. It is most important when
`analysis_status=insufficient_evidence`, but a model may also retain downstream
or unresolved surfaces while reporting a primary:

```json
{
  "id": "o1",
  "line": 30368,
  "causal_role": "unknown",
  "failure_identity": {
    "operation": "distributed_coordination",
    "mechanism": "tcpstore_connection_loss",
    "component": "c10d_tcpstore",
    "direct_failure_object_path": null,
    "affected_artifact_path": null
  },
  "rationale": "The peer-visible connection failed, but the store-owner termination is absent.",
  "evidence_ids": ["e1"]
}
```

Item ids should be unique. A duplicate is retained and reported. If the
selected id resolves to zero or multiple items, L2 makes only the observation
track unavailable. Causal role is `unknown`, `cascade`, or `teardown`;
`initiating` would require `primary_failure`. Every item cites one or more
evidence ids. `selected_observed_failure_id` should be either null or resolve
to exactly one item. It is expected when the model can select one
canonical terminal observation after excluding retry-pending, recovered,
progressed-after, and diagnostic-only candidates. Independent tied candidates
leave it null.

When `analysis_status=primary_identified`, `related_failures` remains the
preferred representation for known relationships to the primary. Redundant or
additional `observed_failures` do not invalidate an otherwise usable response;
L2 grounds and audits them independently. Recovery claims remain attached to
the primary, not to the observation track. When
`analysis_status=no_failure_observed`, both observation fields are empty/null.

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
is retained for corpus calibration and is not an L4 threshold. It MUST be
interpreted with the enclosing `analysis_status` and has no standalone meaning.
A claim is calibration eligible only when its primary or selected-observation
anchor is grounded and the claim value/status are both substantive rather than
`unknown`. Primary-grounded and observation-only calibration are reported
separately.

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
line. Evidence IDs should be unique; duplicates are retained as audit
findings. `supports` uses only the closed claim tags
`primary_failure`, `observed_failures`, `root_cause_assessment`,
`failure_domain`, and `retry_outlook_without_workload_change`. This array is the
canonical citation source; the contract has no second supporting-line list. The
tags identify which claim a citation supports; they do not encode claim
strength or policy. L2 records unavailable or ungrounded citations as
observational findings.

For `analysis_status=primary_identified`, the L1 usability boundary requires
the evidence array to collectively support `primary_failure` and
`root_cause_assessment`. Recovery tags are independently audited by L2, but
their audit result does not change the typed L1 assessment passed to L4. An
`unknown`/`unknown` recovery claim is an abstention and requires no positive
support citation.

For `analysis_status=insufficient_evidence`, each observed failure's
`evidence_ids` is intended to resolve to evidence tagged `observed_failures`.
L1 validates the reference shape, while L2 resolves referential integrity. An
undefined ID becomes a non-blocking `dangling_evidence_reference` finding and
is omitted from L2's grounded citation projection. All observations are
audited, but only a selected observation may still ground independently from
its explicit line when that line exists and was model-visible; raw L1 remains
unchanged. Substantive recovery claims require their corresponding support
tags; unknown abstentions do not.

## L2 Result

**Stage contract**

`L2GroundingInput` is an immutable aggregate rather than a wire payload:

| Direction | Field | Type | Meaning |
| --- | --- | --- | --- |
| Input | `bundle` | `L0Bundle` | Complete structured evidence used to resolve source facts. |
| Input | `model_view` | `L0ModelFacingView` | Exact initial model-facing narrative, compact Decision Evidence view, and supporting evidence. |
| Input | `l1_result` | `L1EvidenceResult` | Raw output, parsed semantic payload, transcript, and tool results. |
| Input | `source_log` | `LogSnapshot` | Immutable source used for citation and line grounding. |
| Output | - | `L2Result` | Mechanically grounded primary and observations, independent primary/observation `AttemptFailureFacts`, and observational audit diagnostics. |

L2 emits a typed result and a trace payload:

```json
{
  "grounding_status": "grounded",
  "track_grounding": {
    "primary": {
      "status": "grounded",
      "method": "exact_source_line",
      "published": true
    },
    "observation": {
      "status": "unavailable",
      "method": null,
      "published": false
    }
  },
  "primary_source_line_available": true,
  "selected_observation_source_line_available": null,
  "primary_model_visible_support": true,
  "selected_observation_model_visible_support": null,
  "identity_lineage": {
    "primary": {
      "identity_kind": "root",
      "model_selected_line": 12083,
      "l0_primary_line": 12083,
      "canonical_identity_anchor_line": 12083,
      "relationship_to_l0": "same_canonical_incident",
      "client_identity_source": "l0_canonical_identity"
    },
    "observation": null
  },
  "audit_status": "clean",
  "grounded_primary_failure": {},
  "grounded_observed_failures": [],
  "grounded_selected_observation": null,
  "enriched_failure_tracks": {
    "primary": {},
    "observation": null
  },
  "audit_influence": "observational_only",
  "field_findings": {},
  "field_finding_codes": {},
  "findings": [],
  "citation_audits": [],
  "grounding_adjustments": [],
  "recovery_field_audits": []
}
```

Grounding status is `grounded`, `unavailable`, or `not_run`. The aggregate is
`grounded` when at least one track is published; `track_grounding` is
authoritative for each L3 input. Audit status is
`clean`, `resolved`, `findings`, or `not_run`. Recovery-field audits are
non-overriding and include `applied=false` when suggesting another
interpretation. `audit_influence=observational_only` is invariant: no finding
changes route quality, NVRx eligibility, L3 identity, L4 input, or action.
L4 receives the exact typed L1 recovery assessment with the grounded primary
track when one exists. It receives that assessment with an observation track
only when L1 reported no primary. Observation-only grounding is explicitly
labeled and can select only root-independent general policy unless a declared
policy context matches.

Missing support for an otherwise valid substantive recovery claim is an L2
finding, not an L1 contract failure or policy gate. Unknown abstentions and
hypothesis-only claims remain weak inputs because of their L1 status and do not
produce a missing-positive-support finding.

Citation audit status is `exact`, `rendered_exact`, `abbreviated_exact`,
`nearby_resolved`, `ambiguous_nearby_match`, `not_model_visible`, or
`ungrounded`. `abbreviated_exact` requires at least two substantial fragments
around an explicit ellipsis to match in order on the cited source line and the
same model-visible rendering; it cannot repair a line number.

The per-track source-line fields report only whether the corresponding anchor
line exists in the immutable source. `primary_model_visible_support` and
`selected_observation_model_visible_support` report whether paired text for the
corresponding selected or uniquely resolved line was visible in L0B or a
successful tool result. Either or both may be true. Each non-null enriched
track independently requires source and model-visible support. A
provenance-only line reference does not satisfy model-visible support.

Each non-null `identity_lineage` entry has `relationship_to_l0` equal to
`same_canonical_incident`, `different_grounded_incident`, or
`l0_identity_unavailable`.
`client_identity_source` is `l0_canonical_identity` only for the first case and
`l2_source_grounding` otherwise. The lineage records provenance without copying
L1 semantic `failure_identity` into the L3-facing recurrence contract.

L2 visibility is evaluated against the exact full `model_visible_payload`
recorded for the initial request and every returned tool payload. A source line
present with original source text or a source quote in any initial-payload
section is therefore visible even when it is absent from the compact
`evidence_bundle` subsection. Narrative summaries and provenance-only line
references do not satisfy this rule. Canonical evidence requires visible
line/quote text; exact source-log content that the model never saw is not
grounded support.
A related-failure line outside the line-reference visibility set is retained in
raw L1 trace but omitted from L2's audited related-failure view.
The same rule applies to `observed_failures`; an ungrounded item remains visible
in raw L1 but is absent from `grounded_observed_failures`. The selected
observation is usable only when its referenced item is grounded.

## L2 Enriched Attempt Tracks

L0 supplies `AttemptRecord.deterministic`; each usable L2 route supplies one
route entry containing independently optional `primary` and `observation`
tracks. Both use the same `AttemptFailureFacts` shape. The runtime assembler
applies the route update atomically:

```json
{
  "route_id": "gpt",
  "primary": {
    "source": "l2_grounded",
    "identity_kind": "root",
    "root_fingerprint": "observed:unicodedecodeerror:checkpoint_metadata",
    "root_fingerprint_source": "observed_exception",
    "observation_fingerprint": null,
    "observation_fingerprint_source": null,
    "fault_outcome": "terminal",
    "primary_line": 12083,
    "selected_observation_line": null,
    "selected_observation_causal_role": null,
    "identity_anchor_line": 12083,
    "identity_anchor_reason": "canonical_episode_terminal",
    "failure_iteration": null,
    "affected_entity": {
      "kind": "artifact",
      "identity": "/path/to/checkpoint",
      "fingerprint": "affected_entity:artifact:7f28...",
      "evidence_line": 12083
    },
    "root_observer_ranks": ["4175"],
    "unattributed_root_occurrence_count": 0,
    "faulting_rank": "4175",
    "faulting_node": null,
    "faulting_gpu": null,
    "rank_to_gpu_map": {}
  },
  "observation": null
}
```

The route entry does not duplicate shared progress. It is inserted only while
the invocation remains open. L2 audit output remains separate and is not copied
into the compact attempt record.

For an observation track, `identity_kind=observation_only`, primary/root,
root-observer, affected-entity, and root-anchor fields are null;
`observation_fingerprint`, its source, `selected_observation_line`, and
`selected_observation_causal_role` are non-null. A route may carry this block
beside a primary block. It may be compared by L3 as observation-only evidence
but cannot enter root/entity ledgers.

## L3 History Summary

**Stage contract**

| Direction | Field | Type | Meaning |
| --- | --- | --- | --- |
| Input | `current_record` | `AttemptRecord` | Current attempt with shared progress plus deterministic and route-keyed primary/observation facts. |
| Input | `prior_attempts` | `PriorAttemptView` | Runtime-selected bounded exact-job earlier records. |
| Output | - | `CycleHistoryComparison` | Shared job-progress comparison plus deterministic and route-keyed primary/observation history summaries. |

These input fields are carried together as immutable
`HistoryEvaluationInput`. L3 receives no raw log, L1 transcript, model
confidence, or retry policy. It compares like with like: deterministic against
deterministic, a route's primary against that route's prior primary, and a
route's observation against that route's prior observation. Missing route facts
remain unavailable; L3 does not silently substitute deterministic facts.

```json
{
  "job_progress": {},
  "deterministic": {},
  "routes": [
    {
      "route_id": "gpt",
      "primary": {},
      "observation": {}
    }
  ]
}
```

`job_progress` is computed once from the shared progress record. Each other
non-null member is a `HistorySummary` for exactly one identity track. Multiple
routes agreeing about one prior cycle do not turn that cycle into multiple
attempts; counts are per route and deduplicated by `(job_id, cycle_id)`.

The following example is one root-bearing track summary:

```json
{
  "available": true,
  "availability_reason": "ready",
  "identity_kind": "root",
  "same_job_attempts": 2,
  "job_history_available": true,
  "job_history_availability_reason": "ready",
  "observation_history_available": false,
  "observation_history_availability_reason": "current_identity_is_root",
  "matching_observation_attempts": 0,
  "observation_comparisons": [],
  "consecutive_same_observation_no_advance_attempts": 0,
  "job_comparisons": [
    {
      "prior_cycle_id": 1,
      "selected_basis": "completed_step_and_checkpoint_step",
      "dimension_comparisons": [],
      "positive_progress_conflict": false,
      "relation": "same",
      "prior_attempt_progress": {},
      "prior_fault_outcome": "terminal",
      "same_failure_iteration": false,
      "same_rank": false,
      "affected_entity_relation": "unknown",
      "same_root_observer_count": false,
      "same_unattributed_root_occurrence_count": false
    }
  ],
  "consecutive_same_job_no_advance_attempts": 1,
  "consecutive_same_job_unknown_progress_attempts": 0,
  "job_progress_advanced": false,
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

Observation-only comparison is a parallel, explicitly weaker scope.
`observation_history_available=true` requires
`identity_kind=observation_only` and a non-null current observation
fingerprint. `observation_comparisons` uses the same progress-comparison shape
for prior exact-job records with the same observation fingerprint, and
`consecutive_same_observation_no_advance_attempts` applies the same newest-first
boundary rules. These fields are diagnostic by default and never contribute to
root/entity aggregates.

The two consecutive fields are independent observations, not alternate views
selected by L3:

- `consecutive_same_root_no_advance_attempts` continues across entity changes;
- `consecutive_same_root_and_entity_no_advance_attempts` requires exact entity
  kind and fingerprint equality at every newest-first step.

A different or unknown entity stops only the second count. Root mismatch,
nonqualifying prior outcome, observed advance, or unknown progress stops both
as applicable. The current failure is not included in either prior-attempt
count.

`job_history_available=true` means same-job lookup was eligible and completed,
including when the prior view contains zero records. For one track summary,
`available=true` additionally means the current track facts have a root fingerprint and
root-scoped comparisons are available. A missing root therefore leaves job
guards usable while setting `available=false` with
`availability_reason=missing_root_fingerprint`. Current/future-cycle filtering
and its counts belong to the runtime-history trace because the view supplied to
L3 already contains only prior records.

A missing root may therefore coexist with available observation-only and
same-job history. Null root or observation fingerprints are never comparable,
and `identity_kind=none` leaves both identity scopes unavailable while retaining
the job scope.

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
| Input | `failure_tracks` | `CycleFailureEntry` | Deterministic facts plus independently grounded route-primary and route-observation facts. |
| Input | `history` | `CycleHistoryComparison` | Matching L3 summaries for every available track plus shared job progress. |
| Input | `model_recovery_assessment` | route-keyed `ModelRecoveryAssessment` or null | Exact typed L1 domain and unchanged-workload retry outlook. It applies to the primary track when one is grounded; it applies to the observation track only when L1 had no primary. |
| Input | `retry_policy` | `RetryPolicyConfig` | Effective confirmation, general, and job-guard retry counts. |
| Input | `policy_contexts` | `PolicyContextConfig` | Trusted policy contexts and their independent retry budgets. |
| Output | `path_selection` | `L4PathSelection` | `primary`, `observation`, `deterministic`, or `none`, with route and reason. |
| Output | `retry_policy` | `RetryPolicyEvaluation` | Deterministic decision, basis, selected rule, and concurrent ledger accounting. |

The inputs are carried as `L4PolicyInput`; the outputs are carried as
`L4PolicyOutcome`.

L4 composes the parallel tracks in this order:

```text
if a grounded route-primary track is eligible:
    select primary and its primary history
else if a grounded route-observation track is eligible:
    select observation and its observation history
else if deterministic facts are eligible:
    select deterministic and its deterministic history
else:
    select none

apply shared same-job progress guards independently
```

Route preference or arbitration must be supplied explicitly by the caller; L4
does not use completion order as preference. Path selection affects the current
recommendation only. All tracks remain in the `AttemptRecord` for later history
and review.

`matching_prior_attempts` counts prior attempts only; the current attempt is
never included in ledger consumption.

```json
{
  "policy_version": "retry_budget.v1",
  "identity_kind": "root",
  "base_rule": "concrete_confirmation_retry",
  "effective_policy": {
    "source": "base_rule",
    "rule": "concrete_confirmation_retry",
    "history_match_scope": "root_and_entity",
    "allowed_retries": 1,
    "policy_context_id": null
  },
  "applied_policy_context": null,
  "decision": "RESTART",
  "decision_basis": "concrete_confirmation_retry_available",
  "retry_budget_exhausted": false,
  "exhausted_by": [],
  "general_root_ceiling": {
    "ledger_id": "general_root_ceiling",
    "applicable": true,
    "rule": "general_retry",
    "history_match_scope": "root_only",
    "allowed_retries": 3,
    "matching_prior_attempts": 1,
    "observed_advance": false,
    "exhausted": false,
    "inapplicable_reason": null
  },
  "selected_policy_ledger": {
    "ledger_id": "selected_policy_ledger",
    "applicable": true,
    "rule": "concrete_confirmation_retry",
    "history_match_scope": "root_and_entity",
    "allowed_retries": 1,
    "matching_prior_attempts": 0,
    "observed_advance": false,
    "exhausted": false,
    "inapplicable_reason": null
  },
  "job_no_progress_guard": {
    "ledger_id": "job_no_progress_guard",
    "applicable": true,
    "rule": "job_no_progress_guard",
    "history_match_scope": "same_job_no_progress",
    "allowed_retries": 3,
    "matching_prior_attempts": 1,
    "observed_advance": false,
    "exhausted": false,
    "inapplicable_reason": null
  },
  "job_unknown_progress_guard": {
    "ledger_id": "job_unknown_progress_guard",
    "applicable": true,
    "rule": "job_unknown_progress_guard",
    "history_match_scope": "same_job_unknown_progress",
    "allowed_retries": 3,
    "matching_prior_attempts": 0,
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
  "current_evidence_qualified": false,
  "current_affected_entity": {
    "kind": "artifact",
    "identity": "/checkpoints/iter_5000/metadata.json",
    "fingerprint": "affected_entity:artifact:3d97...",
    "evidence_line": 12083
  },
  "match_requirements": {
    "job_id": "exact",
    "root_fingerprint": "exact",
    "affected_entity": "exact",
    "progress": "no_observed_advance"
  }
}
```

`general_root_ceiling` is present for every root-identified, recoverable
primary and uses `consecutive_same_root_no_advance_attempts`. It is
inapplicable for observation-only, identity-none, and immediate
`workload_unrecoverable` outcomes.

`selected_policy_ledger` is nullable. It is present only when the effective rule
declares a budget narrower than the general ceiling. Its
`history_match_scope` selects the corresponding L3 count:

- `root_only` uses `consecutive_same_root_no_advance_attempts`;
- `root_and_entity` uses
  `consecutive_same_root_and_entity_no_advance_attempts`;
- `same_job_no_progress` uses
  `consecutive_same_job_no_advance_attempts` for observation-only
  `general_retry`.

The root-independent `job_no_progress_guard` counts consecutive same-job
attempts whose completed progress is the same or regressed. The separate
`job_unknown_progress_guard` bounds consecutive attempts whose progress cannot
be compared. Both remain usable when no primary/root is available.

`retry_budget_exhausted` aggregates every active ledger and guard.
`exhausted_by` names each exhausted component; multiple entries may be present.
An immediate `workload_unrecoverable` STOP has
`retry_budget_exhausted=false` and is identified by `decision_basis`.

Each ledger is a `RetryLedgerEvaluation` with the exact fields shown above.
`inapplicable_reason` is null when active and otherwise explains the missing
identity or unavailable history.

Allowed non-null base rules are `workload_unrecoverable`,
`concrete_confirmation_retry`, `workload_confirmation_retry`, and
`general_retry`. Root-identified `general_retry` uses root-only history;
observation-only `general_retry` uses same-job no-progress history and never
uses the observation fingerprint as a root. `effective_policy` initially mirrors the base rule; a
declared policy context may replace only that selected policy while the general
and job guards continue independently. See `L4.md` for selection and action
semantics.

## Public Analysis Response

**Boundary contract**

| Direction | Type | Meaning |
| --- | --- | --- |
| Input | L0-L4 outputs and execution health | Preserved stage results plus the explicit L4 path selection. |
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
  "l1_assessment": {},
  "l2_grounding": {},
  "primary_failure": {},
  "observed_failures": [],
  "selected_observed_failure": null,
  "secondary_failures": [],
  "cascades": [],
  "evidence_coverage": {},
  "justification": "..."
}
```

Required top-level fields are `schema_version`, `decision`, `decision_basis`,
`retry_policy`, `failure_domain`, `result_provenance`, `primary_failure`,
`observed_failures`, `selected_observed_failure`, `l1_assessment`,
`l2_grounding`, `secondary_failures`, `cascades`, `evidence_coverage`, and
`justification`. `l1_assessment` is `null` when L1 did not produce a parsed
semantic object.

### Stage Ownership In The Public Result

`l1_assessment` is the contract-defined portion of the parsed
`restart_agent_evidence.v1` object emitted by the model. L1 strips unknown
fields at its receiving boundary; the raw model response and ignored field
paths remain traceable. L2 does not rewrite, normalize, or replace fields inside
this block. It includes the model's primary selection, root-cause assessment,
recovery assessment, related failures, evidence, and rationales exactly as L1
accepted them after response-contract normalization.

`l2_grounding` is a separate client-produced block:

```json
{
  "grounding_status": "grounded",
  "track_grounding": {
    "primary": {"status": "grounded", "published": true},
    "observation": {"status": "unavailable", "published": false}
  },
  "audit_status": "clean",
  "not_run_reason": null,
  "grounded_primary_failure": {},
  "grounded_observed_failures": [],
  "grounded_selected_observation": null,
  "grounded_related_failures": [],
  "grounded_evidence": [],
  "audit_influence": "observational_only",
  "grounded_failure_identities": {
    "primary": {
      "direct_failure_object_path": {
        "model_value": "/checkpoint/step_100/metadata",
        "grounded_value": "/checkpoint/step_100/metadata",
        "evidence_lines": [12083],
        "status": "grounded"
      },
      "affected_artifact_path": {
        "model_value": "/checkpoint/step_100",
        "grounded_value": "/checkpoint/step_100",
        "evidence_lines": [12010],
        "status": "grounded"
      }
    },
    "observation": null
  },
  "affected_entity_selection": {
    "source_field": "affected_artifact_path",
    "selection_reason": "grounded_affected_artifact_preferred",
    "evidence_lines": [12010],
    "entity": {
      "kind": "artifact",
      "identity": "/checkpoint/step_100",
      "fingerprint": "affected_entity:artifact:...",
      "evidence_line": 12010
    }
  },
  "history_identities": {
    "primary": {
      "ready": true,
      "identity_kind": "root",
      "anchor_line": 12083,
      "anchor_reason": "model_primary_is_episode_identity_anchor:terminal_exception",
      "root_fingerprint": "observed:...",
      "root_fingerprint_source": "observed_exception"
    },
    "observation": null
  },
  "grounding_adjustments": [],
  "track_findings": {
    "primary": [],
    "observation": []
  },
  "findings": []
}
```

This block says what L2 accepted, rejected, resolved, or derived. It does not
repeat the model's root-cause or recovery prose. Full citation audits and other
operational diagnostics remain in the trace.
`track_findings` keeps primary and observation diagnostics independent.
`findings` is their flattened public view, with each entry labeled by `track`.
An unresolved or ambiguous selected-observation ID therefore makes only the
observation track unavailable and remains visible even when the primary track
is published.

`grounded_failure_identities` grounds the model's direct failure object and
enclosing affected artifact independently. For a grounded primary, L2 prefers
the grounded enclosing artifact as `affected_entity`; when it is unavailable,
L2 uses the grounded direct object. `affected_entity_selection` preserves which
field supplied the typed L3/L4 entity. Neither path is inferred from the other.

For an observation track, `history_identities.observation.identity_kind` is
`observation_only`, root fields are null, and observation fingerprint/source
and anchor line are populated. The fingerprint itself is not a generic root
policy key. A primary and observation identity may coexist in this block;
neither overwrites the other.

The top-level `primary_failure`, `observed_failures`,
`selected_observed_failure`, `secondary_failures`, and `cascades` are canonical
facts published for review. A selected observation may coexist with a primary,
but it never substitutes for `primary_failure` and never creates a root
fingerprint. `retry_policy`, `decision`, and `decision_basis` are L4 outputs;
`result_provenance` records which L4 path was selected.
Top-level `justification` explains the final policy outcome; the model's own
rationale remains under `l1_assessment`.

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
  "source": "l0_deterministic_with_l1_relationship_l2_grounded",
  "reason": "appears after primary candidate at line 12083",
  "relationship_rationales": [
    "Finalizer cleanup occurred after the checkpoint-load failure."
  ]
}
```

L0 owns structural `cascade` and `teardown` roles. When L2 grounds an L1
related-failure rationale for the same event, the rationale annotates the
deterministic group. A grounded L1-only downstream event is retained as a
single-entry group with `source=l1_proposed_l2_grounded`. Downstream events do
not remain in `secondary_failures`.

### Result Provenance

The compact result provenance includes:

- `candidate_kind`: `deterministic` or `l1_enriched`;
- `selected_evidence_path`: `primary`, `observation`, `deterministic`, or
  `none`;
- `selected_route_id` and `path_selection_reason`;
- availability of deterministic, primary, and observation tracks at L4;
- `evidence_source`;
- `model_contribution`;
- `history_contribution`;
- `result_quality`: `normal`, `degraded`, or `unusable`;
- `nvrx_use`: `eligible`, `eligible_degraded`, or
  `fallback_to_nvrx_default`;
- L1 execution status/issues;
- concise notes.

`result_quality` measures result usability, not failure ownership.

`model_contribution` is a closed provenance value:

| Value | Meaning |
| --- | --- |
| `not_enabled` | No model route was configured for this candidate. |
| `pending_not_used` | The deterministic candidate was published while model enrichment was still in flight. |
| `not_needed_l0` | L0 remained authoritative because no usable model contribution was needed or available. |
| `attempted_used` | L1 completed and L2 mechanically grounded the model contribution. L2 audit status is reported separately. |
| `attempted_not_used_timeout` | The route result carried a terminal `provider_timeout` condition; this also covers a route worker that did not return before the analysis deadline. |
| `attempted_not_used_truncated` | The final model output was truncated. |
| `attempted_not_used_provider_error` | Another provider or deadline-clamped request failure prevented usable model evidence. |
| `attempted_not_used_contract_invalid` | JSON decoded, but the object violated `L1ResponseContract`. |
| `attempted_not_used_malformed` | Assistant content could not be decoded as one semantic object. |
| `attempted_not_used_ungrounded` | L1 was structurally usable, but L2 could not establish minimum grounding. |

This field records whether model evidence affected the candidate. It does not
replace `L1ExecutionAssessment`, which retains the exact execution cause and
quality.

### Deterministic And Degraded Results

| Condition | Decision / basis | Primary | Quality |
| --- | --- | --- | --- |
| Accepted path unavailable or empty | `RESTART / log_unavailable` | null | `unusable` |
| Provider timeout/truncation with L0 primary | Deterministic L4 result | L0 candidate | normally `degraded` |
| No primary with one selected terminal observation | Root-independent `general_retry` using same-job progress | null | normally `normal`; selected observation remains separately visible |
| Malformed L1 and neither L0 primary nor selected observation | `RESTART / malformed_model_output` | null | `unusable` |
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
      "l1_execution_assessment": {
        "execution_status": "completed",
        "result_quality": "usable",
        "parse_status": "valid",
        "usable": true,
        "degraded": false,
        "evidence_present": true,
        "reason_codes": [],
        "unusable_reason": null,
        "errors": []
      },
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

`model_results[].execution_status` is lifecycle-only and uses the same closed
values as `L1ExecutionAssessment.execution_status`: `not_run`, `in_flight`,
`completed`, or `failed`. It never encodes a failure cause. Consumers use
`l1_execution_assessment.unusable_reason` and `reason_codes` to distinguish an
analysis deadline, local context-budget failure, provider context rejection,
timeout, HTTP/transport failure, malformed output, or contract failure.

`analysis_result` is the route's final product response after L2, L3, and L4;
it is not the raw L1 model answer. The exact raw and parsed semantic response and tool
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
| L0A | Bundle schema, source size, source decode time, source index/classification time, canonical reduction time, coverage/lossiness, primary, selected observation, identity kind, and fingerprint provenance. |
| Decision Evidence | Selection timing, deterministic primary, selected observation, canonical identity, references, and shared-object provenance. |
| L0B | View schema, projection timing, narrative event/reference coverage, per-section characters/tokens, fanout and overlap compaction, selected/omitted/truncated counts, and payload hash. |
| L1 | Exact credential-free requests, prompts/messages, advertised tools, raw and parsed responses, model/tool calls, retries, provider errors, finish reasons, token usage, and timing. |
| L2 | Per-track grounding status/method, citation audits, adjustments, findings, recovery-field audits, enriched primary/observation identities, and timing. |
| L3 | Shared job progress plus deterministic and route-keyed primary/observation comparisons, aggregates, and timing. |
| L4 | Evidence-path selection, full `retry_policy` evaluation, result provenance, decision/basis, and timing. |

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

## Attrsvc And Progressive Boundary

Attrsvc request/response shapes, environment settings, polling, terminal drain,
and result retention are integration contracts owned by
`ATTRSVC_INTEGRATION.md`. They are not part of
`restart_agent_config.v1` or the Restart Agent public request schema.

`ProgressiveL0State`, `FinalizedL0A`, source-boundary capture, and progressive
finalization are internal lifecycle concepts owned by `PROGRESSIVE.md` and
`RUNTIME.md`; they are not caller-visible wire contracts. Once analysis is
complete, attrsvc exposes the same `restart_agent_response.v1` under its
integration response's `result` field.

## Eval Boundary

`EVALUATION.md` owns the product/harness boundary. This file defines only the
product artifacts and measurements consumed by evaluation; harness gold,
scores, panel reports, and aggregate schemas are not repeated here.
