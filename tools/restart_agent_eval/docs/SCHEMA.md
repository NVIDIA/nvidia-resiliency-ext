# Restart Agent Eval Schemas

## Versions

| Object | Version |
|---|---|
| scored label and case result | `restart_agent_eval.v1` |
| one-log run manifest | `restart_agent_eval_run.v1` |
| one-log review index | `restart_agent_review_index.v1` |
| per-route review summary | `restart_agent_review.v1` |
| panel summary | `restart_agent_panel.v1` |
| product response under test | `restart_agent_response.v1` |
| product collect-all result | `restart_agent_collect_all.v1` |
| live product status | `restart_agent_live_status.v1` |
| live product event | `restart_agent_live_event.v1` |

The harness validates product trace envelopes through `product_trace.py`. It
does not redefine product-internal layer schemas.

## Scored Label

Corpus cases use JSON at `<gold-root>/<relative-log-path>/gold.json`. The
harness has no alternate YAML or embedded-oracle case format. Examples below
use YAML notation only to keep partial objects readable; durable files are JSON.

```yaml
schema_version: restart_agent_eval.v1
label_version: 1
case_id: checkpoint-metadata-decode
review_status: human_approved
source_sha256: 0d1983c9db0c83893623c1491ff5681fc6ce8fbd1ffd8fcc19719d71f08f2643

primary_anchor_expectation:
  accepted_lines: [12083]
  rejected_downstream_lines: [12135]
  tolerance_lines: 2

# Use this instead of primary_anchor_expectation when the initiating failure is
# absent but one terminal failure surface is reviewable.
observation_expectation:
  require_primary_absent: true
  accepted_lines: [30368]
  line_tolerance: 2
  accepted_failure_classes: [tcpstore_connection_loss]
  accepted_causal_roles: [cascade]
  accepted_observation_fingerprints: [terminal_transport_failure_surface:tcpstore_connection_loss]

root_cause_expectation:
  accepted_operations: [checkpoint_load]
  rejected_mechanism_terms: [write, save]
  required_concept_groups:
    - [checkpoint, metadata]
    - [decode, deserialize, unicode]
  require_uncertainty_preserved: true
  uncertainty_terms_any: [transient, ambiguous, unproven]

recovery_assessment_expectation:
  failure_domain: [workload, infrastructure, unknown]
  failure_domain_status: [supported_but_unconfirmed, unknown]
  retry_outlook_without_workload_change: [may_recover, unknown]
  retry_outlook_status: [supported_but_unconfirmed, unknown]

retry_policy_expectation:
  accepted_rules: [concrete_confirmation_retry, workload_confirmation_retry, general_retry]
  retry_budget_exhausted: false

action_expectation:
  accepted: [RESTART]

cascade_expectation:
  expected_groups:
    - causal_role: cascade
      first_line: 12123
      minimum_count: 2
      minimum_rank_count: 2
  teardown_lines: [12135]

history_identity_expectation:
  identity_kind: root
  operation: checkpoint_load
  mechanism: checkpoint_metadata_unicode_decode_error
  canonical_anchor_line: 12083
  same_episode_lines: [12083, 12123]
  expected_cross_route_identity_count: 1

unsupported_claims:
  - id: proven_corruption
    text_patterns:
      - proves the checkpoint is corrupt
```

`accepted_rules` scores the L4 base rule. Optional
`accepted_effective_rules` scores the effective rule independently when a
policy context is expected, for example
`rejected_iteration_retry_then_skip`. When both are present, both must match.

All expectation blocks except `action_expectation` are optional. A scalar or a
list may be used for each recovery field; lists represent acceptable ambiguity,
not a preference order.

`expected_lines` remains available for a specific downstream line.
`expected_groups` scores normalized cascade/fanout semantics using causal role,
first line, minimum event count, and minimum rank spread instead of requiring
every duplicate rendering. `teardown_lines` additionally requires that the event retain
`causal_role=teardown`; a generic cascade at that line does not pass the role
check.

`history_identity_expectation` checks the L2 identity kind, canonical anchor,
comparable L0 operation context, normalized mechanism terms, and panel-wide
identity count. For `identity_kind=observation_only`, the mechanism is checked
against the observation fingerprint and root/entity fields are expected to be
absent. Gold normally avoids opaque implementation hashes; an accepted
observation fingerprint may be supplied when its normalized public value is
itself part of the reviewed contract.
`accepted_operations` and `rejected_mechanism_terms` score L1's structured
`primary_failure.failure_identity` so a contradictory mechanism label cannot
pass merely because correct words appear elsewhere in the response.

Gold labels are strict contracts. `review_status` must be `human_approved`,
`human_reviewed_ambiguous_rca`, or
`human_reviewed_supported_but_unconfirmed_rca`; these distinguish label
approval from certainty of the underlying RCA. `source_sha256` must match the
exact source-log bytes before semantic scoring begins. The harness
rejects unsupported fields in
scored expectation blocks, including unknown `required_coverage` names and
operation-comparison fields. Human rationale remains free-form under
`human_assessment`. This prevents schema drift from silently becoming a failed
KPI; for example, operation history is represented by
`operation_artifact_comparisons`, not the obsolete `operation_history` name.

### L0 Expectations

```yaml
l0_expectation:
  required_setup_marker_types: [checkpoint_metadata_load, checkpoint_load_start]
  minimum_setup_marker_count: 1
  required_coverage:
    setup_progress: found
  accepted_primary_lines: [12083]
  accepted_root_fingerprints: [observed:unicode_decode_error]
  required_checkpoint_lines: [12050]
  expected_primary_phase: setup
  expected_checkpoint_load_iteration: 622125
  expected_progress_after_failure_episode: false
  required_cascade_lines: [12123, 12135]
  required_operation_artifact_comparisons:
    - operation: checkpoint_save
      minimum_success_count: 2
      current_outcome: started_not_completed
  required_failure_episodes:
    - lifecycle_family: rdma_port
      required_source_dialects: [nccl_net_ib]
      status: recovered
      accepted_start_lines: [32172]
      minimum_fault_count: 4
      minimum_recovery_attempt_count: 8
      minimum_recovery_confirmation_count: 8
      required_entities: [nvl72012-T09/mlx5_1:1]
      first_progress_after_line: 32211

l0b_expectation:
  required_evidence_lines: [12050, 12083, 12123]
  accepted_primary_lines: [12083]
  line_tolerance: 0
```

L0B expectations may identify required evidence lines/references and projection
integrity. Fingerprint gold may independently specify expected L0 deterministic and
L2 enriched roots plus merge/split groups.

`required_failure_episodes` scores typed L0 lifecycle evidence independently
from semantic RCA and action policy. A requirement selects one episode by
`lifecycle_family` and accepted start line, then may check outcome, compacted
fanout counts, exact affected entities, source dialect, a downstream terminal
line, or later progress. `nccl_net_ib` identifies the NCCL log dialect and does
not assert InfiniBand rather than RoCE. Each observed episode can satisfy at
most one requirement.

Observation-only cases use `observation_expectation`. L0 scoring checks the
selected line, optional public observation fingerprint, and required primary
absence. L1 and final semantic scoring resolve `selected_observed_failure_id`
against the unchanged `observed_failures` list; L2 scoring checks the separate
grounded observation identity. Observation scoring never substitutes an
observation fingerprint into a root-fingerprint KPI.

### L2 Audit Expectations

```yaml
l2_audit_expectation:
  - field: model_recovery_assessment
    expected: findings
    reason_class: affirmative_persistence_same_event_fanout
```

These expectations score audit behavior. They do not authorize L2 to rewrite
L1 semantics.

### Three-Track Review Projection

The review adapter projects these independent stage-owned views without
reconstructing a missing track from the final result:

```text
failure_tracks.deterministic
failure_tracks.primary
failure_tracks.observation
l3_history.job_progress
l3_history.deterministic
l3_history.primary
l3_history.observation
l4_path_selection(path, route_id, reason)
```

This is a normalized review shape, not a `gold.json` extension. Absent product
tracks are labeled `not_available`; existing approved labels do not silently
change meaning.

## Per-Route Review JSON

The review JSON is a compact index over the product result and trace:

```json
{
  "schema_version": "restart_agent_review.v1",
  "target": "qwen235b",
  "model": "nvidia/qwen/eccn-qwen-235b",
  "decision": "RESTART",
  "decision_basis": "general_retry_available",
  "retry_policy": {
    "policy_version": "retry_budget.v1",
    "base_rule": "workload_confirmation_retry",
    "effective_policy": {
      "source": "base_rule",
      "rule": "workload_confirmation_retry",
      "history_match_scope": "root_only",
      "allowed_retries": 1,
      "policy_context_id": null
    },
    "applied_policy_context": null,
    "retry_budget_exhausted": false,
    "exhausted_by": [],
    "general_root_ceiling": {
      "applicable": true,
      "history_match_scope": "root_only",
      "allowed_retries": 2,
      "matching_prior_attempts": 0,
      "observed_advance": false,
      "exhausted": false
    },
    "selected_policy_ledger": {
      "rule": "workload_confirmation_retry",
      "history_match_scope": "root_only",
      "allowed_retries": 1,
      "matching_prior_attempts": 0,
      "observed_advance": false,
      "exhausted": false
    },
    "job_no_progress_guard": {
      "history_match_scope": "same_job_no_progress",
      "allowed_retries": 3,
      "matching_prior_attempts": 0,
      "exhausted": false
    },
    "job_unknown_progress_guard": {
      "history_match_scope": "same_job_unknown_progress",
      "allowed_retries": 3,
      "matching_prior_attempts": 0,
      "exhausted": false
    }
  },
  "model_recovery_assessment": {
    "failure_domain": {
      "value": "workload",
      "status": "supported_but_unconfirmed",
      "confidence": 78
    },
    "retry_outlook_without_workload_change": {
      "value": "may_recover",
      "status": "supported_but_unconfirmed",
      "confidence": 72
    },
    "rationale": "A clean restart may recover, but the current log cannot prove it."
  },
  "primary_failure": {},
  "observed_failures": [],
  "selected_observed_failure": {},
  "failure_tracks": {
    "deterministic": {"status": "available", "identity_kind": "root"},
    "primary": {"status": "grounded"},
    "observation": {"status": "grounded"}
  },
  "l3_history": {
    "job_progress": {},
    "deterministic": {},
    "routes": {"qwen235b": {"primary": {}, "observation": {}}}
  },
  "l4_path_selection": {
    "path": "primary",
    "route_id": "qwen235b",
    "reason": "grounded_primary_available"
  },
  "primary_selection_by_stage": {},
  "observation_selection_by_stage": {},
  "current_failure_facts": {},
  "l0_kpis": {},
  "l1_kpis": {},
  "l2_kpis": {},
  "l3_kpis": {},
  "l4_kpis": {},
  "model_selection_signals": {},
  "tool_efficiency": {},
  "path_redaction_audit": {},
  "gold_score": null,
  "artifacts": {}
}
```

The review JSON retains `observed_failures`,
`selected_observed_failure`, and `observation_selection_by_stage` for
observation-only routes. The latter contains independent L0, raw L1, and
grounded L2 selections plus their line/episode relationship. The three-track
projection extends this without changing the meaning of those fields.

The product response exposes concurrent ledgers rather than a flattened retry
count. Gold `allowed_retries` scoring compares against the selected-rule budget
when present, the general root ceiling otherwise, and zero for an immediate
`workload_unrecoverable` result.

`gold_score` is null in unscored review mode.

### Model Selection Signals

`model_selection_signals` separates model behavior, route-budget outcomes, and
endpoint reliability. Its call-level source is the product trace's
`L1EvidenceResult.model_calls`; the harness does not infer endpoint health from
the final decision.

```json
{
  "model_call_summary": {
    "calls": 2,
    "model_turns": 1,
    "successful_calls": 1,
    "failed_calls": 1,
    "retried_calls": 1,
    "timeout_calls": 0,
    "deadline_exceeded_calls": 0,
    "timeout_kind_counts": {},
    "http_error_calls": 1,
    "context_window_error_calls": 0,
    "context_budget_error_calls": 0,
    "endpoint_failed_calls": 1,
    "endpoint_retried_calls": 1,
    "failed_latency_s_total": 0.42
  },
  "endpoint_status": "endpoint_issue",
  "failed_endpoint_attempts": 1,
  "context_window_error_calls": 0,
  "context_budget_error_calls": 0
}
```

The classifications are intentionally disjoint:

- `timeout_calls` counts endpoint timeouts only. `timeout_kind_counts` retains
  `connect`, `pool`, `write`, `read`, or `unknown` diagnostic subtypes for all
  timed-out HTTPX operations, including a deadline-clamped operation.
- `deadline_exceeded_calls` counts requests stopped by the route's absolute
  analysis budget. They are route-budget outcomes, not endpoint failures.
- `context_budget_error_calls` counts client-side context preflight rejection.
- `context_window_error_calls` counts recognized provider/model context-window
  rejection, regardless of HTTP status.
- `endpoint_failed_calls` excludes all three local/model-limit categories above
  and counts transport, gateway, retryable HTTP, and provider-service failures.
- `endpoint_retried_calls` is the subset of endpoint failures for which another
  provider attempt was actually scheduled.

Provider attempts and model turns are different counters: a provider retry adds
an attempt to the same logical model turn. Malformed assistant JSON, response
contract violations, and unsupported tool requests are L1 validation or model
behavior findings and do not count as endpoint failures.

`path_redaction_audit` scans model-visible transcript text, tool arguments, and
tool results for the full source path, filename, and distinctive parent path
components. It uses token boundaries for components and excludes opaque
provider metadata such as thought signatures and generated call identifiers;
incidental substrings in those fields are not evidence that the model saw a
source path.

## Gold Comparison

```json
{
  "case_id": "checkpoint-metadata-decode",
  "label_version": 1,
  "l0a": {
    "primary_evidence_coverage": true,
    "selected_primary_accuracy": true,
    "root_fingerprint_accuracy": true,
    "overall_pass": true
  },
  "l0b": {
    "required_evidence_line_recall": true,
    "primary_retained_from_l0a": true,
    "overall_pass": true
  },
  "l1": {
    "root_cause_correct": true,
    "recovery_assessment_correct": true,
    "recovery_field_results": {
      "failure_domain": true,
      "failure_domain_status": true,
      "retry_outlook_without_workload_change": true,
      "retry_outlook_status": true
    },
    "related_failure_recall": false,
    "unsupported_claims": [],
    "model_recovery_confidence": {
      "failure_domain": 78,
      "retry_outlook": 72
    },
    "core_semantic_pass": true,
    "overall_semantic_pass": false
  },
  "l2": {
    "audit_correct": null,
    "grounding_score": {
      "track": "primary",
      "grounding_status": "grounded",
      "grounded_failure_available": true,
      "grounded_anchor_line": 12083,
      "primary_anchor_correct": true,
      "causal_role_correct": true,
      "history_identity_ready": true,
      "identity_kind": "root",
      "root_fingerprint": "checkpoint:metadata_decode"
    },
    "root_fingerprint_accuracy": true,
    "primary_grounding_effect": "neutral"
  },
  "l4": {
    "retry_rule_correct": true,
    "allowed_retries_correct": true,
    "retry_exhaustion_correct": true,
    "action_correct": true,
    "cascade_correct": true,
    "policy_action_pass": true,
    "overall_semantic_pass": true
  },
  "deterministic_l4": {
    "retry_rule_correct": false,
    "action_correct": false,
    "policy_action_pass": false
  },
  "enriched_l4": {
    "retry_rule_correct": true,
    "action_correct": true,
    "policy_action_pass": true
  },
  "l4_path_comparison": {
    "action_effect": "improved",
    "policy_action_effect": "improved"
  },
  "calibration_eligibility": {
    "analysis_status": "primary_identified",
    "eligible_claim_count": 2,
    "claims": {
      "failure_domain": {
        "eligible": true,
        "reason": "substantive_primary_claim",
        "raw_confidence": 78,
        "calibration_confidence": 78,
        "gold_correct": true
      },
      "retry_outlook_without_workload_change": {
        "eligible": true,
        "reason": "substantive_primary_claim",
        "raw_confidence": 72,
        "calibration_confidence": 72,
        "gold_correct": true
      }
    }
  }
}
```

Confidence is retained per case. Calibration is a corpus-level output. Only
substantive claims under `primary_identified` are eligible; unknown claims and
canonical non-primary `confidence=1` placeholders retain `raw_confidence` but
have `calibration_confidence: null`.

`core_semantic_pass` is the L1 headline semantic result. It covers primary/RCA,
recovery fields, and unsupported claims. `related_failure_recall` remains a
separate L1 diagnostic. `l4.cascade_correct` scores the final analyzer output,
which can contain deterministic cascades even when the model did not enumerate
them. `l4.policy_action_pass` covers only retry rule/budget/exhaustion fields
that are present in gold plus the final action; it is independent of L1
semantic correctness. Deterministic-only routes have `l1: null`.

`deterministic_l4` scores the result recorded under
`analyzer_trace.decision_candidates.deterministic.result`.
`enriched_l4` exists only when the product emitted an `l1_enriched` candidate;
it is not synthesized from a deterministic final result. Both use the same gold
action and retry-policy expectations. `l4_path_comparison` classifies whether
L1 improved, regressed, or left correctness unchanged.

## Artifact Layout And Run Manifest

```text
<log-root>/<relative-log-path>
<gold-root>/<relative-log-path>/gold.json
<run-root>/<relative-log-path>/<run-id>/
```

The path remains relative to `log-root`; it is not flattened into a filename.
The generated run directory is disposable. Gold is independently durable.

Completed runs use these canonical result names; the JSON remains indented and
human-readable even though the filename no longer contains `.pretty`:

```text
restart_agent.result.json
model.<provider_model>.result.json
model.<provider_model>.trace.json
model.<provider_model>.review.json
model.<provider_model>.review.md
```

Each individual file is atomically replaced. For a model route, the product
writes trace and then result; result existence means the trace is ready. The
harness then writes review JSON and review Markdown. Routes appear in completion
order rather than configured panel order.
`restart_agent.result.json`, `restart_agent.trace.json`, `review_index.*`, and
`panel_summary.*` remain batch-final artifacts.

While a model panel is running, the run directory also contains:

```text
l0_bundle.json
decision_evidence.json
l0_model_view.json
deterministic.json
restart_agent_route_artifacts.json
live/run_status.json
live/events.jsonl
```

The three L0 files are canonical shared stage outputs. Each appears as a complete
atomic file after L0 finishes, while model routes may still be running. They contain,
respectively, the replayable L0A envelope, exact typed Decision Evidence, and
exact bounded L0B model view. They are not regenerated per route.

`restart_agent_route_artifacts.json` maps configured route IDs to canonical
`model.*.result.json` and `model.*.trace.json` paths. `run_status.json` is an
atomically replaced complete snapshot. `events.jsonl` is append-only and is the
source for console lifecycle updates, including `l0_artifacts_ready` and
`route_completed`. Events reference canonical files at the run root. The final
collect-all result appears only after completion or deadline handling, and the
final panel is then derived from the canonical product batch artifacts.

```json
{
  "schema_version": "restart_agent_eval_run.v1",
  "run_id": "20260717T120000123456Z",
  "source": {
    "relative_path": "checkpoint_logs/job.log",
    "sha256": "...",
    "byte_size": 123456
  },
  "roots": {
    "log": "/corpus/logs",
    "gold": "/corpus/restart_agent_gold",
    "run": "/corpus/restart_agent_runs"
  },
  "expected_gold_path": "/corpus/restart_agent_gold/checkpoint_logs/job.log/gold.json",
  "gold_attached": true,
  "live_artifacts": {
    "directory": "/corpus/restart_agent_runs/.../live",
    "status": "/corpus/restart_agent_runs/.../live/run_status.json",
    "events": "/corpus/restart_agent_runs/.../live/events.jsonl"
  },
  "route_artifact_manifest": "/corpus/restart_agent_runs/.../restart_agent_route_artifacts.json",
  "deterministic": "/corpus/restart_agent_runs/.../deterministic.json",
  "repositories": {
    "product": {"path": "/src/nvrx", "commit": "...", "dirty": false},
    "harness": {"path": "/src/nvrx", "commit": "...", "dirty": false}
  },
  "routes": []
}
```

`review_index.json` additionally records `shared_l0_bundle`,
`shared_decision_evidence`, and `shared_l0_model_view` paths.

`review_index.json` uses `restart_agent_review_index.v1` and indexes the same
manifest plus per-route review artifacts. `panel_summary.json` is the complete
machine-readable panel. `panel_summary.md` is the compact reviewer view;
`panel_diagnostics.md` retains exhaustive stage and interaction tables.

`review_index.md` is navigation-only: it links to the panel and per-route
artifacts instead of repeating panel KPIs. Each `model.*.review.json` contains
harness-owned evaluation data: compact route/result comparison fields, stage
KPIs, gold scores, tool/endpoint diagnostics, concerns, and artifact references.
It does not copy `l1_assessment`, `l2_grounding`, the full retry policy, or other
complete product stage outputs. Detailed L0 execution observations needed by the
panel are normalized once under `l0_kpis`; they are not repeated at the review
root.

Its sibling `review.md` renders compact semantic, KPI, and gold summaries and
links to `result.json` for complete stage output. `result.json` remains the
semantic source of truth:
`l1_assessment` is the exact parsed model output, `l2_grounding` records accepted
grounding, identity, adjustments, and findings, and top-level fields are the
final composed policy result. `trace.json` remains the source for raw provider
responses, tool interaction, and complete internal stage diagnostics.
An unresolved observation `evidence_ids` reference appears in L2 findings as
`dangling_evidence_reference`; it does not mutate `l1_assessment` or by itself
invalidate an observation grounded from its direct model-visible source line.

The compact review comparison fields are intentionally derived and named:

- `semantic_comparison`: root-cause status plus failure-domain and retry-outlook
  value/status/confidence;
- `primary_selection_by_stage`: compact L0/L1/L2 primary identity and relation;
- `history_identity_comparison`: current root/entity identifiers needed by the
  panel;
- `decision_evidence_sha256`: consistency identity for the separately stored
  shared Decision Evidence payload.

## Decision-Stability Artifact

`summarize_decision_stability.py` writes
`restart_agent_eval_stability.v1` JSON plus a Markdown view. One cohort has this
shape:

```json
{
  "cohort_id": "12-character-id",
  "target": "qwen397b",
  "model": "nvidia/qwen/eccn-qwen3-5-397b-a17b",
  "sample_count": 10,
  "minimum_samples": 10,
  "status": "observed_stable",
  "comparability": {
    "status": "verified",
    "missing_fields": [],
    "identity": {
      "source_sha256": "...",
      "product_commit": "...",
      "config_fingerprint": "...",
      "request_sha256": "...",
      "route_profile_sha256": "...",
      "l0a_sha256": "...",
      "l0b_sha256": "...",
      "initial_request_sha256": "..."
    }
  },
  "availability": {
    "usable_l1_count": 10,
    "usable_l1_rate": 1.0
  },
  "decision_stability": {
    "distribution": {"RESTART": 10},
    "modal_agreement": 1.0,
    "sequential_flips": 0,
    "sequential_flip_rate": 0.0
  },
  "semantic_stability": {
    "exact_policy_tuple": {},
    "fields": {}
  },
  "primary_and_identity_stability": {},
  "behavioral_variability": {},
  "endpoint_reliability": {},
  "gold_accuracy": {},
  "samples": []
}
```

Changed comparability identities create separate cohorts. `samples` retains
the per-run values used by every aggregate so the report can be audited without
reconstructing data from Markdown. The artifact does not contain a promotion
verdict.

## Case Result JSONL

Each line emitted by `eval_harness.py` follows:

```json
{
  "schema_version": "restart_agent_eval.v1",
  "run_id": "20260717T120000Z",
  "case_id": "checkpoint-metadata-decode",
  "target": "qwen235b",
  "status": "scored",
  "l0a_quality_correct": true,
  "l0b_quality_correct": true,
  "l1_recovery_correct": true,
  "l1_recovery_fields": {
    "failure_domain": true,
    "failure_domain_status": true,
    "retry_outlook_without_workload_change": true,
    "retry_outlook_status": true
  },
  "l2_audit_correct": null,
  "l4_retry_rule_correct": true,
  "l4_allowed_retries_correct": true,
  "l4_exhaustion_correct": true,
  "accepted_decisions": ["RESTART"],
  "actual_decision": "RESTART",
  "decision_correct": true,
  "deterministic_decision": "STOP",
  "deterministic_decision_correct": false,
  "deterministic_policy_action_correct": false,
  "enriched_decision": "RESTART",
  "enriched_decision_correct": true,
  "enriched_policy_action_correct": true,
  "l1_action_effect": "improved",
  "l1_policy_action_effect": "improved",
  "evidence_line_hit": true,
  "primary_anchor_hit": true,
  "result_path": "...",
  "trace_path": "...",
  "error": null
}
```

`status` is `scored`, `unavailable`, or `analyzer_error`.
`l4_allowed_retries_correct` means the effective rule allowance described
above; it does not refer to a top-level product response field.

## Aggregate JSON

```json
{
  "schema_version": "restart_agent_eval.v1",
  "run_id": "20260717T120000Z",
  "target": "qwen235b",
  "cases": 40,
  "scored_cases": 38,
  "unavailable_cases": 1,
  "analyzer_errors": 1,
  "l0a_quality_accuracy": 0.95,
  "l0b_quality_accuracy": 0.92,
  "l1_recovery_accuracy": 0.84,
  "l4_retry_rule_accuracy": 0.92,
  "l4_allowed_retries_accuracy": 0.97,
  "l4_exhaustion_accuracy": 0.97,
  "decision_accuracy": 0.95,
  "deterministic_decision_accuracy": 0.81,
  "deterministic_policy_action_accuracy": 0.76,
  "enriched_decision_accuracy": 0.95,
  "enriched_policy_action_accuracy": 0.92,
  "l1_action_effect_counts": {
    "improved": 7,
    "regressed": 1,
    "unchanged_correct": 28,
    "unchanged_incorrect": 2
  },
  "l1_action_improvement_rate": 0.184,
  "l1_action_regression_rate": 0.026,
  "evidence_line_hit_rate": 0.89,
  "primary_anchor_hit_rate": 0.92
}
```

`l4_allowed_retries_accuracy` aggregates the effective-rule allowance check.
The per-route review and panel artifacts retain both underlying ledgers for
diagnosis.

Each rate uses only cases where that expectation was declared. Missing gold is
not converted to failure or success.

## Panel Summary JSON

```text
run_context
shared_decision_evidence
decision_evidence_consistency
comparison_axes.semantic_quality[]
comparison_axes.behavioral_efficiency[]
comparison_axes.endpoint_reliability[]
comparison_axes.route_outcome[]
decision_path_comparison.shared_deterministic
decision_path_comparison.model_routes[]
decision_path_comparison.action_effect_counts
decision_path_comparison.policy_action_effect_counts
rows[]
concerns[]
notes[]
```

Rows preserve raw L1 recovery fields, L2 grounding/identity, L3 history facts,
and L4 retry rule/budget. The panel contains no user/not-user score.

## Run Manifest

The manifest records run id/mode/timestamps, eval and product commit/dirty
state, product schema, case and label versions, target, artifact root, and
non-secret command options. Secrets, auth headers, key-file contents, and
model-visible source labels are prohibited.
