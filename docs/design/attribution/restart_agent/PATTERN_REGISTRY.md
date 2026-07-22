# Pattern Registry

This file is canonical for the executable and candidate registry used in
deterministic L0 bundle construction. `TAXONOMY.md` owns semantic vocabulary and
structural roles. `POLICY.md` owns STOP/RESTART policy and history
interpretation. This registry owns how raw log lines become progress markers,
candidate failures, cascades, lifecycle facts, diagnostics, or ignored noise.

The first rows were seeded from a small sanitized corpus; subsequent rows were
reviewed against the external eval corpus described by the Restart Agent
harness. Rows marked `built_in` have been accepted as generic deterministic
bundle-construction rules. Rows marked `candidate` or `profile` need narrower
review before production use.

## Source Corpus

The historical seed was the five sanitized nvbug 6323419 logs listed below.
Those logs are not shipped with the product, and no developer-local path is a
registry dependency. Current qualification logs and their human-reviewed gold
are discovered through the eval harness's mirrored log/gold roots; gold labels
bind provenance with `source_sha256`.

Samples:

- `llama4_scout_llama4_gpu_error_first_iter_post`
- `llama4_scout_llama4_gpu_error_steady_mid`
- `n3_super_gpu_error_scheduler_step_real_002`
- `n3_super_gpu_error_rank31_optimizer_step_real_002`
- `n3_super_gpu_error_loss_reduce_real_002`

Historical seed coverage:

- 5 logs, 32,882 total lines.
- 22 Megatron iteration summary lines.
- 5 explicit `training.runtime` injected GPU error lines.
- 5 CUDA index-kernel bounds assertions.
- 70 CUDA device-side assert lines, many repeated downstream symptoms.
- 109 NCCL collective-timeout dump lines.
- 335 NCCL CUDA-failure warning lines.
- No completed checkpoint-save lines were found. Checkpoint config lines were
  present and are intentionally not treated as checkpoint progress.

## Authority Model

Each row has an authority state:

- `candidate`: observed in logs, review required before production use.
- `profile`: accepted for a workload/profile after review or validation.
- `built_in`: accepted as generic deterministic behavior.
- `l1_proposed`: proposed by an L1 model or discovery job; never authoritative
  until promoted to `profile` or `built_in`.

Production bundle construction SHOULD use `built_in` and selected `profile`
rows. `candidate` and `l1_proposed` rows can be emitted as observability or eval
signals, but high-impact policy such as history-based STOP should not depend on
them as authoritative facts.

## Row Contract

Each row SHOULD define:

- `pattern_id`: stable identifier.
- `group`: `progress`, `checkpoint`, `recovery`, `root_candidate`,
  `cascade_candidate`, `lifecycle`, `diagnostic`, `job_metadata`, or `noise`.
- `matcher`: deterministic regex or parser contract.
- `extracts`: stable fields L0 should parse.
- `validation`: checks required before emitting structured evidence.
- `taxonomy_link`: matching `TAXONOMY.md` row, when this is failure evidence.
- `bundle_use`: how this row contributes to the evidence bundle.
- `promotion_note`: review or rollout condition.

## Built-In Failure Signature Rows

These executable L0 rows supplement structural exception parsing for important
signals that may not have ordinary exception syntax:

| Registry id | Signal | Candidate role | Semantic constraint |
| --- | --- | --- | --- |
| `gpu_hardware_fault` | Xid, uncorrectable ECC, GPU off-bus, explicit NVLink/PCIe link-health failure, thermal shutdown/fault | root candidate | A bare mention of NVLink, PCIe, or temperature is insufficient. |
| `peer_gpu_memory_access_failure` | Invalid peer-GPU memory access, including access reported over NVLink | root candidate | Hardware is plausible, but invalid workload/library access remains possible without corroboration. |
| `infra_policy_event` | Preemption or explicit node failure | root candidate | Observed external/infrastructure event. |
| `time_limit` | Scheduler or wall-time limit | root candidate | Context distinguishes expected policy from configuration error. |
| `user_cancelled` | Explicit user cancellation | root candidate | External user action, not a code exception. |
| `user_config_error` | Explicit invalid or missing configuration | root candidate | Must not override a better traceback terminal exception. |
| `filesystem_permission_denied` | `PermissionError`, EACCES, or permission denied | ambiguous root candidate | Establishes failed access, not ownership or persistence across restart. |
| `cuda_oom` | CUDA allocation failure | root candidate | Restartability remains semantic/history dependent. |
| `nan_or_inf` | Non-finite loss, gradient, or activation | root candidate | May require workload recovery or history. |
| `bad_token_or_window` | Explicit bad token/sample/window handling | root candidate | Workload-managed retry/skip policy remains deferred. |
| `framework_crash` | Segfault, illegal instruction, or core dump | root candidate | L1 still determines semantic domain. |
| `linux_oom_kill_confirmation` | Explicit scheduler/kernel OOM-kill record | cause confirmation | A bare `Killed` line never matches this row. |
| `observed_distributed_operation_timeout` | Direct watchdog/collective timeout | root candidate or downstream symptom | Becomes a cascade when a stronger initiating failure precedes it. |
| `nccl_cascade` | Downstream NCCL watchdog/abort/dump fallout | cascade candidate | Search for an earlier initiating failure. |
| `cuda_previous_error_cascade` | Explicit failure due to a previous capture error | cascade candidate | Generic `CUDA error` text is insufficient. |

`TAXONOMY.md` owns the meanings of structural and semantic roles. This table owns
the executable detector inventory; it does not assign `failure_domain`, recovery
semantics, or `STOP`/`RESTART`.

## Registry Rows

### `megatron_iteration_summary.v1`

- `authority`: `built_in`
- `group`: `progress`
- `matcher`:

```regex
^\s*(?P<rank_prefix>\d+:\s*)?\[(?P<timestamp>[^\]]+)\]\s+iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total_iterations>\d+)\s*\|\s*consumed samples:\s*(?P<consumed_samples>\d+)\s*\|
```

- `extracts`: `rank_prefix`, `timestamp`, `iteration`, `total_iterations`,
  `consumed_samples`; optionally metrics such as learning rate, loss, skipped
  iterations, and nan iterations as context only.
- `validation`: `iteration` and `consumed_samples` must be comparable with
  previous markers from the same attempt. Greater values are application
  progress. Equal or lower values are not progress.
- `bundle_use`: emit `application_progress` marker with
  `marker_type=iteration`, `state=completed`, `value=iteration`, and
  `secondary_value.consumed_samples`.
- `evidence`: 22 hits across all 5 logs. Examples include line 1151 and 1162
  in `llama4_scout_llama4_gpu_error_first_iter_post`, line 1173 in
  `llama4_scout_llama4_gpu_error_steady_mid`, and line 5159 in each n3 log.
- `promotion_note`: accepted as a built-in Megatron-style progress row. It
  remains subject to monotonic validation; matching text alone does not prove
  progress if the marker value does not advance.

### `megatron_training_iterations_total.v1`

- `authority`: `candidate`
- `group`: `lifecycle`
- `matcher`:

```regex
\bsetting training iterations to (?P<total_iterations>\d+)\b
```

- `extracts`: `total_iterations`.
- `validation`: numeric parse only.
- `bundle_use`: attempt metadata. This is not progress.
- `evidence`: 5 hits, one per log.
- `promotion_note`: useful to interpret `iteration N/T`, but should never prove
  application progress by itself.

### `world_size_config.v1`

- `authority`: `built_in`
- `group`: `job_metadata`
- `matcher`:

```regex
\bworld_size\b\s*(?:[.=:\-\s]+)\s*(?P<world_size>\d+)\b
```

- `extracts`: explicit `world_size`.
- `validation`: numeric parse only. Treat as job metadata, not progress and not
  a failure signal.
- `bundle_use`: emit `job_metadata.explicit_world_size`,
  `job_metadata.explicit_world_size_line`, and set
  `job_metadata.world_size_source=explicit`. If no explicit row is found, L0 may
  still emit `inferred_world_size_lower_bound=max(observed_rank)+1`, but that is
  only a lower bound.
- `evidence`: observed in ELK production-style logs such as
  `bad_token_grad_inf_logs/55B_hybrid_moe_25T_phase2_1745223_date_26-02-20_time_03-29-36.log`.
- `promotion_note`: accepted as built-in job-shape metadata. It must not affect
  STOP/RESTART directly.

### `megatron_training_start_datetime.v1`

- `authority`: `candidate`
- `group`: `lifecycle`
- `matcher`:

```regex
\[before the start of training step\] datetime: (?P<timestamp>.+)$
```

- `extracts`: `timestamp`.
- `validation`: timestamp parse when possible.
- `bundle_use`: lifecycle boundary for context windows and progressive status.
  This is not progress.
- `evidence`: 5 hits, one per log.
- `promotion_note`: safe metadata row if used only as a boundary.

### `megatron_rerun_iteration_reset.v1`

- `authority`: `candidate`
- `group`: `lifecycle`
- `matcher`:

```regex
\bOverwriting rerun_state_machine\.current_iteration from (?P<old>-?\d+) to (?P<new>-?\d+)
```

- `extracts`: `old`, `new`.
- `validation`: numeric parse only.
- `bundle_use`: baseline marker for restart/rerun machinery. This is not
  application progress.
- `evidence`: 5 hits, one per log.
- `promotion_note`: keep as lifecycle metadata. Do not compare this with
  application iteration progress.

### `megatron_checkpoint_saved_iteration.v1`

- `authority`: `built_in`
- `group`: `checkpoint`
- `matcher`:

```regex
\bsuccessfully saved checkpoint from iteration\s+(?P<iteration>\d+)\b
```

- `extracts`: checkpoint iteration.
- `validation`: require explicit successful-save language. Do not match
  checkpoint timers, checkpoint configuration, checkpoint deletion, checkpoint
  start/in-progress messages, or failed/partial checkpoint writes.
- `bundle_use`: emit `checkpoint_progress` marker with
  `marker_type=checkpoint`, `state=completed`, and `value=iteration`.
- `evidence`: observed in ELK production-style logs such as
  `bad_token_grad_inf_logs/55B_hybrid_moe_25T_phase2_1745223_date_26-02-20_time_03-29-36.log`.
- `promotion_note`: accepted as built-in progress evidence because it is a
  committed-state marker, not an error signature.

### `observed_exception.v1`

- `authority`: `built_in`
- `group`: `structural_failure_anchor`
- `matcher`: generic exception-summary or assertion-occurrence syntax, not a
  list of error messages.
- `extracts`: exception type, complete raw message, source line, and nearby
  traceback context when available.
- `validation`: debugging advice and traceback frames are not exception
  summaries. Causal role and semantic class remain unresolved for L1.
- `bundle_use`: seed the complete failure episode and bounded raw excerpt.
- `fingerprint`: client-derived from observed exception type, normalized
  message, and stable traceback context.
- `promotion_note`: `index out of bounds`, device-side assertions, and similar
  strings remain message content and eval examples; they are not standalone L0
  taxonomy rules.

### `cuda_pytorch_diagnostic_context.v1`

- `authority`: `built_in`
- `group`: `diagnostic_context`
- `matcher`: stable CUDA/PyTorch advice for asynchronous reporting,
  `CUDA_LAUNCH_BLOCKING`, or `TORCH_USE_CUDA_DSA`.
- `validation`: match the advice form, not an actual assertion occurrence.
- `bundle_use`: retain the line in the raw excerpt with
  `line_role=diagnostic_context`; never seed a primary candidate or root
  fingerprint.
- `promotion_note`: a suggestion does not prove that the suggested condition
  occurred and is not positive evidence for a transient alternative.

### Setup progress patterns

`checkpoint_load_complete.v1` recognizes an explicit successful checkpoint
load and extracts its iteration when present. `cuda_graph_build_complete.v1`
recognizes an explicit completed CUDA-graph build. L0 deduplicates these into
bounded `setup_progress` markers while retaining grouped line/rank counts.
Neither pattern is a failure candidate, forward-progress marker, recovery
signal, or history-comparison value.

### `nccl_collective_timeout_dump.v1`

- `authority`: `built_in`
- `group`: `cascade_candidate`
- `matcher`:

```regex
Received a dump signal due to a collective timeout from\s+(?P<peer_rank>this local rank|rank\s+\d+).*Last enqueued NCCL work:\s*(?P<last_enqueued>\d+), last completed NCCL work:\s*(?P<last_completed>\d+)
```

- `extracts`: `peer_rank`, `last_enqueued`, `last_completed`, local rank when
  parseable from rank prefix.
- `validation`: if an earlier root candidate exists in the same failure window,
  classify as cascade. If this is the first strong error in a log, emit as
  ambiguous NCCL candidate for L1/policy review.
- `taxonomy_link`: `nccl_cascade`.
- `bundle_use`: cascade grouping, affected rank fanout, context-window seed.
- `fingerprint`: `nccl_cascade:collective_timeout_dump`.
- `evidence`: 109 hits across all logs.
- `promotion_note`: accepted as a built-in cascade row. If no earlier strong
  root candidate exists, it may still seed an ambiguous NCCL root candidate.

### `nccl_flight_recorder_failed_update.v1`

- `authority`: `built_in`
- `group`: `cascade_candidate`
- `matcher`:

```regex
Failed to update state for entry (?P<entry_id>\d+): nccl:(?P<collective>[A-Za-z0-9_]+) with error: CUDA error: device-side assert triggered
```

- `extracts`: `entry_id`, `collective`.
- `validation`: high-volume repeats should be grouped by normalized collective
  and error, not emitted as separate roots.
- `taxonomy_link`: `nccl_cascade`.
- `bundle_use`: cascade evidence and collective context.
- `fingerprint`: `nccl_cascade:failed_update:{collective}:cuda_runtime_error`.
- `evidence`: 51 hits in the two llama4 logs.
- `promotion_note`: accepted as a built-in cascade row.

### `nccl_cuda_failure_warn.v1`

- `authority`: `built_in`
- `group`: `cascade_candidate`
- `matcher`:

```regex
NCCL WARN Cuda failure(?:\s+710)? 'device-side assert triggered'
```

- `extracts`: optional CUDA status code `710`.
- `validation`: group aggressively. This is typically late cleanup noise after
  the actual CUDA assert.
- `taxonomy_link`: `cuda_previous_error_cascade` when the line explicitly says
  it follows a previous error; otherwise `nccl_cascade`.
- `bundle_use`: cascade count and affected-rank evidence; do not seed primary
  root unless no earlier root/error candidate exists.
- `fingerprint`: `nccl_cascade:cuda_failure:{normalized_error}`.
- `evidence`: 335 hits, 67 per log.
- `promotion_note`: accepted as a built-in high-volume cascade/noise reducer.

### `process_group_watchdog_exception.v1`

- `authority`: `built_in`
- `group`: `cascade_candidate`
- `matcher`:

```regex
Process group watchdog thread terminated with exception: (?P<exception>.+)$
```

- `extracts`: `exception`, process group id/name when parseable.
- `validation`: if exception matches CUDA/NCCL symptom and an earlier root
  exists, classify as cascade.
- `taxonomy_link`: `nccl_cascade` or `cuda_previous_error_cascade` when the
  message explicitly identifies a previous error.
- `bundle_use`: cascade evidence and process-group context.
- `fingerprint`: `nccl_cascade:process_group_watchdog:{normalized_exception}`.
- `evidence`: 6 hits in the two llama4 logs.
- `promotion_note`: accepted as a built-in cascade row.

### `rank_gpu_mapping_warning.v1`

- `authority`: `built_in`
- `group`: `diagnostic`
- `matcher`:

```regex
Guessing device ID based on global rank\. This can cause a hang if rank to GPU mapping is heterogeneous
```

- `extracts`: local rank when parseable.
- `validation`: warning only.
- `taxonomy_link`: none.
- `bundle_use`: diagnostic context. It can explain why rank is not equivalent
  to GPU, but it is not root failure evidence by itself.
- `evidence`: 28 hits across all logs.
- `promotion_note`: accepted as diagnostic context only. It is not root failure
  evidence and must not be treated as a generic fault.

### `nccl_version.v1`

- `authority`: `candidate`
- `group`: `diagnostic`
- `matcher`:

```regex
\bNCCL version (?P<version>\S+)
```

- `extracts`: `version`.
- `validation`: version parse only.
- `taxonomy_link`: none.
- `bundle_use`: environment metadata. This is not progress and not failure
  evidence.
- `evidence`: 112 hits across all logs.
- `promotion_note`: optional metadata row; low policy value.

## Initial Nvbug Checkpoint Findings

The initial nvbug 6323419 source-log corpus did not justify a checkpoint-progress
row. It contains checkpoint configuration keys such as
`ckpt_fully_parallel_save`, `ckpt_format`, and `ckpt_step`, but no completed
checkpoint-save event.

Do not infer `checkpoint_progress` from configuration. The
`megatron_checkpoint_saved_iteration.v1` row is based on later ELK
production-style logs and requires explicit successful-save language plus a
parseable iteration.

## Resolved Review Decisions

1. `megatron_iteration_summary.v1` is accepted as built-in Megatron-style
   application progress, provided marker values advance monotonically.
2. NCCL timeout, flight-recorder, CUDA-failure warning, and watchdog rows are
   accepted as built-in cascade rows. If no earlier strong root exists, an NCCL
   timeout can still seed an ambiguous root candidate.
3. `rank_gpu_mapping_warning.v1` is diagnostic only. It may preserve caution
   around rank-to-GPU inference, but it is not a fault signature.
4. Test-injection markers and example-specific CUDA messages stay in eval/gold
   metadata, not the generic product registry.
5. Generic exception/assertion structure seeds evidence episodes; CUDA/PyTorch
   debugging advice is diagnostic context only.

## Open Review Questions

1. Should any lifecycle rows above be promoted from `candidate` to `built_in`,
   given that they do not affect policy directly?
