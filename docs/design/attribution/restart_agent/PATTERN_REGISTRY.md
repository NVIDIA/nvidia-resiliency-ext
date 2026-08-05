# L0 Pattern Registry

This document catalogs active deterministic detectors that turn raw log text
into typed L0 observations. It answers three questions:

1. What does L0 recognize?
2. What fact or candidate does a match emit?
3. What safety constraint prevents the match from being overinterpreted?

`L0A.md` owns how observations become occurrence groups, episodes, incidents,
Decision Evidence, and deterministic primary selection. `TAXONOMY.md` defines
shared role meanings. L1-L4 own semantics, history, and action.

## Executable Shape

The implementation has two detector forms:

| Form | Code owner | Output |
| --- | --- | --- |
| Signature registry | `l0/registry.py` | Failure candidate, cascade candidate, cause confirmation, fingerprint input, and optional recovery behavior |
| Structural parsers | `l0/assembly.py` | Progress, checkpoint, setup, lifecycle, metadata, operation/artifact, diagnostic, and failure-episode facts |

The executable code is canonical for exact matching syntax. This document
records behavior and safety constraints rather than duplicating regexes.

`SignatureRegistryRow` contains only `registry_id`, `pattern`, `role`, and
`recovery_behavior`. Every row listed below is active. The implementation does
not currently have runtime `candidate`, `profile`, `built_in`, or
`l1_proposed` authority states.

## Failure Signatures

| Registry id | Observed signal | L0 role or annotation | Safety constraint |
| --- | --- | --- | --- |
| `gpu_hardware_fault` | Explicit Xid, uncorrectable ECC, GPU off-bus, link-health, PCIe fatal, or thermal fault | root candidate | A bare hardware-component mention is insufficient. |
| `peer_gpu_memory_access_failure` | Explicit invalid peer-GPU memory access | root candidate | Hardware is plausible, but invalid workload/library access remains possible. |
| `infra_policy_event` | Explicit preemption or node-failure event | root candidate | Records the observed event; L0 does not choose policy. |
| `time_limit` | Explicit scheduler or wall-time termination event | root candidate | Records the observation only. Scheduler time policy is outside Restart Agent authority and receives no special L4 exemption. |
| `user_cancelled` | Explicit user cancellation | root candidate | Requires cancellation language, not generic teardown. |
| `observed_exception` | Generic exception-summary or assertion syntax | root candidate | Traceback frames and diagnostic advice are excluded; semantic ownership remains unresolved. |
| `configuration_validation_failure` | Explicit invalid option/configuration or missing configuration/key | root candidate | Records validation failure without assigning user ownership or recoverability. |
| `argument_validation_failure` | Explicit invalid argument | root candidate | The failing caller or component remains unresolved. |
| `artifact_or_path_not_found` | Explicit missing file or path | root candidate | Does not distinguish bad configuration from transient storage visibility. |
| `checkpoint_compatibility_mismatch` | Explicit checkpoint mismatch | root candidate | Does not establish whether the checkpoint, reader, or configuration is incompatible. |
| `shape_mismatch` | Explicit shape mismatch | root candidate | Does not establish which input, component, or configuration is responsible. |
| `filesystem_permission_denied` | `PermissionError`, EACCES, or permission denied | root candidate | Establishes failed access, not ownership or persistence. |
| `cuda_oom` | CUDA allocation failure | root candidate | Does not establish whether restart can recover. |
| `nan_or_inf` | Non-finite loss, gradient, activation, or signal | root candidate | Workload recovery and recurrence remain separate questions. |
| `bad_token_or_window` | Explicit bad token/sample/window handling | root candidate plus `retry_then_skip` annotation | L4 uses the annotation only with a matching declared capability and affected entity. |
| `framework_crash` | Segfault, illegal instruction, or core dump | root candidate | Does not determine workload versus infrastructure domain. |
| `linux_oom_kill_confirmation` | Explicit scheduler/kernel OOM-kill record | cause confirmation | A bare `Killed` line does not match. |
| `observed_distributed_operation_timeout` | Direct watchdog or collective-operation timeout | root candidate | Chronology may later associate it as a cascade behind an earlier root. |
| `nccl_cascade` | NCCL watchdog, abort, timeout, or system-error fallout | cascade candidate | Search for an earlier initiating failure. |
| `cuda_previous_error_cascade` | Failure explicitly attributed to an earlier CUDA/capture error | cascade candidate | Generic CUDA error text is insufficient. |

Registry matches emit an observed failure class and source provenance. They do
not emit failure domain, retry outlook, or `STOP`/`RESTART`.

## Progress And Setup Detectors

| Detector id | Typed observation | Required interpretation |
| --- | --- | --- |
| `megatron_iteration_summary.v1` | Completed iteration, total iterations, consumed samples, timestamp, and locality | Only an increasing completed marker is application progress. |
| `checkpoint_complete.v1` | Completed checkpoint step | Requires explicit completion/success plus a parseable step; start, timer, config, deletion, and failed write text are not progress. |
| `recovery_marker.v1` | Recovery/continuation wording | Context only; it does not replace a later completed progress marker. |
| `checkpoint_load_complete.v1` | Completed checkpoint load and optional iteration | Setup progress, not training/checkpoint-save progress. |
| `checkpoint_load_start.v1` | Started checkpoint load and iteration | Started work does not prove completion. |
| `checkpoint_reshard.v1` | Checkpoint sharding change observed | Setup context only. |
| `checkpoint_metadata_load_complete.v1` | Completed checkpoint metadata load | Setup context for the observed operation. |
| `optimizer_setup_start.v1` | Optimizer setup started | Started work does not prove completion. |
| `cuda_graph_build_complete.v1` | CUDA graph build completed | Setup progress, not application iteration progress. |

## Lifecycle, Metadata, And Diagnostic Detectors

| Detector id | Observation | Constraint |
| --- | --- | --- |
| `megatron_training_iterations_total.v1` | Configured training-iteration total | Metadata, not progress. |
| `megatron_training_start_datetime.v1` | Training-start lifecycle boundary | Boundary, not progress. |
| `megatron_rerun_iteration_reset.v1` | Rerun iteration baseline reset | Must not be compared as completed application progress. |
| `world_size_config.v1` | Explicit world size | Job metadata; inferred rank count remains only a lower bound. |
| `rank_gpu_mapping_warning.v1` | Warning that rank-to-GPU mapping may be heterogeneous | Diagnostic; rank is not treated as GPU identity. |
| `nccl_version.v1` | NCCL version | Environment metadata, not a failure. |
| `cuda_async_reporting_advice.v1` | CUDA asynchronous-reporting advice | Diagnostic context, never a failure anchor. |
| `cuda_launch_blocking_advice.v1` | `CUDA_LAUNCH_BLOCKING` advice | Diagnostic context, not evidence the suggested condition occurred. |
| `cuda_dsa_compile_advice.v1` | `TORCH_USE_CUDA_DSA` advice | Diagnostic context, not proof of a device-side assertion. |
| `conditional_cause_language.v1` | “may/might/could be caused by” language | Diagnostic hypothesis, not an observed cause. |

## Operation And Artifact Detectors

These parser families produce comparisons rather than registry failure classes:

| Family | Facts retained | Constraint |
| --- | --- | --- |
| Checkpoint save/load | Operation, iteration, artifact path, start/completion/failure lines, and observer locality | Success applies only at the observed artifact identity strength. |
| Dataloader read | Physical unit, data region, integrity marker, outcome, and observer locality | Success on another file, shard, or region does not establish current-unit health. |
| Path access | Configured read/write/cache paths and failed-access paths | Paths are string evidence, not proof of effective UID, owner, mode, or ACL. |

## Failure-Structure Detectors

L0 also recognizes traceback starts, exception chains, terminal operation
timeouts, bare process kills, explicit process termination, scheduler
cancellation, cleanup frames, and teardown calls. These establish chronology
and causal-role hints:

- a bare process kill has unknown cause until confirmation appears;
- process termination and scheduler cancellation are downstream lifecycle facts;
- cleanup exceptions remain teardown rather than competing roots; and
- a later confirmation may be associated with the preceding compatible episode
  only when no incompatible progress intervenes.

## Global Safety Rules

- Matching text records an observation; it does not establish semantic domain,
  restart persistence, or policy.
- Diagnostic and lifecycle rows cannot seed a primary or root fingerprint.
- Completed progress requires an explicit completed marker with a comparable
  value; configuration and starts are not progress.
- Same-attempt copies are grouped rather than treated as recurrence.
- Cascade and teardown variants retain bounded representatives and aggregate
  counts instead of becoming independent roots.
- Rank, node, and GPU remain separate unless the log supplies an explicit
  mapping.
- Generic exception structure is preferred over accumulating message-specific
  error patterns.
- Fast trigger gates and volatile-token normalization are execution
  optimizations only. Boundary-aware gates must admit every line that can match
  a registry row, and optimized normalization must preserve canonical
  occurrence shapes.
- L0A preserves source references and coverage/lossiness information for every
  detector family used in decision evidence.

## Change Qualification

A new detector should be added only when:

1. it represents a reusable structural or operational concept rather than one
   incident's wording;
2. its emitted typed fact and safety constraint are explicit;
3. reviewed corpus examples demonstrate both matches and important non-matches;
4. regression tests cover primary, cascade, diagnostic, and progress effects as
   applicable; and
5. the eval harness shows that it improves evidence quality without introducing
   material false anchors or fingerprint collisions.

Pattern discovery, corpus hit counts, candidate promotion discussions, and
per-run evidence belong to the eval harness or issue history, not this product
reference.
