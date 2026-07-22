# Failure Vocabulary And Structural Roles

This file is canonical for structural-role meanings and the semantic vocabulary
shared by L1, L3, and L4. `PATTERN_REGISTRY.md` owns executable deterministic
matchers and registry rows. `POLICY.md` owns retry-rule selection,
history-budget accounting, and action mapping.

## Authority Boundary

L0 does not assign semantic root cause or STOP/RESTART policy. It identifies
observed structure needed to assemble evidence:

- traceback starts and causal exception chains;
- terminal Python/framework exception summaries;
- C/C++ assertion occurrences;
- fatal/process-termination records;
- progress, checkpoint, recovery, scheduler, and teardown markers;
- diagnostic context that must remain visible but cannot become a failure
  anchor.

L1 supplies a structured operation/mechanism/component/artifact identity,
root-cause assessment, and two recovery claims. L2 audits evidence support and
derives the client-owned history identity without rewriting L1 semantics. L3
computes history facts, and L4 selects a retry rule and final action.

## Structural Failure Anchors

`observed_exception` is the generic L0 anchor for exception syntax such as:

```text
RuntimeError: ...
UnicodeDecodeError: ...
torch.AcceleratorError: ...
CheckpointException: ...
Assertion ... failed
```

The complete raw message and traceback context are preserved. The anchor does
not imply that the exception is a workload failure, infrastructure failure,
root cause, or cascade. Device-side assertions, `IndexKernel.cu`, and index
bounds failures are examples handled by this generic structure; they are not
separate hardcoded policy taxonomy rows.

The deterministic client derives the stable observed fingerprint from the
exception type, normalized message, and stable traceback context. L1 instead
describes the selected failure through its structured
`operation`/`mechanism`/`component`/`artifact_path` identity. Any `fine_class`
in the final result is client-derived L0/L2 metadata, not a model-authored L1
field.

## Diagnostic Context

The following common CUDA/PyTorch lines are diagnostic advice, not observed
failures:

- asynchronous CUDA error-reporting/possibly incorrect stacktrace warnings;
- `CUDA_LAUNCH_BLOCKING` debugging suggestions;
- `TORCH_USE_CUDA_DSA` compilation suggestions.

L0 keeps them in the raw excerpt and labels them `diagnostic_context`. They MUST
NOT become primary candidates, root fingerprints, or evidence that the
suggested condition actually occurred. In particular, a suggestion to compile
with `TORCH_USE_CUDA_DSA` does not prove a device-side assertion.

## Registry Semantic Boundary

`PATTERN_REGISTRY.md` lists the built-in detector ids, signals, matchers,
candidate roles, and fingerprint shapes. A registry match is observed structure,
not semantic root cause or an action. Model-facing registry rows are provisional
retrieval hints and omit policy authority.

Broad rows such as `python_user_exception`, `cuda_device_assert`, or a generic
`cuda_error_cascade` are intentionally excluded. They collapse observation,
cause, and policy and can promote diagnostic text or downstream symptoms.

An observed peer-GPU access failure MUST NOT be promoted deterministically to
`gpu_hardware_fault`. L1 may identify infrastructure as the leading explanation,
but without Xid, ECC, driver, fabric-health, DCGM, kernel, or equivalent evidence,
the hardware root-cause status is at most `supported_but_unconfirmed`. The model
must retain invalid or nonexistent client-selected peer memory as a material
alternative.

## Semantic And Policy Vocabulary

L1 emits a structured primary identity and exactly two recovery claims:

- `failure_domain.value`: `workload`, `infrastructure`, or `unknown`;
- `retry_outlook_without_workload_change.value`: `cannot_recover`,
  `may_recover`, or `unknown`;
- each claim's `status`: `established_by_current_log`,
  `supported_but_unconfirmed`, `hypothesis_only`, or `unknown`;
- each claim's independent `confidence` from 1 to 99.

`workload` includes application code, model/data/configuration, and
workload-selected framework or library behavior. Ownership within that stack is
not policy ambiguity.

The recovery outlook uses the next NVRx cycle after normal process teardown and
restart delay as its reference point. The workload is unchanged, while process
state is recreated and hardware allocation or mutable external-service state
may change. `cannot_recover` with `established_by_current_log` therefore
requires direct current-log evidence that those normal transition effects are
insufficient. Same-attempt fanout and diagnostic suggestions do not establish
cross-attempt persistence. Long-term remediation and cross-attempt persistence
are not part of the L1 vocabulary; L3 owns observed recurrence.

A fixed request for a mutable resource does not prove that conflicting state
will persist across the cycle boundary. Absence of release or cleanup messages
is not persistence evidence.

`POLICY.md` owns the L4 retry-rule vocabulary and selection order. Registry
`policy_class` values remain evidence-origin metadata in the current
implementation; they are not L4 inputs or public policy scores.
