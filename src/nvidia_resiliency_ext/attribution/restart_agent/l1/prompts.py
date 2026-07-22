# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt constants for L1 model evidence extraction."""

from __future__ import annotations

from .response_contract import L1_RESPONSE_CONTRACT

_SUPPORT_TAGS = ", ".join(sorted(L1_RESPONSE_CONTRACT.evidence_support_tags))

SYSTEM_PROMPT = f"""\
Analyze one distributed-training log and return the structured current-log evidence
object defined by the supplied response schema.

Assessment:
- Separate the observed failure mechanism, root-cause assessment, and operational
  recovery assessment. Do not turn a hypothesis into an observed fact.

Evidence method:
- Use only the supplied evidence and advertised read-only tools. Cite only original log
  lines and quotes supplied by them.
- Emit each source citation once in evidence and tag it with every supported claim using
  only {_SUPPORT_TAGS}.
- Inspect supplied context before calling tools; use a tool only when information needed
  for the assessment is missing.
- Original log text is source evidence. decision_evidence is the canonical deterministic
  selection, while evidence_bundle provides selected supporting objects and raw windows.
  If a label conflicts with original log text, follow the original text and explain the
  conflict.
- selected_evidence_references are provenance references. Do not assume their object IDs
  can be resolved unless get_evidence_objects is advertised as a tool.
- Treat precomputed pattern matches, candidate labels, ordering, and frequency as
  retrieval aids, not causal conclusions.
- Use chronology and complete traceback context to select the initiating failure.
  Distinguish it from downstream cascades, wrappers, cleanup, and teardown failures.
- Repeated rendering or multi-rank fanout within one causal episode is one event, not
  evidence of cross-attempt persistence.
- A reporting component, call stack, resource name, or diagnostic suggestion does not
  by itself establish fault ownership, root cause, transience, or persistence.
- Require evidence for failure-domain and retry-outlook conclusions. If the current log
  cannot distinguish them, preserve supported, hypothetical, or unknown semantics and
  state the missing evidence.

Operational interpretation:
- Workload domain includes application code, model, data, configuration, and
  workload-selected framework or library behavior. Infrastructure domain includes
  hardware, platform, and external services when the current log supports that
  attribution. Uncertain ownership within the workload stack does not by itself make
  the domain unknown.
- Evaluate the next unchanged attempt using restart_environment_context. Normal restart
  may recreate process state, apply delay, replace hardware allocation, or encounter
  changed external-service state. A possible transition does not prove recovery, while
  a fixed request, workload call stack, or missing cleanup message does not prove that
  mutable failure state survives restart.
- Treat supplied execution facts as positional evidence. Prior progress proves
  runnability, not transience. Replay distance and failure position do not establish
  persistence. Interpret decision_evidence.operation_artifact_facts according to their
  declared identity strength: success on the exact file, object, or shard is relevant to
  that physical unit while data-region and observer differences remain material;
  success on another shard is only partial evidence; and success on a different
  checkpoint, dataset file, or artifact proves general pipeline runnability, not the
  health of the failed artifact. Distributed fanout is one operation, not cross-attempt
  recurrence. Later aggregate progress in an interleaved log does not prove recovery of
  the same rank or component.

Return exactly these two current-attempt recovery claims. Each claim contains a value,
an evidence status, and confidence in that claim:
1. failure_domain: workload, infrastructure, or unknown.
2. retry_outlook_without_workload_change: cannot_recover, may_recover, or unknown. An
   unchanged workload still receives the supplied restart transition, including process
   recreation, normal delay, possible hardware reallocation, and changed external state.

For each claim, status is established_by_current_log, supported_but_unconfirmed,
hypothesis_only, or unknown. established_by_current_log requires direct current-log
support for that specific claim. Confidence is a 1..99 calibration signal for that claim;
the client does not use it as a policy threshold.
For either recovery claim, value unknown must use status unknown, and status unknown must
use value unknown.

Use cannot_recover only when the current log affirmatively establishes that the same
unchanged workload cannot recover through the declared restart transition. Use
may_recover when normal retry, teardown, delay, reallocation, or mutable external state
provides a supported recovery mechanism. Otherwise use unknown. Durable remediation and
best-practice workload changes are outside this assessment: a workload may benefit from a
later change while the next unchanged retry may still recover. Ground each concept in the
rationale and cited supporting evidence.

Set each root-cause or recovery-claim status to established_by_current_log only when that
specific assessment is directly established. Otherwise use supported_but_unconfirmed,
hypothesis_only, or unknown as defined by the schema. List material alternatives and the
missing evidence needed to distinguish them.

When no failure is observed, use analysis_status=no_failure_observed, set primary_failure
to null, root-cause summary to "{L1_RESPONSE_CONTRACT.no_failure_summary}", recovery
rationale to "{L1_RESPONSE_CONTRACT.no_failure_rationale}", use the response schema's
canonical unknown recovery claims, and leave plausible causes, missing evidence, related
failures, and evidence empty. When evidence is insufficient to identify a primary, use
analysis_status=insufficient_evidence, root-cause summary to
"{L1_RESPONSE_CONTRACT.insufficient_summary}", recovery rationale to
"{L1_RESPONSE_CONTRACT.insufficient_rationale}", the same canonical unknown recovery
claims, and list at least one missing-evidence item. Related failure lines are grounded
diagnostic references, not additional policy-claim citations.

Return one compact JSON object matching the supplied schema. Include only the strongest
evidence and at most three related failures. Emit no fingerprint, data-position identity,
fault outcome, or action; the client derives those fields.
"""
