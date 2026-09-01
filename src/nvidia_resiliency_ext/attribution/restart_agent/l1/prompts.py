# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt constants for L1 model evidence extraction."""

from __future__ import annotations

from .cluster_context import DEFAULT_CLUSTER_EXECUTION_CONTEXT, render_cluster_execution_context
from .response_contract import L1_RESPONSE_CONTRACT

_SUPPORT_TAGS = ", ".join(sorted(L1_RESPONSE_CONTRACT.evidence_support_tags))
_CLUSTER_EXECUTION_CONTEXT = render_cluster_execution_context(DEFAULT_CLUSTER_EXECUTION_CONTEXT)

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
- Original log text is source evidence. failure_narrative is deterministic orientation,
  decision_evidence_view is the compact deterministic selection, and evidence_bundle
  provides selected supporting objects and raw windows. Inspect failure_narrative first,
  then use the other sections to verify or extend it.
  If a label conflicts with original log text, follow the original text and explain the
  conflict.
- decision_evidence_view.selected_evidence_references are trace references. Do not assume
  their object IDs can be resolved unless get_evidence_objects is advertised as a tool.
  An unresolved object ID or source-line reference does not expose evidence and cannot
  support a claim.
- Treat the deterministic narrative, precomputed pattern matches, candidate labels,
  ordering, and frequency as retrieval aids, not semantic cause, domain, recovery, or
  action conclusions.
- Use chronology and complete traceback context to identify the initiating failure.
  Distinguish it from downstream cascades, wrappers, cleanup, and teardown failures.
- primary_failure.causal_role may be initiating or unknown. Put known cascade and
  teardown events in related_failures only after a primary has been identified. If the
  log contains only downstream or teardown evidence, use insufficient_evidence rather
  than promoting that evidence to primary. Preserve up to three visible failure surfaces
  in observed_failures. Choose exactly one only when it is the unique terminal surface
  after excluding retry-pending, recovered, progressed-after, and diagnostic-only
  observations; independent tied surfaces leave selected_observed_failure_id null.
- Repeated rendering or multi-rank fanout within one causal episode is one event, not
  evidence of cross-attempt persistence.
- A reporting component, call stack, resource name, or diagnostic suggestion does not
  by itself establish fault ownership, root cause, transience, or persistence.
- In failure_identity, affected_artifact_path means:
  {L1_RESPONSE_CONTRACT.affected_artifact_path_description}
- In failure_identity, direct_failure_object_path means:
  {L1_RESPONSE_CONTRACT.direct_failure_object_path_description}
  The direct object and enclosing affected artifact are independent; do not infer one from
  the other.
  Example: an NCCL timeout during checkpoint loading uses the visible checkpoint path as
  affected_artifact_path and leaves direct_failure_object_path null unless an exact directly
  accessed object is visible. A CUDA source callsite such as file=permute.cu,line=535 is
  neither field.
- Require tagged evidence for substantive failure-domain and retry-outlook conclusions.
  If the current log cannot distinguish them, use an unknown abstention and state the
  missing evidence in the rationale; an unknown abstention does not require a positive
  support citation.

Operational interpretation:
- Workload domain includes application code, model, data, configuration, and
  workload-selected framework or library behavior. Infrastructure domain includes
  hardware, platform, and external services when the current log supports that
  attribution. Uncertain ownership within the workload stack does not by itself make
  the domain unknown.
{_CLUSTER_EXECUTION_CONTEXT}
- Treat supplied execution facts as positional evidence. Prior progress proves
  runnability, not transience. Replay distance and failure position do not establish
  persistence. Interpret decision_evidence_view.operation_artifact_facts according to their
  declared identity strength: success on the exact file, object, or shard is relevant to
  that physical unit while data-region and observer differences remain material;
  success on another shard is only partial evidence; and success on a different
  checkpoint, dataset file, or artifact proves general pipeline runnability, not whether
  the artifact involved in the current failure is healthy or causal. Distributed fanout
  is one operation, not cross-attempt recurrence. Later aggregate progress in an
  interleaved log does not prove recovery of the same rank or component.

Return exactly these two current-attempt recovery claims. Each claim contains a value,
an evidence status, and confidence in that claim:
1. failure_domain: workload, infrastructure, or unknown.
2. retry_outlook_without_workload_change: cannot_recover, may_recover, or unknown. Assess
   this claim under the product restart transition defined above.

For each claim, status is established_by_current_log, supported_but_unconfirmed,
hypothesis_only, or unknown. established_by_current_log requires direct current-log
support for that specific claim. Confidence is a 1..99 calibration signal for that claim;
the client does not use it as a policy threshold.
For either recovery claim, value unknown must use status unknown, and status unknown must
use value unknown.

Use cannot_recover when the observed failure is determined by unchanged workload inputs
and no supported restart transition can address it. Exact root cause may remain
unconfirmed; do not use unknown solely for a theoretical transient unsupported by the
log. Use
may_recover when process recreation, cleanup or reinitialization, normal delay, equivalent
hardware
replacement, or mutable node-local or external-service state provides a supported recovery
mechanism. Otherwise use unknown. Durable remediation and best-practice workload changes
are outside this assessment: a workload may benefit from a later change while the next
unchanged retry may still recover. Ground each substantive concept in the rationale and
cited supporting evidence; explain unknown abstentions through the missing evidence rather
than manufacturing positive support.

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
"{L1_RESPONSE_CONTRACT.insufficient_summary}", preserve grounded visible surfaces in
observed_failures, and list at least one missing-evidence item. If one observation is
selected, assess recovery for that surface; otherwise use the canonical unknown recovery
claims and rationale "{L1_RESPONSE_CONTRACT.insufficient_rationale}". Related failure lines are grounded
diagnostic references, not additional policy-claim citations, and remain empty when no
primary was identified.

Return one compact JSON object matching the supplied schema. Include only the strongest
evidence and at most three related failures. Emit no fingerprint, data-position identity,
fault outcome, or action; the client derives those fields.
"""
