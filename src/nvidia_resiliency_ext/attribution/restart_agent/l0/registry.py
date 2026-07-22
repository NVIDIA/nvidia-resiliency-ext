# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supplemental signatures and diagnostic-line roles for L0 assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from ..identity import fingerprint_for
from ..models import PolicyClass, RecoveryBehavior, RegistryRole


@dataclass(frozen=True)
class SignatureRegistryRow:
    registry_id: str
    pattern: Pattern[str]
    policy_class: str
    role: str
    recovery_behavior: str = RecoveryBehavior.NONE.value


_DIAGNOSTIC_CONTEXT_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "cuda_async_reporting_advice",
        re.compile(
            r"CUDA kernel errors might be asynchronously reported"
            r"|stacktrace (?:below )?might be incorrect",
            re.I,
        ),
    ),
    (
        "cuda_launch_blocking_advice",
        re.compile(r"(?:consider|set|passing).*CUDA_LAUNCH_BLOCKING", re.I),
    ),
    (
        "cuda_dsa_compile_advice",
        re.compile(r"Compile with [`']?TORCH_USE_CUDA_DSA", re.I),
    ),
)

_CONDITIONAL_CAUSE_RE = re.compile(
    r"\b(?:might|may|could) be caused by\b"
    r"|\bit is possible that\b"
    r"|\bpossibly due to\b"
    r"|\bplease try\b",
    re.I,
)

_DISTRIBUTED_OPERATION_TIMEOUT_RE = re.compile(
    r"\b(?:watchdog\s+)?(?:caught\s+)?(?:collective\s+)?operation\s+timeout\b"
    r"|\boperation\b.*\btimed out\b",
    re.I,
)


MVP_SIGNATURES: tuple[SignatureRegistryRow, ...] = (
    SignatureRegistryRow(
        registry_id="gpu_hardware_fault",
        pattern=re.compile(
            r"\bXid\b|ECC.*(?:uncorrectable|DBE)|GPU.*(?:off bus|fallen off)"
            r"|NVLink.{0,24}(?:link down|uncorrectable(?: error)?|CRC error|recovery failed)"
            r"|NVLink\s+fatal\s+(?:error|failure)"
            r"|PCIe.{0,24}(?:AER(?: fatal)?|link down|fatal error|uncorrectable(?: error)?)"
            r"|thermal.{0,80}(?:shutdown|violation|fault)",
            re.I,
        ),
        policy_class=PolicyClass.NOT_USER_FAILURE.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="peer_gpu_memory_access_failure",
        pattern=re.compile(
            r"invalid access of peer GPU memory(?:\s+over\s+NVLink)?",
            re.I,
        ),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="infra_policy_event",
        pattern=re.compile(
            r"SLURM.*(?:preempt|node failure|NODE_FAIL)|\bpreempted\b|\bnode failure\b",
            re.I,
        ),
        policy_class=PolicyClass.NOT_USER_FAILURE.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="time_limit",
        pattern=re.compile(r"\btime limit\b|DUE TO TIME LIMIT|\bwall[- ]?time\b", re.I),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="user_cancelled",
        pattern=re.compile(r"\bscancel\b|\bcancelled by user\b|\buser .*cancel", re.I),
        policy_class=PolicyClass.USER_FAILURE.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="observed_exception",
        pattern=re.compile(
            r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):(?:\s|$)"
            r"|\bAssertion(?:Error)?\b.*\bfailed\b",
            re.I,
        ),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="user_config_error",
        pattern=re.compile(
            r"No such file or directory|invalid (?:argument|option|config)"
            r"|missing (?:config|key)|checkpoint .*mismatch|shape mismatch",
            re.I,
        ),
        policy_class=PolicyClass.USER_FAILURE.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="filesystem_permission_denied",
        pattern=re.compile(r"\bPermissionError\b|\bpermission denied\b|\bEACCES\b", re.I),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="cuda_oom",
        pattern=re.compile(r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.I),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="nan_or_inf",
        pattern=re.compile(
            r"(?:loss|grad(?:ient)?|activation)[=:\s]+(?:nan|[-+]?inf(?:inity)?)\b"
            r"|(?:nan|[-+]?inf(?:inity)?)\s+(?:loss|grad(?:ient)?|detected|encountered)"
            r"|non[- ]finite",
            re.I,
        ),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="bad_token_or_window",
        pattern=re.compile(
            r"bad token|bad sample|token window|skip(?:ping)? .*token|quarantine .*token",
            re.I,
        ),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
        recovery_behavior=RecoveryBehavior.RETRY_THEN_SKIP.value,
    ),
    SignatureRegistryRow(
        registry_id="framework_crash",
        pattern=re.compile(
            r"segmentation fault|\bsegfault\b|illegal instruction|core dumped", re.I
        ),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="linux_oom_kill_confirmation",
        pattern=re.compile(
            r"\bDetected\s+\d+\s+oom_kill events\b"
            r"|\b(?:Out of memory|Memory cgroup out of memory):\s+Killed process\b"
            r"|\bSome of the step tasks have been OOM Killed\b",
            re.I,
        ),
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.CAUSE_CONFIRMATION.value,
    ),
    SignatureRegistryRow(
        registry_id="observed_distributed_operation_timeout",
        pattern=_DISTRIBUTED_OPERATION_TIMEOUT_RE,
        policy_class=PolicyClass.AMBIGUOUS.value,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="nccl_cascade",
        pattern=re.compile(
            r"^(?!.*(?:watchdog\s+)?(?:caught\s+)?(?:collective\s+)?operation\s+timeout)"
            r".*NCCL.*(?:timeout|watchdog|abort|unhandled system error)",
            re.I,
        ),
        policy_class=PolicyClass.CASCADE.value,
        role=RegistryRole.CASCADE_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="cuda_previous_error_cascade",
        pattern=re.compile(
            r"operation failed due to a previous error during capture" r"|NCCL WARN Cuda failure",
            re.I,
        ),
        policy_class=PolicyClass.CASCADE.value,
        role=RegistryRole.CASCADE_CANDIDATE.value,
    ),
)

_REGISTRY_TRIGGER_TERMS = (
    "error",
    "exception",
    "failed",
    "assert",
    "xid",
    "ecc",
    "gpu",
    "nvlink",
    "pcie",
    "thermal",
    "slurm",
    "preempt",
    "node fail",
    "time limit",
    "wall-time",
    "walltime",
    "scancel",
    "cancelled by user",
    "no such file",
    "permission denied",
    "invalid ",
    "missing ",
    "checkpoint",
    "shape mismatch",
    "out of memory",
    "cublas_status_alloc_failed",
    "nan",
    "inf",
    "non-finite",
    "bad token",
    "bad sample",
    "token window",
    "skipping",
    "quarantine",
    "segmentation fault",
    "segfault",
    "illegal instruction",
    "core dumped",
    "oom_kill",
    "oom killed",
    "out of memory: killed process",
    "memory cgroup out of memory",
    "nccl",
    "watchdog",
)


def match_registry(
    line: str,
    *,
    diagnostic_checked: bool = False,
) -> list[SignatureRegistryRow]:
    lowered = line.lower()
    if not any(term in lowered for term in _REGISTRY_TRIGGER_TERMS):
        return []
    if not diagnostic_checked and diagnostic_context_kind(line) is not None:
        return []
    return [row for row in MVP_SIGNATURES if row.pattern.search(line)]


def diagnostic_context_kind(line: str) -> str | None:
    """Return the stable role for non-causal CUDA/PyTorch debugging advice."""

    for kind, pattern in _DIAGNOSTIC_CONTEXT_PATTERNS:
        if pattern.search(line):
            return kind
    return None


def diagnostic_uncertainty_kind(line: str) -> str | None:
    """Identify causal suggestions without hiding the observed error line."""

    if _CONDITIONAL_CAUSE_RE.search(line):
        return "conditional_cause_language"
    return None


def signature_for(row: SignatureRegistryRow, line: str) -> str:
    match = row.pattern.search(line)
    if match is None:
        return line.strip()
    return match.group(0).strip()


def fingerprint_components(row: SignatureRegistryRow, line: str) -> list[str]:
    lowered = line.lower()
    if row.registry_id == "cuda_oom":
        return ["allocation_failure"]
    if row.registry_id == "nccl_cascade":
        if "watchdog" in lowered:
            return ["watchdog_timeout"]
        return ["comm_abort"]
    if row.registry_id == "observed_distributed_operation_timeout":
        return ["collective_operation_timeout"]
    if row.registry_id == "cuda_previous_error_cascade":
        return ["previous_capture_error"]
    if row.registry_id == "time_limit":
        return ["time_limit"]
    if row.registry_id == "bad_token_or_window":
        return ["bad_token_or_window"]
    if row.registry_id == "nan_or_inf":
        if "grad norm" in lowered or "gradient" in lowered:
            return ["non_finite_gradient"]
        if "loss" in lowered:
            return ["non_finite_loss"]
        return ["non_finite_signal"]
    if row.registry_id == "gpu_hardware_fault":
        return ["hardware_event"]
    if row.registry_id == "peer_gpu_memory_access_failure":
        return ["peer_gpu_memory_access"]
    if row.registry_id == "infra_policy_event":
        return ["scheduler_or_node_event"]
    if row.registry_id == "user_cancelled":
        return ["user_cancelled"]
    if row.registry_id == "user_config_error":
        return [signature_for(row, line)]
    return [signature_for(row, line)]


def root_fingerprint(row: SignatureRegistryRow, line: str) -> str | None:
    if row.registry_id == "observed_exception":
        return None
    return fingerprint_for(row.registry_id, fingerprint_components(row, line))
