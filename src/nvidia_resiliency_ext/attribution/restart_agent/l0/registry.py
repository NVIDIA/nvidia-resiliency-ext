# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supplemental signatures and diagnostic-line roles for L0 assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from ..identity import fingerprint_for
from ..models import RecoveryBehavior, RegistryRole


@dataclass(frozen=True)
class SignatureRegistryRow:
    registry_id: str
    pattern: Pattern[str]
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
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="peer_gpu_memory_access_failure",
        pattern=re.compile(
            r"invalid access of peer GPU memory(?:\s+over\s+NVLink)?",
            re.I,
        ),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="infra_policy_event",
        pattern=re.compile(
            r"SLURM.*(?:preempt|node failure|NODE_FAIL)|\bpreempted\b|\bnode failure\b",
            re.I,
        ),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="time_limit",
        pattern=re.compile(
            r"\bDUE TO TIME LIMIT\b"
            r"|\b(?:time limit|wall[- ]?time(?: limit)?)\s+"
            r"(?:reached|exceeded|expired)\b"
            r"|\b(?:cancelled|canceled|terminated|killed)\b[^\n]{0,120}"
            r"\b(?:time limit|wall[- ]?time)\b",
            re.I,
        ),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="user_cancelled",
        pattern=re.compile(r"\bscancel\b|\bcancelled by user\b|\buser .*cancel", re.I),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="observed_exception",
        pattern=re.compile(
            r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):(?:\s|$)"
            r"|\bAssertion(?:Error)?\b.*\bfailed\b",
            re.I,
        ),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="configuration_validation_failure",
        pattern=re.compile(
            r"\binvalid (?:option|config(?:uration)?)\b" r"|\bmissing (?:config(?:uration)?|key)\b",
            re.I,
        ),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="argument_validation_failure",
        pattern=re.compile(r"\binvalid argument\b", re.I),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="artifact_or_path_not_found",
        pattern=re.compile(r"\bNo such file or directory\b", re.I),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="checkpoint_compatibility_mismatch",
        pattern=re.compile(r"\bcheckpoint\b.{0,160}\bmismatch\b", re.I),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="shape_mismatch",
        pattern=re.compile(r"\bshape mismatch\b", re.I),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="filesystem_permission_denied",
        pattern=re.compile(r"\bPermissionError\b|\bpermission denied\b|\bEACCES\b", re.I),
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="cuda_oom",
        pattern=re.compile(r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.I),
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
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="bad_token_or_window",
        pattern=re.compile(
            r"bad token|bad sample|token window|skip(?:ping)? .*token|quarantine .*token",
            re.I,
        ),
        role=RegistryRole.ROOT_CANDIDATE.value,
        recovery_behavior=RecoveryBehavior.RETRY_THEN_SKIP.value,
    ),
    SignatureRegistryRow(
        registry_id="framework_crash",
        pattern=re.compile(
            r"segmentation fault|\bsegfault\b|illegal instruction|core dumped", re.I
        ),
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
        role=RegistryRole.CAUSE_CONFIRMATION.value,
    ),
    SignatureRegistryRow(
        registry_id="observed_distributed_operation_timeout",
        pattern=_DISTRIBUTED_OPERATION_TIMEOUT_RE,
        role=RegistryRole.ROOT_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="nccl_cascade",
        pattern=re.compile(
            r"^(?!.*(?:watchdog\s+)?(?:caught\s+)?(?:collective\s+)?operation\s+timeout)"
            r".*NCCL.*(?:timeout|watchdog|abort|unhandled system error)",
            re.I,
        ),
        role=RegistryRole.CASCADE_CANDIDATE.value,
    ),
    SignatureRegistryRow(
        registry_id="cuda_previous_error_cascade",
        pattern=re.compile(
            r"operation failed due to a previous error during capture" r"|NCCL WARN Cuda failure",
            re.I,
        ),
        role=RegistryRole.CASCADE_CANDIDATE.value,
    ),
)

_REGISTRY_TRIGGER_TERMS = (
    "error",
    "exception",
    "nccl",
    "checkpoint",
    "failed",
    "assert",
    "out of memory",
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
    "shape mismatch",
    "cublas_status_alloc_failed",
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
    "watchdog",
)
_NONFINITE_TRIGGER_RE = re.compile(
    r"(?<![a-z0-9_])(?:nan|[-+]?inf(?:inity)?)(?![a-z0-9_])" r"|non[- ]?finite",
    re.I,
)

_ROW_TRIGGER_TERMS: dict[str, tuple[str, ...]] = {
    "gpu_hardware_fault": ("xid", "ecc", "gpu", "nvlink", "pcie", "thermal"),
    "peer_gpu_memory_access_failure": ("peer gpu",),
    "infra_policy_event": ("slurm", "preempt", "node fail", "node_fail"),
    "time_limit": ("time limit", "wall-time", "walltime", "wall time"),
    "user_cancelled": ("scancel", "cancelled by user", "user "),
    "observed_exception": ("error", "exception", "assertion"),
    "configuration_validation_failure": ("invalid ", "missing "),
    "argument_validation_failure": ("invalid argument",),
    "artifact_or_path_not_found": ("no such file or directory",),
    "checkpoint_compatibility_mismatch": ("checkpoint",),
    "shape_mismatch": ("shape mismatch",),
    "filesystem_permission_denied": ("permissionerror", "permission denied", "eacces"),
    "cuda_oom": ("cuda out of memory", "cublas_status_alloc_failed"),
    "bad_token_or_window": (
        "bad token",
        "bad sample",
        "token window",
        "skipping",
        "skip ",
        "quarantine",
    ),
    "framework_crash": (
        "segmentation fault",
        "segfault",
        "illegal instruction",
        "core dumped",
    ),
    "linux_oom_kill_confirmation": (
        "oom_kill",
        "oom killed",
        "out of memory",
        "memory cgroup",
    ),
    "observed_distributed_operation_timeout": ("timeout", "timed out"),
    "nccl_cascade": ("nccl",),
    "cuda_previous_error_cascade": ("previous error", "nccl warn cuda failure"),
}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if term in text:
            return True
    return False


def _row_triggered(
    registry_id: str,
    lowered: str,
    *,
    has_nonfinite_trigger: bool,
) -> bool:
    if registry_id == "nan_or_inf":
        return has_nonfinite_trigger
    return _contains_any(lowered, _ROW_TRIGGER_TERMS[registry_id])


def match_registry(
    line: str,
    *,
    diagnostic_checked: bool = False,
    lowered: str | None = None,
) -> list[SignatureRegistryRow]:
    lowered = lowered if lowered is not None else line.lower()
    has_nonfinite_trigger = _NONFINITE_TRIGGER_RE.search(lowered) is not None
    if not _contains_any(lowered, _REGISTRY_TRIGGER_TERMS) and not has_nonfinite_trigger:
        return []
    if not diagnostic_checked and diagnostic_context_kind(line) is not None:
        return []
    return [
        row
        for row in MVP_SIGNATURES
        if _row_triggered(
            row.registry_id,
            lowered,
            has_nonfinite_trigger=has_nonfinite_trigger,
        )
        and row.pattern.search(line)
    ]


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
    if row.registry_id == "configuration_validation_failure":
        if "invalid option" in lowered:
            return ["invalid_option"]
        if "invalid config" in lowered or "invalid configuration" in lowered:
            return ["invalid_configuration"]
        if "missing key" in lowered:
            return ["missing_key"]
        return ["missing_configuration"]
    if row.registry_id == "argument_validation_failure":
        return ["invalid_argument"]
    if row.registry_id == "artifact_or_path_not_found":
        return ["not_found"]
    if row.registry_id == "checkpoint_compatibility_mismatch":
        return ["compatibility_mismatch"]
    if row.registry_id == "shape_mismatch":
        return ["shape_mismatch"]
    return [signature_for(row, line)]


def root_fingerprint(row: SignatureRegistryRow, line: str) -> str | None:
    if row.registry_id == "observed_exception":
        return None
    return fingerprint_for(row.registry_id, fingerprint_components(row, line))
