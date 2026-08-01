# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint and lightweight locality parsing helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from .models import AffectedEntity, AffectedEntityKind

_EXCEPTION_TYPE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure|Fault))\s*:",
)
_PYTHON_FRAME_RE = re.compile(r'File "[^"]+", line \d+, in ([A-Za-z_][A-Za-z0-9_]*)')
_ROUTING_PREFIX_RE = re.compile(r"^\s*\d+:\s*(?:\[rank\d+\]:\s*)?")
_ISO_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\s*\[\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?\]\s*",
    re.I,
)
_NCCL_HOST_ROUTING_PREFIX_RE = re.compile(
    r"^\s*[A-Za-z0-9_.-]+:\d+:\d+\s+\[\d+\]\s*",
)
_CONDITIONAL_DIAGNOSTIC_RE = re.compile(
    r"\b(?:might|may|could) be caused by\b"
    r"|\bit is possible that\b"
    r"|\bpossibly due to\b"
    r"|\bplease try\b",
    re.I,
)
_FAILURE_ITERATION_RE = re.compile(r"\biteration\s*(?:=|:)?\s*(\d+)\b", re.I)

_VOLATILE_PATTERN_SOURCES = (
    # PyTorch c10d prefixes use severity plus month/day, for example E207 for
    # an error emitted on February 7. It is routing metadata, not mechanism.
    r"\b[ewif]\d{3,4}(?=\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
    r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?\b",
    r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b",
    r"\b0x[0-9a-f]+\b",
    r"\bpid[=:\s-]*\d+\b",
    r"\brank[=:\s_-]*\d+\b",
    r"\bworker[=:\s_-]*\d+\b",
    r"\btask[=:\s_-]*\d+\b",
    r"\breplica[=:\s_-]*\d+\b",
    r"\bcuda:\d+\b",
    r"\bgpu[=:\s_-]*\d+\b",
    r"\bdevice[=:\s_-]*\d+\b",
    r"\bnode[-_.a-z0-9]*\d+\b",
    r"\bnode[=:\s]+[a-z0-9_.-]+\b",
    r"\biteration[=:\s_-]*\d+\b",
    r"\bbucket[=:\s_#-]*\d+\b",
    r"\b\d+(?:\.\d+)?\s*(?:b|kb|kib|mb|mib|gb|gib|tb|tib)\b",
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds|min|minutes)\b",
    r"\bretry[=:\s_-]*\d+\b",
    r"\battempt[=:\s_-]*\d+\b",
    r"\bline\s+\d+\b",
)
_VOLATILE_PATTERN = re.compile(
    "|".join(f"(?:{source})" for source in _VOLATILE_PATTERN_SOURCES),
    re.I,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_UNDERSCORE_RUN_RE = re.compile(r"_+")
_STANDALONE_NUMBER_RE = re.compile(r"(?<![a-z0-9])\d+(?![a-z0-9])")


def path_hints(log_path: str, max_parts: int = 5) -> list[str]:
    path = Path(log_path)
    parts = [part for part in path.parts if part not in {"/", ""}]
    return parts[-max_parts:]


def normalize_token(text: str) -> str:
    normalized = _VOLATILE_PATTERN.sub(" ", text.lower())
    normalized = _NON_ALNUM_RE.sub("_", normalized)
    normalized = _UNDERSCORE_RUN_RE.sub("_", normalized).strip("_")
    return normalized


def normalized_pattern(text: str) -> str:
    normalized = normalize_token(text)
    normalized = _STANDALONE_NUMBER_RE.sub("n", normalized)
    normalized = _UNDERSCORE_RUN_RE.sub("_", normalized).strip("_")
    return normalized


def fingerprint_for(failure_class: str, components: Iterable[str]) -> str | None:
    tokens = [normalize_token(failure_class)]
    for component in components:
        token = normalize_token(component)
        if token:
            tokens.append(token)
    if len(tokens) == 1:
        return None
    return ":".join(tokens)


def _strip_routing_prefix(text: str) -> str:
    stripped = _ROUTING_PREFIX_RE.sub("", text)
    stripped = _ISO_TIMESTAMP_PREFIX_RE.sub("", stripped)
    return _NCCL_HOST_ROUTING_PREFIX_RE.sub("", stripped)


def canonical_observed_fingerprint(
    terminal_text: str,
    context_before: Iterable[str] = (),
) -> str | None:
    """Build a stable history key from observed text, not model vocabulary."""

    stripped = _strip_routing_prefix(terminal_text)
    exception_match = _EXCEPTION_TYPE_RE.search(stripped)
    exception_type = exception_match.group(1) if exception_match else "observed_failure"
    callsite = None
    for text in context_before:
        frame_match = _PYTHON_FRAME_RE.search(text)
        if frame_match:
            callsite = frame_match.group(1)
    components = [exception_type]
    if callsite:
        components.append(callsite)
    pattern = normalized_pattern(_observed_mechanism_text(stripped))
    if pattern:
        components.append(pattern)
    return fingerprint_for("observed", components)


def _observed_mechanism_text(text: str) -> str:
    match = _CONDITIONAL_DIAGNOSTIC_RE.search(text)
    if match is None:
        return text

    exception_match = _EXCEPTION_TYPE_RE.search(text)
    exception_end = exception_match.end() if exception_match else 0
    sentence_start = max(
        text.rfind(". ", exception_end, match.start()),
        text.rfind("! ", exception_end, match.start()),
        text.rfind("? ", exception_end, match.start()),
    )
    cut = sentence_start + 1 if sentence_start >= 0 else match.start()
    return text[:cut].rstrip(" .,:;-")


def grounded_artifact_path(value: object, *, texts: Iterable[str]) -> str | None:
    """Return a model-proposed artifact path only when source text contains it."""

    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    return candidate if any(candidate in text for text in texts) else None


def extract_rank(text: str) -> str | None:
    prefix_match = re.search(r"^\s*(\d+):\s+", text)
    if prefix_match:
        return prefix_match.group(1)
    match = re.search(r"(?:^|[\s\[])(?:rank|global_rank)[=:\s_-]*(\d+)", text, re.I)
    return match.group(1) if match else None


def extract_gpu(text: str) -> str | None:
    match = re.search(
        r"\bcuda:(\d+)\b|\bgpu[=:\s_-]*(\d+)\b|\bdevice[=:\s]+(\d+)\b",
        text,
        re.I,
    )
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def extract_node(text: str) -> str | None:
    match = re.search(r"\bnode[=:\s]+([a-z0-9_.-]+)", text, re.I)
    return match.group(1) if match else None


def extract_failure_iteration(text: str) -> int | None:
    """Return an explicit iteration attached to a failure observation."""

    match = _FAILURE_ITERATION_RE.search(_strip_routing_prefix(text))
    return int(match.group(1)) if match else None


def extract_data_position_fingerprint(text: str) -> str | None:
    identity = extract_data_position_identity(text)
    if identity is None:
        return None
    _kind, _separator, value = identity.partition(":")
    return fingerprint_for("data_position", [value])


def extract_data_position_identity(text: str) -> str | None:
    """Return an exact typed data position suitable for entity comparison."""

    patterns = (
        ("token", r"\btoken_id[=:\s]+([a-z0-9_.-]+)"),
        ("sample", r"\bsample_id[=:\s]+([a-z0-9_.-]+)"),
        ("window", r"\bwindow_id[=:\s]+([a-z0-9_.-]+)"),
        ("token", r"\btoken[=:]+([a-z0-9_.-]+)"),
        ("sample", r"\bsample[=:]+([a-z0-9_.-]+)"),
        ("window", r"\bwindow[=:]+([a-z0-9_.-]+)"),
        ("token", r"\btoken\s+(\d+)\b"),
        ("sample", r"\bsample\s+(\d+)\b"),
        ("window", r"\bwindow\s+(\d+)\b"),
    )
    for kind, pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return f"{kind}:{match.group(1)}"
    return None


def build_affected_entity(
    kind: AffectedEntityKind,
    identity: str,
    *,
    evidence_line: int | None = None,
) -> AffectedEntity:
    """Build the canonical exact-entity identity shared by L0 and L2."""

    digest = hashlib.sha256(f"{kind.value}\0{identity}".encode("utf-8")).hexdigest()
    return AffectedEntity(
        kind=kind,
        identity=identity,
        fingerprint=f"affected_entity:{kind.value}:{digest[:24]}",
        evidence_line=evidence_line,
    )
