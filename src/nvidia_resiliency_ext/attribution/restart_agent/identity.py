# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint and lightweight locality parsing helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

_EXCEPTION_TYPE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure|Fault))\s*:",
)
_PYTHON_FRAME_RE = re.compile(r'File "[^"]+", line \d+, in ([A-Za-z_][A-Za-z0-9_]*)')
_PYTHON_FRAME_DETAIL_RE = re.compile(r'File "([^"]+)", line \d+, in ([A-Za-z_][A-Za-z0-9_]*)')
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
_FAILURE_POSITION_RE = re.compile(r"\b(?:position|offset)\s*(?:=|:|at)?\s*(\d+)\b", re.I)
_FAILURE_ITERATION_RE = re.compile(r"\biteration\s*(?:=|:)?\s*(\d+)\b", re.I)
_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])(/[^\s'\"(),]+)")
_ARTIFACT_OPERATION_RE = re.compile(
    r"\b(?:checkpoint|load(?:ing|ed)?|read(?:ing)?|open(?:ing|ed)?|"
    r"writ(?:e|ing|ten)|sav(?:e|ing|ed)|restore|FileNotFoundError|No such file)\b",
    re.I,
)
_ARTIFACT_SUCCESS_RE = re.compile(
    r"\b(?:successfully|completed|scheduled|deleted)\b",
    re.I,
)
_CHECKPOINT_ITERATION_RE = re.compile(
    r"\b(?:at|from)\s+iteration\s+(?P<iteration>\d+)\b",
    re.I,
)
_OPERATION_SIGNATURE_RE = re.compile(
    r"\b(?P<name>SeqNum|OpType|NumelIn|NumelOut)=(?P<value>[A-Za-z0-9_.-]+)\b",
    re.I,
)
_IDENTITY_SCHEMA_VERSION = "restart_agent_failure_identity.experimental.v1"

_VOLATILE_PATTERNS = (
    # PyTorch c10d prefixes use severity plus month/day, for example E207 for
    # an error emitted on February 7. It is routing metadata, not mechanism.
    re.compile(r"\b[ewif]\d{3,4}(?=\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)", re.I),
    re.compile(
        r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?\b",
        re.I,
    ),
    re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    re.compile(r"\b0x[0-9a-f]+\b", re.I),
    re.compile(r"\bpid[=:\s-]*\d+\b", re.I),
    re.compile(r"\brank[=:\s_-]*\d+\b", re.I),
    re.compile(r"\bworker[=:\s_-]*\d+\b", re.I),
    re.compile(r"\btask[=:\s_-]*\d+\b", re.I),
    re.compile(r"\breplica[=:\s_-]*\d+\b", re.I),
    re.compile(r"\bcuda:\d+\b", re.I),
    re.compile(r"\bgpu[=:\s_-]*\d+\b", re.I),
    re.compile(r"\bdevice[=:\s_-]*\d+\b", re.I),
    re.compile(r"\bnode[-_.a-z0-9]*\d+\b", re.I),
    re.compile(r"\bnode[=:\s]+[a-z0-9_.-]+\b", re.I),
    re.compile(r"\biteration[=:\s_-]*\d+\b", re.I),
    re.compile(r"\bbucket[=:\s_#-]*\d+\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:b|kb|kib|mb|mib|gb|gib|tb|tib)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds|min|minutes)\b", re.I),
    re.compile(r"\bretry[=:\s_-]*\d+\b", re.I),
    re.compile(r"\battempt[=:\s_-]*\d+\b", re.I),
    re.compile(r"\bline\s+\d+\b", re.I),
)


def path_hints(log_path: str, max_parts: int = 5) -> list[str]:
    path = Path(log_path)
    parts = [part for part in path.parts if part not in {"/", ""}]
    return parts[-max_parts:]


def normalize_token(text: str) -> str:
    normalized = text.lower()
    for pattern in _VOLATILE_PATTERNS:
        normalized = pattern.sub(" ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalized_pattern(text: str) -> str:
    normalized = normalize_token(text)
    normalized = re.sub(r"(?<![a-z0-9])\d+(?![a-z0-9])", "n", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def fingerprint_for(fine_class: str, components: Iterable[str]) -> str | None:
    tokens = [normalize_token(fine_class)]
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


def build_experimental_failure_identity(
    terminal_text: str,
    context_before: Iterable[str] = (),
    *,
    model_identity: Mapping[str, Any] | None = None,
    model_context: Iterable[str] = (),
    observed_phase: str | None = None,
) -> dict[str, Any]:
    """Build inspectable family and concrete identities without applying policy."""

    build = _build_identity_sections(
        terminal_text,
        context_before,
        model_identity=model_identity,
        model_context=model_context,
        observed_phase=observed_phase,
    )
    return {
        "schema_version": _IDENTITY_SCHEMA_VERSION,
        "experimental": True,
        "policy_active": False,
        "family": build.family,
        "concrete": build.concrete,
        "client_concrete": build.client_concrete,
        "sources": build.sources,
    }


@dataclass(frozen=True)
class _IdentitySections:
    family: Mapping[str, Any]
    concrete: Mapping[str, Any]
    client_concrete: Mapping[str, Any]
    sources: Mapping[str, Any]


def _build_identity_sections(
    terminal_text: str,
    context_before: Iterable[str],
    *,
    model_identity: Mapping[str, Any] | None,
    model_context: Iterable[str],
    observed_phase: str | None,
) -> _IdentitySections:
    source_context = tuple(context_before)
    combined_model_context = (*source_context, *tuple(model_context))
    stripped = _strip_routing_prefix(terminal_text)
    exception_match = _EXCEPTION_TYPE_RE.search(stripped)
    exception_type = exception_match.group(1) if exception_match else None
    identity = model_identity if isinstance(model_identity, Mapping) else {}
    preferred_rank = extract_rank(terminal_text)
    position_match = _FAILURE_POSITION_RE.search(stripped)
    failure_position = position_match.group(1) if position_match else None
    failure_iteration = extract_failure_iteration(stripped)

    family_fields = _family_identity_fields(identity, exception_type)
    concrete_fields, observed_component = _model_concrete_identity_fields(
        terminal_text,
        combined_model_context,
        identity,
        preferred_rank=preferred_rank,
        failure_position=failure_position,
        failure_iteration=failure_iteration,
    )
    client_fields = _client_concrete_identity_fields(
        terminal_text,
        source_context,
        stripped,
        exception_type=exception_type,
        preferred_rank=preferred_rank,
        observed_phase=observed_phase,
        failure_position=failure_position,
        failure_iteration=failure_iteration,
    )
    family = _family_identity_section(family_fields)
    concrete = _model_concrete_identity_section(family_fields, concrete_fields)
    client_concrete = _client_concrete_identity_section(client_fields)
    sources = _identity_sources(
        family_fields,
        concrete_fields,
        client_fields,
        exception_type=exception_type,
        observed_component=observed_component,
    )
    return _IdentitySections(family, concrete, client_concrete, sources)


def _family_identity_fields(
    model_identity: Mapping[str, Any],
    exception_type: str | None,
) -> dict[str, Any]:
    return {
        "operation": _identity_name(model_identity.get("operation")),
        "mechanism": _identity_name(model_identity.get("mechanism")),
        "exception_type": _identity_name(exception_type),
    }


def _model_concrete_identity_fields(
    terminal_text: str,
    context: tuple[str, ...],
    model_identity: Mapping[str, Any],
    *,
    preferred_rank: str | None,
    failure_position: str | None,
    failure_iteration: int | None,
) -> tuple[dict[str, Any], str | None]:
    frames = _python_frames(context, preferred_rank=preferred_rank)
    stack_path = [frame[1] for frame in frames[-6:]]
    observed_component = _component_from_frames(frames)
    component = observed_component or _optional_identity_text(model_identity.get("component"))
    return (
        {
            "component": component,
            "callsite": stack_path[-1] if stack_path else None,
            "artifact_path": _grounded_artifact_path(
                model_identity.get("artifact_path"),
                texts=(*context, terminal_text),
            ),
            "failure_position": failure_position,
            "failure_iteration": failure_iteration,
            "stack_path": stack_path,
        },
        observed_component,
    )


def _client_concrete_identity_fields(
    terminal_text: str,
    source_context: tuple[str, ...],
    stripped_terminal_text: str,
    *,
    exception_type: str | None,
    preferred_rank: str | None,
    observed_phase: str | None,
    failure_position: str | None,
    failure_iteration: int | None,
) -> dict[str, Any]:
    client_context = _terminal_episode_context(source_context, preferred_rank=preferred_rank)
    client_frames = _python_frames(client_context, preferred_rank=preferred_rank)
    artifact_context_relevant = _artifact_context_relevant(
        stripped_terminal_text,
        observed_phase,
        client_context,
    )
    return {
        "exception_type": _identity_name(exception_type),
        "message_signature": normalized_pattern(_observed_mechanism_text(stripped_terminal_text))
        or None,
        "source_file": client_frames[-1][0] if client_frames else None,
        "callsite": client_frames[-1][1] if client_frames else None,
        "stack_path": [frame[1] for frame in client_frames[-6:]],
        "artifact_path": (
            _observed_artifact_path(
                (*client_context, terminal_text),
                source_paths={path for path, _ in client_frames},
            )
            if artifact_context_relevant
            else None
        ),
        "failure_position": failure_position,
        "failure_iteration": failure_iteration,
        "phase": _identity_name(observed_phase),
        "checkpoint_iteration": (
            _observed_checkpoint_iteration(client_context) if artifact_context_relevant else None
        ),
        "operation_signature": _observed_operation_signature(terminal_text),
    }


def _family_identity_section(fields: Mapping[str, Any]) -> dict[str, Any]:
    label = "|".join(value or "unknown" for value in fields.values())
    return {
        **fields,
        "label": label,
        "fingerprint": _identity_hash("family", fields),
        "complete": all(fields.values()),
    }


def _model_concrete_identity_section(
    family_fields: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    parts = ["|".join(value or "unknown" for value in family_fields.values())]
    for key in (
        "component",
        "callsite",
        "artifact_path",
        "failure_position",
        "failure_iteration",
    ):
        if fields.get(key):
            parts.append(f"{key}={fields[key]}")
    stack_path = fields.get("stack_path") or []
    if stack_path:
        parts.append("stack=" + ">".join(stack_path))
    return {
        **fields,
        "label": "|".join(parts),
        "fingerprint": _identity_hash(
            "concrete",
            {"family": family_fields, "concrete": fields},
        ),
        "complete": bool(
            all(family_fields.values())
            and fields.get("callsite")
            and (
                fields.get("artifact_path")
                or fields.get("failure_position")
                or fields.get("failure_iteration") is not None
            )
        ),
    }


def _client_concrete_identity_section(fields: Mapping[str, Any]) -> dict[str, Any]:
    parts = [
        str(fields.get("exception_type") or "unknown"),
        str(fields.get("message_signature") or "unknown"),
    ]
    for key in (
        "source_file",
        "callsite",
        "artifact_path",
        "failure_position",
        "failure_iteration",
        "phase",
        "checkpoint_iteration",
        "operation_signature",
    ):
        if fields.get(key):
            parts.append(f"{key}={fields[key]}")
    stack_path = fields.get("stack_path") or []
    if stack_path:
        parts.append("stack=" + ">".join(stack_path))
    return {
        **fields,
        "label": "|".join(parts),
        "fingerprint": _identity_hash("client_concrete", fields),
        "complete": bool(
            fields.get("exception_type")
            and fields.get("message_signature")
            and any(
                fields.get(key)
                for key in (
                    "callsite",
                    "artifact_path",
                    "failure_position",
                    "failure_iteration",
                    "operation_signature",
                )
            )
        ),
    }


def _identity_sources(
    family_fields: Mapping[str, Any],
    concrete_fields: Mapping[str, Any],
    client_fields: Mapping[str, Any],
    *,
    exception_type: str | None,
    observed_component: str | None,
) -> dict[str, Any]:
    component = concrete_fields.get("component")
    return {
        "operation": "l1_semantic" if family_fields.get("operation") else None,
        "mechanism": "l1_semantic" if family_fields.get("mechanism") else None,
        "exception_type": "observed_exception" if exception_type else None,
        "component": (
            "observed_stack" if observed_component else ("l1_semantic" if component else None)
        ),
        "callsite": "observed_stack" if concrete_fields.get("callsite") else None,
        "artifact_path": "grounded_l1_path" if concrete_fields.get("artifact_path") else None,
        "failure_position": (
            "observed_exception" if concrete_fields.get("failure_position") else None
        ),
        "failure_iteration": (
            "observed_exception" if concrete_fields.get("failure_iteration") is not None else None
        ),
        "phase": "l0_execution_context" if client_fields.get("phase") else None,
        "checkpoint_iteration": (
            "observed_checkpoint_context" if client_fields.get("checkpoint_iteration") else None
        ),
        "operation_signature": (
            "observed_exception" if client_fields.get("operation_signature") else None
        ),
        "stack_path": "observed_stack" if concrete_fields.get("stack_path") else None,
    }


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


def _python_frames(
    context: Iterable[str],
    *,
    preferred_rank: str | None = None,
) -> list[tuple[str, str]]:
    frames: list[tuple[str, str]] = []
    for text in context:
        line_rank = extract_rank(text)
        if preferred_rank is not None and line_rank is not None and line_rank != preferred_rank:
            continue
        match = _PYTHON_FRAME_DETAIL_RE.search(text)
        if match:
            frames.append((match.group(1), match.group(2)))
    return frames


def _terminal_episode_context(
    context: tuple[str, ...],
    *,
    preferred_rank: str | None,
) -> tuple[str, ...]:
    traceback_start = None
    for index, text in enumerate(context):
        if "Traceback (most recent call last)" not in text:
            continue
        line_rank = extract_rank(text)
        if preferred_rank is not None and line_rank not in {None, preferred_rank}:
            continue
        traceback_start = index
    return context[traceback_start:] if traceback_start is not None else context


def _component_from_frames(frames: Iterable[tuple[str, str]]) -> str | None:
    for path, _ in reversed(tuple(frames)):
        normalized = path.replace("\\", "/")
        relative = None
        for marker in ("/site-packages/", "/dist-packages/"):
            if marker in normalized:
                relative = normalized.split(marker, 1)[1]
                break
        if relative is None:
            continue
        parts = [part for part in relative.split("/")[:-1] if part]
        if parts:
            return ".".join(parts)
    return None


def _grounded_artifact_path(value: Any, *, texts: Iterable[str]) -> str | None:
    candidate = _optional_identity_text(value)
    if not candidate:
        return None
    return candidate if any(candidate in text for text in texts) else None


def _observed_artifact_path(
    texts: Iterable[str],
    *,
    source_paths: set[str],
) -> str | None:
    candidates: list[str] = []
    for text in texts:
        if not _ARTIFACT_OPERATION_RE.search(text):
            continue
        if _ARTIFACT_SUCCESS_RE.search(text):
            continue
        if _PYTHON_FRAME_DETAIL_RE.search(text) or re.search(r"\bframe\s+#\d+", text, re.I):
            continue
        for match in _ABSOLUTE_PATH_RE.finditer(text):
            candidate = match.group(1).rstrip(":;].")
            normalized = candidate.lower()
            if (
                re.search(r"\.py(?::\d+)?$", normalized)
                or "/site-packages/" in normalized
                or "/dist-packages/" in normalized
            ):
                continue
            if candidate not in source_paths:
                candidates.append(candidate)
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _observed_checkpoint_iteration(texts: Iterable[str]) -> int | None:
    for text in reversed(tuple(texts)):
        if not _ARTIFACT_OPERATION_RE.search(text) or "checkpoint" not in text.lower():
            continue
        match = _CHECKPOINT_ITERATION_RE.search(text)
        if match:
            return int(match.group("iteration"))
    return None


def _artifact_context_relevant(
    terminal_text: str,
    observed_phase: str | None,
    client_context: Iterable[str],
) -> bool:
    if _ARTIFACT_OPERATION_RE.search(terminal_text):
        return True
    phase = (observed_phase or "").lower()
    if "checkpoint" in phase or phase in {"load", "save", "restore"}:
        return True
    context = tuple(client_context)
    return any("Traceback (most recent call last)" in text for text in context) and any(
        _ARTIFACT_OPERATION_RE.search(text) for text in context
    )


def _observed_operation_signature(text: str) -> str | None:
    fields: dict[str, str] = {}
    for match in _OPERATION_SIGNATURE_RE.finditer(text):
        fields[match.group("name").lower()] = match.group("value").lower()
    if not fields:
        return None
    order = ("optype", "seqnum", "numelin", "numelout")
    return ",".join(f"{name}={fields[name]}" for name in order if name in fields)


def _optional_identity_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _identity_name(value: Any) -> str | None:
    text = _optional_identity_text(value)
    if text is None:
        return None
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return normalize_token(snake) or None


def _identity_hash(kind: str, value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{kind}:sha256:{hashlib.sha256(serialized).hexdigest()}"


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
    patterns = (
        r"\btoken(?:_id)?[=:\s]+([a-z0-9_.-]+)",
        r"\bsample(?:_id)?[=:\s]+([a-z0-9_.-]+)",
        r"\bwindow(?:_id)?[=:\s]+([a-z0-9_.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return fingerprint_for("data_position", [match.group(1)])
    return None
