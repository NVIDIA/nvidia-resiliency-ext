"""Immutable review-panel model and tool profiles."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

PRIMARY_KEY_ENV = "LLM_API_KEY_FILE"
SECONDARY_KEY_ENV = "LLM_API_KEY_OLD_FILE"
DEFAULT_INFERENCE_ENDPOINT = "https://inference-api.nvidia.com/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 64_000
DEFAULT_TOOL_ADVERTISEMENT: Mapping[str, bool] = MappingProxyType(
    {
        "overview": True,
        "grep_log": True,
        "read_window": True,
        "get_evidence_objects": False,
    }
)
MODEL_TARGETS = ("qwen235b", "qwen397b", "nemotron", "gpt", "claude", "gemini")


@dataclass(frozen=True)
class ModelProfile:
    model: str
    credential_env: str
    endpoint: str = DEFAULT_INFERENCE_ENDPOINT
    context_window_tokens: int | None = None
    tool_loop_profile_id: str | None = None
    max_tool_rounds: int | None = None


MODEL_PROFILES: Mapping[str, ModelProfile] = MappingProxyType(
    {
        "qwen": ModelProfile("nvidia/qwen/qwen3.5-35b-a3b", PRIMARY_KEY_ENV),
        "qwen235b": ModelProfile(
            "nvidia/qwen/eccn-qwen-235b",
            SECONDARY_KEY_ENV,
            context_window_tokens=200_000,
            tool_loop_profile_id="qwen235b.experimental.one_tool_round.v1",
            max_tool_rounds=1,
        ),
        "qwen397b": ModelProfile(
            "nvidia/qwen/eccn-qwen3-5-397b-a17b",
            SECONDARY_KEY_ENV,
            context_window_tokens=262_144,
            tool_loop_profile_id="qwen397b.tools_supported.v1",
        ),
        "nemotron": ModelProfile(
            "nvidia/nvidia/eccn-nemotron-3-ultra",
            SECONDARY_KEY_ENV,
        ),
        "gpt": ModelProfile("us/azure/openai/eccn-gpt-5.5", SECONDARY_KEY_ENV),
        "claude": ModelProfile(
            "us/azure/anthropic/eccn-claude-sonnet-5",
            SECONDARY_KEY_ENV,
        ),
        "gemini": ModelProfile(
            "us/gcp/google/eccn-gemini-3.5-flash",
            SECONDARY_KEY_ENV,
        ),
    }
)


@dataclass(frozen=True)
class RunTarget:
    name: str
    enable_l1: bool = False
    model: str | None = None
    credential_env: str | None = None
    max_output_tokens: int | None = None
    context_window_tokens: int | None = None
    tool_loop_profile_id: str | None = None
    max_tool_rounds: int | None = None
    endpoint: str | None = None


def expand_targets(raw_targets: list[str]) -> list[RunTarget]:
    """Expand panel aliases and remove duplicates while retaining order."""

    expanded: list[str] = []
    for target in raw_targets:
        if target == "all":
            expanded.extend(("deterministic", *MODEL_TARGETS))
        elif target in {"models", "all-models"}:
            expanded.extend(MODEL_TARGETS)
        else:
            expanded.append(target)

    result: list[RunTarget] = []
    seen: set[str] = set()
    for target in expanded:
        if target in seen:
            continue
        seen.add(target)
        if target == "deterministic":
            result.append(RunTarget(name=target))
            continue
        if target == "configured":
            result.append(
                RunTarget(
                    name=target,
                    enable_l1=True,
                    credential_env=PRIMARY_KEY_ENV,
                )
            )
            continue
        profile = MODEL_PROFILES.get(target)
        if profile is None:
            raise SystemExit(f"unknown target: {target}")
        result.append(
            RunTarget(
                name=target,
                enable_l1=True,
                model=profile.model,
                credential_env=profile.credential_env,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                context_window_tokens=profile.context_window_tokens,
                tool_loop_profile_id=profile.tool_loop_profile_id,
                max_tool_rounds=profile.max_tool_rounds,
                endpoint=profile.endpoint,
            )
        )
    return result
