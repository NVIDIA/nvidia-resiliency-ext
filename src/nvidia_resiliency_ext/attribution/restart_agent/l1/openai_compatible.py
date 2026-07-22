# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible L1 model evidence extraction and read-only tool loop."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..models import L1_EVIDENCE_SCHEMA_VERSION, L0ModelFacingView
from ..runtime import SYSTEM_CLOCK, SYSTEM_SLEEPER, Clock, Sleeper
from .contracts import L1EvidenceContext, L1EvidenceResult
from .prompts import SYSTEM_PROMPT
from .provider_profiles import NVIDIA_INFERENCE_HUB
from .response_contract import model_response_schema
from .tools import LogTools
from .validation import model_evidence_contract_errors

DEFAULT_BASE_URL = NVIDIA_INFERENCE_HUB.base_url
DEFAULT_MODEL = NVIDIA_INFERENCE_HUB.model
DEFAULT_OMIT_SAMPLING_PARAMS_MODEL_RE = r"gpt-5\.5"
THINKING_MODES = ("auto", "disable", "allow")
MAX_ERROR_DETAIL_CHARS = 4000
DEFAULT_LLM_MAX_RETRIES = 1
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 0.5
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 64_000
DEFAULT_CONTEXT_SAFETY_TOKENS = 4_096
ESTIMATED_CHARS_PER_INPUT_TOKEN = 3
CONTEXT_TOKEN_ESTIMATE_MULTIPLIER = 1.15
DEFAULT_ADVERTISED_TOOLS = ("overview", "grep_log", "read_window")
IMPLEMENTED_TOOL_NAMES = frozenset((*DEFAULT_ADVERTISED_TOOLS, "get_evidence_objects"))
KNOWN_MODEL_CONTEXT_WINDOWS = {
    "nvidia/qwen/eccn-qwen-235b": 200_000,
    "nvidia/qwen/eccn-qwen3-5-397b-a17b": 262_144,
}
PROVIDER_TIMING_HEADERS = {
    "x-litellm-timing-llm-api-ms": "downstream_llm_api_ms",
    "x-litellm-timing-pre-processing-ms": "proxy_pre_processing_ms",
    "x-litellm-timing-post-processing-ms": "proxy_post_processing_ms",
    "x-litellm-timing-message-copy-ms": "proxy_message_copy_ms",
}


@dataclass(frozen=True)
class LlmConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    api_key_file: str | None = None
    timeout_seconds: float = 120.0
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS
    context_window_tokens: int | None = None
    context_safety_tokens: int = DEFAULT_CONTEXT_SAFETY_TOKENS
    temperature: float | None = 0.2
    top_p: float | None = 0.7
    omit_sampling_params_for_model_regex: str = DEFAULT_OMIT_SAMPLING_PARAMS_MODEL_RE
    max_tool_rounds: int = 8
    max_retries: int = DEFAULT_LLM_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    tools_enabled: bool = True
    advertised_tools: tuple[str, ...] = DEFAULT_ADVERTISED_TOOLS
    thinking_mode: str = "auto"
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        unknown = set(self.advertised_tools).difference(IMPLEMENTED_TOOL_NAMES)
        if unknown:
            raise ValueError("unknown advertised tools: " + ", ".join(sorted(unknown)))

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key_file: str | None = None,
        timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_rounds: int | None = None,
        tools_enabled: bool | None = None,
        advertised_tools: Sequence[str] | None = None,
        thinking_mode: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> "LlmConfig":
        environment = os.environ if environ is None else environ
        return cls(
            base_url=base_url or environment.get("NVRX_LLM_BASE_URL", DEFAULT_BASE_URL),
            model=model or environment.get("NVRX_LLM_MODEL", DEFAULT_MODEL),
            api_key=environment.get("LLM_API_KEY") or environment.get("NVIDIA_API_KEY"),
            api_key_file=api_key_file or environment.get("LLM_API_KEY_FILE"),
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else _float_env(environment, "NVRX_LLM_TIMEOUT_SECONDS", 120.0)
            ),
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else _int_env(
                    environment,
                    "NVRX_LLM_MAX_OUTPUT_TOKENS",
                    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
                )
            ),
            context_window_tokens=(
                context_window_tokens
                if context_window_tokens is not None
                else _optional_int_env(environment, "NVRX_LLM_CONTEXT_WINDOW_TOKENS")
            ),
            context_safety_tokens=_int_env(
                environment,
                "NVRX_LLM_CONTEXT_SAFETY_TOKENS",
                DEFAULT_CONTEXT_SAFETY_TOKENS,
            ),
            temperature=(
                temperature
                if temperature is not None
                else _optional_float_env(environment, "NVRX_LLM_TEMPERATURE", 0.2)
            ),
            top_p=(
                top_p
                if top_p is not None
                else _optional_float_env(environment, "NVRX_LLM_TOP_P", 0.7)
            ),
            max_tool_rounds=(
                max_tool_rounds
                if max_tool_rounds is not None
                else _int_env(environment, "NVRX_LLM_MAX_TOOL_ROUNDS", 8)
            ),
            max_retries=_int_env(
                environment,
                "NVRX_LLM_MAX_RETRIES",
                DEFAULT_LLM_MAX_RETRIES,
            ),
            retry_backoff_seconds=_float_env(
                environment,
                "NVRX_LLM_RETRY_BACKOFF_SECONDS",
                DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
            ),
            tools_enabled=(
                tools_enabled
                if tools_enabled is not None
                else _bool_env(environment, "NVRX_LLM_TOOLS_ENABLED", True)
            ),
            advertised_tools=(
                tuple(advertised_tools)
                if advertised_tools is not None
                else _tool_names_env(
                    environment,
                    "NVRX_LLM_ADVERTISED_TOOLS",
                    DEFAULT_ADVERTISED_TOOLS,
                )
            ),
            thinking_mode=(thinking_mode or environment.get("NVRX_LLM_THINKING_MODE", "auto")),
            reasoning_effort=(reasoning_effort or environment.get("NVRX_LLM_REASONING_EFFORT")),
        )

    def request_temperature(self) -> float | None:
        if self.omit_sampling_params_for_model_regex and re.search(
            self.omit_sampling_params_for_model_regex,
            self.model,
            re.IGNORECASE,
        ):
            return None
        return self.temperature

    def request_top_p(self) -> float | None:
        if self.omit_sampling_params_for_model_regex and re.search(
            self.omit_sampling_params_for_model_regex,
            self.model,
            re.IGNORECASE,
        ):
            return None
        return self.top_p

    def disable_thinking(self) -> bool:
        if self.thinking_mode not in THINKING_MODES:
            raise ValueError(f"unknown thinking_mode: {self.thinking_mode}")
        if self.thinking_mode == "disable":
            return True
        if self.thinking_mode == "allow":
            return False
        return "qwen" in self.model.lower()

    def resolved_context_window_tokens(self) -> int | None:
        if self.context_window_tokens is not None:
            return self.context_window_tokens
        return KNOWN_MODEL_CONTEXT_WINDOWS.get(self.model)

    def resolved_advertised_tools(self) -> tuple[str, ...]:
        if not self.tools_enabled:
            return ()
        return tuple(dict.fromkeys(self.advertised_tools))

    def tools_active(self) -> bool:
        return bool(self.resolved_advertised_tools())


class LlmCallError(RuntimeError):
    """Provider call failed after a request was attempted."""

    def __init__(
        self,
        message: str,
        call_record: dict[str, Any],
        *,
        prior_call_records: Sequence[Mapping[str, Any]] = (),
        transcript_events: Sequence[Mapping[str, Any]] = (),
    ):
        super().__init__(message)
        self.call_record = call_record
        self.prior_call_records = tuple(dict(item) for item in prior_call_records)
        self.transcript_events = tuple(dict(item) for item in transcript_events)


class CredentialProvider(Protocol):
    """Resolve a provider credential at the infrastructure boundary."""

    def load(self, config: LlmConfig) -> str: ...


class ConfigCredentialProvider:
    """Resolve inline or file-backed credentials declared by ``LlmConfig``."""

    def load(self, config: LlmConfig) -> str:
        return _load_api_key(config)


class ToolLoopSession:
    """Execute one bounded model/tool conversation and contract-repair turn."""

    def __init__(
        self,
        config: LlmConfig,
        call_model_with_retries: Any,
        *,
        credential_provider: CredentialProvider,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._config = config
        self._call_model_with_retries = call_model_with_retries
        self._credential_provider = credential_provider
        self._clock = clock

    def run(
        self,
        context: L1EvidenceContext,
        *,
        deadline_monotonic: float | None,
    ) -> L1EvidenceResult:
        state = _ToolConversation.start(
            self._config,
            context,
            api_key=self._credential_provider.load(self._config),
        )
        loop_outcome = ToolRoundExecutor(
            self._config,
            self._call_model_with_retries,
            clock=self._clock,
        ).run(
            state,
            deadline_monotonic=deadline_monotonic,
        )
        if loop_outcome.result is not None:
            return loop_outcome.result
        return ContractRepairExecutor(
            self._config,
            self._call_model_with_retries,
            clock=self._clock,
        ).run(
            state,
            previous_error=loop_outcome.final_error,
            deadline_monotonic=deadline_monotonic,
        )


@dataclass
class _ToolConversation:
    api_key: str
    tools: Any
    messages: list[dict[str, Any]]
    transcript_events: list[dict[str, Any]]
    model_calls: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    unsupported_tool_requests: list[dict[str, Any]]
    raw_output: str | None = None
    last_message: dict[str, Any] | None = None

    @classmethod
    def start(
        cls,
        config: LlmConfig,
        context: L1EvidenceContext,
        *,
        api_key: str,
    ) -> "_ToolConversation":
        model_view = context.model_view
        payload = model_view.to_payload()
        initial_user_message = _initial_user_message(model_view)
        model_visible_payload = json.loads(initial_user_message)
        return cls(
            api_key=api_key,
            tools=context.tools,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial_user_message},
            ],
            transcript_events=[
                {"event_type": "tool_loop_profile", **_tool_loop_profile(config)},
                {
                    "event_type": "bundle_snapshot",
                    "schema_version": model_view.schema_version,
                    "bundle": payload["evidence_bundle"],
                    "model_view": payload,
                    "model_visible_payload": model_visible_payload,
                },
            ],
            model_calls=[],
            tool_calls=[],
            unsupported_tool_requests=[],
        )


@dataclass(frozen=True)
class _ToolLoopOutcome:
    result: L1EvidenceResult | None
    final_error: str | None = None


class ToolRoundExecutor:
    """Execute bounded model turns and dispatch advertised read-only tools."""

    def __init__(
        self,
        config: LlmConfig,
        call_model_with_retries: Any,
        *,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._config = config
        self._call_model_with_retries = call_model_with_retries
        self._clock = clock

    def run(
        self,
        state: _ToolConversation,
        *,
        deadline_monotonic: float | None,
    ) -> _ToolLoopOutcome:
        for model_turn in range(1, max(1, self._config.max_tool_rounds) + 1):
            provider_failure = self._request_model(
                state,
                model_turn=model_turn,
                deadline_monotonic=deadline_monotonic,
            )
            if provider_failure is not None:
                return _ToolLoopOutcome(provider_failure)
            call_record = state.model_calls[-1]
            message = state.last_message or {}
            parsed_tool_calls = _message_tool_calls(message)
            if parsed_tool_calls:
                failure = self._dispatch_tools(
                    state,
                    message,
                    parsed_tool_calls,
                    model_turn=model_turn,
                    deadline_monotonic=deadline_monotonic,
                )
                if failure is not None:
                    return _ToolLoopOutcome(failure)
                continue
            if call_record.get("finish_reason") == "length":
                state.messages.append({"role": "assistant", "content": state.raw_output})
                return _ToolLoopOutcome(
                    None,
                    "model output hit max_tokens before final evidence completed",
                )
            try:
                evidence = _parse_model_evidence(state.raw_output or "")
            except ValueError as exc:
                state.messages.append({"role": "assistant", "content": state.raw_output})
                return _ToolLoopOutcome(None, str(exc))
            return _ToolLoopOutcome(_successful_evidence_result(self._config, state, evidence))
        return _ToolLoopOutcome(None)

    def _request_model(
        self,
        state: _ToolConversation,
        *,
        model_turn: int,
        deadline_monotonic: float | None,
    ) -> L1EvidenceResult | None:
        try:
            response, call_record = self._call_model_with_retries(
                api_key=state.api_key,
                messages=state.messages,
                include_tools=self._config.tools_active(),
                model_turn=model_turn,
                model_calls=state.model_calls,
                transcript_events=state.transcript_events,
                deadline_monotonic=deadline_monotonic,
            )
        except LlmCallError as exc:
            return _provider_failure_from_state(self._config, state, exc)
        state.model_calls.append(call_record)
        message = _response_message(response)
        state.last_message = message
        state.raw_output = message.get("content") or ""
        parsed_tool_calls = _message_tool_calls(message)
        state.transcript_events.append(
            {
                "event_type": "model_response",
                "model_turn": model_turn,
                "model": self._config.model,
                "finish_reason": call_record.get("finish_reason"),
                "raw_model_output": state.raw_output,
                "tool_calls": parsed_tool_calls,
                "usage": call_record.get("usage"),
            }
        )
        return None

    def _dispatch_tools(
        self,
        state: _ToolConversation,
        message: Mapping[str, Any],
        tool_requests: list[dict[str, Any]],
        *,
        model_turn: int,
        deadline_monotonic: float | None,
    ) -> L1EvidenceResult | None:
        state.messages.append(_assistant_tool_call_message(message))
        for tool_call in tool_requests:
            if _deadline_expired(deadline_monotonic, clock=self._clock):
                return _deadline_failure_from_state(
                    self._config,
                    state,
                    model_turn=model_turn,
                    reason="before_tool_call",
                    clock=self._clock,
                )
            tool_result, tool_record, unsupported_record = _execute_tool_call(
                state.tools,
                tool_call,
                model_turn=model_turn,
                advertised_tools=self._config.resolved_advertised_tools(),
                clock=self._clock,
            )
            state.tool_calls.append(tool_record)
            if unsupported_record is not None:
                state.unsupported_tool_requests.append(unsupported_record)
            state.transcript_events.append(
                {
                    "event_type": "tool_result",
                    "model_turn": model_turn,
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "result": tool_result,
                    "record": tool_record,
                }
            )
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "content": json.dumps(tool_result, sort_keys=True),
                }
            )
            if _deadline_expired(deadline_monotonic, clock=self._clock):
                return _deadline_failure_from_state(
                    self._config,
                    state,
                    model_turn=model_turn,
                    reason="after_tool_call",
                    clock=self._clock,
                )
        return None


class ContractRepairExecutor:
    """Request one final schema-conforming response without tool advertisement."""

    def __init__(
        self,
        config: LlmConfig,
        call_model_with_retries: Any,
        *,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._config = config
        self._call_model_with_retries = call_model_with_retries
        self._clock = clock

    def run(
        self,
        state: _ToolConversation,
        *,
        previous_error: str | None,
        deadline_monotonic: float | None,
    ) -> L1EvidenceResult:
        contract_repair_requested = bool(previous_error and "max_tokens" not in previous_error)
        model_turn = len(state.model_calls) + 1
        if _deadline_expired(deadline_monotonic, clock=self._clock):
            return _deadline_failure_from_state(
                self._config,
                state,
                model_turn=model_turn,
                reason="before_final_evidence_call",
                clock=self._clock,
            )
        _append_contract_repair_prompt(
            state,
            previous_error=previous_error,
            contract_repair_requested=contract_repair_requested,
        )
        try:
            try:
                response, call_record = self._call_model_with_retries(
                    api_key=state.api_key,
                    messages=state.messages,
                    include_tools=False,
                    model_turn=model_turn,
                    model_calls=state.model_calls,
                    transcript_events=state.transcript_events,
                    deadline_monotonic=deadline_monotonic,
                )
            except LlmCallError as exc:
                return _provider_failure_from_state(self._config, state, exc)
            state.model_calls.append(call_record)
            message = _response_message(response)
            state.raw_output = message.get("content") or ""
            state.transcript_events.append(
                {
                    "event_type": "model_response",
                    "model_turn": len(state.model_calls),
                    "model": self._config.model,
                    "finish_reason": call_record.get("finish_reason"),
                    "raw_model_output": state.raw_output,
                    "usage": call_record.get("usage"),
                }
            )
            if call_record.get("finish_reason") == "length":
                raise ValueError("model output hit max_tokens during final evidence response")
            evidence = _parse_model_evidence(state.raw_output)
        except Exception as exc:
            return _malformed_evidence_result(
                self._config,
                state,
                exc,
                contract_repair_requested=contract_repair_requested,
            )
        return _successful_evidence_result(
            self._config,
            state,
            evidence,
            forced_final=True,
            contract_repair_requested=contract_repair_requested,
        )


def _append_contract_repair_prompt(
    state: _ToolConversation,
    *,
    previous_error: str | None,
    contract_repair_requested: bool,
) -> None:
    state.transcript_events.append(
        {
            "event_type": "budget_event",
            "reason": (
                "contract_repair_call"
                if contract_repair_requested
                else "forced_final_evidence_call"
            ),
            "previous_error": previous_error,
        }
    )
    repair_context = (
        f"The previous response could not be accepted: {previous_error}. " if previous_error else ""
    )
    state.messages.append(
        {
            "role": "user",
            "content": (
                f"{repair_context}Return one complete {L1_EVIDENCE_SCHEMA_VERSION} "
                "JSON object now. Copy schema_version exactly. Include the primary "
                "failure, root_cause_assessment, model_recovery_assessment, "
                "related_failures, and grounded evidence. Do not call "
                "tools. Do not include scores or decisions."
            ),
        }
    )


def _successful_evidence_result(
    config: LlmConfig,
    state: _ToolConversation,
    evidence: Mapping[str, Any],
    *,
    forced_final: bool = False,
    contract_repair_requested: bool = False,
) -> L1EvidenceResult:
    anomalies: dict[str, Any] = {
        "unsupported_tool_request_seen": bool(state.unsupported_tool_requests),
    }
    if forced_final:
        anomalies.update(
            {
                "forced_final_evidence_call": True,
                "contract_repair_requested": contract_repair_requested,
            }
        )
    return L1EvidenceResult(
        evidence=evidence,
        model=config.model,
        raw_model_output=state.raw_output,
        success=True,
        model_calls=tuple(state.model_calls),
        tool_calls=tuple(state.tool_calls),
        unsupported_tool_requests=tuple(state.unsupported_tool_requests),
        transcript_events=tuple(state.transcript_events),
        anomalies=anomalies,
    )


def _malformed_evidence_result(
    config: LlmConfig,
    state: _ToolConversation,
    error: Exception,
    *,
    contract_repair_requested: bool,
) -> L1EvidenceResult:
    return L1EvidenceResult(
        evidence=None,
        model=config.model,
        raw_model_output=state.raw_output,
        success=False,
        malformed=True,
        errors=(str(error),),
        model_calls=tuple(state.model_calls),
        tool_calls=tuple(state.tool_calls),
        unsupported_tool_requests=tuple(state.unsupported_tool_requests),
        transcript_events=tuple(state.transcript_events),
        anomalies={
            "forced_final_evidence_call": True,
            "contract_repair_requested": contract_repair_requested,
            "malformed_model_evidence": True,
            "model_output_truncated": bool(
                state.model_calls and state.model_calls[-1].get("finish_reason") == "length"
            ),
            "unsupported_tool_request_seen": bool(state.unsupported_tool_requests),
        },
    )


def _provider_failure_from_state(
    config: LlmConfig,
    state: _ToolConversation,
    error: LlmCallError,
) -> L1EvidenceResult:
    return _provider_failure_result(
        config=config,
        exc=error,
        raw_output=state.raw_output,
        model_calls=state.model_calls,
        tool_calls=state.tool_calls,
        unsupported_tool_requests=state.unsupported_tool_requests,
        transcript_events=state.transcript_events,
    )


def _deadline_failure_from_state(
    config: LlmConfig,
    state: _ToolConversation,
    *,
    model_turn: int,
    reason: str,
    clock: Clock = SYSTEM_CLOCK,
) -> L1EvidenceResult:
    return _deadline_failure_result(
        config=config,
        raw_output=state.raw_output,
        model_calls=state.model_calls,
        tool_calls=state.tool_calls,
        unsupported_tool_requests=state.unsupported_tool_requests,
        transcript_events=state.transcript_events,
        model_turn=model_turn,
        reason=reason,
        clock=clock,
    )


class RetryingChatTransport:
    """Apply deadline-aware retry policy around one provider request callable."""

    def __init__(
        self,
        config: LlmConfig,
        request_call: Any,
        *,
        clock: Clock = SYSTEM_CLOCK,
        sleeper: Sleeper = SYSTEM_SLEEPER,
    ) -> None:
        self._config = config
        self._request_call = request_call
        self._clock = clock
        self._sleeper = sleeper

    def call(
        self,
        *,
        api_key: str,
        messages: list[dict[str, Any]],
        include_tools: bool,
        model_turn: int,
        deadline_monotonic: float | None,
    ) -> "RetryOutcome":
        max_retries = max(0, self._config.max_retries)
        max_attempts = max_retries + 1
        prior_call_records: list[dict[str, Any]] = []
        transcript_events: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            if _deadline_expired(deadline_monotonic, clock=self._clock):
                deadline_error = _deadline_call_error(
                    config=self._config,
                    model_turn=model_turn,
                    attempt=attempt,
                    max_retries=max_retries,
                    reason="before_model_request",
                    clock=self._clock,
                )
                deadline_error.prior_call_records = tuple(prior_call_records)
                deadline_error.transcript_events = tuple(transcript_events)
                raise deadline_error
            transcript_events.append(
                _model_request_event(
                    config=self._config,
                    messages=messages,
                    include_tools=include_tools,
                    model_turn=model_turn,
                    attempt=attempt,
                    max_retries=max_retries,
                    deadline_monotonic=deadline_monotonic,
                    clock=self._clock,
                )
            )
            try:
                response, call_record = self._request_call(
                    api_key=api_key,
                    messages=messages,
                    include_tools=include_tools,
                    model_turn=model_turn,
                    attempt=attempt,
                    max_retries=max_retries,
                    deadline_monotonic=deadline_monotonic,
                )
                return RetryOutcome(
                    response=response,
                    call_record=call_record,
                    prior_call_records=tuple(prior_call_records),
                    transcript_events=tuple(transcript_events),
                )
            except LlmCallError as exc:
                retryable = bool(exc.call_record.get("retryable"))
                if not retryable or attempt >= max_attempts:
                    exc.prior_call_records = tuple(prior_call_records)
                    exc.transcript_events = tuple(transcript_events)
                    raise
                backoff_s = max(0.0, self._config.retry_backoff_seconds)
                remaining_s = _remaining_deadline_seconds(
                    deadline_monotonic,
                    clock=self._clock,
                )
                if remaining_s is not None and remaining_s <= backoff_s:
                    exc.call_record["retry_scheduled"] = False
                    exc.call_record["retry_blocked_by_deadline"] = True
                    prior_call_records.append(exc.call_record)
                    transcript_events.append(
                        {
                            "event_type": "provider_error",
                            "model_turn": model_turn,
                            "model": self._config.model,
                            "attempt": attempt,
                            "error_type": exc.call_record.get("error_type"),
                            "http_status": exc.call_record.get("http_status"),
                            "timeout": exc.call_record.get("timeout"),
                            "retryable": retryable,
                            "retry_scheduled": False,
                            "retry_blocked_by_deadline": True,
                            "error": exc.call_record.get("error"),
                            "latency_s": exc.call_record.get("latency_s"),
                        }
                    )
                    deadline_error = _deadline_call_error(
                        config=self._config,
                        model_turn=model_turn,
                        attempt=attempt,
                        max_retries=max_retries,
                        reason="before_retry_backoff",
                        clock=self._clock,
                    )
                    deadline_error.prior_call_records = tuple(prior_call_records)
                    deadline_error.transcript_events = tuple(transcript_events)
                    raise deadline_error from exc
                exc.call_record["retry_scheduled"] = True
                exc.call_record["retry_after_s"] = backoff_s
                prior_call_records.append(exc.call_record)
                transcript_events.append(
                    {
                        "event_type": "provider_error",
                        "model_turn": model_turn,
                        "model": self._config.model,
                        "attempt": attempt,
                        "error_type": exc.call_record.get("error_type"),
                        "http_status": exc.call_record.get("http_status"),
                        "timeout": exc.call_record.get("timeout"),
                        "retryable": retryable,
                        "retry_scheduled": True,
                        "retry_after_s": backoff_s,
                        "error": exc.call_record.get("error"),
                        "latency_s": exc.call_record.get("latency_s"),
                    }
                )
                if backoff_s:
                    self._sleeper.sleep(backoff_s)
        raise AssertionError("unreachable retry loop exit")


@dataclass(frozen=True)
class RetryOutcome:
    """Provider response plus retry attempts and their trace events."""

    response: Mapping[str, Any]
    call_record: Mapping[str, Any]
    prior_call_records: tuple[Mapping[str, Any], ...]
    transcript_events: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _PreparedProviderRequest:
    request: urllib.request.Request
    model_turn: int
    include_tools: bool
    started: float
    attempt: int
    max_retries: int | None
    context_budget: Mapping[str, Any]
    effective_timeout_seconds: float
    remaining_before_call_s: float | None


def _prepare_provider_request(
    config: LlmConfig,
    *,
    api_key: str,
    messages: list[dict[str, Any]],
    include_tools: bool,
    model_turn: int,
    attempt: int,
    max_retries: int | None,
    deadline_monotonic: float | None,
    clock: Clock = SYSTEM_CLOCK,
) -> _PreparedProviderRequest:
    started = clock.monotonic()
    remaining_before_call_s = _remaining_deadline_seconds(
        deadline_monotonic,
        clock=clock,
    )
    if remaining_before_call_s is not None and remaining_before_call_s <= 0:
        raise _deadline_call_error(
            config=config,
            model_turn=model_turn,
            attempt=attempt,
            max_retries=config.max_retries if max_retries is None else max_retries,
            reason="before_http_request",
            clock=clock,
        )
    effective_timeout_seconds = config.timeout_seconds
    if remaining_before_call_s is not None:
        effective_timeout_seconds = min(effective_timeout_seconds, remaining_before_call_s)
    context_budget = _request_context_budget(config, messages, include_tools=include_tools)
    body = _request_body(config, messages, include_tools=include_tools)
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=_encoded_request_body(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return _PreparedProviderRequest(
        request=request,
        model_turn=model_turn,
        include_tools=include_tools,
        started=started,
        attempt=attempt,
        max_retries=max_retries,
        context_budget=context_budget,
        effective_timeout_seconds=effective_timeout_seconds,
        remaining_before_call_s=remaining_before_call_s,
    )


class HttpClient(Protocol):
    """Open a prepared HTTP request."""

    def open(self, request: urllib.request.Request, *, timeout: float) -> Any: ...


class UrllibHttpClient:
    def open(self, request: urllib.request.Request, *, timeout: float) -> Any:
        return urllib.request.urlopen(request, timeout=timeout)


class OpenAICompatibleTransport:
    """Issue one OpenAI-compatible HTTP request and classify its outcome."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        http_client: HttpClient | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._config = config
        self._http_client = http_client or UrllibHttpClient()
        self._clock = clock

    def call(
        self,
        *,
        api_key: str,
        messages: list[dict[str, Any]],
        include_tools: bool,
        model_turn: int,
        attempt: int = 1,
        max_retries: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prepared = _prepare_provider_request(
            self._config,
            api_key=api_key,
            messages=messages,
            include_tools=include_tools,
            model_turn=model_turn,
            attempt=attempt,
            max_retries=max_retries,
            deadline_monotonic=deadline_monotonic,
            clock=self._clock,
        )
        try:
            with self._http_client.open(
                prepared.request,
                timeout=prepared.effective_timeout_seconds,
            ) as response:
                provider_reported_timing = _provider_reported_timing(
                    getattr(response, "headers", None)
                )
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise self._classified_http_error(exc, prepared) from exc
        except urllib.error.URLError as exc:
            raise self._classified_url_error(exc, prepared) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise self._classified_timeout(exc, prepared) from exc
        except json.JSONDecodeError as exc:
            raise self._classified_decode_error(exc, prepared) from exc

        if _deadline_expired(deadline_monotonic, clock=self._clock):
            raise _deadline_call_error(
                config=self._config,
                model_turn=model_turn,
                attempt=attempt,
                max_retries=self._config.max_retries if max_retries is None else max_retries,
                reason="after_http_response",
                started=prepared.started,
                effective_timeout_seconds=prepared.effective_timeout_seconds,
                remaining_before_call_s=prepared.remaining_before_call_s,
                clock=self._clock,
            )

        choice = (payload.get("choices") or [{}])[0]
        call_record = self._call_record(
            model_turn=model_turn,
            include_tools=include_tools,
            started=prepared.started,
            attempt=attempt,
            max_retries=max_retries,
            success=True,
            finish_reason=choice.get("finish_reason"),
            usage=payload.get("usage"),
            provider_reported_timing=provider_reported_timing,
            context_budget=prepared.context_budget,
            effective_timeout_seconds=prepared.effective_timeout_seconds,
            remaining_before_call_s=prepared.remaining_before_call_s,
        )
        return payload, call_record

    def _classified_http_error(
        self,
        error: urllib.error.HTTPError,
        prepared: "_PreparedProviderRequest",
    ) -> LlmCallError:
        detail = error.read().decode("utf-8", errors="replace")
        context_window_exceeded = _is_context_window_exceeded_error(detail)
        record = self._failure_record(
            prepared,
            error_type=("context_window_exceeded" if context_window_exceeded else "http_error"),
            error=f"HTTP {error.code}",
            http_status=error.code,
            response_body=_truncate_error_detail(detail),
            retryable=_is_retryable_http_status(error.code),
            timeout=_is_http_timeout_status(error.code),
        )
        return LlmCallError(f"HTTP {error.code}: {detail}", record)

    def _classified_url_error(
        self,
        error: urllib.error.URLError,
        prepared: "_PreparedProviderRequest",
    ) -> LlmCallError:
        timeout = isinstance(error.reason, (TimeoutError, socket.timeout))
        record = self._failure_record(
            prepared,
            error_type="timeout" if timeout else "url_error",
            error=str(error),
            retryable=True,
            timeout=timeout,
        )
        return LlmCallError(f"failed to reach LLM endpoint: {error}", record)

    def _classified_timeout(
        self,
        error: Exception,
        prepared: "_PreparedProviderRequest",
    ) -> LlmCallError:
        record = self._failure_record(
            prepared,
            error_type="timeout",
            error=str(error),
            retryable=True,
            timeout=True,
        )
        return LlmCallError(f"LLM request timed out: {error}", record)

    def _classified_decode_error(
        self,
        error: json.JSONDecodeError,
        prepared: "_PreparedProviderRequest",
    ) -> LlmCallError:
        record = self._failure_record(
            prepared,
            error_type="provider_response_decode_error",
            error=str(error),
            retryable=True,
        )
        return LlmCallError(f"LLM response was not JSON: {error}", record)

    def _failure_record(
        self,
        prepared: "_PreparedProviderRequest",
        **failure: Any,
    ) -> dict[str, Any]:
        return self._call_record(
            model_turn=prepared.model_turn,
            include_tools=prepared.include_tools,
            started=prepared.started,
            attempt=prepared.attempt,
            max_retries=prepared.max_retries,
            success=False,
            context_budget=prepared.context_budget,
            effective_timeout_seconds=prepared.effective_timeout_seconds,
            remaining_before_call_s=prepared.remaining_before_call_s,
            **failure,
        )

    def _call_record(
        self,
        *,
        model_turn: int,
        include_tools: bool,
        started: float,
        attempt: int,
        max_retries: int | None,
        success: bool,
        finish_reason: str | None = None,
        usage: Mapping[str, Any] | None = None,
        provider_reported_timing: Mapping[str, Any] | None = None,
        error_type: str | None = None,
        error: str | None = None,
        http_status: int | None = None,
        response_body: str | None = None,
        retryable: bool = False,
        timeout: bool = False,
        context_budget: Mapping[str, Any] | None = None,
        effective_timeout_seconds: float | None = None,
        remaining_before_call_s: float | None = None,
    ) -> dict[str, Any]:
        record = {
            "layer": "L1",
            "model": self._config.model,
            "model_turn": model_turn,
            "attempt": attempt,
            "max_retries": self._config.max_retries if max_retries is None else max_retries,
            "success": success,
            "latency_s": round(self._clock.monotonic() - started, 3),
            "finish_reason": finish_reason,
            "usage": usage,
            "tools_advertised": include_tools,
            "reasoning_mode_requested": self._config.thinking_mode,
            "reasoning_mode_resolved": (
                "disabled" if self._config.disable_thinking() else "provider_default"
            ),
            "provider_options": (
                {"enable_thinking": False} if self._config.disable_thinking() else {}
            ),
            "context_budget": dict(context_budget or {}),
            "configured_request_timeout_seconds": self._config.timeout_seconds,
            "effective_request_timeout_seconds": effective_timeout_seconds,
            "remaining_analysis_budget_before_call_s": remaining_before_call_s,
        }
        if provider_reported_timing:
            record["provider_reported_timing"] = dict(provider_reported_timing)
        if not success:
            record.update(
                {
                    "error_type": error_type,
                    "error": error,
                    "http_status": http_status,
                    "response_body": response_body,
                    "retryable": retryable,
                    "retry_scheduled": False,
                    "timeout": timeout,
                }
            )
        return record


class ChatTransport(Protocol):
    """One provider request transport used by the L1 tool loop."""

    def call(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]: ...


RetryTransportFactory = Callable[[LlmConfig, Any], RetryingChatTransport]


class LlmEvidenceExtractor:
    """Bounded OpenAI-compatible tool-loop client for L1 evidence extraction."""

    def __init__(
        self,
        config: LlmConfig | None = None,
        *,
        transport: ChatTransport | None = None,
        retry_transport_factory: RetryTransportFactory | None = None,
        credential_provider: CredentialProvider | None = None,
        clock: Clock = SYSTEM_CLOCK,
        sleeper: Sleeper = SYSTEM_SLEEPER,
    ):
        self._config = config or LlmConfig.from_env()
        self._clock = clock
        self._sleeper = sleeper
        self._transport = transport or OpenAICompatibleTransport(
            self._config,
            clock=clock,
        )
        self._retry_transport_factory = retry_transport_factory
        self._credential_provider = credential_provider or ConfigCredentialProvider()

    def extract_evidence(
        self,
        context: L1EvidenceContext,
        *,
        deadline_monotonic: float | None = None,
    ) -> L1EvidenceResult:
        try:
            return self._extract_evidence(
                context,
                deadline_monotonic=deadline_monotonic,
            )
        except Exception as exc:  # pragma: no cover - defensive provider boundary
            return L1EvidenceResult(
                evidence=None,
                model=self._config.model,
                success=False,
                errors=(str(exc),),
                anomalies={"provider_error": True},
            )

    def _extract_evidence(
        self,
        context: L1EvidenceContext,
        *,
        deadline_monotonic: float | None,
    ) -> L1EvidenceResult:
        return ToolLoopSession(
            self._config,
            self._call_model_with_retries,
            credential_provider=self._credential_provider,
            clock=self._clock,
        ).run(
            context,
            deadline_monotonic=deadline_monotonic,
        )

    def _call_model_with_retries(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_calls = kwargs.pop("model_calls")
        transcript_events = kwargs.pop("transcript_events")
        try:
            retry_transport = (
                self._retry_transport_factory(self._config, self._call_model)
                if self._retry_transport_factory is not None
                else RetryingChatTransport(
                    self._config,
                    self._call_model,
                    clock=self._clock,
                    sleeper=self._sleeper,
                )
            )
            outcome = retry_transport.call(**kwargs)
        except LlmCallError as exc:
            model_calls.extend(dict(item) for item in exc.prior_call_records)
            transcript_events.extend(dict(item) for item in exc.transcript_events)
            raise
        model_calls.extend(dict(item) for item in outcome.prior_call_records)
        transcript_events.extend(dict(item) for item in outcome.transcript_events)
        return dict(outcome.response), dict(outcome.call_record)

    def _call_model(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._transport.call(**kwargs)


def _provider_reported_timing(headers: Any) -> dict[str, Any]:
    """Extract optional proxy timing headers without inferring missing spans."""

    if headers is None:
        return {}
    normalized_headers: dict[str, Any] = {}
    try:
        normalized_headers = {str(key).lower(): value for key, value in headers.items()}
    except (AttributeError, TypeError, ValueError):
        return {}

    components: dict[str, float] = {}
    for header_name, field_name in PROVIDER_TIMING_HEADERS.items():
        raw_value = normalized_headers.get(header_name)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value < 0:
            continue
        components[field_name] = value
    if not components:
        return {}
    return {"source": "response_headers", **components}


def _request_body(
    config: LlmConfig,
    messages: list[dict[str, Any]],
    *,
    include_tools: bool,
) -> dict[str, Any]:
    budget = _request_context_budget(config, messages, include_tools=include_tools)
    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "max_tokens": budget["effective_max_output_tokens"],
    }
    if config.request_temperature() is not None:
        body["temperature"] = config.request_temperature()
    if config.request_top_p() is not None:
        body["top_p"] = config.request_top_p()
    if config.reasoning_effort:
        body["reasoning_effort"] = config.reasoning_effort
    if config.disable_thinking():
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if include_tools and config.advertised_tools:
        body["tools"] = _tool_schemas(config)
        body["tool_choice"] = "auto"
    return body


def _request_context_budget(
    config: LlmConfig,
    messages: list[dict[str, Any]],
    *,
    include_tools: bool,
) -> dict[str, Any]:
    context_window = config.resolved_context_window_tokens()
    estimate_payload: dict[str, Any] = {"messages": messages}
    if include_tools:
        estimate_payload["tools"] = _tool_schemas(config)
    input_chars = len(
        json.dumps(
            estimate_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    raw_estimated_input_tokens = math.ceil(input_chars / ESTIMATED_CHARS_PER_INPUT_TOKEN)
    estimated_input_tokens = math.ceil(
        raw_estimated_input_tokens * CONTEXT_TOKEN_ESTIMATE_MULTIPLIER
    )
    effective_max_output_tokens = config.max_output_tokens
    if context_window is not None:
        available = max(
            1,
            context_window - estimated_input_tokens - max(0, config.context_safety_tokens),
        )
        effective_max_output_tokens = min(config.max_output_tokens, available)
    return {
        "context_window_tokens": context_window,
        "raw_estimated_input_tokens": raw_estimated_input_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "estimation_chars_per_token": ESTIMATED_CHARS_PER_INPUT_TOKEN,
        "estimation_multiplier": CONTEXT_TOKEN_ESTIMATE_MULTIPLIER,
        "input_chars": input_chars,
        "configured_max_output_tokens": config.max_output_tokens,
        "effective_max_output_tokens": effective_max_output_tokens,
        "safety_tokens": max(0, config.context_safety_tokens),
        "adjusted": effective_max_output_tokens != config.max_output_tokens,
    }


def _model_request_event(
    *,
    config: LlmConfig,
    messages: list[dict[str, Any]],
    include_tools: bool,
    model_turn: int,
    attempt: int,
    max_retries: int,
    deadline_monotonic: float | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> dict[str, Any]:
    body = _request_body(config, messages, include_tools=include_tools)
    context_budget = _request_context_budget(config, messages, include_tools=include_tools)
    encoded = _encoded_request_body(body)
    request_body = json.loads(encoded.decode("utf-8"))
    return {
        "event_type": "model_request",
        "model_turn": model_turn,
        "attempt": attempt,
        "max_retries": max_retries,
        "model": config.model,
        "provider_route": config.base_url,
        "endpoint": config.base_url.rstrip("/") + "/chat/completions",
        "context_budget": context_budget,
        "timeout_seconds": config.timeout_seconds,
        "remaining_analysis_budget_s": _remaining_deadline_seconds(
            deadline_monotonic,
            clock=clock,
        ),
        "request_body": request_body,
        "advertised_tool_schemas": request_body.get("tools", []),
        "payload_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "payload_bytes": len(encoded),
        "redaction_status": "api_credentials_excluded",
        "truncated": False,
        "decision_budget": None,
    }


def _encoded_request_body(body: Mapping[str, Any]) -> bytes:
    return json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _provider_failure_result(
    *,
    config: LlmConfig,
    exc: LlmCallError,
    raw_output: str | None,
    model_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    unsupported_tool_requests: list[dict[str, Any]],
    transcript_events: list[dict[str, Any]],
) -> L1EvidenceResult:
    model_calls.append(exc.call_record)
    transcript_events.append(
        {
            "event_type": "provider_error",
            "model_turn": exc.call_record.get("model_turn"),
            "model": config.model,
            "error_type": exc.call_record.get("error_type"),
            "http_status": exc.call_record.get("http_status"),
            "timeout": exc.call_record.get("timeout"),
            "retryable": exc.call_record.get("retryable"),
            "error": exc.call_record.get("error"),
            "latency_s": exc.call_record.get("latency_s"),
        }
    )
    error_type = str(exc.call_record.get("error_type") or "provider_error")
    return L1EvidenceResult(
        evidence=None,
        model=config.model,
        raw_model_output=raw_output,
        success=False,
        malformed=False,
        errors=(str(exc),),
        model_calls=tuple(model_calls),
        tool_calls=tuple(tool_calls),
        unsupported_tool_requests=tuple(unsupported_tool_requests),
        transcript_events=tuple(transcript_events),
        anomalies={
            "provider_error": True,
            "provider_error_type": error_type,
            "provider_timeout": bool(exc.call_record.get("timeout")),
            "deadline_exceeded": error_type == "analysis_deadline_exceeded",
            "context_window_exceeded": error_type == "context_window_exceeded",
            "unsupported_tool_request_seen": bool(unsupported_tool_requests),
        },
    )


def _remaining_deadline_seconds(
    deadline_monotonic: float | None,
    *,
    clock: Clock = SYSTEM_CLOCK,
) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, deadline_monotonic - clock.monotonic())


def _deadline_expired(
    deadline_monotonic: float | None,
    *,
    clock: Clock = SYSTEM_CLOCK,
) -> bool:
    remaining = _remaining_deadline_seconds(deadline_monotonic, clock=clock)
    return remaining is not None and remaining <= 0


def _deadline_call_error(
    *,
    config: LlmConfig,
    model_turn: int,
    attempt: int,
    max_retries: int,
    reason: str,
    started: float | None = None,
    effective_timeout_seconds: float | None = None,
    remaining_before_call_s: float | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> LlmCallError:
    message = f"analysis deadline exceeded: {reason}"
    call_record = {
        "layer": "L1",
        "model": config.model,
        "model_turn": model_turn,
        "attempt": attempt,
        "max_retries": max_retries,
        "success": False,
        "latency_s": round(clock.monotonic() - started, 3) if started is not None else 0.0,
        "finish_reason": None,
        "usage": None,
        "tools_advertised": False,
        "reasoning_mode_requested": config.thinking_mode,
        "error_type": "analysis_deadline_exceeded",
        "error": message,
        "http_status": None,
        "response_body": None,
        "retryable": False,
        "retry_scheduled": False,
        "timeout": True,
        "deadline_exceeded": True,
        "deadline_reason": reason,
        "configured_request_timeout_seconds": config.timeout_seconds,
        "effective_request_timeout_seconds": effective_timeout_seconds,
        "remaining_analysis_budget_before_call_s": remaining_before_call_s,
    }
    return LlmCallError(message, call_record)


def _deadline_failure_result(
    *,
    config: LlmConfig,
    raw_output: str | None,
    model_calls: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    unsupported_tool_requests: list[dict[str, Any]],
    transcript_events: list[dict[str, Any]],
    model_turn: int,
    reason: str,
    clock: Clock = SYSTEM_CLOCK,
) -> L1EvidenceResult:
    return _provider_failure_result(
        config=config,
        exc=_deadline_call_error(
            config=config,
            model_turn=model_turn,
            attempt=1,
            max_retries=config.max_retries,
            reason=reason,
            clock=clock,
        ),
        raw_output=raw_output,
        model_calls=model_calls,
        tool_calls=tool_calls,
        unsupported_tool_requests=unsupported_tool_requests,
        transcript_events=transcript_events,
    )


def _initial_user_message(model_view: L0ModelFacingView) -> str:
    payload = {
        "response_schema": model_response_schema(),
        **model_view.prompt_payload(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _tool_loop_profile(config: LlmConfig) -> dict[str, Any]:
    advertised_tools = config.resolved_advertised_tools()
    tools_active = bool(advertised_tools)
    max_tool_rounds = max(1, config.max_tool_rounds) if tools_active else 0
    return {
        "tools_enabled": tools_active,
        "advertised_tools": list(advertised_tools),
        "max_tool_rounds": max_tool_rounds,
        "max_model_turns": max_tool_rounds + 1 if tools_active else 1,
        "meaning": (
            "tool-enabled model rounds followed by one tools-disabled final turn"
            if tools_active
            else "single tools-disabled model turn"
        ),
    }


def _tool_schemas(config: LlmConfig) -> list[dict[str, Any]]:
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "overview",
                "description": "Return a compact summary, head/tail preview, and L0 findings.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_log",
                "description": "Search the bound log with a regular expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "ignore_case": {"type": "boolean", "default": True},
                        "max_matches": {"type": "integer", "default": 50},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_window",
                "description": "Read original log lines around one center line.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "center_line": {"type": "integer"},
                        "before": {"type": "integer", "default": 20},
                        "after": {"type": "integer", "default": 80},
                    },
                    "required": ["center_line"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_evidence_objects",
                "description": (
                    "Resolve stable L0A evidence references from the current attempt "
                    "without rescanning the log."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "refs": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 128},
                            "maxItems": 8,
                        }
                    },
                    "required": ["refs"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    advertised = set(config.resolved_advertised_tools())
    return [schema for schema in schemas if schema["function"]["name"] in advertised]


def _execute_tool_call(
    tools: LogTools,
    tool_call: dict[str, Any],
    *,
    model_turn: int,
    advertised_tools: Sequence[str] = DEFAULT_ADVERTISED_TOOLS,
    clock: Clock = SYSTEM_CLOCK,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    started = clock.monotonic()
    name = str(tool_call.get("name") or "")
    args = _tool_args(tool_call)
    unsupported: dict[str, Any] | None = None
    try:
        if name not in advertised_tools:
            result = {"error": "tool_not_advertised", "requested_tool_name": name}
            unsupported = {
                "model_turn": model_turn,
                "requested_tool_name": name,
                "args_summary": _compact_args(args),
                "rejection_reason": "tool_not_advertised",
            }
        elif name == "overview":
            result = tools.overview()
        elif name == "grep_log":
            result = tools.grep_log(
                str(args.get("pattern") or ""),
                ignore_case=bool(args.get("ignore_case", True)),
                max_matches=int(args.get("max_matches", 50)),
            )
        elif name == "read_window":
            result = tools.read_window(
                int(args["center_line"]),
                before=int(args.get("before", 20)),
                after=int(args.get("after", 80)),
            )
        elif name == "get_evidence_objects":
            refs = args.get("refs")
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                raise ValueError("refs must be an array of strings")
            result = tools.get_evidence_objects(refs)
        else:
            result = {"error": "tool_not_implemented", "requested_tool_name": name}
            unsupported = {
                "model_turn": model_turn,
                "requested_tool_name": name,
                "args_summary": _compact_args(args),
                "rejection_reason": "tool_not_implemented",
            }
    except Exception as exc:
        result = {"error": str(exc), "requested_tool_name": name}

    result_text = json.dumps(result, sort_keys=True)
    record = {
        "tool_call_id": tool_call.get("id"),
        "model_turn": model_turn,
        "name": name,
        "args_summary": _compact_args(args),
        "latency_ms": round((clock.monotonic() - started) * 1000),
        "result_chars": len(result_text),
        "result_lines": _result_line_count(result),
        "total_matches": result.get("total_matches") if isinstance(result, dict) else None,
        "truncated": bool(result.get("truncated")) if isinstance(result, dict) else False,
        "error": result.get("error") if isinstance(result, dict) else None,
    }
    return result, record, unsupported


def _tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _compact_args(args: dict[str, Any]) -> str:
    return "|".join(f"{key}={args[key]}" for key in sorted(args))[:300]


def _result_line_count(result: dict[str, Any]) -> int:
    if "matches" in result and isinstance(result["matches"], list):
        return len(result["matches"])
    if "lines" in result and isinstance(result["lines"], list):
        return len(result["lines"])
    if "objects" in result and isinstance(result["objects"], list):
        return sum(
            len(payload["lines"])
            for item in result["objects"]
            if isinstance(item, dict)
            and isinstance((payload := item.get("payload")), dict)
            and isinstance(payload.get("lines"), list)
        )
    return 0


def _response_message(response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message if isinstance(message, dict) else {}


def _message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        result.append(
            {
                "id": item.get("id"),
                "name": function.get("name"),
                "arguments": function.get("arguments") or "{}",
            }
        )
    return result


def _assistant_tool_call_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {"role": "assistant", "content": message.get("content") or ""}
    if "tool_calls" in message:
        result["tool_calls"] = message["tool_calls"]
    return result


def _parse_model_evidence(text: str | None) -> dict[str, Any]:
    if not text:
        raise ValueError("empty model output")
    payload_text = _extract_json_text(text)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(payload_text)
        except Exception as exc:
            raise ValueError(f"model output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("model output JSON must be an object")
    contract_errors = model_evidence_contract_errors(payload)
    if contract_errors:
        raise ValueError("model evidence contract failed: " + "; ".join(contract_errors))
    return payload


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    if start < 0:
        return stripped
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(stripped[start:])
        return stripped[start : start + end]
    except json.JSONDecodeError:
        return stripped[start:]


def _load_api_key(config: LlmConfig) -> str:
    if config.api_key:
        return config.api_key.strip()
    if not config.api_key_file:
        raise RuntimeError("no API key configured")
    key_file = Path(os.path.expanduser(config.api_key_file))
    key = key_file.read_text(encoding="utf-8", errors="replace").strip()
    if not key:
        raise RuntimeError(f"empty API key file: {key_file}")
    return key


def _is_retryable_http_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _is_http_timeout_status(status: int) -> bool:
    return status == 504


def _is_context_window_exceeded_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(
        marker in lowered
        for marker in (
            "contextwindowexceeded",
            "context window exceeded",
            "maximum context length",
            "max context length",
        )
    )


def _truncate_error_detail(detail: str) -> str:
    if len(detail) <= MAX_ERROR_DETAIL_CHARS:
        return detail
    return detail[:MAX_ERROR_DETAIL_CHARS] + "...[truncated]"


def _float_env(environment: Mapping[str, str], name: str, default: float) -> float:
    value = environment.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _optional_float_env(
    environment: Mapping[str, str],
    name: str,
    default: float | None,
) -> float | None:
    value = environment.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"none", "null", "default"}:
        return None
    return float(value)


def _int_env(environment: Mapping[str, str], name: str, default: int) -> int:
    value = environment.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _optional_int_env(environment: Mapping[str, str], name: str) -> int | None:
    value = environment.get(name)
    if value is None or value == "":
        return None
    if value.lower() in {"none", "null", "default"}:
        return None
    return int(value)


def _bool_env(environment: Mapping[str, str], name: str, default: bool) -> bool:
    value = environment.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _tool_names_env(
    environment: Mapping[str, str],
    name: str,
    default: Sequence[str],
) -> tuple[str, ...]:
    value = environment.get(name)
    if value is None:
        return tuple(default)
    return tuple(item.strip() for item in value.split(",") if item.strip())
