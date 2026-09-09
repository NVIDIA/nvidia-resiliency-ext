# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for L1 provider-failure classification."""

import httpx
import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l1.openai_compatible import (
    LlmCallError,
    LlmConfig,
    OpenAICompatibleTransport,
    _httpx_request_error,
    _is_retryable_http_status,
    _request_body,
)


def test_http_retryability_is_closed_to_rate_limits_and_server_failures():
    retryable = {429, 500, 503, 599}
    non_retryable = {400, 401, 404, 422, 499}

    assert all(_is_retryable_http_status(status) for status in retryable)
    assert not any(_is_retryable_http_status(status) for status in non_retryable)


def test_provider_requests_explicitly_disable_streaming():
    body = _request_body(LlmConfig(), [], include_tools=False)

    assert body["stream"] is False


@pytest.mark.parametrize(
    "model",
    [
        "us/azure/openai/eccn-gpt-5.5",
        "us/azure/openai/eccn-gpt-5.6-sol",
    ],
)
def test_gpt_provider_profiles_omit_unsupported_sampling_parameters(model):
    body = _request_body(LlmConfig(model=model), [], include_tools=False)

    assert "temperature" not in body
    assert "top_p" not in body


def test_transient_transport_failures_are_retryable_with_stable_categories():
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    expected = {
        httpx.ConnectError: "connect_error",
        httpx.ReadError: "read_error",
        httpx.WriteError: "write_error",
        httpx.CloseError: "close_error",
        httpx.ProxyError: "proxy_error",
        httpx.RemoteProtocolError: "remote_protocol_error",
    }

    for error_type, category in expected.items():
        assert _httpx_request_error(error_type("failure", request=request)) == (
            category,
            True,
        )


def test_non_transient_transport_failures_are_not_retried():
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    expected = {
        httpx.LocalProtocolError: "local_protocol_error",
        httpx.UnsupportedProtocol: "unsupported_protocol",
        httpx.DecodingError: "response_decoding_error",
    }

    for error_type, category in expected.items():
        assert _httpx_request_error(error_type("failure", request=request)) == (
            category,
            False,
        )
    assert _httpx_request_error(httpx.RequestError("failure", request=request)) == (
        "request_error",
        False,
    )


def test_invalid_provider_json_is_a_retryable_provider_failure():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"{not-json"))
    )
    transport = OpenAICompatibleTransport(
        LlmConfig(base_url="https://provider.example/v1"),
        http_client=client,
    )

    try:
        with pytest.raises(LlmCallError) as failure:
            transport.call(
                api_key="secret",
                messages=[],
                include_tools=False,
                model_turn=1,
            )
    finally:
        client.close()

    assert failure.value.call_record["error_type"] == "provider_response_decode_error"
    assert failure.value.call_record["retryable"] is True


def test_context_window_error_envelope_at_http_200_is_not_semantic_output():
    detail = "ContextWindowExceededError: maximum context length is 200000 tokens"
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"error": {"message": detail}})
        )
    )
    transport = OpenAICompatibleTransport(
        LlmConfig(base_url="https://provider.example/v1"),
        http_client=client,
    )

    try:
        with pytest.raises(LlmCallError) as failure:
            transport.call(
                api_key="secret",
                messages=[],
                include_tools=False,
                model_turn=1,
            )
    finally:
        client.close()

    record = failure.value.call_record
    assert record["http_status"] == 200
    assert record["error_type"] == "context_window_exceeded"
    assert record["retryable"] is False
    assert record["timeout"] is False


@pytest.mark.parametrize("status", [429, 503, 504])
def test_context_window_rejection_precedes_http_retry_and_timeout_status(status):
    detail = "ContextWindowExceededError: this model's maximum context length is " "200000 tokens"
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(status, text=detail))
    )
    transport = OpenAICompatibleTransport(
        LlmConfig(base_url="https://provider.example/v1"),
        http_client=client,
    )

    try:
        with pytest.raises(LlmCallError) as failure:
            transport.call(
                api_key="secret",
                messages=[],
                include_tools=False,
                model_turn=1,
            )
    finally:
        client.close()

    record = failure.value.call_record
    assert record["http_status"] == status
    assert record["error_type"] == "context_window_exceeded"
    assert record["retryable"] is False
    assert record["retry_scheduled"] is False
    assert record["timeout"] is False
    assert "maximum context length" in record["response_body"]
