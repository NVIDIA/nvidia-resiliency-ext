# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
import uuid
from pathlib import Path


def _load_score_attribution_script(monkeypatch, *, response_content='{"notes":"ok"}'):
    calls = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": response_content}}]}

    class _Client:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, headers, json):
            calls.append(
                {
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": self.timeout,
                }
            )
            return _Response()

    api_keys = types.ModuleType("nvidia_resiliency_ext.attribution.api_keys")
    api_keys.load_llm_api_key = lambda: ""
    config = types.ModuleType("nvidia_resiliency_ext.attribution.orchestration.config")
    config.DEFAULT_LLM_BASE_URL = "https://example.test/v1"

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=_Client))
    monkeypatch.setitem(sys.modules, "nvidia_resiliency_ext.attribution.api_keys", api_keys)
    monkeypatch.setitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.orchestration.config",
        config,
    )

    script_path = (
        Path(__file__).resolve().parents[3]
        / "src/nvidia_resiliency_ext/skills/nvrx-attr/scripts/score_attribution.py"
    )
    module_name = f"_score_attribution_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, calls


def test_qwen_judge_request_disables_thinking(monkeypatch):
    module, calls = _load_score_attribution_script(monkeypatch)

    assert (
        module.invoke_judge_model(
            "prompt",
            model="nvidia/qwen/qwen3.5-35b-a3b",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
        == '{"notes":"ok"}'
    )

    assert calls[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_non_qwen_judge_request_omits_thinking_toggle(monkeypatch):
    module, calls = _load_score_attribution_script(monkeypatch)

    module.invoke_judge_model(
        "prompt",
        model="us/azure/openai/eccn-gpt-5.6-sol",
        api_key="test-key",
        base_url="https://example.test/v1/chat/completions",
    )

    assert calls[0]["url"] == "https://example.test/v1/chat/completions"
    assert "chat_template_kwargs" not in calls[0]["json"]


def test_restart_agent_json_is_pretty_printed_for_judge(monkeypatch):
    module, _ = _load_score_attribution_script(monkeypatch)

    rendered = module.render_log_output_for_judge('{"decision":"RESTART","primary_failure":null}')

    assert '"decision": "RESTART"' in rendered
    assert '"primary_failure": null' in rendered
