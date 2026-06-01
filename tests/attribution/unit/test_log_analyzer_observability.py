# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys
import threading
import time
import types
from typing import Any, Dict

from nvidia_resiliency_ext.attribution.orchestration.analysis_pipeline import AnalysisPipelineMode
from nvidia_resiliency_ext.attribution.orchestration.config import LogSageExecutionConfig


def _stub_module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _import_log_analyzer_with_optional_dependency_stubs(monkeypatch):
    monkeypatch.delitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.orchestration.log_analyzer",
        raising=False,
    )
    monkeypatch.delitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage",
        raising=False,
    )

    langchain_openai = _stub_module(monkeypatch, "langchain_openai")
    langchain_openai.ChatOpenAI = object

    _stub_module(monkeypatch, "logsage")
    _stub_module(monkeypatch, "logsage.auto_resume_policy")
    attribution_classes = _stub_module(
        monkeypatch, "logsage.auto_resume_policy.attribution_classes"
    )
    attribution_classes.ApplicationData = object
    attribution_classes.LRUCache = object

    error_attribution = _stub_module(monkeypatch, "logsage.auto_resume_policy.error_attribution")
    error_attribution.get_proposed_solution_cat = lambda *args, **kwargs: None

    error_extraction = _stub_module(monkeypatch, "logsage.auto_resume_policy.error_extraction")
    error_extraction.return_application_errors = lambda *args, **kwargs: []

    httpx = _stub_module(monkeypatch, "httpx")
    httpx.post = lambda *args, **kwargs: types.SimpleNamespace(status_code=201, text="created")

    module = importlib.import_module("nvidia_resiliency_ext.attribution.orchestration.log_analyzer")
    return module.LogAnalyzer


class _FakeRunner:
    def __init__(self, log_result: Dict[str, Any]):
        self.config = LogSageExecutionConfig(use_lib_log_analysis=True)
        self.log_result = log_result

    async def fetch_log_result(self, path: str) -> Dict[str, Any]:
        return self.log_result

    async def shutdown_async(self) -> None:
        pass


async def _track_submission(path: str) -> None:
    pass


def _log_result() -> Dict[str, Any]:
    return {
        "module": "log_analyzer",
        "result": [
            {
                "raw_text": "RESTART IMMEDIATE",
                "auto_resume": "RESTART IMMEDIATE",
                "auto_resume_explanation": "transient infra issue",
                "attribution_text": "transient infra issue",
                "checkpoint_saved_flag": 0,
                "action": "RESTART",
                "primary_issues": [],
                "secondary_issues": [],
            }
        ],
        "recommendation": {"action": "RESTART", "source": "log_analyzer"},
    }


def test_run_attribution_returns_before_observability_post_completes(tmp_path, monkeypatch):
    LogAnalyzer = _import_log_analyzer_with_optional_dependency_stubs(monkeypatch)

    async def run() -> None:
        log_path = tmp_path / "job.log"
        log_path.write_text("training failed\n", encoding="utf-8")
        runner = _FakeRunner(_log_result())
        analyzer = LogAnalyzer(
            allowed_root=str(tmp_path),
            log_sage=runner.config,
            track_submission=_track_submission,
            trace_analyzer=None,
            analysis_pipeline_mode=AnalysisPipelineMode.LOG_ONLY,
            runner=runner,
        )

        post_started = threading.Event()
        release_post = threading.Event()
        post_calls = []

        def blocking_post(*args: Any, **kwargs: Any) -> None:
            post_started.set()
            release_post.wait(timeout=2.0)
            post_calls.append((args, kwargs))
            raise RuntimeError("dataflow unavailable")

        monkeypatch.setattr(analyzer, "_post_analysis_results", blocking_post)

        started_at = time.perf_counter()
        result = await analyzer.run_attribution_for_path(
            str(log_path),
            user="user",
            job_id="123",
        )
        elapsed = time.perf_counter() - started_at

        assert result.log_result == runner.log_result
        assert elapsed < 0.5

        started = await asyncio.to_thread(post_started.wait, 1.0)
        assert started
        tasks = tuple(analyzer._post_tasks)
        assert tasks

        release_post.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
        assert post_calls
        await analyzer.shutdown_async()

    asyncio.run(run())


def test_log_sage_runner_omits_unset_llm_overrides(monkeypatch):
    _import_log_analyzer_with_optional_dependency_stubs(monkeypatch)
    module = importlib.import_module("nvidia_resiliency_ext.attribution.orchestration.log_analyzer")
    runner = module.LogSageRunner(LogSageExecutionConfig(use_lib_log_analysis=True))
    captured: dict[str, dict[str, Any]] = {}

    class FakeLogAnalyzer:
        async def run(self, kwargs: dict[str, Any]) -> list[Any]:
            captured["run"] = dict(kwargs)
            return []

    async def fake_get_lib_log_analyzer(kwargs: dict[str, Any]) -> FakeLogAnalyzer:
        captured["init"] = dict(kwargs)
        return FakeLogAnalyzer()

    runner._get_lib_log_analyzer = fake_get_lib_log_analyzer

    async def run() -> None:
        await runner._fetch_log_result_lib("/tmp/job.log")

    asyncio.run(run())

    for kwargs in captured.values():
        assert "model" not in kwargs
        assert "base_url" not in kwargs
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "max_tokens" not in kwargs
        assert kwargs["log_path"] == "/tmp/job.log"
        assert kwargs["exclude_nvrx_logs"] is False


def test_log_sage_runner_preserves_explicit_llm_overrides(monkeypatch):
    _import_log_analyzer_with_optional_dependency_stubs(monkeypatch)
    module = importlib.import_module("nvidia_resiliency_ext.attribution.orchestration.log_analyzer")
    runner = module.LogSageRunner(
        LogSageExecutionConfig(
            use_lib_log_analysis=True,
            llm_model="override-model",
            llm_base_url="https://llm.example.test/v1",
            llm_temperature=0.0,
            llm_top_p=0.0,
            llm_max_tokens=0,
        )
    )
    captured: dict[str, dict[str, Any]] = {}

    class FakeLogAnalyzer:
        async def run(self, kwargs: dict[str, Any]) -> list[Any]:
            captured["run"] = dict(kwargs)
            return []

    async def fake_get_lib_log_analyzer(kwargs: dict[str, Any]) -> FakeLogAnalyzer:
        captured["init"] = dict(kwargs)
        return FakeLogAnalyzer()

    runner._get_lib_log_analyzer = fake_get_lib_log_analyzer

    async def run() -> None:
        await runner._fetch_log_result_lib("/tmp/job.log")

    asyncio.run(run())

    for kwargs in captured.values():
        assert kwargs["model"] == "override-model"
        assert kwargs["base_url"] == "https://llm.example.test/v1"
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] == 0.0
        assert kwargs["max_tokens"] == 0


def test_log_fr_analyzer_mcp_uses_top_level_log_contract(monkeypatch):
    _import_log_analyzer_with_optional_dependency_stubs(monkeypatch)
    module = importlib.import_module("nvidia_resiliency_ext.attribution.orchestration.log_analyzer")
    runner = object.__new__(module.LogSageRunner)
    runner.config = LogSageExecutionConfig(use_lib_log_analysis=False)
    runner._log_analysis_lock = asyncio.Lock()
    captured: dict[str, Any] = {}

    class FakeMCPClient:
        session = object()

        async def run_module_resilient(self, name: str, **kwargs: Any) -> dict[str, Any]:
            captured["name"] = name
            captured["kwargs"] = kwargs
            return {
                "module": "log_fr_analyzer",
                "result": _log_result()["result"],
                "recommendation": {"action": "RESTART", "source": "log_analyzer"},
                "fr": {
                    "result": {
                        "analysis_text": "collective table",
                        "hanging_ranks": "hanging ranks: [1]",
                    },
                    "state": "CONTINUE",
                },
                "llm_merged_summary": "ignored unless merge_llm=true",
                "result_id": "rid",
                "resource_uri": "attribution://log_fr_analyzer/rid",
            }

    runner._mcp_client = FakeMCPClient()

    async def run() -> None:
        log_dict, fr_analysis, summary = await runner.fetch_log_fr_analyzer_mcp(
            "/tmp/job.log",
            "/tmp/fr",
        )
        assert captured["name"] == "log_fr_analyzer"
        assert captured["kwargs"]["merge_llm"] is False
        assert log_dict == {
            "module": "log_fr_analyzer",
            "result": _log_result()["result"],
            "recommendation": {"action": "RESTART", "source": "log_analyzer"},
            "result_id": "rid",
            "resource_uri": "attribution://log_fr_analyzer/rid",
        }
        assert fr_analysis is not None
        assert fr_analysis.analysis_text == "collective table"
        assert fr_analysis.hanging_ranks == "hanging ranks: [1]"
        assert summary is None

        _log_dict, _fr_analysis, merged_summary = await runner.fetch_log_fr_analyzer_mcp(
            "/tmp/job.log",
            "/tmp/fr",
            merge_llm=True,
        )
        assert captured["kwargs"]["merge_llm"] is True
        assert merged_summary == "ignored unless merge_llm=true"

    asyncio.run(run())
