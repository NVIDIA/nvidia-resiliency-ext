# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import logging
import sys
import threading
import time
import types
from typing import Any, Dict

from nvidia_resiliency_ext.attribution.coalescing.coalescer import RequestCoalescer
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

    output_parsers = _stub_module(monkeypatch, "langchain_core.output_parsers")
    output_parsers.StrOutputParser = object
    prompts = _stub_module(monkeypatch, "langchain_core.prompts")
    prompts.ChatPromptTemplate = types.SimpleNamespace(
        from_template=lambda *_args, **_kwargs: object()
    )
    runnables = _stub_module(monkeypatch, "langchain_core.runnables")
    runnables.RunnablePassthrough = object

    _stub_module(monkeypatch, "logsage")
    _stub_module(monkeypatch, "logsage.auto_resume_policy")
    attribution_classes = _stub_module(
        monkeypatch, "logsage.auto_resume_policy.attribution_classes"
    )

    class StubErrorAttribution:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    stub_attribution = types.SimpleNamespace(
        APPLICATION_DONE="APPLICATION_DONE",
        ERRORS_NOT_FOUND="ERRORS_NOT_FOUND",
        LLM_FAILURE="LLM_FAILURE",
        SLURM_STEP_CANCELLED="SLURM_STEP_CANCELLED",
        SLURM_STEP_CANCELLED_JOB_REQUEUE="SLURM_STEP_CANCELLED_JOB_REQUEUE",
    )
    stub_auto_resume = types.SimpleNamespace(
        ERRORS_NOT_FOUND="ERRORS_NOT_FOUND",
        LLM_FAILURE="LLM_FAILURE",
        RESTART_IMMEDIATE="RESTART IMMEDIATE",
        STOP_NO_RESTART="STOP - DONT RESTART IMMEDIATE",
    )
    stub_finished = types.SimpleNamespace(
        APPLICATION_DONE="APPLICATION_DONE",
        LLM_FAILURE="LLM_FAILURE",
        SLURM_CANCELLED="SLURM_CANCELLED",
        SLURM_CANCELLED_JOB_REQUEUE="SLURM_CANCELLED_JOB_REQUEUE",
        SLURM_CANCELLED_TIME_LIMIT="SLURM_CANCELLED_TIME_LIMIT",
    )
    attribution_classes.ApplicationData = object
    attribution_classes.Attribution = stub_attribution
    attribution_classes.AutoResumeAction = stub_auto_resume
    attribution_classes.ErrorAttribution = StubErrorAttribution
    attribution_classes.FinishedStatus = stub_finished
    attribution_classes.LRUCache = object

    error_attribution = _stub_module(monkeypatch, "logsage.auto_resume_policy.error_attribution")
    error_attribution.CONTEXT_SIZE = 4096
    error_attribution.get_attribution = lambda *args, **kwargs: (None, None, None, None)
    error_attribution.get_auto_resume = lambda *args, **kwargs: ("", "")
    error_attribution.get_proposed_solution_cat = lambda *args, **kwargs: None

    error_extraction = _stub_module(monkeypatch, "logsage.auto_resume_policy.error_extraction")
    error_extraction.finished_validation = lambda _llm, data: data
    error_extraction.return_application_errors = lambda *args, **kwargs: []
    error_extraction.return_application_errors_rt = lambda *args, **kwargs: types.SimpleNamespace(
        checkpoint_saved=False
    )
    prompts_mod = _stub_module(monkeypatch, "logsage.auto_resume_policy.prompts")
    prompts_mod.template_post_error_check = ""
    util_postprocessing = _stub_module(
        monkeypatch, "logsage.auto_resume_policy.util_postprocessing"
    )
    util_postprocessing.get_auto_resume_postprocessing = lambda *args, **kwargs: False
    utils = _stub_module(monkeypatch, "logsage.auto_resume_policy.utils")
    utils.chunk_indices = lambda *args, **kwargs: []

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


def test_log_sage_runner_lib_run_allows_coalescer_timeout(monkeypatch):
    _import_log_analyzer_with_optional_dependency_stubs(monkeypatch)
    module = importlib.import_module("nvidia_resiliency_ext.attribution.orchestration.log_analyzer")
    runner = module.LogSageRunner(LogSageExecutionConfig(use_lib_log_analysis=True))
    started = threading.Event()
    finished = threading.Event()

    class BlockingFakeLogAnalyzer:
        async def run(self, kwargs: dict[str, Any]) -> list[Any]:
            started.set()
            time.sleep(0.5)
            finished.set()
            return []

    async def fake_get_lib_log_analyzer(kwargs: dict[str, Any]) -> BlockingFakeLogAnalyzer:
        return BlockingFakeLogAnalyzer()

    runner._get_lib_log_analyzer = fake_get_lib_log_analyzer

    async def run() -> None:
        coalescer = RequestCoalescer(compute_timeout=0.05)

        t0 = time.monotonic()
        result = await coalescer.get_or_compute(
            "/tmp/job.log",
            lambda: runner._fetch_log_result_lib("/tmp/job.log"),
        )
        elapsed = time.monotonic() - t0

        assert started.is_set()
        assert result["state"] == "timeout"
        assert result["recommendation"]["action"] == "TIMEOUT"
        assert elapsed < 0.3

        assert await asyncio.to_thread(finished.wait, 1.0)
        await asyncio.wait_for(runner._log_analysis_lock.acquire(), timeout=0.1)
        runner._log_analysis_lock.release()

    asyncio.run(run())


def test_log_sage_runner_logs_worker_exception_after_timeout(monkeypatch, caplog):
    _import_log_analyzer_with_optional_dependency_stubs(monkeypatch)
    module = importlib.import_module("nvidia_resiliency_ext.attribution.orchestration.log_analyzer")
    runner = module.LogSageRunner(LogSageExecutionConfig(use_lib_log_analysis=True))

    class FailingFakeLogAnalyzer:
        async def run(self, kwargs: dict[str, Any]) -> list[Any]:
            time.sleep(0.2)
            raise RuntimeError("logsage worker failed")

    async def fake_get_lib_log_analyzer(kwargs: dict[str, Any]) -> FailingFakeLogAnalyzer:
        return FailingFakeLogAnalyzer()

    runner._get_lib_log_analyzer = fake_get_lib_log_analyzer

    async def run() -> None:
        coalescer = RequestCoalescer(compute_timeout=0.05)
        result = await coalescer.get_or_compute(
            "/tmp/job.log",
            lambda: runner._fetch_log_result_lib("/tmp/job.log"),
        )

        assert result["state"] == "timeout"

        deadline = time.monotonic() + 1.0
        while runner._lib_log_analysis_tasks and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not runner._lib_log_analysis_tasks

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        asyncio.run(run())

    assert "Lib LogSage analysis worker failed after caller cancellation or timeout" in caplog.text
    assert "logsage worker failed" in caplog.text


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


def test_log_fr_analyzer_mcp_uses_cycle_counter_for_per_cycle_log(monkeypatch):
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
                "fr": {"result": None, "state": "CONTINUE"},
            }

    runner._mcp_client = FakeMCPClient()

    async def run() -> None:
        await runner.fetch_log_fr_analyzer_mcp(
            "/tmp/train_cycle5.log",
            "/tmp/fr",
        )

        assert captured["name"] == "log_fr_analyzer"
        assert captured["kwargs"]["is_per_cycle"] is True
        assert captured["kwargs"]["cycle_counter"] == 5

    asyncio.run(run())
