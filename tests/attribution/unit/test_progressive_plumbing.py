# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys
import types
from typing import Any

from nvidia_resiliency_ext.attribution.orchestration.config import ErrorCode, LogSageExecutionConfig
from nvidia_resiliency_ext.attribution.orchestration.job import JobMode
from nvidia_resiliency_ext.attribution.orchestration.progressive import (
    ANALYSIS_INTENT_PROGRESSIVE,
    MODULE_LOG_ANALYZER_PROGRESSIVE_START,
    PROGRESSIVE_STATUS_UNSUPPORTED,
    ProgressiveLogAnalysisStartTool,
    ProgressiveStartResult,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    LogAnalyzerError,
    LogAnalyzerSubmitResult,
)


def _stub_module(monkeypatch, name: str):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _import_analyzer_with_optional_dependency_stubs(monkeypatch):
    for module_name in (
        "nvidia_resiliency_ext.attribution.analyzer.engine",
        "nvidia_resiliency_ext.attribution.orchestration.log_analyzer",
        "nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

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

    module = importlib.import_module("nvidia_resiliency_ext.attribution.analyzer.engine")
    return module.Analyzer


class FakeLogAnalyzer:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, str, str | None]] = []
        self.progressive_starts: list[tuple[str, str, str | None]] = []
        self.progressive_started: asyncio.Event | None = None
        self.progressive_release: asyncio.Event | None = None

    def register_callbacks(self, _callback: Any) -> None:
        pass

    async def submit(
        self,
        log_path: str,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> LogAnalyzerSubmitResult:
        self.submissions.append((log_path, user, job_id))
        return LogAnalyzerSubmitResult(
            submitted=True,
            normalized_path=log_path,
            mode=JobMode.SINGLE.value,
        )

    async def start(
        self,
        log_path: str,
        *,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> ProgressiveStartResult:
        self.progressive_starts.append((log_path, user, job_id))
        if self.progressive_started is not None:
            self.progressive_started.set()
        if self.progressive_release is not None:
            await self.progressive_release.wait()
        return ProgressiveStartResult(status="accepted")

    async def start_progressive_analysis(
        self,
        log_path: str,
        *,
        user: str = "unknown",
        job_id: str | None = None,
    ) -> ProgressiveStartResult:
        return await self.start(log_path, user=user, job_id=job_id)


class FakeMcpClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.session = object()
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run_module_resilient(
        self,
        module_name: str,
        *,
        max_attempts: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((module_name, {"max_attempts": max_attempts, **kwargs}))
        return self.response


def test_progressive_submit_starts_backend_when_enabled(monkeypatch, tmp_path):
    Analyzer = _import_analyzer_with_optional_dependency_stubs(monkeypatch)
    log_analyzer = FakeLogAnalyzer()
    analyzer = Analyzer(
        allowed_root=str(tmp_path),
        log_analyzer=log_analyzer,
        progressive_analysis_enabled=True,
    )

    async def run() -> None:
        log_analyzer.progressive_started = asyncio.Event()
        log_analyzer.progressive_release = asyncio.Event()

        result = await asyncio.wait_for(
            analyzer.submit(
                str(tmp_path / "train_cycle0.log"),
                user="alice",
                analysis_intent=ANALYSIS_INTENT_PROGRESSIVE,
            ),
            timeout=0.5,
        )
        assert result.submitted is True
        await asyncio.wait_for(log_analyzer.progressive_started.wait(), timeout=0.5)
        log_analyzer.progressive_release.set()
        await asyncio.sleep(0)

    asyncio.run(run())

    assert log_analyzer.submissions == [(str(tmp_path / "train_cycle0.log"), "alice", None)]
    assert log_analyzer.progressive_starts == [(str(tmp_path / "train_cycle0.log"), "alice", None)]


def test_progressive_submit_is_ignored_when_feature_gate_disabled(monkeypatch, tmp_path):
    Analyzer = _import_analyzer_with_optional_dependency_stubs(monkeypatch)
    log_analyzer = FakeLogAnalyzer()
    analyzer = Analyzer(
        allowed_root=str(tmp_path),
        log_analyzer=log_analyzer,
        progressive_analysis_enabled=False,
    )

    async def run() -> None:
        await analyzer.submit(
            str(tmp_path / "train_cycle0.log"),
            analysis_intent=ANALYSIS_INTENT_PROGRESSIVE,
        )

    asyncio.run(run())

    assert log_analyzer.progressive_starts == []


def test_invalid_analysis_intent_uses_parameter_error(monkeypatch, tmp_path):
    Analyzer = _import_analyzer_with_optional_dependency_stubs(monkeypatch)
    log_analyzer = FakeLogAnalyzer()
    analyzer = Analyzer(
        allowed_root=str(tmp_path),
        log_analyzer=log_analyzer,
        progressive_analysis_enabled=True,
    )

    async def run() -> LogAnalyzerError:
        result = await analyzer.submit(
            str(tmp_path / "train_cycle0.log"),
            analysis_intent="unexpected",
        )
        assert isinstance(result, LogAnalyzerError)
        return result

    result = asyncio.run(run())

    assert result.error_code == ErrorCode.INVALID_PARAMETER
    assert log_analyzer.submissions == []


def test_progressive_start_tool_returns_unsupported_without_final_result():
    tool = ProgressiveLogAnalysisStartTool({})

    async def run() -> dict[str, str | None]:
        return await tool.run({"log_path": "/tmp/train_cycle0.log", "is_per_cycle": True})

    result = asyncio.run(run())

    assert result == {
        "module": MODULE_LOG_ANALYZER_PROGRESSIVE_START,
        "status": PROGRESSIVE_STATUS_UNSUPPORTED,
        "message": "LogSage progressive start API is not configured",
        "handle": None,
    }


def test_progressive_start_uses_mcp_loganalysis_tool(monkeypatch, tmp_path):
    _import_analyzer_with_optional_dependency_stubs(monkeypatch)
    module = importlib.import_module("nvidia_resiliency_ext.attribution.orchestration.log_analyzer")
    fake_client = FakeMcpClient(
        {
            "module": MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            "status": PROGRESSIVE_STATUS_UNSUPPORTED,
            "message": "not wired yet",
            "handle": None,
        }
    )
    monkeypatch.setattr(module, "create_mcp_client", lambda **_kwargs: fake_client)
    runner = module.LogSageRunner(LogSageExecutionConfig(use_lib_log_analysis=False))

    async def run() -> ProgressiveStartResult:
        return await runner.start_progressive_analysis(
            str(tmp_path / "train_cycle0.log"),
            user="alice",
            job_id="123",
        )

    result = asyncio.run(run())

    assert result == ProgressiveStartResult(
        status=PROGRESSIVE_STATUS_UNSUPPORTED,
        message="not wired yet",
    )
    assert fake_client.calls == [
        (
            MODULE_LOG_ANALYZER_PROGRESSIVE_START,
            {
                "max_attempts": 3,
                "log_path": str(tmp_path / "train_cycle0.log"),
                "is_per_cycle": True,
                "user": "alice",
                "job_id": "123",
            },
        )
    ]
