# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys
import types

from nvidia_resiliency_ext.attribution.base import AttributionState
from nvidia_resiliency_ext.attribution.orchestration.types import (
    AttributionRecommendation,
    LogSageAnalysisResult,
    RawAnalysisResultItem,
)


def _stub_module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch):
    monkeypatch.delitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr",
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
    attribution_classes.ErrorAttribution = object
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

    return importlib.import_module(
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr"
    )


def test_excluded_rank_count_parses_comma_separated_ranks(monkeypatch):
    module = _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch)

    assert module._excluded_ranks_from_attribution_result(
        "Summary\n- List of ranks to be excluded: 1,2,3\n"
    ) == {1, 2, 3}


def test_combined_log_fr_threshold_uses_rank_count_not_line_count(monkeypatch):
    module = _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch)
    analyzer = object.__new__(module.CombinedLogFR)
    analyzer.threshold = 2

    async def run():
        return await analyzer.print_output("List of ranks to be excluded: 1,2,3")

    _result, state = asyncio.run(run())

    assert state == AttributionState.STOP


def test_combined_log_fr_exact_threshold_continues(monkeypatch):
    module = _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch)
    analyzer = object.__new__(module.CombinedLogFR)
    analyzer.threshold = 3

    async def run():
        return await analyzer.print_output("List of ranks to be excluded: [1, 2, 3]")

    _result, state = asyncio.run(run())

    assert state == AttributionState.CONTINUE


def test_log_fr_mcp_path_collects_without_merge_by_default(monkeypatch):
    _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp",
        raising=False,
    )
    module = importlib.import_module(
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp"
    )

    raw_item = {
        "raw_text": "ERRORS NOT FOUND",
        "auto_resume": "ERRORS NOT FOUND",
        "auto_resume_explanation": "",
        "attribution_text": "",
        "checkpoint_saved_flag": 0,
        "action": "CONTINUE",
        "primary_issues": [],
        "secondary_issues": [],
    }

    class FakeLogAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return ([raw_item], AttributionState.CONTINUE)

    class FakeFRAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return (
                {"analysis_text": "fr table", "hanging_ranks": "hanging ranks: [1, 2, 3]"},
                AttributionState.STOP,
            )

    class FakeCombinedLogFR:
        def __init__(self, _kwargs):
            raise AssertionError("merge LLM should not be constructed unless merge_llm=True")

    monkeypatch.setattr(module, "NVRxLogAnalyzer", FakeLogAnalyzer)
    monkeypatch.setattr(module, "CollectiveAnalyzer", FakeFRAnalyzer)
    monkeypatch.setattr(module, "CombinedLogFR", FakeCombinedLogFR)
    monkeypatch.setattr(module, "fr_path_resolvable_for_collective_analyzer", lambda _path: True)

    async def run():
        orchestrator = module.CombinedLogFRMCPOrchestrator()
        return await orchestrator._run_from_paths(
            {"log_path": "/tmp/job.log", "fr_path": "/tmp/fr"}
        )

    payload, state = asyncio.run(run())

    assert state == AttributionState.CONTINUE
    assert payload["recommendation"] == {"action": "CONTINUE", "source": "log_analyzer"}
    assert payload["fr"]["state"] == "STOP"
    assert "llm_merged_summary" not in payload


def test_log_fr_mcp_path_state_ignores_fr_and_merge_stop_when_merge_enabled(monkeypatch):
    _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp",
        raising=False,
    )
    module = importlib.import_module(
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp"
    )

    raw_item = {
        "raw_text": "ERRORS NOT FOUND",
        "auto_resume": "ERRORS NOT FOUND",
        "auto_resume_explanation": "",
        "attribution_text": "",
        "checkpoint_saved_flag": 0,
        "action": "CONTINUE",
        "primary_issues": [],
        "secondary_issues": [],
    }

    class FakeLogAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return ([raw_item], AttributionState.CONTINUE)

    class FakeFRAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return (
                {"analysis_text": "fr table", "hanging_ranks": "hanging ranks: [1, 2, 3]"},
                AttributionState.STOP,
            )

    class FakeCombinedLogFR:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return (
                "List of ranks to be excluded: 1,2,3",
                AttributionState.STOP,
            )

    monkeypatch.setattr(module, "NVRxLogAnalyzer", FakeLogAnalyzer)
    monkeypatch.setattr(module, "CollectiveAnalyzer", FakeFRAnalyzer)
    monkeypatch.setattr(module, "CombinedLogFR", FakeCombinedLogFR)
    monkeypatch.setattr(module, "fr_path_resolvable_for_collective_analyzer", lambda _path: True)

    async def run():
        orchestrator = module.CombinedLogFRMCPOrchestrator()
        return await orchestrator._run_from_paths(
            {"log_path": "/tmp/job.log", "fr_path": "/tmp/fr", "merge_llm": True}
        )

    payload, state = asyncio.run(run())

    assert state == AttributionState.CONTINUE
    assert payload["recommendation"] == {"action": "CONTINUE", "source": "log_analyzer"}
    assert payload["fr"]["state"] == "STOP"
    assert payload["llm_merged_summary"] == "List of ranks to be excluded: 1,2,3"


def test_log_fr_mcp_path_merge_unwraps_logsage_result_for_llm(monkeypatch):
    _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp",
        raising=False,
    )
    module = importlib.import_module(
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp"
    )

    item = RawAnalysisResultItem(
        raw_text="STOP - DONT RESTART IMMEDIATE\ncheckpoint was not saved",
        auto_resume="STOP - DONT RESTART IMMEDIATE",
        auto_resume_explanation="checkpoint was not saved",
        attribution_text="Primary issues: [NCCL TIMEOUT], Secondary issues: []",
        checkpoint_saved_flag=0,
        action="STOP",
        primary_issues=["NCCL TIMEOUT"],
        secondary_issues=[],
    )
    logsage_result = LogSageAnalysisResult(
        [item],
        AttributionRecommendation(action="STOP", source="log_analyzer"),
    )
    captured = {}

    class FakeLogAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return (logsage_result, AttributionState.STOP)

    class FakeFRAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return (
                {"analysis_text": "fr table", "hanging_ranks": "hanging ranks: [1, 2, 3]"},
                AttributionState.CONTINUE,
            )

    class FakeCombinedLogFR:
        def __init__(self, kwargs):
            captured["init_log_input"] = kwargs["input_data"][0]

        async def run(self, kwargs):
            captured["run_log_input"] = kwargs["input_data"][0]
            return ("merged", AttributionState.CONTINUE)

    monkeypatch.setattr(module, "NVRxLogAnalyzer", FakeLogAnalyzer)
    monkeypatch.setattr(module, "CollectiveAnalyzer", FakeFRAnalyzer)
    monkeypatch.setattr(module, "CombinedLogFR", FakeCombinedLogFR)
    monkeypatch.setattr(module, "fr_path_resolvable_for_collective_analyzer", lambda _path: True)

    async def run():
        orchestrator = module.CombinedLogFRMCPOrchestrator()
        return await orchestrator._run_from_paths(
            {"log_path": "/tmp/job.log", "fr_path": "/tmp/fr", "merge_llm": True}
        )

    payload, state = asyncio.run(run())

    assert captured["init_log_input"] == [item]
    assert captured["run_log_input"] == [item]
    assert state == AttributionState.STOP
    assert payload["result"] == [item.to_payload()]
    assert payload["llm_merged_summary"] == "merged"


def test_log_fr_mcp_skips_merge_when_fr_data_is_missing(monkeypatch):
    _import_combined_log_fr_with_optional_dependency_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp",
        raising=False,
    )
    module = importlib.import_module(
        "nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp"
    )

    raw_item = {
        "raw_text": "ERRORS NOT FOUND",
        "auto_resume": "ERRORS NOT FOUND",
        "auto_resume_explanation": "",
        "attribution_text": "",
        "checkpoint_saved_flag": 0,
        "action": "CONTINUE",
        "primary_issues": [],
        "secondary_issues": [],
    }

    class FakeLogAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return ([raw_item], AttributionState.CONTINUE)

    class FakeFRAnalyzer:
        def __init__(self, _kwargs):
            pass

        async def run(self, _kwargs):
            return (None, AttributionState.CONTINUE)

    class FakeCombinedLogFR:
        def __init__(self, _kwargs):
            raise AssertionError("merge LLM should not be constructed without FR data")

    monkeypatch.setattr(module, "NVRxLogAnalyzer", FakeLogAnalyzer)
    monkeypatch.setattr(module, "CollectiveAnalyzer", FakeFRAnalyzer)
    monkeypatch.setattr(module, "CombinedLogFR", FakeCombinedLogFR)
    monkeypatch.setattr(module, "fr_path_resolvable_for_collective_analyzer", lambda _path: True)

    async def run():
        orchestrator = module.CombinedLogFRMCPOrchestrator()
        return await orchestrator._run_from_paths(
            {"log_path": "/tmp/job.log", "fr_path": "/tmp/fr"}
        )

    payload, state = asyncio.run(run())

    assert state == AttributionState.CONTINUE
    assert payload["fr"]["result"] is None
    assert payload["fr"]["state"] == "CONTINUE"
    assert "llm_merged_summary" not in payload
