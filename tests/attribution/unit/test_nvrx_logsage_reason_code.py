import asyncio
import importlib
import sys
import types


def _stub_module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch):
    module_name = "nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

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

    return importlib.import_module(module_name)


def test_stop_no_restart_action_wins_over_restart_substring(monkeypatch):
    nvrx_logsage = _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch)

    assert nvrx_logsage._action_from_logsage_head("STOP - DONT RESTART IMMEDIATE") == "STOP"
    assert nvrx_logsage._action_from_logsage_head("RESTART IMMEDIATE") == "RESTART"


def test_cycle_counter_key_normalizes_ft_per_cycle_log_names(monkeypatch):
    nvrx_logsage = _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch)

    assert (
        nvrx_logsage._cycle_counter_key("/mnt/logs/test_job_cycle0.log") == "/mnt/logs/test_job.log"
    )
    assert (
        nvrx_logsage._cycle_counter_key("/mnt/logs/test_job_cycle4.log") == "/mnt/logs/test_job.log"
    )
    assert nvrx_logsage._cycle_counter_key("/mnt/logs/nvrx_4.log") == "/mnt/logs/nvrx.log"


def test_per_cycle_non_progressive_path_applies_repeated_error_policy(monkeypatch):
    nvrx_logsage = _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch)
    analyzer = object.__new__(nvrx_logsage.NVRxLogAnalyzer)
    analyzer._init_config = {
        "log_path": "/mnt/logs/test_job_cycle2.log",
        "is_per_cycle": True,
        "cycle_counter": 2,
    }
    analyzer.is_per_cycle = True
    analyzer.job_inline_data_dict = {}
    analyzer.attribution_dict = {
        "/mnt/logs/test_job_cycle1.log": "Primary issues: [OOM], Secondary issues: []"
    }
    analyzer.cycle_counter_dict = {"/mnt/logs/test_job.log": 2}
    analyzer.repeated_amount = 3
    analyzer.llm = object()

    monkeypatch.setattr(
        nvrx_logsage,
        "get_proposed_solution_cat",
        lambda *_args, **_kwargs: (
            "RESTART IMMEDIATE",
            "",
            "Attribution: Primary issues: [OOM], Secondary issues: []",
            "",
            "False",
        ),
    )
    monkeypatch.setattr(
        nvrx_logsage,
        "get_auto_resume_postprocessing",
        lambda *_args, **_kwargs: True,
    )
    output = types.SimpleNamespace(
        finished="RUNNING",
        original_text=["RuntimeError: OOM"],
        application_errors_list_full=[("RuntimeError: OOM", "", 0)],
        checkpoint_saved=False,
    )

    fields = asyncio.run(analyzer.llm_analyze([output]))

    assert fields[0][0] == "STOP - DONT RESTART IMMEDIATE"
    assert fields[0][1] == "Stop job due to repeated issue"
    assert analyzer.attribution_dict["/mnt/logs/test_job_cycle2.log"] == (
        "Primary issues: [OOM], Secondary issues: []"
    )
    assert analyzer.cycle_counter_dict["/mnt/logs/test_job.log"] == 2


def test_print_output_returns_structured_item_without_downstream_reparse(monkeypatch):
    nvrx_logsage = _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch)

    analysis_result, state = asyncio.run(
        nvrx_logsage.NVRxLogAnalyzer.print_output(
            object(),
            [
                (
                    "STOP - DONT RESTART IMMEDIATE",
                    "checkpoint was not saved",
                    "Attribution: Primary issues: [NCCL TIMEOUT], Secondary issues: [GPU XID]",
                    "",
                    "False",
                )
            ],
        )
    )

    assert state.name == "STOP"
    assert analysis_result.recommendation.action == "STOP"
    items = analysis_result.items
    assert len(items) == 1
    assert items[0].action == "STOP"
    assert items[0].auto_resume == "STOP - DONT RESTART IMMEDIATE"
    assert items[0].auto_resume_explanation == "checkpoint was not saved"
    assert (
        items[0].attribution_text == "Primary issues: [NCCL TIMEOUT], Secondary issues: [GPU XID]"
    )
    assert items[0].checkpoint_saved_flag == 0
    assert items[0].primary_issues == ["NCCL TIMEOUT"]
    assert items[0].secondary_issues == ["GPU XID"]


def test_logsage_tuple_parser_preserves_issue_names_with_brackets(monkeypatch):
    nvrx_logsage = _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch)

    item = nvrx_logsage._result_item_from_logsage_fields(
        (
            "STOP - DONT RESTART IMMEDIATE",
            "checkpoint was not saved",
            "Attribution: Primary issues: [NCCL error [code 5]], "
            "Secondary issues: [GPU XID [79]]",
            "",
            "False",
        ),
        raw_text="STOP - DONT RESTART IMMEDIATE\ncheckpoint was not saved",
        action="STOP",
    )

    assert item.primary_issues == ["NCCL error [code 5]"]
    assert item.secondary_issues == ["GPU XID [79]"]


def test_logsage_tuple_parser_falls_back_when_issue_literal_recurses(monkeypatch):
    nvrx_logsage = _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch)

    def raise_recursion(_value):
        raise RecursionError("too deep")

    monkeypatch.setattr(nvrx_logsage.ast, "literal_eval", raise_recursion)

    item = nvrx_logsage._result_item_from_logsage_fields(
        (
            "STOP - DONT RESTART IMMEDIATE",
            "checkpoint was not saved",
            "Attribution: Primary issues: [NCCL TIMEOUT], Secondary issues: []",
            "",
            "False",
        ),
        raw_text="STOP - DONT RESTART IMMEDIATE\ncheckpoint was not saved",
        action="STOP",
    )

    assert item.primary_issues == ["NCCL TIMEOUT"]
    assert item.secondary_issues == []
