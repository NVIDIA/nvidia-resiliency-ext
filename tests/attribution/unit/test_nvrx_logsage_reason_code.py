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

    return importlib.import_module(module_name)


def test_stop_no_restart_action_wins_over_restart_substring(monkeypatch):
    nvrx_logsage = _import_nvrx_logsage_with_optional_dependency_stubs(monkeypatch)

    assert nvrx_logsage._action_from_logsage_head("STOP - DONT RESTART IMMEDIATE") == "STOP"
    assert nvrx_logsage._action_from_logsage_head("RESTART IMMEDIATE") == "RESTART"


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
