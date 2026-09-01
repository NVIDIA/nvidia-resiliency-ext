# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the single-source L1 evidence-tool contracts."""

import json

from nvidia_resiliency_ext.attribution.restart_agent.identity import normalized_pattern
from nvidia_resiliency_ext.attribution.restart_agent.infrastructure.log_source import LogSnapshot
from nvidia_resiliency_ext.attribution.restart_agent.l0 import build_l0_bundle
from nvidia_resiliency_ext.attribution.restart_agent.l0.decision import build_decision_evidence
from nvidia_resiliency_ext.attribution.restart_agent.l0.projection import build_l0_model_facing_view
from nvidia_resiliency_ext.attribution.restart_agent.l1 import L1EvidenceResult
from nvidia_resiliency_ext.attribution.restart_agent.l1.openai_compatible import (
    LlmConfig,
    _tool_loop_profile,
    _tool_schemas,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.tool_contracts import (
    DEFAULT_ADVERTISED_TOOLS,
    L1_TOOL_CONTRACTS,
    TOOL_RESULT_SCHEMA_VERSION,
    advertised_tool_schemas,
    execute_tool_request,
)
from nvidia_resiliency_ext.attribution.restart_agent.l1.tools import LogTools
from nvidia_resiliency_ext.attribution.restart_agent.l2.grounding import (
    model_visible_line_numbers,
    model_visible_line_texts,
)
from nvidia_resiliency_ext.attribution.restart_agent.models import (
    DistributedFailureIncident,
    L0Bundle,
    NormalizedOccurrenceGroup,
)


def _tools(lines: tuple[str, ...] | None = None) -> LogTools:
    lines = lines or (
        "iteration 1 completed",
        "RuntimeError: observed failure",
        "scheduler cancelled step",
    )
    bundle = L0Bundle(
        log_path="/not/read.log",
        byte_size=sum(len(line) + 1 for line in lines),
        line_count=len(lines),
    )
    return LogTools(
        bundle,
        LogSnapshot(path=bundle.log_path, lines=lines, byte_size=bundle.byte_size),
    )


def _execute(name: str, arguments, *, advertised=None):
    return execute_tool_request(
        _tools(),
        name=name,
        raw_arguments=arguments,
        advertised_tools=advertised or DEFAULT_ADVERTISED_TOOLS,
    )


def test_advertised_schemas_come_from_the_executable_contract_registry():
    schemas = advertised_tool_schemas(DEFAULT_ADVERTISED_TOOLS)

    assert [item["function"]["name"] for item in schemas] == list(DEFAULT_ADVERTISED_TOOLS)
    grep_parameters = next(
        item["function"]["parameters"] for item in schemas if item["function"]["name"] == "grep_log"
    )
    assert grep_parameters["properties"]["max_matches"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 200,
        "default": 50,
    }
    assert grep_parameters["properties"]["result_mode"] == {
        "type": "string",
        "enum": ["compact", "raw"],
        "default": "compact",
    }
    assert set(L1_TOOL_CONTRACTS) == {
        "overview",
        "grep_log",
        "read_window",
        "get_evidence_objects",
    }


def test_zero_tool_rounds_resolves_to_one_tools_disabled_model_turn():
    config = LlmConfig(tools_enabled=True, max_tool_rounds=0)

    assert config.resolved_advertised_tools() == ()
    assert config.tools_active() is False
    assert _tool_schemas(config) == []
    assert _tool_loop_profile(config) == {
        "tools_enabled": False,
        "advertised_tools": [],
        "max_tool_rounds": 0,
        "max_model_turns": 1,
        "meaning": "single tools-disabled model turn",
    }


def test_default_tool_round_budget_is_three():
    assert LlmConfig().max_tool_rounds == 3


def test_negative_tool_rounds_are_rejected_at_l1_config_construction():
    try:
        LlmConfig(max_tool_rounds=-1)
    except ValueError as exc:
        assert str(exc) == "max_tool_rounds must not be negative"
    else:
        raise AssertionError("negative tool rounds must be rejected")


def test_successful_tool_result_uses_the_common_envelope():
    result, arguments, unsupported = _execute(
        "grep_log",
        json.dumps({"pattern": "RuntimeError"}),
    )

    assert result["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert result["tool"] == "grep_log"
    assert result["status"] == "ok"
    assert result["error"] is None
    assert result["data"]["matches"] == [
        {
            "line": 2,
            "text": "RuntimeError: observed failure",
            "group_kind": "individual_match",
            "occurrence_count": 1,
        }
    ]
    assert result["data"]["total_raw_matches"] == 1
    assert result["data"]["total_match_groups"] == 1
    assert result["data"]["collapsed_matches"] == 0
    assert result["data"]["scan_complete"] is True
    assert result["data"]["samples_truncated"] is False
    assert result["data"]["initial_view_overlap"] == {
        "available": False,
        "matched_group_count": 1,
        "represented_group_count": None,
        "new_group_count": None,
        "new_evidence_beyond_initial_view": None,
    }
    assert result["limits"]["max_matches"] == 50
    assert arguments == {
        "pattern": "RuntimeError",
        "ignore_case": True,
        "max_matches": 50,
        "result_mode": "compact",
    }
    assert unsupported is False


def test_explicit_grep_limit_above_the_default_is_honored():
    lines = tuple(f"RuntimeError: failure {index}" for index in range(75))

    result, arguments, unsupported = execute_tool_request(
        _tools(lines),
        name="grep_log",
        raw_arguments={"pattern": "RuntimeError", "max_matches": 60},
        advertised_tools=DEFAULT_ADVERTISED_TOOLS,
    )

    assert result["status"] == "ok"
    assert len(result["data"]["matches"]) == 60
    assert result["data"]["total_raw_matches"] == 75
    assert result["data"]["total_match_groups"] == 75
    assert result["data"]["collapsed_matches"] == 0
    assert result["data"]["scan_complete"] is True
    assert result["data"]["samples_truncated"] is True
    assert result["truncated"] is True
    assert result["limits"]["max_matches"] == 60
    assert arguments["max_matches"] == 60
    assert unsupported is False


def test_compact_grep_preserves_unique_evidence_after_12k_rank_fanout():
    fanout_count = 12_000
    lines = tuple(
        [
            *(
                f"[rank{rank}] RuntimeError: NCCL collective timeout"
                for rank in range(fanout_count)
            ),
            "[rank7] RuntimeError: unique CUDA launch failure",
        ]
    )
    bundle = L0Bundle(
        log_path="/not/read.log",
        byte_size=sum(len(line) + 1 for line in lines),
        line_count=len(lines),
        distributed_failure_incidents=(
            DistributedFailureIncident(
                incident_id="di-1",
                incident_kind="distributed_fanout",
                incident_type="distributed_exception_fanout",
                status="terminal",
                first_observed_line=1,
                last_observed_line=fanout_count,
                primary_observed_line=1,
                primary_observed_quote=lines[0],
                member_event_lines=tuple(range(1, fanout_count + 1)),
                event_count=fanout_count,
                observed_rank_count=fanout_count,
            ),
        ),
    )
    tools = LogTools(
        bundle,
        LogSnapshot(path=bundle.log_path, lines=lines, byte_size=bundle.byte_size),
    )

    compact = tools.grep_log("RuntimeError", max_matches=10)
    raw = tools.grep_log("RuntimeError", max_matches=10, result_mode="raw")

    assert compact["total_raw_matches"] == fanout_count + 1
    assert compact["total_match_groups"] == 2
    assert compact["collapsed_matches"] == fanout_count - 1
    assert compact["scan_complete"] is True
    assert compact["samples_truncated"] is False
    assert compact["matches"][0] == {
        "line": 1,
        "text": "[rank0] RuntimeError: NCCL collective timeout",
        "group_kind": "distributed_fanout",
        "incident_id": "di-1",
        "occurrence_count": fanout_count,
        "distinct_rank_count": fanout_count,
        "unattributed_occurrence_count": 0,
        "first_line": 1,
        "last_line": fanout_count,
        "sample_lines": [1, 2, 3, 4, 5],
        "sample_ranks": ["0", "1", "2", "3", "4"],
    }
    assert compact["matches"][1]["text"] == "[rank7] RuntimeError: unique CUDA launch failure"
    assert raw["total_match_groups"] == fanout_count + 1
    assert raw["collapsed_matches"] == 0
    assert len(raw["matches"]) == 10
    assert raw["scan_complete"] is True
    assert raw["samples_truncated"] is True


def test_compact_grep_reuses_l0_occurrence_groups_outside_distributed_fanout():
    lines = (
        "[rank0] NCCL WARN NET/IB : Got async event : PORT_ERR",
        "[rank1] NCCL WARN NET/IB : Got async event : PORT_ERR",
        "[rank2] RuntimeError: unique CUDA launch failure",
    )
    occurrence_group = NormalizedOccurrenceGroup(
        occurrence_group_id="og-7",
        normalized_shape=normalized_pattern(lines[0]),
        first_line=1,
        count=2,
        sample_lines=(1, 2),
        rank_spread=("0", "1"),
        registry_id="nccl_rdma_port_fault_event.v1",
        classification="error",
    )
    bundle = L0Bundle(
        log_path="/not/read.log",
        byte_size=sum(len(line) + 1 for line in lines),
        line_count=len(lines),
        occurrence_groups=(occurrence_group,),
    )

    result = LogTools(
        bundle,
        LogSnapshot(path=bundle.log_path, lines=lines, byte_size=bundle.byte_size),
    ).grep_log("NCCL|RuntimeError")

    assert result["scan_complete"] is True
    assert result["samples_truncated"] is False
    assert result["total_raw_matches"] == 3
    assert result["total_match_groups"] == 2
    assert result["collapsed_matches"] == 1
    assert result["matches"][0] == {
        "line": 1,
        "text": lines[0],
        "group_kind": "normalized_occurrence_group",
        "occurrence_count": 2,
        "distinct_rank_count": 2,
        "unattributed_occurrence_count": 0,
        "first_line": 1,
        "last_line": 2,
        "sample_lines": [1, 2],
        "sample_ranks": ["0", "1"],
        "occurrence_group_id": "og-7",
        "normalized_shape": normalized_pattern(lines[0]),
        "occurrence_group_total_count": 2,
        "occurrence_group_distinct_rank_count": 2,
        "classification": "error",
        "registry_id": "nccl_rdma_port_fault_event.v1",
    }


def test_compact_grep_reports_semantic_overlap_with_initial_model_view(tmp_path):
    log_path = tmp_path / "attempt.log"
    log_path.write_text(
        "iteration 7 completed\nRuntimeError: NCCL collective timeout\n",
        encoding="utf-8",
    )
    bundle = build_l0_bundle(str(log_path))
    model_view = build_l0_model_facing_view(bundle, build_decision_evidence(bundle))
    result = LogTools(
        bundle,
        LogSnapshot.read(log_path),
        model_view=model_view,
    ).grep_log("RuntimeError")

    assert result["initial_view_overlap"]["available"] is True
    assert result["initial_view_overlap"]["matched_group_count"] == 1
    assert result["initial_view_overlap"]["represented_group_count"] == 1
    assert result["initial_view_overlap"]["new_group_count"] == 0
    assert result["initial_view_overlap"]["new_evidence_beyond_initial_view"] is False


def test_compact_grep_prioritizes_l0_groups_over_earlier_unclassified_lines():
    lines = (
        "NCCL configuration detail",
        "[rank7] NCCL watchdog caught collective operation timeout",
    )
    group = NormalizedOccurrenceGroup(
        occurrence_group_id="og-1",
        normalized_shape=normalized_pattern(lines[1]),
        first_line=2,
        count=1,
        sample_lines=(2,),
        rank_spread=("7",),
        registry_id="observed_distributed_operation_timeout",
        classification="error",
    )
    bundle = L0Bundle(
        log_path="/not/read.log",
        byte_size=sum(len(line) + 1 for line in lines),
        line_count=len(lines),
        occurrence_groups=(group,),
    )

    result = LogTools(
        bundle,
        LogSnapshot(path=bundle.log_path, lines=lines, byte_size=bundle.byte_size),
    ).grep_log("NCCL", max_matches=1)

    assert result["total_match_groups"] == 2
    assert result["samples_truncated"] is True
    assert result["matches"][0]["occurrence_group_id"] == "og-1"


def test_compact_grep_does_not_merge_distinct_distributed_incidents():
    lines = tuple(f"[rank{rank}] RuntimeError: collective timeout" for rank in range(4))
    incidents = tuple(
        DistributedFailureIncident(
            incident_id=f"di-{index + 1}",
            incident_kind="distributed_fanout",
            incident_type="distributed_exception_fanout",
            status="terminal",
            first_observed_line=(index * 2) + 1,
            last_observed_line=(index * 2) + 2,
            primary_observed_line=(index * 2) + 1,
            primary_observed_quote=lines[index * 2],
            member_event_lines=((index * 2) + 1, (index * 2) + 2),
            event_count=2,
            observed_rank_count=2,
        )
        for index in range(2)
    )
    bundle = L0Bundle(
        log_path="/not/read.log",
        byte_size=sum(len(line) + 1 for line in lines),
        line_count=len(lines),
        distributed_failure_incidents=incidents,
    )

    result = LogTools(
        bundle,
        LogSnapshot(path=bundle.log_path, lines=lines, byte_size=bundle.byte_size),
    ).grep_log("RuntimeError")

    assert result["total_raw_matches"] == 4
    assert result["total_match_groups"] == 2
    assert [item["incident_id"] for item in result["matches"]] == ["di-1", "di-2"]
    assert [item["occurrence_count"] for item in result["matches"]] == [2, 2]


def test_symmetric_read_window_limit_includes_the_center_line():
    lines = tuple(f"line {index}" for index in range(1, 302))

    result, arguments, unsupported = execute_tool_request(
        _tools(lines),
        name="read_window",
        raw_arguments={"center_line": 151, "before": 120, "after": 120},
        advertised_tools=DEFAULT_ADVERTISED_TOOLS,
    )

    assert result["status"] == "ok"
    assert result["data"]["start_line"] == 31
    assert result["data"]["end_line"] == 271
    assert len(result["data"]["lines"]) == 241
    assert result["limits"]["max_lines"] == 241
    assert arguments == {"center_line": 151, "before": 120, "after": 120}
    assert unsupported is False


def test_malformed_json_and_type_coercion_are_rejected_with_closed_codes():
    malformed, _, _ = _execute("grep_log", "{")
    wrong_type, _, _ = _execute(
        "grep_log",
        {"pattern": "failure", "ignore_case": "false"},
    )
    invalid_result_mode, _, _ = _execute(
        "grep_log",
        {"pattern": "failure", "result_mode": "summarize"},
    )

    assert malformed["status"] == "error"
    assert malformed["error"]["code"] == "malformed_arguments_json"
    assert wrong_type["error"] == {
        "code": "invalid_arguments",
        "field": "ignore_case",
        "message": "ignore_case must be a boolean.",
    }
    assert invalid_result_mode["error"] == {
        "code": "invalid_arguments",
        "field": "result_mode",
        "message": "result_mode must be compact or raw.",
    }


def test_invalid_regex_and_out_of_range_line_are_rejected_before_execution():
    invalid_regex, _, _ = _execute("grep_log", {"pattern": "["})
    out_of_range, _, _ = _execute("read_window", {"center_line": 4})

    assert invalid_regex["error"]["code"] == "invalid_regex"
    assert out_of_range["error"]["code"] == "line_out_of_range"
    assert out_of_range["error"]["field"] == "center_line"


def test_every_registry_entry_enforces_its_required_success_shape():
    calls = (
        ("overview", {}),
        ("grep_log", {"pattern": "failure"}),
        ("read_window", {"center_line": 2}),
        ("get_evidence_objects", {"refs": ["missing"]}),
    )

    for name, arguments in calls:
        result, _, _ = _execute(
            name,
            arguments,
            advertised=tuple(L1_TOOL_CONTRACTS),
        )
        assert set(result) == {
            "schema_version",
            "tool",
            "status",
            "data",
            "error",
            "truncated",
            "limits",
        }
        assert result["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
        assert result["tool"] == name
        assert result["status"] == "ok", (name, result)
        assert set(L1_TOOL_CONTRACTS[name].result_required_fields).issubset(result["data"])


def test_unknown_arguments_are_rejected_instead_of_ignored():
    result, _, _ = _execute(
        "overview",
        {"surprise": True},
        advertised=tuple(L1_TOOL_CONTRACTS),
    )

    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["field"] == "surprise"


def test_unadvertised_tool_uses_the_same_error_envelope():
    result, _, unsupported = _execute(
        "get_evidence_objects",
        {"refs": ["missing"]},
        advertised=("grep_log", "read_window"),
    )

    assert result["status"] == "error"
    assert result["error"]["code"] == "tool_not_advertised"
    assert unsupported is True


def test_tool_name_rejection_uses_advertisement_first_precedence():
    unadvertised, _, unsupported = _execute(
        "invented_tool",
        {},
        advertised=DEFAULT_ADVERTISED_TOOLS,
    )
    advertised_but_unimplemented, _, advertised_unsupported = _execute(
        "invented_tool",
        {},
        advertised=(*DEFAULT_ADVERTISED_TOOLS, "invented_tool"),
    )

    assert unadvertised["error"]["code"] == "tool_not_advertised"
    assert unsupported is True
    assert advertised_but_unimplemented["error"]["code"] == "tool_not_implemented"
    assert advertised_unsupported is True


def test_failed_tool_results_do_not_expand_l2_model_visibility(tmp_path):
    log_path = tmp_path / "attempt.log"
    log_path.write_text("iteration 1 completed\nRuntimeError: failure\n", encoding="utf-8")
    bundle = build_l0_bundle(str(log_path))
    model_view = build_l0_model_facing_view(bundle, build_decision_evidence(bundle))
    result = L1EvidenceResult(
        semantic_payload=None,
        model="test-model",
        transcript_events=(
            {
                "event_type": "tool_result",
                "result": {
                    "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                    "tool": "read_window",
                    "status": "error",
                    "data": None,
                    "error": {
                        "code": "line_out_of_range",
                        "field": "center_line",
                        "message": "line 999999 is unavailable",
                    },
                    "truncated": False,
                    "limits": {},
                },
            },
            {
                "event_type": "tool_result",
                "result": {
                    "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                    "tool": "read_window",
                    "status": "ok",
                    "data": {
                        "start_line": 444,
                        "end_line": 444,
                        "lines": [{"line": 444, "text": "visible tool evidence"}],
                        "truncated": False,
                    },
                    "error": None,
                    "truncated": False,
                    "limits": {"max_lines": 241},
                },
            },
        ),
    )

    visible_lines = model_visible_line_numbers(model_view, result)
    visible_texts = model_visible_line_texts(model_view, result)
    assert 999999 not in visible_lines
    assert 999999 not in visible_texts
    assert 444 in visible_lines
    assert visible_texts[444] == {"visible tool evidence"}
