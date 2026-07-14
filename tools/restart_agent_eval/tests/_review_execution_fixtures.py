# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem and process fixtures for review execution scenarios."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from unittest import mock

from _builders import retry_policy
from _mocks import process_result
from restart_agent_eval import product_trace
from restart_agent_eval import review as review_log
from restart_agent_eval.profiles import expand_targets


def collect_all_trace(
    log_path: Path,
    batch_result: dict,
    analyzer_trace: dict,
    *,
    l0_bundle: dict | None = None,
) -> dict:
    return {
        "schema_version": product_trace.COLLECT_ALL_TRACE_SCHEMA,
        "request": {"log_path": str(log_path)},
        "collect_all_result": batch_result,
        "l0_bundle": {"log_path": str(log_path)} if l0_bundle is None else l0_bundle,
        "analyzer_trace": analyzer_trace,
    }


def write_collect_all_batch(
    command,
    *,
    log_path,
    batch_result,
    model_targets,
    stderr="",
):
    trace_path = Path(command[command.index("--trace-json") + 1])
    l0_path = Path(command[command.index("--l0-bundle-json-out") + 1])
    decision_path = Path(command[command.index("--decision-evidence-json-out") + 1])
    model_view_path = Path(command[command.index("--l0-model-view-json-out") + 1])
    trace_path.write_text(
        json.dumps(
            collect_all_trace(
                log_path,
                batch_result,
                {
                    "deterministic": {"analyzer_trace": {"layers": {}}},
                    "model_routes": {
                        target.name: {"analyzer_trace": {"layers": {}}} for target in model_targets
                    },
                },
            )
        ),
        encoding="utf-8",
    )
    for path in (l0_path, decision_path, model_view_path):
        path.write_text("{}\n", encoding="utf-8")
    return process_result(command, stdout=json.dumps(batch_result), stderr=stderr)


def write_route_then_batch(
    command,
    *,
    log_path,
    target,
    analysis_result,
    batch_result,
    on_event,
):
    l0_path = Path(command[command.index("--l0-bundle-json-out") + 1])
    l0_path.write_text(json.dumps({"bundle": {"log_path": str(log_path)}}), encoding="utf-8")
    manifest_path = Path(command[command.index("--route-artifact-manifest") + 1])
    route_paths = json.loads(manifest_path.read_text(encoding="utf-8"))["routes"][target.name]
    route_result_path = Path(route_paths["result_json"])
    route_trace_path = Path(route_paths["trace_json"])
    route_result_path.write_text(json.dumps(analysis_result), encoding="utf-8")
    route_trace_path.write_text(
        json.dumps(
            {
                "schema_version": product_trace.SINGLE_TRACE_SCHEMA,
                "request": {"log_path": str(log_path)},
                "analysis_result": analysis_result,
                "analyzer_trace": {"layers": {}},
                "l0_bundle": {"log_path": str(log_path)},
            }
        ),
        encoding="utf-8",
    )
    on_event(
        {
            "event": "route_completed",
            "route_id": target.name,
            "result_artifact": route_result_path.name,
            "trace_artifact": route_trace_path.name,
        }
    )
    trace_path = Path(command[command.index("--trace-json") + 1])
    trace_path.write_text(
        json.dumps(
            collect_all_trace(
                log_path,
                batch_result,
                {"model_routes": {target.name: {"analyzer_trace": {"layers": {}}}}},
            )
        ),
        encoding="utf-8",
    )
    result_path = Path(command[command.index("--result-json") + 1])
    result_path.write_text(json.dumps(batch_result), encoding="utf-8")
    return process_result(command, stdout=json.dumps(batch_result))


def completed_route_result(target, analysis_result: dict) -> dict:
    return {
        "route_id": target.name,
        "model": target.model,
        "execution_status": "completed",
        "l1_usable": True,
        "analysis_result": analysis_result,
    }


def collect_all_result(analysis_result: dict, model_targets) -> dict:
    return {
        "schema_version": "restart_agent_collect_all.v1",
        "deterministic_result": analysis_result,
        "model_results": [
            completed_route_result(target, analysis_result) for target in model_targets
        ],
        "shared_analysis": {"routing_mode": "collect_all"},
    }


@contextlib.contextmanager
def collect_all_execution(*, replay_l0: bool = False):
    targets = expand_targets(["all"])
    model_targets = [target for target in targets if target.enable_l1]
    args = review_log.parse_review_args(["--log", "/tmp/example.log", "all"])
    analysis_result = {
        "schema_version": "restart_agent_response.v1",
        "decision": "RESTART",
        "decision_basis": "general_retry_available",
        "retry_policy": retry_policy(
            policy_version="retry_budget.v1",
            matching_prior_failures=0,
        ),
    }
    batch_result = collect_all_result(analysis_result, model_targets)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log_path = root / "input.log"
        log_path.write_text("RuntimeError: failure\n", encoding="utf-8")
        replay_bundle = root / "prior_l0_bundle.json"
        if replay_l0:
            replay_bundle.write_text("{}\n", encoding="utf-8")

        def fake_run(*, cmd, **kwargs):
            return write_collect_all_batch(
                cmd,
                log_path=log_path,
                batch_result=batch_result,
                model_targets=model_targets,
                stderr="batch warning",
            )

        with mock.patch.object(
            review_log,
            "_run_process_with_live_events",
            side_effect=fake_run,
        ) as run:
            target_runs = review_log.run_collect_all_targets(
                targets=targets,
                model_targets=model_targets,
                args=args,
                log_path=log_path,
                product_repo=root,
                run_dir=root,
                l0_bundle_in=replay_bundle if replay_l0 else None,
                l0_bundle_out=root / "l0_bundle.json",
                decision_evidence_out=root / "decision_evidence.json",
                l0_model_view_out=root / "l0_model_view.json",
            )
        yield root, targets, model_targets, target_runs, run
