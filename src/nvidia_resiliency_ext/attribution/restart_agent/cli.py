# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entrypoint for local restart-policy analysis."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .agent_runtime import RestartAgentRuntime, build_restart_agent_runtime
from .attempt_records import InMemoryAttemptRecordStore
from .config import load_restart_agent_config
from .infrastructure import (
    DETERMINISTIC_FALLBACK_SCHEMA_VERSION,
    L0ArtifactPublisher,
    LiveArtifactWriter,
    RouteArtifactPublisher,
    load_route_artifact_manifest,
    write_json_atomic,
)
from .l0 import read_l0_bundle
from .l1 import THINKING_MODES, LlmConfig, LlmEvidenceExtractor
from .models import (
    AnalysisResult,
    AttemptRecord,
    CollectAllAnalysisResult,
    RestartAgentRequest,
    normalize_attempt_records,
)
from .observability import CLI_COLLECT_ALL_TRACE_SCHEMA_VERSION, CLI_TRACE_SCHEMA_VERSION
from .pipeline import RestartAgent


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    attempt_records = _load_attempt_records(args.attempt_records_json_in)
    request = RestartAgentRequest(
        log_path=args.log_path,
        job_id=args.job_id,
        cycle_id=args.cycle_id,
    )
    replayed_bundle = (
        read_l0_bundle(args.l0_bundle_json_in, expected_log_path=args.log_path)
        if args.l0_bundle_json_in
        else None
    )
    if args.config:
        return _run_configured(args, request, attempt_records, replayed_bundle)

    runtime = RestartAgentRuntime(
        RestartAgent(evidence_extractor=_evidence_extractor_from_args(args)),
        attempt_record_store=InMemoryAttemptRecordStore(),
    )
    if attempt_records:
        runtime.attempt_record_control.seed(attempt_records)
    l0_publisher = L0ArtifactPublisher(
        bundle_path=args.l0_bundle_json_out,
        decision_evidence_path=args.decision_evidence_json_out,
        model_view_path=args.l0_model_view_json_out,
    )
    try:

        def fallback_callback(candidate: Any) -> None:
            if args.fallback_json_out:
                write_json_atomic(
                    args.fallback_json_out,
                    {
                        "schema_version": DETERMINISTIC_FALLBACK_SCHEMA_VERSION,
                        "candidate": candidate.to_payload(),
                    },
                )

        run = runtime.analyze_one(
            request,
            l0_bundle=replayed_bundle,
            on_l0_ready=l0_publisher.publish,
            on_fallback_ready=fallback_callback,
        )
        l0_publisher.wait()
    finally:
        l0_publisher.close()
    result = run.result
    if args.trace_json:
        _write_trace_json(
            args.trace_json,
            request,
            result,
            run.trace,
            run.bundle,
        )
    if args.result_json:
        write_json_atomic(args.result_json, result.to_payload())
    _write_attempt_records(args.attempt_records_json_out, runtime)
    if args.summary:
        _print_summary(args.trace_json, result, run.trace)
    print(json.dumps(result.to_payload(), sort_keys=True))
    return 0


def _run_configured(
    args: argparse.Namespace,
    request: RestartAgentRequest,
    attempt_records: tuple[AttemptRecord, ...],
    replayed_bundle: Any,
) -> int:
    config = load_restart_agent_config(args.config)
    runtime = build_restart_agent_runtime(config)
    if attempt_records:
        runtime.attempt_record_control.seed(attempt_records)
    model_routes = runtime.model_routes
    route_artifact_publisher = (
        RouteArtifactPublisher(
            load_route_artifact_manifest(
                args.route_artifact_manifest,
                expected_route_ids=[route.route_id for route in model_routes],
            ),
            request=request.to_payload(),
            batch_trace_path=args.trace_json,
        )
        if args.route_artifact_manifest
        else None
    )
    live_writer = (
        LiveArtifactWriter(args.incremental_artifact_dir) if args.incremental_artifact_dir else None
    )
    if live_writer is not None:
        live_writer.start(
            routes=[
                {
                    "route_id": route.route_id,
                    "model": route.model,
                    "endpoint": route.endpoint,
                    "credential_ref": route.credential_ref,
                }
                for route in model_routes
            ],
            config_metadata=config.metadata(),
        )

    def fallback_callback(candidate: Any) -> None:
        candidate_payload = candidate.to_payload()
        if args.fallback_json_out:
            write_json_atomic(
                args.fallback_json_out,
                {
                    "schema_version": DETERMINISTIC_FALLBACK_SCHEMA_VERSION,
                    "candidate": candidate_payload,
                },
            )
        if live_writer is not None:
            live_writer.publish_fallback(
                candidate_payload,
                artifact_path=args.fallback_json_out,
            )

    def route_callback(route_result: Any, route_trace: Any) -> None:
        artifact_paths = (
            route_artifact_publisher.publish(route_result, route_trace)
            if route_artifact_publisher is not None
            else {}
        )
        if live_writer is not None:
            live_writer.publish_route(
                route_result.to_payload(),
                artifact_paths=artifact_paths,
            )

    l0_publisher = L0ArtifactPublisher(
        bundle_path=args.l0_bundle_json_out,
        decision_evidence_path=args.decision_evidence_json_out,
        model_view_path=args.l0_model_view_json_out,
        on_published=(live_writer.publish_l0_artifacts if live_writer is not None else None),
    )

    def l0_callback(artifacts: Any) -> None:
        if route_artifact_publisher is not None:
            route_artifact_publisher.set_l0_artifacts(artifacts)
        l0_publisher.publish(artifacts)

    try:
        run = runtime.analyze_many(
            request,
            model_routes,
            l0_bundle=replayed_bundle,
            max_parallel_models=config.max_parallel_models,
            config_metadata=config.metadata(),
            on_l0_ready=l0_callback,
            on_fallback_ready=fallback_callback,
            on_route_complete=route_callback,
            timeout_seconds=config.timeout_seconds,
        )
        result = run.result
        l0_publisher.wait()
        if args.trace_json:
            _write_collect_all_trace_json(
                args.trace_json,
                request,
                result,
                run.trace,
                run.bundle,
            )
        if args.result_json:
            write_json_atomic(args.result_json, result.to_payload())
        _write_attempt_records(args.attempt_records_json_out, runtime)
        if live_writer is not None:
            live_writer.complete(
                final_artifacts={
                    "result_json": args.result_json,
                    "trace_json": args.trace_json,
                    "l0_bundle_json": args.l0_bundle_json_out,
                    "decision_evidence_json": args.decision_evidence_json_out,
                    "l0_model_view_json": args.l0_model_view_json_out,
                },
            )
        if args.summary:
            _print_collect_all_summary(args.trace_json, result)
        print(json.dumps(result.to_payload(), sort_keys=True))
    except BaseException as exc:
        if live_writer is not None:
            live_writer.fail(exc)
        raise
    finally:
        l0_publisher.close()
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one training log for restart policy.")
    parser.add_argument("log_path")
    parser.add_argument("--job-id")
    parser.add_argument("--cycle-id", type=int)
    parser.add_argument(
        "--attempt-records-json-in",
        help="Seed runtime history from a plain AttemptRecord JSON array",
    )
    parser.add_argument(
        "--attempt-records-json-out",
        help="Write the complete post-analysis AttemptRecord JSON array",
    )
    parser.add_argument(
        "--enable-l1",
        action="store_true",
        help="Run the profile-style L1 LLM evidence extractor for readable non-empty logs",
    )
    parser.add_argument(
        "--config",
        help=(
            "Run the versioned restart-agent configuration in this JSON file; "
            "credential_ref values resolve through environment variables"
        ),
    )
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key-file")
    parser.add_argument("--llm-timeout-seconds", type=float)
    parser.add_argument("--llm-max-output-tokens", type=int)
    parser.add_argument("--llm-context-window-tokens", type=int)
    parser.add_argument("--llm-max-tool-rounds", type=int)
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--llm-top-p", type=float)
    parser.add_argument(
        "--llm-thinking-mode",
        choices=THINKING_MODES,
        help="Provider thinking/reasoning knob policy; auto disables Qwen thinking",
    )
    parser.add_argument("--llm-reasoning-effort")
    parser.add_argument(
        "--disable-l1-tools",
        action="store_true",
        help="Do not advertise read-only L1 log tools to the model",
    )
    parser.add_argument(
        "--trace-json",
        help="Write the request, L0 bundle, and final response to this JSON file",
    )
    parser.add_argument(
        "--result-json",
        help="Write the final single-route or collect-all result to this JSON file",
    )
    parser.add_argument(
        "--fallback-json-out",
        help="Write the deterministic fallback candidate when it becomes ready",
    )
    parser.add_argument(
        "--route-artifact-manifest",
        help="Write completed collect-all routes to caller-declared result/trace paths",
    )
    parser.add_argument(
        "--l0-bundle-json-in",
        help="Reuse a versioned L0 bundle previously built for this unchanged log",
    )
    parser.add_argument(
        "--l0-bundle-json-out",
        help="Write the versioned L0 bundle used by this analysis",
    )
    parser.add_argument(
        "--decision-evidence-json-out",
        help="Write canonical Decision Evidence when shared L0 completes",
    )
    parser.add_argument(
        "--l0-model-view-json-out",
        help="Write the bounded L0B model evidence view when shared L0 completes",
    )
    parser.add_argument(
        "--incremental-artifact-dir",
        help=(
            "Publish collect-all lifecycle status and append-only events while "
            "analysis is running"
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact human-readable analysis summary to stderr",
    )
    args = parser.parse_args(argv)
    if args.incremental_artifact_dir and not args.config:
        parser.error("--incremental-artifact-dir requires --config")
    if args.route_artifact_manifest and not args.config:
        parser.error("--route-artifact-manifest requires --config")
    if args.config:
        conflicting_options = _config_cli_conflicts(args)
        if conflicting_options:
            parser.error(
                "--config owns model-route behavior; remove: " + ", ".join(conflicting_options)
            )
    return args


def _config_cli_conflicts(args: argparse.Namespace) -> list[str]:
    values = {
        "--enable-l1": args.enable_l1,
        "--llm-base-url": args.llm_base_url,
        "--llm-model": args.llm_model,
        "--llm-api-key-file": args.llm_api_key_file,
        "--llm-timeout-seconds": args.llm_timeout_seconds,
        "--llm-max-output-tokens": args.llm_max_output_tokens,
        "--llm-context-window-tokens": args.llm_context_window_tokens,
        "--llm-max-tool-rounds": args.llm_max_tool_rounds,
        "--llm-temperature": args.llm_temperature,
        "--llm-top-p": args.llm_top_p,
        "--llm-thinking-mode": args.llm_thinking_mode,
        "--llm-reasoning-effort": args.llm_reasoning_effort,
        "--disable-l1-tools": args.disable_l1_tools,
    }
    return [option for option, value in values.items() if value not in (None, False)]


def _evidence_extractor_from_args(
    args: argparse.Namespace,
) -> LlmEvidenceExtractor | None:
    if not args.enable_l1:
        return None
    config = LlmConfig.from_env(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key_file=args.llm_api_key_file,
        timeout_seconds=args.llm_timeout_seconds,
        max_output_tokens=args.llm_max_output_tokens,
        context_window_tokens=args.llm_context_window_tokens,
        max_tool_rounds=args.llm_max_tool_rounds,
        tools_enabled=not args.disable_l1_tools,
        thinking_mode=args.llm_thinking_mode,
        reasoning_effort=args.llm_reasoning_effort,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
    )
    return LlmEvidenceExtractor(config)


def _load_attempt_records(path: str | None) -> tuple[AttemptRecord, ...]:
    if path is None:
        return ()
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise TypeError("attempt-record fixture must be a JSON array")
    return normalize_attempt_records(value)


def _write_attempt_records(path: str | None, runtime: RestartAgentRuntime) -> None:
    if path is None:
        return
    write_json_atomic(
        path,
        [record.to_payload() for record in runtime.attempt_record_control.records()],
    )


def _write_trace_json(
    trace_json: str,
    request: RestartAgentRequest,
    result: AnalysisResult,
    analyzer_trace: dict[str, Any],
    l0_bundle: Any,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": CLI_TRACE_SCHEMA_VERSION,
        "request": request.to_payload(),
        "analysis_result": result.to_payload(),
        "analyzer_trace": _to_jsonable(analyzer_trace),
        "l0_bundle": _to_jsonable(l0_bundle) if l0_bundle is not None else None,
    }
    write_json_atomic(trace_json, payload)


def _write_collect_all_trace_json(
    trace_json: str,
    request: RestartAgentRequest,
    result: CollectAllAnalysisResult,
    analyzer_trace: dict[str, Any],
    l0_bundle: Any,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": CLI_COLLECT_ALL_TRACE_SCHEMA_VERSION,
        "request": request.to_payload(),
        "collect_all_result": result.to_payload(),
        "analyzer_trace": _to_jsonable(analyzer_trace),
        "l0_bundle": _to_jsonable(l0_bundle) if l0_bundle is not None else None,
    }
    write_json_atomic(trace_json, payload)


def _print_collect_all_summary(
    trace_json: str | None,
    result: CollectAllAnalysisResult,
) -> None:
    shared = result.shared_analysis
    print(
        "restart_agent collect_all: "
        f"routes={len(result.model_results)} "
        f"max_parallel={shared.get('max_parallel_models')} "
        f"batch_wall_clock_s={shared.get('batch_wall_clock_s')} "
        f"trace_json={trace_json or '<disabled>'}",
        file=sys.stderr,
    )
    for route_result in result.model_results:
        payload = route_result.analysis_result.to_payload()
        primary = payload.get("primary_failure") or {}
        print(
            "restart_agent collect_all: "
            f"route={route_result.route_id} "
            f"model={route_result.model} "
            f"status={route_result.execution_status} "
            f"l1_usable={route_result.l1_usable} "
            f"decision={payload.get('decision')} "
            f"primary={primary.get('fine_class')}@{primary.get('line')}",
            file=sys.stderr,
        )


def _print_summary(
    trace_json: str | None,
    result: AnalysisResult,
    analyzer_trace: dict[str, Any],
) -> None:
    payload = result.to_payload()
    primary = payload.get("primary_failure") or {}
    retry_policy = payload.get("retry_policy") or {}
    print(
        "restart_agent summary: "
        f"decision={payload.get('decision')} "
        f"basis={payload.get('decision_basis')} "
        f"rule={retry_policy.get('rule')} "
        f"retries={retry_policy.get('matching_prior_failures')}/"
        f"{retry_policy.get('allowed_retries')} "
        f"exhausted={retry_policy.get('retry_budget_exhausted')}",
        file=sys.stderr,
    )
    if primary:
        print(
            "restart_agent summary: "
            f"primary={primary.get('fine_class')} "
            f"policy_class={primary.get('policy_class')} "
            f"line={primary.get('line')} "
            f"fingerprint={primary.get('root_fingerprint')}",
            file=sys.stderr,
        )
    provenance = payload.get("result_provenance") or {}
    if provenance:
        print(
            "restart_agent summary: "
            f"source={provenance.get('evidence_source')} "
            f"model_contribution={provenance.get('model_contribution')} "
            f"quality={provenance.get('result_quality')} "
            f"nvrx_use={provenance.get('nvrx_use')}",
            file=sys.stderr,
        )
    print(
        "restart_agent summary: "
        f"secondary_failures={len(payload.get('secondary_failures') or [])} "
        f"cascades={len(payload.get('cascades') or [])} "
        f"trace_json={trace_json or '<disabled>'}",
        file=sys.stderr,
    )
    timing = (analyzer_trace.get("timing") or {}) if analyzer_trace else {}
    if timing:
        print(
            "restart_agent summary: "
            f"timing_total_s={timing.get('total_wall_clock_s')} "
            f"l0_s={timing.get('l0_wall_clock_s')} "
            f"l0a_s={timing.get('l0a_wall_clock_s')} "
            f"l0b_s={timing.get('l0b_wall_clock_s')} "
            f"l1_s={timing.get('l1_wall_clock_s')} "
            f"l1_model_s={timing.get('l1_model_call_wall_clock_s')} "
            f"l1_tool_s={timing.get('l1_tool_wall_clock_s')}",
            file=sys.stderr,
        )
    token_usage = (analyzer_trace.get("token_usage") or {}) if analyzer_trace else {}
    if token_usage:
        print(
            "restart_agent summary: "
            f"tokens_total={token_usage.get('total_tokens')} "
            f"prompt={token_usage.get('prompt_tokens')} "
            f"completion={token_usage.get('completion_tokens')} "
            f"reasoning={token_usage.get('reasoning_tokens')} "
            f"cached_prompt={token_usage.get('cached_prompt_tokens')}",
            file=sys.stderr,
        )
    token_limit = (analyzer_trace.get("token_limit") or {}) if analyzer_trace else {}
    if token_limit:
        print(
            "restart_agent summary: "
            f"token_limit_hit={token_limit.get('hit')} "
            f"token_limit_hit_count={token_limit.get('hit_count')}",
            file=sys.stderr,
        )
    l1 = (analyzer_trace.get("l1") or {}) if analyzer_trace else {}
    l2 = (analyzer_trace.get("l2_audit") or {}) if analyzer_trace else {}
    if l1.get("enabled"):
        print(
            "restart_agent summary: "
            f"l1_model={l1.get('model')} "
            f"l1_response_parsed={l1.get('success')} "
            f"l2_audit={l2.get('audit_status')} "
            f"model_calls={len(l1.get('model_calls') or [])} "
            f"tool_calls={len(l1.get('tool_calls') or [])}",
            file=sys.stderr,
        )
        if l1.get("errors"):
            print(
                "restart_agent summary: "
                f"l1_errors={json.dumps(l1.get('errors'), sort_keys=True)}",
                file=sys.stderr,
            )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, AnalysisResult):
        return value.to_payload()
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
