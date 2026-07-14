#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run product restart-policy analysis for one log and emit review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifact_io import LOCAL_ARTIFACT_STORE, ArtifactStore
from .artifact_io import read_json as _read_json
from .artifact_io import write_json as _write_json
from .artifact_io import write_text_atomic as _write_text_atomic
from .artifacts import ArtifactLayout, resolve_artifact_layout
from .paths import REPO_ROOT, path_from_env, product_repo_from_env
from .product_contract import collect_all_config, route_artifact_manifest
from .product_process import (
    ProcessExecutor,
    ProcessResult,
    ProcessTimeoutError,
    SubprocessExecutor,
    product_cli_command,
)
from .product_trace import SINGLE_TRACE_SCHEMA, ProductTrace, decision_candidate_result
from .profiles import DEFAULT_TOOL_ADVERTISEMENT, PRIMARY_KEY_ENV, RunTarget, expand_targets
from .repository_identity import git_identity as _git_identity
from .review_context import ReviewContext
from .review_markdown import write_review_markdown
from .runtime import SYSTEM_CLOCK, SYSTEM_SLEEPER, Clock, Sleeper
from .schemas import REVIEW_SUMMARY_SCHEMA_VERSION
from .scoring import _int_or_none, _line_in_failure_episode, _list_of_dicts
from .scoring import distributed_incident_summaries as summarize_distributed_incidents
from .scoring import l0_bundle_kpis as build_l0_bundle_kpis
from .scoring import l1_kpis as build_l1_kpis
from .scoring import l2_kpis as build_l2_kpis
from .scoring import l3_kpis as build_l3_kpis
from .scoring import l4_kpis as build_l4_kpis
from .scoring import line_numbering_summary
from .scoring import model_call_summary as summarize_model_calls
from .scoring import model_selection_signals as build_model_selection_signals
from .scoring import (
    path_redaction_audit,
    read_gold_label,
    score_against_gold,
    semantic_safety_summary,
    tool_efficiency_summary,
)

DEFAULT_RUN_TARGETS = ("deterministic",)


def main(argv: list[str] | None = None) -> int:
    environment = dict(os.environ)
    args = parse_review_args(argv, environ=environment)
    return ReviewApplication(environment=environment).run(args)


@dataclass(frozen=True)
class ReviewApplication:
    """Composition root for one-log review execution."""

    environment: Mapping[str, str]
    process_executor: ProcessExecutor = SubprocessExecutor()
    clock: Clock = SYSTEM_CLOCK
    sleeper: Sleeper = SYSTEM_SLEEPER
    artifact_store: ArtifactStore = LOCAL_ARTIFACT_STORE

    def run(self, args: argparse.Namespace) -> int:
        return _run_review(
            args,
            process_executor=self.process_executor,
            environment=self.environment,
            clock=self.clock,
            sleeper=self.sleeper,
            artifact_store=self.artifact_store,
        )


def _run_review(
    args: argparse.Namespace,
    *,
    process_executor: ProcessExecutor,
    environment: Mapping[str, str],
    clock: Clock,
    sleeper: Sleeper,
    artifact_store: ArtifactStore,
) -> int:
    log_path = args.log.expanduser()
    if not log_path.is_absolute():
        raise SystemExit(f"--log must be absolute: {log_path}")
    if not log_path.is_file():
        raise SystemExit(f"--log is not a readable file: {log_path}")

    l0_bundle_in = None
    if args.l0_bundle_json_in is not None:
        l0_bundle_in = args.l0_bundle_json_in.expanduser().resolve()
        if not l0_bundle_in.is_file():
            raise SystemExit(f"--l0-bundle-json-in is not a readable file: {l0_bundle_in}")

    product_repo = args.product_repo.expanduser().resolve()
    package_dir = product_repo / "src" / "nvidia_resiliency_ext" / "attribution" / "restart_agent"
    if not package_dir.is_dir():
        raise SystemExit(f"--product-repo does not look like an NVRx product repo: {product_repo}")

    layout = resolve_artifact_layout(
        log_path=log_path,
        log_root=args.log_root,
        gold_root=args.gold_root,
        run_root=args.run_root,
        run_dir=args.run_dir,
        gold_label=args.gold_label,
        clock=clock,
    )
    run_dir = layout.run_dir
    run_dir.mkdir(parents=True, exist_ok=args.run_dir is not None)
    print(f"run directory: {run_dir}", flush=True)
    gold_path = layout.gold_path
    gold_label = (
        read_gold_label(gold_path, source_log=log_path)
        if gold_path is not None and gold_path.is_file()
        else None
    )

    targets = expand_targets(args.targets or list(DEFAULT_RUN_TARGETS))
    validate_model_environment(targets, environment=environment)
    summaries: list[dict[str, Any]] = []
    completed_summaries: dict[str, dict[str, Any]] = {}
    max_exit = 0
    shared_l0_bundle = run_dir / "l0_bundle.json"
    shared_decision_evidence = run_dir / "decision_evidence.json"
    shared_l0_model_view = run_dir / "l0_model_view.json"
    model_targets = [target for target in targets if target.enable_l1]
    live_artifact_dir = run_dir / "live" if model_targets else None

    def publish_target_review(
        target: RunTarget,
        completed: ProcessResult,
        paths: dict[str, Path],
    ) -> None:
        summary = write_review_summary(
            target=target,
            completed=completed,
            paths=paths,
            source_log=log_path,
            gold_label=gold_label,
            effective_tool_profile=effective_tool_profile(target, args),
            artifact_store=artifact_store,
        )
        completed_summaries[target.name] = summary
        _print_run_summary(summary, paths)

    if model_targets:
        target_runs = run_collect_all_targets(
            targets=targets,
            model_targets=model_targets,
            args=args,
            log_path=log_path,
            product_repo=product_repo,
            run_dir=run_dir,
            l0_bundle_in=l0_bundle_in,
            l0_bundle_out=shared_l0_bundle,
            decision_evidence_out=shared_decision_evidence,
            l0_model_view_out=shared_l0_model_view,
            on_target_ready=publish_target_review,
            process_executor=process_executor,
            environment=environment,
            sleeper=sleeper,
        )
    else:
        target_runs = {}
        for target in targets:
            target_runs[target.name] = run_target(
                target=target,
                args=args,
                log_path=log_path,
                product_repo=product_repo,
                run_dir=run_dir,
                l0_bundle_in=l0_bundle_in,
                l0_bundle_out=shared_l0_bundle,
                decision_evidence_out=shared_decision_evidence,
                l0_model_view_out=shared_l0_model_view,
                process_executor=process_executor,
                environment=environment,
            )

    for target in targets:
        completed, paths = target_runs[target.name]
        max_exit = max(max_exit, completed.returncode)
        summary = completed_summaries.get(target.name)
        if summary is None:
            publish_target_review(target, completed, paths)
            summary = completed_summaries[target.name]
        summaries.append(summary)

    write_review_index(
        run_dir,
        log_path,
        product_repo,
        summaries,
        layout=layout,
        shared_l0_bundle=shared_l0_bundle,
        shared_decision_evidence=shared_decision_evidence,
        shared_l0_model_view=shared_l0_model_view,
        gold_label_path=gold_path if gold_label is not None else None,
        live_artifact_dir=live_artifact_dir,
        clock=clock,
        artifact_store=artifact_store,
    )
    _write_panel_summary(run_dir, summaries)
    print()
    print(f"all review artifacts: {run_dir}")
    return max_exit


def parse_review_args(
    argv: list[str] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    """Parse one-log review options using an explicit environment."""

    environment = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(
        description=(
            "Run the product restart agent for one log and write " "review-mode artifacts."
        )
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help=(
            "deterministic, configured, qwen, qwen235b, qwen397b, nemotron, gpt, "
            "claude, gemini, models, or all; configured uses NVRX_LLM_MODEL "
            "and models excludes qwen by default"
        ),
    )
    parser.add_argument("--log", type=Path, required=True, help="absolute log path to analyze")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="explicit artifact directory; bypasses mirrored run-root placement",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=path_from_env("RESTART_AGENT_EVAL_LOG_ROOT", environment),
        help="source corpus root; defaults to RESTART_AGENT_EVAL_LOG_ROOT",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=path_from_env("RESTART_AGENT_EVAL_GOLD_ROOT", environment),
        help="durable gold root; defaults to RESTART_AGENT_EVAL_GOLD_ROOT",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=path_from_env("RESTART_AGENT_EVAL_RUN_ROOT", environment),
        help="disposable generated-run root; defaults to RESTART_AGENT_EVAL_RUN_ROOT",
    )
    parser.add_argument(
        "--gold-label",
        type=Path,
        help=(
            "explicit human-approved label; otherwise resolves to "
            "<gold-root>/<relative-log-path>/gold.json"
        ),
    )
    parser.add_argument(
        "--l0-bundle-json-in",
        type=Path,
        help=(
            "reuse a versioned L0 bundle previously built for this unchanged log; "
            "the product validates schema, path, size, and mtime"
        ),
    )
    parser.add_argument(
        "--product-repo",
        type=Path,
        default=product_repo_from_env(environment),
        help=(
            "NVRx checkout containing restart_agent; defaults to "
            "NVRX_RESTART_AGENT_PRODUCT_REPO or the containing checkout"
        ),
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable for product CLI"
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible LLM base URL; defaults to product environment",
    )
    parser.add_argument("--llm-timeout-seconds", type=float)
    parser.add_argument("--llm-max-output-tokens", type=int)
    parser.add_argument("--llm-context-window-tokens", type=int)
    parser.add_argument("--llm-max-tool-rounds", type=int)
    parser.add_argument("--llm-thinking-mode", choices=("auto", "disable", "allow"))
    parser.add_argument("--llm-reasoning-effort")
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--llm-top-p", type=float)
    parser.add_argument("--disable-l1-tools", action="store_true")
    return parser.parse_args(argv)


def artifact_label(target: RunTarget) -> str:
    """Return the stable artifact prefix for a review target."""

    if target.model is None:
        return "model.configured" if target.enable_l1 else target.name
    model_token = target.model.replace("/", "_")
    return f"model.{model_token}"


def _target_paths(
    target: RunTarget,
    run_dir: Path,
) -> dict[str, Path]:
    label = artifact_label(target)
    paths = {
        "result_json": run_dir / f"{label}.result.json",
        "trace_json": run_dir / f"{label}.trace.json",
        "review_json": run_dir / f"{label}.review.json",
        "review_md": run_dir / f"{label}.review.md",
    }
    return paths


def run_target(
    *,
    target: RunTarget,
    args: argparse.Namespace,
    log_path: Path,
    product_repo: Path,
    run_dir: Path,
    l0_bundle_in: Path | None = None,
    l0_bundle_out: Path | None = None,
    decision_evidence_out: Path | None = None,
    l0_model_view_out: Path | None = None,
    process_executor: ProcessExecutor | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[ProcessResult, dict[str, Path]]:
    """Execute one review target and publish its raw product artifacts."""

    paths = _target_paths(target, run_dir)

    cmd = [
        *product_cli_command(args.python, log_path),
        "--trace-json",
        str(paths["trace_json"]),
    ]
    if l0_bundle_in is not None:
        cmd.extend(["--l0-bundle-json-in", str(l0_bundle_in)])
    if l0_bundle_out is not None:
        cmd.extend(["--l0-bundle-json-out", str(l0_bundle_out)])
    if decision_evidence_out is not None:
        cmd.extend(["--decision-evidence-json-out", str(decision_evidence_out)])
    if l0_model_view_out is not None:
        cmd.extend(["--l0-model-view-json-out", str(l0_model_view_out)])
    if target.enable_l1:
        cmd.append("--enable-l1")
        if target.model is not None:
            cmd.extend(["--llm-model", target.model])
        _append_optional(cmd, "--llm-base-url", args.base_url)
        _append_optional(cmd, "--llm-timeout-seconds", args.llm_timeout_seconds)
        _append_optional(
            cmd,
            "--llm-max-output-tokens",
            (
                args.llm_max_output_tokens
                if args.llm_max_output_tokens is not None
                else target.max_output_tokens
            ),
        )
        _append_optional(
            cmd,
            "--llm-context-window-tokens",
            args.llm_context_window_tokens,
        )
        _append_optional(
            cmd,
            "--llm-max-tool-rounds",
            effective_tool_profile(target, args).get("max_tool_rounds"),
        )
        _append_optional(cmd, "--llm-thinking-mode", args.llm_thinking_mode)
        _append_optional(cmd, "--llm-reasoning-effort", args.llm_reasoning_effort)
        _append_optional(cmd, "--llm-temperature", args.llm_temperature)
        _append_optional(cmd, "--llm-top-p", args.llm_top_p)
        if args.disable_l1_tools:
            cmd.append("--disable-l1-tools")

    env = dict(os.environ if environment is None else environment)
    if target.enable_l1:
        credential_env = target.credential_env or PRIMARY_KEY_ENV
        env[PRIMARY_KEY_ENV] = env[credential_env]
    env["PYTHONPATH"] = _prepend_path("src", env.get("PYTHONPATH"))
    env.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/nvrx-restart-agent-pycache")

    executor = process_executor or SubprocessExecutor()
    completed = executor.run(
        cmd,
        cwd=product_repo,
        env=env,
    )
    stderr_path = run_dir / f"{artifact_label(target)}.stderr.log"
    if persist_process_diagnostics(completed, stderr_path):
        paths["stderr_log"] = stderr_path
    _write_pretty_json_from_text(completed.stdout, paths["result_json"])
    return completed, paths


def run_collect_all_targets(
    *,
    targets: list[RunTarget],
    model_targets: list[RunTarget],
    args: argparse.Namespace,
    log_path: Path,
    product_repo: Path,
    run_dir: Path,
    l0_bundle_in: Path | None = None,
    l0_bundle_out: Path,
    decision_evidence_out: Path | None = None,
    l0_model_view_out: Path | None = None,
    on_target_ready: (
        Callable[
            [RunTarget, ProcessResult, dict[str, Path]],
            None,
        ]
        | None
    ) = None,
    process_executor: ProcessExecutor | None = None,
    environment: Mapping[str, str] | None = None,
    sleeper: Sleeper = SYSTEM_SLEEPER,
) -> dict[str, tuple[ProcessResult, dict[str, Path]]]:
    """Execute the product's shared-L0 multi-route review path."""

    invocation = _prepare_collect_all_invocation(
        model_targets=model_targets,
        args=args,
        log_path=log_path,
        run_dir=run_dir,
        l0_bundle_in=l0_bundle_in,
        l0_bundle_out=l0_bundle_out,
        decision_evidence_out=decision_evidence_out,
        l0_model_view_out=l0_model_view_out,
        environment=environment,
    )
    collector = _PublishedRouteCollector(
        targets_by_route={target.name: target for target in model_targets},
        run_dir=run_dir,
        on_target_ready=on_target_ready,
    )
    completed = _run_process_with_live_events(
        cmd=invocation.command,
        cwd=product_repo,
        env=invocation.environment,
        events_path=invocation.live_artifact_dir / "events.jsonl",
        on_event=collector.on_event,
        process_executor=process_executor,
        sleeper=sleeper,
    )
    persist_process_diagnostics(completed, run_dir / "restart_agent.stderr.log")
    if not invocation.batch_paths["result_json"].is_file():
        _write_pretty_json_from_text(completed.stdout, invocation.batch_paths["result_json"])
    if completed.returncode != 0:
        detail = completed.stderr or completed.stdout
        raise RuntimeError(
            f"collect-all product invocation failed with exit {completed.returncode}: "
            f"{detail[-2000:]}"
        )

    return _materialize_collect_all_routes(
        targets=targets,
        run_dir=run_dir,
        completed=completed,
        batch_paths=invocation.batch_paths,
        existing_runs=collector.target_runs,
        on_target_ready=on_target_ready,
    )


TargetRun = tuple[ProcessResult, dict[str, Path]]


@dataclass(frozen=True)
class _CollectAllInvocation:
    command: list[str]
    environment: dict[str, str]
    batch_paths: dict[str, Path]
    live_artifact_dir: Path


def _prepare_collect_all_invocation(
    *,
    model_targets: list[RunTarget],
    args: argparse.Namespace,
    log_path: Path,
    run_dir: Path,
    l0_bundle_in: Path | None,
    l0_bundle_out: Path,
    decision_evidence_out: Path | None,
    l0_model_view_out: Path | None,
    environment: Mapping[str, str] | None = None,
) -> _CollectAllInvocation:
    route_config_path = run_dir / "restart_agent.json"
    route_artifact_manifest_path = run_dir / "restart_agent_route_artifacts.json"
    batch_paths = {
        "result_json": run_dir / "restart_agent.result.json",
        "trace_json": run_dir / "restart_agent.trace.json",
    }
    live_artifact_dir = run_dir / "live"
    _write_json(
        route_config_path,
        collect_all_config(
            [build_route_payload(target, args, environment=environment) for target in model_targets]
        ),
    )
    _write_json(
        route_artifact_manifest_path,
        route_artifact_manifest(
            {
                target.name: {
                    "result_json": str(_target_paths(target, run_dir)["result_json"]),
                    "trace_json": str(_target_paths(target, run_dir)["trace_json"]),
                }
                for target in model_targets
            }
        ),
    )
    command = [
        *product_cli_command(args.python, log_path),
        "--config",
        str(route_config_path),
        "--trace-json",
        str(batch_paths["trace_json"]),
        "--result-json",
        str(batch_paths["result_json"]),
        "--fallback-json-out",
        str(run_dir / "deterministic_fallback.json"),
        "--route-artifact-manifest",
        str(route_artifact_manifest_path),
        "--incremental-artifact-dir",
        str(live_artifact_dir),
    ]
    if l0_bundle_in is not None:
        command.extend(["--l0-bundle-json-in", str(l0_bundle_in)])
    command.extend(["--l0-bundle-json-out", str(l0_bundle_out)])
    if decision_evidence_out is not None:
        command.extend(["--decision-evidence-json-out", str(decision_evidence_out)])
    if l0_model_view_out is not None:
        command.extend(["--l0-model-view-json-out", str(l0_model_view_out)])
    process_environment = dict(os.environ if environment is None else environment)
    process_environment["PYTHONPATH"] = _prepend_path(
        "src",
        process_environment.get("PYTHONPATH"),
    )
    process_environment.setdefault(
        "PYTHONPYCACHEPREFIX",
        "/private/tmp/nvrx-restart-agent-pycache",
    )
    return _CollectAllInvocation(command, process_environment, batch_paths, live_artifact_dir)


@dataclass
class _PublishedRouteCollector:
    targets_by_route: dict[str, RunTarget]
    run_dir: Path
    on_target_ready: Callable[[RunTarget, ProcessResult, dict[str, Path]], None] | None
    target_runs: dict[str, TargetRun] | None = None

    def __post_init__(self) -> None:
        self.target_runs = {} if self.target_runs is None else self.target_runs

    def on_event(self, event: dict[str, Any]) -> None:
        if str(event.get("event") or "") != "route_completed":
            return
        route_id = str(event.get("route_id") or "")
        target = self.targets_by_route.get(route_id)
        if target is None or route_id in self.target_runs:
            return
        try:
            completed_route, paths = _load_published_route(
                target=target,
                event=event,
                run_dir=self.run_dir,
            )
            if self.on_target_ready is not None:
                self.on_target_ready(target, completed_route, paths)
            self.target_runs[route_id] = (completed_route, paths)
        except Exception as exc:
            print(
                f"live: route {route_id} artifact publication deferred: {exc}",
                file=sys.stderr,
                flush=True,
            )


def _materialize_collect_all_routes(
    *,
    targets: list[RunTarget],
    run_dir: Path,
    completed: ProcessResult,
    batch_paths: dict[str, Path],
    existing_runs: dict[str, TargetRun] | None,
    on_target_ready: Callable[[RunTarget, ProcessResult, dict[str, Path]], None] | None,
) -> dict[str, TargetRun]:
    target_runs = dict(existing_runs or {})
    batch_result = _read_json(batch_paths["result_json"])
    batch_trace = ProductTrace.from_payload(_read_json(batch_paths["trace_json"]))
    model_results = {
        str(item.get("route_id")): item
        for item in batch_result.get("model_results") or []
        if isinstance(item, dict) and item.get("route_id")
    }
    analyzer_trace = batch_trace.analyzer_trace
    route_traces = analyzer_trace.get("model_routes") or {}
    deterministic_trace = analyzer_trace.get("deterministic") or {}
    for target in targets:
        if target.name in target_runs:
            continue
        if target.enable_l1:
            route_result = model_results.get(target.name)
            route_trace = route_traces.get(target.name)
            if not isinstance(route_result, dict) or not isinstance(route_trace, dict):
                raise RuntimeError(f"collect-all result is missing route {target.name!r}")
            analysis_result = route_result.get("analysis_result") or {}
            single_analyzer_trace = route_trace.get("analyzer_trace") or {}
            route_status = str(route_result.get("execution_status") or "unknown")
        else:
            analysis_result = batch_result.get("deterministic_result") or {}
            single_analyzer_trace = deterministic_trace.get("analyzer_trace") or {}
            route_status = "deterministic"
        paths = _target_paths(target, run_dir)
        result_text = json.dumps(analysis_result, sort_keys=True) + "\n"
        _write_json(paths["result_json"], analysis_result)
        _write_json(
            paths["trace_json"],
            {
                "schema_version": SINGLE_TRACE_SCHEMA,
                "request": dict(batch_trace.request),
                "analysis_result": analysis_result,
                "analyzer_trace": single_analyzer_trace,
                "l0_bundle": dict(batch_trace.l0_bundle or {}),
                "collect_all_context": {
                    "shared_analysis": batch_result.get("shared_analysis"),
                    "route_id": target.name,
                    "execution_status": route_status,
                    "batch_trace": str(batch_paths["trace_json"]),
                },
            },
        )
        target_runs[target.name] = (
            ProcessResult(
                command=completed.command,
                returncode=completed.returncode,
                stdout=result_text,
                stderr=completed.stderr,
            ),
            paths,
        )
        if target.enable_l1 and on_target_ready is not None:
            on_target_ready(target, target_runs[target.name][0], paths)
    return target_runs


def _load_published_route(
    *,
    target: RunTarget,
    event: dict[str, Any],
    run_dir: Path,
) -> tuple[ProcessResult, dict[str, Path]]:
    """Validate and load canonical route files written directly by the product."""

    paths = _target_paths(target, run_dir)
    event_result_path = _resolved_published_artifact(run_dir, event.get("result_artifact"))
    event_trace_path = _resolved_published_artifact(run_dir, event.get("trace_artifact"))
    if event_result_path != paths["result_json"].resolve():
        raise ValueError(f"route result path does not match its manifest: {event_result_path}")
    if event_trace_path != paths["trace_json"].resolve():
        raise ValueError(f"route trace path does not match its manifest: {event_trace_path}")
    analysis_result = _read_json(paths["result_json"])
    ProductTrace.from_payload(_read_json(paths["trace_json"]))
    if not isinstance(analysis_result, dict):
        raise ValueError(f"route result is malformed: {paths['result_json']}")
    result_text = json.dumps(analysis_result, sort_keys=True) + "\n"
    return (
        ProcessResult(
            command=("restart-agent", target.name),
            returncode=0,
            stdout=result_text,
            stderr="",
        ),
        paths,
    )


def _resolved_published_artifact(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("route completion event is missing an artifact path")
    resolved_root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"published artifact escapes its run directory: {relative_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"published artifact is not ready: {candidate}")
    return candidate


def _run_process_with_live_events(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    events_path: Path,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    process_executor: ProcessExecutor | None = None,
    sleeper: Sleeper = SYSTEM_SLEEPER,
) -> ProcessResult:
    event_offset = 0
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_handle:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_handle:
            executor = process_executor or SubprocessExecutor()
            process = executor.start(
                cmd,
                cwd=cwd,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            try:
                while process.poll() is None:
                    event_offset = _print_new_live_events(
                        events_path,
                        event_offset,
                        on_event=on_event,
                    )
                    sleeper.sleep(0.1)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except ProcessTimeoutError:
                    process.kill()
                    process.wait()
                raise
            event_offset = _print_new_live_events(
                events_path,
                event_offset,
                on_event=on_event,
            )
            stdout_handle.seek(0)
            stderr_handle.seek(0)
            return ProcessResult(
                command=tuple(cmd),
                returncode=returncode,
                stdout=stdout_handle.read(),
                stderr=stderr_handle.read(),
            )


def _print_new_live_events(
    path: Path,
    offset: int,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    batch = read_live_progress(path, offset)
    for item in batch.items:
        print(item.message, flush=True)
        if on_event is not None and item.event is not None:
            on_event(item.event)
    return batch.next_offset


@dataclass(frozen=True)
class LiveProgressItem:
    """One complete lifecycle event and its user-visible representation."""

    message: str
    event: dict[str, Any] | None


@dataclass(frozen=True)
class LiveProgressBatch:
    """New complete lifecycle events read after a byte offset."""

    next_offset: int
    items: tuple[LiveProgressItem, ...]


def read_live_progress(path: Path, offset: int) -> LiveProgressBatch:
    """Read complete lifecycle events without performing console I/O."""

    if not path.is_file():
        return LiveProgressBatch(next_offset=offset, items=())
    items: list[LiveProgressItem] = []
    with path.open("rb") as handle:
        if handle.seek(0, os.SEEK_END) < offset:
            offset = 0
        handle.seek(offset)
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.endswith(b"\n"):
                return LiveProgressBatch(next_offset=line_start, items=tuple(items))
            offset = handle.tell()
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                items.append(LiveProgressItem("live: malformed lifecycle event", None))
                continue
            items.append(LiveProgressItem(format_live_event(event), event))
    return LiveProgressBatch(next_offset=offset, items=tuple(items))


def format_live_event(event: dict[str, Any]) -> str:
    """Format one lifecycle event for the review CLI."""
    event_name = str(event.get("event") or "unknown")
    elapsed = event.get("elapsed_s")
    suffix = f" t={elapsed}s" if elapsed is not None else ""
    if event_name == "run_started":
        return f"live: analysis started routes={event.get('route_count')}{suffix}"
    if event_name == "deterministic_fallback_ready":
        return "live: deterministic fallback ready " f"decision={event.get('decision')}{suffix}"
    if event_name == "l0_artifacts_ready":
        return "live: L0 artifacts ready " f"l0={event.get('l0_wall_clock_s')}s{suffix}"
    if event_name == "route_completed":
        return (
            f"live: route {event.get('route_id')} "
            f"status={event.get('execution_status')} "
            f"decision={event.get('decision')}{suffix}"
        )
    if event_name == "run_completed":
        return (
            "live: analysis completed "
            f"routes={event.get('completed_routes')}/{event.get('total_routes')}{suffix}"
        )
    if event_name == "run_failed":
        return f"live: analysis failed error={event.get('error')}{suffix}"
    return f"live: {event_name}{suffix}"


def build_route_payload(
    target: RunTarget,
    args: argparse.Namespace,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one product model-route payload from target and CLI options."""

    resolved_environment = os.environ if environment is None else environment
    route: dict[str, Any] = {
        "route_id": target.name,
        "base_url": (
            args.base_url or resolved_environment.get("NVRX_LLM_BASE_URL") or target.endpoint
        ),
        "credential_ref": target.credential_env or PRIMARY_KEY_ENV,
        "tools": {
            "enabled": not args.disable_l1_tools,
            "advertisement": dict(DEFAULT_TOOL_ADVERTISEMENT),
        },
    }
    route.update(
        {key: value for key, value in {"model": target.model}.items() if value is not None}
    )
    request_values = {
        "timeout_seconds": args.llm_timeout_seconds,
        "max_output_tokens": (
            args.llm_max_output_tokens
            if args.llm_max_output_tokens is not None
            else target.max_output_tokens
        ),
        "context_window_tokens": args.llm_context_window_tokens,
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
    }
    if request_values["context_window_tokens"] is None:
        request_values["context_window_tokens"] = target.context_window_tokens
    request = {key: value for key, value in request_values.items() if value is not None}
    if request:
        route["request"] = request

    max_tool_rounds = effective_tool_profile(target, args).get("max_tool_rounds")
    if max_tool_rounds is not None:
        route["tools"]["max_rounds"] = max_tool_rounds

    reasoning = {
        key: value
        for key, value in {
            "thinking_mode": args.llm_thinking_mode,
            "reasoning_effort": args.llm_reasoning_effort,
        }.items()
        if value is not None
    }
    if reasoning:
        route["reasoning"] = reasoning
    return route


def validate_model_environment(
    targets: list[RunTarget],
    *,
    environment: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Validate all credential files required by selected model targets."""

    resolved_environment = os.environ if environment is None else environment
    required_environments = {
        target.credential_env or PRIMARY_KEY_ENV for target in targets if target.enable_l1
    }
    for environment_name in sorted(required_environments):
        key_file = resolved_environment.get(environment_name)
        if not key_file:
            raise SystemExit(f"{environment_name} is required for the selected model targets")
        path = Path(key_file).expanduser()
        if not path.is_file() or not os.access(path, os.R_OK):
            raise SystemExit(f"{environment_name} must name a readable file")
    return frozenset(required_environments)


def _append_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def effective_tool_profile(
    target: RunTarget,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Resolve the effective tool-loop behavior for one target."""

    tools_enabled = target.enable_l1 and not args.disable_l1_tools
    max_tool_rounds = (
        args.llm_max_tool_rounds if args.llm_max_tool_rounds is not None else target.max_tool_rounds
    )
    source = (
        "cli_override"
        if args.llm_max_tool_rounds is not None
        else ("target_profile" if target.max_tool_rounds is not None else "product_default")
    )
    return {
        "profile_id": target.tool_loop_profile_id,
        "experimental": bool(target.tool_loop_profile_id),
        "tools_enabled": tools_enabled,
        "max_tool_rounds": max_tool_rounds,
        "max_model_turns": (
            max_tool_rounds + 1
            if tools_enabled and max_tool_rounds is not None
            else 1 if not tools_enabled else None
        ),
        "source": source,
    }


def _prepend_path(value: str, existing: str | None) -> str:
    return value if not existing else f"{value}{os.pathsep}{existing}"


def _write_pretty_json_from_text(text: str, path: Path) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        _write_text_atomic(path, text)
        return
    _write_json(path, payload)


def write_review_summary(
    *,
    target: RunTarget,
    completed: ProcessResult,
    paths: dict[str, Path],
    source_log: Path,
    gold_label: dict[str, Any] | None = None,
    effective_tool_profile: dict[str, Any] | None = None,
    artifact_store: ArtifactStore = LOCAL_ARTIFACT_STORE,
) -> dict[str, Any]:
    """Build and publish the normalized review artifacts for one route."""

    context = ReviewContext.read(paths, artifact_store=artifact_store)
    summary = _build_review_summary(
        context=context,
        target=target,
        completed=completed,
        paths=paths,
        source_log=source_log,
        gold_label=gold_label,
        effective_tool_profile=effective_tool_profile,
    )
    artifact_store.write_json(paths["review_json"], summary)
    write_review_markdown(paths["review_md"], summary)
    return summary


def _build_review_summary(
    *,
    context: ReviewContext,
    target: RunTarget,
    completed: ProcessResult,
    paths: dict[str, Path],
    source_log: Path,
    gold_label: dict[str, Any] | None,
    effective_tool_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis = context.analysis
    analyzer_trace = context.analyzer_trace
    route_execution_status = context.route_execution_status
    l2_audit = context.l2_audit
    l1 = context.l1
    l1_model_output = context.l1_model_output
    l1_was_run = target.enable_l1 or bool(l1.get("enabled"))
    l1_layer = context.l1_layer
    timing = context.timing
    latency_measurement = context.latency_measurement
    token_usage = context.token_usage
    token_limit = context.token_limit
    l0_bundle = context.l0_bundle
    l0_model_view = context.l0_model_view
    decision_evidence = context.decision_evidence
    l0_deterministic_primary = context.l0_primary
    l1_semantic_primary = context.l1_primary
    l2_grounded_primary = context.l2_primary
    current_failure_facts = context.current_failure_facts
    primary = context.primary
    provenance = context.provenance
    fallback_candidate = context.fallback_candidate
    enriched_candidate = context.enriched_candidate
    fallback_analysis = context.fallback_analysis
    enriched_analysis = context.enriched_analysis
    model_calls = context.model_calls
    tool_calls = context.tool_calls
    interaction_transcript = context.interaction_transcript
    metrics = _build_review_metrics(
        context=context,
        source_log=source_log,
        l1_was_run=l1_was_run,
    )
    leak_audit = metrics.path_redaction_audit
    model_call_summary = metrics.model_call_summary
    tool_efficiency = metrics.tool_efficiency
    semantic_safety = metrics.semantic_safety
    model_selection_signals = metrics.model_selection_signals
    l0_bundle_kpis = metrics.l0_kpis
    l1_kpis = metrics.l1_kpis
    l2_kpis = metrics.l2_kpis
    l3_kpis = metrics.l3_kpis
    l4_kpis = metrics.l4_kpis

    summary = {
        "schema_version": REVIEW_SUMMARY_SCHEMA_VERSION,
        "run_label": artifact_label(target),
        "target": target.name,
        "model": l1.get("model") or target.model,
        "credential_source_env": (target.credential_env if target.enable_l1 else None),
        "effective_tool_profile": effective_tool_profile or {},
        "exit_code": completed.returncode,
        "l1_response_parsed": l1.get("success") if l1.get("enabled") else None,
        "l1_model_output": l1_model_output or None,
        "l2_audit_ran": l2_audit.get("used"),
        "l2_audit_status": l2_audit.get("audit_status"),
        "decision": analysis.get("decision"),
        "decision_basis": analysis.get("decision_basis"),
        "retry_policy": analysis.get("retry_policy") or {},
        "primary_failure": {
            "fine_class": primary.get("fine_class"),
            "policy_class": primary.get("policy_class"),
            "line": primary.get("line"),
            "rank": primary.get("rank"),
            "fault_outcome": primary.get("fault_outcome"),
            "causal_role": primary.get("causal_role"),
            "root_fingerprint": primary.get("root_fingerprint"),
            "root_fingerprint_source": primary.get("root_fingerprint_source"),
            "failure_identity": primary.get("failure_identity"),
        },
        "decision_evidence": decision_evidence,
        "primary_selection_by_stage": {
            "l0_deterministic": _primary_stage_view(l0_deterministic_primary),
            "l1_semantic": _primary_stage_view(l1_semantic_primary),
            "l2_grounded": _primary_stage_view(l2_grounded_primary),
            "l1_relation_to_l0": _primary_selection_relation(
                l0_deterministic_primary,
                l1_semantic_primary,
                l0_bundle,
            ),
            "l2_relation_to_l0": _primary_selection_relation(
                l0_deterministic_primary,
                l2_grounded_primary,
                l0_bundle,
            ),
        },
        "current_failure_facts": current_failure_facts,
        "root_cause_assessment": analysis.get("root_cause_assessment"),
        "model_recovery_assessment": analysis.get("model_recovery_assessment"),
        "secondary_failures": analysis.get("secondary_failures") or [],
        "cascades": analysis.get("cascades") or [],
        "justification": analysis.get("justification"),
        "result_provenance": provenance,
        "decision_paths": {
            "deterministic_fallback": _decision_path_summary(
                fallback_candidate,
                fallback_analysis,
            ),
            "l1_enriched": _decision_path_summary(
                enriched_candidate,
                enriched_analysis,
            ),
        },
        "route_execution_status": route_execution_status,
        "l2_audit": l2_audit,
        "timing": timing,
        "latency_measurement": latency_measurement,
        "token_usage": token_usage,
        "token_limit": token_limit,
        "job_metadata": l0_bundle.get("job_metadata") or {},
        "run_progress_summary": l0_bundle.get("run_progress_summary") or {},
        "later_progress_after_fault_observations": (
            l0_bundle.get("later_progress_after_fault_observations") or []
        ),
        "operation_artifact_comparisons": (l0_bundle.get("operation_artifact_comparisons") or []),
        "distributed_failure_incidents": (summarize_distributed_incidents(l0_bundle)),
        "l0_kpis": l0_bundle_kpis,
        "line_numbering": {
            **line_numbering_summary(source_log),
            **((l0_bundle.get("anomalies") or {}).get("line_numbering") or {}),
        },
        "l1_kpis": l1_kpis,
        "l2_kpis": l2_kpis,
        "l3_kpis": l3_kpis,
        "l4_kpis": l4_kpis,
        "model_calls": len(model_calls),
        "model_call_summary": model_call_summary,
        "tool_calls": len(tool_calls),
        "tool_names": [call.get("name") for call in tool_calls if isinstance(call, dict)],
        "tool_efficiency": tool_efficiency,
        "semantic_safety": semantic_safety,
        "model_selection_signals": model_selection_signals,
        "errors": l1.get("errors") or [],
        "anomalies": l1.get("anomalies") or {},
        "path_redaction_audit": leak_audit,
        "artifacts": {name: str(path) for name, path in paths.items() if path.is_file()},
    }
    summary["gold_score"] = score_against_gold(
        analysis,
        gold_label,
        l0_bundle=l0_bundle,
        l0_model_view=l0_model_view,
        l1_evidence=l1.get("parsed_evidence"),
        l2_grounded_semantics=analyzer_trace.get("l2_grounded_semantics"),
        l2_audit=l2_audit,
        include_l1=l1_was_run,
        fallback_analysis=fallback_analysis,
        enriched_analysis=enriched_analysis,
    )
    return summary


@dataclass(frozen=True)
class _ReviewMetrics:
    path_redaction_audit: dict[str, Any]
    model_call_summary: dict[str, Any]
    tool_efficiency: dict[str, Any]
    semantic_safety: dict[str, Any]
    model_selection_signals: dict[str, Any]
    l0_kpis: dict[str, Any]
    l1_kpis: dict[str, Any]
    l2_kpis: dict[str, Any]
    l3_kpis: dict[str, Any]
    l4_kpis: dict[str, Any]


def _build_review_metrics(
    *,
    context: ReviewContext,
    source_log: Path,
    l1_was_run: bool,
) -> _ReviewMetrics:
    model_call_summary = summarize_model_calls(context.model_calls)
    tool_efficiency = tool_efficiency_summary(
        l1=context.l1,
        timing=context.timing,
        interaction_transcript=context.interaction_transcript,
        analysis=context.analysis,
        l0_bundle=context.l0_bundle,
    )
    semantic_safety = semantic_safety_summary(
        l2_audit=context.l2_audit,
        l4_policy=(context.analyzer_trace.get("l4_policy") or {}),
    )
    selection_signals = build_model_selection_signals(
        model_call_summary=model_call_summary,
        tool_efficiency=tool_efficiency,
        semantic_safety=semantic_safety,
        route_execution_status=context.route_execution_status,
    )
    l1_kpis = build_l1_kpis(
        l1=context.l1,
        l1_layer=context.l1_layer,
        timing=context.timing,
        token_usage=context.token_usage,
        token_limit=context.token_limit,
        model_call_summary=model_call_summary,
        tool_efficiency=tool_efficiency,
        model_selection_signals=selection_signals,
        route_execution_status=context.route_execution_status,
    )
    if not l1_was_run:
        l1_kpis.update(
            {
                "execution_status": "not_run",
                "execution_issues": [],
                "output_status": "not_run",
                "output_usable": False,
            }
        )
    return _ReviewMetrics(
        path_redaction_audit=path_redaction_audit(
            source_log,
            context.interaction_transcript,
        ),
        model_call_summary=model_call_summary,
        tool_efficiency=tool_efficiency,
        semantic_safety=semantic_safety,
        model_selection_signals=selection_signals,
        l0_kpis=build_l0_bundle_kpis(
            analysis=context.analysis,
            l0_bundle=context.l0_bundle,
            l0_model_view=context.l0_model_view,
            timing=context.timing,
            tool_efficiency=tool_efficiency,
        ),
        l1_kpis=l1_kpis,
        l2_kpis=build_l2_kpis(
            l2_audit=context.l2_audit,
            l0_primary=context.l0_primary,
            timing=context.timing,
        ),
        l3_kpis=build_l3_kpis(
            analyzer_trace=context.analyzer_trace,
            timing=context.timing,
        ),
        l4_kpis=build_l4_kpis(
            analysis=context.analysis,
            analyzer_trace=context.analyzer_trace,
            timing=context.timing,
        ),
    )


def _decision_candidate_result(candidate: Any) -> dict[str, Any]:
    return decision_candidate_result(candidate)


def _decision_path_summary(
    candidate: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    candidate = candidate if isinstance(candidate, dict) else {}
    retry_policy = result.get("retry_policy") or {}
    if not isinstance(retry_policy, dict):
        retry_policy = {}
    return {
        "available": bool(result),
        "candidate_kind": candidate.get("candidate_kind"),
        "decision": result.get("decision"),
        "decision_basis": result.get("decision_basis"),
        "retry_rule": retry_policy.get("rule"),
        "allowed_retries": retry_policy.get("allowed_retries"),
        "retry_budget_exhausted": retry_policy.get("retry_budget_exhausted"),
        "ready_wall_clock_s": candidate.get("ready_wall_clock_s"),
        "l1_execution_status": candidate.get("l1_execution_status"),
    }


def _primary_stage_view(primary: dict[str, Any]) -> dict[str, Any]:
    return {
        "fine_class": primary.get("fine_class"),
        "line": _int_or_none(primary.get("line")),
        "policy_class": primary.get("policy_class"),
        "fault_outcome": primary.get("fault_outcome"),
        "causal_role": primary.get("causal_role"),
        "root_fingerprint": primary.get("root_fingerprint"),
        "root_fingerprint_source": primary.get("root_fingerprint_source"),
    }


def _primary_selection_relation(
    l0_primary: dict[str, Any],
    selected_primary: dict[str, Any],
    l0_bundle: dict[str, Any],
) -> str:
    l0_line = _int_or_none(l0_primary.get("line"))
    selected_line = _int_or_none(selected_primary.get("line"))
    if l0_line is None or selected_line is None:
        return "not_available"
    if l0_line == selected_line:
        return "same_line"
    for episode in _list_of_dicts(l0_bundle.get("failure_episodes")):
        if _line_in_failure_episode(l0_line, episode) and _line_in_failure_episode(
            selected_line,
            episode,
        ):
            return "same_failure_episode"
    for incident in _list_of_dicts(l0_bundle.get("distributed_failure_incidents")):
        lines = {
            _int_or_none(incident.get("primary_observed_line")),
            *(_int_or_none(line) for line in incident.get("member_event_lines") or []),
        }
        if l0_line in lines and selected_line in lines:
            return "same_distributed_incident"
    return "different_selection"


def write_review_index(
    run_dir: Path,
    source_log: Path,
    product_repo: Path,
    summaries: list[dict[str, Any]],
    *,
    layout: ArtifactLayout,
    shared_l0_bundle: Path | None = None,
    shared_decision_evidence: Path | None = None,
    shared_l0_model_view: Path | None = None,
    gold_label_path: Path | None = None,
    live_artifact_dir: Path | None = None,
    clock: Clock = SYSTEM_CLOCK,
    artifact_store: ArtifactStore = LOCAL_ARTIFACT_STORE,
) -> None:
    """Publish the run-level index and manifest for completed route reviews."""

    source = {
        "absolute_path": str(source_log.resolve()),
        "relative_path": layout.relative_log_path.as_posix(),
        "sha256": _file_sha256(source_log),
        "byte_size": source_log.stat().st_size,
    }
    repositories = {
        "product": _git_identity(product_repo),
        "harness": _git_identity(REPO_ROOT),
    }
    run_manifest = _build_run_manifest(
        run_dir=run_dir,
        source=source,
        repositories=repositories,
        summaries=summaries,
        layout=layout,
        gold_label_path=gold_label_path,
        live_artifact_dir=live_artifact_dir,
        created_at_utc=clock.now_utc().isoformat(),
    )
    index = _build_review_index_payload(
        run_manifest=run_manifest,
        source_log=source_log,
        source=source,
        product_repo=product_repo,
        repositories=repositories,
        summaries=summaries,
        layout=layout,
        shared_l0_bundle=shared_l0_bundle,
        shared_decision_evidence=shared_decision_evidence,
        shared_l0_model_view=shared_l0_model_view,
        gold_label_path=gold_label_path,
    )
    artifact_store.write_json(run_dir / "run_manifest.json", run_manifest)
    artifact_store.write_json(run_dir / "review_index.json", index)
    artifact_store.write_text(
        run_dir / "review_index.md",
        _review_index_markdown(
            source_log=source_log,
            source=source,
            product_repo=product_repo,
            repositories=repositories,
            summaries=summaries,
            layout=layout,
            shared_l0_bundle=shared_l0_bundle,
            shared_decision_evidence=shared_decision_evidence,
            shared_l0_model_view=shared_l0_model_view,
            gold_label_path=gold_label_path,
        ),
    )


def _build_run_manifest(
    *,
    run_dir: Path,
    source: dict[str, Any],
    repositories: dict[str, Any],
    summaries: list[dict[str, Any]],
    layout: ArtifactLayout,
    gold_label_path: Path | None,
    live_artifact_dir: Path | None,
    created_at_utc: str,
) -> dict[str, Any]:
    routes = [
        {
            "target": summary.get("target"),
            "run_label": summary.get("run_label"),
            "model": summary.get("model"),
            "effective_tool_profile": summary.get("effective_tool_profile"),
        }
        for summary in summaries
    ]
    return {
        "schema_version": "restart_agent_eval_run.v1",
        "run_id": layout.run_id,
        "created_at_utc": created_at_utc,
        "source": source,
        "roots": {
            "log": str(layout.log_root) if layout.log_root is not None else None,
            "gold": str(layout.gold_root) if layout.gold_root is not None else None,
            "run": str(layout.run_root) if layout.run_root is not None else None,
        },
        "run_dir": str(run_dir),
        "expected_gold_path": str(layout.gold_path) if layout.gold_path is not None else None,
        "gold_attached": gold_label_path is not None,
        "live_artifacts": (
            {
                "directory": str(live_artifact_dir),
                "status": str(live_artifact_dir / "run_status.json"),
                "events": str(live_artifact_dir / "events.jsonl"),
            }
            if live_artifact_dir is not None
            else None
        ),
        "route_artifact_manifest": (
            str(run_dir / "restart_agent_route_artifacts.json")
            if live_artifact_dir is not None
            else None
        ),
        "deterministic_fallback": (
            str(run_dir / "deterministic_fallback.json") if live_artifact_dir is not None else None
        ),
        "repositories": repositories,
        "routes": routes,
    }


def _build_review_index_payload(
    *,
    run_manifest: dict[str, Any],
    source_log: Path,
    source: dict[str, Any],
    product_repo: Path,
    repositories: dict[str, Any],
    summaries: list[dict[str, Any]],
    layout: ArtifactLayout,
    shared_l0_bundle: Path | None,
    shared_decision_evidence: Path | None,
    shared_l0_model_view: Path | None,
    gold_label_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": "restart_agent_review_index.v1",
        "run_manifest": run_manifest,
        "source_log_path": str(source_log),
        "source_log_relative_path": layout.relative_log_path.as_posix(),
        "source_log_sha256": source["sha256"],
        "source_log_byte_size": source["byte_size"],
        "product_repo": str(product_repo),
        "product_commit": repositories["product"]["commit"],
        "harness_repo": str(REPO_ROOT),
        "harness_commit": repositories["harness"]["commit"],
        "shared_l0_bundle": str(shared_l0_bundle) if shared_l0_bundle else None,
        "shared_decision_evidence": (
            str(shared_decision_evidence) if shared_decision_evidence else None
        ),
        "shared_l0_model_view": str(shared_l0_model_view) if shared_l0_model_view else None,
        "gold_label_path": str(gold_label_path) if gold_label_path else None,
        "runs": summaries,
    }


def _review_index_markdown(
    *,
    source_log: Path,
    source: dict[str, Any],
    product_repo: Path,
    repositories: dict[str, Any],
    summaries: list[dict[str, Any]],
    layout: ArtifactLayout,
    shared_l0_bundle: Path | None,
    shared_decision_evidence: Path | None,
    shared_l0_model_view: Path | None,
    gold_label_path: Path | None,
) -> str:
    lines = [
        "# Restart Agent Review Index",
        "",
        "## Start Here",
        "",
        "1. Open [panel_summary.md](panel_summary.md) to compare model outcomes.",
        "2. Open a model's `review.md` below to inspect its complete L1 answer and route assessment.",
        "3. Use [panel_diagnostics.md](panel_diagnostics.md) for cross-route engineering detail.",
        "",
        "The per-model `result.json` is the final composed product result, not the raw model answer. "
        "The per-model `trace.json` is the deep diagnostic record. Its "
        "`analyzer_trace.l1.parsed_evidence` field is reproduced at the top of `review.md`.",
        "",
        "## Run Identity",
        "",
        f"- source_log_path: `{source_log}`",
        f"- source_log_relative_path: `{layout.relative_log_path.as_posix()}`",
        f"- source_log_sha256: `{source['sha256']}`",
        f"- product_repo: `{product_repo}`",
        f"- product_commit: `{repositories['product']['commit']}`",
        f"- harness_commit: `{repositories['harness']['commit']}`",
        f"- shared_l0_bundle: `{shared_l0_bundle}`",
        f"- shared_decision_evidence: `{shared_decision_evidence}`",
        f"- shared_l0_model_view: `{shared_l0_model_view}`",
        f"- gold_label_path: `{gold_label_path}`",
        "",
        "## Artifact Map",
        "",
        "| question | artifact |",
        "|---|---|",
        "| How do the models compare? | [panel_summary.md](panel_summary.md) |",
        "| What exactly did one model say in L1? | The model's `review.md` below |",
        "| What final result did the full pipeline produce? | The model's `result.json` below |",
        "| Why did a stage behave that way? | The model's `trace.json` below |",
        "| What low-level cross-route diagnostics exist? | [panel_diagnostics.md](panel_diagnostics.md) |",
        "",
        "## Model Reviews",
        "",
        "| target | model | L1 output | decision | review | final result | deep trace |",
        "|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        label = str(summary.get("run_label") or summary.get("target") or "route")
        parsed = summary.get("l1_response_parsed")
        l1_status = "not_run" if parsed is None else "parsed" if parsed else "unavailable"
        lines.append(
            "| "
            f"{summary.get('target')} | "
            f"{summary.get('model')} | "
            f"{l1_status} | "
            f"{summary.get('decision')} | "
            f"[open]({label}.review.md) | "
            f"[open]({label}.result.json) | "
            f"[open]({label}.trace.json) |"
        )
    lines.append("")
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_panel_summary(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    try:
        from .panel import write_panel_summary
    except ImportError as exc:
        print(
            f"restart_agent review warning: panel summary unavailable: {exc}",
            file=sys.stderr,
        )
        return
    try:
        write_panel_summary(run_dir, summaries)
    except Exception as exc:  # pragma: no cover - best-effort review artifact.
        print(
            f"restart_agent review warning: failed to write panel summary: {exc}",
            file=sys.stderr,
        )


def persist_process_diagnostics(
    completed: ProcessResult,
    path: Path,
) -> bool:
    """Publish stderr or a synthesized failure diagnostic when useful."""

    stderr = completed.stderr or ""
    if completed.returncode == 0 and not stderr.strip():
        return False
    if not stderr.strip():
        stderr = f"restart_agent process exited with code {completed.returncode} without stderr\n"
    path.write_text(stderr, encoding="utf-8", errors="replace")
    return True


def _print_run_summary(summary: dict[str, Any], paths: dict[str, Path]) -> None:
    signals = summary.get("model_selection_signals") or {}
    tool_efficiency = summary.get("tool_efficiency") or {}
    print()
    print(f"== {summary['run_label']} ==")
    print(
        "decision={decision} primary={fine_class}@{line} "
        "l1_s={l1_s} model_calls={model_calls} tool_calls={tool_calls} "
        "ctx={context_efficiency} sem={semantic_safety} endpoint={endpoint_reliability} "
        "dup_tools={dup_tools} failed_attempts={failed_attempts} "
        "retries={retries} timeouts={timeouts} "
        "redaction={redaction}".format(
            decision=summary.get("decision"),
            fine_class=summary["primary_failure"].get("fine_class"),
            line=summary["primary_failure"].get("line"),
            l1_s=summary["timing"].get("l1_wall_clock_s"),
            model_calls=summary.get("model_calls"),
            tool_calls=summary.get("tool_calls"),
            context_efficiency=signals.get("context_efficiency"),
            semantic_safety=signals.get("semantic_safety"),
            endpoint_reliability=signals.get("endpoint_reliability"),
            dup_tools=tool_efficiency.get("duplicate_prompt_context_calls"),
            failed_attempts=signals.get("failed_endpoint_attempts"),
            retries=signals.get("retried_model_calls"),
            timeouts=signals.get("timeout_model_calls"),
            redaction=summary["path_redaction_audit"].get("passed"),
        )
    )
    print(f"result: {paths['result_json']}")
    print(f"trace:  {paths['trace_json']}")
    if paths.get("stderr_log"):
        print(f"stderr: {paths['stderr_log']}")
    print(f"review: {paths['review_md']}")


if __name__ == "__main__":
    raise SystemExit(main())
