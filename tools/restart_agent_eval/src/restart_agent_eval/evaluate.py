#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run human-labeled Restart Agent cases through the product review path."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import validate_artifact_roots
from .corpus import SCHEMA_VERSION, Case, discover_cases
from .paths import REPO_ROOT, TOOL_ROOT, path_from_env, product_repo_from_env
from .product_process import ProcessExecutor, SubprocessExecutor
from .product_trace import ProductTrace, decision_candidate_result
from .repository_identity import git_identity as _git_identity
from .runtime import SYSTEM_CLOCK, Clock
from .scoring import (
    score_l0_against_gold,
    score_l0b_against_gold,
    score_path_effect,
    score_semantic_view,
)

EVAL_ROOT = TOOL_ROOT
REVIEW_LOG = TOOL_ROOT / "src" / "review_log.py"
RECOVERY_FIELDS = (
    "failure_domain",
    "failure_domain_status",
    "retry_outlook_without_workload_change",
    "retry_outlook_status",
)


@dataclass
class CaseResult:
    schema_version: str
    run_id: str
    case_id: str
    target: str
    status: str
    l0a_quality_correct: bool | None
    l0b_quality_correct: bool | None
    l1_recovery_correct: bool | None
    l1_recovery_fields: dict[str, bool | None]
    l2_audit_correct: bool | None
    l4_retry_rule_correct: bool | None
    l4_allowed_retries_correct: bool | None
    l4_exhaustion_correct: bool | None
    accepted_decisions: tuple[str, ...]
    actual_decision: str | None
    decision_correct: bool | None
    fallback_decision: str | None
    fallback_decision_correct: bool | None
    fallback_policy_action_correct: bool | None
    enriched_decision: str | None
    enriched_decision_correct: bool | None
    enriched_policy_action_correct: bool | None
    l1_action_effect: str
    l1_policy_action_effect: str
    evidence_line_hit: bool | None
    primary_anchor_hit: bool | None
    result_path: str | None
    trace_path: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        checks = [
            self.l0a_quality_correct,
            self.l0b_quality_correct,
            self.l1_recovery_correct,
            self.l2_audit_correct,
            self.l4_retry_rule_correct,
            self.l4_allowed_retries_correct,
            self.l4_exhaustion_correct,
            self.decision_correct,
            self.evidence_line_hit,
            self.primary_anchor_hit,
        ]
        scored_checks = [value for value in checks if value is not None]
        return self.status == "scored" and all(value is True for value in scored_checks)


def score_case(
    *,
    run_id: str,
    target: str,
    case: Case,
    verdict: dict[str, Any],
    result_path: Path | None = None,
    trace_path: Path | None = None,
) -> CaseResult:
    trace = _read_trace(trace_path)
    analyzer_trace = dict(trace.analyzer_trace) if trace is not None else {}
    l0_bundle = dict(trace.l0_bundle or {}) if trace is not None else {}
    l0_model_view = analyzer_trace.get("l0_model_view") or {}
    l0a_score = score_l0_against_gold(l0_bundle, case.label)
    l0b_score = score_l0b_against_gold(l0_bundle, l0_model_view, case.label)
    l1_evidence = (analyzer_trace.get("l1") or {}).get("parsed_evidence") or {}
    l1_assessment = (
        l1_evidence.get("model_recovery_assessment") if isinstance(l1_evidence, dict) else None
    )
    if not isinstance(l1_assessment, dict):
        l1_assessment = verdict.get("model_recovery_assessment") or {}
    recovery_fields, recovery_correct = _score_recovery_assessment(
        l1_assessment,
        case.recovery_expectation,
    )
    rule_correct, allowed_correct, exhausted_correct = _score_retry_policy(
        verdict.get("retry_policy") or {},
        case.retry_policy_expectation,
    )
    actual_decision = str(verdict.get("decision") or "") or None
    decision_candidates = analyzer_trace.get("decision_candidates") or {}
    if not isinstance(decision_candidates, dict):
        decision_candidates = {}
    fallback_verdict = decision_candidate_result(decision_candidates.get("deterministic_fallback"))
    enriched_verdict = decision_candidate_result(decision_candidates.get("l1_enriched"))
    candidate_kind = str((verdict.get("result_provenance") or {}).get("candidate_kind") or "")
    if not fallback_verdict and (
        target == "deterministic" or candidate_kind == "deterministic_fallback"
    ):
        fallback_verdict = verdict
    if not enriched_verdict and candidate_kind == "l1_enriched":
        enriched_verdict = verdict
    fallback_score = (
        score_semantic_view(fallback_verdict, case.label, include_action=True)
        if fallback_verdict
        else None
    )
    enriched_score = (
        score_semantic_view(enriched_verdict, case.label, include_action=True)
        if enriched_verdict
        else None
    )
    return CaseResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        case_id=case.case_id,
        target=target,
        status="scored",
        l0a_quality_correct=(l0a_score.get("overall_pass") if l0a_score else None),
        l0b_quality_correct=(l0b_score.get("overall_pass") if l0b_score else None),
        l1_recovery_correct=recovery_correct,
        l1_recovery_fields=recovery_fields,
        l2_audit_correct=_l2_audit_correct(analyzer_trace.get("l2_audit") or {}, case.label),
        l4_retry_rule_correct=rule_correct,
        l4_allowed_retries_correct=allowed_correct,
        l4_exhaustion_correct=exhausted_correct,
        accepted_decisions=case.accepted_decisions,
        actual_decision=actual_decision,
        decision_correct=actual_decision in case.accepted_decisions,
        fallback_decision=(
            str(fallback_verdict.get("decision") or "") or None if fallback_verdict else None
        ),
        fallback_decision_correct=(
            fallback_score.get("action_correct") if fallback_score is not None else None
        ),
        fallback_policy_action_correct=(
            fallback_score.get("policy_action_pass") if fallback_score is not None else None
        ),
        enriched_decision=(
            str(enriched_verdict.get("decision") or "") or None if enriched_verdict else None
        ),
        enriched_decision_correct=(
            enriched_score.get("action_correct") if enriched_score is not None else None
        ),
        enriched_policy_action_correct=(
            enriched_score.get("policy_action_pass") if enriched_score is not None else None
        ),
        l1_action_effect=score_path_effect(
            fallback_score,
            enriched_score,
            "action_correct",
        ),
        l1_policy_action_effect=score_path_effect(
            fallback_score,
            enriched_score,
            "policy_action_pass",
        ),
        evidence_line_hit=_evidence_line_hit(verdict, case.label),
        primary_anchor_hit=_primary_anchor_hit(verdict, case.label),
        result_path=str(result_path) if result_path else None,
        trace_path=str(trace_path) if trace_path else None,
        error=None,
    )


def _score_recovery_assessment(
    assessment: Any,
    expectation: dict[str, Any],
) -> tuple[dict[str, bool | None], bool | None]:
    assessment = assessment if isinstance(assessment, dict) else {}
    results: dict[str, bool | None] = {}
    for field in RECOVERY_FIELDS:
        accepted = _accepted_values(expectation.get(field))
        actual = _recovery_field_value(assessment, field)
        results[field] = str(actual) in accepted if accepted else None
    scored = [value for value in results.values() if value is not None]
    return results, all(scored) if scored else None


def _recovery_field_value(assessment: dict[str, Any], field: str) -> Any:
    if field in {"failure_domain", "failure_domain_status"}:
        claim = assessment.get("failure_domain") or {}
        key = "status" if field.endswith("_status") else "value"
    else:
        claim = assessment.get("retry_outlook_without_workload_change") or {}
        key = "status" if field.endswith("_status") else "value"
    return claim.get(key) if isinstance(claim, dict) else None


def _score_retry_policy(
    retry_policy: Any,
    expectation: dict[str, Any],
) -> tuple[bool | None, bool | None, bool | None]:
    retry_policy = retry_policy if isinstance(retry_policy, dict) else {}
    accepted_rules = _accepted_values(expectation.get("accepted_rules"))
    rule_correct = str(retry_policy.get("rule")) in accepted_rules if accepted_rules else None
    expected_allowed = expectation.get("allowed_retries")
    allowed_correct = (
        _int_or_none(retry_policy.get("allowed_retries")) == _int_or_none(expected_allowed)
        if expected_allowed is not None
        else None
    )
    expected_exhausted = expectation.get("retry_budget_exhausted")
    exhausted_correct = (
        bool(retry_policy.get("retry_budget_exhausted")) == bool(expected_exhausted)
        if expected_exhausted is not None
        else None
    )
    return rule_correct, allowed_correct, exhausted_correct


def _accepted_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_trace(path: Path | None) -> ProductTrace | None:
    if path is None or not path.is_file():
        return None
    try:
        return ProductTrace.from_payload(
            json.loads(path.read_text(encoding="utf-8", errors="replace"))
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _l2_audit_correct(audit: dict[str, Any], label: dict[str, Any]) -> bool | None:
    expectations = label.get("l2_audit_expectation") or []
    if not expectations:
        return None
    field_audits = audit.get("field_audits") or {}
    checks: list[bool] = []
    for expectation in expectations:
        if not isinstance(expectation, dict):
            continue
        field = str(expectation.get("field") or "")
        expected = str(expectation.get("expected") or "")
        actual = field_audits.get(field) or {}
        check = actual.get("status") == expected
        reason = expectation.get("reason_class")
        if reason:
            check = check and str(reason) in {
                str(value) for value in actual.get("finding_classes") or []
            }
        checks.append(check)
    return all(checks) if checks else None


def _evidence_line_hit(verdict: dict[str, Any], label: dict[str, Any]) -> bool | None:
    expected = {
        int(line)
        for line in (label.get("l0b_expectation") or {}).get("required_evidence_lines") or []
    }
    if not expected:
        return None
    actual = {
        int(item["line"])
        for item in verdict.get("evidence") or []
        if isinstance(item, dict) and item.get("line") is not None
    }
    return bool(expected.intersection(actual))


def _primary_anchor_hit(verdict: dict[str, Any], label: dict[str, Any]) -> bool | None:
    expectation = label.get("primary_anchor_expectation") or {}
    accepted = [int(line) for line in expectation.get("accepted_lines") or []]
    if not accepted:
        return None
    primary = verdict.get("primary_failure") or {}
    line = primary.get("line") if isinstance(primary, dict) else None
    if line is None:
        return False
    tolerance = int(expectation.get("tolerance_lines") or 0)
    return any(abs(int(line) - accepted_line) <= tolerance for accepted_line in accepted)


class ProductRunner:
    def __init__(
        self,
        *,
        product_repo: Path,
        target: str,
        artifacts_dir: Path,
        python: str,
        process_executor: ProcessExecutor | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.product_repo = product_repo.resolve()
        self.target = target
        self.artifacts_dir = artifacts_dir.resolve()
        self.python = python
        self.process_executor = process_executor or SubprocessExecutor()
        self.environment = dict(os.environ if environment is None else environment)

    def __call__(self, case: Case) -> tuple[dict[str, Any], Path, Path | None]:
        case_dir = self.artifacts_dir / _safe_token(case.case_id) / _safe_token(self.target)
        case_dir.mkdir(parents=True, exist_ok=True)
        completed = self.process_executor.run(
            [
                self.python,
                str(REVIEW_LOG),
                "--log",
                str(case.log_path),
                "--product-repo",
                str(self.product_repo),
                "--run-dir",
                str(case_dir),
                self.target,
            ],
            cwd=EVAL_ROOT,
            env=self.environment,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"review target failed with exit {completed.returncode}: {detail[-1000:]}"
            )
        result_paths = sorted(case_dir.glob("*.result.json"))
        result_paths = [path for path in result_paths if path.name != "restart_agent.result.json"]
        if len(result_paths) != 1:
            raise RuntimeError(
                f"expected one product result under {case_dir}, found {len(result_paths)}"
            )
        result_path = result_paths[0]
        verdict = json.loads(result_path.read_text(encoding="utf-8"))
        trace_paths = sorted(case_dir.glob("*.trace.json"))
        trace_paths = [path for path in trace_paths if path.name != "restart_agent.trace.json"]
        trace_path = trace_paths[0] if len(trace_paths) == 1 else None
        return verdict, result_path, trace_path


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "case"


def _unavailable_result(run_id: str, target: str, case: Case) -> CaseResult:
    return CaseResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        case_id=case.case_id,
        target=target,
        status="unavailable",
        l0a_quality_correct=None,
        l0b_quality_correct=None,
        l1_recovery_correct=None,
        l1_recovery_fields={},
        l2_audit_correct=None,
        l4_retry_rule_correct=None,
        l4_allowed_retries_correct=None,
        l4_exhaustion_correct=None,
        accepted_decisions=case.accepted_decisions,
        actual_decision=None,
        decision_correct=None,
        fallback_decision=None,
        fallback_decision_correct=None,
        fallback_policy_action_correct=None,
        enriched_decision=None,
        enriched_decision_correct=None,
        enriched_policy_action_correct=None,
        l1_action_effect="not_available",
        l1_policy_action_effect="not_available",
        evidence_line_hit=None,
        primary_anchor_hit=None,
        result_path=None,
        trace_path=None,
        error=f"missing input.log: {case.log_path}",
    )


def _error_result(run_id: str, target: str, case: Case, error: Exception) -> CaseResult:
    result = _unavailable_result(run_id, target, case)
    result.status = "analyzer_error"
    result.error = f"{type(error).__name__}: {error}"
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def aggregate_results(run_id: str, target: str, results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate observable case outcomes for one evaluated target."""
    scored = [result for result in results if result.status == "scored"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "target": target,
        "cases": len(results),
        "scored_cases": len(scored),
        "unavailable_cases": sum(result.status == "unavailable" for result in results),
        "analyzer_errors": sum(result.status == "analyzer_error" for result in results),
        "l0a_quality_accuracy": _rate_defined(result.l0a_quality_correct for result in scored),
        "l0b_quality_accuracy": _rate_defined(result.l0b_quality_correct for result in scored),
        "l1_recovery_accuracy": _rate_defined(result.l1_recovery_correct for result in scored),
        "l4_retry_rule_accuracy": _rate_defined(result.l4_retry_rule_correct for result in scored),
        "l4_allowed_retries_accuracy": _rate_defined(
            result.l4_allowed_retries_correct for result in scored
        ),
        "l4_exhaustion_accuracy": _rate_defined(result.l4_exhaustion_correct for result in scored),
        "decision_accuracy": _rate_defined(result.decision_correct for result in scored),
        "fallback_decision_accuracy": _rate_defined(
            result.fallback_decision_correct for result in scored
        ),
        "fallback_policy_action_accuracy": _rate_defined(
            result.fallback_policy_action_correct for result in scored
        ),
        "enriched_decision_accuracy": _rate_defined(
            result.enriched_decision_correct for result in scored
        ),
        "enriched_policy_action_accuracy": _rate_defined(
            result.enriched_policy_action_correct for result in scored
        ),
        "l1_action_effect_counts": _effect_counts(result.l1_action_effect for result in scored),
        "l1_policy_action_effect_counts": _effect_counts(
            result.l1_policy_action_effect for result in scored
        ),
        "l1_action_improvement_rate": _effect_rate(
            (result.l1_action_effect for result in scored),
            "improved",
        ),
        "l1_action_regression_rate": _effect_rate(
            (result.l1_action_effect for result in scored),
            "regressed",
        ),
        "evidence_line_hit_rate": _rate_defined(result.evidence_line_hit for result in scored),
        "primary_anchor_hit_rate": _rate_defined(result.primary_anchor_hit for result in scored),
    }


def _rate_defined(values: Iterable[bool | None]) -> float | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return sum(value is True for value in items) / len(items)


def _effect_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value).items()))


def _effect_rate(values: Iterable[str], expected: str) -> float | None:
    scored = [value for value in values if value not in {"not_available", "unscored"}]
    if not scored:
        return None
    return sum(value == expected for value in scored) / len(scored)


def _print_report(results: list[CaseResult], aggregate: dict[str, Any]) -> None:
    print(
        "case | status | L1 recovery | L4 rule | retry count | exhaustion | fallback | enriched | L1 effect | decision | evidence | primary"
    )
    print("--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---")
    for result in results:
        print(
            f"{result.case_id} | {result.status} | {_mark(result.l1_recovery_correct)} | "
            f"{_mark(result.l4_retry_rule_correct)} | "
            f"{_mark(result.l4_allowed_retries_correct)} | "
            f"{_mark(result.l4_exhaustion_correct)} | "
            f"{_mark(result.fallback_decision_correct)} | "
            f"{_mark(result.enriched_decision_correct)} | "
            f"{result.l1_action_effect} | "
            f"{_mark(result.decision_correct)} | {_mark(result.evidence_line_hit)} | "
            f"{_mark(result.primary_anchor_hit)}"
        )
        if result.error:
            print(f"  error: {result.error}")
    print()
    print(
        f"scored={aggregate['scored_cases']} unavailable={aggregate['unavailable_cases']} "
        f"errors={aggregate['analyzer_errors']} "
        f"l1_recovery_accuracy={aggregate['l1_recovery_accuracy']} "
        f"l4_retry_rule_accuracy={aggregate['l4_retry_rule_accuracy']} "
        f"l4_allowed_retries_accuracy={aggregate['l4_allowed_retries_accuracy']} "
        f"l4_exhaustion_accuracy={aggregate['l4_exhaustion_accuracy']} "
        f"decision_accuracy={aggregate['decision_accuracy']} "
        f"fallback_decision_accuracy={aggregate['fallback_decision_accuracy']} "
        f"enriched_decision_accuracy={aggregate['enriched_decision_accuracy']} "
        f"l1_improvement_rate={aggregate['l1_action_improvement_rate']} "
        f"l1_regression_rate={aggregate['l1_action_regression_rate']}"
    )


def _mark(value: bool | None) -> str:
    if value is None:
        return "-"
    return "pass" if value else "FAIL"


def parse_evaluation_args(
    argv: list[str] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    """Parse corpus-evaluation options using an explicit environment."""

    environment = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Score labeled Restart Agent cases.")
    parser.add_argument("--target", default="deterministic")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=path_from_env("RESTART_AGENT_EVAL_LOG_ROOT", environment),
        help="source corpus root for mirrored gold discovery",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=path_from_env("RESTART_AGENT_EVAL_GOLD_ROOT", environment),
        help="durable gold root containing <relative-log-path>/gold.json",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=path_from_env("RESTART_AGENT_EVAL_RUN_ROOT", environment),
        help="disposable generated-run root",
    )
    parser.add_argument(
        "--product-repo",
        type=Path,
        default=product_repo_from_env(environment),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--results-jsonl", type=Path)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    environment = dict(os.environ)
    args = parse_evaluation_args(argv, environ=environment)
    return EvaluationApplication(environment=environment).run(args)


@dataclass(frozen=True)
class EvaluationApplication:
    """Composition root for labeled-corpus evaluation."""

    environment: Mapping[str, str]
    process_executor: ProcessExecutor = SubprocessExecutor()
    clock: Clock = SYSTEM_CLOCK

    def run(self, args: argparse.Namespace) -> int:
        return _run_evaluation(
            args,
            process_executor=self.process_executor,
            clock=self.clock,
            environment=self.environment,
        )


def _run_evaluation(
    args: argparse.Namespace,
    *,
    process_executor: ProcessExecutor,
    clock: Clock,
    environment: Mapping[str, str],
) -> int:
    started_at = clock.now_utc()
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_root = args.run_root.expanduser().resolve() if args.run_root else None
    artifacts_dir = (
        args.artifacts_dir.expanduser()
        if args.artifacts_dir
        else (
            run_root / "corpus" / run_id
            if run_root is not None
            else Path(tempfile.gettempdir()) / "nvrx-restart-agent-eval" / run_id
        )
    ).resolve()
    if args.log_root is None or args.gold_root is None:
        raise SystemExit("--log-root and --gold-root are required")
    log_root = args.log_root.expanduser().resolve()
    gold_root = args.gold_root.expanduser().resolve()
    validate_artifact_roots([root for root in (log_root, gold_root, run_root) if root is not None])
    cases = discover_cases(log_root, gold_root)
    discovery = {
        "mode": "mirrored_gold",
        "log_root": str(log_root),
        "gold_root": str(gold_root),
    }
    no_cases_message = f"no gold.json labels found under {gold_root}"
    if not cases:
        print(no_cases_message, file=sys.stderr)
        return 2

    product_repo = args.product_repo.expanduser().resolve() if args.product_repo else None
    eval_identity = _git_identity(REPO_ROOT) or {}
    product_identity = _git_identity(product_repo) or {}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "scored",
        "started_at": started_at.isoformat(),
        "runner": "product",
        "target": args.target,
        "discovery": discovery,
        "artifact_root": str(artifacts_dir),
        "run_root": str(run_root) if run_root is not None else None,
        "eval_commit": eval_identity.get("commit"),
        "eval_dirty": eval_identity.get("dirty"),
        "product_repo": str(product_repo) if product_repo else None,
        "product_commit": product_identity.get("commit"),
        "product_dirty": product_identity.get("dirty"),
        "product_schema_version": None,
        "case_ids": [case.case_id for case in cases],
        "label_versions": {case.case_id: case.label_version for case in cases},
        "options": {"python": args.python, "fail_on_mismatch": args.fail_on_mismatch},
    }
    _write_json(artifacts_dir / "run_manifest.json", manifest)

    product_runner = ProductRunner(
        product_repo=product_repo,  # type: ignore[arg-type]
        target=args.target,
        artifacts_dir=artifacts_dir,
        python=args.python,
        process_executor=process_executor,
        environment=environment,
    )

    results: list[CaseResult] = []
    for case in cases:
        if not case.available:
            results.append(_unavailable_result(run_id, args.target, case))
            continue
        try:
            verdict, result_path, trace_path = product_runner(case)
            results.append(
                score_case(
                    run_id=run_id,
                    target=args.target,
                    case=case,
                    verdict=verdict,
                    result_path=result_path,
                    trace_path=trace_path,
                )
            )
        except Exception as error:  # One analyzer failure must not abort the run.
            results.append(_error_result(run_id, args.target, case, error))

    results_jsonl = args.results_jsonl or artifacts_dir / "case_results.jsonl"
    _write_jsonl(results_jsonl, results)
    aggregate = aggregate_results(run_id, args.target, results)
    _write_json(artifacts_dir / "aggregate.json", aggregate)
    manifest["completed_at"] = clock.now_utc().isoformat()
    manifest["product_schema_version"] = _product_schema_version(results)
    manifest["case_status_counts"] = {
        "scored": aggregate["scored_cases"],
        "unavailable": aggregate["unavailable_cases"],
        "analyzer_error": aggregate["analyzer_errors"],
    }
    _write_json(artifacts_dir / "run_manifest.json", manifest)
    _print_report(results, aggregate)
    print(f"artifacts: {artifacts_dir}")

    if not any(result.status == "scored" for result in results):
        return 2
    if args.fail_on_mismatch and any(
        not result.ok for result in results if result.status == "scored"
    ):
        return 1
    return 0


def _product_schema_version(results: list[CaseResult]) -> str | None:
    versions: set[str] = set()
    for result in results:
        if not result.result_path:
            continue
        try:
            payload = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("schema_version") if isinstance(payload, dict) else None
        if value:
            versions.add(str(value))
    return next(iter(versions)) if len(versions) == 1 else None


if __name__ == "__main__":
    raise SystemExit(main())
