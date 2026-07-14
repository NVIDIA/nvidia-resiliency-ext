#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture or verify deterministic restart-agent behavior for gold cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import SOURCE_ROOT
from .product_process import ProcessExecutor, SubprocessExecutor

FIXTURE_SCHEMA_VERSION = "restart_agent_behavior_fixture.v1"
FIXTURE_NAME = "behavior.json"
ELAPSED_KEYS = frozenset(
    {
        "ready_wall_clock_s",
        "remaining_at_return_s",
        "terminal_total_wall_clock_s",
        "total_wall_clock_s",
        "wall_clock_s",
    }
)


def _product_modules(product_repo: Path) -> Mapping[str, Any]:
    source_root = product_repo.resolve() / "src"
    if not source_root.is_dir():
        raise SystemExit(f"product source directory does not exist: {source_root}")
    sys.path.insert(0, str(source_root))

    from nvidia_resiliency_ext.attribution.restart_agent import RestartAgent, RuntimeInputs
    from nvidia_resiliency_ext.attribution.restart_agent.l0 import (
        build_decision_evidence,
        build_l0_bundle,
        build_l0_model_facing_view,
    )

    return {
        "RestartAgent": RestartAgent,
        "RuntimeInputs": RuntimeInputs,
        "build_decision_evidence": build_decision_evidence,
        "build_l0_bundle": build_l0_bundle,
        "build_l0_model_facing_view": build_l0_model_facing_view,
    }


def _source_identity(log_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with log_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "relative_path": None,
        "sha256": digest.hexdigest(),
        "byte_size": log_path.stat().st_size,
    }


def _normalized(value: Any, *, log_path: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name == "log_path":
                normalized[name] = "<LOG>"
            elif name in ELAPSED_KEYS or name.endswith("_wall_clock_s"):
                normalized[name] = 0.0
            else:
                normalized[name] = _normalized(item, log_path=log_path)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalized(item, log_path=log_path) for item in value]
    if isinstance(value, str):
        return value.replace(log_path, "<LOG>")
    return value


def build_fixture(
    log_path: Path,
    product_repo: Path,
    *,
    process_executor: ProcessExecutor | None = None,
    environment: Mapping[str, str] | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Capture product behavior in an isolated interpreter."""

    process_environment = dict(os.environ if environment is None else environment)
    process_environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(SOURCE_ROOT), process_environment.get("PYTHONPATH")) if item
    )
    completed = (process_executor or SubprocessExecutor()).run(
        [
            python_executable,
            "-m",
            "restart_agent_eval.behavior_worker",
            "--log",
            str(log_path),
            "--product-repo",
            str(product_repo),
        ],
        cwd=product_repo.resolve(),
        env=process_environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"behavior fixture worker failed: {detail[-2000:]}")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise TypeError("behavior fixture worker returned a non-object")
    return result


def build_fixture_in_worker(log_path: Path, product_repo: Path) -> dict[str, Any]:
    """Capture deterministic behavior inside the isolated worker process."""
    modules = _product_modules(product_repo)
    bundle = modules["build_l0_bundle"](log_path)
    decision_evidence = modules["build_decision_evidence"](bundle)
    model_view = modules["build_l0_model_facing_view"](bundle, decision_evidence)
    run = modules["RestartAgent"]().run(
        modules["RuntimeInputs"](log_path=str(log_path)),
        l0_bundle=bundle,
    )
    source = _source_identity(log_path)
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "source": source,
        "behavior": _normalized(
            {
                "l0_bundle": asdict(bundle),
                "decision_evidence": decision_evidence.to_payload(),
                "l0_model_view": model_view.to_payload(),
                "deterministic_result": run.result.to_payload(),
                "deterministic_trace": dict(run.trace),
            },
            log_path=str(log_path),
        ),
    }


def _first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}: keys differ; missing={sorted(expected_keys - actual_keys)} "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected_keys):
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _first_difference(
                expected_item,
                actual_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cases(log_root: Path, gold_root: Path) -> Sequence[tuple[Path, Path]]:
    cases: list[tuple[Path, Path]] = []
    for gold_path in sorted(gold_root.glob("**/gold.json")):
        relative_log = gold_path.parent.relative_to(gold_root)
        log_path = log_root / relative_log
        if not log_path.is_file():
            raise SystemExit(f"gold case source log does not exist: {log_path}")
        cases.append((log_path, gold_path.parent / FIXTURE_NAME))
    if not cases:
        raise SystemExit(f"no gold cases found below {gold_root}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--product-repo", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    failures = 0
    for log_path, fixture_path in _cases(args.log_root.resolve(), args.gold_root.resolve()):
        fixture = build_fixture(log_path.resolve(), args.product_repo)
        fixture["source"]["relative_path"] = str(log_path.relative_to(args.log_root.resolve()))
        if args.check:
            if not fixture_path.is_file():
                print(f"MISSING {fixture_path}")
                failures += 1
                continue
            expected = json.loads(fixture_path.read_text(encoding="utf-8"))
            difference = _first_difference(expected, fixture)
            if difference is not None:
                print(f"DIFF {fixture_path}: {difference}")
                failures += 1
                continue
            print(f"OK {fixture_path}")
        else:
            _write_json(fixture_path, fixture)
            print(f"WROTE {fixture_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
