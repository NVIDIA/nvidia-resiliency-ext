# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _mocks import process_result as _process_result  # noqa: E402
from _panel_fixtures import panel_summary as _panel_summary  # noqa: E402
from restart_agent_eval import behavior  # noqa: E402
from restart_agent_eval import repository_identity  # noqa: E402
from restart_agent_eval import panel as summarize_review_panel  # noqa: E402
from restart_agent_eval.profiles import MODEL_PROFILES  # noqa: E402


class BoundaryIsolationTest(unittest.TestCase):
    def test_should_keep_tests_on_public_production_boundaries(self) -> None:
        violations = []
        for test_path in sorted(Path(__file__).parent.glob("test_*.py")):
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
            production_aliases = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("restart_agent_eval"):
                        for imported in node.names:
                            if imported.name.startswith("_"):
                                violations.append(
                                    f"{test_path.name}:{node.lineno}: imports {imported.name}"
                                )
                            production_aliases.add(imported.asname or imported.name)
                elif isinstance(node, ast.Import):
                    for imported in node.names:
                        if imported.name.startswith("restart_agent_eval"):
                            production_aliases.add(
                                imported.asname or imported.name.partition(".")[0]
                            )

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in production_aliases
                    and node.attr.startswith("_")
                ):
                    violations.append(
                        f"{test_path.name}:{node.lineno}: accesses {node.value.id}.{node.attr}"
                    )
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "assertRaisesRegex"
                ):
                    violations.append(
                        f"{test_path.name}:{node.lineno}: pins exception message text"
                    )
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr.startswith("assert")
                ):
                    assertion_values = [*node.args, *(item.value for item in node.keywords)]
                    for value in assertion_values:
                        for call in (
                            item for item in ast.walk(value) if isinstance(item, ast.Call)
                        ):
                            function = call.func
                            module_call = (
                                isinstance(function, ast.Attribute)
                                and isinstance(function.value, ast.Name)
                                and function.value.id in production_aliases
                            )
                            imported_call = (
                                isinstance(function, ast.Name) and function.id in production_aliases
                            )
                            if module_call or imported_call:
                                violations.append(
                                    f"{test_path.name}:{node.lineno}: executes production "
                                    "behavior inside an assertion"
                                )
                                break
                        else:
                            continue
                        break

        self.assertEqual(violations, [])

    def test_panel_input_separates_loading_from_panel_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "review_index.json").write_text(
                json.dumps({"run_manifest": {"run_id": "run-1"}}),
                encoding="utf-8",
            )
            context = summarize_review_panel.PanelInput.read(
                run_dir,
                [{"target": "deterministic"}],
            )

        self.assertEqual(context.run_manifest, {"run_id": "run-1"})
        self.assertEqual(context.summaries[0]["target"], "deterministic")

    def test_panel_payload_has_explicit_schema(self) -> None:
        panel = _panel_summary([])

        self.assertEqual(panel["schema_version"], "restart_agent_panel.v1")

    def test_behavior_capture_uses_injected_worker_without_mutating_import_path(self) -> None:
        class _Executor:
            def __init__(self) -> None:
                self.calls = []

            def run(self, command, *, cwd, env=None):
                self.calls.append((list(command), cwd, dict(env or {})))
                return _process_result(
                    command,
                    stdout='{"schema_version":"fixture.v1"}\n',
                )

        executor = _Executor()
        original_path = list(sys.path)
        result = behavior.build_fixture(
            Path("/logs/input.log"),
            Path("/product"),
            process_executor=executor,
        )

        self.assertEqual(result, {"schema_version": "fixture.v1"})
        self.assertEqual(sys.path, original_path)
        self.assertIn("restart_agent_eval.behavior_worker", executor.calls[0][0])

    def test_model_profiles_are_immutable(self) -> None:
        with self.assertRaises(TypeError):
            MODEL_PROFILES["new"] = MODEL_PROFILES["qwen"]  # type: ignore[index]

    def test_repository_identity_degrades_when_git_cannot_run(self) -> None:
        class _UnavailableExecutor:
            def run(self, command, *, cwd, env=None):
                raise OSError("git unavailable")

        identity = repository_identity.git_identity(
            Path("/repo"),
            process_executor=_UnavailableExecutor(),
        )

        self.assertEqual(
            identity,
            {"path": "/repo", "commit": None, "dirty": None},
        )


if __name__ == "__main__":
    unittest.main()
