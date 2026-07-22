# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review route planning, credentials, endpoints, and product handoff."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _assertions import assert_mapping_fields  # noqa: E402
from _mocks import RecordingExecutor as _RecordingExecutor  # noqa: E402
from _mocks import isolated_environment, process_result  # noqa: E402
from restart_agent_eval import review as review_log  # noqa: E402
from restart_agent_eval.profiles import (  # noqa: E402
    DEFAULT_INFERENCE_ENDPOINT,
    SECONDARY_KEY_ENV,
    expand_targets,
)


class ReviewRoutePlanningTest(unittest.TestCase):
    def test_review_cli_accepts_l0_bundle_replay_path(self) -> None:
        args = review_log.parse_review_args(
            [
                "--log",
                "/tmp/example.log",
                "--l0-bundle-json-in",
                "/tmp/l0_bundle.json",
                "all",
            ]
        )

        self.assertEqual(args.l0_bundle_json_in, Path("/tmp/l0_bundle.json"))

    def test_deterministic_target_requires_no_model_credentials(self) -> None:
        targets = expand_targets(["deterministic"])

        actual = review_log.validate_model_environment(
            targets,
            environment={},
        )

        self.assertEqual(actual, frozenset())

    def test_panel_model_preflight_requires_secondary_key_environment(self) -> None:
        targets = expand_targets(["qwen235b"])
        with isolated_environment():
            with self.assertRaises(SystemExit):
                review_log.validate_model_environment(targets)

    def test_panel_model_preflight_accepts_readable_secondary_key_file(self) -> None:
        targets = expand_targets(["qwen235b"])
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "key"
            key_file.write_text("secret", encoding="utf-8")
            with isolated_environment({"LLM_API_KEY_OLD_FILE": str(key_file)}):
                required = review_log.validate_model_environment(targets)

        self.assertEqual(required, frozenset({"LLM_API_KEY_OLD_FILE"}))

    def test_panel_model_preflight_rejects_missing_key_path(self) -> None:
        with self.assertRaises(SystemExit):
            review_log.validate_model_environment(
                expand_targets(["qwen235b"]),
                environment={"LLM_API_KEY_OLD_FILE": "/missing/key"},
            )

    def test_qwen_and_panel_preflight_require_both_key_environments(self) -> None:
        targets = expand_targets(["qwen", "qwen235b"])
        with tempfile.TemporaryDirectory() as tmp:
            primary_key = Path(tmp) / "primary"
            primary_key.write_text("secret", encoding="utf-8")
            with isolated_environment({"LLM_API_KEY_FILE": str(primary_key)}):
                with self.assertRaises(SystemExit):
                    review_log.validate_model_environment(targets)

    def test_model_run_inherits_key_environment_without_cli_secret_flag(self) -> None:
        target = expand_targets(["configured"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "configured"])
        completed = process_result(stdout="{}")
        executor = _RecordingExecutor(completed)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key_file = root / "key"
            key_file.write_text("secret", encoding="utf-8")
            with isolated_environment({"LLM_API_KEY_FILE": str(key_file)}):
                review_log.run_target(
                    target=target,
                    args=args,
                    log_path=root / "input.log",
                    product_repo=root,
                    run_dir=root,
                    process_executor=executor,
                )

        command, _, child_environment = executor.calls[0]
        self.assertIn("--enable-l1", command)
        self.assertNotIn("--llm-api-key-file", command)
        self.assertEqual(child_environment["LLM_API_KEY_FILE"], str(key_file))

    def test_panel_model_maps_secondary_key_into_product_environment(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "qwen235b"])
        completed = process_result(stdout="{}")
        executor = _RecordingExecutor(completed)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary_key = root / "primary"
            secondary_key = root / "secondary"
            primary_key.write_text("primary", encoding="utf-8")
            secondary_key.write_text("secondary", encoding="utf-8")
            with isolated_environment(
                {
                    "LLM_API_KEY_FILE": str(primary_key),
                    "LLM_API_KEY_OLD_FILE": str(secondary_key),
                }
            ):
                review_log.run_target(
                    target=target,
                    args=args,
                    log_path=root / "input.log",
                    product_repo=root,
                    run_dir=root,
                    process_executor=executor,
                )

        command, _, child_environment = executor.calls[0]
        self.assertNotIn("--llm-api-key-file", command)
        self.assertEqual(
            child_environment["LLM_API_KEY_FILE"],
            str(secondary_key),
        )

    def test_collect_all_route_payload_preserves_per_model_credential_reference(
        self,
    ) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "qwen235b"])

        with isolated_environment():
            route = review_log.build_route_payload(target, args)

        assert_mapping_fields(
            self,
            route,
            {
                "route_id": "qwen235b",
                "model": "nvidia/qwen/eccn-qwen-235b",
                "base_url": DEFAULT_INFERENCE_ENDPOINT,
                "credential_ref": SECONDARY_KEY_ENV,
            },
        )
        self.assertEqual(route["tools"]["max_rounds"], 1)
        self.assertEqual(
            route["tools"]["advertisement"],
            {
                "overview": True,
                "grep_log": True,
                "read_window": True,
                "get_evidence_objects": False,
            },
        )
        self.assertEqual(route["request"]["max_output_tokens"], 64_000)
        self.assertEqual(route["request"]["context_window_tokens"], 200_000)

    def test_collect_all_route_payload_materializes_base_url_override(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(
            [
                "--log",
                "/tmp/example.log",
                "--base-url",
                "https://regulated.example.test/v1",
                "qwen235b",
            ]
        )

        route = review_log.build_route_payload(target, args)

        self.assertEqual(route["base_url"], "https://regulated.example.test/v1")

    def test_collect_all_route_payload_materializes_base_url_environment(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "qwen235b"])

        with isolated_environment({"NVRX_LLM_BASE_URL": "https://environment.example.test/v1"}):
            route = review_log.build_route_payload(target, args)

        self.assertEqual(route["base_url"], "https://environment.example.test/v1")

    def test_route_payload_materializes_all_request_reasoning_and_tool_overrides(self) -> None:
        target = expand_targets(["qwen397b"])[0]
        args = review_log.parse_review_args(
            [
                "--log",
                "/tmp/example.log",
                "--llm-timeout-seconds",
                "30",
                "--llm-max-output-tokens",
                "1024",
                "--llm-context-window-tokens",
                "4096",
                "--llm-temperature",
                "0.2",
                "--llm-top-p",
                "0.7",
                "--llm-thinking-mode",
                "allow",
                "--llm-reasoning-effort",
                "high",
                "--llm-max-tool-rounds",
                "2",
                "qwen397b",
            ]
        )

        route = review_log.build_route_payload(target, args, environment={})

        self.assertEqual(
            route["request"],
            {
                "timeout_seconds": 30.0,
                "max_output_tokens": 1024,
                "context_window_tokens": 4096,
                "temperature": 0.2,
                "top_p": 0.7,
            },
        )
        self.assertEqual(route["reasoning"], {"thinking_mode": "allow", "reasoning_effort": "high"})
        self.assertEqual(route["tools"]["max_rounds"], 2)

    def test_disabling_tools_removes_tool_turns_from_effective_profile(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(
            ["--log", "/tmp/example.log", "--disable-l1-tools", "qwen235b"]
        )

        profile = review_log.effective_tool_profile(target, args)
        route = review_log.build_route_payload(target, args, environment={})

        self.assertFalse(profile["tools_enabled"])
        self.assertEqual(profile["max_model_turns"], 1)
        self.assertFalse(route["tools"]["enabled"])
