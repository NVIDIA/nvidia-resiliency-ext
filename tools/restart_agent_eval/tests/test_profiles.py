# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static model profile expansion and tool-policy contracts."""

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import review as review_log  # noqa: E402
from restart_agent_eval.profiles import (  # noqa: E402
    MODEL_TARGETS,
    SECONDARY_KEY_ENV,
    expand_targets,
)


class ProfileContractTest(unittest.TestCase):
    def test_empty_target_list_is_empty(self) -> None:
        actual = expand_targets([])

        self.assertEqual(actual, [])

    def test_aliases_expand_in_order_and_remove_duplicates(self) -> None:
        targets = expand_targets(["deterministic", "models", "qwen235b", "deterministic"])

        self.assertEqual([target.name for target in targets], ["deterministic", *MODEL_TARGETS])

    def test_all_alias_includes_deterministic_and_all_models(self) -> None:
        targets = expand_targets(["all"])

        self.assertEqual([target.name for target in targets], ["deterministic", *MODEL_TARGETS])

    def test_unknown_target_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            expand_targets(["unknown-model"])

    def test_model_panel_uses_shared_64k_output_ceiling(self) -> None:
        targets = expand_targets(["models"])

        self.assertEqual(
            {target.name: target.max_output_tokens for target in targets},
            {
                "qwen235b": 64_000,
                "qwen397b": 64_000,
                "nemotron": 64_000,
                "gpt": 64_000,
                "claude": 64_000,
                "gemini": 64_000,
            },
        )
        self.assertTrue(all(target.enable_l1 for target in targets))
        self.assertEqual(
            {target.credential_env for target in targets},
            {SECONDARY_KEY_ENV},
        )

    def test_configured_target_uses_product_model_environment(self) -> None:
        target = expand_targets(["configured"])[0]
        label = review_log.artifact_label(target)

        self.assertTrue(target.enable_l1)
        self.assertIsNone(target.model)
        self.assertEqual(target.credential_env, review_log.PRIMARY_KEY_ENV)
        self.assertEqual(label, "model.configured")

    def test_qwen235b_uses_one_tool_round_experimental_profile(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "qwen235b"])

        profile = review_log.effective_tool_profile(target, args)

        self.assertEqual(
            profile["profile_id"],
            "qwen235b.experimental.one_tool_round.v1",
        )
        self.assertTrue(profile["tools_enabled"])
        self.assertEqual(profile["max_tool_rounds"], 1)
        self.assertEqual(profile["max_model_turns"], 2)
        self.assertEqual(profile["source"], "target_profile")

    def test_qwen397b_uses_supported_tools_without_route_specific_cap(self) -> None:
        target = expand_targets(["qwen397b"])[0]
        args = review_log.parse_review_args(["--log", "/tmp/example.log", "qwen397b"])

        profile = review_log.effective_tool_profile(target, args)
        route = review_log.build_route_payload(target, args)

        self.assertEqual(
            profile["profile_id"],
            "qwen397b.tools_supported.v1",
        )
        self.assertTrue(profile["tools_enabled"])
        self.assertIsNone(profile["max_tool_rounds"])
        self.assertIsNone(profile["max_model_turns"])
        self.assertEqual(profile["source"], "product_default")
        self.assertEqual(route["model"], "nvidia/qwen/eccn-qwen3-5-397b-a17b")
        self.assertEqual(route["credential_ref"], SECONDARY_KEY_ENV)
        self.assertEqual(route["request"]["context_window_tokens"], 262_144)
        self.assertNotIn("max_rounds", route["tools"])

    def test_cli_tool_round_override_wins_over_qwen_profile(self) -> None:
        target = expand_targets(["qwen235b"])[0]
        args = review_log.parse_review_args(
            [
                "--log",
                "/tmp/example.log",
                "--llm-max-tool-rounds",
                "3",
                "qwen235b",
            ]
        )

        profile = review_log.effective_tool_profile(target, args)

        self.assertEqual(profile["max_tool_rounds"], 3)
        self.assertEqual(profile["max_model_turns"], 4)
        self.assertEqual(profile["source"], "cli_override")
