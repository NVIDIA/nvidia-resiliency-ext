# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import product_contract  # noqa: E402


class ProductContractTest(unittest.TestCase):
    def test_collect_all_config_preserves_route_order_and_parallelism(self) -> None:
        routes = [{"route_id": "fast"}, {"route_id": "accurate"}]

        payload = product_contract.collect_all_config(routes)

        self.assertEqual(payload["schema_version"], "restart_agent_config.v1")
        self.assertEqual(payload["routing"], {"mode": "collect_all", "max_parallel_models": 2})
        self.assertEqual(payload["model_routes"], routes)
        self.assertIsNot(payload["model_routes"][0], routes[0])

    def test_route_artifact_manifest_copies_route_paths(self) -> None:
        paths = {"fast": {"result_json": "/run/fast.result.json"}}

        payload = product_contract.route_artifact_manifest(paths)
        paths["fast"]["result_json"] = "changed"

        self.assertEqual(payload["schema_version"], "restart_agent_route_artifacts.v1")
        self.assertEqual(
            payload["routes"],
            {"fast": {"result_json": "/run/fast.result.json"}},
        )


if __name__ == "__main__":
    unittest.main()
