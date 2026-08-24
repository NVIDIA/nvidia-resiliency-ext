# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from _mocks import isolated_environment  # noqa: E402
from restart_agent_eval import paths  # noqa: E402


class PathConfigurationTest(unittest.TestCase):
    def test_product_repo_defaults_to_containing_checkout(self) -> None:
        with isolated_environment():
            product_repo = paths.product_repo_from_env()

        self.assertEqual(product_repo, paths.REPO_ROOT)

    def test_product_repo_uses_shared_environment_resolution(self) -> None:
        configured = paths.product_repo_from_env({"NVRX_RESTART_AGENT_PRODUCT_REPO": "~/product"})

        self.assertEqual(configured, Path("~/product").expanduser())

    def test_missing_optional_path_is_none(self) -> None:
        actual = paths.path_from_env("MISSING", {})

        self.assertIsNone(actual)


if __name__ == "__main__":
    unittest.main()
