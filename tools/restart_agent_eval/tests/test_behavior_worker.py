# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from _bootstrap import configure_test_imports

configure_test_imports()

from restart_agent_eval import behavior_worker  # noqa: E402


class BehaviorWorkerTest(unittest.TestCase):
    def test_main_resolves_inputs_and_emits_one_json_object(self) -> None:
        output = io.StringIO()
        inputs = []

        def build(log_path, product_repo):
            inputs.append((log_path, product_repo))
            return {"schema_version": "fixture.v1"}

        with contextlib.redirect_stdout(output):
            exit_code = behavior_worker.main(
                ["--log", "input.log", "--product-repo", "product"],
                fixture_builder=build,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"schema_version": "fixture.v1"})
        self.assertEqual(inputs, [(Path("input.log").resolve(), Path("product").resolve())])


if __name__ == "__main__":
    unittest.main()
