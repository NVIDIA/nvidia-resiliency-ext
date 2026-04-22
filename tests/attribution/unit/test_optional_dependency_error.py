# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from unittest.mock import patch

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    import nvidia_resiliency_ext.attribution as attribution


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestOptionalDependencyMessaging(unittest.TestCase):
    def test_analyzer_missing_optional_dependency_shows_extra_hint(self):
        with patch(
            "nvidia_resiliency_ext.attribution.import_module",
            side_effect=ModuleNotFoundError("No module named 'mcp'", name="mcp"),
        ):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                _ = attribution.Analyzer

        self.assertIn("nvidia-resiliency-ext[attribution]", str(ctx.exception))
        self.assertIn("missing module: mcp", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
