# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from unittest.mock import patch

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    import nvidia_resiliency_ext.attribution as attribution
    from nvidia_resiliency_ext.attribution._optional import (
        reraise_if_missing_attribution_dependency,
    )


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestOptionalDependencyMessaging(unittest.TestCase):
    def _evict_modules(self, module_roots):
        evicted = {}
        for module_name in list(sys.modules):
            if any(
                module_name == module_root or module_name.startswith(f"{module_root}.")
                for module_root in module_roots
            ):
                evicted[module_name] = sys.modules.pop(module_name)
        return evicted

    def _restore_modules(self, module_roots, evicted):
        for module_name in list(sys.modules):
            if any(
                module_name == module_root or module_name.startswith(f"{module_root}.")
                for module_root in module_roots
            ):
                del sys.modules[module_name]
        sys.modules.update(evicted)

    def test_analyzer_missing_optional_dependency_shows_extra_hint(self):
        with patch(
            "nvidia_resiliency_ext.attribution.import_module",
            side_effect=ModuleNotFoundError("No module named 'mcp'", name="mcp"),
        ):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                _ = attribution.Analyzer

        self.assertIn("nvidia-resiliency-ext[attribution]", str(ctx.exception))
        self.assertIn("missing module: mcp", str(ctx.exception))

    def test_service_missing_optional_dependency_shows_extra_hint(self):
        with self.assertRaises(ModuleNotFoundError) as ctx:
            reraise_if_missing_attribution_dependency(
                ModuleNotFoundError("No module named 'fastapi'", name="fastapi"),
                feature="nvrx-attrsvc",
            )

        self.assertIn("nvidia-resiliency-ext[attribution]", str(ctx.exception))
        self.assertIn("missing module: fastapi", str(ctx.exception))

    def test_attrsvc_package_import_is_lightweight(self):
        module_roots = ("nvidia_resiliency_ext.services.attrsvc",)
        evicted = self._evict_modules(module_roots)
        try:
            __import__("nvidia_resiliency_ext.services.attrsvc")
            self.assertNotIn("nvidia_resiliency_ext.services.attrsvc.app", sys.modules)
        finally:
            self._restore_modules(module_roots, evicted)

    def test_smonsvc_package_import_is_lightweight(self):
        module_roots = (
            "nvidia_resiliency_ext.services.attrsvc",
            "nvidia_resiliency_ext.services.smonsvc",
        )
        evicted = self._evict_modules(module_roots)
        try:
            __import__("nvidia_resiliency_ext.services.smonsvc")
            self.assertNotIn("nvidia_resiliency_ext.services.attrsvc.app", sys.modules)
        finally:
            self._restore_modules(module_roots, evicted)


if __name__ == "__main__":
    unittest.main()
