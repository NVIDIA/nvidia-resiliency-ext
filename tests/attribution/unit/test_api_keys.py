# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for API key loading and LLM merge guardrails."""

import asyncio
import sys
import tempfile
import unittest
from unittest.mock import patch

if sys.version_info < (3, 10):
    raise unittest.SkipTest(
        "Attribution package requires Python 3.10+ (e.g. dataclass(slots=True) in log_analyzer.job)."
    )

from nvidia_resiliency_ext.attribution.api_keys import load_llm_api_key
from nvidia_resiliency_ext.attribution.combined_log_fr.llm_merge import merge_log_fr_llm


class TestLoadLlmApiKey(unittest.TestCase):
    def test_reads_and_strips_llm_env(self):
        with patch.dict("os.environ", {"LLM_API_KEY": "  sk-test  "}):
            self.assertEqual(load_llm_api_key(), "sk-test")

    def test_returns_empty_when_unset_and_no_key_files(self):
        def getenv_side_effect(key: str, default=None):
            if key in ("LLM_API_KEY", "LLM_API_KEY_FILE"):
                return None
            return default

        with (
            patch(
                "nvidia_resiliency_ext.attribution.api_keys.os.getenv",
                side_effect=getenv_side_effect,
            ),
            patch("nvidia_resiliency_ext.attribution.api_keys.os.path.isfile", return_value=False),
        ):
            self.assertEqual(load_llm_api_key(), "")

    def test_llm_api_key_file_used_when_set(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".key") as f:
            f.write("key-from-file\n")
            path = f.name
        try:
            with patch.dict("os.environ", {"LLM_API_KEY_FILE": path}, clear=True):
                self.assertEqual(load_llm_api_key(), "key-from-file")
        finally:
            import os as _os

            _os.unlink(path)


class TestMergeLogFrLlm(unittest.TestCase):
    def test_raises_when_api_key_empty(self):
        async def run():
            with self.assertRaises(ValueError) as ctx:
                await merge_log_fr_llm("log", "fr", llm_api_key="", model="dummy-model")
            self.assertIn("LLM API key", str(ctx.exception))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
