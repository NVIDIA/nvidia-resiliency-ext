# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nvdataflow result handling in post_backend."""

import sys
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest(
        "Importing attribution.postprocessing requires Python 3.10+ (dataclass slots)."
    )

from nvidia_resiliency_ext.attribution.postprocessing import post_backend
from nvidia_resiliency_ext.attribution.postprocessing.post_backend import _nvdataflow_result_ok


class TestNvdataflowResultOk(unittest.TestCase):
    def test_bool(self):
        self.assertIs(_nvdataflow_result_ok(True), True)
        self.assertIs(_nvdataflow_result_ok(False), False)

    def test_none_is_success(self):
        self.assertIs(_nvdataflow_result_ok(None), True)

    def test_int_exit_code(self):
        self.assertIs(_nvdataflow_result_ok(0), True)
        self.assertIs(_nvdataflow_result_ok(200), True)
        self.assertIs(_nvdataflow_result_ok(201), True)
        self.assertIs(_nvdataflow_result_ok(1), False)


class TestPostWithRetries(unittest.TestCase):
    def tearDown(self) -> None:
        post_backend.set_post_override(None)

    def test_false_return_is_not_retried(self):
        calls: list[int] = []

        def override(data, index):
            calls.append(1)
            return False

        post_backend.set_post_override(override)
        self.assertIs(post_backend._post_with_retries({}, "idx"), False)
        self.assertEqual(len(calls), 1)

    def test_exceptions_are_retried_until_success(self):
        calls: list[int] = []

        def override(data, index):
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("transient")
            return True

        post_backend.set_post_override(override)
        self.assertIs(post_backend._post_with_retries({}, "idx"), True)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
