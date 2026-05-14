# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataflow HTTP posting in post_backend."""

import sys
import unittest
from unittest import mock

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.postprocessing import post_backend


@unittest.skipUnless(
    PY310_PLUS,
    "Importing attribution.postprocessing requires Python 3.10+ (dataclass slots).",
)
class TestPostWithRetries(unittest.TestCase):
    def tearDown(self) -> None:
        post_backend.set_post_override(None)
        post_backend._default_http_post_fn = None

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


@unittest.skipUnless(
    PY310_PLUS,
    "Importing attribution.postprocessing requires Python 3.10+ (dataclass slots).",
)
class TestDataflowHttpPost(unittest.TestCase):
    def tearDown(self) -> None:
        post_backend.set_post_override(None)
        post_backend._default_http_post_fn = None

    def test_full_endpoint_posts_json(self):
        response = mock.Mock(status_code=201, text="created")
        post = post_backend.make_dataflow_http_post_fn(
            endpoint="https://dataflow.example.test/dataflow2/sandbox-nvrx/posting",
            timeout_seconds=3.0,
            queue="fast",
        )

        with mock.patch.object(post_backend.httpx, "post", return_value=response) as http_post:
            self.assertIs(post({"s_name": "nvrx"}, ""), True)

        http_post.assert_called_once_with(
            "https://dataflow.example.test/dataflow2/sandbox-nvrx/posting",
            json={"s_name": "nvrx"},
            params={"queue": "fast"},
            timeout=3.0,
        )

    def test_endpoint_is_used_as_configured(self):
        response = mock.Mock(status_code=200, text="ok")
        post = post_backend.make_dataflow_http_post_fn(
            endpoint="https://dataflow.example.test/custom/posting",
            timeout_seconds=2.0,
        )

        with mock.patch.object(post_backend.httpx, "post", return_value=response) as http_post:
            self.assertIs(post({"s_name": "nvrx"}, "ignored"), True)

        http_post.assert_called_once_with(
            "https://dataflow.example.test/custom/posting",
            json={"s_name": "nvrx"},
            params=None,
            timeout=2.0,
        )

    def test_default_retrying_post_requires_configured_endpoint(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIs(
                post_backend._post_with_retries({"s_name": "nvrx"}, "sandbox-nvrx"),
                False,
            )

    def test_default_retrying_post_uses_configured_http_backend(self):
        response = mock.Mock(status_code=201, text="created")

        with mock.patch.dict(
            "os.environ",
            {
                "NVRX_ATTRSVC_EXPORT_URL": (
                    "https://dataflow.example.test/dataflow2/sandbox-nvrx/posting"
                )
            },
            clear=True,
        ):
            with mock.patch.object(post_backend.httpx, "post", return_value=response) as http_post:
                self.assertIs(
                    post_backend._post_with_retries({"s_name": "nvrx"}, "sandbox-nvrx"),
                    True,
                )

        http_post.assert_called_once()

    def test_default_retrying_post_reuses_http_poster_for_retries(self):
        calls: list[int] = []

        def post(data, index):
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionError("transient")
            return True

        factory = mock.Mock(return_value=post)

        with mock.patch.object(post_backend, "make_dataflow_http_post_fn", factory):
            with mock.patch.object(post_backend.time, "sleep"):
                self.assertIs(
                    post_backend._post_with_retries({"s_name": "nvrx"}, "sandbox-nvrx"),
                    True,
                )

        factory.assert_called_once_with()
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
