# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

import nvidia_resiliency_ext.attribution.base as attr


class TestNVRxAttribution(unittest.TestCase):
    """Test cases for NVRxAttribution class."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple test functions for single values
        self.sync_preprocess = lambda x: x * 2
        self.sync_attribution = lambda x: x + 10
        self.sync_output_handler = lambda x: x * 3

        # Test functions for lists
        self.list_preprocess = lambda x: [item * 2 for item in x] if isinstance(x, list) else x * 2
        self.list_attribution = lambda x: (
            [item + 10 for item in x] if isinstance(x, list) else x + 10
        )
        self.list_output_handler = lambda x: (
            [item * 3 for item in x] if isinstance(x, list) else x * 3
        )

    def test_init_with_custom_kwargs(self):
        """Test initialization with custom attribution kwargs."""
        custom_kwargs = {"param1": "value1", "param2": 42}
        attribution = attr.NVRxAttribution(
            preprocess_input=self.sync_preprocess,
            attribution=self.sync_attribution,
            output_handler=self.sync_output_handler,
            attribution_kwargs=custom_kwargs,
        )

        self.assertEqual(attribution.attribution_kwargs, custom_kwargs)

    def test_init_with_custom_thread_pool(self):
        """Test initialization with custom thread pool."""
        custom_pool = ThreadPoolExecutor(max_workers=4)
        attribution = attr.NVRxAttribution(
            preprocess_input=self.sync_preprocess,
            attribution=self.sync_attribution,
            output_handler=self.sync_output_handler,
            thread_pool=custom_pool,
        )

        self.assertEqual(attribution._thread_pool, custom_pool)

    def test_get_shared_loop(self):
        """Test getting the shared event loop."""
        # Reset the shared loop
        attr.NVRxAttribution._shared_loop = None

        loop1 = attr.NVRxAttribution.get_shared_loop()
        loop2 = attr.NVRxAttribution.get_shared_loop()

        self.assertIs(loop1, loop2)
        self.assertFalse(loop1.is_closed())

    def test_run_sync_single_input(self):
        """Test running the attribution pipeline with single input synchronously."""
        attribution = attr.NVRxAttribution(
            preprocess_input=self.sync_preprocess,
            attribution=self.sync_attribution,
            output_handler=self.sync_output_handler,
        )

        result = attribution.run_sync(5)
        # Expected: (5 * 2 + 10) * 3 = 60
        self.assertEqual(result, 60)

    def test_run_sync_list_input(self):
        """Test running the attribution pipeline with list input synchronously."""
        attribution = attr.NVRxAttribution(
            preprocess_input=self.list_preprocess,
            attribution=self.list_attribution,
            output_handler=self.list_output_handler,
        )

        result = attribution.run_sync([1, 2, 3])
        # Expected: ([2, 4, 6] + 10) * 3 = [36, 42, 48]
        self.assertEqual(result, [36, 42, 48])

    def test_attribution_with_kwargs(self):
        """Test attribution function with custom kwargs."""

        def attribution_with_kwargs(x, multiplier=1, adder=0):
            return x * multiplier + adder

        attribution = attr.NVRxAttribution(
            preprocess_input=self.sync_preprocess,
            attribution=attribution_with_kwargs,
            output_handler=self.sync_output_handler,
            attribution_kwargs={"multiplier": 3, "adder": 5},
        )

        result = attribution.run_sync(5)
        # Expected: ((5 * 2) * 3 + 5) * 3 = 105
        self.assertEqual(result, 105)

    def test_cleanup_on_deletion(self):
        """Test that thread pool is properly cleaned up on deletion."""
        custom_pool = ThreadPoolExecutor(max_workers=2)
        attribution = attr.NVRxAttribution(
            preprocess_input=self.sync_preprocess,
            attribution=self.sync_attribution,
            output_handler=self.sync_output_handler,
            thread_pool=custom_pool,
        )

        # Mock the shutdown method
        with patch.object(custom_pool, 'shutdown') as mock_shutdown:
            del attribution
            mock_shutdown.assert_called_once_with(wait=False)

    def test_error_handling_in_preprocess(self):
        """Test error handling in preprocessing step."""

        def failing_preprocess(x):
            raise ValueError("Preprocessing failed")

        attribution = attr.NVRxAttribution(
            preprocess_input=failing_preprocess,
            attribution=self.sync_attribution,
            output_handler=self.sync_output_handler,
        )

        with self.assertRaises(ValueError):
            attribution.run_sync(5)

    def test_error_handling_in_attribution(self):
        """Test error handling in attribution step."""

        def failing_attribution(x):
            raise RuntimeError("Attribution failed")

        attribution = attr.NVRxAttribution(
            preprocess_input=self.sync_preprocess,
            attribution=failing_attribution,
            output_handler=self.sync_output_handler,
        )

        with self.assertRaises(RuntimeError):
            attribution.run_sync(5)

    def test_error_handling_in_output_handler(self):
        """Test error handling in output handler step."""

        def failing_output_handler(x):
            raise TypeError("Output handling failed")

        attribution = attr.NVRxAttribution(
            preprocess_input=self.sync_preprocess,
            attribution=self.sync_attribution,
            output_handler=failing_output_handler,
        )

        with self.assertRaises(TypeError):
            attribution.run_sync(5)

    def test_preprocess_input_with_complex_data(self):
        """Test preprocessing function with complex data structures."""

        def complex_preprocess(data):
            if isinstance(data, list):
                return [x * 2 for x in data]
            return data * 2

        attribution = attr.NVRxAttribution(
            preprocess_input=complex_preprocess,
            attribution=self.list_attribution,
            output_handler=self.list_output_handler,
        )

        result = attribution.run_sync([1, 2, 3])
        # Expected: ([2, 4, 6] + 10) * 3 = [36, 42, 48]
        self.assertEqual(result, [36, 42, 48])

    def test_async_preprocess_with_complex_logic(self):
        """Test async preprocessing with complex logic."""

        async def async_complex_preprocess(data):
            await asyncio.sleep(0.001)  # Simulate async work
            if isinstance(data, list):
                return [x * 3 for x in data]
            return data * 3

        def list_attribution_for_async(x):
            if isinstance(x, list):
                return [item + 10 for item in x]
            return x + 10

        def list_output_handler_for_async(x):
            if isinstance(x, list):
                return [item * 3 for item in x]
            return x * 3

        attribution = attr.NVRxAttribution(
            preprocess_input=async_complex_preprocess,
            attribution=list_attribution_for_async,
            output_handler=list_output_handler_for_async,
        )

        async def test_run():
            return await attribution.run([1, 2, 3])

        result = asyncio.run(test_run())
        # Expected: ([3, 6, 9] + 10) * 3 = [39, 48, 57]
        self.assertEqual(result, [39, 48, 57])

    def test_pipeline_with_mixed_sync_async_functions(self):
        """Test pipeline with mixed sync and async functions."""

        def sync_preprocess(x):
            return x * 2

        async def async_attribution(x):
            await asyncio.sleep(0.001)  # Simulate async work
            return x + 10

        def sync_output_handler(x):
            return x * 3

        attribution = attr.NVRxAttribution(
            preprocess_input=sync_preprocess,
            attribution=async_attribution,
            output_handler=sync_output_handler,
        )

        async def test_run():
            return await attribution.run(5)

        result = asyncio.run(test_run())
        # Expected: (5 * 2 + 10) * 3 = 60
        self.assertEqual(result, 60)


if __name__ == '__main__':
    unittest.main()
