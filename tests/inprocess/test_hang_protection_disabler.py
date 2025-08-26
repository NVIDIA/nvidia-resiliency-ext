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

import datetime
import os
import time
import unittest
import unittest.mock
from typing import Optional

import nvidia_resiliency_ext.inprocess as inprocess

from . import common


@common.apply_all_tests(common.retry())
@unittest.mock.patch.dict(
    os.environ,
    {
        'RANK': '0',
        'WORLD_SIZE': '1',
        'MASTER_ADDR': 'localhost',
    },
)
class TestHangProtectionDisabler(unittest.TestCase):
    def test_disable_hang_protection_context_manager(self):
        """Test that disable_hang_protection context manager works correctly."""
        completed_successfully = False
        sleep_duration = 2  # Longer than hard_timeout

        test_kwargs = {
            'store_kwargs': {
                'port': common.find_free_port(),
                'timeout': datetime.timedelta(seconds=10),
            },
            'soft_timeout': datetime.timedelta(seconds=1.25),
            'hard_timeout': datetime.timedelta(seconds=1.5),
            'heartbeat_timeout': datetime.timedelta(seconds=1.75),
            'barrier_timeout': datetime.timedelta(seconds=3),
        }

        @inprocess.Wrapper(**test_kwargs)
        def test_fn(call_wrapper: Optional[inprocess.CallWrapper] = None):
            nonlocal completed_successfully

            # Use the disable_hang_protection context manager
            with call_wrapper.disable_hang_protection():
                # Sleep longer than both soft and hard timeouts
                # This should NOT trigger a restart because hang protection is disabled
                time.sleep(sleep_duration)
                completed_successfully = True

            return "success"

        # Execute the test and measure elapsed time
        start_time = time.perf_counter()
        result = test_fn()
        elapsed_time = time.perf_counter() - start_time

        # Verify the result
        self.assertEqual(result, "success")
        self.assertTrue(completed_successfully, "Context manager should complete successfully")

        # Verify timing: should take at least as long as sleep_duration (proves hang protection disabled)
        self.assertGreaterEqual(
            elapsed_time,
            sleep_duration,
            f"Test should take at least {sleep_duration}s to prove hang protection is disabled",
        )

        # Verify timing: should not take much longer than sleep_duration (proves no restarts)
        max_expected_time = sleep_duration + 2  # Allow 2s overhead for setup/teardown
        self.assertLess(
            elapsed_time,
            max_expected_time,
            f"Test took {elapsed_time:.2f}s, expected < {max_expected_time}s (possible restarts)",
        )

    def test_hang_protection_reenables(self):
        """Test that hang protection is properly re-enabled after exiting disable_hang_protection context."""
        call_count = 0

        test_kwargs = {
            'store_kwargs': {
                'port': common.find_free_port(),
                'timeout': datetime.timedelta(seconds=10),
            },
            'soft_timeout': datetime.timedelta(seconds=1.25),
            'hard_timeout': datetime.timedelta(seconds=1.5),
            'heartbeat_timeout': datetime.timedelta(seconds=1.75),
            'barrier_timeout': datetime.timedelta(seconds=3),
        }

        @inprocess.Wrapper(**test_kwargs)
        def test_fn(call_wrapper: Optional[inprocess.CallWrapper] = None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: test disable_hang_protection works, then trigger timeout outside context

                # Use disable_hang_protection - this should succeed
                with call_wrapper.disable_hang_protection():
                    time.sleep(2)  # Longer than hard_timeout (1.5s) - should NOT restart

                # Now sleep outside the context - this SHOULD cause a timeout and restart
                time.sleep(1.3)  # Longer than soft_timeout (1.25s) - should restart

                # We should never reach this point due to timeout restart
                self.fail("Expected timeout restart did not occur")

            elif call_count == 2:
                # Second call (after restart): return successfully
                return "restarted_success"
            else:
                self.fail(f"Unexpected call count: {call_count}")

        # Execute the test
        result = test_fn()

        # Verify that restart occurred and function was called twice
        self.assertEqual(
            call_count, 2, "Function should have been called twice (restart should have occurred)"
        )
        self.assertEqual(result, "restarted_success", "Should return success after restart")

    def test_exception_in_disabled_region_still_restarts(self):
        """Test that exceptions inside disable_hang_protection context still trigger restarts."""
        call_count = 0

        test_kwargs = {
            'store_kwargs': {
                'port': common.find_free_port(),
                'timeout': datetime.timedelta(seconds=10),
            },
            'soft_timeout': datetime.timedelta(seconds=1.25),
            'hard_timeout': datetime.timedelta(seconds=1.5),
            'heartbeat_timeout': datetime.timedelta(seconds=1.75),
            'barrier_timeout': datetime.timedelta(seconds=3),
        }

        @inprocess.Wrapper(**test_kwargs)
        def test_fn(call_wrapper: Optional[inprocess.CallWrapper] = None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: raise exception inside disable_hang_protection context

                with call_wrapper.disable_hang_protection():
                    # Raise an exception inside the disabled context
                    # This SHOULD still trigger a restart (exceptions should always restart)
                    raise ValueError("Test exception inside disabled hang protection")

                # We should never reach this point due to exception restart
                self.fail("Expected exception restart did not occur")

            elif call_count == 2:
                # Second call (after restart): return successfully
                return "exception_restarted_success"
            else:
                self.fail(f"Unexpected call count: {call_count}")

        # Execute the test
        result = test_fn()

        # Verify that restart occurred and function was called twice
        self.assertEqual(
            call_count,
            2,
            "Function should have been called twice (exception restart should have occurred)",
        )
        self.assertEqual(
            result, "exception_restarted_success", "Should return success after exception restart"
        )


if __name__ == '__main__':
    unittest.main()
