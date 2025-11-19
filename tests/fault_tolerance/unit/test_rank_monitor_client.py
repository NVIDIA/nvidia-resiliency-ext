# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nvidia_resiliency_ext.fault_tolerance.rank_monitor_client import RankMonitorClient


class TestRankMonitorClient(unittest.TestCase):
    """Tests for RankMonitorClient, focusing on iteration detection and reporting."""

    def setUp(self):
        """Set up test fixtures."""
        # Clean up any Megatron modules that might exist
        if 'megatron.training.global_vars' in sys.modules:
            del sys.modules['megatron.training.global_vars']

    def tearDown(self):
        """Clean up after tests."""
        # Clean up any Megatron modules we created
        if 'megatron.training.global_vars' in sys.modules:
            del sys.modules['megatron.training.global_vars']

    @patch('nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.get_rank')
    def test_no_megatron_detection(self, mock_get_rank):
        """Test that client works without Megatron-LM."""
        mock_get_rank.return_value = 0

        # Create client without Megatron module
        client = RankMonitorClient()

        # Should not detect workload module
        self.assertIsNone(client._workload_global_vars_module)
        self.assertIsNone(client._cached_workload_args)

        # Should not be able to report iterations
        self.assertFalse(client._can_report_iterations())

        # _get_current_iteration should return None
        self.assertIsNone(client._get_current_iteration())

    @patch('nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.get_rank')
    def test_megatron_detection_without_global_args(self, mock_get_rank):
        """Test Megatron detection when _GLOBAL_ARGS is not yet initialized."""
        mock_get_rank.return_value = 0

        # Create mock Megatron module without _GLOBAL_ARGS
        mock_megatron_module = SimpleNamespace()
        sys.modules['megatron.training.global_vars'] = mock_megatron_module

        # Create client
        client = RankMonitorClient()

        # Should detect workload module
        self.assertIsNotNone(client._workload_global_vars_module)
        self.assertEqual(client._workload_global_vars_module, mock_megatron_module)

        # Should indicate it can report iterations (framework detected)
        self.assertTrue(client._can_report_iterations())

        # But _get_current_iteration should return None (args not initialized)
        self.assertIsNone(client._get_current_iteration())
        self.assertIsNone(client._cached_workload_args)

    @patch('nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.get_rank')
    def test_megatron_detection_with_global_args(self, mock_get_rank):
        """Test Megatron detection when _GLOBAL_ARGS is initialized."""
        mock_get_rank.return_value = 0

        # Create mock Megatron module with _GLOBAL_ARGS
        mock_args = SimpleNamespace(curr_iteration=100)
        mock_megatron_module = SimpleNamespace(_GLOBAL_ARGS=mock_args)
        sys.modules['megatron.training.global_vars'] = mock_megatron_module

        # Create client
        client = RankMonitorClient()

        # Should detect workload module and args
        self.assertIsNotNone(client._workload_global_vars_module)
        self.assertTrue(client._can_report_iterations())

        # Should return current iteration
        iteration = client._get_current_iteration()
        self.assertEqual(iteration, 100)

        # Should cache the args
        self.assertIsNotNone(client._cached_workload_args)
        self.assertEqual(client._cached_workload_args, mock_args)

    @patch('nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.get_rank')
    def test_global_args_caching(self, mock_get_rank):
        """Test that _GLOBAL_ARGS is cached after first successful retrieval."""
        mock_get_rank.return_value = 0

        # Create mock Megatron module without _GLOBAL_ARGS initially
        mock_megatron_module = SimpleNamespace()
        sys.modules['megatron.training.global_vars'] = mock_megatron_module

        # Create client
        client = RankMonitorClient()

        # First call - no args yet
        self.assertIsNone(client._get_current_iteration())
        self.assertIsNone(client._cached_workload_args)

        # Now add _GLOBAL_ARGS (simulating late initialization)
        mock_args = SimpleNamespace(curr_iteration=200)
        mock_megatron_module._GLOBAL_ARGS = mock_args

        # Second call - should find and cache args
        iteration = client._get_current_iteration()
        self.assertEqual(iteration, 200)
        self.assertIsNotNone(client._cached_workload_args)

        # Modify the iteration in the module
        mock_args.curr_iteration = 300

        # Third call - should use cached args and get updated iteration
        iteration = client._get_current_iteration()
        self.assertEqual(iteration, 300)

    @patch('nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.get_rank')
    def test_global_args_without_curr_iteration(self, mock_get_rank):
        """Test when _GLOBAL_ARGS exists but doesn't have curr_iteration."""
        mock_get_rank.return_value = 0

        # Create mock Megatron module with _GLOBAL_ARGS but no curr_iteration
        mock_args = SimpleNamespace(other_field=123)
        mock_megatron_module = SimpleNamespace(_GLOBAL_ARGS=mock_args)
        sys.modules['megatron.training.global_vars'] = mock_megatron_module

        # Create client
        client = RankMonitorClient()

        # Should detect module
        self.assertIsNotNone(client._workload_global_vars_module)

        # Should cache args
        iteration = client._get_current_iteration()
        self.assertIsNone(iteration)  # No curr_iteration field
        self.assertIsNotNone(client._cached_workload_args)

    @patch('nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.get_rank')
    def test_iteration_updates_dynamically(self, mock_get_rank):
        """Test that iteration values update as training progresses."""
        mock_get_rank.return_value = 0

        # Create mock Megatron module with _GLOBAL_ARGS
        mock_args = SimpleNamespace(curr_iteration=1)
        mock_megatron_module = SimpleNamespace(_GLOBAL_ARGS=mock_args)
        sys.modules['megatron.training.global_vars'] = mock_megatron_module

        # Create client
        client = RankMonitorClient()

        # Check initial iteration
        self.assertEqual(client._get_current_iteration(), 1)

        # Simulate training progress
        mock_args.curr_iteration = 100
        self.assertEqual(client._get_current_iteration(), 100)

        mock_args.curr_iteration = 200
        self.assertEqual(client._get_current_iteration(), 200)

    @patch('nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.get_rank')
    def test_can_report_iterations_logic(self, mock_get_rank):
        """Test _can_report_iterations logic for arming progress tracking."""
        mock_get_rank.return_value = 0

        # Case 1: No Megatron module
        client1 = RankMonitorClient()
        self.assertFalse(client1._can_report_iterations())

        # Case 2: Megatron module present but no _GLOBAL_ARGS yet
        mock_megatron_module = SimpleNamespace()
        sys.modules['megatron.training.global_vars'] = mock_megatron_module
        client2 = RankMonitorClient()
        self.assertTrue(client2._can_report_iterations())  # Can report (framework detected)
        self.assertIsNone(client2._get_current_iteration())  # But no iteration yet

        # Case 3: Megatron module with _GLOBAL_ARGS
        mock_args = SimpleNamespace(curr_iteration=50)
        mock_megatron_module._GLOBAL_ARGS = mock_args
        client3 = RankMonitorClient()
        self.assertTrue(client3._can_report_iterations())
        self.assertEqual(client3._get_current_iteration(), 50)


if __name__ == "__main__":
    unittest.main()
