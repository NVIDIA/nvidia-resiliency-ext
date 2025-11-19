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

import unittest
from unittest.mock import patch

from nvidia_resiliency_ext.fault_tolerance.progress_tracker import TrainingProgressTracker


class TestTrainingProgressTracker(unittest.TestCase):
    """Tests for TrainingProgressTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.min_progress_iterations = 200
        self.max_no_progress_restarts = 3

    def test_initialization(self):
        """Test tracker initialization with default values."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        self.assertEqual(tracker.min_progress_iterations, 200)
        self.assertEqual(tracker.max_no_progress_restarts, 3)
        self.assertEqual(tracker.current_max_iteration, 0)
        self.assertEqual(tracker.last_restart_iteration, 0)
        self.assertEqual(tracker.no_progress_count, 0)
        self.assertEqual(tracker.restart_count, 0)
        self.assertFalse(tracker._iterations_not_reported_warned)

    def test_update_iteration(self):
        """Test that iteration updates track maximum value."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Update with increasing values
        tracker.update_iteration(100)
        self.assertEqual(tracker.current_max_iteration, 100)

        tracker.update_iteration(200)
        self.assertEqual(tracker.current_max_iteration, 200)

        # Update with smaller value should not decrease max
        tracker.update_iteration(150)
        self.assertEqual(tracker.current_max_iteration, 200)

        # Update with larger value should increase max
        tracker.update_iteration(300)
        self.assertEqual(tracker.current_max_iteration, 300)

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_analyze_previous_cycle_initial_run(self, mock_logger):
        """Test analyze_previous_cycle during initial run (Cycle 0)."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Initial run - no analysis should happen
        tracker.analyze_previous_cycle()

        # Verify restart_count incremented but no progress check
        self.assertEqual(tracker.restart_count, 1)
        self.assertEqual(tracker.no_progress_count, 0)
        # No INFO or WARNING logs should be generated for initial run
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_analyze_previous_cycle_with_progress(self, mock_logger):
        """Test analyze_previous_cycle when progress is made."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Simulate Cycle 0 (initial run)
        tracker.update_iteration(500)
        tracker.analyze_previous_cycle()

        # Simulate Cycle 1 (first restart) with good progress
        tracker.update_iteration(800)  # 800 - 500 = 300 iterations
        tracker.analyze_previous_cycle()

        # Verify progress detected
        self.assertEqual(tracker.restart_count, 2)
        self.assertEqual(tracker.no_progress_count, 0)
        self.assertEqual(tracker.last_restart_iteration, 800)

        # Check INFO log was generated (logs as #1 since restart_count=1 when check happens)
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("Training cycle #1 made progress", log_message)
        self.assertIn("500 → 800", log_message)
        self.assertIn("(300 iterations)", log_message)

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_analyze_previous_cycle_no_progress(self, mock_logger):
        """Test analyze_previous_cycle when no progress is made."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Simulate Cycle 0 (initial run)
        tracker.update_iteration(500)
        tracker.analyze_previous_cycle()

        # Simulate Cycle 1 with insufficient progress (only 50 iterations)
        tracker.update_iteration(550)
        tracker.analyze_previous_cycle()

        # Verify no progress detected
        self.assertEqual(tracker.restart_count, 2)
        self.assertEqual(tracker.no_progress_count, 1)

        # Check WARNING log was generated (logs as #1 since restart_count=1 when check happens)
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        self.assertIn("Training cycle #1 made NO progress", log_message)
        self.assertIn("500 → 550", log_message)
        self.assertIn("(50 iterations, need 200)", log_message)
        self.assertIn("Consecutive cycles without progress: 1/3", log_message)

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_analyze_previous_cycle_no_iterations_reported(self, mock_logger):
        """Test analyze_previous_cycle when no iterations are reported."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Simulate Cycle 0 (initial run) - no iterations reported
        tracker.analyze_previous_cycle()

        # Simulate Cycle 1 (first restart) - still no iterations reported
        tracker.analyze_previous_cycle()

        # Verify warning logged once and no_progress_count not incremented
        self.assertEqual(tracker.restart_count, 2)
        self.assertEqual(tracker.no_progress_count, 0)
        self.assertTrue(tracker._iterations_not_reported_warned)

        # Check WARNING log was generated once
        self.assertEqual(mock_logger.warning.call_count, 1)
        log_message = mock_logger.warning.call_args[0][0]
        self.assertIn(
            "Progress tracking is enabled but no iterations are being reported", log_message
        )

        # Simulate Cycle 2 - still no iterations, should not warn again
        mock_logger.reset_mock()
        tracker.analyze_previous_cycle()
        # No WARNING should be logged this time (already warned)
        self.assertEqual(mock_logger.warning.call_count, 0)

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_analyze_previous_cycle_progress_reset(self, mock_logger):
        """Test that no_progress_count resets after progress is made."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Cycle 0: initial run
        tracker.update_iteration(500)
        tracker.analyze_previous_cycle()

        # Cycle 1: no progress
        tracker.update_iteration(550)
        tracker.analyze_previous_cycle()
        self.assertEqual(tracker.no_progress_count, 1)

        # Cycle 2: good progress should reset counter
        tracker.update_iteration(800)
        tracker.analyze_previous_cycle()
        self.assertEqual(tracker.no_progress_count, 0)

    def test_should_terminate_early_false(self):
        """Test should_terminate_early returns False when threshold not reached."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # No progress yet
        self.assertFalse(tracker.should_terminate_early())

        # Some no-progress cycles but below threshold
        tracker.no_progress_count = 2
        self.assertFalse(tracker.should_terminate_early())

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_should_terminate_early_true(self, mock_logger):
        """Test should_terminate_early returns True when threshold reached."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Set no_progress_count to threshold
        tracker.no_progress_count = 3
        tracker.last_restart_iteration = 100

        # Should trigger early termination
        self.assertTrue(tracker.should_terminate_early())

        # Check ERROR log was generated
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        self.assertIn("EARLY TERMINATION", log_message)
        self.assertIn("3 consecutive training cycles", log_message)
        self.assertIn("stuck at iteration ~100", log_message)

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_full_scenario_early_termination(self, mock_logger):
        """Test a complete scenario leading to early termination."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Cycle 0: Initial run with good progress
        tracker.update_iteration(300)
        tracker.analyze_previous_cycle()
        self.assertFalse(tracker.should_terminate_early())

        # Cycle 1: Good progress
        tracker.update_iteration(600)
        tracker.analyze_previous_cycle()
        self.assertEqual(tracker.no_progress_count, 0)
        self.assertFalse(tracker.should_terminate_early())

        # Cycle 2: No progress (stuck at iteration 605)
        tracker.update_iteration(605)
        tracker.analyze_previous_cycle()
        self.assertEqual(tracker.no_progress_count, 1)
        self.assertFalse(tracker.should_terminate_early())

        # Cycle 3: Still no progress
        tracker.update_iteration(610)
        tracker.analyze_previous_cycle()
        self.assertEqual(tracker.no_progress_count, 2)
        self.assertFalse(tracker.should_terminate_early())

        # Cycle 4: Still no progress - should trigger termination
        tracker.update_iteration(615)
        tracker.analyze_previous_cycle()
        self.assertEqual(tracker.no_progress_count, 3)
        self.assertTrue(tracker.should_terminate_early())

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_exact_threshold_progress(self, mock_logger):
        """Test progress exactly at the minimum threshold."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Cycle 0: Initial run
        tracker.update_iteration(500)
        tracker.analyze_previous_cycle()

        # Cycle 1: Exactly min_progress_iterations (200)
        tracker.update_iteration(700)  # 700 - 500 = 200
        tracker.analyze_previous_cycle()

        # Should count as progress (>= threshold)
        self.assertEqual(tracker.no_progress_count, 0)
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("made progress", log_message)

    @patch('nvidia_resiliency_ext.fault_tolerance.progress_tracker.logger')
    def test_just_below_threshold_progress(self, mock_logger):
        """Test progress just below the minimum threshold."""
        tracker = TrainingProgressTracker(
            min_progress_iterations=self.min_progress_iterations,
            max_no_progress_restarts=self.max_no_progress_restarts,
        )

        # Cycle 0: Initial run
        tracker.update_iteration(500)
        tracker.analyze_previous_cycle()

        # Cycle 1: Just below threshold (199 iterations)
        tracker.update_iteration(699)
        tracker.analyze_previous_cycle()

        # Should count as no progress
        self.assertEqual(tracker.no_progress_count, 1)
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        self.assertIn("made NO progress", log_message)


if __name__ == "__main__":
    unittest.main()
