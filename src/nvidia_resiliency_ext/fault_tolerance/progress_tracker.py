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

"""
Training Progress Tracker for detecting stuck restarts.

This module tracks training iteration progress across worker restarts to detect
when restarts are not making meaningful progress (e.g., due to HW failures).
If multiple restarts fail to progress beyond a certain iteration, the job can
be early terminated to save resources.

Progress is defined as the iteration from the checkpoint file having increased
by at least min_progress_iterations (default 1) since the start of the cycle.
When checkpoint_iteration_file is set, the tracker reads it in analyze_previous_cycle().

Note: The launcher agent process runs continuously, so all state is kept in memory.
Only worker processes restart.
"""

import logging
from typing import Optional

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)


class TrainingProgressTracker:
    """
    Tracks training progress across worker restarts to detect stuck training.

    This tracker:
    1. Monitors the maximum iteration reached in each cycle (from checkpoint file when set)
    2. Compares progress between cycles (including the initial cycle 0)
    3. Detects when training is stuck (not progressing)
    4. Recommends early termination when too many consecutive cycles make no progress

    State is kept in memory since the launcher agent runs continuously.
    """

    def __init__(
        self,
        min_progress_iterations: int = 1,
        max_no_progress_cycles: int = 2,
        initial_cycle_number: int = 0,
        checkpoint_iteration_file: Optional[str] = None,
    ):
        """
        Initialize the progress tracker.

        Args:
            min_progress_iterations: Minimum iteration increase to consider as "making progress"
                                     (default 1 = any increase).
            max_no_progress_cycles: Max consecutive cycles (including cycle 0) without progress
                                    before early termination. E.g. 2 = allow cycle 0 and 1 with
                                    no progress, then terminate before starting cycle 2.
                                    Set to <= 0 to disable progress tracking.
            initial_cycle_number: Initial global cycle number (for job array replacement nodes)
                                  Cycle 0 = initial attempt, cycle 1 = first restart, etc.
            checkpoint_iteration_file: Optional path to file containing latest checkpoint iteration.
                                      When set, read at startup for baseline and in analyze_previous_cycle().
        """
        self.min_progress_iterations = min_progress_iterations
        self.max_no_progress_cycles = max_no_progress_cycles
        self.checkpoint_iteration_file = checkpoint_iteration_file

        # State (kept in memory across worker restarts)
        self.current_max_iteration = 0
        self.last_restart_iteration = 0
        self.no_progress_count = 0
        self.cycle_number = (
            initial_cycle_number  # Track global cycle number (0 = initial, 1 = first restart, etc.)
        )
        self._iterations_not_reported_warned = (
            False  # Track if we've warned about missing iterations
        )

        # Training resumes from last checkpoint; use it as baseline for progress (don't assume 0)
        self._read_iteration_from_checkpoint_file()
        self.last_restart_iteration = self.current_max_iteration

    def update_iteration(self, iteration: int):
        """
        Update the current iteration (used when reading from checkpoint file or other source).

        Args:
            iteration: Current training iteration
        """
        if iteration > self.current_max_iteration:
            self.current_max_iteration = iteration

    def sync_cycle_number(self, new_cycle_number: int):
        """
        Sync cycle number when rendezvous round is updated.

        This is called when a stale rendezvous round is detected and corrected,
        ensuring the progress tracker stays in sync with the global cycle.

        Args:
            new_cycle_number: The corrected cycle number to sync to
        """
        self.cycle_number = new_cycle_number

    def _read_iteration_from_checkpoint_file(self) -> None:
        """Read checkpoint iteration file and update current_max_iteration if present."""
        if not self.checkpoint_iteration_file:
            return
        try:
            with open(self.checkpoint_iteration_file, "r") as f:
                raw = f.read().strip()
            if raw:
                iteration = int(raw)
                self.update_iteration(iteration)
        except FileNotFoundError:
            pass  # File may not exist yet
        except (ValueError, OSError) as e:
            logger.debug(
                "Could not read checkpoint iteration file %s: %s",
                self.checkpoint_iteration_file,
                e,
            )

    def analyze_previous_cycle(self):
        """
        Analyze progress made in the previous cycle (including initial cycle 0).

        Called by the launcher before deciding whether to restart. If checkpoint_iteration_file
        is set, reads it to get the iteration reached in the previous cycle, then checks if
        that cycle made meaningful progress and updates tracking state.

        If no iteration source is available (no file and both current and last are 0),
        the progress check is skipped and a warning is logged once.
        """
        # Skip if tracking is disabled
        if self.max_no_progress_cycles <= 0:
            return

        # When using checkpoint file, read it now (before new workers start)
        self._read_iteration_from_checkpoint_file()

        # Check if previous cycle made progress (includes cycle 0; last_restart_iteration starts at 0)
        if self.last_restart_iteration == 0 and self.current_max_iteration == 0:
            # Iterations are not being reported - skip progress check
            if not self._iterations_not_reported_warned:
                logger.warning(
                    "Progress tracking is enabled but no iterations are being reported. "
                    "Provide --ft-checkpoint-iteration-file with path to a file containing "
                    "the latest checkpoint iteration (e.g. phase1/latest_checkpointed_iteration.txt). "
                    "Progress tracking will be inactive until iterations are reported."
                )
                self._iterations_not_reported_warned = True
            # Don't count as no-progress if iterations aren't being reported
        else:
            # Iterations available - check for progress
            progress_made = self.current_max_iteration - self.last_restart_iteration

            if progress_made >= self.min_progress_iterations:
                # Good progress made
                logger.info(
                    f"Training cycle #{self.cycle_number} made progress: "
                    f"{self.last_restart_iteration} → {self.current_max_iteration} "
                    f"({progress_made} iterations)"
                )
                self.no_progress_count = 0  # Reset counter
            else:
                # No meaningful progress
                self.no_progress_count += 1
                logger.warning(
                    f"Training cycle #{self.cycle_number} made NO progress: "
                    f"{self.last_restart_iteration} → {self.current_max_iteration} "
                    f"({progress_made} iterations, need {self.min_progress_iterations}). "
                    f"Consecutive cycles without progress: {self.no_progress_count}/{self.max_no_progress_cycles}"
                )

        # Update for next cycle
        self.cycle_number += 1
        self.last_restart_iteration = self.current_max_iteration
        self.current_max_iteration = 0  # Will be updated during new run

    def should_terminate_early(self) -> bool:
        """
        Check if training should be early terminated due to lack of progress.

        Returns:
            True if training should be terminated (too many consecutive cycles without progress)
        """
        # Tracking is disabled if max_no_progress_cycles <= 0
        if self.max_no_progress_cycles <= 0:
            return False

        if self.no_progress_count >= self.max_no_progress_cycles:
            logger.error(
                f"EARLY TERMINATION: {self.no_progress_count} consecutive training cycles "
                f"without progress (threshold: {self.max_no_progress_cycles}). "
                f"Training stuck at iteration ~{self.last_restart_iteration}. "
                f"This may indicate a non-recoverable failure (hardware issue, bad data, configuration problem, etc.)"
            )
            return True

        return False
