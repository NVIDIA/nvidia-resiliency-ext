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

import io
import logging
import re
import unittest

from nvidia_resiliency_ext.inprocess.nested_restarter import (
    NestedRestarterCallback,
    NestedRestarterHandlingCompleted,
    NestedRestarterHandlingStarting,
    NestedRestarterLogger,
)
from nvidia_resiliency_ext.inprocess.state import State

from . import common


@common.apply_all_tests(common.retry())
class TestNestedRestarterLogging(unittest.TestCase):
    """Test that the nested restarter logging produces messages that match expected regex patterns"""

    def setUp(self):
        # Create a frozen state for testing
        self.state = State(
            rank=0,
            world_size=2,
            active_rank=0,
            active_world_size=2,
            initial_rank=0,
            initial_world_size=2,
        ).freeze()

        # Capture log output
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_formatter = logging.Formatter("%(message)s")
        self.log_handler.setFormatter(self.log_formatter)

        # Create logger
        self.logger = NestedRestarterLogger()
        self.logger.restarter_logger.handlers = [self.log_handler]
        self.logger.restarter_logger.setLevel(logging.DEBUG)

    def test_callback_log_matches_state_regex(self):
        """Test that callback logs match regex pattern with state only"""
        # Define the regex pattern for state
        state_regex = r'^.*NestedRestarter.*state\=(?P<state>[a-zA-Z]+).*$'

        # Create a callback with state only
        callback = NestedRestarterCallback(
            restarter_state="initialize", logger=self.logger, special_rank=0
        )

        # Execute the callback
        callback(self.state)

        # Get the log output
        log_output = self.log_capture.getvalue()

        # Check if the log message matches the regex
        match = re.search(state_regex, log_output, re.MULTILINE)
        self.assertIsNotNone(match, f"Log output doesn't match the state regex: {log_output}")
        self.assertEqual(match.group('state'), "initialize")

    def test_callback_log_matches_state_and_stage_regex(self):
        """Test that callback logs match regex pattern with state and stage"""
        # Define the regex pattern for state and stage
        state_stage_regex = (
            r'^.*NestedRestarter.*state\=(?P<state>[a-zA-Z]+)\s+stage\=(?P<stage>[a-zA-Z]+).*$'
        )

        # Create a callback with both state and stage
        callback = NestedRestarterCallback(
            restarter_state="handling",
            restarter_stage="starting",
            logger=self.logger,
            special_rank=0,
        )

        # Execute the callback
        callback(self.state)

        # Get the log output
        log_output = self.log_capture.getvalue()

        # Check if the log message matches the regex
        match = re.search(state_stage_regex, log_output, re.MULTILINE)
        self.assertIsNotNone(
            match, f"Log output doesn't match the state and stage regex: {log_output}"
        )
        self.assertEqual(match.group('state'), "handling")
        self.assertEqual(match.group('stage'), "starting")

    def test_handling_completed_logs(self):
        """Test that NestedRestarterHandlingCompleted produces logs that match the regex"""
        # Create a handling completed instance
        handling_completed = NestedRestarterHandlingCompleted(special_rank=0)
        handling_completed.logger = self.logger

        # Call twice to test both state transitions
        handling_completed(
            self.state
        )  # First call, transitions from initialize to handling/completed

        # Get the log output
        log_output = self.log_capture.getvalue()

        # Define the regex patterns
        state_regex = r'^.*NestedRestarter.*state\=(?P<state>[a-zA-Z]+).*$'
        state_stage_regex = (
            r'^.*NestedRestarter.*state\=(?P<state>[a-zA-Z]+)\s+stage\=(?P<stage>[a-zA-Z]+).*$'
        )

        # Check for initial state (should be 'initialize' first)
        first_match = re.search(state_regex, log_output, re.MULTILINE)
        self.assertIsNotNone(first_match, f"Log output doesn't match the state regex: {log_output}")
        self.assertEqual(first_match.group('state'), "initialize")

        # Reset log capture to check for the state after transition
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setFormatter(self.log_formatter)
        self.logger.restarter_logger.handlers = [self.log_handler]

        # Call again to check that the state has transitioned
        handling_completed(self.state)

        log_output = self.log_capture.getvalue()

        # Now it should have state 'handling' and stage 'completed'
        second_match = re.search(state_stage_regex, log_output, re.MULTILINE)
        self.assertIsNotNone(
            second_match, f"Log output doesn't match the state and stage regex: {log_output}"
        )
        self.assertEqual(second_match.group('state'), "handling")
        self.assertEqual(second_match.group('stage'), "completed")

    def test_handling_starting_logs(self):
        """Test that NestedRestarterHandlingStarting produces logs that match the regex"""
        # Create a handling starting instance
        handling_starting = NestedRestarterHandlingStarting()
        handling_starting.logger = self.logger

        # Execute the handler
        handling_starting(self.state)

        # Get the log output
        log_output = self.log_capture.getvalue()

        # Define the regex pattern
        state_stage_regex = (
            r'^.*NestedRestarter.*state\=(?P<state>[a-zA-Z]+)\s+stage\=(?P<stage>[a-zA-Z]+).*$'
        )

        # Check if the log message matches the regex
        match = re.search(state_stage_regex, log_output, re.MULTILINE)
        self.assertIsNotNone(
            match, f"Log output doesn't match the state and stage regex: {log_output}"
        )
        self.assertEqual(match.group('state'), "handling")
        self.assertEqual(match.group('stage'), "starting")


if __name__ == '__main__':
    unittest.main()
