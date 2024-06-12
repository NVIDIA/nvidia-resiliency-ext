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

from enum import Enum, auto


class InvalidStateTransitionException(Exception):
    """Custom exception for invalid state transitions."""

    pass


class RankMonitorState(Enum):
    UNINITIALIZED = auto()
    INITIALIZE = auto()
    HANDLING_START = auto()
    HANDLING_PROCESSING = auto()
    HANDLING_COMPLETED = auto()
    FINALIZED = auto()
    ABORTED = auto()


class RankMonitorStateMachine:
    def __init__(self, logger):
        self.state = RankMonitorState.UNINITIALIZED
        self.logger = logger

    def is_restarting(self):
        return self.state in [RankMonitorState.HANDLING_START, RankMonitorState.HANDLING_PROCESSING]

    def handle_heartbeat_msg(self):
        self._handle_msg()

    def handle_section_msg(self):
        self._handle_msg()

    def _handle_msg(self):
        if self.state == RankMonitorState.UNINITIALIZED:
            self.transition_to(RankMonitorState.INITIALIZE)
        elif self.state in [RankMonitorState.HANDLING_START, RankMonitorState.HANDLING_PROCESSING]:
            self.transition_to(RankMonitorState.HANDLING_COMPLETED)
        elif self.state in [RankMonitorState.INITIALIZE, RankMonitorState.HANDLING_COMPLETED]:
            pass  # it is perfectly fine if we have INITILIZE or HANDLING_COMPLETED state during handle_heartbeat_msg
        else:
            raise AssertionError(f"Unexpected _handle_msg call, current state: {self.state}")

    def periodic_restart_check(self):
        if self.state in [RankMonitorState.HANDLING_START, RankMonitorState.HANDLING_PROCESSING]:
            self.transition_to(RankMonitorState.HANDLING_PROCESSING)
        else:
            raise AssertionError(
                f"Unexpected periodic_restart_check all, current state: {self.state}"
            )

    def handle_ipc_connection_lost(self):
        if self.state in [RankMonitorState.INITIALIZE, RankMonitorState.HANDLING_COMPLETED]:
            self.transition_to(RankMonitorState.HANDLING_START)
        elif self.state in [RankMonitorState.HANDLING_START, RankMonitorState.HANDLING_PROCESSING]:
            self.transition_to(RankMonitorState.ABORTED)
        elif self.state == RankMonitorState.ABORTED:
            pass  # we are already aborted from in-job, ingnore following handle_ipc_connection_lost
        elif self.state == RankMonitorState.UNINITIALIZED:
            pass  # ignore in case of not yet initialized state
        else:
            raise AssertionError(
                f"Unexpected handle_ipc_connection_lost all, current state: {self.state}"
            )

    def handle_signal(self):
        if self.state in [RankMonitorState.HANDLING_START, RankMonitorState.HANDLING_PROCESSING]:
            self.transition_to(RankMonitorState.ABORTED)
        elif self.state == RankMonitorState.UNINITIALIZED:
            pass  # ignore in case of not yet initialized state
        else:
            self.transition_to(RankMonitorState.FINALIZED)

    def transition_to(self, new_state):
        if self.can_transition_to(new_state):
            self._log_state_transition(new_state)
            self.state = new_state
        else:
            error_message = f"Invalid transition attempted from {self.state} to {new_state}"
            self.logger.log_for_restarter(error_message)
            raise InvalidStateTransitionException(error_message)

    def can_transition_to(self, new_state):
        allowed_transitions = {
            RankMonitorState.UNINITIALIZED: [
                RankMonitorState.INITIALIZE,
                RankMonitorState.FINALIZED,
            ],
            RankMonitorState.INITIALIZE: [
                RankMonitorState.HANDLING_START,
                RankMonitorState.FINALIZED,
            ],
            RankMonitorState.HANDLING_START: [
                RankMonitorState.HANDLING_COMPLETED,
                RankMonitorState.HANDLING_PROCESSING,
                RankMonitorState.ABORTED,
            ],
            RankMonitorState.HANDLING_PROCESSING: [
                RankMonitorState.HANDLING_COMPLETED,
                RankMonitorState.HANDLING_PROCESSING,
                RankMonitorState.ABORTED,
            ],
            RankMonitorState.HANDLING_COMPLETED: [
                RankMonitorState.HANDLING_START,
                RankMonitorState.FINALIZED,
            ],
            RankMonitorState.ABORTED: [RankMonitorState.FINALIZED],
            RankMonitorState.FINALIZED: [],
        }
        return new_state in allowed_transitions[self.state]

    def _log_state_transition(self, new_state):
        if new_state == RankMonitorState.INITIALIZE:
            self.logger.log_for_restarter("[NestedRestarter] name=[InJob] state=initialize")
        elif new_state == RankMonitorState.HANDLING_START:
            self.logger.log_for_restarter(
                "[NestedRestarter] name=[InJob] state=handling stage=starting"
            )
        elif new_state == RankMonitorState.HANDLING_PROCESSING:
            self.logger.log_for_restarter(
                "[NestedRestarter] name=[InJob] state=handling stage=processing"
            )
        elif new_state == RankMonitorState.HANDLING_COMPLETED:
            self.logger.log_for_restarter(
                "[NestedRestarter] name=[InJob] state=handling stage=completed"
            )
        elif new_state == RankMonitorState.FINALIZED:
            self.logger.log_for_restarter("[NestedRestarter] name=[InJob] state=finalized")
        elif new_state == RankMonitorState.ABORTED:
            self.logger.log_for_restarter("[NestedRestarter] name=[InJob] state=aborted")
