import unittest
from unittest.mock import MagicMock, call

from nvidia_resiliency_ext.fault_tolerance import (
    InvalidStateTransitionException,
    RankMonitorState,
    RankMonitorStateMachine,
)


class TestRankMonitorStateMachine(unittest.TestCase):
    def setUp(self):
        # Initialize the mock logger and RankMonitorStateMachine
        self.logger = MagicMock()
        self.rank = 0
        self.state_machine = RankMonitorStateMachine(self.logger)

    def test_initial_state(self):
        # Test initial state is UNINITIALIZED
        self.assertEqual(self.state_machine.state, RankMonitorState.UNINITIALIZED)

    def test_initialize_transition(self):
        # Valid transition from UNINITIALIZED to INITIALIZE
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.logger.log_for_restarter.assert_called_with(
            "[NestedRestarter] name=[InJob] state=initialize"
        )

    def test_handle_signal_while_uninitialized_to_aborted(self):
        # Attempt the invalid transition and assert that it raises an exception
        with self.assertRaises(InvalidStateTransitionException) as context:
            self.state_machine.transition_to(RankMonitorState.ABORTED)

        # Check that the exception message is as expected
        self.assertEqual(
            str(context.exception),
            "Invalid transition attempted from RankMonitorState.UNINITIALIZED to RankMonitorState.ABORTED",
        )

        # Ensure that the state has not changed and remains UNINITIALIZED
        self.assertEqual(self.state_machine.state, RankMonitorState.UNINITIALIZED)

    def test_invalid_transition_from_uninitialized_to_completed(self):
        # Invalid transition directly to HANDLING_COMPLETED from UNINITIALIZED
        with self.assertRaises(InvalidStateTransitionException) as context:
            self.state_machine.transition_to(RankMonitorState.HANDLING_COMPLETED)

        # Check that the exception message is as expected
        self.assertEqual(
            str(context.exception),
            "Invalid transition attempted from RankMonitorState.UNINITIALIZED to RankMonitorState.HANDLING_COMPLETED",
        )

        # State should not change and should log a warning
        self.assertEqual(self.state_machine.state, RankMonitorState.UNINITIALIZED)

    def test_finalize_to_start_invalid_transition(self):
        # Transition to FINALIZED and attempt an invalid transition to HANDLING_START
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.transition_to(RankMonitorState.FINALIZED)

        # Attempt the invalid transition and assert that it raises an exception
        with self.assertRaises(InvalidStateTransitionException) as context:
            self.state_machine.transition_to(RankMonitorState.HANDLING_START)

        # Check that the exception message is as expected
        self.assertEqual(
            str(context.exception),
            "Invalid transition attempted from RankMonitorState.FINALIZED to RankMonitorState.HANDLING_START",
        )

        # Ensure that the state has not changed and remains FINALIZED
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_handle_signal_from_handling_start(self):
        # Valid signal handling from HANDLING_START, transitioning to ABORTED
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.transition_to(RankMonitorState.HANDLING_START)
        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)
        self.logger.log_for_restarter.assert_called_with(
            "[NestedRestarter] name=[InJob] state=aborted"
        )

    def test_handle_signal_from_initialize(self):
        # Valid signal handling from INITIALIZE, transitioning to FINALIZED
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)
        self.logger.log_for_restarter.assert_called_with(
            "[NestedRestarter] name=[InJob] state=finalized"
        )

    def test_invalid_handle_signal_from_finalized(self):
        # Invalid signal handling from FINALIZED, should remain in FINALIZED
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.transition_to(RankMonitorState.FINALIZED)

        # It is ok to receive signal in FINALIZED state.
        self.state_machine.handle_signal()
        # Ensure that the state has not changed and remains FINALIZED
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_periodic_restart_check_processing_state(self):
        # Transition to HANDLING_PROCESSING and check periodic restart processing
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.transition_to(RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)
        self.logger.log_for_restarter.assert_called_with(
            "[NestedRestarter] name=[InJob] state=handling stage=processing"
        )

    def test_handle_ipc_connection_lost_from_initialize(self):
        # Test IPC connection lost handling from INITIALIZE, valid transition to HANDLING_START
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.logger.log_for_restarter.assert_called_with(
            "[NestedRestarter] name=[InJob] state=handling stage=starting"
        )

    def test_handle_ipc_connection_lost_invalid_state(self):
        # Invalid IPC connection lost handling from ABORTED, should remain ABORTED
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.transition_to(RankMonitorState.HANDLING_START)
        self.state_machine.transition_to(RankMonitorState.ABORTED)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_normal_flow_with_max_restarts_3_with_periodic_check(self):
        # Normal flow with max_restarts = 3
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_normal_flow_2_with_max_restarts_3_with_periodic_check(self):
        # Normal flow with max_restarts = 3
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_handle_signal_while_processing(self):
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_handle_signal_while_starting(self):
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_handle_signal_while_initialized(self):
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_handle_ipc_connection_loss_while_processing(self):
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_handle_ipc_connection_loss_while_starting(self):
        self.state_machine.handle_heartbeat_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_multiple_valid_transitions(self):
        # Test multiple valid transitions in sequence
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.transition_to(RankMonitorState.HANDLING_START)
        self.state_machine.transition_to(RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.transition_to(RankMonitorState.HANDLING_COMPLETED)
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)
        expected_calls = [
            call("[NestedRestarter] name=[InJob] state=initialize"),
            call("[NestedRestarter] name=[InJob] state=handling stage=starting"),
            call("[NestedRestarter] name=[InJob] state=handling stage=processing"),
            call("[NestedRestarter] name=[InJob] state=handling stage=completed"),
        ]
        self.logger.log_for_restarter.assert_has_calls(expected_calls)

    def test_normal_flow_with_max_restarts_3_with_periodic_check_section(self):
        # Normal flow with max_restarts = 3
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_normal_flow_2_with_max_restarts_3_with_periodic_check_section(self):
        # Normal flow with max_restarts = 3
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_handle_signal_while_processing_section(self):
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_handle_signal_while_starting_section(self):
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_handle_signal_while_initialized_section(self):
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)

        self.state_machine.handle_signal()
        self.assertEqual(self.state_machine.state, RankMonitorState.FINALIZED)

    def test_handle_ipc_connection_loss_while_processing_section(self):
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)
        self.state_machine.periodic_restart_check()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_PROCESSING)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_handle_ipc_connection_loss_while_starting_section(self):
        self.state_machine.handle_section_msg()
        self.assertEqual(self.state_machine.state, RankMonitorState.INITIALIZE)
        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_START)

        self.state_machine.handle_ipc_connection_lost()
        self.assertEqual(self.state_machine.state, RankMonitorState.ABORTED)

    def test_multiple_valid_transitions_section(self):
        # Test multiple valid transitions in sequence
        self.state_machine.transition_to(RankMonitorState.INITIALIZE)
        self.state_machine.transition_to(RankMonitorState.HANDLING_START)
        self.state_machine.transition_to(RankMonitorState.HANDLING_PROCESSING)
        self.state_machine.transition_to(RankMonitorState.HANDLING_COMPLETED)
        self.assertEqual(self.state_machine.state, RankMonitorState.HANDLING_COMPLETED)
        expected_calls = [
            call("[NestedRestarter] name=[InJob] state=initialize"),
            call("[NestedRestarter] name=[InJob] state=handling stage=starting"),
            call("[NestedRestarter] name=[InJob] state=handling stage=processing"),
            call("[NestedRestarter] name=[InJob] state=handling stage=completed"),
        ]
        self.logger.log_for_restarter.assert_has_calls(expected_calls)


if __name__ == "__main__":
    unittest.main()
