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

import errno
import signal
import unittest
from unittest.mock import patch

from nvidia_resiliency_ext.shared_utils.auto_restart import fork_and_monitor


class TestAutoRestart(unittest.TestCase):
    """Test cases for the monitor_and_restart function."""

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WIFEXITED')
    @patch('os.WEXITSTATUS')
    @patch('os.WIFSIGNALED')
    @patch('os.WTERMSIG')
    @patch('time.sleep')
    @patch('sys.stderr.write')
    def test_successful_child_exit(
        self,
        mock_stderr_write,
        mock_sleep,
        mock_wtermsig,
        mock_wifsignaled,
        mock_wexitstatus,
        mock_wifexited,
        mock_waitpid,
        mock_fork,
    ):
        """Test that parent exits when child exits with status 0."""
        # Mock fork to return 0 (child process)
        mock_fork.return_value = 0

        # Call the function - should return immediately in child
        fork_and_monitor()

        # Verify fork was called
        mock_fork.assert_called_once()
        # Verify stderr message was written (with timestamp)
        mock_stderr_write.assert_called_once()
        call_args = mock_stderr_write.call_args[0][0]
        self.assertIn("Starting auto-restart mode\n", call_args)

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WIFEXITED')
    @patch('os.WEXITSTATUS')
    @patch('os.WIFSIGNALED')
    @patch('os.WTERMSIG')
    @patch('time.sleep')
    @patch('sys.stderr.write')
    @patch('os._exit')
    def test_parent_monitors_child_success(
        self,
        mock_exit,
        mock_stderr_write,
        mock_sleep,
        mock_wtermsig,
        mock_wifsignaled,
        mock_wexitstatus,
        mock_wifexited,
        mock_waitpid,
        mock_fork,
    ):
        """Test parent process monitoring when child exits successfully."""
        # Mock fork to return child PID (parent process)
        mock_fork.return_value = 12345

        # Mock waitpid to return child exit info
        mock_waitpid.return_value = (12345, 0)

        # Mock exit status checks
        mock_wifexited.return_value = True
        mock_wexitstatus.return_value = 0
        mock_wifsignaled.return_value = False

        # Mock os._exit to raise SystemExit for testing
        def mock_exit_func(status):
            raise SystemExit(status)

        mock_exit.side_effect = mock_exit_func

        # Mock stderr to capture output
        stderr_calls = []

        def mock_write(msg):
            stderr_calls.append(msg)

        mock_stderr_write.side_effect = mock_write

        # Call the function - should exit in parent
        with self.assertRaises(SystemExit) as cm:
            fork_and_monitor()

        # Verify exit status
        self.assertEqual(cm.exception.code, 0)

        # Verify the monitoring sequence
        mock_fork.assert_called_once()
        mock_waitpid.assert_called_once_with(12345, 0)
        mock_wifexited.assert_called_once_with(0)
        mock_wexitstatus.assert_called_once_with(0)
        mock_exit.assert_called_once_with(0)

        # Verify stderr messages (with timestamps)
        self.assertEqual(len(stderr_calls), 3)
        self.assertIn("Starting auto-restart mode\n", stderr_calls[0])
        self.assertIn("Launched child 12345\n", stderr_calls[1])
        self.assertIn("Child 12345 exited with status 0\n", stderr_calls[2])

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WIFEXITED')
    @patch('os.WEXITSTATUS')
    @patch('os.WIFSIGNALED')
    @patch('os.WTERMSIG')
    @patch('time.sleep')
    @patch('sys.stderr.write')
    @patch('os._exit')
    def test_parent_monitors_child_failure_and_restart(
        self,
        mock_exit,
        mock_stderr_write,
        mock_sleep,
        mock_wtermsig,
        mock_wifsignaled,
        mock_wexitstatus,
        mock_wifexited,
        mock_waitpid,
        mock_fork,
    ):
        """Test parent process restarts child when it fails."""
        # Mock fork to return child PID (parent process)
        mock_fork.return_value = 12345

        # Mock waitpid to return child exit info
        mock_waitpid.return_value = (12345, 256)  # Exit code 1

        # Mock exit status checks - child failed
        mock_wifexited.return_value = True
        mock_wexitstatus.return_value = 1
        mock_wifsignaled.return_value = False

        # Mock stderr to capture output
        stderr_calls = []

        def mock_write(msg):
            stderr_calls.append(msg)

        mock_stderr_write.side_effect = mock_write

        # Mock sleep to avoid actual delay in test
        def mock_sleep_func(delay):
            # Break out of the loop after first iteration
            raise KeyboardInterrupt("Test complete")

        mock_sleep.side_effect = mock_sleep_func

        # Call the function - should restart child
        with self.assertRaises(KeyboardInterrupt):
            fork_and_monitor()

        # Verify the monitoring sequence
        mock_fork.assert_called_once()
        mock_waitpid.assert_called_once_with(12345, 0)
        mock_wifexited.assert_called_once_with(256)
        mock_wexitstatus.assert_called_once_with(256)
        mock_sleep.assert_called_once_with(5.0)  # Default restart delay

        # Verify stderr messages (with timestamps)
        self.assertEqual(len(stderr_calls), 3)
        self.assertIn("Starting auto-restart mode\n", stderr_calls[0])
        self.assertIn("Launched child 12345\n", stderr_calls[1])
        self.assertIn("Child 12345 exited with status 1\n", stderr_calls[2])

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WIFEXITED')
    @patch('os.WEXITSTATUS')
    @patch('os.WIFSIGNALED')
    @patch('os.WTERMSIG')
    @patch('time.sleep')
    @patch('sys.stderr.write')
    @patch('os._exit')
    def test_parent_monitors_child_signal_termination(
        self,
        mock_exit,
        mock_stderr_write,
        mock_sleep,
        mock_wtermsig,
        mock_wifsignaled,
        mock_wexitstatus,
        mock_wifexited,
        mock_waitpid,
        mock_fork,
    ):
        """Test parent process handles child terminated by signal."""
        # Mock fork to return child PID (parent process)
        mock_fork.return_value = 12345

        # Mock waitpid to return child exit info
        mock_waitpid.return_value = (12345, 9)  # SIGKILL

        # Mock exit status checks - child was signaled
        mock_wifexited.return_value = False
        mock_wifsignaled.return_value = True
        mock_wtermsig.return_value = signal.SIGKILL

        # Mock stderr to capture output
        stderr_calls = []

        def mock_write(msg):
            stderr_calls.append(msg)

        mock_stderr_write.side_effect = mock_write

        # Mock sleep to avoid actual delay in test
        def mock_sleep_func(delay):
            # Break out of the loop after first iteration
            raise KeyboardInterrupt("Test complete")

        mock_sleep.side_effect = mock_sleep_func

        # Call the function - should restart child
        with self.assertRaises(KeyboardInterrupt):
            fork_and_monitor()

        # Verify the monitoring sequence
        mock_fork.assert_called_once()
        mock_waitpid.assert_called_once_with(12345, 0)
        mock_wifexited.assert_called_once_with(9)
        mock_wifsignaled.assert_called_once_with(9)
        mock_wtermsig.assert_called_once_with(9)
        mock_sleep.assert_called_once_with(5.0)

        # Verify stderr messages (with timestamps)
        self.assertEqual(len(stderr_calls), 3)
        self.assertIn("Starting auto-restart mode\n", stderr_calls[0])
        self.assertIn("Launched child 12345\n", stderr_calls[1])
        self.assertIn("Child 12345 exited with signal 9\n", stderr_calls[2])

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WIFEXITED')
    @patch('os.WEXITSTATUS')
    @patch('os.WIFSIGNALED')
    @patch('os.WTERMSIG')
    @patch('time.sleep')
    @patch('sys.stderr.write')
    @patch('os._exit')
    def test_parent_monitors_child_abnormal_exit(
        self,
        mock_exit,
        mock_stderr_write,
        mock_sleep,
        mock_wtermsig,
        mock_wifsignaled,
        mock_wexitstatus,
        mock_wifexited,
        mock_waitpid,
        mock_fork,
    ):
        """Test parent process handles abnormal child exit."""
        # Mock fork to return child PID (parent process)
        mock_fork.return_value = 12345

        # Mock waitpid to return child exit info
        mock_waitpid.return_value = (12345, 0x1234)  # Abnormal exit

        # Mock exit status checks - abnormal exit
        mock_wifexited.return_value = False
        mock_wifsignaled.return_value = False

        # Mock stderr to capture output
        stderr_calls = []

        def mock_write(msg):
            stderr_calls.append(msg)

        mock_stderr_write.side_effect = mock_write

        # Mock sleep to avoid actual delay in test
        def mock_sleep_func(delay):
            # Break out of the loop after first iteration
            raise KeyboardInterrupt("Test complete")

        mock_sleep.side_effect = mock_sleep_func

        # Call the function - should restart child
        with self.assertRaises(KeyboardInterrupt):
            fork_and_monitor()

        # Verify the monitoring sequence
        mock_fork.assert_called_once()
        mock_waitpid.assert_called_once_with(12345, 0)
        mock_wifexited.assert_called_once_with(0x1234)
        mock_sleep.assert_called_once_with(5.0)

        # Verify stderr messages (with timestamps)
        self.assertEqual(len(stderr_calls), 3)
        self.assertIn("Starting auto-restart mode\n", stderr_calls[0])
        self.assertIn("Launched child 12345\n", stderr_calls[1])
        self.assertIn("Child 12345 exited abnormally (1234)\n", stderr_calls[2])

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os.WIFEXITED')
    @patch('os.WEXITSTATUS')
    @patch('os.WIFSIGNALED')
    @patch('os.WTERMSIG')
    @patch('time.sleep')
    @patch('sys.stderr.write')
    @patch('os._exit')
    def test_custom_restart_delay(
        self,
        mock_exit,
        mock_stderr_write,
        mock_sleep,
        mock_wtermsig,
        mock_wifsignaled,
        mock_wexitstatus,
        mock_wifexited,
        mock_waitpid,
        mock_fork,
    ):
        """Test that custom restart delay is used."""
        # Mock fork to return child PID (parent process)
        mock_fork.return_value = 12345

        # Mock waitpid to return child exit info
        mock_waitpid.return_value = (12345, 256)  # Exit code 1

        # Mock exit status checks - child failed
        mock_wifexited.return_value = True
        mock_wexitstatus.return_value = 1
        mock_wifsignaled.return_value = False

        # Mock sleep to avoid actual delay in test
        def mock_sleep_func(delay):
            # Break out of the loop after first iteration
            raise KeyboardInterrupt("Test complete")

        mock_sleep.side_effect = mock_sleep_func

        # Call the function with custom delay
        with self.assertRaises(KeyboardInterrupt):
            fork_and_monitor(restart_delay=10.0)

        # Verify custom delay was used
        mock_sleep.assert_called_once_with(10.0)

    @patch('os.fork')
    @patch('sys.stderr.write')
    @patch('os._exit')
    def test_fork_failure(self, mock_exit, mock_stderr_write, mock_fork):
        """Test handling of fork failure."""
        # Mock fork to raise OSError
        mock_fork.side_effect = OSError("Fork failed")

        # Mock os._exit to raise SystemExit for testing
        def mock_exit_func(status):
            raise SystemExit(status)

        mock_exit.side_effect = mock_exit_func

        # Call the function - should exit with status 1
        with self.assertRaises(SystemExit) as cm:
            fork_and_monitor()

        # Verify exit status
        self.assertEqual(cm.exception.code, 1)

        # Verify stderr messages were written (with timestamps)
        self.assertEqual(mock_stderr_write.call_count, 2)
        call_args_list = [call[0][0] for call in mock_stderr_write.call_args_list]
        self.assertIn("Starting auto-restart mode\n", call_args_list[0])
        self.assertIn("fork: Fork failed\n", call_args_list[1])

        # Verify os._exit was called with status 1
        mock_exit.assert_called_once_with(1)

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os._exit')
    def test_waitpid_interrupted(self, mock_exit, mock_waitpid, mock_fork):
        """Test handling of waitpid interruption."""
        # Mock fork to return child PID (parent process)
        mock_fork.return_value = 12345

        # Mock waitpid to raise EINTR first, then succeed
        mock_waitpid.side_effect = [OSError(errno.EINTR, "Interrupted"), (12345, 0)]

        # Mock os._exit to raise SystemExit for testing
        def mock_exit_func(status):
            raise SystemExit(status)

        mock_exit.side_effect = mock_exit_func

        # Mock exit status checks
        with (
            patch('os.WIFEXITED', return_value=True),
            patch('os.WEXITSTATUS', return_value=0),
            patch('os.WIFSIGNALED', return_value=False),
            patch('sys.stderr.write'),
        ):

            # Call the function - should exit in parent with status 0
            with self.assertRaises(SystemExit) as cm:
                fork_and_monitor()

            # Verify exit status
            self.assertEqual(cm.exception.code, 0)

        # Verify waitpid was called twice (retry after EINTR)
        self.assertEqual(mock_waitpid.call_count, 2)

    @patch('os.fork')
    @patch('os.waitpid')
    @patch('os._exit')
    def test_waitpid_other_error(self, mock_exit, mock_waitpid, mock_fork):
        """Test handling of waitpid error other than EINTR."""
        # Mock fork to return child PID (parent process)
        mock_fork.return_value = 12345

        # Mock waitpid to raise non-EINTR error
        mock_waitpid.side_effect = OSError(errno.EPERM, "Other error")

        # Mock os._exit to raise SystemExit for testing
        def mock_exit_func(status):
            raise SystemExit(status)

        mock_exit.side_effect = mock_exit_func

        # Call the function - should exit with status 1
        with self.assertRaises(SystemExit) as cm:
            fork_and_monitor()

        # Verify exit status
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
