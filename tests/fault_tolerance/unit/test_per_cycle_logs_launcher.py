# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for launcher pipe functionality in per_cycle_logs module."""

import os
import tempfile
import time

import pytest

from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import (
    MultiplexingReaderThread,
    PipeBasedLogsSpecs,
)


class TestLauncherPipeFunctionality:
    """Tests for launcher pipe functionality."""

    @pytest.fixture
    def temp_log_files(self):
        """Create temporary log files."""
        fd1, worker_log = tempfile.mkstemp(suffix='_worker.log')
        fd2, launcher_log = tempfile.mkstemp(suffix='_launcher.log')
        os.close(fd1)
        os.close(fd2)
        yield worker_log, launcher_log
        # Cleanup
        for path in [worker_log, launcher_log]:
            if os.path.exists(path):
                os.remove(path)

    def test_launcher_pipe_no_prefix(self, temp_log_files):
        """Test that launcher logs have no rank prefix."""
        worker_log, launcher_log = temp_log_files

        # Create launcher pipe
        launcher_read_fd, launcher_write_fd = os.pipe()

        # Create worker pipe
        worker_read_fd, worker_write_fd = os.pipe()

        # Create reader thread with both launcher and worker pipes
        reader = MultiplexingReaderThread(
            pipes={0: worker_read_fd},  # Worker rank 0
            log_file_path=worker_log,
            world_size=1,
            local_to_global_rank={0: 0},
            launcher_pipe_fd=launcher_read_fd,
            launcher_log_file_path=launcher_log,
        )
        reader.start()

        # Write to launcher pipe
        os.write(launcher_write_fd, b'Launcher log line 1\n')
        os.write(launcher_write_fd, b'Launcher log line 2\n')

        # Write to worker pipe
        os.write(worker_write_fd, b'Worker log line 1\n')
        os.write(worker_write_fd, b'Worker log line 2\n')

        # Close pipes
        os.close(launcher_write_fd)
        os.close(worker_write_fd)

        # Wait for thread to process
        reader.shutdown()
        reader.join(timeout=5.0)

        # Check launcher log (no prefix)
        with open(launcher_log, 'r') as f:
            launcher_content = f.read()
        assert 'Launcher log line 1\n' in launcher_content
        assert 'Launcher log line 2\n' in launcher_content
        assert '0:' not in launcher_content  # No rank prefix

        # Check worker log (with prefix)
        with open(worker_log, 'r') as f:
            worker_content = f.read()
        assert '0: Worker log line 1\n' in worker_content
        assert '0: Worker log line 2\n' in worker_content

    def test_launcher_only_no_workers(self, temp_log_files):
        """Test launcher pipe with no worker pipes."""
        worker_log, launcher_log = temp_log_files

        # Create launcher pipe
        launcher_read_fd, launcher_write_fd = os.pipe()

        # Create reader thread with only launcher pipe
        reader = MultiplexingReaderThread(
            pipes={},  # No worker pipes
            log_file_path=worker_log,  # Won't be used but needs valid path
            world_size=None,
            local_to_global_rank={},
            launcher_pipe_fd=launcher_read_fd,
            launcher_log_file_path=launcher_log,
        )
        reader.start()

        # Write to launcher pipe
        os.write(launcher_write_fd, b'Launcher only line 1\n')
        os.write(launcher_write_fd, b'Launcher only line 2\n')

        # Close pipe
        os.close(launcher_write_fd)

        # Give thread time to process (since no pipes, it relies on polling timeout)
        time.sleep(0.5)

        # Shutdown thread
        reader.shutdown()
        reader.join(timeout=2.0)

        # Check launcher log
        with open(launcher_log, 'r') as f:
            content = f.read()
        assert 'Launcher only line 1\n' in content
        assert 'Launcher only line 2\n' in content

    def test_launcher_pipe_incomplete_line(self, temp_log_files):
        """Test that incomplete launcher logs are flushed properly."""
        worker_log, launcher_log = temp_log_files

        # Create launcher pipe
        launcher_read_fd, launcher_write_fd = os.pipe()

        # Create reader thread
        reader = MultiplexingReaderThread(
            pipes={},
            log_file_path=worker_log,
            world_size=None,
            local_to_global_rank={},
            launcher_pipe_fd=launcher_read_fd,
            launcher_log_file_path=launcher_log,
        )
        reader.start()

        # Write incomplete line (no trailing newline)
        os.write(launcher_write_fd, b'Incomplete launcher line')

        # Close pipe (should flush incomplete line)
        os.close(launcher_write_fd)

        # Give thread time to process
        time.sleep(0.5)

        # Shutdown thread
        reader.shutdown()
        reader.join(timeout=2.0)

        # Check that incomplete line was flushed
        with open(launcher_log, 'r') as f:
            content = f.read()
        assert 'Incomplete launcher line\n' in content

    def test_pipe_based_logs_specs_early_thread_start(self):
        """Test that PipeBasedLogsSpecs starts thread immediately when launcher pipe provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_log = os.path.join(tmpdir, "base.log")
            launcher_log = os.path.join(tmpdir, "launcher.log")

            # Create launcher pipe
            launcher_read_fd, launcher_write_fd = os.pipe()

            # Create PipeBasedLogsSpecs - should start thread immediately
            logs_specs = PipeBasedLogsSpecs(
                base_log_file=base_log,
                launcher_pipe_fd=launcher_read_fd,
                launcher_log_file=launcher_log,
            )

            # Verify thread was started
            assert logs_specs._reader_thread is not None
            assert logs_specs._reader_thread.is_alive()

            # Write to launcher pipe
            os.write(launcher_write_fd, b'Early launcher log\n')
            os.close(launcher_write_fd)

            # Give thread time to process
            time.sleep(0.1)

            # Cleanup
            logs_specs.cleanup()

            # Verify log was written to launcher log file
            assert os.path.exists(launcher_log)
            with open(launcher_log, 'r') as f:
                content = f.read()
            assert 'Early launcher log\n' in content

    def test_update_pipes_preserves_launcher_pipe(self, temp_log_files):
        """Test that update_pipes maintains launcher pipe across cycles."""
        worker_log1, launcher_log = temp_log_files
        worker_log2 = worker_log1.replace('_worker.log', '_worker2.log')

        try:
            # Create launcher pipe
            launcher_read_fd, launcher_write_fd = os.pipe()

            # Create first worker pipe
            worker_read_fd1, worker_write_fd1 = os.pipe()

            # Create reader thread
            reader = MultiplexingReaderThread(
                pipes={0: worker_read_fd1},
                log_file_path=worker_log1,
                world_size=1,
                local_to_global_rank={0: 0},
                launcher_pipe_fd=launcher_read_fd,
                launcher_log_file_path=launcher_log,
            )
            reader.start()

            # Write to both pipes
            os.write(launcher_write_fd, b'Launcher cycle 0\n')
            os.write(worker_write_fd1, b'Worker cycle 0\n')
            os.close(worker_write_fd1)

            # Give time to process
            time.sleep(0.1)

            # Update pipes for new cycle (simulate restart)
            worker_read_fd2, worker_write_fd2 = os.pipe()
            reader.update_pipes(
                new_pipes={0: worker_read_fd2},
                new_log_file=worker_log2,
                new_world_size=1,
                new_local_to_global_rank={0: 0},
                new_launcher_pipe_fd=launcher_read_fd,  # Same launcher pipe
                new_launcher_log_file=launcher_log,
            )

            # Write to both pipes again
            os.write(launcher_write_fd, b'Launcher cycle 1\n')
            os.write(worker_write_fd2, b'Worker cycle 1\n')
            os.close(worker_write_fd2)
            os.close(launcher_write_fd)

            # Wait for thread
            reader.shutdown()
            reader.join(timeout=5.0)

            # Check launcher log has both cycles
            with open(launcher_log, 'r') as f:
                launcher_content = f.read()
            assert 'Launcher cycle 0\n' in launcher_content
            assert 'Launcher cycle 1\n' in launcher_content

            # Check worker logs are separate
            with open(worker_log1, 'r') as f:
                assert 'Worker cycle 0\n' in f.read()
            with open(worker_log2, 'r') as f:
                assert 'Worker cycle 1\n' in f.read()

        finally:
            if os.path.exists(worker_log2):
                os.remove(worker_log2)
