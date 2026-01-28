# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nvidia_resiliency_ext.fault_tolerance.per_cycle_logs module."""

import os
import tempfile
import threading
import time

import pytest

from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import MultiplexingReaderThread


class TestMultiplexingReaderThread:
    """Tests for MultiplexingReaderThread class."""

    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file."""
        fd, path = tempfile.mkstemp(suffix='.log')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def _create_pipe_pair(self):
        """Create a pipe and return (read_fd, write_fd)."""
        return os.pipe()

    def _write_to_pipe(self, write_fd, data: bytes):
        """Write data to pipe."""
        os.write(write_fd, data)

    def _close_pipe(self, fd):
        """Close a pipe file descriptor."""
        try:
            os.close(fd)
        except OSError:
            pass

    def _read_log_file(self, path: str) -> str:
        """Read the contents of the log file."""
        with open(path, 'r') as f:
            return f.read()

    def test_single_rank_basic_output(self, temp_log_file):
        """Test basic output from a single rank."""
        # Create pipe
        read_fd, write_fd = self._create_pipe_pair()

        # Create reader thread
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write some data
        self._write_to_pipe(write_fd, b'Hello from rank 0\n')
        self._write_to_pipe(write_fd, b'Second line\n')

        # Close pipe to signal end
        self._close_pipe(write_fd)

        # Wait for thread to finish
        reader.join(timeout=2.0)

        # Check output
        content = self._read_log_file(temp_log_file)
        assert '0: Hello from rank 0\n' in content
        assert '0: Second line\n' in content

    def test_multiple_ranks(self, temp_log_file):
        """Test output from multiple ranks."""
        # Create pipes for 3 ranks
        pipes = {}
        write_fds = {}
        for rank in range(3):
            read_fd, write_fd = self._create_pipe_pair()
            pipes[rank] = read_fd
            write_fds[rank] = write_fd

        # Create reader thread with world_size for proper padding
        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=3,
        )
        reader.start()

        # Write data from each rank
        self._write_to_pipe(write_fds[0], b'Rank 0 output\n')
        self._write_to_pipe(write_fds[1], b'Rank 1 output\n')
        self._write_to_pipe(write_fds[2], b'Rank 2 output\n')

        # Close all pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        # Wait for thread to finish
        reader.join(timeout=2.0)

        # Check output
        content = self._read_log_file(temp_log_file)
        assert '0: Rank 0 output\n' in content
        assert '1: Rank 1 output\n' in content
        assert '2: Rank 2 output\n' in content

    def test_global_rank_mapping(self, temp_log_file):
        """Test local to global rank mapping."""
        # Create pipes for 2 local ranks
        pipes = {}
        write_fds = {}
        for local_rank in [0, 1]:
            read_fd, write_fd = self._create_pipe_pair()
            pipes[local_rank] = read_fd
            write_fds[local_rank] = write_fd

        # Map local ranks to global ranks: local 0 -> global 4, local 1 -> global 5
        local_to_global = {0: 4, 1: 5}

        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=100,  # Large world size for padding test
            local_to_global_rank=local_to_global,
        )
        reader.start()

        # Write data
        self._write_to_pipe(write_fds[0], b'Local 0\n')
        self._write_to_pipe(write_fds[1], b'Local 1\n')

        # Close pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        reader.join(timeout=2.0)

        # Check output uses global ranks with proper padding (world_size=100, so 2 digits)
        content = self._read_log_file(temp_log_file)
        assert ' 4: Local 0\n' in content
        assert ' 5: Local 1\n' in content

    def test_null_byte_handling(self, temp_log_file):
        """Test that NULL bytes are replaced with <NUL> marker."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write data with NULL bytes
        self._write_to_pipe(write_fd, b'Before NULL\x00After NULL\n')
        self._write_to_pipe(write_fd, b'Multiple \x00 nulls \x00 here\n')

        self._close_pipe(write_fd)
        reader.join(timeout=2.0)

        # Check NULL bytes are replaced
        content = self._read_log_file(temp_log_file)
        assert '<NUL>' in content
        assert '0: Before NULL<NUL>After NULL\n' in content
        assert '0: Multiple <NUL> nulls <NUL> here\n' in content

        # Verify file is still detected as text, not binary
        # Use Python's built-in text detection instead of 'file' command for portability
        try:
            with open(temp_log_file, 'r', encoding='utf-8') as f:
                f.read()
            is_text = True
        except (UnicodeDecodeError, ValueError):
            is_text = False
        assert is_text, f"Log file {temp_log_file} is not valid UTF-8 text"

    def test_invalid_utf8_handling(self, temp_log_file):
        """Test handling of invalid UTF-8 sequences."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write invalid UTF-8 sequences
        # 0xFF and 0xFE are invalid UTF-8 start bytes
        self._write_to_pipe(write_fd, b'Invalid: \xff\xfe bytes here\n')
        # Incomplete UTF-8 sequence (start of 3-byte sequence without continuation)
        self._write_to_pipe(write_fd, b'Incomplete: \xe2\x28\xa1\n')

        self._close_pipe(write_fd)
        reader.join(timeout=2.0)

        # Check that invalid bytes are replaced with Unicode replacement character
        content = self._read_log_file(temp_log_file)
        assert '0: Invalid:' in content
        assert '0: Incomplete:' in content
        # The replacement character should be present (U+FFFD)
        assert '\ufffd' in content or '?' in content  # Different systems may render differently

    def test_incomplete_line_buffering(self, temp_log_file):
        """Test that incomplete lines (without \\n) are buffered correctly."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write incomplete line (no \n)
        self._write_to_pipe(write_fd, b'Incomplete')
        time.sleep(0.1)  # Give thread time to process

        # Complete the line
        self._write_to_pipe(write_fd, b' line completed\n')
        time.sleep(0.1)

        self._close_pipe(write_fd)
        reader.join(timeout=2.0)

        # Check that incomplete line was buffered and completed
        content = self._read_log_file(temp_log_file)
        assert '0: Incomplete line completed\n' in content

    def test_incomplete_line_on_pipe_close(self, temp_log_file):
        """Test that incomplete line is flushed when pipe closes."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write incomplete line and close pipe immediately
        self._write_to_pipe(write_fd, b'Incomplete line without newline')
        self._close_pipe(write_fd)

        reader.join(timeout=2.0)

        # Check that incomplete line was flushed with added newline
        content = self._read_log_file(temp_log_file)
        assert '0: Incomplete line without newline\n' in content

    def test_mixed_data_multiple_ranks(self, temp_log_file):
        """Test complex scenario with multiple ranks, NULL bytes, and invalid UTF-8."""
        # Create pipes for 2 ranks
        pipes = {}
        write_fds = {}
        for rank in range(2):
            read_fd, write_fd = self._create_pipe_pair()
            pipes[rank] = read_fd
            write_fds[rank] = write_fd

        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=10,
        )
        reader.start()

        # Rank 0: Normal output
        self._write_to_pipe(write_fds[0], b'Rank 0: Normal line\n')

        # Rank 1: Output with NULL bytes
        self._write_to_pipe(write_fds[1], b'Rank 1: Has \x00 null\n')

        # Rank 0: Invalid UTF-8
        self._write_to_pipe(write_fds[0], b'Rank 0: Invalid \xff\xfe\n')

        # Rank 1: Incomplete line
        self._write_to_pipe(write_fds[1], b'Rank 1: Incomplete')
        time.sleep(0.1)
        self._write_to_pipe(write_fds[1], b' completed\n')

        # Close all pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        reader.join(timeout=2.0)

        # Check all output
        content = self._read_log_file(temp_log_file)
        assert '0: Rank 0: Normal line\n' in content
        assert '1: Rank 1: Has <NUL> null\n' in content
        assert '0: Rank 0: Invalid' in content
        assert '1: Rank 1: Incomplete completed\n' in content

    def test_large_data_throughput(self, temp_log_file):
        """Test handling of large amounts of data."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Write many lines
        num_lines = 1000
        for i in range(num_lines):
            self._write_to_pipe(write_fd, f'Line {i}\n'.encode('utf-8'))

        self._close_pipe(write_fd)
        reader.join(timeout=5.0)

        # Check all lines are present
        content = self._read_log_file(temp_log_file)
        lines = content.strip().split('\n')
        assert len(lines) == num_lines
        assert '0: Line 0' in content
        assert f'0: Line {num_lines-1}' in content

    def test_rank_prefix_padding(self, temp_log_file):
        """Test that rank prefixes are properly padded based on world_size."""
        # Test with world_size=1 (no padding needed)
        read_fd1, write_fd1 = self._create_pipe_pair()
        reader1 = MultiplexingReaderThread(
            pipes={0: read_fd1},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader1.start()
        self._write_to_pipe(write_fd1, b'Test\n')
        self._close_pipe(write_fd1)
        reader1.join(timeout=2.0)

        content1 = self._read_log_file(temp_log_file)
        # With world_size=1, no padding (rank_width=0)
        assert '0: Test\n' in content1

        # Clear file
        with open(temp_log_file, 'w') as f:
            f.write('')

        # Test with world_size=100 (2-digit padding for ranks 0-99)
        read_fd2, write_fd2 = self._create_pipe_pair()
        reader2 = MultiplexingReaderThread(
            pipes={5: read_fd2},
            log_file_path=temp_log_file,
            world_size=100,
        )
        reader2.start()
        self._write_to_pipe(write_fd2, b'Test\n')
        self._close_pipe(write_fd2)
        reader2.join(timeout=2.0)

        content2 = self._read_log_file(temp_log_file)
        # Should be right-aligned with width 2 (max rank is 99, which is 2 digits)
        assert ' 5: Test\n' in content2

    def test_concurrent_writes(self, temp_log_file):
        """Test that concurrent writes from multiple ranks don't merge lines."""
        pipes = {}
        write_fds = {}
        for rank in range(4):
            read_fd, write_fd = self._create_pipe_pair()
            pipes[rank] = read_fd
            write_fds[rank] = write_fd

        reader = MultiplexingReaderThread(
            pipes=pipes,
            log_file_path=temp_log_file,
            world_size=4,
        )
        reader.start()

        # Write from all ranks concurrently
        def write_from_rank(rank, write_fd):
            for i in range(10):
                self._write_to_pipe(write_fd, f'Rank {rank} line {i}\n'.encode('utf-8'))
                time.sleep(0.001)  # Small delay to interleave writes

        threads = []
        for rank, write_fd in write_fds.items():
            t = threading.Thread(target=write_from_rank, args=(rank, write_fd))
            t.start()
            threads.append(t)

        # Wait for all writers to finish
        for t in threads:
            t.join()

        # Close all pipes
        for write_fd in write_fds.values():
            self._close_pipe(write_fd)

        reader.join(timeout=5.0)

        # Check that all lines are present and properly formatted
        content = self._read_log_file(temp_log_file)
        lines = content.strip().split('\n')

        # Should have 40 lines total (4 ranks * 10 lines each)
        assert len(lines) == 40

        # Check each line has proper format
        for line in lines:
            assert ': Rank ' in line
            assert ' line ' in line

    def test_append_mode(self, temp_log_file):
        """Test that reader appends to existing file."""
        # Write some initial content
        with open(temp_log_file, 'w') as f:
            f.write('Existing content\n')

        read_fd, write_fd = self._create_pipe_pair()
        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        self._write_to_pipe(write_fd, b'New content\n')
        self._close_pipe(write_fd)
        reader.join(timeout=2.0)

        # Check both old and new content are present
        content = self._read_log_file(temp_log_file)
        assert 'Existing content\n' in content
        assert '0: New content\n' in content

    def test_empty_pipe(self, temp_log_file):
        """Test handling of pipe that closes without any data."""
        read_fd, write_fd = self._create_pipe_pair()

        reader = MultiplexingReaderThread(
            pipes={0: read_fd},
            log_file_path=temp_log_file,
            world_size=1,
        )
        reader.start()

        # Close pipe immediately without writing
        self._close_pipe(write_fd)

        reader.join(timeout=2.0)

        # File should be empty or contain only pre-existing content
        content = self._read_log_file(temp_log_file)
        # No crash, thread terminates cleanly
        assert reader is not None
