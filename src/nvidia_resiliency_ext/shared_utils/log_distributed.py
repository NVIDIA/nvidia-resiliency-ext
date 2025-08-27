# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import heapq
import logging
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List


class DistributedLogHandler(logging.Handler):
    """Custom log handler that logs messages to the file system."""

    def __init__(
        self,
        rank_id: int,
        file_path: str,
        max_file_size: int,
        max_backup_files: int,
        proc_name: str,
    ):
        super().__init__()
        self.fname = None
        self.flock = threading.Lock()
        self.rank_id = rank_id
        self.file_path = file_path
        self.max_file_size = max_file_size
        self.max_backup_files = max_backup_files
        self.proc_name = proc_name

    def emit(self, record: logging.LogRecord):
        """Emit a log record."""
        try:
            # Format the message using the formatter (which handles rank info dynamically)
            msg = self.format(record)
            self._write_message(message=msg)
        except (OSError, IOError, RuntimeError):
            # Fallback to stderr if logging fails
            sys.stderr.write(f"Log handler error: {record.getMessage()}\n")
            sys.stderr.flush()

    def _log_file_namer(self):
        return f"rank_{self.rank_id}_{self.proc_name}.msg.{int(time.time()*1000)}"

    def _cleanup_old_backup_files(self):
        """Clean up old log files, keeping only the most recent one's."""
        backup_files = []
        for filename in os.listdir(self.file_path):
            match = re.match(rf"rank_{self.rank_id}_{self.proc_name}.msg\.(\d+)", filename)
            if not match:
                continue
            backup_files.append(filename)
        backup_files.sort()
        for old_file in backup_files[: -self.max_backup_files]:
            try:
                os.remove(os.path.join(self.file_path, old_file))
            except (OSError, IOError) as e:
                # Log the error but don't fail the entire operation
                sys.stderr.write(f"Failed to remove backup file {old_file}: {e}\n")
                sys.stderr.flush()

    def _write_message(self, message: str):
        with self.flock:
            if self.fname is None:
                os.makedirs(self.file_path, exist_ok=True)
                self.fname = os.path.join(self.file_path, self._log_file_namer())
            # Check if file needs rotation
            if os.path.exists(self.fname):
                try:
                    file_size = os.path.getsize(self.fname)
                    if file_size > self.max_file_size:
                        self.fname = os.path.join(self.file_path, self._log_file_namer())
                        self._cleanup_old_backup_files()
                except (OSError, IOError) as e:
                    sys.stderr.write(f"File rotation error for {self.fname}: {e}\n")
                    sys.stderr.flush()

            # Append message to the rank's message file
            with open(self.fname, 'a') as f:
                f.write(f"{message}\n")
                f.flush()  # Ensure message is written immediately


class DynamicLogFormatter(logging.Formatter):
    """Dynamic formatter that reads rank information from LogManager."""

    def __init__(
        self,
        workload_rank=None,
        workload_local_rank=None,
        infra_rank=None,
        infra_local_rank=None,
        fmt=None,
        datefmt=None,
    ):
        super().__init__(fmt, datefmt)
        self.workload_rank = workload_rank
        self.workload_local_rank = workload_local_rank
        self.infra_rank = infra_rank
        self.infra_local_rank = infra_local_rank

    def format(self, record):
        # Fallback to "?" for None values
        record.workload_rank = self.workload_rank if self.workload_rank is not None else "?"
        record.workload_local_rank = (
            self.workload_local_rank if self.workload_local_rank is not None else "?"
        )
        record.infra_rank = self.infra_rank if self.infra_rank is not None else "?"
        record.infra_local_rank = (
            self.infra_local_rank if self.infra_local_rank is not None else "?"
        )

        # Use the parent's format method
        return super().format(record)


class LogMessage:
    """Represents a log message."""

    log_pattern = re.compile(
        r"(?P<asctime>[\d-]+\s[\d:,]+) \[(?P<levelname>\w+)\] \[(?P<hostname>[\w.-]+)\] "
        r"\[workload:(?P<workload_rank>\d+)\((?P<workload_local_rank>\d+)\) infra:(?P<infra_rank>\d+)\((?P<infra_local_rank>\d+)\)\] "
        r"(?P<filename>[\w.]+):(?P<lineno>\d+) (?P<message>.+)"
    )

    def __init__(self, log_message: str):
        self.log_message = log_message
        self.hash_table = {}
        match = LogMessage.log_pattern.match(log_message)
        if match:
            log_fields = match.groupdict()
            for key, value in log_fields.items():
                if key == 'asctime':
                    # Convert asctime to a datetime object, then to a Unix timestamp
                    dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S,%f')
                    timestamp = int(dt.timestamp())
                    self.hash_table[key] = value
                else:
                    self.hash_table[key] = value

        if 'asctime' not in self.hash_table:
            current_datetime = datetime.now()
            self.hash_table['asctime'] = int(current_datetime.timestamp())

    def getts(self):
        return self.hash_table['asctime']

    def __str__(self):
        return self.log_message


class NodeLogAggregator:

    def __init__(
        self, log_dir: str, temp_dir: str, log_file: str, max_file_size: int, en_chrono_ord: bool
    ):
        self._log_dict_queue = {}
        self._aggregator_thread = None
        self._stop_event = threading.Event()
        self._max_msg_file_size = max_file_size

        # Use node_id to ensure all ranks on the same node use the same directory
        self._temp_dir = temp_dir
        os.makedirs(self._temp_dir, exist_ok=True)

        # Create log directory if it doesn't exist
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_file = log_file
        self.en_chrono_ord = en_chrono_ord

        # Track file positions for each rank to avoid re-reading
        self._file_positions = {}

    def shutdown(self):
        self._stop_event.set()
        if self._aggregator_thread:
            self._aggregator_thread.join()
            self._aggregator_thread = None

    def start_aggregator(self):
        """Start the log aggregator thread."""
        if self._aggregator_thread is not None:
            return
        self._aggregator_thread = threading.Thread(
            target=self._aggregator_loop, daemon=True, name="LogAggregator"
        )
        self._aggregator_thread.start()

    def _write_messages_to_file(self, messages: List[LogMessage], output):
        # Write messages to output
        for msg in messages:
            try:
                # The message is already formatted by the formatter, just write it
                output.write(msg.log_message + '\n')
                output.flush()
            except Exception as e:
                # Fallback to stderr if output fails
                sys.stderr.write(f"Log output error: {e}\n")
                sys.stderr.flush()

    def _merge_sort_streaming_lists(
        self, msg_dict: Dict[str, queue.SimpleQueue], heap: List
    ) -> list:
        if not self.en_chrono_ord:
            unsorted_msgs = []
            for key, msg_q in msg_dict.items():
                if msg_q:
                    while not msg_q.empty():
                        lm = msg_q.get()
                        unsorted_msgs.append(lm)
            msg_dict.clear()
            return unsorted_msgs

        # Initialize heap with the first log of each list
        heap_keys = {}
        i = 0
        for _, key, lm in heap:
            heap_keys[key] = i
            i += 1

        for key, msg_q in msg_dict.items():
            if msg_q and msg_q.qsize() > 0:
                if key not in heap_keys:
                    lm = msg_q.get()
                    # push <ts, key, log>
                    heapq.heappush(heap, (lm.getts(), key, lm))

        sorted_msgs = []
        while heap:
            ts, key, log_entry = heapq.heappop(heap)
            sorted_msgs.append(log_entry)
            msg_q = msg_dict[key]
            if msg_q.qsize() > 0:
                next_log = msg_q.get()
                heapq.heappush(heap, (next_log.getts(), key, next_log))
            else:
                break
        return sorted_msgs

    def _process_messages(self, output):
        # Check for pending messages from other ranks
        keep_processing = 50
        msg_dict = {}
        heap = []

        while keep_processing:
            if self._stop_event.is_set():
                # Gives room for aggregator to catch up with writes
                keep_processing -= 1
            # Check for pending messages from other ranks
            self._check_pending_messages()

            # Process queued messages
            for key, lm_q in self._log_dict_queue.items():
                if key in msg_dict:
                    curr_q = msg_dict[key]
                    while not lm_q.empty():
                        curr_q.put(lm_q.get())
                else:
                    msg_dict[key] = lm_q
            self._log_dict_queue.clear()

            sorted_msgs = self._merge_sort_streaming_lists(msg_dict, heap)
            if len(sorted_msgs) > 0:
                self._write_messages_to_file(sorted_msgs, output)
            # Sleep briefly to avoid busy waiting
            time.sleep(0.025)

    def _aggregator_loop(self):
        """Main loop for the log aggregator."""
        # Setup per-node log file
        log_file = os.path.join(self._log_dir, self._log_file)
        output = open(log_file, 'a', buffering=1)  # Line buffered
        try:
            self._process_messages(output)
        finally:
            output.close()

    def _check_pending_messages(self):
        if not os.path.exists(self._temp_dir):
            return

        # Check if we can access the directory
        if not os.access(self._temp_dir, os.R_OK):
            return

        # Look for message files from all ranks (including this aggregator rank)
        for filename in os.listdir(self._temp_dir):
            if not filename.startswith('rank_'):
                continue
            msg_file = os.path.join(self._temp_dir, filename)
            # Process current file
            self._process_message_file(msg_file)

    def _process_message_file(self, msg_file: str):
        """Process a single message file (current or backup)."""
        try:
            file_size = os.path.getsize(msg_file)
        except FileNotFoundError as e:
            # File was deleted/renamed between discovery and processing
            # This can happen due to race conditions, but should be logged for debugging
            sys.stderr.write(f"File not found during processing {msg_file}: {e}\n")
            sys.stderr.flush()
            return
        except (IOError, OSError) as e:
            # Unexpected: Permission issues, disk problems, etc.
            # Log this as it might indicate a real problem
            sys.stderr.write(f"Unexpected error accessing {msg_file}: {e}\n")
            sys.stderr.flush()
            return

        # Get the last known position for this file
        last_position = self._file_positions.get(msg_file, 0)

        # If file hasn't grown, check if can be deleted
        if file_size <= last_position and file_size >= self._max_msg_file_size:
            self._cleanup_old_backup_files(os.path.basename(msg_file))
            return

        # Read new content from the file
        try:
            with open(msg_file, 'r') as f:
                f.seek(last_position)
                lines = f.readlines()
                file_size = f.tell()

        except FileNotFoundError as e:
            # File was deleted between size check and read
            sys.stderr.write(f"File not found during read {msg_file}: {e}\n")
            sys.stderr.flush()
            return
        except (IOError, OSError) as e:
            # File is being written by another process or other I/O error
            # Log this as it might indicate a real problem
            sys.stderr.write(f"IO error reading {msg_file}: {e}\n")
            sys.stderr.flush()
            return

        # Process each line
        log_msg_q = queue.SimpleQueue()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            log_msg = LogMessage(line)
            log_msg_q.put(log_msg)

        self._log_dict_queue[msg_file] = log_msg_q

        # Update the position for this file
        self._file_positions[msg_file] = file_size

    def _cleanup_old_backup_files(self, msg_file: str):
        """Clean up old backup files, keeping only the most recent one."""
        # Find all backup files for this rank
        parts_first = msg_file.split('.', 1)
        parts_last = msg_file.rsplit('.', 1)
        if len(parts_first) < 2 or len(parts_last) < 2:
            sys.stderr.write(f"Skipping '{msg_file}': missing '.' parts")
            return
        to_del_ts = parts_last[-1]
        if not to_del_ts.isdigit():
            sys.stderr.write(f"Skipping '{msg_file}': last part is not numeric")
            return
        to_del_prefix = parts_first[0]
        if not os.path.exists(self._temp_dir):
            return
        for filename in os.listdir(self._temp_dir):
            match = re.match(rf"{to_del_prefix}.msg\.(\d+)", filename)
            if not match:
                continue
            cur_file_ts = match.group(1)
            if int(cur_file_ts) > int(to_del_ts):
                try:
                    os.remove(os.path.join(self._temp_dir, msg_file))
                    break
                except (OSError, IOError) as e:
                    # Log the error but don't fail the entire operation
                    sys.stderr.write(f"Failed to remove backup file {msg_file}: {e}\n")
                    sys.stderr.flush()
