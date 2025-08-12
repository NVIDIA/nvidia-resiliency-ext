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


import unittest
import os
import shutil
import multiprocessing
import time
import random
import tempfile
import sys
from pathlib import Path

sys.path.append(os.getcwd() + "/src/")

from datetime import datetime
from nvidia_resiliency_ext.shared_utils.log_distributed import LogMessage, NodeLogAggregator
import nvidia_resiliency_ext.shared_utils.log_manager as LogMgr


def create_test_workspace():
    # Create a temporary directory
    tmp_dir = Path(tempfile.mkdtemp())

    # Define log and temp directories
    log_dir = tmp_dir / "logs"
    temp_dir = tmp_dir / "tmp"

    # Remove directories if they already exist
    if log_dir.exists():
        shutil.rmtree(log_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Create the directories
    log_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, temp_dir


def setup_vars(global_id, local_id, file_size, dbg_on="0"):
    os.environ["SLURM_PROCID"] = str(global_id)
    os.environ["SLURM_LOCALID"] = str(local_id)
    os.environ["RANK"] = str(global_id)
    os.environ["LOCAL_RANK"] = str(local_id)
    os.environ["NVRX_LOG_MAX_FILE_SIZE_KB"] = str(file_size)
    os.environ["NVRX_LOG_DEBUG"] = dbg_on


def gen_log_msg(logger, num_msg, log_type="info"):
    for i in range(num_msg):
        skip = random.uniform(1, 50)
        skip -= 1
        if skip == 0:
            time.sleep(0.002 + (random.uniform(0, 100)) / 100000)
        if log_type == "info":
            logger.info(f"My Info Logging Message {i}")
        if log_type == "debug":
            logger.debug(f"My Debug Logging Message {i}")


def worker_process(id, num_msg, file_size):
    """Function that each process will execute."""
    setup_vars(id, id, file_size)
    log_dir, temp_dir = create_test_workspace()
    logger = LogMgr.setup_logger(log_dir, temp_dir, True, False)
    gen_log_msg(logger, num_msg)


class TestLogger(unittest.TestCase):

    def count_files_in_dir(self, dir_path):
        filename = []
        file_count = 0
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file():
                    filename.append(entry.name)
                    file_count += 1
        return file_count, filename

    def check_a_file(self, filename, num_lines, global_id, local_id, chrono_on):
        line_count = 0
        curr_ts = 0
        curr_dt = 0

        with open(filename, 'r') as file:
            for line in file:
                match = LogMessage.log_pattern.match(line)
                if match:
                    line_count += 1
                    log_fields = match.groupdict()
                    for key, value in log_fields.items():
                        if key == 'asctime':
                            # Convert asctime to a datetime object, then to a Unix timestamp
                            dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S,%f')
                            line_ts = dt.timestamp()
                            if chrono_on:
                                self.assertLessEqual(
                                    curr_ts,
                                    line_ts,
                                    f'The timestamp of {curr_dt} is > {value}',
                                )
                            curr_ts = line_ts
                            curr_dt = value

                        if key == 'workload_rank':
                            if global_id != -1:
                                self.assertEqual(
                                    int(value),
                                    global_id,
                                    f'The workload_rank should be {global_id} instead {value}',
                                )
                        if key == 'workload_local_rank':
                            if local_id != -1:
                                self.assertEqual(
                                    int(value),
                                    local_id,
                                    f'The workload_local_rank should be {local_id} instead {value}',
                                )
                    if key == 'infra_rank':
                        if local_id != -1:
                            self.assertEqual(
                                int(value),
                                local_id,
                                f'The infra_rank should be {local_id} instead {value}',
                            )
        if num_lines != -1:
            self.assertEqual(
                line_count, num_lines, f'The line_count should be {num_lines} instead {line_count}'
            )

    def check_files(self, log_dir, filenames, num_lines, global_id, local_id, chrono_on):
        for fname in filenames:
            self.check_a_file(
                os.path.join(log_dir, fname), num_lines, global_id, local_id, chrono_on
            )

    def check_msg(self, num_msg, file_size_kb, pm_files, is_agg: bool, log_type="info", dbg_on="0"):
        log_dir, temp_dir = create_test_workspace()
        setup_vars(0, 0, file_size_kb, dbg_on)

        if is_agg:
            aggregator = NodeLogAggregator(
                log_dir=LogMgr.get_log_dir(log_dir),
                temp_dir=LogMgr.get_temp_dir(temp_dir),
                log_file=LogMgr.get_log_file(),
                max_file_size_kb=LogMgr.get_max_file_size_kb(file_size_kb),
                en_chrono_ord=True,
            )
            aggregator.start_aggregator()
        logger = LogMgr.setup_logger(log_dir, temp_dir, True, not is_agg)
        gen_log_msg(logger, num_msg, log_type)

        time.sleep(1)
        pm = LogMgr.get_temp_dir(temp_dir)
        num_files, file_names = self.count_files_in_dir(pm)
        self.assertEqual(
            num_files, pm_files, f'The number of files should be {pm_files}, instead {num_files}'
        )
        self.check_files(pm, file_names, -1, 0, 0, "1")

        if is_agg:
            aggregator.shutdown()
            num_files, file_names = self.count_files_in_dir(log_dir)
            self.assertEqual(num_files, 1, f'The number of files should be 1, instead {num_files}')
            self.check_files(log_dir, file_names, num_msg, 0, 0, "1")

    def test_single_msg(self):
        self.check_msg(1, 1024, 1, True, "info", "0")

    def test_single_dbg_msg(self):
        self.check_msg(1, 1024, 1, True, "debug", "1")

    def test_many_msg(self):
        self.check_msg(2000, 1024, 1, True)

    def test_rotation(self):
        self.check_msg(300, 10, 4, False)

    def test_rotation_cleanup(self):
        self.check_msg(2000, 10, 1, True)

    def multiple_processes(self, num_procs, num_msg, file_size_kb, chrono_on=True):
        log_dir, temp_dir = create_test_workspace()
        setup_vars(
            global_id=0,
            local_id=0,
            file_size=file_size_kb,
            dbg_on="0",
        )

        aggregator = NodeLogAggregator(
            log_dir=LogMgr.get_log_dir(log_dir),
            temp_dir=LogMgr.get_temp_dir(temp_dir),
            log_file=LogMgr.get_log_file(),
            max_file_size_kb=LogMgr.get_max_file_size_kb(file_size_kb),
            en_chrono_ord=True,
        )
        aggregator.start_aggregator()
        logger = LogMgr.setup_logger(log_dir, temp_dir, True, False)

        processes = []
        for i in range(num_procs):
            # Create a new process
            p = multiprocessing.Process(target=worker_process, args=(i + 1, num_msg, file_size_kb))
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()
        aggregator.shutdown()
        if hasattr(LogMgr.setup_logger, '_log_manager'):
            lm = getattr(LogMgr.setup_logger, '_log_manager')
        num_files, file_names = self.count_files_in_dir(log_dir)
        self.assertEqual(num_files, 1, f'The number of files should be 1, instead {num_files}')
        self.check_files(log_dir, file_names, num_msg * num_procs, -1, -1, chrono_on)

    def test_one_proc(self):
        self.multiple_processes(1, 2000, 1024)

    def test_four_proc(self):
        # gb200 has 4 GPU's, check that config
        self.multiple_processes(4, 2000, 1024)

    def test_eight_proc(self):
        # h100 has 8 GPU's, check that config
        self.multiple_processes(8, 2000, 1024)

    def test_one_proc_w_rotate(self):
        self.multiple_processes(1, 2000, 10)

    def test_four_proc_w_rotate(self):
        self.multiple_processes(1, 2000, 10)

    def test_eight_proc_w_rotate(self):
        # h100 has 8 GPU's, check that config
        self.multiple_processes(8, 2000, 10)

    def test_four_proc_w_rotate_nochrono(self):
        self.multiple_processes(1, 2000, 10, False)

    def test_eight_proc_w_rotate_nochrono(self):
        # h100 has 8 GPU's, check that config
        self.multiple_processes(8, 2000, 1000, False)
