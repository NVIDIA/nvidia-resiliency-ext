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
import logging
import os
import shutil
import multiprocessing
import time
import random

from datetime import datetime
from src.nvidia_resiliency_ext.shared_utils.logger import setup_logger, log_pattern


def setup_vars(global_id, local_id, is_agg):
    os.environ["NVRX_LOG_AGGREGATOR"] = is_agg
    os.environ["SLURM_PROCID"] = str(global_id)
    os.environ["SLURM_LOCALID"] = str(local_id)
    os.environ["RANK"] = str(global_id)
    os.environ["LOCAL_RANK"] = str(local_id)


def gen_log_msg(logger, num_msg):
    for i in range(num_msg):
        skip = random.uniform(1, 50)
        skip -= 1
        if skip == 0:
            time.sleep(0.002 + (random.uniform(0, 100)) / 100000)
        logger.info(f"My Logging Message {i}")


def worker_process(n, num_msg):
    """Function that each process will execute."""
    setup_vars(n, n, "0")
    log_dir = os.getcwd() + "/tests/shared_utils/logs/"
    logger = setup_logger(log_dir, log_dir, False)
    gen_log_msg(logger, num_msg)


class TestLogger(unittest.TestCase):

    def count_files_in_dir(self, dir_path):
        filename = ""
        file_count = 0
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file():
                    filename = entry.name
                    file_count += 1
        return file_count, filename

    def check_file(self, file_path, num_lines, global_id, local_id):
        line_count = 0
        curr_ts = 0
        curr_dt = 0
        with open(file_path, 'r') as file:
            for line in file:
                match = log_pattern.match(line)
                if match:
                    line_count += 1
                    log_fields = match.groupdict()
                    for key, value in log_fields.items():
                        if key == 'asctime':
                            # Convert asctime to a datetime object, then to a Unix timestamp
                            dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S,%f')
                            line_ts = dt.timestamp()
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
        self.assertEqual(
            line_count, num_lines, f'The line_count should be {num_lines} instead {line_count}'
        )

    def test_single_msg(self):
        log_dir = os.getcwd() + "/tests/shared_utils/logs/"
        setup_vars(0, 0, "1")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        logger = setup_logger(log_dir, log_dir, True)
        logger.info("My Logging Message 1")
        if hasattr(setup_logger, '_log_manager'):
            lm = getattr(setup_logger, '_log_manager')
            lm.shutdown()
        num_files, file_name = self.count_files_in_dir(log_dir)
        self.assertEqual(num_files, 1, f'The number of files should be 1, instead {num_files}')
        self.check_file(log_dir + file_name, 1, 0, 0)

    def test_many_msg(self):
        log_dir = os.getcwd() + "/tests/shared_utils/logs/"
        setup_vars(0, 0, "1")

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        logger = setup_logger(log_dir, log_dir, True)
        num_msg = 2000
        for i in range(num_msg):
            time.sleep(0.002 + (random.uniform(0, 100)) / 100000)
            logger.info(f"My Logging Message {i}")
        if hasattr(setup_logger, '_log_manager'):
            lm = getattr(setup_logger, '_log_manager')
            lm.shutdown()
        num_files, file_name = self.count_files_in_dir(log_dir)
        self.assertEqual(num_files, 1, f'The number of files should be 1, instead {num_files}')
        self.check_file(log_dir + file_name, num_msg, 0, 0)

    def multiple_processes(self, num_procs, num_msg):
        log_dir = os.getcwd() + "/tests/shared_utils/logs/"
        setup_vars(0, 0, "1")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        logger = setup_logger(log_dir, log_dir, True)

        processes = []
        for i in range(num_procs):
            # Create a new process
            p = multiprocessing.Process(target=worker_process, args=(i + 1, num_msg))
            processes.append(p)
            p.start()

        # process 0 logs
        gen_log_msg(logger, num_msg)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        if hasattr(setup_logger, '_log_manager'):
            lm = getattr(setup_logger, '_log_manager')
            lm.shutdown()
        num_files, file_name = self.count_files_in_dir(log_dir)
        self.assertEqual(num_files, 1, f'The number of files should be 1, instead {num_files}')
        self.check_file(log_dir + file_name, num_msg * (num_procs + 1), -1, -1)

    def test_one_proc(self):
        self.multiple_processes(1, 2000)
