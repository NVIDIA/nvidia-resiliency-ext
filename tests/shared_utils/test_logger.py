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


from src.nvidia_resiliency_ext.shared_utils.logger import setup_logger, log_pattern


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
        with open(file_path, 'r') as file:
            for line in file:
                match = log_pattern.match(line)
                if match:
                    line_count += 1
                    log_fields = match.groupdict()
                    for key, value in log_fields.items():
                        if key == 'workload_rank':
                            self.assertEqual(
                                int(value),
                                global_id,
                                f'The workload_rank should be {global_id} instead {value}',
                            )
                        if key == 'workload_local_rank':
                            self.assertEqual(
                                int(value),
                                local_id,
                                f'The workload_local_rank should be {local_id} instead {value}',
                            )
                    if key == 'infra_rank':
                        self.assertEqual(
                            int(value),
                            local_id,
                            f'The infra_rank should be {local_id} instead {value}',
                        )
        self.assertEqual(
            line_count, num_lines, f'The line_count should be {num_lines} instead {line_count}'
        )

    def setup_vars(self, global_id, local_id):
        os.environ["NVRX_LOG_AGGREGATOR"] = "1"
        os.environ["SLURM_PROCID"] = str(global_id)
        os.environ["SLURM_LOCALID"] = str(local_id)
        os.environ["RANK"] = str(global_id)
        os.environ["LOCAL_RANK"] = str(local_id)
  
  
    def test_single_msg(self):
        log_dir = os.getcwd() + "/tests/shared_utils/logs/"
        self.setup_vars(0, 0)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        logger = setup_logger(log_dir, log_dir, True)
        logger.info("My Logging Message 1")
        if hasattr(setup_logger, '_log_manager'):
            lm = getattr(setup_logger, '_log_manager')
            lm.shutdown()
        num_files, file_name = self.count_files_in_dir(log_dir)
        self.assertEqual(num_files, 1, f'The number of files should be 1, instead {num_files}')
        self.check_file(log_dir+file_name, 1, 0, 0)


    def test_many_msg(self):
        log_dir = os.getcwd() + "/tests/shared_utils/logs/"
        self.setup_vars(0, 0)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        logger = setup_logger(log_dir, log_dir, True)
        num_msg = 10000
        for i in range(num_msg):
            logger.info(f"My Logging Message {i}")
        if hasattr(setup_logger, '_log_manager'):
            lm = getattr(setup_logger, '_log_manager')
            lm.shutdown()
        num_files, file_name = self.count_files_in_dir(log_dir)
        self.assertEqual(num_files, 1, f'The number of files should be 1, instead {num_files}')
        self.check_file(log_dir + file_name, num_msg, 0, 0)
