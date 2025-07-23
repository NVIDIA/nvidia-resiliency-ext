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


from src.nvidia_resiliency_ext.shared_utils.logger import setup_logger


class TestLogger(unittest.TestCase):

    def test_single_msg(self):
        log_dir = os.getcwd() + "/tests/shared_utils/"
        os.environ["NVRX_LOG_AGGREGATOR"] = "1"
        os.environ["SLURM_PROCID"] = "0"
        os.environ["SLURM_LOCALID"] = "0"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        
        logger = setup_logger(log_dir, log_dir, True)
        logger.info("My Logging Message 1")
        if hasattr(setup_logger, '_log_manager'):
            lm = getattr(setup_logger, '_log_manager')
            lm.shutdown()
