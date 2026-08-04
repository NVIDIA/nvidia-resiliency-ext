# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import unittest
from unittest.mock import patch

from nvidia_resiliency_ext.shared_utils.job_metadata import (
    job_id_from_env,
    job_user_from_env,
    slurm_array_job_id_from_env,
    slurm_array_task_id_from_env,
    slurm_restart_count_from_env,
)


class TestJobMetadata(unittest.TestCase):

    def test_job_metadata_helpers_read_slurm_env_first(self):
        with patch.dict(
            os.environ,
            {
                "SLURM_JOB_USER": "slurm-user",
                "USER": "fallback-user",
                "SLURM_ARRAY_JOB_ID": "array-job",
                "SLURM_JOB_ID": "plain-job",
            },
        ):
            self.assertEqual(job_user_from_env(), "slurm-user")
            self.assertEqual(job_id_from_env(), "array-job")

    def test_job_metadata_helpers_fallback_to_user_and_slurm_job_id(self):
        with patch.dict(
            os.environ,
            {
                "USER": "fallback-user",
                "SLURM_JOB_ID": "plain-job",
            },
            clear=True,
        ):
            self.assertEqual(job_user_from_env(), "fallback-user")
            self.assertEqual(job_id_from_env(), "plain-job")

    def test_job_metadata_helpers_return_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(job_user_from_env())
            self.assertIsNone(job_id_from_env())

    def test_slurm_array_metadata_accepts_an_explicit_environment(self):
        env = {
            "SLURM_ARRAY_JOB_ID": " 123 ",
            "SLURM_ARRAY_TASK_ID": " 7 ",
            "SLURM_RESTART_COUNT": "2",
        }

        self.assertEqual(slurm_array_job_id_from_env(env), "123")
        self.assertEqual(slurm_array_task_id_from_env(env), "7")
        self.assertEqual(slurm_restart_count_from_env(env), 2)

    def test_slurm_restart_count_is_strict(self):
        for value in ("invalid", "-1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "SLURM_RESTART_COUNT"),
            ):
                slurm_restart_count_from_env({"SLURM_RESTART_COUNT": value})
