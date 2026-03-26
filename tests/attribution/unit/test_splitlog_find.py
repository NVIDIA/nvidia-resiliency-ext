# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for splitlog log file discovery under LOGS_DIR/slurm."""

import os
import sys
import tempfile
import unittest

if sys.version_info < (3, 10):
    raise unittest.SkipTest("attribution tests require Python 3.10+")

from nvidia_resiliency_ext.attribution.log_analyzer.splitlog import SplitlogTracker


class TestFindLogFiles(unittest.TestCase):
    def test_slurm_subdir_jobid_dot_pattern(self):
        """Paths like 2058365.0.1.main_workload.log under logs/slurm/."""
        with tempfile.TemporaryDirectory() as tmp:
            slurm = os.path.join(tmp, "slurm")
            os.makedirs(slurm)
            path = os.path.join(slurm, "2058365.0.1.main_workload.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write("x")

            tracker = SplitlogTracker()
            found = tracker._find_log_files(tmp, "2058365")
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0].endswith("main_workload.log"))

    def test_legacy_pattern_still_in_slurm_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            slurm = os.path.join(tmp, "slurm")
            os.makedirs(slurm)
            path = os.path.join(slurm, "app_2058365_date_01-01-01_time_00-00-00.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write("x")

            tracker = SplitlogTracker()
            found = tracker._find_log_files(tmp, "2058365")
            self.assertEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()
