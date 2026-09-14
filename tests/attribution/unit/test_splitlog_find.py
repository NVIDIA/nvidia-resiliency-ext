# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for splitlog log file discovery under LOGS_DIR/slurm."""

import os
import sys
import tempfile
import unittest

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.legacy_logsage.orchestration.splitlog import (
        SplitlogTracker,
    )


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
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

    def test_symlinked_log_file_outside_allowed_root_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowed = os.path.join(tmp, "allowed")
            logs_dir = os.path.join(allowed, "logs")
            outside = os.path.join(tmp, "outside")
            os.makedirs(logs_dir)
            os.makedirs(outside)

            valid = os.path.join(logs_dir, "app_2058365_cycle1.log")
            outside_target = os.path.join(outside, "app_2058365_cycle0.log")
            escaped_link = os.path.join(logs_dir, "app_2058365_cycle0.log")
            with open(valid, "w", encoding="utf-8") as f:
                f.write("safe")
            with open(outside_target, "w", encoding="utf-8") as f:
                f.write("outside")
            try:
                os.symlink(outside_target, escaped_link)
            except (AttributeError, NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink not available: {exc}")

            tracker = SplitlogTracker(allowed_root=allowed)
            found = tracker._find_log_files(logs_dir, "2058365")

            self.assertEqual(found, [valid])


if __name__ == "__main__":
    unittest.main()
