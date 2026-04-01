# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import sys
import tempfile
import unittest

# isort: off  # SkipTest before attribution import on Python < 3.10
if sys.version_info < (3, 10):
    raise unittest.SkipTest("attribution tests require Python 3.10+")

from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import (
    extract_fr_dump_path,
    fr_path_resolvable_for_collective_analyzer,
)

# isort: on


class TestFrDumpPathInference(unittest.TestCase):
    def test_checkpoints_sibling_of_logs(self):
        """Infer <run>/checkpoints when log is under <run>/logs/ and _dump_* files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "exp", "20260327", "2")
            logs_slurm = os.path.join(run, "logs", "slurm")
            ckpt = os.path.join(run, "checkpoints")
            os.makedirs(logs_slurm)
            os.makedirs(ckpt)
            # Create dummy trace files so validation passes
            open(os.path.join(ckpt, "_dump_0"), "w").close()
            open(os.path.join(ckpt, "_dump_1"), "w").close()
            log_file = os.path.join(logs_slurm, "2058736.0.1.main_workload.log")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("TORCH_FR_DUMP_TEMP_FILE=/container/_dump_\n")

            resolved = extract_fr_dump_path(log_file)
            self.assertEqual(os.path.realpath(resolved), os.path.realpath(ckpt))

    def test_checkpoints_sibling_no_traces_returns_none(self):
        """<run>/checkpoints exists but is empty — FR analysis must not be triggered."""
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            logs_slurm = os.path.join(run, "logs", "slurm")
            ckpt = os.path.join(run, "checkpoints")
            os.makedirs(logs_slurm)
            os.makedirs(ckpt)  # directory exists but no _dump_* files
            log_file = os.path.join(logs_slurm, "job.log")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("some log content\n")

            self.assertIsNone(extract_fr_dump_path(log_file))

    def test_torch_fallback_prefix_with_traces(self):
        """TORCH_FR_DUMP_TEMP_FILE prefix is valid when matching _dump_<rank> files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create trace files matching the prefix
            prefix = os.path.join(tmp, "_dump_")
            open(prefix + "0", "w").close()
            open(prefix + "1", "w").close()
            log_file = os.path.join(tmp, "slurm-12345.out")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"x\nTORCH_FR_DUMP_TEMP_FILE={prefix}\n")

            self.assertEqual(extract_fr_dump_path(log_file), prefix)

    def test_torch_fallback_prefix_no_traces_returns_none(self):
        """TORCH_FR_DUMP_TEMP_FILE prefix with no matching files — analysis must not be triggered."""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "_dump_")  # no _dump_* files created
            log_file = os.path.join(tmp, "slurm-12345.out")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"x\nTORCH_FR_DUMP_TEMP_FILE={prefix}\n")

            self.assertIsNone(extract_fr_dump_path(log_file))

    def test_torch_fallback_prefix_is_directory_returns_none(self):
        """TORCH_FR_DUMP_TEMP_FILE set to a bare directory — misconfigured, must not trigger."""
        with tempfile.TemporaryDirectory() as tmp:
            bare_dir = os.path.join(tmp, "checkpoints")
            os.makedirs(bare_dir)
            log_file = os.path.join(tmp, "slurm-12345.out")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"TORCH_FR_DUMP_TEMP_FILE={bare_dir}\n")

            self.assertIsNone(extract_fr_dump_path(log_file))

    def test_no_fr_marker_and_no_logs_layout_returns_none(self):
        """No TORCH_FR_DUMP_TEMP_FILE in log and not under logs/ layout — returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            log_file = os.path.join(tmp, "slurm-99999.out")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("training started\nno fr dump here\n")

            self.assertIsNone(extract_fr_dump_path(log_file))

    def test_checkpoints_outside_allowed_root_returns_none(self):
        """Inferred checkpoints dir must resolve under allowed_root when policy is set."""
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "exp", "run")
            logs_slurm = os.path.join(run, "logs", "slurm")
            ckpt = os.path.join(run, "checkpoints")
            os.makedirs(logs_slurm)
            os.makedirs(ckpt)
            open(os.path.join(ckpt, "_dump_0"), "w").close()
            log_file = os.path.join(logs_slurm, "job.log")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("x\n")

            allowed = os.path.join(tmp, "other_root")
            os.makedirs(allowed, exist_ok=True)
            self.assertIsNone(extract_fr_dump_path(log_file, allowed_root=allowed))

    def test_torch_prefix_outside_allowed_root_returns_none(self):
        """TORCH_FR_DUMP_TEMP_FILE prefix must resolve under allowed_root when policy is set."""
        with tempfile.TemporaryDirectory() as tmp:
            evil = os.path.join(tmp, "evil")
            os.makedirs(evil)
            prefix = os.path.join(evil, "_dump_")
            open(prefix + "0", "w").close()
            log_file = os.path.join(tmp, "job.out")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"TORCH_FR_DUMP_TEMP_FILE={prefix}\n")

            allowed = os.path.join(tmp, "allowed_only")
            os.makedirs(allowed, exist_ok=True)
            self.assertIsNone(extract_fr_dump_path(log_file, allowed_root=allowed))

    def test_fr_path_resolvable_accepts_prefix_without_existing_node(self):
        """MCP and CollectiveAnalyzer accept TORCH_FR_DUMP_TEMP_FILE prefix when _dump_* files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "_dump_")
            open(prefix + "0", "w", encoding="utf-8").close()
            self.assertFalse(os.path.exists(prefix))
            self.assertTrue(fr_path_resolvable_for_collective_analyzer(prefix))

    def test_torch_prefix_glob_uses_wildcard_suffix(self):
        """TORCH_FR_DUMP_TEMP_FILE is a path prefix; rank files match glob(prefix + '*'), not glob(prefix)."""
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "_dump_")
            open(os.path.join(tmp, "_dump_0"), "w", encoding="utf-8").close()
            open(os.path.join(tmp, "_dump_1"), "w", encoding="utf-8").close()
            self.assertFalse(os.path.isdir(prefix))
            self.assertFalse(os.path.isfile(prefix))
            self.assertEqual(len(glob.glob(prefix)), 0)
            self.assertEqual(len(glob.glob(prefix + "*")), 2)


if __name__ == "__main__":
    unittest.main()
