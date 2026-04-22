# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for slurm_parser LOGS_DIR extraction."""

import sys
import unittest

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.svc.slurm_parser import parse_slurm_output


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestWritingLogsToFallback(unittest.TestCase):
    def test_launch_out_without_start_paths(self):
        content = """\
>>> START SBATCH 1774652591 (2026-03-27 16:03:11.000)
Running on nodes: nvl72150-T[09-10]
Writing logs to /lustre/fsw/portfolios/coreai/users/u/exp//run/2/logs
>>> START all_node_setup 1774652591 (2026-03-27 16:03:11.000)
"""
        info = parse_slurm_output(content)
        self.assertEqual(
            info.logs_dir,
            "/lustre/fsw/portfolios/coreai/users/u/exp//run/2/logs",
        )

    def test_start_paths_wins_over_writing_logs_to(self):
        content = """\
<< START PATHS >>
LOGS_DIR=/data/from_marker
<< END PATHS >>
Writing logs to /ignored/fallback
"""
        info = parse_slurm_output(content)
        self.assertEqual(info.logs_dir, "/data/from_marker")


if __name__ == "__main__":
    unittest.main()
