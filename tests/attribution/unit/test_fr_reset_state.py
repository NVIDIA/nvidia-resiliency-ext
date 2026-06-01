# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import unittest

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestCollectiveAnalyzerRunState(unittest.IsolatedAsyncioTestCase):
    def _args(self, fr_path: str) -> dict:
        return {
            "fr_path": fr_path,
            "pattern": "_dump_*",
            "verbose": False,
            "health_check": False,
            "llm_analyze": False,
            "model": None,
            "base_url": None,
            "threshold": None,
        }

    async def test_reused_analyzer_clears_previous_fr_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = os.path.join(tmp, "first")
            second_dir = os.path.join(tmp, "second")
            os.makedirs(first_dir)
            os.makedirs(second_dir)
            open(os.path.join(first_dir, "_dump_1"), "w", encoding="utf-8").close()
            open(os.path.join(second_dir, "_dump_2"), "w", encoding="utf-8").close()

            analyzer = CollectiveAnalyzer(self._args(first_dir))

            def fake_process_file(filepath: str) -> bool:
                rank_id = os.path.basename(filepath).rsplit("_", 1)[-1]
                analyzer.collectives_by_file[rank_id] = []
                analyzer.pg_status[rank_id] = {"0": {"last_enqueued_collective": 0}}
                analyzer.pg_configs[rank_id] = {"ranks": {int(rank_id)}}
                analyzer.node_health_status[int(rank_id)] = {}
                return True

            analyzer.process_file = fake_process_file
            analyzer.group_collectives_by_windows = lambda: {}
            analyzer.analyze_matches = lambda verbose=False: ({}, {})

            await analyzer.run(self._args(first_dir))
            self.assertEqual(set(analyzer.collectives_by_file), {"1"})
            self.assertEqual(set(analyzer.pg_status), {"1"})
            self.assertEqual(set(analyzer.node_health_status), {1})

            await analyzer.run(self._args(second_dir))
            self.assertEqual(set(analyzer.collectives_by_file), {"2"})
            self.assertEqual(set(analyzer.pg_status), {"2"})
            self.assertEqual(set(analyzer.node_health_status), {2})


if __name__ == "__main__":
    unittest.main()
