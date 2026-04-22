# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import fr_markdown_appendix


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestFrMarkdownAppendix(unittest.TestCase):
    def test_no_section_when_only_dump_path(self):
        self.assertEqual(
            fr_markdown_appendix(
                dump_path="/lustre/.../checkpoints",
                analysis_text="table...",
                hanging_ranks=None,
            ),
            "",
        )

    def test_no_section_when_hanging_ranks_empty_list(self):
        self.assertEqual(
            fr_markdown_appendix(
                dump_path="/x",
                hanging_ranks="hanging ranks: []",
            ),
            "",
        )

    def test_section_when_hanging_ranks_nonempty(self):
        md = fr_markdown_appendix(
            dump_path="/x",
            hanging_ranks="hanging ranks: [1, 2]",
            analysis_text="PG analysis",
        )
        self.assertIn("*Flight recorder*", md)
        self.assertIn("hanging ranks: [1, 2]", md)
        self.assertIn("FR collective analysis", md)
        self.assertIn("PG analysis", md)
        self.assertNotIn("/x", md)
        self.assertNotIn("Flight recorder dump", md)


if __name__ == "__main__":
    unittest.main()
