# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from unittest.mock import patch

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.mcp_integration.mcp_client import get_server_command


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestGetServerCommand(unittest.TestCase):
    def test_prefers_nvrx_mcp_analysis_on_path(self):
        with patch(
            "nvidia_resiliency_ext.attribution.mcp_integration.mcp_client.shutil.which"
        ) as w:
            w.return_value = "/opt/conda/bin/nvrx-mcp-analysis"
            cmd = get_server_command()
        self.assertEqual(cmd, ["/opt/conda/bin/nvrx-mcp-analysis", "--log-level", "INFO"])

    def test_log_level_override(self):
        with patch(
            "nvidia_resiliency_ext.attribution.mcp_integration.mcp_client.shutil.which"
        ) as w:
            w.return_value = "/opt/conda/bin/nvrx-mcp-analysis"
            cmd = get_server_command(log_level="debug")
        self.assertEqual(cmd, ["/opt/conda/bin/nvrx-mcp-analysis", "--log-level", "DEBUG"])


if __name__ == "__main__":
    unittest.main()
