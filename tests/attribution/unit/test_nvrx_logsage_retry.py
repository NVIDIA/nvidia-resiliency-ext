import importlib
import os
import unittest
from unittest.mock import patch

try:
    nvrx_logsage = importlib.import_module(
        "nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage"
    )
    IMPORT_ERROR = None
except ImportError as exc:
    nvrx_logsage = None
    IMPORT_ERROR = exc


@unittest.skipIf(nvrx_logsage is None, f"missing optional dependency: {IMPORT_ERROR}")
class TestNVRxLogSageRetry(unittest.TestCase):
    def test_with_exponential_backoff_returns_failure_when_retries_zero(self):
        def llm_call():
            raise AssertionError("llm_call should not run when retries=0")

        with patch.dict(os.environ, {"NVRX_LOG_ANALYSIS_LLM_RETRIES": "0"}):
            self.assertEqual(
                nvrx_logsage._with_exponential_backoff(llm_call, checkpoint_saved=True),
                (
                    nvrx_logsage.ATTR_LLM_FAILURE,
                    nvrx_logsage.ATTR_LLM_FAILURE,
                    nvrx_logsage.ATTR_LLM_FAILURE,
                    nvrx_logsage.ATTR_LLM_FAILURE,
                    "True",
                ),
            )


if __name__ == "__main__":
    unittest.main()
