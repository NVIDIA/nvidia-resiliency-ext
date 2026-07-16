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

    def test_endpoint_failure_is_not_retried_by_default(self):
        calls = {"count": 0}

        def llm_call():
            calls["count"] += 1
            return (
                nvrx_logsage.LOGSAGE_LLM_ENDPOINT_FAILED,
                "",
                nvrx_logsage.LOGSAGE_LLM_ENDPOINT_FAILED,
                "",
                "False",
            )

        with patch.dict(
            os.environ,
            {
                "NVRX_LOG_ANALYSIS_LLM_RETRIES": "3",
            },
        ):
            result = nvrx_logsage._with_exponential_backoff(llm_call, checkpoint_saved=False)

        self.assertEqual(calls["count"], 1)
        self.assertEqual(result[0], nvrx_logsage.ATTR_LLM_FAILURE)
        self.assertIn("configured additional retries: 0", result[1])
        self.assertIn(nvrx_logsage.LOGSAGE_LLM_ENDPOINT_FAILED, result[2])

    def test_endpoint_failure_uses_configured_outer_retry(self):
        responses = [
            (
                nvrx_logsage.LOGSAGE_LLM_ENDPOINT_FAILED,
                "",
                nvrx_logsage.LOGSAGE_LLM_ENDPOINT_FAILED,
                "",
                "False",
            ),
            (
                "RESTART IMMEDIATE",
                "not a user failure",
                "Attribution: Primary issues: [NCCL TIMEOUT], Secondary issues: []",
                "",
                "False",
            ),
        ]

        def llm_call():
            return responses.pop(0)

        with patch.dict(
            os.environ,
            {
                "NVRX_LOG_ANALYSIS_LLM_RETRIES": "3",
            },
        ):
            result = nvrx_logsage._with_exponential_backoff(
                llm_call,
                checkpoint_saved=False,
                endpoint_outer_retries=1,
                endpoint_outer_backoff_sec=0,
            )

        self.assertEqual(len(responses), 0)
        self.assertEqual(result[0], "RESTART IMMEDIATE")

    def test_exceptions_still_use_llm_retry_budget(self):
        calls = {"count": 0}

        def llm_call():
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary local failure")
            return (
                "RESTART IMMEDIATE",
                "not a user failure",
                "Attribution: Primary issues: [NCCL TIMEOUT], Secondary issues: []",
                "",
                "False",
            )

        with patch.dict(
            os.environ,
            {
                "NVRX_LOG_ANALYSIS_LLM_RETRIES": "2",
                "NVRX_LOG_ANALYSIS_LLM_INITIAL_BACKOFF_SEC": "0",
                "NVRX_LOG_ANALYSIS_LLM_MAX_BACKOFF_SEC": "0",
                "NVRX_LOG_ANALYSIS_LLM_JITTER_SEC": "0",
            },
        ):
            result = nvrx_logsage._with_exponential_backoff(llm_call, checkpoint_saved=False)

        self.assertEqual(calls["count"], 2)
        self.assertEqual(result[0], "RESTART IMMEDIATE")


if __name__ == "__main__":
    unittest.main()
