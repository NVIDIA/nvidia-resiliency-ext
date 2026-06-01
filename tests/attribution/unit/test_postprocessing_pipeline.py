# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import unittest
from unittest import mock
from unittest.mock import patch

PY310_PLUS = sys.version_info >= (3, 10)

if PY310_PLUS:
    from nvidia_resiliency_ext.attribution.orchestration.log_path_metadata import JobMetadata
    from nvidia_resiliency_ext.attribution.orchestration.posting_markdown import (
        format_posting_markdown_body,
    )
    from nvidia_resiliency_ext.attribution.orchestration.types import RawAnalysisResultItem
    from nvidia_resiliency_ext.attribution.postprocessing import slack
    from nvidia_resiliency_ext.attribution.postprocessing.config import config
    from nvidia_resiliency_ext.attribution.postprocessing.pipeline import (
        ResultPoster,
        build_dataflow_record,
        post_analysis_items,
    )
    from nvidia_resiliency_ext.attribution.postprocessing.slack import should_notify_slack


def _raw_item(
    raw_text,
    action,
    *,
    auto_resume=None,
    auto_resume_explanation="",
    attribution_text="",
    primary_issues=None,
    secondary_issues=None,
):
    return RawAnalysisResultItem(
        raw_text=raw_text,
        auto_resume=auto_resume or raw_text.split("\n", 1)[0],
        auto_resume_explanation=auto_resume_explanation,
        attribution_text=attribution_text,
        checkpoint_saved_flag=0,
        action=action,
        primary_issues=[] if primary_issues is None else primary_issues,
        secondary_issues=[] if secondary_issues is None else secondary_issues,
    )


@unittest.skipUnless(PY310_PLUS, "attribution tests require Python 3.10+")
class TestBuildDataflowRecord(unittest.TestCase):
    def test_raw_analysis_result_item_serializes_canonical_payload(self):
        raw_text = "\n".join(
            [
                "STOP - DONT RESTART IMMEDIATE",
                "checkpoint was not saved",
                "Attribution: Primary issues: [NCCL TIMEOUT], Secondary issues: []",
                "",
                "False",
            ]
        )

        item = _raw_item(
            raw_text,
            "STOP",
            auto_resume="STOP - DONT RESTART IMMEDIATE",
            auto_resume_explanation="checkpoint was not saved",
            attribution_text="Primary issues: [NCCL TIMEOUT], Secondary issues: []",
            primary_issues=["NCCL TIMEOUT"],
        )

        self.assertEqual(item.raw_text, raw_text)
        self.assertEqual(item.action, "STOP")
        self.assertEqual(
            item.to_payload(),
            {
                "raw_text": raw_text,
                "auto_resume": "STOP - DONT RESTART IMMEDIATE",
                "auto_resume_explanation": "checkpoint was not saved",
                "attribution_text": "Primary issues: [NCCL TIMEOUT], Secondary issues: []",
                "checkpoint_saved_flag": 0,
                "primary_issues": ["NCCL TIMEOUT"],
                "secondary_issues": [],
                "action": "STOP",
            },
        )

    def test_raw_analysis_result_item_reads_canonical_json_payload(self):
        item = RawAnalysisResultItem.from_payload(
            _raw_item(
                "ERRORS NOT FOUND\nreason text",
                "CONTINUE",
                auto_resume="ERRORS NOT FOUND",
            ).to_payload()
        )

        self.assertEqual(item.raw_text, "ERRORS NOT FOUND\nreason text")
        self.assertEqual(item.action, "CONTINUE")

    def test_uses_explicit_attribution_timing_fields(self):
        record = build_dataflow_record(
            item=_raw_item(
                "CONTINUE\nsafe to continue",
                "CONTINUE",
                auto_resume="CONTINUE",
                auto_resume_explanation="safe to continue",
                attribution_text="Primary issues: [NCCL TIMEOUT], Secondary issues: [GPU XID]",
                primary_issues=["NCCL TIMEOUT"],
                secondary_issues=["GPU XID"],
            ),
            metadata=JobMetadata(job_id="123", cycle_id=7),
            log_path="/logs/slurm-123.out",
            attribution_analysis_duration_seconds=12.345,
            attribution_analysis_completed_ms=1710000000123,
            cluster_name="test-cluster",
            user="alice",
            recommendation={"action": "RESTART", "source": "log_analyzer"},
        )

        self.assertEqual(record["d_attribution_analysis_duration_seconds"], 12.35)
        self.assertEqual(record["ts_attribution_analysis_completed_ms"], 1710000000123)
        self.assertEqual(record["s_auto_resume"], "CONTINUE")
        self.assertEqual(record["s_auto_resume_explanation"], "safe to continue")
        self.assertEqual(record["s_recommendation_action"], "RESTART")
        self.assertEqual(record["s_recommendation_source"], "log_analyzer")
        self.assertEqual(record["s_primary_issues"], "NCCL TIMEOUT")
        result = json.loads(record["s_attribution_result_json"])
        self.assertEqual(result["raw_auto_resume"], "CONTINUE")
        self.assertEqual(result["primary_issues"], ["NCCL TIMEOUT"])
        self.assertEqual(result["secondary_issues"], ["GPU XID"])
        self.assertFalse(result["checkpoint_saved"])
        self.assertNotIn("attribution_state", result)
        self.assertNotIn("attribution", result)
        self.assertNotIn("d_processing_time", record)
        self.assertNotIn("ts_current_time", record)
        self.assertNotIn("s_attribution", record)
        self.assertNotIn("s_attribution_state", record)
        self.assertNotIn("l_checkpoint_saved", record)

    def test_post_analysis_items_uses_result_state_without_polluting_payload(self):
        old_poster = config.default_poster
        old_cluster = config.cluster_name
        old_slack_token = config.slack_bot_token
        old_slack_channel = config.slack_channel
        captured = []
        try:
            config.default_poster = ResultPoster(
                post_fn=lambda data, index: captured.append((data, index)) or True
            )
            config.cluster_name = "test-cluster"
            config.slack_bot_token = ""
            config.slack_channel = ""

            raw_text = "\n".join(
                [
                    "STOP - DONT RESTART IMMEDIATE",
                    "checkpoint was not saved",
                    "Attribution: Primary issues: [NCCL TIMEOUT], Secondary issues: []",
                    "",
                    "False",
                ]
            )
            with patch(
                "nvidia_resiliency_ext.attribution.postprocessing.pipeline"
                ".maybe_send_slack_notification"
            ) as maybe_send_slack_notification:
                with patch.dict(
                    "os.environ",
                    {
                        "NVRX_ATTRSVC_EXPORT_URL": (
                            "https://dataflow.example.test/dataflow2/test/posting"
                        )
                    },
                ):
                    post_analysis_items(
                        [
                            _raw_item(
                                raw_text,
                                "STOP",
                                auto_resume="STOP - DONT RESTART IMMEDIATE",
                                auto_resume_explanation="checkpoint was not saved",
                                attribution_text=(
                                    "Primary issues: [NCCL TIMEOUT], Secondary issues: []"
                                ),
                                primary_issues=["NCCL TIMEOUT"],
                            )
                        ],
                        attribution_analysis_duration_seconds=1.2,
                        attribution_analysis_completed_ms=1710000000999,
                        path="/logs/slurm-123.out",
                        user="alice",
                        job_id="123",
                        recommendation={"action": "STOP", "source": "log_analyzer"},
                    )
        finally:
            config.default_poster = old_poster
            config.cluster_name = old_cluster
            config.slack_bot_token = old_slack_token
            config.slack_channel = old_slack_channel

        self.assertEqual(len(captured), 1)
        record, index = captured[0]
        self.assertEqual(index, "")
        self.assertEqual(record["s_auto_resume"], "STOP - DONT RESTART IMMEDIATE")
        self.assertEqual(record["s_recommendation_action"], "STOP")
        self.assertEqual(record["s_primary_issues"], "NCCL TIMEOUT")
        result = json.loads(record["s_attribution_result_json"])
        self.assertFalse(result["checkpoint_saved"])
        self.assertEqual(result["raw_auto_resume"], "STOP - DONT RESTART IMMEDIATE")
        self.assertNotIn("attribution", result)
        self.assertNotIn("attribution_state", result)
        self.assertNotIn("s_attribution_state", record)
        self.assertNotIn("s_attribution", record)
        self.assertNotIn("l_checkpoint_saved", record)
        maybe_send_slack_notification.assert_called_once()
        self.assertIs(maybe_send_slack_notification.call_args.args[0], record)

    def test_posting_markdown_reconstructs_attribution_from_structured_issues(self):
        record = build_dataflow_record(
            item=_raw_item(
                "STOP - DONT RESTART IMMEDIATE\ncheckpoint was not saved",
                "STOP",
                auto_resume="STOP - DONT RESTART IMMEDIATE",
                auto_resume_explanation="checkpoint was not saved",
                attribution_text="Primary issues: [NCCL TIMEOUT], Secondary issues: [GPU XID]",
                primary_issues=["NCCL TIMEOUT"],
                secondary_issues=["GPU XID"],
            ),
            metadata=JobMetadata(job_id="123", cycle_id=0),
            log_path="/logs/slurm-123.out",
            attribution_analysis_duration_seconds=1.0,
            attribution_analysis_completed_ms=1710000000123,
            cluster_name="test-cluster",
            user="alice",
            recommendation={"action": "STOP", "source": "log_analyzer"},
        )

        body = format_posting_markdown_body(record)

        self.assertIn("Primary issues: [NCCL TIMEOUT], Secondary issues: [GPU XID]", body)

    def test_slack_notification_gating_uses_recommendation_action(self):
        self.assertTrue(should_notify_slack("STOP"))
        self.assertFalse(should_notify_slack("CONTINUE"))
        self.assertFalse(should_notify_slack("RESTART"))

    def test_slack_send_does_not_require_auto_resume_explanation(self):
        client = mock.Mock()

        with (
            patch.object(slack, "HAS_SLACK", True),
            patch.object(slack, "WebClient", return_value=client),
            patch.object(slack, "get_slack_user_id", return_value=None),
        ):
            sent = slack.send_slack_notification(
                {
                    "s_job_id": "123",
                    "s_user": "alice",
                    "s_recommendation_action": "STOP",
                    "s_auto_resume": "STOP",
                    "s_auto_resume_explanation": "",
                },
                "xoxb-token",
                "#alerts",
            )

        self.assertTrue(sent)
        client.chat_postMessage.assert_called_once()


if __name__ == "__main__":
    unittest.main()
