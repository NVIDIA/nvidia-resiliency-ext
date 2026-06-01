# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown text for posting pipelines: human-readable attribution + optional FR appendix.

FR in Slack omits dump paths and appears only when hanging/missing ranks are non-empty
(see :func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.fr_markdown_appendix`).
Used by Slack notifications and structured logging so message bodies stay aligned with
:dataflow record keys (see :func:`~nvidia_resiliency_ext.attribution.orchestration.llm_output.log_fields_for_dataflow_record`).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import fr_markdown_appendix


def _format_issue_list(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if str(item))
    return str(value or "")


def _format_issues(primary_issues: Any, secondary_issues: Any) -> Optional[str]:
    primary = _format_issue_list(primary_issues)
    secondary = _format_issue_list(secondary_issues)
    if not primary and not secondary:
        return None
    return f"Primary issues: [{primary}], Secondary issues: [{secondary}]"


def format_attribution_markdown(
    *,
    job_id: Optional[str] = None,
    attribution_text: Optional[str] = None,
    auto_resume_explanation: Optional[str] = None,
    log_path: Optional[str] = None,
) -> str:
    """Build Markdown for core attribution fields (job id, failure text, terminal explanation)."""
    jid = job_id if job_id else "unknown"
    attr = attribution_text if attribution_text else "No attribution available"
    expl = auto_resume_explanation if auto_resume_explanation else "No explanation available"
    body = (
        f"*Job ID:* `{jid}`\n"
        "*Failed due to:*\n"
        f"```{attr}```\n"
        "*Terminal issue:*\n"
        f"```{expl}```"
    )
    if log_path:
        body += f"*Log path:*\n```{log_path}```"
    return body


def format_attribution_markdown_from_record(data: Mapping[str, Any]) -> str:
    """Same as :func:`format_attribution_markdown`, reading keys from a posting dict.

    Reads canonical dataflow keys (``s_attribution_result_json`` and, if present,
    ``s_primary_issues``).
    """
    result_payload = {}
    raw_result_json = data.get("s_attribution_result_json")
    if raw_result_json:
        try:
            result_payload = json.loads(str(raw_result_json))
        except json.JSONDecodeError:
            result_payload = {}
    primary_issues = data.get("s_primary_issues") or result_payload.get("primary_issues")
    secondary_issues = result_payload.get("secondary_issues")
    attribution_text = _format_issues(primary_issues, secondary_issues)
    return format_attribution_markdown(
        job_id=data.get("s_job_id"),
        attribution_text=attribution_text,
        auto_resume_explanation=data.get("s_auto_resume_explanation"),
        log_path=data.get("s_log_path"),
    )


def format_posting_markdown_body(data: Mapping[str, Any]) -> str:
    """Full Markdown body for a posting: attribution sections plus optional FR appendix.

    Same text shape as Slack messages before user mention; also used when logging posted results.
    """
    body = format_attribution_markdown_from_record(data)
    fr = fr_markdown_appendix(
        dump_path=data.get("s_fr_dump_path"),
        analysis_text=data.get("s_fr_analysis"),
        hanging_ranks=data.get("s_hanging_ranks"),
    )
    return f"{body}{fr}"
