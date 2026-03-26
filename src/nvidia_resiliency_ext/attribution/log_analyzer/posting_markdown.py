# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown text for posting pipelines: human-readable attribution + optional FR appendix.

FR in Slack omits dump paths and appears only when hanging/missing ranks are non-empty
(see :func:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_support.fr_markdown_appendix`).
Used by Slack notifications and structured logging so message bodies stay aligned with
:dataflow record keys (see :func:`~nvidia_resiliency_ext.attribution.log_analyzer.llm_output.log_fields_for_dataflow_record`).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from nvidia_resiliency_ext.attribution.trace_analyzer.fr_support import fr_markdown_appendix


def format_attribution_markdown(
    *,
    job_id: Optional[str] = None,
    attribution_text: Optional[str] = None,
    auto_resume_explanation: Optional[str] = None,
) -> str:
    """Build Markdown for core attribution fields (job id, failure text, terminal explanation).

    Aligns with dataflow keys ``s_job_id``, ``s_attribution``, ``s_auto_resume_explanation`` from
    :func:`~nvidia_resiliency_ext.attribution.log_analyzer.llm_output.log_fields_for_dataflow_record`.
    """
    jid = job_id if job_id else "unknown"
    attr = attribution_text if attribution_text else "No attribution available"
    expl = auto_resume_explanation if auto_resume_explanation else "No explanation available"
    return (
        f"*Job ID:* `{jid}`\n"
        "*Failed due to:*\n"
        f"```{attr}```\n"
        "*Terminal issue:*\n"
        f"```{expl}```"
    )


def format_attribution_markdown_from_record(data: Mapping[str, Any]) -> str:
    """Same as :func:`format_attribution_markdown`, reading keys from a dataflow posting dict."""
    return format_attribution_markdown(
        job_id=data.get("s_job_id"),
        attribution_text=data.get("s_attribution"),
        auto_resume_explanation=data.get("s_auto_resume_explanation"),
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
