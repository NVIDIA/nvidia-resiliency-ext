#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utilities for log_analyzer: LLM output parsing, log path metadata, and record building."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Regex patterns for log file path parsing and splitlog file discovery

# Per-cycle log files (e.g., foo_cycle3.log); raw string for re.search(), compiled for .search()
CYCLE_LOG_PATTERN = r"_cycle(\d+)\.log$"
CYCLE_NUM_PATTERN = re.compile(r"_cycle(\d+)\.log$", re.IGNORECASE)

# Date/time in filename: *_date_YY-MM-DD_time_HH-MM-SS.log (for splitlog file sorting)
DATE_TIME_PATTERN = re.compile(
    r"_date_(\d{2})-(\d{2})-(\d{2})_time_(\d{2})-(\d{2})-(\d{2})\.log$", re.IGNORECASE
)

# Job ID extraction patterns (tried in order of specificity)
JOB_ID_PATTERNS = [
    r"_(\d+)_date_",  # foo_12345_date_2024... (most specific)
    r"[/\\]job_(\d+)[/\\]",  # job_12345/slurm.log (directory pattern)
    r"slurm-(\d+)\.(out|err|log)$",  # slurm-12345.out (SLURM default naming)
    r"[/\\](\d{6,})\.(out|err|log)$",  # /12345678.out (6+ digit job ID in filename)
    r"_(\d{6,})\.(out|err|log)$",  # prefix_12345678.out (6+ digit job ID after underscore)
]


@dataclass
class ParsedLLMResponse:
    """Parsed fields from LLM response.

    The log_analyzer module returns structured text that includes:
    - auto_resume decision (first line)
    - auto_resume explanation (second line)
    - Attribution section with failure attribution
    - checkpoint_saved flag
    """

    auto_resume: str
    auto_resume_explanation: str
    attribution_text: str
    checkpoint_saved_flag: int


def parse_llm_response(raw_text: str) -> ParsedLLMResponse:
    """
    Parse raw LLM response text to extract structured fields.

    The expected format from log_analyzer is:
        <auto_resume_decision>
        <auto_resume_explanation>
        ...
        Attribution: <attribution_text>

        <checkpoint_saved>

    Args:
        raw_text: Raw text from LLM response

    Returns:
        ParsedLLMResponse with extracted fields
    """
    # Extract auto_resume (first line) and explanation (second line)
    lines = raw_text.split("\n")
    auto_resume = lines[0] if lines else ""
    if len(lines) > 1:
        auto_resume_explanation = lines[1]
    else:
        auto_resume_explanation = ""
        logger.warning("Failed to extract auto_resume_explanation: insufficient lines in response")

    # Extract text after 'Attribution:' marker
    attribution_parts = raw_text.split("Attribution:")
    if len(attribution_parts) > 1:
        attribution_section = attribution_parts[1].strip()
        parts = attribution_section.split("\n\n")
        attribution_text = parts[0].replace('"\\', "").replace('\\"', "")
        if len(parts) > 1:
            checkpoint_saved = parts[1]
        else:
            checkpoint_saved = "false"
            logger.debug("No checkpoint_saved field in attribution response")
    else:
        attribution_text = ""
        checkpoint_saved = "false"
        # For ERRORS NOT FOUND, missing Attribution: marker is expected
        if "ERRORS NOT FOUND" in auto_resume:
            logger.debug("No 'Attribution:' marker in LLM response (expected for ERRORS NOT FOUND)")
        else:
            logger.warning("No 'Attribution:' marker found in LLM response")

    # Normalize checkpoint_saved to int flag
    checkpoint_saved_flag = 0
    if isinstance(checkpoint_saved, str) and checkpoint_saved.strip().lower() != "false":
        checkpoint_saved_flag = 1

    return ParsedLLMResponse(
        auto_resume=auto_resume,
        auto_resume_explanation=auto_resume_explanation,
        attribution_text=attribution_text,
        checkpoint_saved_flag=checkpoint_saved_flag,
    )


@dataclass
class JobMetadata:
    """Metadata extracted from log file path."""

    job_id: str
    cycle_id: int


def extract_job_metadata(log_path: str, warn_on_missing_job_id: bool = True) -> JobMetadata:
    """
    Extract job ID and cycle ID from log file path.

    Job ID: tried in order of specificity; see JOB_ID_PATTERNS (inline comments).
    Cycle ID: from ..._cycle<N>.log (see CYCLE_LOG_PATTERN).

    Args:
        log_path: Path to the log file
        warn_on_missing_job_id: If True, log warning when job ID extraction fails.
            Set to False when job_id is provided externally (e.g., from POST request).

    Returns:
        JobMetadata with extracted fields (empty/zero if not found)
    """
    # Try each job ID pattern in order (most specific first)
    job_id = ""
    for pattern in JOB_ID_PATTERNS:
        match = re.search(pattern, log_path)
        if match:
            job_id = match.group(1)
            break

    if not job_id and warn_on_missing_job_id:
        logger.warning(f"Failed to extract job ID from path: {log_path}")

    # Extract cycle ID from path pattern: _cycle<N>.log
    match = re.search(CYCLE_LOG_PATTERN, log_path)
    if match:
        cycle_id = int(match.group(1))
    else:
        cycle_id = 0
        logger.debug(f"No cycle ID in path (not a per-cycle log): {log_path}")

    return JobMetadata(job_id=job_id, cycle_id=cycle_id)


def build_dataflow_record(
    parsed: ParsedLLMResponse,
    metadata: JobMetadata,
    log_path: str,
    processing_time: float,
    cluster_name: str,
    user: str = "nemotron_run",
) -> Dict[str, Any]:
    """
    Build a dataflow record from parsed LLM results.

    This creates the data dictionary suitable for posting to dataflow/elasticsearch.
    The actual posting is left to the caller (to avoid nvdataflow dependency in library).

    Args:
        parsed: Parsed LLM response
        metadata: Job metadata from path
        log_path: Path to the log file
        processing_time: Time taken for analysis in seconds
        cluster_name: Cluster name for dataflow
        user: User identifier (default: "nemotron_run")

    Returns:
        Dictionary with dataflow record fields
    """
    return {
        "s_cluster": cluster_name,
        "s_user": user,
        "s_attribution": parsed.attribution_text,
        "s_auto_resume": parsed.auto_resume,
        "s_auto_resume_explanation": parsed.auto_resume_explanation,
        "s_job_id": metadata.job_id,
        "l_cycle_id": metadata.cycle_id,
        "s_log_path": log_path,
        "l_checkpoint_saved": parsed.checkpoint_saved_flag,
        "d_processing_time": round(processing_time, 2),
        "ts_current_time": round(datetime.now().timestamp() * 1000),
    }
