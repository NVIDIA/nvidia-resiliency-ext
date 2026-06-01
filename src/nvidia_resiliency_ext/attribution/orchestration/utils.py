# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path validation and LogSage/MCP result shaping for the log analyzer stack.

For LLM response parsing, path-based job metadata, regex patterns for splitlog/cycles, and
dataflow field extraction from parsed LLM output, see :mod:`nvidia_resiliency_ext.attribution.orchestration.llm_output`.

Sections here:
- Path validation under an allowed root (:func:`validate_log_path`; containment helper in :mod:`~nvidia_resiliency_ext.attribution.path_utils`)
- In-process LogSage result shaping to MCP-shaped dict (:func:`nvrx_run_result_to_log_dict`)
"""

from __future__ import annotations

import logging
import os
import stat
from typing import Any, Dict, Union

from .config import MODULE_LOG_ANALYZER, RESP_MODULE, RESP_RESULT, ErrorCode
from .llm_output import logsage_recommendation, recommendation_payload
from .types import LogAnalyzerError, LogSageAnalysisResult, RawAnalysisResultItem

logger = logging.getLogger(__name__)


# --- Path validation ----------------------------------------------------------


def validate_log_path(
    user_path: str,
    allowed_root: str,
    *,
    require_regular_file: bool = True,
    reject_empty: bool = False,
) -> Union[str, LogAnalyzerError]:
    """Resolve ``user_path`` under ``allowed_root``; return real path or :class:`LogAnalyzerError`."""
    if not os.path.isabs(user_path):
        return LogAnalyzerError(
            error_code=ErrorCode.INVALID_PATH,
            message="path must be absolute",
        )

    try:
        real = os.path.realpath(user_path)
    except ValueError:
        return LogAnalyzerError(
            error_code=ErrorCode.INVALID_PATH,
            message="invalid path characters",
        )

    allowed = os.path.realpath(allowed_root)

    try:
        common = os.path.commonpath([real, allowed])
    except ValueError:
        return LogAnalyzerError(
            error_code=ErrorCode.OUTSIDE_ROOT,
            message="access outside allowed root is not permitted",
        )

    if common != allowed:
        return LogAnalyzerError(
            error_code=ErrorCode.OUTSIDE_ROOT,
            message="access outside allowed root is not permitted",
        )

    try:
        st = os.stat(real)
    except FileNotFoundError:
        return LogAnalyzerError(
            error_code=ErrorCode.NOT_FOUND,
            message="path not found",
        )
    except PermissionError:
        return LogAnalyzerError(
            error_code=ErrorCode.NOT_READABLE,
            message="permission denied",
        )
    except OSError as e:
        return LogAnalyzerError(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=f"filesystem error: {e}",
        )

    if require_regular_file and not stat.S_ISREG(st.st_mode):
        return LogAnalyzerError(
            error_code=ErrorCode.NOT_REGULAR,
            message="path must be a regular file",
        )

    if not os.access(real, os.R_OK):
        return LogAnalyzerError(
            error_code=ErrorCode.NOT_READABLE,
            message="path is not readable",
        )

    if reject_empty and require_regular_file and st.st_size == 0:
        return LogAnalyzerError(
            error_code=ErrorCode.EMPTY_FILE,
            message="file is empty",
        )

    return real


# --- LogSage result shaping ---------------------------------------------------


LOG_ANALYZER_MODULE = MODULE_LOG_ANALYZER


def log_analyzer_result_payload(
    actual_result: list[Any] | LogSageAnalysisResult,
    *,
    module: str = LOG_ANALYZER_MODULE,
) -> Dict[str, Any]:
    """Build the serialized LogSage payload plus the source-derived recommendation."""
    if isinstance(actual_result, LogSageAnalysisResult):
        analysis_result = actual_result
    else:
        items = [RawAnalysisResultItem.from_payload(item) for item in actual_result]
        analysis_result = LogSageAnalysisResult(
            items,
            logsage_recommendation(items, source=module),
        )
    payload: Dict[str, Any] = {
        RESP_MODULE: module,
        RESP_RESULT: [item.to_payload() for item in analysis_result.items],
    }
    payload["recommendation"] = recommendation_payload(analysis_result.recommendation)
    return payload


def selected_log_analyzer_cycle_payload(
    log_result: Dict[str, Any],
    result_item: Any,
) -> Dict[str, Any]:
    """Return a canonical one-cycle LogSage payload preserving response metadata."""
    module = str(log_result.get(RESP_MODULE) or LOG_ANALYZER_MODULE)
    recommendation = log_result.get("recommendation")
    source = module
    if isinstance(recommendation, dict):
        rec_source = recommendation.get("source")
        if isinstance(rec_source, str) and rec_source.strip():
            source = rec_source.strip()
    selected = {
        key: value
        for key, value in log_result.items()
        if key not in {RESP_RESULT, "recommendation"}
    }
    cycle_payload = log_analyzer_result_payload([result_item], module=module)
    cycle_payload["recommendation"]["source"] = source
    selected.update(cycle_payload)
    return selected


def nvrx_run_result_to_log_dict(result: Any, path: str) -> Dict[str, Any]:
    """Normalize :meth:`NVRxLogAnalyzer.run` output to the service/MCP ``log_result`` shape."""
    if result is None:
        logger.error("Lib log analyzer run returned None for path=%s", path)
        raise RuntimeError("LogSage run returned None")

    actual_result = result
    if isinstance(result, tuple) and len(result) == 2:
        try:
            actual_result, _state = result
        except (ValueError, TypeError) as e:
            logger.error(
                "Lib log analyzer result unpack failed for path=%s: %s",
                path,
                e,
                exc_info=True,
            )
            raise RuntimeError(f"LogSage result had unexpected shape for path={path}: {e}") from e

    if not isinstance(actual_result, (list, LogSageAnalysisResult)):
        raise RuntimeError(
            f"LogSage result must be LogSageAnalysisResult or list[RawAnalysisResultItem] for path={path}, "
            f"got {type(actual_result).__name__}"
        )

    return log_analyzer_result_payload(actual_result)
