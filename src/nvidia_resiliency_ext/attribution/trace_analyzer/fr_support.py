# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL flight recorder helpers (path from logs, analysis, dataflow fields, Markdown).

Heavy collective parsing lives in :mod:`fr_attribution` (``CollectiveAnalyzer``). This module
holds the attrsvc-facing pipeline: discover path → :func:`analyze_fr_dump` →
:func:`fr_fields_for_dataflow_record` / :func:`fr_markdown_appendix`.
"""

from __future__ import annotations

import ast
import asyncio
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from nvidia_resiliency_ext.attribution.base import NVRxAttribution
from nvidia_resiliency_ext.attribution.path_utils import path_is_under_allowed_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log scan: optional activation (Megatron prints TORCH_FR_DUMP_TEMP_FILE=...)
# ---------------------------------------------------------------------------

FR_DUMP_PATH_LOG_LINE_PATTERN = re.compile(r"TORCH_FR_DUMP_TEMP_FILE=(\S+)")
FR_DUMP_PATH_LOG_SCAN_LINES = 1000


def _fr_traces_exist_in_dir(directory: str) -> bool:
    """Return True if at least one ``_dump_*`` file exists directly inside ``directory``."""
    return bool(glob.glob(os.path.join(directory, "_dump_*")))


def _fr_traces_exist_for_prefix(prefix: str) -> bool:
    """Return True if at least one file matches ``prefix*`` (TORCH_FR_DUMP_TEMP_FILE is a path prefix).

    e.g. prefix='/tmp/checkpoints/_dump_' matches '/tmp/checkpoints/_dump_0', '_dump_1', ...
    """
    return bool(glob.glob(prefix + "*"))


def fr_path_resolvable_for_collective_analyzer(fr_path: str) -> bool:
    """Return True if ``fr_path`` is valid for :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution.CollectiveAnalyzer` ``fr_path``.

    Matches :meth:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution.CollectiveAnalyzer.preprocess_FR_dumps`:
    an existing directory, an existing dump file, or a **path prefix** when ``glob.glob(prefix + "*")``
    finds at least one file (Megatron ``TORCH_FR_DUMP_TEMP_FILE``).
    """
    if os.path.isdir(fr_path) or os.path.isfile(fr_path):
        return True
    return _fr_traces_exist_for_prefix(fr_path)


def _infer_checkpoints_dir_from_log_path(
    log_path: str, allowed_root: Optional[str] = None
) -> Optional[str]:
    """If ``log_path`` lives under a ``.../logs/...`` tree, return sibling ``.../checkpoints`` when
    it exists **and** contains at least one ``_dump_*`` trace file.

    Many training runs use ``<run>/logs/`` for Slurm step logs and ``<run>/checkpoints/`` for FR dumps
    (``_dump_<rank>``). The log line ``TORCH_FR_DUMP_TEMP_FILE=`` often points at a container-local path
    that attrsvc cannot read; the shared run directory is derivable from the log file path.
    """
    try:
        d = os.path.dirname(os.path.abspath(log_path))
    except (OSError, ValueError):
        return None
    while True:
        if os.path.basename(d) == "logs":
            run_root = os.path.dirname(d)
            cand = os.path.join(run_root, "checkpoints")
            if os.path.isdir(cand):
                if not _fr_traces_exist_in_dir(cand):
                    logger.debug(
                        "Inferred checkpoints dir %s has no _dump_* traces; skipping FR analysis",
                        cand,
                    )
                    return None
                if allowed_root is not None and not path_is_under_allowed_root(cand, allowed_root):
                    logger.warning(
                        "Inferred FR checkpoints path %r is outside allowed_root %r; skipping FR analysis",
                        cand,
                        allowed_root,
                    )
                    return None
                logger.debug(
                    "FR dump path inferred from log layout: log_path=%s -> checkpoints=%s",
                    log_path,
                    cand,
                )
                return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _read_torch_fr_dump_from_log(
    log_path: str, allowed_root: Optional[str] = None
) -> Optional[str]:
    """Scan the first ``FR_DUMP_PATH_LOG_SCAN_LINES`` lines of ``log_path`` for TORCH_FR_DUMP_TEMP_FILE.

    ``TORCH_FR_DUMP_TEMP_FILE`` is a **path prefix**, not a directory.  Megatron writes e.g.
    ``TORCH_FR_DUMP_TEMP_FILE=/tmp/checkpoints/_dump_`` and each rank appends its rank index
    (``_dump_0``, ``_dump_1``, …).

    Returns the prefix only when:
    - the extracted value is not a directory (a bare directory means the env var was set incorrectly), and
    - at least one file matching ``prefix*`` exists on the shared filesystem.
    """
    try:
        with open(log_path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= FR_DUMP_PATH_LOG_SCAN_LINES:
                    break
                m = FR_DUMP_PATH_LOG_LINE_PATTERN.search(line)
                if m:
                    prefix = m.group(1)
                    if os.path.isdir(prefix):
                        logger.warning(
                            "TORCH_FR_DUMP_TEMP_FILE=%r is a directory, not a path prefix; "
                            "env var is misconfigured — skipping FR analysis",
                            prefix,
                        )
                        return None
                    if not _fr_traces_exist_for_prefix(prefix):
                        logger.debug(
                            "TORCH_FR_DUMP_TEMP_FILE prefix %r has no matching trace files; "
                            "skipping FR analysis",
                            prefix,
                        )
                        return None
                    if allowed_root is not None and not path_is_under_allowed_root(
                        prefix, allowed_root
                    ):
                        logger.warning(
                            "TORCH_FR_DUMP_TEMP_FILE prefix %r is outside allowed_root %r; "
                            "skipping FR analysis",
                            prefix,
                            allowed_root,
                        )
                        return None
                    return prefix
    except OSError:
        pass
    return None


def extract_fr_dump_path(log_path: str, allowed_root: Optional[str] = None) -> Optional[str]:
    """Resolve FR dump path for attrsvc.

    Prefer ``<run>/checkpoints`` (directory containing ``_dump_*`` files) when ``log_path`` is
    under ``<run>/logs/`` (shared filesystem).  Otherwise scan the log for
    ``TORCH_FR_DUMP_TEMP_FILE=`` (a path prefix; may be container-local).

    When ``allowed_root`` is set, inferred directories and ``TORCH_FR_DUMP_TEMP_FILE`` prefixes must
    resolve under that root (same containment as log path validation); otherwise discovery returns
    ``None`` so log-injected paths cannot bypass the service path policy.

    Returns ``None`` if no valid FR traces are found — analysis should not be triggered.
    """
    inferred = _infer_checkpoints_dir_from_log_path(log_path, allowed_root=allowed_root)
    if inferred is not None:
        return inferred
    return _read_torch_fr_dump_from_log(log_path, allowed_root=allowed_root)


# ---------------------------------------------------------------------------
# Dump analysis (delegates to CollectiveAnalyzer in-process)
# ---------------------------------------------------------------------------


@dataclass
class FRAnalysisResult:
    """Result from running CollectiveAnalyzer on a NCCL flight recorder dump."""

    analysis_text: str  # process-group table (hanging/completed ranks per PG)
    hanging_ranks: str  # "hanging ranks: [...]" summary string from print_output


def fr_result_from_mcp_module_response(resp: Any) -> Optional[FRAnalysisResult]:
    """Build :class:`FRAnalysisResult` from :meth:`NVRxMCPClient.run_module` / ``run_module_resilient`` JSON.

    Expects the usual MCP envelope ``{\"result\": ..., \"recommendation\": ...}``.
    ``result`` is either the structured dict from
    :class:`~nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution.CollectiveAnalyzer`
    (``analysis_text`` + ``hanging_ranks``) or a plain string (hanging ranks only).
    The recommendation is intentionally ignored because FR is monitor-only.
    """
    if not isinstance(resp, dict):
        return None
    if resp.get("error"):
        logger.warning("FR MCP response error: %s", resp.get("error"))
        return None
    inner = resp.get("result")
    if isinstance(inner, dict) and "hanging_ranks" in inner:
        return FRAnalysisResult(
            analysis_text=str(inner.get("analysis_text") or ""),
            hanging_ranks=str(inner.get("hanging_ranks") or ""),
        )
    if isinstance(inner, str):
        return FRAnalysisResult(analysis_text="", hanging_ranks=inner)
    return None


async def analyze_fr_dump(dump_path: str) -> Optional[FRAnalysisResult]:
    """Run CollectiveAnalyzer on a flight recorder dump (deterministic table analysis only)."""
    try:
        from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import (
            CollectiveAnalyzer,
        )

        args = {
            "fr_path": dump_path,
            "pattern": "_dump_*",
            "model": None,
            "verbose": False,
            "health_check": False,
            "llm_analyze": False,
            "threshold": None,
        }

        def _run_sync() -> FRAnalysisResult:
            try:
                analyzer = CollectiveAnalyzer(args)
                loop = analyzer._loop
                analysis_text = loop.run_until_complete(analyzer.preprocess_FR_dumps())
                packed, _ = loop.run_until_complete(analyzer.print_output(analysis_text))
            finally:
                NVRxAttribution.reset_thread_event_loop()
            if isinstance(packed, dict):
                return FRAnalysisResult(
                    analysis_text=str(packed.get("analysis_text") or analysis_text),
                    hanging_ranks=str(packed.get("hanging_ranks") or ""),
                )
            return FRAnalysisResult(analysis_text=analysis_text, hanging_ranks=str(packed))

        return await asyncio.get_running_loop().run_in_executor(None, _run_sync)
    except Exception:
        logger.exception("FR dump analysis failed for %s", dump_path)
        return None


# ---------------------------------------------------------------------------
# Dataflow / Elasticsearch (s_fr_*)
# ---------------------------------------------------------------------------


def fr_fields_for_dataflow_record(
    fr_dump_path: Optional[str] = None,
    fr_analysis: Optional[FRAnalysisResult] = None,
) -> Dict[str, str]:
    """Build ``s_fr_*`` keys to merge into a posting record."""
    out: Dict[str, str] = {}
    if fr_dump_path:
        out["s_fr_dump_path"] = fr_dump_path
    if fr_analysis:
        out["s_fr_analysis"] = fr_analysis.analysis_text
        out["s_hanging_ranks"] = fr_analysis.hanging_ranks
        # Also provide a machine-friendly numeric array of hanging ranks for aggregations
        try:
            nums = [int(x) for x in re.findall(r"\d+", str(fr_analysis.hanging_ranks or ""))]
            out["l_hanging_ranks"] = nums
        except Exception:
            # Keep best-effort; don't fail record assembly on parse errors
            pass
    return out


# ---------------------------------------------------------------------------
# Markdown (Slack mrkdwn–compatible; also logs)
# ---------------------------------------------------------------------------


def _fr_hanging_ranks_meaningful(hanging_ranks: Optional[str]) -> bool:
    """True when FR output reports at least one hanging or missing rank to surface in Slack."""
    if not hanging_ranks or not hanging_ranks.strip():
        return False
    t = hanging_ranks.strip()
    low = t.lower()
    if not low.startswith("hanging ranks:"):
        return True
    bracket = t.find("[")
    if bracket == -1:
        rest = t.split(":", 1)[1].strip() if ":" in t else t
        return bool(rest) and rest not in ("[]", "None")
    try:
        payload = ast.literal_eval(t[bracket:])
    except (ValueError, SyntaxError):
        return True
    if isinstance(payload, list):
        return len(payload) > 0
    return bool(payload)


def fr_markdown_appendix(
    *,
    dump_path: Optional[str] = None,
    analysis_text: Optional[str] = None,
    hanging_ranks: Optional[str] = None,
) -> str:
    """Slack/posting appendix for FR: path is omitted; section only if ranks are non-empty.

    ``dump_path`` is ignored here (still stored in dataflow via :func:`fr_fields_for_dataflow_record`).
    """
    if not _fr_hanging_ranks_meaningful(hanging_ranks):
        return ""
    parts: list[str] = ["\n*Flight recorder*"]
    if hanging_ranks:
        parts.append(f"\n*Hanging / missing ranks:*\n`{hanging_ranks.strip()}`")
    if analysis_text and analysis_text.strip():
        parts.append(f"\n*FR collective analysis:*\n```{analysis_text.strip()}```")
    return "".join(parts)


def fr_markdown_appendix_from_result(
    dump_path: Optional[str],
    result: Optional[FRAnalysisResult],
) -> str:
    """Same as :func:`fr_markdown_appendix` but takes a :class:`FRAnalysisResult`."""
    if result is None:
        return fr_markdown_appendix(dump_path=dump_path)
    return fr_markdown_appendix(
        dump_path=dump_path,
        analysis_text=result.analysis_text,
        hanging_ranks=result.hanging_ranks,
    )
