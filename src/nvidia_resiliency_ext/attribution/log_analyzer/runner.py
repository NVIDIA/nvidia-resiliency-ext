# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run log analysis in-process (lib) or via MCP; sync entry points for FT launcher.

Runs log analysis with a timeout (from config); blocks then returns result or None (skip).

Uses one long-lived LogAnalyzer (and its RequestCoalescer) per process so that
results are cached per file path—same mapping as the HTTP service. We run a
dedicated thread with an event loop and submit work to it from sync code.

The HTTP service (nvrx_attrsvc) does not use this module; it builds
:class:`~nvidia_resiliency_ext.attribution.analyzer.engine.Analyzer` with
``ALLOWED_ROOT`` from config, and ``analyze()`` / ``submit()`` validate paths
under that root.
"""

import asyncio
import logging
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Long-lived loop and analyzer for the library path so the coalescer is reused.
_lib_loop: Optional[asyncio.AbstractEventLoop] = None
_lib_analyzer: Any = None
_lib_loop_ready = threading.Event()
_lib_loop_starting = False
_lib_loop_thread: Optional[threading.Thread] = None
_lib_lock = threading.Lock()

_LOOP_START_TIMEOUT_S = 5.0


def _ensure_analyzer_event_loop() -> None:
    """Start the dedicated thread and event loop if not already running."""
    global _lib_loop, _lib_loop_starting, _lib_loop_thread
    with _lib_lock:
        if _lib_loop is not None:
            return
        if not _lib_loop_starting:
            _lib_loop_starting = True

            def _run_loop() -> None:
                global _lib_loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _lib_loop = loop
                _lib_loop_ready.set()
                loop.run_forever()

            t = threading.Thread(
                target=_run_loop,
                daemon=True,
                name="nvrx-attrib-log-loop",
            )
            _lib_loop_thread = t
            t.start()

    _lib_loop_ready.wait(timeout=_LOOP_START_TIMEOUT_S)

    with _lib_lock:
        if _lib_loop is not None:
            _lib_loop_starting = False
            return

        worker = _lib_loop_thread
        if worker is not None and worker.is_alive():
            logger.warning(
                "log analysis lib: event loop thread still starting after %.0fs; "
                "not spawning another worker (wait for readiness on next call)",
                _LOOP_START_TIMEOUT_S,
            )
            return

        _lib_loop_starting = False
        _lib_loop_thread = None

    logger.warning("log analysis lib: event loop thread did not start in time")


def _get_or_create_analyzer(
    timeout_seconds: float = 60.0,
    use_lib_log_analysis: Optional[bool] = None,
) -> bool:
    """Ensure the long-lived analyzer exists; sets _lib_loop and _lib_analyzer globals.

    use_lib_log_analysis is only used when creating; pass None to reuse existing (set at init).
    Returns True if ready, False otherwise.
    """
    global _lib_analyzer
    _ensure_analyzer_event_loop()
    if not _lib_loop_ready.is_set() or _lib_loop is None:
        return False
    with _lib_lock:
        if _lib_analyzer is not None:
            return True
        use_lib = use_lib_log_analysis if use_lib_log_analysis is not None else True
        try:
            from ..analyzer import Analyzer

            # Permissive root: only enforces absolute paths under /. Callers must
            # restrict paths themselves if needed (e.g. FT host validates before calling).
            _lib_analyzer = Analyzer(
                allowed_root="/",
                use_lib_log_analysis=use_lib,
                compute_timeout=timeout_seconds,
            )
            _lib_analyzer.set_event_loop(_lib_loop)
            if not use_lib:
                future = asyncio.run_coroutine_threadsafe(_lib_analyzer.connect_mcp(), _lib_loop)
                future.result(timeout=30)
        except Exception as e:
            _lib_analyzer = None
            logger.warning("log analysis lib: failed to create analyzer: %s", e)
            return False
    return True


def ensure_analyzer_ready(
    timeout_seconds: float = 60.0,
    use_lib_log_analysis: bool = True,
) -> bool:
    """Eagerly create the analyzer (event loop, Analyzer, set_event_loop).
    Call at client init for fail-fast. Returns True if ready."""
    return _get_or_create_analyzer(timeout_seconds, use_lib_log_analysis)


def _raw_to_result_dict(raw: Any) -> Optional[Dict[str, Any]]:
    """Convert analyzer result to the dict shape used by attribution_no_restart."""
    if hasattr(raw, "result"):
        r = getattr(raw, "result", None)
        return r if isinstance(r, dict) else None
    return None


def run_log_analysis_sync(
    log_path: str,
    wl_restart: Optional[int] = None,
    user: str = "",
    job_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Run log analysis synchronously with a timeout.

    If the analysis does not complete within the configured timeout, the result is skipped
    (returns None). Timeout comes from the analyzer coalescer (set at init).

    Uses the analyzer's RequestCoalescer: results are cached per file path (same
    as the HTTP service). Repeat calls for the same path return the cached result;
    wl_restart selects the cycle when one file has multiple cycles.

    Args:
        log_path: Path to the cycle log file to analyze. With this module's
            ``allowed_root="/"``, enforce a stricter allowed prefix at the call site if required.
        wl_restart: Workload restart index within file (None = first or all).
            When a file contains multiple cycles, use this to select which cycle's result.

    Returns:
        Result dict from the analyzer on success, or None on timeout/error/skip.
    """
    from .types import LogAnalyzerError

    if not _get_or_create_analyzer():
        return None

    validated = _lib_analyzer.validate_path(log_path, require_regular_file=True, reject_empty=False)
    if isinstance(validated, LogAnalyzerError):
        logger.debug("log analysis lib: skip (path validation): %s", validated.message)
        return None

    timeout = _lib_analyzer.compute_timeout

    async def _run() -> Any:
        await _lib_analyzer.submit(validated, user=user, job_id=job_id)
        return await _lib_analyzer.analyze(validated, wl_restart=wl_restart)

    try:
        future = asyncio.run_coroutine_threadsafe(_run(), _lib_loop)
        raw = future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.info("log analysis lib: skipped (timeout after %.0fs): %s", timeout, log_path)
        return None
    except Exception as e:
        logger.warning("log analysis lib: skip (exception): %s: %s", type(e).__name__, e)
        return None

    if isinstance(raw, LogAnalyzerError):
        logger.debug("log analysis lib: analysis error for %s: %s", log_path, raw.message)
        return None
    return _raw_to_result_dict(raw)
