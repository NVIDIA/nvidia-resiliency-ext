#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Run log analysis in-process (lib) or via MCP; sync entry points for FT launcher.

Runs log analysis with a timeout (from config); blocks then returns result or None (skip).

Uses one long-lived LogAnalyzer (and its RequestCoalescer) per process so that
results are cached per file path—same mapping as the HTTP service. We run a
dedicated thread with an event loop and submit work to it from sync code.
"""

import asyncio
import logging
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Long-lived loop, config, and analyzer for the library path so the coalescer is reused.
_lib_loop: Optional[asyncio.AbstractEventLoop] = None
_lib_config: Any = None
_lib_analyzer: Any = None
_lib_loop_ready = threading.Event()
_lib_loop_starting = False
_lib_lock = threading.Lock()


def _ensure_analyzer_event_loop() -> None:
    """Start the dedicated thread and event loop if not already running"""
    global _lib_loop, _lib_loop_starting
    with _lib_lock:
        if _lib_loop is not None:
            return
        if _lib_loop_starting:
            pass  # another caller already started the thread; wait below
        else:
            _lib_loop_starting = True

            def _run_loop() -> None:
                global _lib_loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _lib_loop = loop
                _lib_loop_ready.set()
                loop.run_forever()

            threading.Thread(target=_run_loop, daemon=True).start()
    _lib_loop_ready.wait(timeout=5.0)
    if _lib_loop is None:
        with _lib_lock:
            _lib_loop_starting = False
        logger.warning("log analysis lib: event loop thread did not start in time")


def _get_or_create_analyzer(
    timeout_seconds: float = 60.0,
    use_lib_log_analysis: Optional[bool] = None,
) -> bool:
    """Ensure the long-lived analyzer exists; sets _lib_loop and _lib_analyzer globals.

    use_lib_log_analysis is only used when creating; pass None to reuse existing (set at init).
    Returns True if ready, False otherwise.
    """
    global _lib_config, _lib_analyzer
    _ensure_analyzer_event_loop()
    if not _lib_loop_ready.is_set() or _lib_loop is None:
        return False
    with _lib_lock:
        if _lib_analyzer is not None and _lib_config is not None:
            return True
        use_lib = use_lib_log_analysis if use_lib_log_analysis is not None else True
        try:
            from .analyzer import AnalyzerConfig, LogAnalyzer

            _lib_config = AnalyzerConfig(
                allowed_root="/",
                compute_timeout=timeout_seconds,
                use_lib_log_analysis=use_lib,
            )
            _lib_analyzer = LogAnalyzer(config=_lib_config)
            _lib_analyzer.set_event_loop(_lib_loop)
            if not use_lib:
                future = asyncio.run_coroutine_threadsafe(_lib_analyzer.connect_mcp(), _lib_loop)
                future.result(timeout=30)
        except Exception as e:
            _lib_analyzer = None
            _lib_config = None
            logger.warning("log analysis lib: failed to create analyzer: %s", e)
            return False
    return True


def ensure_analyzer_ready(
    timeout_seconds: float = 60.0,
    use_lib_log_analysis: bool = True,
) -> bool:
    """Eagerly create the analyzer (event loop, AnalyzerConfig, set_event_loop).
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
    (returns None). Timeout comes from _lib_config (set at init).

    Uses the analyzer's RequestCoalescer: results are cached per file path (same
    as the HTTP service). Repeat calls for the same path return the cached result;
    wl_restart selects the cycle when one file has multiple cycles.

    Args:
        log_path: Path to the cycle log file to analyze.
        wl_restart: Workload restart index within file (None = first or all).
            When a file contains multiple cycles, use this to select which cycle's result.

    Returns:
        Result dict from the analyzer on success, or None on timeout/error/skip.
    """
    from .analyzer import AnalyzerError

    if not _get_or_create_analyzer():
        return None

    validated = _lib_analyzer.validate_path(log_path, require_regular_file=True, reject_empty=False)
    if isinstance(validated, AnalyzerError):
        logger.debug("log analysis lib: skip (path validation): %s", validated.message)
        return None

    timeout = _lib_config.compute_timeout

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

    if isinstance(raw, AnalyzerError):
        logger.debug("log analysis lib: analysis error for %s: %s", log_path, raw.message)
        return None
    return _raw_to_result_dict(raw)
