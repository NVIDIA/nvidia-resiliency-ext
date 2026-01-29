#  Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

import asyncio
import logging
import os
import re
import stat
import sys
import time
from datetime import datetime
from importlib.resources import files as pkg_files
from typing import Any, Awaitable, Callable, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from nvdataflow import post as nv_post

    HAS_NVDATAFLOW = True
except ImportError:
    nv_post = None
    HAS_NVDATAFLOW = False

from slack_bolt.app import App
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from nvidia_resiliency_ext.attribution.mcp_integration.mcp_client import NVRxMCPClient

# Setup logging (configurable via NVRX_ATTRSVC_LOG_LEVEL_NAME env: DEBUG|INFO|WARNING|ERROR|CRITICAL)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Typed configuration loaded from environment/.env (pydantic-settings v2)."""

    FAST_API_ROOT_PATH: str = Field(default="", description="FastAPI root path")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LLM_MODEL: str = Field(default="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1")
    ALLOWED_ROOT: str = Field(
        ..., description="Absolute base directory allowed for input paths (required)"
    )
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    LOG_LEVEL_NAME: str = Field(default="INFO")

    CLUSTER_NAME: str = Field(default="", description="Cluster name")
    NVDATAFLOW_PROJECT: str = Field(
        default="coreai_resiliency_osiris", description="nvdataflow index"
    )

    SLACK_BOT_TOKEN: str = Field(default="", description="Slack bot token")
    SLACK_CHANNEL: str = Field(default="#osiris-alerts", description="Slack channel")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="NVRX_ATTRSVC_",
    )


def get_slack_user_email(userID: str, token: str) -> str | None:
    client = WebClient(token=token)

    try:
        # Fetch all users (pagination may be needed for very large teams)
        user_id = client.users_lookupByEmail(email=f"{userID}@nvidia.com").get("user")["id"]

        return user_id  # User not found
    except SlackApiError as e:
        logger.error(f"Error fetching user email: {e.response['error']}")
        return None


def send_slack_notification(data: dict, slack_bot_token: str, slack_channel: str):
    """
    Send slack notification.
    """

    app = App(token=slack_bot_token)

    client = WebClient(token=slack_bot_token)

    slack_user_id = get_slack_user_email(data['s_user'], slack_bot_token)

    mention = f"\n<@{slack_user_id}>" if slack_user_id else ""
    if not slack_user_id:
        logger.error(f"User {data['s_user']} not found in Slack")

    text = (
        f"*Job ID:* `{data['s_job_id']}`\n"
        "*Failed due to:*\n"
        f"```{data['s_attribution']}```\n"
        "*Terminal issue:*\n"
        f"```{data['s_auto_resume_explanation']}```"
        f"{mention}"
    )
    try:
        response = client.chat_postMessage(
            channel=slack_channel,  # or channel ID like "C1234567890"
            text=text,
        )
    except SlackApiError as e:
        logger.error(f"Error posting message: {e.response['error']}")


class RequestCoalescer:
    """
    Ensures only one request processes a given log file at a time.
    Subsequent requests for the same file wait for the result and get it from cache.

    This implements the "single-flight" pattern to avoid duplicate LLM calls
    for the same log file when multiple requests arrive simultaneously.
    """

    def __init__(self, cleanup_interval_hours: float = 1.0):
        """
        Args:
            cleanup_interval_hours: How often to clean up old cache entries (default 1 hour).
                                   Entries older than this are removed during cleanup.
        """
        self._cache: Dict[str, tuple[Any, float]] = {}  # path -> (result, timestamp)
        self._in_flight: Dict[str, asyncio.Event] = {}  # path -> completion event
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval_hours * 3600.0  # Convert to seconds
        self._last_cleanup = time.monotonic()

        # Statistics counters
        self._stats_cache_hits = 0
        self._stats_cache_misses = 0
        self._stats_coalesced_requests = 0  # Requests that waited for in-flight
        self._stats_total_computes = 0  # Total compute_fn invocations
        self._stats_total_expired_cleaned = 0  # Total entries removed by cleanup
        self._stats_compute_errors = 0  # Compute failures

    def _cleanup_expired_locked(self) -> int:
        """
        Remove all expired entries from cache.

        Note: Caller must hold self._lock.

        Returns:
            Number of entries removed
        """
        now = time.monotonic()
        expired_keys = [
            path
            for path, (_, cache_time) in self._cache.items()
            if now - cache_time >= self._cleanup_interval
        ]
        for path in expired_keys:
            del self._cache[path]
        if expired_keys:
            self._stats_total_expired_cleaned += len(expired_keys)
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics about the cache and request coalescing.

        Returns:
            Dictionary with statistics
        """
        async with self._lock:
            total_requests = (
                self._stats_cache_hits + self._stats_cache_misses + self._stats_coalesced_requests
            )
            return {
                "cache_size": len(self._cache),
                "in_flight_count": len(self._in_flight),
                "total_requests": total_requests,
                "cache_hits": self._stats_cache_hits,
                "cache_misses": self._stats_cache_misses,
                "coalesced_requests": self._stats_coalesced_requests,
                "total_computes": self._stats_total_computes,
                "compute_errors": self._stats_compute_errors,
                "total_expired_cleaned": self._stats_total_expired_cleaned,
                "cleanup_interval_hours": self._cleanup_interval / 3600.0,
            }

    async def get_or_compute(self, path: str, compute_fn: Callable[[], Awaitable[Any]]) -> Any:
        """
        Get cached result or compute it (only once per path).

        Args:
            path: The normalized log file path (cache key)
            compute_fn: Async function to call if not cached/in-flight

        Returns:
            The result (from cache or freshly computed)
        """
        while True:
            event_to_wait: Optional[asyncio.Event] = None
            should_compute = False

            async with self._lock:
                # Periodically clean up expired entries to prevent memory leaks
                now = time.monotonic()
                if now - self._last_cleanup >= self._cleanup_interval:
                    self._cleanup_expired_locked()
                    self._last_cleanup = now

                # Check cache first (log files are immutable, so cached results are always valid)
                if path in self._cache:
                    result, _ = self._cache[path]
                    self._stats_cache_hits += 1
                    logger.info(f"Cache hit for {path}")
                    return result

                # Check if another request is already processing this path
                if path in self._in_flight:
                    # Capture the event while holding the lock
                    event_to_wait = self._in_flight[path]
                    self._stats_coalesced_requests += 1
                    logger.info(f"Waiting for in-flight request for {path}")
                else:
                    # We're the first - create event and mark as in-flight
                    event = asyncio.Event()
                    self._in_flight[path] = event
                    self._stats_cache_misses += 1
                    should_compute = True

            # Outside the lock: either wait for in-flight or compute
            if event_to_wait is not None:
                # Wait for the in-flight request to complete (using captured event)
                await event_to_wait.wait()
                # Loop back to check cache (the in-flight request should have cached it)
                continue

            if should_compute:
                # We're responsible for computing
                try:
                    async with self._lock:
                        self._stats_total_computes += 1
                    logger.info(f"Computing result for {path}")
                    result = await compute_fn()

                    # Cache the result
                    async with self._lock:
                        self._cache[path] = (result, time.monotonic())
                        logger.info(f"Cached result for {path}")

                    return result
                except Exception as e:
                    async with self._lock:
                        self._stats_compute_errors += 1
                    logger.error(f"Failed to compute result for {path}: {e}", exc_info=True)
                    raise
                finally:
                    # Signal completion and clean up in-flight tracking
                    async with self._lock:
                        if path in self._in_flight and self._in_flight[path] is event:
                            del self._in_flight[path]
                    event.set()


# Global request coalescer instance
_request_coalescer = RequestCoalescer()


def setup() -> "Settings":
    """
    Group environment configuration and logging setup for nvrx_attrsvc.
    Returns a configured Settings instance.
    """
    try:
        cfg = Settings()  # type: ignore[call-arg]
    except Exception as e:
        # Fail fast if required settings are missing or invalid
        raise SystemExit(f"nvrx_attrsvc configuration error: {e}")
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL_NAME, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    # Validate ALLOWED_ROOT
    allowed_root = os.path.realpath(cfg.ALLOWED_ROOT)
    if not os.path.isabs(allowed_root):
        raise SystemExit("ALLOWED_ROOT must be an absolute path")
    if not os.path.isdir(allowed_root):
        raise SystemExit(f"ALLOWED_ROOT is not a directory: {allowed_root}")
    # Ensure we can traverse/read the root (execute bit for directories)
    if not os.access(allowed_root, os.X_OK | os.R_OK):
        raise SystemExit(f"ALLOWED_ROOT is not accessible: {allowed_root}")
    return cfg


# Response models
class AttrSvcResult(BaseModel):
    """Response model for log analysis."""

    result: Any
    status: str = "completed"


class ErrorResponse(BaseModel):
    """Standard error body for nvrx_attrsvc."""

    error_code: str
    message: str
    details: Any | None = None


class SubmitRequest(BaseModel):
    """Submission model for analysis requests."""

    log_path: str


class SubmitResponse(BaseModel):
    """Response model for submit endpoint."""

    submitted: bool


def create_app(cfg: Settings) -> FastAPI:
    """
    Construct and return the FastAPI app for the NVRX Attribution Service (nvrx_attrsvc).
    """
    app = FastAPI(
        title="NVRX Attribution Service",
        summary="nvrx_attrsvc - NVRX attribution service for artifact/log analysis",
        contact={
            "name": "NVRX Attribution Service",
            "email": "nvrx@nvidia.com",
        },
        root_path=cfg.FAST_API_ROOT_PATH,
        debug=cfg.DEBUG,
    )

    # Global exception handlers to standardize error bodies
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException):
        # Support structured detail; otherwise fall back to generic mapping
        detail = exc.detail
        if isinstance(detail, dict):
            error_code = str(detail.get("error_code", exc.status_code)).lower()
            message = str(detail.get("message", "error")).lower().rstrip(".")
            body = ErrorResponse(error_code=error_code, message=message).model_dump()
        else:
            body = ErrorResponse(
                error_code=str(exc.status_code).lower(),
                message=str(detail).lower().rstrip("."),
            ).model_dump()
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_, exc: Exception):
        logger.exception("Unhandled exception in nvrx_attrsvc", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="internal_error", message="internal server error"
            ).model_dump(),
        )

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        """
        nvrx_attrsvc health check endpoint.
        """
        return {"status": "OK"}

    @app.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """
        Get cache and request coalescing statistics.

        Returns statistics about cache hits, misses, in-flight requests, etc.
        """
        return await _request_coalescer.get_stats()

    @app.post(
        "/logs",
        response_model=SubmitResponse,
        responses={
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def submit_analysis(req: SubmitRequest) -> SubmitResponse:
        """
        Submit a new analysis job to nvrx_attrsvc. For this example server the POST is a no-op and
        analysis is performed synchronously via GET. We still accept the request
        to align with the client flow.

        Possible errors:
          - 400: missing or invalid log_path
        """
        if not req.log_path:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "invalid_request", "message": "log_path is required"},
            )
        # Log the request (file may not exist yet - validation happens at GET time)
        logger.info(f"POST /logs - received: {req.log_path}")
        # Validate the path exists and is accessible
        _normalize_and_validate_path(req.log_path, cfg, require_regular_file=True)
        return SubmitResponse(submitted=True)

    @app.get(
        "/print",
        responses={
            200: {"content": {"text/plain": {}}},
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def print_log_path(
        log_path: str = Query(
            ..., description="Absolute path to a file or directory under allowed root"
        )
    ) -> str:
        """
        Return the first 4KB of a file for preview.
        """
        max_bytes = 4096
        try:
            normalized = _normalize_and_validate_path(log_path, cfg, require_regular_file=False)
            with open(normalized, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
                return content
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "file_error",
                    "message": f"file error: {str(e)}".lower().rstrip("."),
                },
            )

    @app.get(
        "/logs",
        response_model=AttrSvcResult,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            422: {
                "content": {
                    "application/json": {
                        "examples": {
                            "validation_error": {
                                "summary": "Validation error",
                                "value": {
                                    "detail": [
                                        {
                                            "loc": ["query", "log_path"],
                                            "msg": "field required",
                                            "type": "value_error.missing",
                                        }
                                    ]
                                },
                            }
                        }
                    }
                }
            },
        },
    )
    async def attribution_log_path(
        log_path: str = Query(
            ..., min_length=1, description="Absolute path to a log file under allowed root"
        )
    ) -> AttrSvcResult:
        """
        nvrx_attrsvc: Analyze logs from a specific path.

        Args:
            log_path: Path to the log file to analyze (must be under ALLOWED_ROOT and not a symlink)

        Returns:
            AttrSvcResult with analysis results

        Errors:
            - 400 invalid path (relative, outside ALLOWED_ROOT, symlink, unreadable, not regular file)
            - 404 path not found
        """
        # Log the request first (before validation, so we see what was requested)
        logger.info(f"GET /logs - received: {log_path}")
        try:
            normalized = _normalize_and_validate_path(log_path, cfg, require_regular_file=True)
            logger.info(f"Analyzing log: {normalized}")

            async def _do_analysis():
                """Inner function that performs the actual LLM analysis."""
                try:
                    client = _create_mcp_client()
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error_code": "mcp_init_failed",
                            "message": f"failed to initialize mcp client: {str(e)}".lower().rstrip(
                                "."
                            ),
                        },
                    )
                async with client:
                    # Detect per-cycle logs: filename ends with _cycle<N>.log
                    is_per_cycle = bool(re.search(r'_cycle\d+\.log$', normalized))
                    s_time = time.time()
                    log_result = await client.run_module(
                        module_name="log_analyzer",
                        log_path=normalized,
                        model=cfg.LLM_MODEL,
                        temperature=0.0,
                        exclude_nvrx_logs=False,
                        is_per_cycle=is_per_cycle,
                        top_p=1,
                        max_tokens=8192,
                    )
                    logger.info(f"Result preview: {str(log_result)}")

                    e_time = time.time()
                    # 1. Access the main text blob inside the nested list
                    # data['result'] is a list, the first item is a list, and the text is the first item of that.
                    if 'result' in log_result and len(log_result['result']) > 0:
                        for item in log_result['result']:
                            raw_text = item[0]

                            # 2. Extract the auto_resume raw
                            auto_resume_raw = raw_text.split('\n')
                            # Split by newlines and take the first element
                            auto_resume = auto_resume_raw[0]
                            try:
                                auto_resume_explanation = auto_resume_raw[1][:-1]
                            except Exception as e:
                                auto_resume_explanation = ""
                                logger.info(f"Failed to extract auto resume explanation: {e}")

                            # 3. Extract text after 'Attribution:'
                            # Split the text by the specific key "Attribution:" and take the second part
                            # We use .strip() to remove the leading newline character
                            attribution_text = raw_text.split('Attribution:')
                            if len(attribution_text) > 1:
                                attribution_text = attribution_text[1].strip()
                                try:
                                    checkpoint_saved = attribution_text.split("\n\n")[1]
                                except (IndexError, AttributeError):
                                    checkpoint_saved = "false"
                                attribution_text = (
                                    attribution_text.replace('"\\', "")
                                    .replace('\"', "")
                                    .split("\n\n")[0]
                                )
                            else:
                                attribution_text = ""
                                checkpoint_saved = "false"

                            # normalize checkpoint_saved → int flag
                            checkpoint_saved_flag = 0
                            if (
                                isinstance(checkpoint_saved, str)
                                and checkpoint_saved.strip().lower() != "false"
                            ):
                                checkpoint_saved_flag = 1

                            try:
                                match = re.search(r"_(\d+)_date_", normalized)
                                if not match:
                                    raise ValueError("Job ID not found in path")
                                jobid = match.group(1)
                            except Exception as e:
                                jobid = ""
                                logger.info(f"Failed to extract job ID: {e}")

                            cycle_id = 0
                            try:
                                match = re.search(r"_cycle(\d+)\.log$", normalized)
                                if not match:
                                    raise ValueError("Cycle ID not found in path")

                                cycle_id = int(match.group(1))

                            except Exception as e:
                                logger.info(f"Failed to extract cycle ID: {e}")

                            logger.info("jobid: %s", jobid)
                            logger.info("log_path: %s", normalized)
                            logger.info("auto_resume: %s", auto_resume)
                            logger.info("auto_resume_explanation: %s", auto_resume_explanation)
                            logger.info("attribution_text: %s", attribution_text)
                            data = {
                                "s_cluster": cfg.CLUSTER_NAME,
                                "s_user": "user",
                                "s_attribution": attribution_text,
                                "s_auto_resume": auto_resume,
                                "s_auto_resume_explanation": auto_resume_explanation,
                                "s_job_id": jobid,
                                "l_cycle_id": cycle_id,
                                "s_log_path": normalized,
                                "l_checkpoint_saved": checkpoint_saved_flag,
                                "d_processing_time": round(e_time - s_time, 2),
                                "ts_current_time": round(datetime.now().timestamp() * 1000),
                            }

                            if auto_resume == "STOP - DONT RESTART IMMEDIATE":
                                # Send slack notification
                                send_slack_notification(
                                    data, cfg.SLACK_BOT_TOKEN, cfg.SLACK_CHANNEL
                                )

                            if (
                                HAS_NVDATAFLOW
                                and cfg.NVDATAFLOW_PROJECT
                                and auto_resume != "NO LOGS"
                            ):
                                result = nv_post(data=data, project=cfg.NVDATAFLOW_PROJECT)
                            else:
                                if HAS_NVDATAFLOW:
                                    logger.error("nvdataflow index is missing")
                                if cfg.NVDATAFLOW_PROJECT:
                                    logger.error("can't import nvdataflow")
                                if auto_resume == "NO LOGS":
                                    logger.error("no logs to post")
                    return log_result

            # Use request coalescing: only one request per log file is processed,
            # others wait for the cached result
            log_result = await _request_coalescer.get_or_compute(normalized, _do_analysis)

            return AttrSvcResult(result=log_result, status="completed")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"NVRX Attribution Service error: {e}")
            raise HTTPException(
                status_code=500,
                detail={"error_code": "internal_error", "message": str(e).lower().rstrip(".")},
            )

    return app


def _get_server_command() -> list[str]:
    """
    Resolve and return the server launcher command for the MCP client.
    """
    pkg = "nvidia_resiliency_ext.attribution.mcp_integration"
    try:
        resource = pkg_files(pkg).joinpath("server_launcher.py")
    except Exception as e:
        raise FileNotFoundError(f"failed to locate server_launcher.py in package {pkg}: {e}")
    if not resource.exists():
        raise FileNotFoundError(f"server launcher not found in package: {pkg}/server_launcher.py")
    return [sys.executable, str(resource)]


def _create_mcp_client() -> NVRxMCPClient:
    """
    Create and return an initialized NVRxMCPClient.
    """
    return NVRxMCPClient(_get_server_command())


def _normalize_and_validate_path(
    user_path: str, cfg: Settings, *, require_regular_file: bool
) -> str:
    """
    Normalize and validate an input path:
      - Must be absolute
      - Resolve realpath under allowed root (no traversal outside)
      - Must not be a symlink
      - If require_regular_file=True, must be a regular readable file; otherwise allow directories too
    Returns the normalized absolute path or raises HTTPException(400/404).
    """
    if not os.path.isabs(user_path):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "invalid_path", "message": "log_path must be absolute"},
        )
    real = os.path.realpath(user_path)
    allowed_root = os.path.realpath(cfg.ALLOWED_ROOT)
    try:
        common = os.path.commonpath([real, allowed_root])
    except Exception:
        raise HTTPException(
            status_code=400, detail={"error_code": "invalid_path", "message": "invalid path"}
        )
    if common != allowed_root:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "outside_root",
                "message": "access outside allowed root is not permitted",
            },
        )
    try:
        st = os.lstat(real)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail={"error_code": "not_found", "message": "path not found"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "stat_failed",
                "message": f"stat failed: {str(e)}".lower().rstrip("."),
            },
        )
    if stat.S_ISLNK(st.st_mode):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "symlink_not_allowed", "message": "symlinks are not allowed"},
        )
    if require_regular_file:
        if not stat.S_ISREG(st.st_mode):
            raise HTTPException(
                status_code=400,
                detail={"error_code": "not_regular", "message": "path must be a regular file"},
            )
    # Basic readability
    if not os.access(real, os.R_OK):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "not_readable", "message": "path is not readable"},
        )
    return real


def main():
    """Entry point for the NVRX Attribution Service."""
    cfg = setup()
    logger.info(f"Starting NVRX Attribution Service (nvrx_attrsvc) on {cfg.HOST}:{cfg.PORT}")
    logger.info(f"nvrx_attrsvc API Documentation: http://{cfg.HOST}:{cfg.PORT}/docs")
    uvicorn.run(create_app(cfg), host=cfg.HOST, port=cfg.PORT)


if __name__ == "__main__":
    main()
