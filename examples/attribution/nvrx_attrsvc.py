#  Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging
import os
import stat
from importlib.resources import files as pkg_files
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="NVRX_ATTRSVC_",
    )


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

            # Connect to the MCP server and run analysis
            try:
                client = _create_mcp_client()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "mcp_init_failed",
                        "message": f"failed to initialize mcp client: {str(e)}".lower().rstrip("."),
                    },
                )
            async with client:
                log_result = await client.run_module(
                    module_name="log_analyzer",
                    log_path=normalized,
                    model=cfg.LLM_MODEL,
                    temperature=0.2,
                    exclude_nvrx_logs=False,
                    is_per_cycle=True,
                    top_p=0.7,
                    max_tokens=8192,
                )
                logger.info(f"Result preview: {str(log_result)[:200]}...")

                # 1. Access the main text blob inside the nested list
                # data['result'] is a list, the first item is a list, and the text is the first item of that.
                if 'result' in log_result and len(log_result['result']) > 0:
                    for item in log_result['result']:
                        raw_text = item[0]

                        # 2. Extract the First Line
                        # Split by newlines and take the first element
                        first_line = raw_text.split('\n')[0]

                        # 3. Extract text after 'Attribution:'
                        # Split the text by the specific key "Attribution:" and take the second part
                        # We use .strip() to remove the leading newline character
                        attribution_text = raw_text.split('Attribution:')
                        if len(attribution_text) > 1:
                            attribution_text = attribution_text[1].strip()
                            attribution_text = attribution_text.replace('"\\', "").replace('\"', "").split("\n\n")[0]
                        else:
                            attribution_text = ""
                        data = {
                            "s_cluster": "oci-hsg",
                            "s_user": "nvrx_attr",
                            "s_attribution": attribution_text,
                            "s_auto_resume": first_line,
                            "s_auto_resume_explanation": "",
                            "s_jobid": "111",
                            "ts_current_time": round(datetime.now().timestamp() * 1000),
                        }
                        post(data=data, project="df-nvrxattr-test1")
                

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
    return ["python", str(resource)]


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
