#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Attribution service client for nvrx_smonsvc."""

import logging
import time
from collections.abc import Callable

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from nvrx_attrsvc.routes import (
    PARAM_JOB_ID,
    PARAM_LOG_PATH,
    PARAM_USER,
    PARAM_WL_RESTART,
    ROUTE_LOGS,
)

logger = logging.getLogger(__name__)


class AttrsvcClient:
    """
    HTTP client for interacting with the attribution service (attrsvc).

    Handles POST/GET requests to /logs with retry logic and rate limiting.
    """

    # Default configuration
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_MAX_ATTEMPTS = 3
    DEFAULT_RETRY_DELAY = 5.0
    DEFAULT_RATE_LIMIT_BASE_DELAY = 10.0
    DEFAULT_RATE_LIMIT_MAX_DELAY = 60.0
    DEFAULT_REQUEST_THROTTLE = 0.25  # seconds between requests

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        rate_limit_base_delay: float = DEFAULT_RATE_LIMIT_BASE_DELAY,
        rate_limit_max_delay: float = DEFAULT_RATE_LIMIT_MAX_DELAY,
        request_throttle: float = DEFAULT_REQUEST_THROTTLE,
        on_rate_limited: Callable[[], None] | None = None,
    ):
        """
        Initialize the attribution service client.

        Args:
            base_url: Base URL of the attribution service (e.g., http://localhost:8000)
            timeout: HTTP request timeout in seconds
            max_attempts: Maximum number of retry attempts
            retry_delay: Delay between retries for server errors
            rate_limit_base_delay: Base delay for 429 rate limiting (exponential backoff)
            rate_limit_max_delay: Maximum delay for rate limiting
            request_throttle: Delay between successful requests to prevent rate limiting
            on_rate_limited: Optional callback when rate limited (for stats tracking)
        """
        if httpx is None:
            raise ImportError("httpx is required. Install with: pip install httpx")

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._rate_limit_base_delay = rate_limit_base_delay
        self._rate_limit_max_delay = rate_limit_max_delay
        self._request_throttle = request_throttle
        self._on_rate_limited = on_rate_limited
        self._client = httpx.Client(timeout=timeout)

        # Health check cache to avoid blocking HTTP calls on every /healthz request
        self._health_cache: tuple[bool, float] | None = None
        self._health_cache_ttl = 60.0  # seconds

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "AttrsvcClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def check_health_cached(self) -> bool:
        """
        Check attrsvc connectivity with caching to avoid blocking on every request.

        Does not call attrsvc /stats; uses a lightweight request to the base URL.
        Returns cached result if within TTL, otherwise makes a new request.

        Returns:
            True if attrsvc is reachable, False otherwise
        """
        now = time.time()

        # Check cache
        if self._health_cache is not None:
            is_healthy, timestamp = self._health_cache
            if now - timestamp < self._health_cache_ttl:
                return is_healthy

        # Cache miss or expired - lightweight connectivity check (no /stats)
        is_healthy = False
        if self._client is not None:
            response = None
            try:
                response = self._client.get(self._base_url, timeout=5.0)
                is_healthy = response.status_code < 500
            except Exception:
                pass
            finally:
                if response is not None:
                    response.close()
        self._health_cache = (is_healthy, now)
        return is_healthy

    def request_with_retry(
        self,
        method: str,
        job_id: str,
        log_path: str,
        on_success: Callable,
        on_client_error: Callable[[str], None],
        on_404: Callable | None = None,
        user: str = "unknown",
        cycle: int | None = None,
    ) -> None:
        """
        Make an HTTP request to /logs endpoint with retry logic.

        Args:
            method: "POST" or "GET"
            job_id: Job ID for logging (also sent to attrsvc for splitlog mode)
            log_path: Log file path
            on_success: Callback for 200 response, receives response
            on_client_error: Callback for 4xx errors, receives error message string
            on_404: Optional callback for 404 errors (POST only - file not found)
            user: SLURM job user (for POST requests)
            cycle: Cycle number for GET requests in splitlog mode (None = latest)
        """
        url = f"{self._base_url}{ROUTE_LOGS}"

        for attempt in range(self._max_attempts):
            response = None
            try:
                if method == "POST":
                    response = self._client.post(
                        url,
                        json={
                            PARAM_LOG_PATH: log_path,
                            PARAM_USER: user,
                            PARAM_JOB_ID: job_id,
                        },
                        headers={"accept": "application/json"},
                    )
                else:  # GET
                    params = {PARAM_LOG_PATH: log_path}
                    if cycle is not None:
                        params[PARAM_WL_RESTART] = cycle
                    response = self._client.get(
                        url,
                        params=params,
                    )

                if response.status_code == 200:
                    on_success(response)
                    # Throttle successful requests to prevent rate limiting
                    if self._request_throttle > 0:
                        time.sleep(self._request_throttle)
                    return
                elif response.status_code >= 500 or response.status_code == 429:
                    # Server error or rate limited - retry
                    if response.status_code == 429:
                        if self._on_rate_limited:
                            self._on_rate_limited()
                        # Check for Retry-After header first
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                                logger.warning(
                                    f"[{job_id}] {method} rate limited, waiting {wait_time}s "
                                    f"(Retry-After header, attempt {attempt + 1}/{self._max_attempts})"
                                )
                                time.sleep(wait_time)
                                continue
                            except ValueError:
                                pass
                        # Exponential backoff
                        backoff_delay = min(
                            self._rate_limit_base_delay * (2**attempt),
                            self._rate_limit_max_delay,
                        )
                        logger.warning(
                            f"[{job_id}] {method} rate limited (429), waiting {backoff_delay}s "
                            f"(attempt {attempt + 1}/{self._max_attempts})"
                        )
                        time.sleep(backoff_delay)
                        continue
                    else:
                        logger.warning(
                            f"[{job_id}] {method} server error ({response.status_code}), "
                            f"attempt {attempt + 1}/{self._max_attempts}"
                        )
                elif response.status_code == 404 and on_404:
                    on_404()
                    return
                else:
                    # Client error - permanent failure
                    try:
                        error_detail = response.json()
                        if isinstance(error_detail, dict):
                            error_msg = error_detail.get("detail", str(error_detail))
                        else:
                            error_msg = str(error_detail)
                    except Exception:
                        error_msg = response.text
                    # 400 errors are expected during normal operation:
                    # - INVALID_PATH: log file doesn't exist yet (job still running)
                    # - NOT_REGULAR: path is symlink or directory
                    # - EMPTY_FILE: file just created, not yet written
                    # These resolve on the next poll cycle, so log at debug.
                    # Other 4xx errors (401, 403, 422) indicate real problems.
                    if response.status_code == 400:
                        logger.debug(
                            f"[{job_id}] {method} failed ({response.status_code}): {error_msg}"
                        )
                    else:
                        logger.warning(
                            f"[{job_id}] {method} failed ({response.status_code}): {error_msg}"
                        )
                    on_client_error(error_msg)
                    return

            except httpx.ConnectError:
                logger.debug(
                    f"[{job_id}] {method} connection failed, attempt {attempt + 1}/{self._max_attempts}"
                )
            except Exception as e:
                logger.warning(f"[{job_id}] {method} error: {e}")
                return
            finally:
                if response is not None:
                    response.close()

            # Wait before retry
            if attempt < self._max_attempts - 1:
                time.sleep(self._retry_delay)

        # All retries exhausted without success
        logger.warning(f"[{job_id}] {method} failed after {self._max_attempts} attempts")
        on_client_error("max retries exceeded")
