# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attribution service client for nvidia_resiliency_ext.services.smonsvc."""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from nvidia_resiliency_ext.attribution.orchestration.http_api import (
    ROUTE_HEALTHZ,
    get_log_response,
    post_log,
)

logger = logging.getLogger(__name__)

_EXPECTED_CLIENT_ERROR_CODES = {
    "errorcode.not_readable",
    "not_readable",
    "errorcode.logs_dir_not_readable",
    "logs_dir_not_readable",
}


@dataclass(frozen=True)
class _AttrsvcEndpoint:
    base_url: str
    display_url: str
    uds_path: str | None = None


def _parse_attrsvc_endpoint(endpoint: str) -> _AttrsvcEndpoint:
    value = endpoint.strip()
    if not value:
        raise ValueError("attrsvc endpoint must not be empty")

    if value.startswith("unix://"):
        parsed = urlparse(value)
        if parsed.netloc and parsed.path:
            path = f"/{parsed.netloc}{parsed.path}"
        else:
            path = parsed.path or parsed.netloc
        path = unquote(path)
        if not os.path.isabs(path):
            raise ValueError(
                f"attrsvc unix endpoint must contain an absolute socket path: {endpoint}"
            )
        return _AttrsvcEndpoint(
            base_url="http://nvrx-attrsvc",
            display_url=f"unix://{path}",
            uds_path=path,
        )

    if value.startswith("/"):
        path = os.path.abspath(value)
        return _AttrsvcEndpoint(
            base_url="http://nvrx-attrsvc",
            display_url=f"unix://{path}",
            uds_path=path,
        )

    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    if parsed.scheme != "http":
        raise ValueError(
            "attrsvc endpoint must use http://, unix://, or an absolute "
            f"socket path; got {endpoint}"
        )
    return _AttrsvcEndpoint(base_url=value.rstrip("/"), display_url=value.rstrip("/"))


def _is_expected_client_error(
    status_code: int,
    error_detail: object,
    *,
    permission_denied_is_expected: bool = False,
) -> bool:
    if status_code == 400:
        return True
    if (
        not permission_denied_is_expected
        or status_code != 403
        or not isinstance(error_detail, dict)
    ):
        return False
    return _client_error_code(error_detail) in _EXPECTED_CLIENT_ERROR_CODES


def _client_error_code(error_detail: object) -> str:
    if not isinstance(error_detail, dict):
        return ""
    return str(error_detail.get("error_code", "")).lower()


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
        permission_denied_is_expected: bool = False,
    ):
        """
        Initialize the attribution service client.

        Args:
            base_url: Base URL of the attribution service (e.g., http://localhost:8000)
                or HTTP-over-UDS endpoint (e.g., unix:///tmp/nvrx-attrsvc.sock)
            timeout: HTTP request timeout in seconds
            max_attempts: Maximum number of retry attempts
            retry_delay: Delay between retries for server errors
            rate_limit_base_delay: Base delay for 429 rate limiting (exponential backoff)
            rate_limit_max_delay: Maximum delay for rate limiting
            request_throttle: Delay between successful requests to prevent rate limiting
            on_rate_limited: Optional callback when rate limited (for stats tracking)
            permission_denied_is_expected: Treat attrsvc not_readable/logs_dir_not_readable 403s
                as expected. This is useful for standalone all-users smonsvc deployments where
                attrsvc may not be able to read every user's logs. Keep False for co-deployed
                same-user paths where ft_launcher creates the log and attrsvc should have access.
        """
        if httpx is None:
            raise ImportError("httpx is required. Install with: pip install httpx")

        self._endpoint = _parse_attrsvc_endpoint(base_url)
        self._base_url = self._endpoint.base_url
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._rate_limit_base_delay = rate_limit_base_delay
        self._rate_limit_max_delay = rate_limit_max_delay
        self._request_throttle = request_throttle
        self._on_rate_limited = on_rate_limited
        self._permission_denied_is_expected = permission_denied_is_expected
        if self._endpoint.uds_path:
            transport = httpx.HTTPTransport(uds=self._endpoint.uds_path)
            self._client = httpx.Client(
                transport=transport,
                base_url=self._endpoint.base_url,
                timeout=timeout,
            )
        else:
            self._client = httpx.Client(base_url=self._endpoint.base_url, timeout=timeout)

        # Health check cache to avoid blocking HTTP calls on every /healthz request
        self._health_cache: tuple[bool, float] | None = None
        self._health_cache_ttl = 60.0  # seconds
        self._expected_client_error_counts: dict[tuple[str, int, str], int] = {}

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
                response = self._client.get(ROUTE_HEALTHZ, timeout=5.0)
                is_healthy = response.status_code < 500
            except Exception:
                pass
            finally:
                if response is not None:
                    response.close()
        self._health_cache = (is_healthy, now)
        return is_healthy

    def _log_expected_client_error(
        self,
        method: str,
        job_id: str,
        status_code: int,
        error_detail: object,
        error_msg: object,
    ) -> None:
        error_code = _client_error_code(error_detail)
        key = (method, status_code, error_code)
        count = self._expected_client_error_counts.get(key, 0) + 1
        self._expected_client_error_counts[key] = count
        if count <= 5 or count % 100 == 0:
            logger.debug(
                f"[{job_id}] {method} failed ({status_code}): {error_msg} "
                f"(expected client error count={count})"
            )

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
        for attempt in range(self._max_attempts):
            response = None
            try:
                if method == "POST":
                    response = post_log(
                        self._client,
                        log_path,
                        user=user,
                        job_id=job_id,
                    )
                else:  # GET
                    response = get_log_response(
                        self._client,
                        log_path,
                        wl_restart=cycle,
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
                    error_detail: object = {}
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
                    # Permission-denied 403s are expected only for standalone
                    # all-users monitor deployments; co-deployed same-user
                    # attrsvc paths should be able to read ft_launcher logs.
                    if _is_expected_client_error(
                        response.status_code,
                        error_detail,
                        permission_denied_is_expected=self._permission_denied_is_expected,
                    ):
                        self._log_expected_client_error(
                            method,
                            job_id,
                            response.status_code,
                            error_detail,
                            error_msg,
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
