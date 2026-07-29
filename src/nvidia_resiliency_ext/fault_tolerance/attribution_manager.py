# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle management for launcher-managed attribution service."""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
import shutil
import subprocess  # nosec B404
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

DEFAULT_ATTRIBUTION_PORT = 50050
DEFAULT_ATTRIBUTION_STARTUP_TIMEOUT = 20.0
_ATTRIBUTION_STOP_TIMEOUT = 5.0
_ATTRIBUTION_READY_POLL_INTERVAL = 0.5
_MANAGED_ATTRIBUTION_ENDPOINT = "localhost"
_EXTERNAL_ATTRIBUTION_SCHEMES = {"http", "grpc"}
_ATTRSVC_EXPORT_URL_ENV = "NVRX_ATTRSVC_EXPORT_URL"
_ATTRIBUTION_EXPORT_URL_CONFIG = "--ft-attribution-export-url / attribution_export_url"
_EXPORT_URL_SCHEMES = {"http", "https"}


@dataclass(frozen=True)
class AttributionEndpoint:
    """Endpoint used by the in-launcher attribution client."""

    endpoint: str
    decision_timeout: Optional[float]


@dataclass(frozen=True)
class AttributionConfig:
    """Resolved launcher-managed attribution service configuration."""

    endpoint: Optional[str]
    applog_dir: Optional[str]
    log_file: Optional[str]
    startup_timeout: float
    llm_api_key_file: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    analysis_backend: Optional[str] = None
    decision_timeout: Optional[float] = None
    log_level: Optional[str] = None
    export_url: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "analysis_backend",
            _normalize_analysis_backend(self.analysis_backend),
        )

    @property
    def is_enabled(self) -> bool:
        return self.endpoint is not None

    @property
    def is_managed(self) -> bool:
        if not self.is_enabled:
            return False
        assert self.endpoint is not None
        return is_managed_attribution_endpoint(self.endpoint)

    @property
    def client_endpoint(self) -> AttributionEndpoint:
        if not self.is_enabled:
            raise ValueError("attribution endpoint requested while attribution service is disabled")
        assert self.endpoint is not None
        if self.is_managed:
            return AttributionEndpoint(
                endpoint=_managed_attribution_client_endpoint(),
                decision_timeout=self.decision_timeout,
            )
        return AttributionEndpoint(
            endpoint=self.endpoint,
            decision_timeout=self.decision_timeout,
        )

    @classmethod
    def from_args(
        cls,
        args: Any,
        base_log_file: str,
        ft_cfg: FaultToleranceConfig,
    ) -> "AttributionConfig":
        endpoint = getattr(args, "ft_attribution_endpoint", None)
        if endpoint is None:
            endpoint = ft_cfg.attribution_endpoint
        if endpoint is not None:
            endpoint = str(endpoint).strip() or None
        if endpoint is not None:
            _validate_attribution_endpoint(endpoint)

        export_url = _resolve_export_url(args, ft_cfg)
        if export_url is not None and (
            endpoint is None or not is_managed_attribution_endpoint(endpoint)
        ):
            endpoint_desc = endpoint if endpoint is not None else "disabled"
            raise ValueError(
                f"{_ATTRIBUTION_EXPORT_URL_CONFIG} requires launcher-managed attribution "
                f"with --ft-attribution-endpoint localhost, got {endpoint_desc!r}"
            )

        startup_timeout = getattr(args, "ft_attribution_startup_timeout", None)
        if startup_timeout is None:
            startup_timeout = DEFAULT_ATTRIBUTION_STARTUP_TIMEOUT
        startup_timeout = float(startup_timeout)
        if startup_timeout <= 0:
            raise ValueError("--ft-attribution-startup-timeout must be positive")

        decision_timeout = _resolve_decision_timeout(args, ft_cfg)

        if endpoint is None:
            return cls(
                endpoint=None,
                applog_dir=None,
                log_file=None,
                startup_timeout=startup_timeout,
                llm_api_key_file=getattr(args, "ft_attribution_llm_api_key_file", None),
                llm_base_url=getattr(args, "ft_attribution_llm_base_url", None),
                llm_model=getattr(args, "ft_attribution_llm_model", None),
                analysis_backend=getattr(args, "ft_attribution_analysis_backend", None),
                decision_timeout=decision_timeout,
                log_level=getattr(args, "ft_attribution_log_level", None),
                export_url=None,
            )

        applog_dir = os.path.dirname(os.path.realpath(os.path.abspath(base_log_file)))
        os.makedirs(applog_dir, exist_ok=True)
        _validate_existing_dir(applog_dir, "derived attribution applog_dir")
        base_real = os.path.realpath(os.path.abspath(base_log_file))
        if os.path.commonpath([applog_dir, base_real]) != applog_dir:
            raise ValueError(
                f"derived attribution applog_dir must contain --ft-per-cycle-applog-prefix: "
                f"{applog_dir!r} does not contain {base_real!r}"
            )

        log_file = _attribution_log_path(base_log_file)
        log_file = os.path.realpath(os.path.abspath(os.path.expanduser(log_file)))
        log_parent = os.path.dirname(log_file) or "."
        os.makedirs(log_parent, exist_ok=True)

        return cls(
            endpoint=endpoint,
            applog_dir=applog_dir,
            log_file=log_file,
            startup_timeout=startup_timeout,
            llm_api_key_file=getattr(args, "ft_attribution_llm_api_key_file", None),
            llm_base_url=getattr(args, "ft_attribution_llm_base_url", None),
            llm_model=getattr(args, "ft_attribution_llm_model", None),
            analysis_backend=getattr(args, "ft_attribution_analysis_backend", None),
            decision_timeout=decision_timeout,
            log_level=getattr(args, "ft_attribution_log_level", None),
            export_url=export_url,
        )


class AttributionManager:
    """Start and stop a job-local attribution service process when this launcher owns it."""

    def __init__(self, cfg: AttributionConfig, *, is_store_host: bool):
        self.cfg = cfg
        self.is_store_host = is_store_host
        self.process: Optional[subprocess.Popen] = None

    def start_if_needed(self) -> Optional[AttributionEndpoint]:
        if not self.cfg.is_enabled:
            return None

        if not self.cfg.is_managed:
            logger.debug(
                "Using externally managed attribution service at %s",
                self.cfg.endpoint,
            )
            return self.cfg.client_endpoint

        if not self.is_store_host:
            return None

        assert self.cfg.applog_dir is not None
        assert self.cfg.log_file is not None
        api_key_file = self._resolve_api_key_file()
        env = self._child_env(api_key_file)
        cmd = _attribution_command()

        logger.info(
            "Starting managed attribution service on localhost:%s "
            "(applog_dir=%s, log_file=%s, startup_timeout=%.1fs, decision_timeout=%s)",
            DEFAULT_ATTRIBUTION_PORT,
            self.cfg.applog_dir,
            self.cfg.log_file,
            self.cfg.startup_timeout,
            (
                f"{self.cfg.decision_timeout:.1f}s"
                if self.cfg.decision_timeout is not None
                else "default"
            ),
        )

        log_fd = open(self.cfg.log_file, "w")
        try:
            self.process = subprocess.Popen(  # nosec B603
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                env=env,
                shell=False,
            )
        finally:
            log_fd.close()

        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

        logger.info(
            "Managed attribution service is ready: PID=%s endpoint=http://127.0.0.1:%s",
            self.process.pid if self.process else None,
            DEFAULT_ATTRIBUTION_PORT,
        )
        return self.cfg.client_endpoint

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            return
        if proc.poll() is not None:
            logger.info(
                "Managed attribution service PID=%s already exited with returncode=%s",
                proc.pid,
                proc.returncode,
            )
            self.process = None
            return

        logger.info("Sending SIGTERM to managed attribution service PID=%s", proc.pid)
        with contextlib.suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=_ATTRIBUTION_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Managed attribution service PID=%s did not exit within %.0fs; killing",
                proc.pid,
                _ATTRIBUTION_STOP_TIMEOUT,
            )
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait()
        logger.info(
            "Managed attribution service PID=%s finished with returncode=%s",
            proc.pid,
            proc.returncode,
        )
        self.process = None

    def _resolve_api_key_file(self) -> str:
        raw = self.cfg.llm_api_key_file or os.getenv("LLM_API_KEY_FILE")
        if not raw:
            raise ValueError(
                "managed attribution service requires "
                "--ft-attribution-llm-api-key-file or LLM_API_KEY_FILE"
            )
        path = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
        if not os.path.isfile(path):
            raise ValueError(f"managed attribution service LLM API key file is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise ValueError(
                f"managed attribution service LLM API key file is not readable: {path}"
            )
        return path

    def _child_env(self, api_key_file: str) -> dict[str, str]:
        assert self.cfg.applog_dir is not None
        env = os.environ.copy()
        # Export URL for launcher-managed attrsvc is explicit via CLI/YAML, not parent env.
        env.pop(_ATTRSVC_EXPORT_URL_ENV, None)
        env["LLM_API_KEY_FILE"] = api_key_file
        # nvrx-attrsvc owns the service-side environment variable contract.
        env["NVRX_ATTRSVC_ENDPOINT"] = _managed_attribution_client_endpoint()
        env["NVRX_ATTRSVC_ALLOWED_ROOT"] = self.cfg.applog_dir
        _set_if_not_none(env, "NVRX_ATTRSVC_LLM_BASE_URL", self.cfg.llm_base_url)
        _set_if_not_none(env, "NVRX_ATTRSVC_LLM_MODEL", self.cfg.llm_model)
        _set_if_not_none(env, "NVRX_ATTRSVC_ANALYSIS_BACKEND", self.cfg.analysis_backend)
        _set_if_not_none(env, "NVRX_ATTRSVC_LOG_LEVEL", self.cfg.log_level)
        _set_if_not_none(env, _ATTRSVC_EXPORT_URL_ENV, self.cfg.export_url)
        return env

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.cfg.startup_timeout
        url = f"http://127.0.0.1:{DEFAULT_ATTRIBUTION_PORT}/healthz"
        last_error = "not probed"

        while time.monotonic() < deadline:
            rc = self.process.poll()
            if rc is not None:
                raise RuntimeError(
                    f"managed attribution service exited before becoming ready "
                    f"(returncode={rc}, log_file={self.cfg.log_file})"
                )
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:  # nosec B310
                    if 200 <= resp.status < 300:
                        return
                    last_error = f"HTTP status {resp.status}"
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP status {exc.code}: {exc.reason}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(_ATTRIBUTION_READY_POLL_INTERVAL)

        raise TimeoutError(
            f"managed attribution service did not become ready within "
            f"{self.cfg.startup_timeout:.1f}s "
            f"at {url} (last_error={last_error}, log_file={self.cfg.log_file})"
        )


def is_managed_attribution_endpoint(endpoint: str) -> bool:
    return endpoint.strip().lower() == _MANAGED_ATTRIBUTION_ENDPOINT


def _managed_attribution_client_endpoint() -> str:
    return f"http://127.0.0.1:{DEFAULT_ATTRIBUTION_PORT}"


def _attribution_command() -> list[str]:
    exe = shutil.which("nvrx-attrsvc")
    if exe:
        return [exe]
    return [sys.executable, "-m", "nvidia_resiliency_ext.services.attrsvc"]


def _attribution_log_path(base_log_file: str) -> str:
    if base_log_file.endswith(".log"):
        return base_log_file[:-4] + "_attribution.log"
    return base_log_file + "_attribution.log"


def _validate_attribution_endpoint(endpoint: str) -> None:
    endpoint = endpoint.strip()
    if is_managed_attribution_endpoint(endpoint):
        return

    parsed = urlparse(endpoint)
    if not parsed.scheme:
        _validate_attribution_endpoint_host(_unbracket_host(endpoint))
        raise ValueError(
            "--ft-attribution-endpoint must be exactly localhost for launcher-managed "
            "attribution, or an explicit endpoint such as http://host:port, "
            "grpc://host:port, or unix:///path/to/socket."
        )

    if parsed.scheme == "unix":
        if not parsed.path:
            raise ValueError("--ft-attribution-endpoint unix endpoint must include a socket path")
        return

    if parsed.scheme not in _EXTERNAL_ATTRIBUTION_SCHEMES:
        raise ValueError(
            f"--ft-attribution-endpoint scheme must be one of "
            f"{sorted(_EXTERNAL_ATTRIBUTION_SCHEMES | {'unix'})}, got {parsed.scheme!r}"
        )

    if not parsed.hostname:
        raise ValueError("--ft-attribution-endpoint must include a host")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("--ft-attribution-endpoint must include a valid port") from exc

    if port is None:
        raise ValueError("--ft-attribution-endpoint must include a port")

    _validate_attribution_endpoint_host(parsed.hostname)


def _resolve_export_url(args: Any, ft_cfg: FaultToleranceConfig) -> Optional[str]:
    value = getattr(args, "ft_attribution_export_url", None)
    if value is None:
        value = getattr(ft_cfg, "attribution_export_url", None)
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None
    _validate_export_url(value)
    return value


def _resolve_decision_timeout(args: Any, ft_cfg: FaultToleranceConfig) -> Optional[float]:
    value = getattr(args, "ft_attribution_decision_timeout", None)
    if value is None:
        value = getattr(ft_cfg, "attribution_decision_timeout", None)
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        raise ValueError("--ft-attribution-decision-timeout must be positive")
    return value


def _normalize_analysis_backend(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized != "mcp":
        raise ValueError(
            "--ft-attribution-analysis-backend supports only 'mcp'; "
            f"'lib' is no longer supported by launcher-managed attrsvc, got {value!r}"
        )
    return normalized


def _validate_export_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _EXPORT_URL_SCHEMES or not parsed.netloc:
        raise ValueError(
            f"{_ATTRIBUTION_EXPORT_URL_CONFIG} must be a complete http(s) URL, got {url!r}"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"{_ATTRIBUTION_EXPORT_URL_CONFIG} must not include query parameters or fragments"
        )


def _validate_attribution_endpoint_host(hostname: str) -> None:
    endpoint_ip = _parse_ip_literal(hostname)
    if endpoint_ip is not None and endpoint_ip.is_unspecified:
        raise ValueError(
            "--ft-attribution-endpoint must not be a bind-all address. "
            "Use localhost for launcher-managed attribution, or use a routable "
            "endpoint for an externally managed attribution service."
        )


def _parse_ip_literal(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    # DNS names are valid external endpoint hosts. They are resolved by the client at connect time,
    # so a parsing failure here means "not an IP literal" rather than "invalid endpoint".
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        return None


def _unbracket_host(hostname: str) -> str:
    if hostname.startswith("[") and hostname.endswith("]"):
        return hostname[1:-1]
    return hostname


def _validate_existing_dir(path: str, name: str) -> None:
    if not os.path.isdir(path):
        raise ValueError(f"{name} must be an existing directory, got {path}")
    if not os.access(path, os.R_OK | os.X_OK):
        raise ValueError(f"{name} must be readable/searchable, got {path}")


def _set_if_not_none(env: dict[str, str], key: str, value: Any) -> None:
    if value is not None:
        env[key] = str(value)
