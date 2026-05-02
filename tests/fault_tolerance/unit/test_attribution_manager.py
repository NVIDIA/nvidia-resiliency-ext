# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.fault_tolerance.attribution_manager import (
    DEFAULT_ATTRIBUTION_PORT,
    AttributionConfig,
    AttributionManager,
    _attribution_command,
)
from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig


def _args(**overrides):
    defaults = {
        "ft_attribution_endpoint": None,
        "ft_attribution_startup_timeout": 20.0,
        "ft_attribution_llm_api_key_file": None,
        "ft_attribution_llm_base_url": None,
        "ft_attribution_llm_model": None,
        "ft_attribution_analysis_backend": None,
        "ft_attribution_compute_timeout": None,
        "ft_attribution_log_level": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 123

    def poll(self):
        return self.returncode


class _FakeResponse:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _managed_cfg(tmp_path, *, startup_timeout=1.0):
    return AttributionConfig.from_args(
        _args(
            ft_attribution_endpoint="localhost",
            ft_attribution_startup_timeout=startup_timeout,
        ),
        str(tmp_path / "train.log"),
        FaultToleranceConfig(),
    )


def test_attribution_default_is_disabled_and_does_not_start(tmp_path, monkeypatch):
    base_log = tmp_path / "logs" / "train.log"
    cfg = AttributionConfig.from_args(_args(), str(base_log), FaultToleranceConfig())

    assert cfg.endpoint is None
    assert cfg.applog_dir is None
    assert cfg.log_file is None
    assert not cfg.is_enabled
    assert not cfg.is_managed

    def fail_popen(*args, **kwargs):
        raise AssertionError("Popen should not be called when attribution service is disabled")

    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.subprocess.Popen",
        fail_popen,
    )

    endpoint = AttributionManager(cfg, is_store_host=True).start_if_needed()

    assert endpoint is None


def test_managed_attribution_config_derives_applog_dir_and_log_file(tmp_path):
    base_log = tmp_path / "logs" / "train.log"
    cfg = AttributionConfig.from_args(
        _args(ft_attribution_endpoint="localhost"), str(base_log), FaultToleranceConfig()
    )

    assert cfg.endpoint == "localhost"
    assert cfg.client_endpoint.endpoint == f"http://localhost:{DEFAULT_ATTRIBUTION_PORT}"
    assert cfg.applog_dir == str(tmp_path / "logs")
    assert cfg.log_file == str(tmp_path / "logs" / "train_attribution.log")
    assert cfg.is_enabled
    assert cfg.is_managed


@pytest.mark.parametrize(
    "endpoint",
    ["0.0.0.0", "::", "[::]", "http://0.0.0.0:50050", "grpc://[::]:50050"],
)
def test_attribution_endpoint_rejects_bind_all_addresses(tmp_path, endpoint):
    with pytest.raises(ValueError, match="bind-all address"):
        AttributionConfig.from_args(
            _args(ft_attribution_endpoint=endpoint),
            str(tmp_path / "train.log"),
            FaultToleranceConfig(),
        )


def test_attribution_endpoint_localhost_is_managed(tmp_path):
    cfg = AttributionConfig.from_args(
        _args(ft_attribution_endpoint="localhost"),
        str(tmp_path / "train.log"),
        FaultToleranceConfig(),
    )

    assert cfg.endpoint == "localhost"
    assert cfg.is_managed


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://attribution.external:50123",
        "grpc://attribution.external:50123",
        "unix:///tmp/a.sock",
    ],
)
def test_attribution_endpoint_external_strings_are_not_managed(tmp_path, endpoint):
    cfg = AttributionConfig.from_args(
        _args(ft_attribution_endpoint=endpoint),
        str(tmp_path / "train.log"),
        FaultToleranceConfig(),
    )

    assert cfg.endpoint == endpoint
    assert cfg.client_endpoint.endpoint == endpoint
    assert cfg.is_enabled
    assert not cfg.is_managed


def test_attribution_config_maps_launcher_args(tmp_path):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    applog_dir = tmp_path / "allowed"
    applog_dir.mkdir()
    base_log = applog_dir / "train.log"

    cfg = AttributionConfig.from_args(
        _args(
            ft_attribution_endpoint="localhost",
            ft_attribution_llm_api_key_file=str(api_key_file),
            ft_attribution_llm_base_url="https://llm.example/v1",
            ft_attribution_llm_model="model-a",
            ft_attribution_analysis_backend="lib",
            ft_attribution_compute_timeout=12.5,
            ft_attribution_log_level="DEBUG",
        ),
        str(base_log),
        FaultToleranceConfig(),
    )

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(api_key_file))
    assert env["LLM_API_KEY_FILE"] == str(api_key_file)
    assert env["NVRX_ATTRSVC_ENDPOINT"] == f"http://localhost:{DEFAULT_ATTRIBUTION_PORT}"
    assert env["NVRX_ATTRSVC_ALLOWED_ROOT"] == str(applog_dir)
    assert env["NVRX_ATTRSVC_LLM_BASE_URL"] == "https://llm.example/v1"
    assert env["NVRX_ATTRSVC_LLM_MODEL"] == "model-a"
    assert env["NVRX_ATTRSVC_ANALYSIS_BACKEND"] == "lib"
    assert env["NVRX_ATTRSVC_COMPUTE_TIMEOUT"] == "12.5"
    assert env["NVRX_ATTRSVC_LOG_LEVEL"] == "DEBUG"


def test_child_env_overrides_inherited_attrsvc_endpoint(tmp_path, monkeypatch):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    base_log = tmp_path / "train.log"
    monkeypatch.setenv("NVRX_ATTRSVC_ENDPOINT", "/tmp/nvrx-attrsvc.sock")
    cfg = AttributionConfig.from_args(
        _args(ft_attribution_endpoint="localhost"),
        str(base_log),
        FaultToleranceConfig(),
    )

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(api_key_file))

    assert env["NVRX_ATTRSVC_ENDPOINT"] == f"http://localhost:{DEFAULT_ATTRIBUTION_PORT}"


def test_child_env_preserves_existing_pythonpath(tmp_path, monkeypatch):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    base_log = tmp_path / "train.log"
    monkeypatch.setenv("PYTHONPATH", "/existing")
    cfg = AttributionConfig.from_args(
        _args(ft_attribution_endpoint="localhost"),
        str(base_log),
        FaultToleranceConfig(),
    )

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(api_key_file))

    assert env["PYTHONPATH"] == "/existing"


def test_attribution_command_prefers_console_script(monkeypatch):
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.shutil.which",
        lambda _name: "/opt/conda/bin/nvrx-attrsvc",
    )

    assert _attribution_command() == ["/opt/conda/bin/nvrx-attrsvc"]


def test_attribution_command_falls_back_to_service_module(monkeypatch):
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.shutil.which",
        lambda _name: None,
    )

    assert _attribution_command() == [
        sys.executable,
        "-m",
        "nvidia_resiliency_ext.services.attrsvc",
    ]


def test_managed_attribution_requires_api_key_file_before_popen(tmp_path, monkeypatch):
    base_log = tmp_path / "train.log"
    cfg = AttributionConfig.from_args(
        _args(ft_attribution_endpoint="localhost"), str(base_log), FaultToleranceConfig()
    )
    manager = AttributionManager(cfg, is_store_host=True)
    monkeypatch.delenv("LLM_API_KEY_FILE", raising=False)

    def fail_popen(*args, **kwargs):
        raise AssertionError("Popen should not be called without an API key file")

    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.subprocess.Popen",
        fail_popen,
    )

    with pytest.raises(ValueError, match="LLM_API_KEY_FILE"):
        manager.start_if_needed()


def test_external_attribution_does_not_require_key_or_start_process(tmp_path, monkeypatch):
    base_log = tmp_path / "train.log"
    cfg = AttributionConfig.from_args(
        _args(ft_attribution_endpoint="http://attribution.external:50123"),
        str(base_log),
        FaultToleranceConfig(),
    )
    manager = AttributionManager(cfg, is_store_host=True)
    monkeypatch.delenv("LLM_API_KEY_FILE", raising=False)

    def fail_popen(*args, **kwargs):
        raise AssertionError("Popen should not be called for external attribution")

    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.subprocess.Popen",
        fail_popen,
    )

    endpoint = manager.start_if_needed()

    assert endpoint is not None
    assert endpoint.endpoint == "http://attribution.external:50123"


def test_wait_until_ready_accepts_2xx_health_response(tmp_path, monkeypatch):
    manager = AttributionManager(_managed_cfg(tmp_path), is_store_host=True)
    manager.process = _FakeProcess()

    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(200),
    )

    manager._wait_until_ready()


def test_wait_until_ready_rejects_4xx_health_response(tmp_path, monkeypatch):
    manager = AttributionManager(_managed_cfg(tmp_path), is_store_host=True)
    manager.process = _FakeProcess()
    monotonic_values = iter([0.0, 0.0, 2.0])

    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(404),
    )
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "nvidia_resiliency_ext.fault_tolerance.attribution_manager.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(TimeoutError, match="HTTP status 404"):
        manager._wait_until_ready()


def test_wait_until_ready_raises_when_process_exits_early(tmp_path):
    manager = AttributionManager(_managed_cfg(tmp_path), is_store_host=True)
    manager.process = _FakeProcess(returncode=1)

    with pytest.raises(RuntimeError, match="returncode=1"):
        manager._wait_until_ready()
