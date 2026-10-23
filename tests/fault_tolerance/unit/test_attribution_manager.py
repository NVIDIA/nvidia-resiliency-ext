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


def _config(tmp_path, base_log="train.log", **flags):
    """The attribution config for these --ft-* values, built as the launcher builds it."""
    args = SimpleNamespace(**flags)
    return AttributionConfig.from_args(
        args, str(tmp_path / base_log), FaultToleranceConfig.from_args(args)
    )


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
    return _config(
        tmp_path,
        ft_attribution_endpoint="localhost",
        ft_attribution_startup_timeout=startup_timeout,
    )


def test_attribution_default_is_disabled_and_does_not_start(tmp_path, monkeypatch):
    cfg = _config(tmp_path, base_log="logs/train.log")

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
    cfg = _config(tmp_path, base_log="logs/train.log", ft_attribution_endpoint="localhost")

    assert cfg.endpoint == "localhost"
    assert cfg.client_endpoint.endpoint == f"http://127.0.0.1:{DEFAULT_ATTRIBUTION_PORT}"
    assert cfg.applog_dir == str(tmp_path / "logs")
    assert cfg.log_file == str(tmp_path / "logs" / "nvrx" / "train_attribution.log")
    assert cfg.is_enabled
    assert cfg.is_managed


def test_managed_attribution_log_uses_nvrx_log_stem_and_directory(tmp_path):
    nvrx_log = tmp_path / "nvrx" / "job" / "logs" / "launcher.log"
    cfg = _config(
        tmp_path,
        base_log="application/train.log",
        ft_attribution_endpoint="localhost",
        ft_nvrx_logfile=str(nvrx_log),
    )

    assert cfg.applog_dir == str(tmp_path / "application")
    assert cfg.log_file == str(nvrx_log.parent / "launcher_attribution.log")


@pytest.mark.parametrize(
    "endpoint,match",
    [
        ("0.0.0.0", "bind-all address"),
        ("::", "bind-all address"),
        ("[::]", "bind-all address"),
        ("http://0.0.0.0:50050", "bind-all address"),
        ("grpc://[::]:50050", "bind-all address"),
        ("locahlost", "--ft-attribution-endpoint"),
        ("localhost:50050", "--ft-attribution-endpoint"),
        ("http://:50050", "--ft-attribution-endpoint"),
        ("http://localhost:not-a-port", "--ft-attribution-endpoint"),
        ("grpc://attribution.external", "--ft-attribution-endpoint"),
        ("ftp://attribution.external:50050", "--ft-attribution-endpoint"),
        ("unix://", "--ft-attribution-endpoint"),
    ],
)
def test_attribution_endpoint_rejects_invalid_values(tmp_path, endpoint, match):
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, ft_attribution_endpoint=endpoint)


def test_attribution_endpoint_localhost_is_managed(tmp_path):
    cfg = _config(tmp_path, ft_attribution_endpoint="localhost")

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
    cfg = _config(tmp_path, ft_attribution_endpoint=endpoint)

    assert cfg.endpoint == endpoint
    assert cfg.client_endpoint.endpoint == endpoint
    assert cfg.is_enabled
    assert not cfg.is_managed


def test_attribution_config_maps_launcher_args(tmp_path):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    applog_dir = tmp_path / "allowed"
    applog_dir.mkdir()

    cfg = _config(
        tmp_path,
        base_log="allowed/train.log",
        ft_attribution_endpoint="localhost",
        ft_attribution_llm_api_key_file=str(api_key_file),
        ft_attribution_llm_base_url="https://llm.example/v1",
        ft_attribution_llm_model="model-a",
        ft_attribution_analysis_backend="lib",
        ft_attribution_log_level="DEBUG",
        ft_attribution_export_url=("https://dataflow.example.test/dataflow2/example-index/posting"),
    )

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(api_key_file))
    assert env["LLM_API_KEY_FILE"] == str(api_key_file)
    assert env["NVRX_ATTRSVC_ENDPOINT"] == f"http://127.0.0.1:{DEFAULT_ATTRIBUTION_PORT}"
    assert env["NVRX_ATTRSVC_ALLOWED_ROOT"] == str(applog_dir)
    assert env["NVRX_ATTRSVC_LLM_BASE_URL"] == "https://llm.example/v1"
    assert env["NVRX_ATTRSVC_LLM_MODEL"] == "model-a"
    assert env["NVRX_ATTRSVC_ANALYSIS_BACKEND"] == "lib"
    assert env["NVRX_ATTRSVC_LOG_LEVEL"] == "DEBUG"
    assert (
        env["NVRX_ATTRSVC_EXPORT_URL"]
        == "https://dataflow.example.test/dataflow2/example-index/posting"
    )


def test_attribution_config_accepts_lib_analysis_backend(tmp_path):
    cfg = _config(
        tmp_path, ft_attribution_endpoint="localhost", ft_attribution_analysis_backend="lib"
    )

    assert cfg.analysis_backend == "lib"


def test_attribution_config_rejects_mcp_analysis_backend(tmp_path):
    with pytest.raises(ValueError, match="only supports 'lib'"):
        _config(
            tmp_path, ft_attribution_endpoint="localhost", ft_attribution_analysis_backend="mcp"
        )


def test_managed_attrsvc_inherits_log_convergence_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_RESTART_AGENT_LOG_QUIET_SECONDS", "15")
    monkeypatch.setenv("NVRX_ATTRSVC_RESTART_AGENT_LOG_MAX_WAIT_SECONDS", "60")
    cfg = _managed_cfg(tmp_path)

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(tmp_path / "key"))

    assert env["NVRX_ATTRSVC_RESTART_AGENT_LOG_QUIET_SECONDS"] == "15"
    assert env["NVRX_ATTRSVC_RESTART_AGENT_LOG_MAX_WAIT_SECONDS"] == "60"


def test_attribution_stop_action_defaults_to_log_only(tmp_path):
    """Enabling attribution must not implicitly enable acting on its verdicts."""
    cfg = _config(tmp_path, ft_attribution_endpoint="localhost")

    assert cfg.stop_action == "log"
    assert cfg.enforce_stop is False
    assert cfg.client_endpoint.enforce_stop is False


def test_attribution_stop_action_no_restart_enables_enforcement(tmp_path):
    cfg = _config(
        tmp_path, ft_attribution_endpoint="localhost", ft_attribution_stop_action="no-restart"
    )

    assert cfg.enforce_stop is True
    assert cfg.client_endpoint.enforce_stop is True


@pytest.mark.parametrize("value", ["restart", "stop", ""])
def test_attribution_stop_action_rejects_unknown_values(tmp_path, value):
    with pytest.raises(ValueError, match="--ft-attribution-stop-action"):
        _config(tmp_path, ft_attribution_endpoint="localhost", ft_attribution_stop_action=value)


def test_attribution_child_env_does_not_inherit_export_url_from_launcher_env(tmp_path, monkeypatch):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    monkeypatch.setenv(
        "NVRX_ATTRSVC_EXPORT_URL",
        "https://dataflow.example.test/dataflow2/launcher-env/posting",
    )
    cfg = _config(tmp_path, ft_attribution_endpoint="localhost")

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(api_key_file))

    assert cfg.export_url is None
    assert "NVRX_ATTRSVC_EXPORT_URL" not in env


def test_cli_export_url_overrides_launcher_env(tmp_path, monkeypatch):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    monkeypatch.setenv(
        "NVRX_ATTRSVC_EXPORT_URL",
        "https://dataflow.example.test/old/posting",
    )
    cfg = _config(
        tmp_path,
        ft_attribution_endpoint="localhost",
        ft_attribution_export_url="https://dataflow.example.test/new/posting",
    )

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(api_key_file))

    assert cfg.export_url == "https://dataflow.example.test/new/posting"
    assert env["NVRX_ATTRSVC_EXPORT_URL"] == "https://dataflow.example.test/new/posting"


@pytest.mark.parametrize(
    "url",
    [
        "dataflow.example.test/posting",
        "ftp://dataflow.example.test/posting",
        "https://dataflow.example.test/posting?queue=fast",
    ],
)
def test_export_url_rejects_invalid_values(tmp_path, url):
    with pytest.raises(ValueError, match="--ft-attribution-export-url"):
        _config(tmp_path, ft_attribution_endpoint="localhost", ft_attribution_export_url=url)


def test_export_url_rejects_external_attribution_endpoint(tmp_path):
    with pytest.raises(ValueError, match="launcher-managed attribution"):
        _config(
            tmp_path,
            ft_attribution_endpoint="http://attribution.external:50123",
            ft_attribution_export_url="https://dataflow.example.test/posting",
        )


def test_export_url_rejects_disabled_attribution(tmp_path):
    with pytest.raises(ValueError, match="--ft-attribution-endpoint localhost"):
        _config(tmp_path, ft_attribution_export_url="https://dataflow.example.test/posting")


def test_child_env_overrides_inherited_attrsvc_endpoint(tmp_path, monkeypatch):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    monkeypatch.setenv("NVRX_ATTRSVC_ENDPOINT", "/tmp/nvrx-attrsvc.sock")
    cfg = _config(tmp_path, ft_attribution_endpoint="localhost")

    env = AttributionManager(cfg, is_store_host=True)._child_env(str(api_key_file))

    assert env["NVRX_ATTRSVC_ENDPOINT"] == f"http://127.0.0.1:{DEFAULT_ATTRIBUTION_PORT}"


def test_child_env_preserves_existing_pythonpath(tmp_path, monkeypatch):
    api_key_file = tmp_path / "key"
    api_key_file.write_text("secret")
    monkeypatch.setenv("PYTHONPATH", "/existing")
    cfg = _config(tmp_path, ft_attribution_endpoint="localhost")

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
    cfg = _config(tmp_path, ft_attribution_endpoint="localhost")
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
    cfg = _config(tmp_path, ft_attribution_endpoint="http://attribution.external:50123")
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


def test_external_attribution_ignores_launcher_export_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_EXPORT_URL", "not-a-url")

    cfg = _config(tmp_path, ft_attribution_endpoint="http://attribution.external:50123")

    assert cfg.export_url is None


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
