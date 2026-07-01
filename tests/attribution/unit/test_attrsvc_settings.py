import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

attrsvc_config = pytest.importorskip("nvidia_resiliency_ext.services.attrsvc.config")
Settings = attrsvc_config.Settings


def test_attrsvc_llm_settings_are_override_only(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_LLM_MODEL", raising=False)
    monkeypatch.delenv("NVRX_ATTRSVC_LLM_BASE_URL", raising=False)

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LLM_MODEL is None
    assert cfg.LLM_BASE_URL is None


def test_attrsvc_llm_settings_accept_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_LLM_MODEL", "override-model")
    monkeypatch.setenv("NVRX_ATTRSVC_LLM_BASE_URL", "https://llm.example.test/v1")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LLM_MODEL == "override-model"
    assert cfg.LLM_BASE_URL == "https://llm.example.test/v1"


def test_attrsvc_llm_empty_env_values_are_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_LLM_MODEL", "")
    monkeypatch.setenv("NVRX_ATTRSVC_LLM_BASE_URL", " ")
    monkeypatch.setenv("NVRX_ATTRSVC_LLM_TEMPERATURE", "")
    monkeypatch.setenv("NVRX_ATTRSVC_LLM_TOP_P", "")
    monkeypatch.setenv("NVRX_ATTRSVC_LLM_MAX_TOKENS", "")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LLM_MODEL is None
    assert cfg.LLM_BASE_URL is None
    assert cfg.LLM_TEMPERATURE is None
    assert cfg.LLM_TOP_P is None
    assert cfg.LLM_MAX_TOKENS is None


def test_attrsvc_log_analysis_endpoint_outer_retry_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES", raising=False)
    monkeypatch.delenv("NVRX_ATTRSVC_LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC", raising=False)
    monkeypatch.delenv("NVRX_LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES", raising=False)
    monkeypatch.delenv("NVRX_LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC", raising=False)

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES is None
    assert cfg.LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC is None


def test_attrsvc_log_analysis_endpoint_outer_retry_accepts_attrsvc_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES", "2")
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC", "3.5")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES == 2
    assert cfg.LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC == 3.5


def test_attrsvc_log_analysis_endpoint_outer_retry_accepts_nvrx_env(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES", raising=False)
    monkeypatch.delenv("NVRX_ATTRSVC_LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC", raising=False)
    monkeypatch.setenv("NVRX_LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES", "1")
    monkeypatch.setenv("NVRX_LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC", "0")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES == 1
    assert cfg.LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC == 0.0


def test_attrsvc_log_analysis_endpoint_outer_retry_rejects_negative(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES", "-1")

    with pytest.raises(Exception):
        Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)


def test_attrsvc_settings_pass_endpoint_outer_retry_to_controller_config(tmp_path):
    attrsvc_service = pytest.importorskip("nvidia_resiliency_ext.services.attrsvc.service")
    cfg = Settings(
        ALLOWED_ROOT=str(tmp_path),
        LOG_ANALYSIS_ENDPOINT_OUTER_RETRIES=4,
        LOG_ANALYSIS_ENDPOINT_OUTER_BACKOFF_SEC=2.5,
        _env_file=None,
    )

    controller_cfg = attrsvc_service._controller_config_from_settings(cfg)

    assert controller_cfg.analysis.endpoint_outer_retries == 4
    assert controller_cfg.analysis.endpoint_outer_backoff_sec == 2.5


def test_attrsvc_analysis_backend_uses_current_env_name_only(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_ANALYSIS_BACKEND", "lib")
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_ANALYSIS_BACKEND", "mcp")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.ANALYSIS_BACKEND == "lib"


def test_attrsvc_progressive_analysis_defaults_to_all_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS", raising=False)

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.PROGRESSIVE_ANALYSIS == "all_explicit"


def test_attrsvc_progressive_analysis_accepts_off(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS", "OFF")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.PROGRESSIVE_ANALYSIS == "off"


def test_attrsvc_progressive_analysis_accepts_all_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS", "ALL_EXPLICIT")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.PROGRESSIVE_ANALYSIS == "all_explicit"


def test_attrsvc_log_analysis_backend_env_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_ANALYSIS_BACKEND", raising=False)
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_ANALYSIS_BACKEND", "lib")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.ANALYSIS_BACKEND == "mcp"


def test_attrsvc_log_level_uses_current_env_name_only(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_LEVEL", "debug")
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_LEVEL_NAME", "WARNING")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LOG_LEVEL == "DEBUG"


def test_attrsvc_log_level_name_env_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_LOG_LEVEL", raising=False)
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_LEVEL_NAME", "DEBUG")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LOG_LEVEL == "INFO"
