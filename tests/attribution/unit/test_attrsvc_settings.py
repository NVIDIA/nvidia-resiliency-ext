import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

attrsvc_config = pytest.importorskip("nvidia_resiliency_ext.services.attrsvc.config")
Settings = attrsvc_config.Settings


def test_attrsvc_llm_settings_are_override_only(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_LLM_MODEL", raising=False)
    monkeypatch.delenv("NVRX_ATTRSVC_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("NVRX_ATTRSVC_PROGRESSIVE_CHUNKS_PER_TIME", raising=False)

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LLM_MODEL is None
    assert cfg.LLM_BASE_URL is None
    assert cfg.PROGRESSIVE_CHUNKS_PER_TIME is None


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
    monkeypatch.setenv("NVRX_ATTRSVC_PROGRESSIVE_CHUNKS_PER_TIME", "")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.LLM_MODEL is None
    assert cfg.LLM_BASE_URL is None
    assert cfg.LLM_TEMPERATURE is None
    assert cfg.LLM_TOP_P is None
    assert cfg.LLM_MAX_TOKENS is None
    assert cfg.PROGRESSIVE_CHUNKS_PER_TIME is None


def test_attrsvc_progressive_chunks_per_time_accepts_float_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_PROGRESSIVE_CHUNKS_PER_TIME", "0.01")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.PROGRESSIVE_CHUNKS_PER_TIME == 0.01


def test_attrsvc_progressive_chunks_per_time_must_be_positive(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_PROGRESSIVE_CHUNKS_PER_TIME", "0")

    with pytest.raises(ValueError, match="PROGRESSIVE_CHUNKS_PER_TIME must be positive"):
        Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)


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
