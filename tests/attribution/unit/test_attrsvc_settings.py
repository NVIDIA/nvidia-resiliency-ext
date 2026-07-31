import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

attrsvc_config = pytest.importorskip("nvidia_resiliency_ext.services.attrsvc.config")
Settings = attrsvc_config.Settings


def test_attrsvc_analysis_backend_defaults_to_direct_restart_agent(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_ANALYSIS_BACKEND", raising=False)

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.ANALYSIS_BACKEND == "lib"
    assert cfg.RESTART_AGENT_ENRICHMENT_ENABLED is True


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


def test_attrsvc_analysis_backend_uses_current_env_name_only(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_ANALYSIS_BACKEND", "lib")
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_ANALYSIS_BACKEND", "mcp")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.ANALYSIS_BACKEND == "lib"


def test_attrsvc_progressive_analysis_defaults_to_all_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS", raising=False)

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.PROGRESSIVE_ANALYSIS == "all_explicit"
    assert cfg.RESTART_AGENT_PROGRESSIVE_ENABLED is False
    assert cfg.RESTART_AGENT_PRE_END_POLL_SECONDS == 180.0
    assert cfg.RESTART_AGENT_ACTIVE_IDLE_SECONDS == 900.0
    assert cfg.RESTART_AGENT_MAX_ACTIVE_STATES == 64
    assert cfg.RESTART_AGENT_MAX_COMPLETED_RESULTS == 3000


def test_attrsvc_live_terminal_drain_uses_conservative_defaults(tmp_path):
    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.RESTART_AGENT_LOG_QUIET_SECONDS == 5.0
    assert cfg.RESTART_AGENT_LOG_MAX_WAIT_SECONDS == 40.0
    assert cfg.RESTART_AGENT_LOG_POLL_SECONDS == 0.25


def test_attrsvc_progressive_analysis_accepts_off(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS", "OFF")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.PROGRESSIVE_ANALYSIS == "off"


def test_attrsvc_progressive_analysis_accepts_all_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS", "ALL_EXPLICIT")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.PROGRESSIVE_ANALYSIS == "all_explicit"


def test_attrsvc_restart_agent_progressive_analysis_accepts_explicit_enable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NVRX_ATTRSVC_RESTART_AGENT_PROGRESSIVE_ENABLED", "true")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.RESTART_AGENT_PROGRESSIVE_ENABLED is True


def test_attrsvc_log_analysis_backend_env_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.delenv("NVRX_ATTRSVC_ANALYSIS_BACKEND", raising=False)
    monkeypatch.setenv("NVRX_ATTRSVC_LOG_ANALYSIS_BACKEND", "lib")

    cfg = Settings(ALLOWED_ROOT=str(tmp_path), _env_file=None)

    assert cfg.ANALYSIS_BACKEND == "lib"


def test_attrsvc_restart_agent_config_resolves_existing_llm_environment(tmp_path, monkeypatch):
    resolver = pytest.importorskip(
        "nvidia_resiliency_ext.services.attrsvc.restart_agent_config"
    ).restart_agent_config_from_settings
    key_file = tmp_path / "key"
    key_file.write_text("secret", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(key_file))
    cfg = Settings(
        ALLOWED_ROOT=str(tmp_path),
        LLM_MODEL="model-a",
        LLM_BASE_URL="https://llm.example.test/v1",
        _env_file=None,
    )

    restart_config = resolver(cfg)

    assert restart_config.config_id == "nvrx-attrsvc-environment"
    assert len(restart_config.model_route_specs) == 1
    route = restart_config.model_route_specs[0]
    assert route.route_id == "nvrx-default"
    assert route.model == "model-a"
    assert route.endpoint == "https://llm.example.test/v1"


def test_attrsvc_restart_agent_allows_explicit_deterministic_only_mode(tmp_path):
    resolver = pytest.importorskip(
        "nvidia_resiliency_ext.services.attrsvc.restart_agent_config"
    ).restart_agent_config_from_settings
    cfg = Settings(
        ALLOWED_ROOT=str(tmp_path),
        RESTART_AGENT_ENRICHMENT_ENABLED=False,
        _env_file=None,
    )

    restart_config = resolver(cfg, environ={})

    assert restart_config.config_id == "nvrx-attrsvc-deterministic"
    assert restart_config.enrichment_enabled is False
    assert restart_config.model_route_specs == ()
    assert restart_config.max_parallel_models == 0


def test_attrsvc_restart_agent_config_file_is_authoritative_and_single_route(tmp_path):
    resolver = pytest.importorskip(
        "nvidia_resiliency_ext.services.attrsvc.restart_agent_config"
    ).restart_agent_config_from_settings
    config_path = tmp_path / "restart_agent.json"
    payload = {
        "schema_version": "restart_agent_config.v1",
        "config_id": "file-config",
        "config_version": 1,
        "enrichment": {"enabled": True},
        "routing": {"mode": "collect_all", "max_parallel_models": 2},
        "model_routes": [
            {
                "route_id": route_id,
                "model": route_id,
                "base_url": "https://llm.example.test/v1",
                "credential_ref": "TEST_KEY",
            }
            for route_id in ("one", "two")
        ],
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = Settings(
        ALLOWED_ROOT=str(tmp_path),
        RESTART_AGENT_CONFIG=str(config_path),
        _env_file=None,
    )

    with pytest.raises(ValueError, match="exactly one model route"):
        resolver(cfg, environ={"TEST_KEY": "/unused/key"})


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
