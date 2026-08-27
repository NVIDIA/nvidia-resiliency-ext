# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys

import pytest

from nvidia_resiliency_ext.attribution.mcp_integration.registry import AttributionModuleRegistry


def test_packaged_mcp_definitions_register_restart_agent_and_fr_without_logsage(monkeypatch):
    """The packaged MCP registry must not import source-only LogSage modules."""
    module_name = "nvidia_resiliency_ext.attribution.mcp_integration.module_definitions"
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.legacy_logsage.log_analyzer.nvrx_logsage",
        None,
    )
    monkeypatch.setitem(
        sys.modules,
        "nvidia_resiliency_ext.attribution.legacy_logsage.combined_log_fr.combined_log_fr_mcp",
        None,
    )

    module = importlib.import_module(module_name)
    registry = AttributionModuleRegistry()
    monkeypatch.setattr(module, "global_registry", registry)

    module.register_all_modules()

    assert registry.list_modules() == ["restart_agent", "fr_analyzer"]
    restart_metadata = registry.get_module_metadata("restart_agent")
    assert restart_metadata is not None
    assert restart_metadata.requires_llm is False
    assert "log_path" in restart_metadata.input_schema["properties"]
    assert "model" not in restart_metadata.input_schema["properties"]
    assert module.create_args_from_dict("restart_agent", {"log_path": "/tmp/job.log"}) == {
        "log_path": "/tmp/job.log",
        "timeout_seconds": 240.0,
        "retain_detailed_artifacts": False,
    }

    metadata = registry.get_module_metadata("fr_analyzer")
    assert metadata is not None
    assert metadata.requires_llm is False
    assert "fr_path" in metadata.input_schema["properties"]
    assert "llm_analyze" not in metadata.input_schema["properties"]
    assert "model" not in metadata.input_schema["properties"]
    assert module.create_args_from_dict("fr_analyzer", {"fr_path": "/tmp/fr"}) == {
        "fr_path": "/tmp/fr",
        "verbose": False,
        "health_check": False,
        "pattern": "_dump_*",
    }


def test_restart_agent_mcp_module_returns_recommendation_envelope(tmp_path):
    module = importlib.import_module(
        "nvidia_resiliency_ext.attribution.mcp_integration.module_definitions"
    )
    log_path = tmp_path / "job.log"
    log_path.write_text("RuntimeError: CUDA error: uncorrectable ECC error encountered\n")

    result = asyncio.run(module.RestartAgentMCPModule({}).run({"log_path": str(log_path)}))

    assert result["module"] == "restart_agent"
    assert result["recommendation"] == {
        "action": result["result"]["decision"],
        "source": "restart_agent",
    }
    assert result["result"]["decision"] in {"RESTART", "STOP"}


def test_server_launcher_rejects_legacy_modules_without_legacy_registration():
    launcher = importlib.import_module(
        "nvidia_resiliency_ext.attribution.mcp_integration.server_launcher"
    )

    with pytest.raises(SystemExit) as exc_info:
        launcher._validate_requested_modules(["log_analyzer"], ["restart_agent", "fr_analyzer"])

    assert "Requested module(s) not registered: log_analyzer" in str(exc_info.value)
    assert "--enable-legacy-logsage" in str(exc_info.value)
