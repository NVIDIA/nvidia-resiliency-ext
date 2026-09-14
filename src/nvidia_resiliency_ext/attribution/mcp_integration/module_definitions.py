"""
Packaged MCP module definitions for NVRX Attribution.

The restart-agent log analyzer and Flight Recorder analyzer are registered
here. Legacy LogSage/LangChain-backed tools live under the source-only
``legacy_logsage`` tree and are registered explicitly by ``server_launcher``.
"""

import asyncio
from typing import Any

from nvidia_resiliency_ext.attribution.mcp_integration.registry import global_registry
from nvidia_resiliency_ext.attribution.orchestration.llm_output import recommendation_payload
from nvidia_resiliency_ext.attribution.orchestration.types import (
    RECOMMENDATION_ACTIONS,
    RECOMMENDATION_PAYLOAD_FIELDS,
    AttributionRecommendation,
)
from nvidia_resiliency_ext.attribution.restart_agent import RestartAgent, RestartAgentRequest
from nvidia_resiliency_ext.attribution.restart_agent.l1.contracts import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer

MODULE_RESTART_AGENT = "restart_agent"

_RECOMMENDATION_FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    "action": {
        "type": "string",
        "enum": list(RECOMMENDATION_ACTIONS),
        "description": "Client-facing action derived from analyzer output",
    },
    "source": {
        "type": "string",
        "description": "Signal/source that produced the recommendation",
    },
}


class RestartAgentMCPModule:
    """MCP adapter for the packaged restart-agent log analyzer."""

    def __init__(self, _args: dict[str, Any]):
        self._agent = RestartAgent()

    async def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        timeout_arg = arguments.get("timeout_seconds")
        timeout_seconds = (
            DEFAULT_ANALYSIS_TIMEOUT_SECONDS if timeout_arg is None else float(timeout_arg)
        )
        retain_detailed_artifacts = bool(arguments.get("retain_detailed_artifacts", False))
        request = RestartAgentRequest(
            log_path=str(arguments["log_path"]),
            job_id=arguments.get("job_id"),
            cycle_id=arguments.get("cycle_id"),
        )
        run = await asyncio.to_thread(
            self._agent.run,
            request,
            timeout_seconds=timeout_seconds,
            retain_detailed_artifacts=retain_detailed_artifacts,
        )
        result = run.result
        payload = result.to_payload()
        return {
            "module": MODULE_RESTART_AGENT,
            "result": payload,
            "recommendation": recommendation_payload(
                AttributionRecommendation(
                    action=result.decision,
                    source=MODULE_RESTART_AGENT,
                )
            ),
        }


def _recommendation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field_name: _RECOMMENDATION_FIELD_SCHEMAS[field_name]
            for field_name in RECOMMENDATION_PAYLOAD_FIELDS
        },
        "required": list(RECOMMENDATION_PAYLOAD_FIELDS),
        "description": "Derived client recommendation envelope",
    }


def _restart_agent_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "log_path": {
                "type": "string",
                "description": "Absolute path to the terminal training log",
            },
            "job_id": {
                "type": ["string", "null"],
                "description": "Optional scheduler/job identifier",
            },
            "cycle_id": {
                "type": ["integer", "null"],
                "description": "Optional restart cycle identifier",
            },
            "timeout_seconds": {
                "type": ["number", "null"],
                "description": "Maximum wall-clock seconds for analysis",
                "default": DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
            },
            "retain_detailed_artifacts": {
                "type": ["boolean", "null"],
                "description": "Retain detailed intermediate artifacts in memory",
                "default": False,
            },
        },
        "required": ["log_path"],
    }


def _restart_agent_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "module": {
                "type": "string",
                "description": f"Module name: {MODULE_RESTART_AGENT}",
            },
            "result": {
                "type": "object",
                "description": "Restart-agent decision payload",
            },
            "recommendation": _recommendation_schema(),
        },
        "required": ["module", "result", "recommendation"],
    }


def _fr_analyzer_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fr_path": {
                "type": "string",
                "description": "Paths to FR dump files",
            },
            "verbose": {
                "type": "boolean",
                "description": "Enable verbose output",
                "default": False,
            },
            "health_check": {
                "type": "boolean",
                "description": "Show node health check results",
                "default": False,
            },
            "pattern": {
                "type": "string",
                "description": "File pattern to match",
                "default": "_dump_*",
            },
        },
        "required": ["fr_path"],
    }


def _fr_analyzer_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "module": {
                "type": "string",
                "description": "Module name: fr_analyzer",
            },
            "result": {
                "type": "object",
                "description": "Collective analysis results including hanging ranks",
            },
            "recommendation": _recommendation_schema(),
        },
        "required": ["module", "result", "recommendation"],
    }


def register_all_modules() -> None:
    """Register packaged NVRX attribution MCP modules with the global registry."""

    global_registry.register(
        name=MODULE_RESTART_AGENT,
        module_class=RestartAgentMCPModule,
        description="Analyze terminal distributed-training logs with restart-agent",
        input_schema=_restart_agent_input_schema(),
        output_schema=_restart_agent_output_schema(),
        requires_llm=False,
        dependencies=[],
    )
    global_registry.register(
        name="fr_analyzer",
        module_class=CollectiveAnalyzer,
        description="Analyze PyTorch Flight Recorder traces to identify collective operation hangs",
        input_schema=_fr_analyzer_input_schema(),
        output_schema=_fr_analyzer_output_schema(),
        requires_llm=False,
        dependencies=[],
    )


def create_args_from_dict(module_name: str, config: dict) -> dict:
    """
    Build a module argument dict from schema defaults and ``config`` overrides.

    Args:
        module_name: Name of the module
        config: Configuration dictionary

    Returns:
        Plain dict suitable for module constructors and :meth:`~nvidia_resiliency_ext.attribution.base.NVRxAttribution.run`.
    """
    metadata = global_registry.get_module_metadata(module_name)
    if not metadata:
        raise ValueError(f"Module '{module_name}' not found in registry")

    properties = metadata.input_schema.get("properties", {})

    args_dict = {}
    for prop_name, prop_schema in properties.items():
        default = prop_schema.get("default")
        if default is not None:
            args_dict[prop_name] = default

    args_dict.update(config)

    return args_dict
