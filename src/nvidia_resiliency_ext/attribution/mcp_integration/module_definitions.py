"""
Module definitions for NVRX Attribution modules.

This file registers all available attribution modules with the registry.
"""

from typing import Any

from nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr_mcp import (
    CombinedLogFRMCPOrchestrator,
)
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
from nvidia_resiliency_ext.attribution.mcp_integration.registry import global_registry
from nvidia_resiliency_ext.attribution.orchestration.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_P,
)
from nvidia_resiliency_ext.attribution.orchestration.progressive import (
    MODULE_LOG_ANALYZER_PROGRESSIVE_START,
    PROGRESSIVE_STATUS_UNSUPPORTED,
    ProgressiveLogAnalysisStartTool,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    RAW_ANALYSIS_RESULT_ITEM_PAYLOAD_FIELDS,
    RECOMMENDATION_ACTIONS,
    RECOMMENDATION_PAYLOAD_FIELDS,
)
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer

_RAW_ANALYSIS_RESULT_ITEM_FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    "raw_text": {
        "type": "string",
        "description": "Raw LogSage result text",
    },
    "auto_resume": {
        "type": "string",
        "description": "Parsed restart/stop decision from LogSage",
    },
    "auto_resume_explanation": {
        "type": "string",
        "description": "Parsed explanation for the restart/stop decision",
    },
    "attribution_text": {
        "type": "string",
        "description": "Parsed attribution text without the Attribution: prefix",
    },
    "checkpoint_saved_flag": {
        "type": "integer",
        "enum": [0, 1],
        "description": "Whether LogSage reported a checkpoint was saved",
    },
    "action": {
        "type": "string",
        "enum": list(RECOMMENDATION_ACTIONS),
        "description": "Parsed cycle action from LogSage",
    },
    "primary_issues": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Parsed primary attribution issues",
    },
    "secondary_issues": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Parsed secondary attribution issues",
    },
}

_RECOMMENDATION_FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    "action": {
        "type": "string",
        "enum": list(RECOMMENDATION_ACTIONS),
        "description": "Client-facing action derived from LogSage output",
    },
    "source": {
        "type": "string",
        "description": "Signal/source that produced the recommendation",
    },
}


def _raw_analysis_result_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field_name: _RAW_ANALYSIS_RESULT_ITEM_FIELD_SCHEMAS[field_name]
            for field_name in RAW_ANALYSIS_RESULT_ITEM_PAYLOAD_FIELDS
        },
        "required": list(RAW_ANALYSIS_RESULT_ITEM_PAYLOAD_FIELDS),
    }


def _recommendation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field_name: _RECOMMENDATION_FIELD_SCHEMAS[field_name]
            for field_name in RECOMMENDATION_PAYLOAD_FIELDS
        },
        "required": list(RECOMMENDATION_PAYLOAD_FIELDS),
        "description": (
            "Derived client recommendation; consumers should branch on this, "
            "not raw LogSage fields"
        ),
    }


def _llm_runtime_properties(*, model_description: str) -> dict[str, dict[str, Any]]:
    """Input-schema properties shared by LogSage-backed MCP tools."""
    return {
        "model": {
            "type": "string",
            "description": model_description,
            "default": DEFAULT_LLM_MODEL,
        },
        "base_url": {
            "type": "string",
            "description": "LLM base url",
            "default": DEFAULT_LLM_BASE_URL,
        },
        "temperature": {
            "type": "number",
            "description": "Temperature for LLM sampling",
            "default": DEFAULT_LLM_TEMPERATURE,
        },
        "top_p": {
            "type": "number",
            "description": "Top-p for LLM sampling",
            "default": DEFAULT_LLM_TOP_P,
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum tokens for LLM response",
            "default": DEFAULT_LLM_MAX_TOKENS,
        },
    }


def _log_analyzer_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "log_path": {"type": "string", "description": "Path to log files"},
            **_llm_runtime_properties(model_description="LLM model to use for analysis"),
            "exclude_nvrx_logs": {
                "type": "boolean",
                "description": "Exclude NVRX internal logs",
                "default": False,
            },
            "is_per_cycle": {
                "type": "boolean",
                "description": "Input is already per-cycle data (skip filtering and chunking)",
                "default": False,
            },
        },
        "required": ["log_path"],
    }


def _progressive_start_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "log_path": {"type": "string", "description": "Path to the growing log file"},
            "is_per_cycle": {
                "type": "boolean",
                "description": "Input is already a single ft_launcher cycle log",
                "default": False,
            },
            "user": {
                "type": "string",
                "description": "Optional submitting user for observability",
                "default": "unknown",
            },
            "job_id": {
                "type": "string",
                "description": "Optional scheduler job id for observability",
            },
            **_llm_runtime_properties(
                model_description="LLM model to bind to the progressive session"
            ),
        },
        "required": ["log_path"],
    }


def _progressive_start_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "module": {
                "type": "string",
                "description": f"Module name: {MODULE_LOG_ANALYZER_PROGRESSIVE_START}",
            },
            "status": {
                "type": "string",
                "description": "Progressive start status",
                "default": PROGRESSIVE_STATUS_UNSUPPORTED,
            },
            "message": {
                "type": "string",
                "description": "Human-readable status detail",
            },
            "handle": {
                "type": ["string", "null"],
                "description": "Optional backend session handle",
            },
        },
        "required": ["module", "status", "message", "handle"],
    }


def register_all_modules():
    """Register all NVRX attribution modules with the global registry."""

    # Register Log Analyzer (LogSage)
    global_registry.register(
        name="log_analyzer",
        module_class=NVRxLogAnalyzer,
        description="Analyze application logs using LogSage to identify errors and propose solutions",
        input_schema=_log_analyzer_input_schema(),
        output_schema={
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Module name: log_analyzer",
                },
                "result": {
                    "type": "array",
                    "items": _raw_analysis_result_item_schema(),
                    "description": "Per-cycle attribution results",
                },
                "recommendation": _recommendation_schema(),
            },
            "required": ["module", "result", "recommendation"],
        },
        requires_llm=True,
        dependencies=[],
    )

    # Non-result-producing progressive start entry point. This advertises the
    # NVRx-owned MCP/loganalysis boundary before the LogSage progressive API is
    # available; the tool returns unsupported status metadata today.
    global_registry.register(
        name=MODULE_LOG_ANALYZER_PROGRESSIVE_START,
        module_class=ProgressiveLogAnalysisStartTool,
        description=(
            "Start progressive log analysis for a growing application log. "
            "Returns status metadata only; it does not produce final attribution."
        ),
        input_schema=_progressive_start_input_schema(),
        output_schema=_progressive_start_output_schema(),
        requires_llm=False,
        dependencies=[],
    )

    # Register FR (Flight Recorder) Analyzer
    global_registry.register(
        name="fr_analyzer",
        module_class=CollectiveAnalyzer,
        description="Analyze PyTorch Flight Recorder traces to identify collective operation hangs",
        input_schema={
            "type": "object",
            "properties": {
                "fr_path": {
                    "type": "string",
                    "description": "Paths to FR dump files",
                },
                "model": {
                    "type": "string",
                    "description": "LLM model for analysis (optional)",
                    "default": DEFAULT_LLM_MODEL,
                },
                "base_url": {
                    "type": "string",
                    "description": "LLM base url",
                    "default": DEFAULT_LLM_BASE_URL,
                },
                "scheduling_order_file": {
                    "type": "string",
                    "description": "Process group scheduling order (e.g., 'TP->PP->DP')",
                    "default": "TP->PP->DP",
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
                "llm_analyze": {
                    "type": "boolean",
                    "description": "Use LLM for analysis",
                    "default": False,
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern to match",
                    "default": "_dump_*",
                },
            },
            "required": ["fr_path"],
        },
        output_schema={
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
        },
        requires_llm=False,
        dependencies=[],
    )

    # Log + FR in one MCP round-trip (path mode), with optional LLM merge.
    global_registry.register(
        name="log_fr_analyzer",
        module_class=CombinedLogFRMCPOrchestrator,
        description=(
            "Path mode: run LogSage and FR in parallel in this process. "
            "Set merge_llm=true to also run the Log+FR LLM merge."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "log_path": {
                    "type": "string",
                    "description": "Application log path (path mode; use with fr_path)",
                },
                "fr_path": {
                    "type": "string",
                    "description": (
                        "FR dump directory, a single dump file, or path prefix (TORCH_FR_DUMP_TEMP_FILE "
                        "style, e.g. /shared/_dump_ when _dump_0 exists); path mode with log_path"
                    ),
                },
                "input_data": {
                    "type": "array",
                    "items": {
                        "description": "First item is a LogSage result; second item is an FR result",
                    },
                    "description": "Input-data mode: [log_result, fr_result] from prior tool runs",
                },
                "model": {
                    "type": "string",
                    "description": "LLM model",
                    "default": DEFAULT_LLM_MODEL,
                },
                "base_url": {
                    "type": "string",
                    "description": "LLM base url",
                    "default": DEFAULT_LLM_BASE_URL,
                },
                "temperature": {"type": "number", "default": DEFAULT_LLM_TEMPERATURE},
                "top_p": {"type": "number", "default": DEFAULT_LLM_TOP_P},
                "max_tokens": {"type": "integer", "default": DEFAULT_LLM_MAX_TOKENS},
                "threshold": {"type": "integer", "default": 0},
                "merge_llm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run Log+FR LLM merge after collecting LogSage and FR outputs",
                },
                "exclude_nvrx_logs": {"type": "boolean", "default": False},
                "is_per_cycle": {"type": "boolean", "default": False},
                "pattern": {"type": "string", "default": "_dump_*"},
                "verbose": {"type": "boolean", "default": False},
                "health_check": {"type": "boolean", "default": False},
                "llm_analyze": {"type": "boolean", "default": False},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Module name: log_fr_analyzer",
                },
                "result": {
                    "type": "array",
                    "items": _raw_analysis_result_item_schema(),
                    "description": "Per-cycle LogSage attribution results from combined log+FR analysis",
                },
                "recommendation": _recommendation_schema(),
                "fr": {
                    "type": "object",
                    "description": "Flight recorder analysis payload and internal state",
                },
                "llm_merged_summary": {
                    "type": "string",
                    "description": (
                        "LLM merge summary from LogSage and flight recorder outputs; present only "
                        "when merge ran with FR data"
                    ),
                },
            },
            "required": ["module", "result", "recommendation", "fr"],
        },
        requires_llm=True,
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

    # Get schema defaults
    schema = metadata.input_schema
    properties = schema.get("properties", {})

    # Build args with defaults
    args_dict = {}
    for prop_name, prop_schema in properties.items():
        default = prop_schema.get("default")
        if default is not None:
            args_dict[prop_name] = default

    # Override with provided config
    args_dict.update(config)

    return args_dict
