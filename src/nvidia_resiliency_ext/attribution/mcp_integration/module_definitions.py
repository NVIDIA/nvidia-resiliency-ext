"""
Module definitions for NVRX Attribution modules.

This file registers all available attribution modules with the registry.
"""

import argparse

from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer
from nvidia_resiliency_ext.attribution.mcp_integration.registry import global_registry
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer


def register_all_modules():
    """Register all NVRX attribution modules with the global registry."""

    # Register Log Analyzer (LogSage)
    global_registry.register(
        name="log_analyzer",
        module_class=NVRxLogAnalyzer,
        description="Analyze application logs using LogSage to identify errors and propose solutions",
        input_schema={
            "type": "object",
            "properties": {
                "log_path": {"type": "string", "description": "Path to log files"},
                "model": {
                    "type": "string",
                    "description": "LLM model to use for analysis",
                    "default": "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature for LLM sampling",
                    "default": 0.2,
                },
                "top_p": {
                    "type": "number",
                    "description": "Top-p for LLM sampling",
                    "default": 0.7,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for LLM response",
                    "default": 8192,
                },
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
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "Attribution result with proposed solution",
                },
                "state": {
                    "type": "string",
                    "enum": ["CONTINUE", "STOP"],
                    "description": "Whether to continue pipeline",
                },
            },
        },
        requires_llm=True,
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
                    "default": "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
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
                "result": {
                    "type": "object",
                    "description": "Collective analysis results including hanging ranks",
                },
                "state": {"type": "string", "enum": ["CONTINUE", "STOP"]},
            },
        },
        requires_llm=False,
        dependencies=[],
    )

    # Register Combined Log + FR Analyzer
    try:
        from nvidia_resiliency_ext.attribution.combined_log_fr.combined_log_fr import CombinedLogFR

        global_registry.register(
            name="combined_log_fr",
            module_class=CombinedLogFR,
            description="Combined analysis of application logs and FR traces with LLM synthesis",
            input_schema={
                "type": "object",
                "properties": {
                    "input_data": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "description": "Input data for the module",
                        },
                        "description": "Input data for the module",
                    },
                    "model": {
                        "type": "string",
                        "description": "LLM model for synthesis",
                        "default": "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
                    },
                    "threshold": {
                        "type": "integer",
                        "description": "Threshold for stopping pipeline",
                        "default": 0,
                    },
                },
                "required": ["input_data"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Combined attribution with actionable recommendations",
                    },
                    "state": {"type": "string", "enum": ["CONTINUE", "STOP"]},
                },
            },
            requires_llm=True,
            dependencies=["log_analyzer", "fr_analyzer"],
        )
    except ImportError:
        pass


def create_args_from_dict(module_name: str, config: dict) -> argparse.Namespace:
    """
    Create an argparse.Namespace from a configuration dictionary.

    Args:
        module_name: Name of the module
        config: Configuration dictionary

    Returns:
        argparse.Namespace with module configuration
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

    return argparse.Namespace(**args_dict)
