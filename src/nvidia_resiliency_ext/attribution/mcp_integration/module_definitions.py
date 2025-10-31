"""
Module definitions for NVRX Attribution modules.

This file registers all available attribution modules with the registry.
"""

import argparse

from nvidia_resiliency_ext.attribution.mcp_integration.registry import global_registry
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer

def register_all_modules():
    """Register all NVRX attribution modules with the global registry."""

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
