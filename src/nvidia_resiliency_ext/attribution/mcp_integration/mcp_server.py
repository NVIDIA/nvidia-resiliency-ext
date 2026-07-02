"""
MCP Server implementation for NVRX Attribution modules.

This server exposes multiple attribution modules as MCP tools,
allowing them to be called individually or composed into pipelines.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from nvidia_resiliency_ext.attribution.logging_utils import bounded_log_value
from nvidia_resiliency_ext.attribution.mcp_integration.registry import (
    AttributionModuleRegistry,
    global_registry,
    serialize_result,
)
from nvidia_resiliency_ext.attribution.orchestration.llm_output import recommendation_payload
from nvidia_resiliency_ext.attribution.orchestration.progressive import (
    MODULE_LOG_ANALYZER_PROGRESSIVE_START,
    PROGRESSIVE_STATUS_FAILED,
)
from nvidia_resiliency_ext.attribution.orchestration.types import (
    AttributionRecommendation,
    LogSageAnalysisResult,
)
from nvidia_resiliency_ext.attribution.orchestration.utils import (
    LOG_ANALYZER_MODULE,
    log_analyzer_result_payload,
)

logger = logging.getLogger(__name__)


def _unknown_recommendation_payload(module_name: str) -> dict[str, str]:
    return recommendation_payload(AttributionRecommendation(source=module_name))


class NVRxMCPServer:
    """
    MCP Server for NVRX Attribution modules.

    Features:
    1. Exposes each registered module as an MCP tool
    2. Supports module composition/pipelining
    3. Caches results for efficient multi-module workflows
    4. Provides resource URIs for accessing cached results
    """

    def __init__(
        self,
        registry: Optional[AttributionModuleRegistry] = None,
        server_name: str = "nvidia-resiliency-attribution",
    ):
        """
        Initialize the MCP server.

        Args:
            registry: Module registry (uses global if not provided)
            server_name: Name of the MCP server
        """
        self.registry = registry or global_registry
        self.server = Server(server_name)
        self.module_instances: Dict[str, Any] = {}
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available attribution tools."""
            tools = []

            # Add individual module tools
            for metadata in self.registry.get_all_metadata():
                tools.append(
                    Tool(
                        name=metadata.name,
                        description=metadata.description,
                        inputSchema=metadata.input_schema,
                    )
                )

            # Add result retrieval tool
            tools.append(
                Tool(
                    name="get_result",
                    description="Retrieve a cached attribution result by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "result_id": {
                                "type": "string",
                                "description": "The result ID to retrieve",
                            }
                        },
                        "required": ["result_id"],
                    },
                )
            )

            # Add status tool
            tools.append(
                Tool(
                    name="status",
                    description="Get server status and list available modules",
                    inputSchema={"type": "object", "properties": {}},
                )
            )

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool calls."""
            try:
                # Special tools
                if name == "status":
                    return await self._handle_status()

                if name == "get_result":
                    return await self._handle_get_result(arguments)

                # Regular module tools
                return await self._handle_module_execution(name, arguments)

            except Exception as e:
                # ValueError: common validation / empty FR dir — message is enough at INFO default
                if isinstance(e, ValueError):
                    logger.warning("Error executing tool '%s': %s", name, e)
                else:
                    logger.error("Error executing tool '%s': %s", name, e, exc_info=True)
                if name == MODULE_LOG_ANALYZER_PROGRESSIVE_START:
                    error_body = {
                        "module": name,
                        "status": PROGRESSIVE_STATUS_FAILED,
                        "message": str(e),
                        "handle": None,
                    }
                elif self.registry.get_module_metadata(name):
                    error_body = {
                        "module": name,
                        "result": [],
                        "recommendation": _unknown_recommendation_payload(name),
                        "error": str(e),
                    }
                else:
                    error_body = {"error": str(e), "tool": name, "arguments": arguments}
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(error_body),
                    )
                ]

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources (cached results)."""
            resources = []
            for key in self.registry.list_results_cache_keys():
                module_name, result_id = key.split(":", 1)
                resources.append(
                    Resource(
                        uri=f"attribution://{module_name}/{result_id}",
                        name=f"{module_name} result {result_id}",
                        mimeType="application/json",
                        description=f"Cached result from {module_name} module",
                    )
                )
            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource (cached result)."""
            # Convert AnyUrl object to string if necessary
            uri_str = str(uri)

            # Parse URI: attribution://module_name/result_id
            if not uri_str.startswith("attribution://"):
                raise ValueError(f"Invalid resource URI: {uri_str}")

            path = uri_str.replace("attribution://", "")
            parts = path.split("/")

            if len(parts) != 2:
                raise ValueError(f"Invalid resource path: {path}")

            module_name, result_id = parts
            result = self.registry.get_cached_result_by_uri(module_name, result_id)

            if result is None:
                raise ValueError(f"Result not found: {module_name}:{result_id}")

            return serialize_result(result)

    async def _handle_status(self) -> List[TextContent]:
        """Handle status request."""
        modules = self.registry.list_modules()
        dependency_graph = self.registry.get_dependency_graph()

        status = {
            "status": "running",
            "modules": modules,
            "module_count": len(modules),
            "dependency_graph": dependency_graph,
            "cached_results": self.registry.count_results_cache_entries(),
            "active_instances": list(self.module_instances.keys()),
        }

        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    async def _handle_get_result(self, arguments: dict) -> List[TextContent]:
        """Handle result retrieval request."""
        result_id = arguments["result_id"]

        result = self.registry.get_cached_result_by_id(result_id)
        if result is not None:
            return [TextContent(type="text", text=serialize_result(result))]

        return [
            TextContent(type="text", text=json.dumps({"error": f"Result not found: {result_id}"}))
        ]

    async def _handle_module_execution(
        self, module_name: str, arguments: dict
    ) -> List[TextContent]:
        """Execute a single attribution module."""
        # Apply default values from input schema
        arguments_with_defaults = self.registry.apply_defaults(module_name, arguments)
        tool_arguments = (
            self._progressive_start_arguments(arguments_with_defaults)
            if module_name == MODULE_LOG_ANALYZER_PROGRESSIVE_START
            else arguments_with_defaults
        )

        # Get or create module instance
        if module_name not in self.module_instances:
            constructor_kwargs: Dict[str, Any] = {}
            if module_name == MODULE_LOG_ANALYZER_PROGRESSIVE_START:
                analyzer = self._get_or_create_log_analyzer(tool_arguments)
                if analyzer is not None:
                    constructor_kwargs["analyzer"] = analyzer
            instance_args = (
                self._log_analyzer_constructor_args(tool_arguments)
                if module_name == LOG_ANALYZER_MODULE
                else tool_arguments
            )
            instance = self.registry.create_instance(
                module_name,
                instance_args,
                constructor_kwargs=constructor_kwargs,
            )
            self.module_instances[module_name] = instance
        else:
            instance = self.module_instances[module_name]

        logger.info(
            "module=%s argument_keys=%s",
            module_name,
            sorted(tool_arguments.keys()),
        )
        logger.debug("module=%s arguments=%s", module_name, bounded_log_value(tool_arguments))

        # Run the attribution with defaults applied
        result = await instance.run(tool_arguments)

        if module_name == MODULE_LOG_ANALYZER_PROGRESSIVE_START:
            response = dict(result) if isinstance(result, dict) else {"status": str(result)}
            response.setdefault("module", module_name)
            return [TextContent(type="text", text=serialize_result(response))]

        result_id = self.registry.result_id_for_args(arguments_with_defaults)

        # Handle tuple results (result, AttributionState)
        actual_result = result

        if isinstance(result, tuple) and len(result) == 2:
            actual_result, _state = result

        if module_name == LOG_ANALYZER_MODULE and isinstance(
            actual_result, (list, LogSageAnalysisResult)
        ):
            response = log_analyzer_result_payload(actual_result, module=module_name)
            response.update(
                {
                    "result_id": result_id,
                    "resource_uri": f"attribution://{module_name}/{result_id}",
                }
            )
        elif (
            isinstance(actual_result, dict)
            and isinstance(actual_result.get("result"), list)
            and isinstance(actual_result.get("recommendation"), dict)
        ):
            response = dict(actual_result)
            response.setdefault("module", module_name)
            response.update(
                {
                    "result_id": result_id,
                    "resource_uri": f"attribution://{module_name}/{result_id}",
                }
            )
        else:
            response = {
                "module": module_name,
                "result_id": result_id,
                "resource_uri": f"attribution://{module_name}/{result_id}",
                "result": actual_result,
                "recommendation": _unknown_recommendation_payload(module_name),
            }

        self.registry.cache_result_by_id(module_name, result_id, response)

        return [TextContent(type="text", text=serialize_result(response))]

    @staticmethod
    def _log_analyzer_constructor_args(arguments: dict) -> dict:
        """Drop NVRx orchestration-only request metadata before constructing LogSage."""
        return {key: value for key, value in arguments.items() if key not in {"user", "job_id"}}

    @staticmethod
    def _progressive_start_arguments(arguments: dict) -> dict:
        """Keep NVRx request/reporting metadata out of the MCP progressive tool."""
        return {key: value for key, value in arguments.items() if key not in {"user", "job_id"}}

    def _get_or_create_log_analyzer(self, arguments: dict) -> Optional[Any]:
        """Return the shared LogSage analyzer instance for progressive and terminal MCP tools."""
        if LOG_ANALYZER_MODULE in self.module_instances:
            return self.module_instances[LOG_ANALYZER_MODULE]
        if self.registry.get_module_metadata(LOG_ANALYZER_MODULE) is None:
            return None
        analyzer = self.registry.create_instance(
            LOG_ANALYZER_MODULE,
            self._log_analyzer_constructor_args(arguments),
        )
        self.module_instances[LOG_ANALYZER_MODULE] = analyzer
        return analyzer

    async def run(self):
        """Run the MCP server."""
        import os

        logger.info("Starting NVRX Attribution MCP Server")
        logger.info(f"Registered modules: {self.registry.list_modules()}, pid: {os.getpid()}")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )

    def run_sync(self):
        """Run the server synchronously."""
        asyncio.run(self.run())
