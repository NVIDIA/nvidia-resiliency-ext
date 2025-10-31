"""
MCP Server implementation for NVRX Attribution modules.

This server exposes multiple attribution modules as MCP tools,
allowing them to be called individually or composed into pipelines.
"""

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from nvidia_resiliency_ext.attribution.base import AttributionState
from nvidia_resiliency_ext.attribution.mcp_integration.registry import (
    AttributionModuleRegistry,
    global_registry,
    serialize_result,
)

logger = logging.getLogger(__name__)


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
                logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(e), "tool": name, "arguments": arguments}),
                    )
                ]

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources (cached results)."""
            resources = []
            for key in self.registry._results_cache.keys():
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
            "cached_results": len(self.registry._results_cache),
            "active_instances": list(self.module_instances.keys()),
        }

        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    async def _handle_get_result(self, arguments: dict) -> List[TextContent]:
        """Handle result retrieval request."""
        result_id = arguments["result_id"]

        # Search for result in cache
        for key, value in self.registry._results_cache.items():
            if result_id in key:
                return [TextContent(type="text", text=serialize_result(value))]

        return [
            TextContent(type="text", text=json.dumps({"error": f"Result not found: {result_id}"}))
        ]

    async def _handle_module_execution(
        self, module_name: str, arguments: dict
    ) -> List[TextContent]:
        """Execute a single attribution module."""
        # Apply default values from input schema
        arguments_with_defaults = self.registry.apply_defaults(module_name, arguments)
        
        # Get or create module instance
        if module_name not in self.module_instances:
            # Convert arguments to argparse.Namespace
            args = argparse.Namespace(**arguments_with_defaults)
            instance = self.registry.create_instance(module_name, args)
            self.module_instances[module_name] = instance
        else:
            instance = self.module_instances[module_name]

        # Extract input data
        logger.info(f"arguments: {arguments_with_defaults}")

        # Run the attribution with defaults applied
        result = await instance.run(arguments_with_defaults)

        # Generate result ID and cache
        result_id = self.registry.cache_result(module_name, arguments, result)

        # Handle tuple results (result, AttributionState)
        actual_result = result
        state = AttributionState.CONTINUE

        if isinstance(result, tuple) and len(result) == 2:
            actual_result, state = result

        response = {
            "module": module_name,
            "result_id": result_id,
            "resource_uri": f"attribution://{module_name}/{result_id}",
            "result": actual_result,
            "state": state.name if isinstance(state, AttributionState) else str(state),
        }

        return [TextContent(type="text", text=serialize_result(response))]


    async def run(self):
        """Run the MCP server."""
        import os

        logger.info(f"Starting NVRX Attribution MCP Server")
        logger.info(f"Registered modules: {self.registry.list_modules()}, pid: {os.getpid()}")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )

    def run_sync(self):
        """Run the server synchronously."""
        asyncio.run(self.run())