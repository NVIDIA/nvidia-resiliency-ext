"""
MCP Client for connecting to NVRX Attribution servers.

This client allows:
1. Calling attribution modules on remote MCP servers
2. Chaining modules across different servers
3. Retrieving cached results
"""

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from importlib.resources import files as pkg_files
from typing import Any, Dict, List

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from nvidia_resiliency_ext.attribution.mcp_integration.registry import deserialize_result

logger = logging.getLogger(__name__)


def get_server_command() -> List[str]:
    """
    Resolve and return the server launcher command for the MCP client.

    Returns:
        Command list to launch the MCP server subprocess.
    """
    pkg = "nvidia_resiliency_ext.attribution.mcp_integration"
    try:
        resource = pkg_files(pkg).joinpath("server_launcher.py")
    except Exception as e:
        raise FileNotFoundError(f"failed to locate server_launcher.py in package {pkg}: {e}")
    if not resource.exists():
        raise FileNotFoundError(f"server launcher not found in package: {pkg}/server_launcher.py")
    # Use WARNING to reduce subprocess log verbosity
    return ["python", str(resource), "--log-level", "WARNING"]


def create_mcp_client() -> "NVRxMCPClient":
    """
    Create and return an NVRxMCPClient with the default server command.

    Returns:
        Configured NVRxMCPClient ready for use as async context manager.
    """
    return NVRxMCPClient(get_server_command())


class NVRxMCPClient:
    """
    Client for interacting with NVRX Attribution MCP servers.

    Supports:
    - Calling individual attribution modules
    - Running multi-module pipelines
    - Cross-server module composition
    - Result caching and retrieval
    """

    def __init__(self, server_command: List[str]):
        """
        Initialize the MCP client.

        Args:
            server_command: Command to start the MCP server (e.g., ["python", "server.py"])
        """
        self.server_command = server_command
        self.client = None
        self._context = None
        self.exit_stack = AsyncExitStack()
        logger.info(f"Initialized MCP client with server command: {self.server_command}")

    async def __aenter__(self):
        """Async context manager entry."""
        import os

        # Convert server_command list to StdioServerParameters
        logger.info(f"pid: {os.getpid()} is connecting to server")
        if isinstance(self.server_command, list):
            logger.info(f"Server command is a list: {self.server_command}")
            # Split command list into executable and args
            command = self.server_command[0] if self.server_command else "python"
            args = self.server_command[1:] if len(self.server_command) > 1 else []

            # Pass current environment variables to subprocess (especially PYTHONPATH)
            env = dict(os.environ)

            server_params = StdioServerParameters(command=command, args=args, env=env)
        else:
            server_params = self.server_command

        self._context = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = self._context
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        logger.info(
            f"Client session created with read_stream: {read_stream} and write_stream: {write_stream}"
        )
        # Initialize the client session
        await self.session.initialize()
        logger.info("Client session initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.exit_stack.aclose()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on the server."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        response = await self.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.inputSchema) if tool.inputSchema else {},
            }
            for tool in response.tools
        ]

    async def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        result = await self.call_tool("status", {})
        return deserialize_result(result)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Call a tool on the server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool result as string
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        result = await self.session.call_tool(tool_name, arguments)

        if result and len(result.content) > 0:
            return result.content[0].text

        return json.dumps({"error": "No result returned"})

    async def run_module(self, module_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a single attribution module.

        Args:
            module_name: Name of the module to run
            input_data: Input data for the module
            **kwargs: Additional configuration for the module

        Returns:
            Module execution result
        """
        arguments = kwargs

        result_str = await self.call_tool(module_name, arguments)
        return deserialize_result(result_str)

    async def get_result(self, result_id: str) -> Dict[str, Any]:
        """
        Retrieve a cached result by ID.

        Args:
            result_id: The result ID to retrieve

        Returns:
            The cached result
        """
        result_str = await self.call_tool("get_result", {"result_id": result_id})
        return deserialize_result(result_str)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources (cached results)."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        response = await self.session.list_resources()
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "mime_type": resource.mimeType,
                "description": resource.description,
            }
            for resource in response.resources
        ]

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource by URI.

        Args:
            uri: Resource URI (e.g., "attribution://module_name/result_id")

        Returns:
            Resource content
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        result = await self.session.read_resource(uri)

        # Extract text content from ReadResourceResult
        if result and len(result.contents) > 0:
            content_str = result.contents[0].text
            return deserialize_result(content_str)

        return {"error": "No content in resource"}


class MultiServerClient:
    """
    Client for managing multiple MCP servers and orchestrating cross-server workflows.

    Use this to:
    - Connect to multiple attribution servers
    - Route requests to appropriate servers
    - Compose modules across different servers
    """

    def __init__(self):
        """Initialize the multi-server client."""
        self.servers: Dict[str, NVRxMCPClient] = {}
        self.module_to_server: Dict[str, str] = {}

    def add_server(self, server_name: str, server_command: List[str]):
        """
        Add a server to the client.

        Args:
            server_name: Name for the server
            server_command: Command to start the server
        """
        self.servers[server_name] = NVRxMCPClient(server_command)

    async def connect_all(self):
        """Connect to all registered servers."""
        for name, client in self.servers.items():
            logger.info(f"Connecting to server: {name}")
            await client.__aenter__()

            # Discover modules on this server
            status = await client.get_status()
            modules = status.get("modules", [])

            for module in modules:
                self.module_to_server[module] = name

            logger.info(f"Server {name} provides modules: {modules}")

    async def disconnect_all(self):
        """Disconnect from all servers."""
        for name, client in self.servers.items():
            logger.info(f"Disconnecting from server: {name}")
            await client.__aexit__(None, None, None)

    async def run_module(self, module_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a module (automatically routes to correct server).

        Args:
            module_name: Name of the module
            input_data: Input data
            **kwargs: Additional configuration

        Returns:
            Module result
        """
        server_name = self.module_to_server.get(module_name)
        if not server_name:
            raise ValueError(f"Module '{module_name}' not found on any server")

        client = self.servers[server_name]
        return await client.run_module(module_name, **kwargs)

    async def run_cross_server_pipeline(
        self, modules: List[str], module_configs: Dict[str, Dict[str, Any]], input_data: Any
    ) -> Dict[str, Any]:
        """
        Run a pipeline that spans multiple servers with concurrent execution.

        Modules without dependencies run concurrently. Modules with dependencies
        wait for their dependencies to complete before starting.

        Args:
            modules: List of module names
            module_configs: Configuration for each module
            input_data: Initial input data

        Returns:
            Pipeline execution result
        """
        results = {}
        running_tasks = {}  # module_name -> asyncio.Task
        completed_modules = set()
        stop_requested = False

        # Build dependency graph
        dependencies = {}
        for module_name in modules:
            config = module_configs.get(module_name, {})
            module_deps = config.get("dependencies", {})
            dependencies[module_name] = list(module_deps.values()) if module_deps else []

        async def run_module_with_deps(module_name: str):
            """Run a module after waiting for its dependencies."""
            nonlocal stop_requested

            # Wait for dependencies to complete
            module_deps = dependencies.get(module_name, [])
            if module_deps:
                logger.info(f"Module '{module_name}' waiting for dependencies: {module_deps}")

                # Wait for all dependency tasks to complete
                for dep_name in module_deps:
                    if dep_name in running_tasks:
                        await running_tasks[dep_name]
                    elif dep_name not in completed_modules:
                        raise ValueError(
                            f"Dependency '{dep_name}' not scheduled for module '{module_name}'"
                        )

                # Check if any dependency resulted in STOP
                for dep_name in module_deps:
                    if results.get(dep_name, {}).get("state") == "STOP":
                        logger.info(
                            f"Module '{module_name}' skipped due to STOP from dependency '{dep_name}'"
                        )
                        return None

            # Check if stop was requested
            if stop_requested:
                logger.info(f"Module '{module_name}' skipped due to previous STOP")
                return None

            # Prepare input data
            config = module_configs.get(module_name, {})
            module_input = input_data.copy() if isinstance(input_data, dict) else input_data

            # Inject dependency results
            config_deps = config.get("dependencies", {})
            if config_deps:
                for dep_key, dep_module_name in config_deps.items():
                    if dep_module_name in results:
                        if isinstance(module_input, dict):
                            module_input[dep_key] = results[dep_module_name].get("result")
                        else:
                            module_input = results[dep_module_name].get("result")

            # Run the module
            logger.info(f"Running module: {module_name}")
            try:
                result = await self.run_module(module_name, **config)
                results[module_name] = result
                completed_modules.add(module_name)

                # Check for stop condition
                if result.get("state") == "STOP":
                    logger.info(f"Pipeline stopped at module: {module_name}")
                    stop_requested = True

                return result
            except Exception as e:
                logger.error(f"Error running module '{module_name}': {e}", exc_info=True)
                results[module_name] = {"error": str(e), "state": "STOP"}
                stop_requested = True
                raise

        # Launch all modules (they'll wait for their dependencies internally)
        logger.info(f"Starting concurrent pipeline execution for modules: {modules}")
        logger.info(f"Dependency graph: {dependencies}")

        for module_name in modules:
            task = asyncio.create_task(run_module_with_deps(module_name))
            running_tasks[module_name] = task

        # Wait for all tasks to complete (or fail)
        try:
            await asyncio.gather(*running_tasks.values(), return_exceptions=False)
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # Cancel any still-running tasks
            for module_name, task in running_tasks.items():
                if not task.done():
                    logger.info(f"Cancelling module: {module_name}")
                    task.cancel()

        # Determine final result (last completed module in original order)
        final_result = input_data
        for module_name in reversed(modules):
            if module_name in results and "result" in results[module_name]:
                final_result = results[module_name]["result"]
                break

        return {
            "modules": modules,
            "results": results,
            "final_result": final_result,
            "completed_modules": list(completed_modules),
            "dependencies": dependencies,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()


async def run_distributed_attribution(
    servers: Dict[str, List[str]],
    modules: List[str],
    module_configs: Dict[str, Dict[str, Any]],
    input_data: Any,
) -> Dict[str, Any]:
    """
    Convenience function to run a distributed attribution pipeline.

    Args:
        servers: Dict mapping server names to their commands
        modules: List of modules to execute
        module_configs: Configuration for each module
        input_data: Initial input data

    Returns:
        Pipeline execution result
    """
    client = MultiServerClient()

    for server_name, server_command in servers.items():
        client.add_server(server_name, server_command)

    async with client:
        return await client.run_cross_server_pipeline(modules, module_configs, input_data)
