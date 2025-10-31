"""
Registry for NVRX Attribution Modules to enable dynamic discovery and composition.
This allows multiple attribution modules to be registered and invoked via MCP.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Type

from nvidia_resiliency_ext.attribution.base import NVRxAttribution

logger = logging.getLogger(__name__)


@dataclass
class ModuleMetadata:
    """Metadata about an attribution module."""

    name: str
    description: str
    module_class: Type[NVRxAttribution]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    requires_llm: bool = False
    dependencies: List[str] = None  # Other modules this depends on

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

    def to_mcp_tool_schema(self) -> Dict[str, Any]:
        """Convert module metadata to MCP tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class AttributionModuleRegistry:
    """
    Registry for NVRX Attribution modules.

    This enables:
    1. Dynamic module discovery
    2. Module composition and pipelining
    3. MCP tool generation
    4. Inter-module communication
    """

    def __init__(self):
        self._modules: Dict[str, ModuleMetadata] = {}
        self._instances: Dict[str, NVRxAttribution] = {}
        self._results_cache: Dict[str, Any] = {}

    def register(
        self,
        name: str,
        module_class: Type[NVRxAttribution],
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        requires_llm: bool = False,
        dependencies: Optional[List[str]] = None,
    ):
        """Register an attribution module."""
        metadata = ModuleMetadata(
            name=name,
            description=description,
            module_class=module_class,
            input_schema=input_schema,
            output_schema=output_schema,
            requires_llm=requires_llm,
            dependencies=dependencies or [],
        )
        self._modules[name] = metadata

    def get_module_metadata(self, name: str) -> Optional[ModuleMetadata]:
        """Get metadata for a registered module."""
        return self._modules.get(name)

    def list_modules(self) -> List[str]:
        """List all registered module names."""
        return list(self._modules.keys())

    def get_all_metadata(self) -> List[ModuleMetadata]:
        """Get metadata for all registered modules."""
        return list(self._modules.values())

    def create_instance(self, name: str, args: Any) -> NVRxAttribution:
        """Create an instance of a registered module."""
        metadata = self._modules.get(name)
        if not metadata:
            raise ValueError(f"Module '{name}' not registered")

        # Create instance
        instance = metadata.module_class(args)
        self._instances[name] = instance
        return instance

    def get_instance(self, name: str) -> Optional[NVRxAttribution]:
        """Get an existing instance of a module."""
        return self._instances.get(name)

    def apply_defaults(self, module_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values from input_schema to arguments."""
        metadata = self._modules.get(module_name)
        if not metadata:
            return arguments

        # Create a copy to avoid modifying the original
        result = dict(arguments)

        # Get the properties from the input schema
        input_schema = metadata.input_schema
        properties = input_schema.get("properties", {})

        # Apply defaults for missing arguments
        for param_name, param_schema in properties.items():
            if param_name not in result and "default" in param_schema:
                result[param_name] = param_schema["default"]
                logger.debug(f"Applied default for {param_name}: {param_schema['default']}")

        return result

    def cache_result(self, module_name: str, arguments: Dict[str, Any], result: Any):
        """Cache a module execution result."""
        hash_value = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        key = f"{module_name}:{hash_value}"
        self._results_cache[key] = result
        logger.info(f"Caching result: {key}")
        return hash_value

    def get_cached_result_by_uri(self, module_name: str, result_id: str) -> Optional[Any]:
        """Retrieve a cached result by URI."""
        key = f"{module_name}:{result_id}"
        return self._results_cache.get(key)

    def get_cached_result_by_args(
        self, module_name: str, arguments: Dict[str, Any]
    ) -> Optional[Any]:
        """Retrieve a cached result based on module name and arguments."""
        hash_value = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        key = f"{module_name}:{hash_value}"
        logger.info(f"Looking up cached result: {key}")
        return self._results_cache.get(key)

    def clear_cache(self):
        """Clear all cached results."""
        self._results_cache.clear()

    def to_mcp_tools(self) -> List[Dict[str, Any]]:
        """Convert all registered modules to MCP tool schemas."""
        return [metadata.to_mcp_tool_schema() for metadata in self._modules.values()]

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph of all modules."""
        return {name: metadata.dependencies for name, metadata in self._modules.items()}

    def get_execution_order(self, target_modules: List[str]) -> List[str]:
        """
        Get the execution order for a set of modules based on dependencies.
        Uses topological sort.
        """
        from collections import deque

        # Build in-degree map
        in_degree = {name: 0 for name in target_modules}
        graph = {name: [] for name in target_modules}

        for name in target_modules:
            metadata = self._modules.get(name)
            if metadata:
                for dep in metadata.dependencies:
                    if dep in target_modules:
                        graph[dep].append(name)
                        in_degree[name] += 1

        # Topological sort
        queue = deque([name for name in target_modules if in_degree[name] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(target_modules):
            raise ValueError("Circular dependency detected in modules")

        return result


def serialize_result(result: Any) -> str:
    """Serialize attribution result to JSON string."""
    if result is None:
        return json.dumps(None)

    if is_dataclass(result):
        return json.dumps(asdict(result), default=str)

    if hasattr(result, 'to_dict'):
        return json.dumps(result.to_dict(), default=str)

    if hasattr(result, '__dict__'):
        return json.dumps(result.__dict__, default=str)

    if isinstance(result, (dict, list, str, int, float, bool)):
        return json.dumps(result, default=str)

    # Fallback to string representation
    return json.dumps(str(result))


def deserialize_result(result_str: str) -> Any:
    """Deserialize JSON string to result object."""
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        return result_str


# Global registry instance
global_registry = AttributionModuleRegistry()
