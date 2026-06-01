"""
Registry for NVRX Attribution Modules to enable dynamic discovery and composition.
This allows multiple attribution modules to be registered and invoked via MCP.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class MCPModule(Protocol):
    """Registered tool implementation: constructible with call-time args, ``async run(arguments)``."""

    async def run(self, arguments: Dict[str, Any]) -> Any: ...


@dataclass
class ModuleMetadata:
    """Metadata about an attribution module."""

    name: str
    description: str
    module_class: type[MCPModule]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    requires_llm: bool = False
    dependencies: List[str] = field(default_factory=list)  # Other modules this depends on

    def to_mcp_tool_schema(self) -> Dict[str, Any]:
        """Convert module metadata to MCP tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def _mcp_results_cache_max_entries() -> int:
    raw = os.environ.get("NVRX_MCP_RESULTS_CACHE_MAX_ENTRIES", "").strip()
    if not raw:
        return 256
    try:
        v = int(raw, 10)
        return v if v > 0 else 256
    except ValueError:
        return 256


class AttributionModuleRegistry:
    """
    Registry for NVRX Attribution modules.

    This enables:
    1. Dynamic module discovery
    2. Module composition and pipelining
    3. MCP tool generation
    4. Inter-module communication
    """

    def __init__(self, cache_max_entries: Optional[int] = None):
        self._modules: Dict[str, ModuleMetadata] = {}
        self._instances: Dict[str, MCPModule] = {}
        max_e = (
            int(cache_max_entries)
            if cache_max_entries is not None
            else _mcp_results_cache_max_entries()
        )
        self._results_cache_max_entries: int = max_e if max_e > 0 else 256
        # value is (created_at monotonic, payload); when full, drop oldest half by created_at
        self._results_cache: Dict[str, Tuple[float, Any]] = {}

    def register(
        self,
        name: str,
        module_class: type[MCPModule],
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

    def unregister(self, name: str):
        """Unregister a module."""
        if name in self._modules:
            del self._modules[name]
        else:
            raise ValueError(f"Module '{name}' not registered")

    def get_module_metadata(self, name: str) -> Optional[ModuleMetadata]:
        """Get metadata for a registered module."""
        return self._modules.get(name)

    def list_modules(self) -> List[str]:
        """List all registered module names."""
        return list(self._modules.keys())

    def get_all_metadata(self) -> List[ModuleMetadata]:
        """Get metadata for all registered modules."""
        return list(self._modules.values())

    def create_instance(self, name: str, args: Any) -> MCPModule:
        """Create an instance of a registered module."""
        metadata = self._modules.get(name)
        if not metadata:
            raise ValueError(f"Module '{name}' not registered")

        # Create instance
        instance = metadata.module_class(args)
        self._instances[name] = instance
        return instance

    def get_instance(self, name: str) -> Optional[MCPModule]:
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

    def _evict_oldest_half(self) -> None:
        n = len(self._results_cache)
        if n == 0:
            return
        n_drop = max(1, n // 2)
        oldest_first = sorted(self._results_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in oldest_first[:n_drop]:
            del self._results_cache[k]

    def _results_cache_get(self, key: str) -> Optional[Any]:
        item = self._results_cache.get(key)
        if item is None:
            return None
        _, value = item
        return value

    def _results_cache_put(self, key: str, value: Any) -> None:
        if key in self._results_cache:
            del self._results_cache[key]
        if len(self._results_cache) >= self._results_cache_max_entries:
            self._evict_oldest_half()
        self._results_cache[key] = (time.monotonic(), value)

    def list_results_cache_keys(self) -> List[str]:
        """Keys for cached results (MCP resources)."""
        return list(self._results_cache.keys())

    def count_results_cache_entries(self) -> int:
        """Number of cached results."""
        return len(self._results_cache)

    def iter_results_cache_items(self) -> Iterator[Tuple[str, Any]]:
        """Yield (cache_key, result) for each cache entry."""
        for key, (_, payload) in self._results_cache.items():
            yield key, payload

    def result_id_for_args(self, arguments: Dict[str, Any]) -> str:
        """Stable result ID for a module invocation argument payload."""
        return hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()

    def cache_result_by_id(self, module_name: str, result_id: str, result: Any) -> None:
        """Cache a module execution result under a known result ID."""
        key = f"{module_name}:{result_id}"
        self._results_cache_put(key, result)
        logger.info(f"Caching result: {key}")

    def cache_result(self, module_name: str, arguments: Dict[str, Any], result: Any) -> str:
        """Cache a module execution result."""
        result_id = self.result_id_for_args(arguments)
        self.cache_result_by_id(module_name, result_id, result)
        return result_id

    def get_cached_result_by_id(self, result_id: str) -> Optional[Any]:
        """Retrieve a cached result by exact result ID, regardless of module."""
        for key, payload in self.iter_results_cache_items():
            _module_name, cached_result_id = key.split(":", 1)
            if cached_result_id == result_id:
                return payload
        return None

    def get_cached_result_by_uri(self, module_name: str, result_id: str) -> Optional[Any]:
        """Retrieve a cached result by URI."""
        key = f"{module_name}:{result_id}"
        return self._results_cache_get(key)

    def get_cached_result_by_args(
        self, module_name: str, arguments: Dict[str, Any]
    ) -> Optional[Any]:
        """Retrieve a cached result based on module name and arguments."""
        hash_value = self.result_id_for_args(arguments)
        key = f"{module_name}:{hash_value}"
        logger.info(f"Looking up cached result: {key}")
        return self._results_cache_get(key)

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
    return json.dumps(_jsonable_result(result), default=str)


def _jsonable_result(result: Any) -> Any:
    if result is None:
        return None

    if is_dataclass(result):
        return _jsonable_result(asdict(result))

    if isinstance(result, Enum):
        return result.name

    if hasattr(result, 'to_dict'):
        return _jsonable_result(result.to_dict())

    if hasattr(result, '__dict__'):
        return _jsonable_result(result.__dict__)

    if isinstance(result, dict):
        return {str(key): _jsonable_result(value) for key, value in result.items()}

    if isinstance(result, (list, tuple)):
        return [_jsonable_result(value) for value in result]

    if isinstance(result, (str, int, float, bool)):
        return result

    return str(result)


def deserialize_result(result_str: str) -> Any:
    """Deserialize JSON string to result object."""
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        return result_str


# Global registry instance
global_registry = AttributionModuleRegistry()
