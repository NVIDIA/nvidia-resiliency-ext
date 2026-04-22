"""MCP integration for NVRX Attribution modules."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from nvidia_resiliency_ext.attribution._optional import reraise_if_missing_attribution_dependency

if TYPE_CHECKING:
    from .mcp_client import NVRxMCPClient, create_mcp_client, get_server_command
    from .mcp_server import NVRxMCPServer
    from .registry import AttributionModuleRegistry
    from .transport_errors import is_mcp_connection_error

_EXPORTS = {
    "NVRxMCPServer": ".mcp_server",
    "NVRxMCPClient": ".mcp_client",
    "AttributionModuleRegistry": ".registry",
    "get_server_command": ".mcp_client",
    "create_mcp_client": ".mcp_client",
    "is_mcp_connection_error": ".transport_errors",
}

__all__ = [
    "NVRxMCPServer",
    "NVRxMCPClient",
    "AttributionModuleRegistry",
    "get_server_command",
    "create_mcp_client",
    "is_mcp_connection_error",
]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        reraise_if_missing_attribution_dependency(
            exc,
            feature=f"{__name__}.{name}",
        )
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
