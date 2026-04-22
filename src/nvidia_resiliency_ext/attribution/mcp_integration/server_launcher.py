#!/usr/bin/env python3
"""
Launcher for the NVRX Attribution MCP server (tool-agnostic: clients call ``log_analyzer``,
``fr_analyzer``, etc. by MCP tool name).

Installed as ``nvrx-mcp-analysis`` for a short ``ps`` name; process title matches via
``setproctitle`` when available.

Usage:
    # Launch server with all modules
    nvrx-mcp-analysis
    python server_launcher.py

    # Launch with specific modules only
    nvrx-mcp-analysis --modules log_analyzer fr_analyzer

    # Launch with custom server name
    nvrx-mcp-analysis --server-name my-attribution-server
"""

import argparse
import logging
import sys

from nvidia_resiliency_ext.attribution._optional import reraise_if_missing_attribution_dependency

_PROC_TITLE = "nvrx-mcp-analysis"


def _set_process_title(title: str) -> None:
    """Set argv-style process name for ps/top (Linux/macOS; no-op if setproctitle unavailable)."""
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle(title)
    except Exception:
        pass


logger = logging.getLogger(__name__)


def main():
    """Main entry point for the MCP server."""
    _set_process_title(_PROC_TITLE)

    try:
        from nvidia_resiliency_ext.attribution.mcp_integration.mcp_server import NVRxMCPServer
        from nvidia_resiliency_ext.attribution.mcp_integration.module_definitions import (
            register_all_modules,
        )
        from nvidia_resiliency_ext.attribution.mcp_integration.registry import global_registry
    except ModuleNotFoundError as exc:
        reraise_if_missing_attribution_dependency(
            exc,
            feature="nvrx-mcp-analysis",
        )
        raise

    parser = argparse.ArgumentParser(description='Launch NVRX Attribution MCP Server')
    parser.add_argument(
        '--server-name', default='nvidia-resiliency-attribution', help='Name of the MCP server'
    )
    parser.add_argument('--modules', nargs='*', help='Specific modules to enable (default: all)')
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level',
    )

    args = parser.parse_args()

    _level = getattr(logging, args.log_level)
    if not logging.root.handlers:
        logging.basicConfig(
            level=_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
    else:
        logging.getLogger("nvidia_resiliency_ext").setLevel(_level)

    # Register all modules
    logger.info("Registering attribution modules...")
    register_all_modules()

    # Filter modules if specified
    if args.modules:
        all_modules = global_registry.list_modules()
        for module in list(all_modules):
            if module not in args.modules:
                global_registry.unregister(module)
                logger.info(f"Unregistered module: {module}")
        logger.info(f"Enabled modules: {args.modules}")
    else:
        logger.info(f"Enabled modules: {global_registry.list_modules()}")

    # Create and run server
    server = NVRxMCPServer(registry=global_registry, server_name=args.server_name)

    logger.info(f"Starting server: {args.server_name}")

    try:
        server.run_sync()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
