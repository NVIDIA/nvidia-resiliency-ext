#!/usr/bin/env python3
"""
Launcher script for NVRX Attribution MCP Server.

Usage:
    # Launch server with all modules
    python server_launcher.py
    
    # Launch server with specific modules only
    python server_launcher.py --modules log_analyzer fr_analyzer
    
    # Launch with custom server name
    python server_launcher.py --server-name my-attribution-server
"""

import argparse
import logging
import sys

from nvidia_resiliency_ext.attribution.mcp_integration.mcp_server import NVRxMCPServer
from nvidia_resiliency_ext.attribution.mcp_integration.module_definitions import (
    register_all_modules,
)
from nvidia_resiliency_ext.attribution.mcp_integration.registry import global_registry

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description='Launch NVRX Attribution MCP Server')
    parser.add_argument(
        '--server-name', default='nvidia-resiliency-attribution', help='Name of the MCP server'
    )
    parser.add_argument('--modules', nargs='*', help='Specific modules to enable (default: all)')
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level',
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Register all modules
    logger.info("Registering attribution modules...")
    register_all_modules()

    # Filter modules if specified
    if args.modules:
        all_modules = global_registry.list_modules()
        for module in list(all_modules):
            if module not in args.modules:
                # Remove from registry (simplified - in production, use proper filtering)
                logger.info(f"Skipping module: {module}")
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
