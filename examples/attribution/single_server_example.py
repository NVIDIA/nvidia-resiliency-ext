#!/usr/bin/env python3
"""
Example: Single MCP Server with Packaged Attribution Modules

This example demonstrates:
1. Starting an MCP server with packaged restart-agent and FR modules
2. Calling the restart-agent log analyzer
3. Optionally calling the Flight Recorder analyzer
4. Accessing cached results
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nvidia_resiliency_ext.attribution.mcp_integration.mcp_client import NVRxMCPClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main(args: argparse.Namespace):
    """Run examples with a single MCP server."""

    # Server command - adjust path as needed
    server_command = [
        "python",
        "src/nvidia_resiliency_ext/attribution/mcp_integration/server_launcher.py",
        "--modules",
        "restart_agent",
        "fr_analyzer",
    ]

    logger.info("=" * 80)
    logger.info("NVRX Attribution MCP Integration - Single Server Example")
    logger.info("=" * 80)
    logger.info(f"Server command: {server_command}")
    # Connect to the server
    client = NVRxMCPClient(server_command)
    async with client as client:

        # 1. Get server status
        logger.info("\n1. Server Status:")
        logger.info("-" * 80)
        status = await client.get_status()
        logger.info(json.dumps(status, indent=2))

        # 2. List available tools
        logger.info("\n2. Available Tools:")
        logger.info("-" * 80)
        tools = await client.list_tools()
        for tool in tools:
            logger.info(f"  - {tool['name']}: {tool['description']}")

        # 3. Run restart-agent log analyzer
        logger.info("\n3. Running Restart Agent:")
        logger.info("-" * 80)
        log_args = {
            "log_path": args.log_path,
            "timeout_seconds": args.timeout_seconds,
        }
        if args.job_id is not None:
            log_args["job_id"] = args.job_id
        if args.cycle_id is not None:
            log_args["cycle_id"] = args.cycle_id
        log_result = await client.run_module(module_name="restart_agent", **log_args)
        logger.info(f"Result preview: {str(log_result)[:200]}...")

        # 4. Optionally run FR analyzer
        if args.fr_path:
            logger.info("\n4. Running FR Analyzer:")
            logger.info("-" * 80)
            fr_result = await client.run_module(
                module_name="fr_analyzer",
                fr_path=args.fr_path,
                verbose=True,
                health_check=True,
                pattern="_dump_*",
            )
            logger.info(f"Result preview: {str(fr_result)[:200]}...")
        else:
            logger.info("\n4. Skipping FR Analyzer: --fr-path not provided")

        # 5. List and access cached resources
        logger.info("\n5. Cached Resources:")
        logger.info("-" * 80)
        resources = await client.list_resources()
        logger.info(f"Number of cached results: {len(resources)}")
        for resource in resources[:3]:  # Show first 3
            logger.info(f"  - {resource['uri']}: {resource['name']}")

        # 6. Retrieve a specific cached result
        if resources:
            logger.info("\n6. Retrieving Cached Result:")
            logger.info("-" * 80)
            uri = resources[0]["uri"]
            cached_result = await client.read_resource(uri)
            logger.info(f"Retrieved from {uri}")
            logger.info(f"Content preview: {str(cached_result)[:200]}...")

        logger.info("\n" + "=" * 80)
        logger.info("Example completed successfully!")
        logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single MCP Server with Packaged Attribution Modules"
    )
    parser.add_argument("--log-path", type=str, required=True, help="Absolute path to log file")
    parser.add_argument("--fr-path", type=str, help="Path to FR dumps")
    parser.add_argument("--job-id", type=str, help="Optional scheduler/job identifier")
    parser.add_argument("--cycle-id", type=int, help="Optional restart cycle identifier")
    parser.add_argument("--timeout-seconds", type=float, default=240.0, help="Analysis timeout")
    args = parser.parse_args()

    asyncio.run(main(args))
