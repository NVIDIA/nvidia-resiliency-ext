#!/usr/bin/env python3
"""
Example: Single MCP Server with Multiple Attribution Modules

This example demonstrates:
1. Starting an MCP server with all attribution modules
2. Calling individual modules
3. Running a pipeline of modules
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

        # 3. Run log analyzer
        logger.info("\n3. Running Log Analyzer:")
        logger.info("-" * 80)
        log_result = await client.run_module(
            module_name="log_analyzer",
            log_path=args.log_path,
            model="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
            temperature=0.2,
            exclude_nvrx_logs=True,
            top_p=0.7,
            max_tokens=8192,
        )
        logger.info(f"Result preview: {str(log_result)[:200]}...")

        # 4. Run FR analyzer
        logger.info("\n4. Running FR Analyzer:")
        logger.info("-" * 80)
        fr_result = await client.run_module(
            module_name="fr_analyzer",
            fr_path=args.fr_path,
            model="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            verbose=True,
            scheduling_order_file="TP->PP->DP",
            health_check=True,
            llm_analyze=False,
            pattern="_dump_*",
        )
        logger.info(f"Result preview: {str(fr_result)[:200]}...")

        # 5. Run combined_log_fr with cached results
        logger.info("\n5. Running Combined Analysis with Cached Results:")
        logger.info("-" * 80)

        # Extract the actual result data from the previous runs
        log_analysis_result = (
            log_result.get("result") if isinstance(log_result, dict) else log_result
        )
        fr_analysis_result = fr_result.get("result") if isinstance(fr_result, dict) else fr_result

        # Run combined_log_fr with the cached results
        combined_result = await client.run_module(
            module_name="combined_log_fr",
            input_data=[
                (log_analysis_result, log_result["state"]),
                (fr_analysis_result, fr_result["state"]),
            ],
            model="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
            threshold=5,
        )
        logger.info(f"Combined Result: {combined_result}")
        logger.info(f"Combined Result ID: {combined_result.get('result_id')}")
        logger.info(
            f"Combined Result: {combined_result.get('result')[:500]}..."
            if isinstance(combined_result.get("result"), str)
            else combined_result
        )

        # 6. List and access cached resources
        logger.info("\n6. Cached Resources:")
        logger.info("-" * 80)
        resources = await client.list_resources()
        logger.info(f"Number of cached results: {len(resources)}")
        for resource in resources[:3]:  # Show first 3
            logger.info(f"  - {resource['uri']}: {resource['name']}")

        # 7. Retrieve a specific cached result
        if resources:
            logger.info("\n7. Retrieving Cached Result:")
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
        description="Single MCP Server with Multiple Attribution Modules"
    )
    parser.add_argument("--log-path", type=str, help="Path to log file")
    parser.add_argument("--fr-path", type=str, help="Path to FR dumps")
    args = parser.parse_args()

    asyncio.run(main(args))
