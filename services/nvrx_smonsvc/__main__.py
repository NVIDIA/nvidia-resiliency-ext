#!/usr/bin/env python3
#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
CLI entry point for nvrx_smonsvc.

Usage:
    python -m nvrx_smonsvc --attrsvc-url http://localhost:8000 --port 8100
"""

import argparse
import logging
import os

from .monitor import SlurmJobMonitor

# Configure logging (read level from env, default to INFO)
_log_level_name = os.environ.get("NVRX_SMONSVC_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress verbose httpcore/httpx debug logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor SLURM jobs and integrate with attribution service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor jobs in default partitions (batch, batch_long)
  %(prog)s --attrsvc-url http://localhost:8000 --port 8100

  # Monitor specific partitions
  %(prog)s --partitions gpu interactive --attrsvc-url http://localhost:8000 --port 8100

  # Monitor specific user's jobs
  %(prog)s --user foo_bar --attrsvc-url http://attrsvc.cluster.local:8000 --port 8100

  # Monitor jobs matching a pattern
  %(prog)s --job-pattern "train_.*" --attrsvc-url http://localhost:8000 --port 8100

  # Run both services via SLURM
  sbatch scripts/nvrx_services.sbatch
        """,
    )

    # Environment variable prefix: NVRX_SMONSVC_
    parser.add_argument(
        "--attrsvc-url",
        default=os.environ.get("NVRX_ATTRSVC_URL", "http://localhost:8000"),
        help="Attribution service URL (env: NVRX_ATTRSVC_URL, default: http://localhost:8000)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("NVRX_SMONSVC_INTERVAL", "180")),
        help="Poll interval in seconds (env: NVRX_SMONSVC_INTERVAL, default: 180)",
    )

    parser.add_argument(
        "--user",
        default=os.environ.get("NVRX_SMONSVC_USER"),
        help="Filter jobs by specific user (env: NVRX_SMONSVC_USER, default: all users)",
    )

    def parse_partitions(val: str | None) -> list | None:
        return val.split() if val else None

    parser.add_argument(
        "--partitions",
        nargs="+",
        default=parse_partitions(os.environ.get("NVRX_SMONSVC_PARTITIONS")),
        help="SLURM partitions to monitor (env: NVRX_SMONSVC_PARTITIONS space-separated, default: batch batch_long)",
    )

    parser.add_argument(
        "--job-pattern",
        default=os.environ.get("NVRX_SMONSVC_JOB_PATTERN"),
        help="Regex pattern to match job names (env: NVRX_SMONSVC_JOB_PATTERN)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("NVRX_SMONSVC_TIMEOUT", "60.0")),
        help="HTTP request timeout in seconds (env: NVRX_SMONSVC_TIMEOUT, default: 60)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=(
            int(os.environ.get("NVRX_SMONSVC_PORT"))
            if os.environ.get("NVRX_SMONSVC_PORT")
            else None
        ),
        help="Port for HTTP server (env: NVRX_SMONSVC_PORT, disabled if not set)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=os.environ.get("NVRX_SMONSVC_VERBOSE", "").lower() in ("true", "1", "yes"),
        help="Enable debug logging (env: NVRX_SMONSVC_VERBOSE)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # If --user is specified, disable all_users mode
    all_users = args.user is None

    with SlurmJobMonitor(
        attrsvc_url=args.attrsvc_url,
        poll_interval=args.interval,
        user=args.user,
        all_users=all_users,
        partitions=args.partitions,
        job_pattern=args.job_pattern,
        timeout=args.timeout,
        port=args.port,
    ) as monitor:
        monitor.run()


if __name__ == "__main__":
    main()
