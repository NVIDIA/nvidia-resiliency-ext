#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI entry point for nvidia_resiliency_ext.services.smonsvc.

Usage:
    python -m nvidia_resiliency_ext.services.smonsvc --attrsvc-endpoint http://localhost:8000 --port 8100
    python -m nvidia_resiliency_ext.services.smonsvc --attrsvc-endpoint unix:///tmp/nvrx-attrsvc.sock --port 8100
"""

import argparse
import logging
import os

from .monitor import SlurmJobMonitor
from .status_server import DEFAULT_STATUS_HOST

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
  %(prog)s --attrsvc-endpoint http://localhost:8000 --port 8100

  # Monitor through attrsvc HTTP-over-UDS
  %(prog)s --attrsvc-endpoint unix:///tmp/nvrx-attrsvc.sock --port 8100

  # Monitor specific partitions
  %(prog)s --partitions gpu interactive --attrsvc-endpoint http://localhost:8000 --port 8100

  # Monitor specific user's jobs
  %(prog)s --user foo_bar --attrsvc-endpoint http://attrsvc.cluster.local:8000 --port 8100

  # Monitor jobs matching a pattern
  %(prog)s --job-pattern "train_.*" --attrsvc-endpoint http://localhost:8000 --port 8100

  # Run both services via SLURM
  sbatch scripts/nvrx_services.sbatch
        """,
    )

    # Environment variable prefix: NVRX_SMONSVC_
    default_attrsvc_endpoint = os.environ.get("NVRX_ATTRSVC_ENDPOINT", "http://localhost:8000")

    parser.add_argument(
        "--attrsvc-endpoint",
        default=default_attrsvc_endpoint,
        dest="attrsvc_url",
        help=(
            "Attribution service URL or UDS endpoint "
            "(env: NVRX_ATTRSVC_ENDPOINT, default: http://localhost:8000)"
        ),
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
        "--host",
        default=os.environ.get("NVRX_SMONSVC_HOST", DEFAULT_STATUS_HOST),
        help=(
            "Host/interface for HTTP status server "
            f"(env: NVRX_SMONSVC_HOST, default: {DEFAULT_STATUS_HOST})"
        ),
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
        logging.getLogger("nvidia_resiliency_ext.services.smonsvc").setLevel(logging.DEBUG)
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
        host=args.host,
    ) as monitor:
        monitor.run()


if __name__ == "__main__":
    main()
