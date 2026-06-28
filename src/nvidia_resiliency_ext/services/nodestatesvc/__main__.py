#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for nvidia_resiliency_ext.services.nodestatesvc."""

from __future__ import annotations

import argparse
import logging
import os

from .server import DEFAULT_HOST, DEFAULT_PORT, NodeStateService, run_server
from .slurm import SlurmNodeStateClient


def _default_int(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise SystemExit(
            f"Invalid value for {env_name!r}: expected an integer, got {value!r}"
        ) from None


def _default_float(env_name: str, default: float) -> float:
    value = os.environ.get(env_name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise SystemExit(
            f"Invalid value for {env_name!r}: expected a float, got {value!r}"
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Slurm node state for NVRx rendezvous restart policy",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("NVRX_NODESTATESVC_HOST", DEFAULT_HOST),
        help=f"HTTP bind host (env: NVRX_NODESTATESVC_HOST, default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_default_int("NVRX_NODESTATESVC_HTTP_PORT", DEFAULT_PORT),
        help=("HTTP bind port " f"(env: NVRX_NODESTATESVC_HTTP_PORT, default: {DEFAULT_PORT})"),
    )
    parser.add_argument(
        "--slurm-timeout",
        type=float,
        default=_default_float("NVRX_NODESTATESVC_SLURM_TIMEOUT", 30.0),
        help="Timeout in seconds for each Slurm command",
    )
    parser.add_argument(
        "--slurm-batch-size",
        type=int,
        default=_default_int("NVRX_NODESTATESVC_SLURM_BATCH_SIZE", 512),
        help="Maximum nodes per sinfo command",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("NVRX_NODESTATESVC_LOG_LEVEL", "INFO"),
        help="Python logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    service = NodeStateService(
        SlurmNodeStateClient(
            timeout=args.slurm_timeout,
            batch_size=args.slurm_batch_size,
        )
    )
    server = run_server(args.host, args.port, service)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
