#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone example of NodeHealthCheck usage.

Usage:
    # With default socket path
    python examples/node_health_check_example.py

    # With custom socket path
    python examples/node_health_check_example.py --socket-path /var/run/nvhcd/nvhcd.sock
"""

import argparse
import logging
import sys

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Example of NodeHealthCheck usage')
    parser.add_argument(
        '--socket-path',
        type=str,
        default='/var/run/nvhcd/nvhcd.sock',
        help='Path to the Unix domain socket (default: /var/run/nvhcd/nvhcd.sock)',
    )
    args = parser.parse_args()

    # Import here to allow running from project root with PYTHONPATH
    try:
        from nvidia_resiliency_ext.shared_utils.health_check import NodeHealthCheck
    except ImportError:
        logger.error(
            "Failed to import NodeHealthCheck. "
            "Make sure to run with PYTHONPATH=./src or install the package."
        )
        sys.exit(1)

    # Create NodeHealthCheck instance
    logger.info("Creating NodeHealthCheck...")
    logger.info(f"  socket_path: {args.socket_path}")
    checker = NodeHealthCheck(
        socket_path=args.socket_path,
    )

    # Check if gRPC is available
    if checker._grpc is None:
        logger.error("gRPC module not available. Install grpcio package.")
        sys.exit(1)

    # Check channel target
    logger.info(f"  channel_target: {checker._channel_target}")
    if checker._channel_target is None:
        logger.warning("No valid channel target. Health check will be skipped.")

    # Perform health check
    logger.info("Performing health check...")
    result = checker._perform_health_check()

    # Report result
    if result:
        logger.info("Health check PASSED")
        print("\n✓ Health check PASSED")
        sys.exit(0)
    else:
        logger.error("Health check FAILED")
        print("\n✗ Health check FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
