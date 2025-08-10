# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
NVRx Log Aggregator Service

This module provides a standalone log aggregator service that can run independently
of training processes. It's designed to be used in SLURM environments where the
aggregator runs in step 0 and training processes run in step 1.

The service monitors a shared temporary directory for log messages from training
processes and aggregates them into per-node log files.

Usage:
    # Start the aggregator service (step 0 in slurm)
    python -m nvidia_resiliency_ext.shared_utils.log_aggregator \
        --log-dir /path/to/logs \
        --temp-dir /path/to/temp \
        --wait-file /path/to/shutdown.signal

    # In training processes (step 1 in slurm)
    export NVRX_LOG_DIR=/path/to/logs
    export NVRX_LOG_TEMP_DIR=/path/to/temp
    export NVRX_LOG_AGGREGATOR=1
    ft_launcher ... your_training_script.py
"""

import argparse
import os
import time
from nvidia_resiliency_ext.shared_utils.log_distributed import NodeLogAggregator
import nvidia_resiliency_ext.shared_utils.log_manager as LogMgr


def main():
    """Main function for running the log aggregator as a separate service."""
    parser = argparse.ArgumentParser(description="NVRx Log Aggregator Service")
    parser.add_argument("--log-dir", help="Directory for log files")
    parser.add_argument("--temp-dir", help="Directory for temporary files)")
    parser.add_argument("--en_chrono_ord", help="Enable Chronological Ordering")
    parser.add_argument(
        "--wait-file",
        required=True,
        help="File to wait for before shutting down (required to keep service running)",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=1.0,
        help="Interval in seconds to check for shutdown file (default: 1.0)",
    )

    args = parser.parse_args()

    log_dir = LogMgr.get_log_dir(args.log_dir)
    log_file = LogMgr.get_log_file()
    temp_dir = LogMgr.get_temp_dir(args.temp_dir)
    max_file_size_kb = LogMgr.get_max_file_size_kb()
    en_chrono_ord = LogMgr.get_en_chrono_ord()

    if log_dir is None:
        raise RuntimeError("Log directory must be set for log aggregator service")
    print(log_dir)
    print(log_file)
    print(f"Starting NVRx Log Aggregator Service")
    print(f"  Log Path: {os.path.join(log_dir, log_file)}")
    print(f"  Temp directory: {temp_dir}")
    print(f"  en_chrono_ord: {en_chrono_ord}")

    aggregator = NodeLogAggregator(
        log_dir=log_dir,
        temp_dir=temp_dir,
        log_file=log_file,
        max_file_size_kb=max_file_size_kb,
        en_chrono_ord=en_chrono_ord,
    )
    aggregator.start_aggregator()
    print("Log aggregator service is running...")

    # Wait for shutdown file
    print(f"Waiting for shutdown file: {args.wait_file}")
    while not os.path.exists(args.wait_file):
        time.sleep(args.check_interval)
    print("Shutdown file detected")

    # Shutdown gracefully
    print("Shutting down log aggregator service...")
    aggregator.shutdown()
    print("Log aggregator service stopped")


if __name__ == "__main__":
    main()
