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
of training processes. The service monitors a shared temporary directory for log messages
from training processes and aggregates them into per-node log files.

Example sbatch Usage:
    export NVRX_DIST_LOG_DIR=/tmp/nvrx
    NVRX_REPO=/../nvidia-resiliency-ext:/nvrx_repo

    # all node setup, if installing from source
    srun \
        bash -c '
            echo "export NVRX_DIST_LOG_DIR=$NVRX_DIST_LOG_DIR" >> /tmp/.myenv_${SLURM_JOB_ID}.sh
            cd /nvrx_repo && pip install -e .
        '
    # main workload with aggregator
    srun \
        bash -c '
          source /tmp/.myenv_${SLURM_JOB_ID}.sh
          if [[ $SLURM_LOCALID -eq 0 ]]; then
            cd /nvrx_repo && PYTHONPATH=./src:$PYTHONPATH \
                python src/nvidia_resiliency_ext/shared_utils/log_aggregator.py \
                    --wait-file ./stop \
                    --log-dir /logs/slurm/${SLURM_JOB_ID} &
          fi
          $LAUNCHER_CMD $LAUNCHER_ARGS $WORKLOAD_CMD $WORKLOAD_ARGS
          touch /nvrx_repo/stop
        '
"""

import argparse
import os
import time
from nvidia_resiliency_ext.shared_utils.log_distributed import NodeLogAggregator
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig


def main():
    """Main function for running the log aggregator as a separate service."""
    parser = argparse.ArgumentParser(description="NVRx Log Aggregator Service")
    parser.add_argument("--log-dir", help="Directory for log files")
    parser.add_argument(
        "--en-chronological-ordering", action="store_true", help="Enable Chronological Ordering"
    )
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

    log_dir = args.log_dir
    log_file = LogConfig.get_log_file()
    dist_log_dir = LogConfig.get_dist_log_dir()
    max_file_size = LogConfig.get_max_file_size()
    en_chrono_ord = args.en_chronological_ordering

    if log_dir is None:
        raise RuntimeError("Log directory must be set for log aggregator service")
    print(f"Starting NVRx Log Aggregator Service")
    print(f"  Log Path: {os.path.join(log_dir, log_file)}")
    print(f"  Temp directory: {dist_log_dir}")
    print(f"  en_chronological_ordering: {en_chrono_ord}")

    aggregator = NodeLogAggregator(
        log_dir=log_dir,
        temp_dir=dist_log_dir,
        log_file=log_file,
        max_file_size=max_file_size,
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
