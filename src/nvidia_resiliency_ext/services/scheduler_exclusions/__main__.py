# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the NVRx Scheduler Exclusion Service."""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from .config import SchedulerExclusionServiceSettings
from .monitor import SchedulerExclusionMonitor
from .server import SchedulerExclusionHttpServer


def _parser(defaults: SchedulerExclusionServiceSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NVRx Scheduler Exclusion Service")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--slurm-bin-dir", default=defaults.slurm_bin_dir)
    parser.add_argument("--slurm-conf", default=defaults.slurm_conf)
    parser.add_argument(
        "--output-dir",
        dest="scheduler_exclusion_dir",
        default=defaults.scheduler_exclusion_dir,
        help="Shared directory in which to publish scheduler-exclusion artifacts.",
    )
    parser.add_argument(
        "--refresh-interval-seconds",
        type=float,
        default=defaults.refresh_interval_seconds,
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=float,
        default=defaults.cache_ttl_seconds,
    )
    parser.add_argument(
        "--query-timeout-seconds",
        type=float,
        default=defaults.query_timeout_seconds,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    defaults = SchedulerExclusionServiceSettings.from_env()
    args = _parser(defaults).parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = SchedulerExclusionServiceSettings(
        host=args.host,
        port=args.port,
        slurm_bin_dir=args.slurm_bin_dir,
        slurm_conf=args.slurm_conf,
        scheduler_exclusion_dir=args.scheduler_exclusion_dir,
        refresh_interval_seconds=args.refresh_interval_seconds,
        cache_ttl_seconds=args.cache_ttl_seconds,
        query_timeout_seconds=args.query_timeout_seconds,
    )
    monitor = SchedulerExclusionMonitor(settings.monitor_config())
    server = SchedulerExclusionHttpServer((settings.host, settings.port), monitor)
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    server.timeout = 0.5
    monitor.start()
    host, port = server.server_address[:2]
    logging.getLogger(__name__).info("Listening on http://%s:%s", host, port)
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        server.server_close()
        monitor.stop()


if __name__ == "__main__":
    main()
