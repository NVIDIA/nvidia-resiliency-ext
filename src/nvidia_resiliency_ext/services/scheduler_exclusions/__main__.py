# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI configuration for the reusable scheduler exclusion publisher."""

from __future__ import annotations

import argparse
import logging
import os

from nvidia_resiliency_ext.attribution.scheduler_exclusions import (
    PublisherConfig,
    SchedulerExclusionPublisher,
)
from nvidia_resiliency_ext.attribution.scheduler_exclusions.publisher import positive_integer
from nvidia_resiliency_ext.attribution.scheduler_exclusions.runtime import parent_lifetime, serve

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish scheduler exclusions for one Slurm array")
    parser.add_argument(
        "--once", action="store_true", help="Poll once and exit (0 success, 1 poll failure)"
    )
    parser.add_argument("--parent-pid", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = PublisherConfig.from_env()
        interval = (
            positive_integer(
                "SEGMENT_HEALTH_POLL_INTERVAL",
                os.environ.get("SEGMENT_HEALTH_POLL_INTERVAL") or "300",
            )
            if not args.once
            else 300
        )
    except ValueError as exc:
        parser.error(str(exc))
    with parent_lifetime(args.parent_pid):
        try:
            publisher = SchedulerExclusionPublisher(config)
        except OSError as exc:
            logger.error("Could not initialize publisher: %s", exc)
            return 1
        if args.once:
            return 0 if publisher.poll_once() else 1
        serve(publisher, interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
