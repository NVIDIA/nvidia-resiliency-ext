# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CLI: ``python -m nvrx_watch``.

Exit codes are what a cron wrapper or systemd unit keys off:
    0  pass completed, nothing critical
    1  degraded -- a source could not be observed; no heartbeat was sent
    2  at least one critical finding
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from . import config as config_module
from . import detectors, parsing
from . import platform as platform_module
from . import runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nvrx-watch",
        description="Watch an NVRx run from outside the job: chain reconciliation and "
        "restart-anomaly detection.",
        epilog="Every flag also reads from NVRX_WATCH_<FLAG> in the environment. "
        "Given a job_id, job name, owner and work dir are read from Slurm; --job-name / "
        "--work-dir override that (and are required under --platform none).",
    )
    parser.add_argument(
        "job_id",
        nargs="?",
        help="a Slurm job id of the chain (any generation). Its job name, owner and "
        "work dir are resolved from Slurm.",
    )
    parser.add_argument("--config", help="JSON config file; CLI flags and env override it")
    parser.add_argument("--job-name", help="override the job name (else derived from job_id)")
    parser.add_argument("--work-dir", help="override NVRX_WORK_DIR (else derived from job_id)")
    parser.add_argument("--cycle-info-glob", help="override the cycle-info file glob")
    parser.add_argument("--checkpoint-iteration-file", help="override the checkpoint file path")
    parser.add_argument(
        "--platform",
        choices=("slurm", "none"),
        help="'none' disables chain reconciliation and watches cycle infos only",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        help="ft_launcher --max-restarts, for the restart-budget detector",
    )
    parser.add_argument("--state-dir", help="watcher state and log directory")
    parser.add_argument(
        "--user", help="scope squeue/sacct to this owner (else derived from job_id)"
    )
    parser.add_argument(
        "--act",
        dest="observe_only",
        action="store_false",
        default=None,
        help="take corrective actions (owner mode). Default is observe-only: detect and "
        "page but never touch a job -- for SREs monitoring runs they do not own.",
    )
    parser.add_argument("--disable", help="comma-separated detector names to skip")
    parser.add_argument("--heartbeat-url", help="dead-man URL, pinged after each observing pass")
    parser.add_argument("--pd-routing-key", help="PagerDuty Events v2 routing key")
    parser.add_argument("--webhook-url", help="generic JSON webhook for findings")
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        help="seconds before a still-present finding pages again (default 3600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="decide and log, change nothing and page nobody (start here)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="run continuously every N seconds instead of a single pass",
    )
    parser.add_argument("--list-detectors", action="store_true", help="print detectors and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def configure_logging(log_file: str, verbose: bool, dry_run: bool) -> None:
    prefix = "[DRY] " if dry_run else ""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    except OSError as exc:  # a read-only home must not stop the pass
        print(f"warning: cannot open log file {log_file}: {exc}", file=sys.stderr)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=f"%(asctime)s {prefix}%(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_detectors:
        for detector in detectors.ALL:
            print(f"{detector.name:24s} requires: {', '.join(detector.requires)}")
        return 0

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in ("config", "interval", "list_detectors", "verbose") and value is not None
    }
    try:
        config = config_module.load(config_file=args.config, overrides=overrides)
    except (OSError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    configure_logging(config.log_file, args.verbose, config.dry_run)
    plat = platform_module.create(config.platform, timeout=config.command_timeout, user=config.user)

    # Resolve a job id into job name and owner -- so the whole invocation is
    # `nvrx-watch <job_id>`. The watcher then monitors by *name*: squeue and the cycle-info
    # glob span every generation, so it follows the singleton chain automatically as
    # generations turn over. The id only bootstraps the identity; explicit flags still win.
    if config.platform == "slurm":
        if not config.job_id:
            print("a job_id is required (or use --platform none with --work-dir).", file=sys.stderr)
            return 1
        desc = plat.describe_job(config.job_id)
        if desc is None:
            print(
                f"could not resolve job id {config.job_id} via scontrol/sacct; pass --job-name "
                "and --work-dir explicitly, or check the id.",
                file=sys.stderr,
            )
            return 1
        if "job_name" not in overrides and desc.job_name:
            config.job_name = desc.job_name
        if "user" not in overrides and desc.user:
            config.user = desc.user
        plat = platform_module.create(
            config.platform, timeout=config.command_timeout, user=config.user
        )

        # Candidate ids to read run details from: the given id, then whatever generation is
        # live now (the given id may have aged out; the batch script is identical across a
        # chain's generations).
        candidates = [config.job_id]
        try:
            candidates += [g.gen_id for g in plat.list_generations(config.job_name)]
        except platform_module.PlatformError:
            pass

        # Read the actual --ft-cycle-info-dir / --ft-checkpoint-iteration-file from the batch
        # script, so nvrx-watch works for any InJob sbatch. Resolution falls through (returns
        # None) when the script is unreadable or its paths come from an env not defined in the
        # script; then the caller should pass --work-dir and the work-dir layout is used.
        want_cyc = "cycle_info_glob" not in overrides
        want_ckpt = "checkpoint_iteration_file" not in overrides
        if want_cyc or want_ckpt:
            for jid in candidates:
                script = plat.batch_script(jid)
                if not script:
                    continue
                # script_path roots any libraries the sbatch sources relative to itself.
                cyc, ckpt = parsing.resolve_ft_launcher_paths(
                    script, script_path=plat.batch_script_path(jid)
                )
                if want_cyc and cyc:
                    config.cycle_info_glob = cyc
                if want_ckpt and ckpt:
                    config.checkpoint_iteration_file = ckpt
                if config.cycle_info_glob or config.checkpoint_iteration_file:
                    break
    elif config.platform == "none" and not config.work_dir:
        print(
            "--platform none needs --work-dir (nothing to resolve a job id from).", file=sys.stderr
        )
        return 1

    if args.interval is None:
        return runner.run_once(config, plat).exit_code

    # Daemon mode. Cron is still the recommended deployment -- it restarts the watcher
    # for free if it dies, which is one less thing to watch.
    last = 0
    while True:
        try:
            last = runner.run_once(config, plat).exit_code
        except KeyboardInterrupt:
            return last
        except Exception:  # a detector bug must not end the watch
            logging.getLogger("nvrx_watch").exception("pass failed")
            last = 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
