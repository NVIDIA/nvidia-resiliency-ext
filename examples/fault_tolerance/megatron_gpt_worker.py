# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run unmodified pretrain_gpt.py with an optional, once-per-run demonstration fault."""

import argparse
import json
import os
import runpy
import signal
import sys
import threading
from pathlib import Path


def read_iteration(path, *, startup=False):
    """Only an absent startup tracker establishes a fresh-run baseline of zero."""
    try:
        return int(path.read_text().strip())
    except FileNotFoundError:
        return 0 if startup else None
    except (OSError, ValueError) as exc:
        if startup:
            raise RuntimeError(
                f'Cannot establish startup checkpoint iteration from {path}'
            ) from exc
        return None


def inject_after_checkpoint(iteration_file, marker, fault, stop, baseline):
    """Wait for a newly completed save, then claim a shared once-only marker and signal self."""
    while not stop.wait(0.1):
        if marker.exists():
            return
        iteration = read_iteration(iteration_file)
        if iteration is None or iteration <= baseline:
            continue
        try:
            # Shared across all worker incarnations and nodes of this launch attempt.
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, 'w') as stream:
            json.dump({'fault': fault, 'checkpoint': iteration, 'rank': os.environ['RANK']}, stream)
        print(
            f'GPT demo: injecting {fault} on rank {os.environ["RANK"]} '
            f'after completed checkpoint {iteration}',
            flush=True,
        )
        os.kill(os.getpid(), signal.SIGKILL if fault == 'kill' else signal.SIGSTOP)
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--megatron-path', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--fault-marker', type=Path, required=True)
    parser.add_argument('--fault', choices=('none', 'kill', 'hang'), default='kill')
    parser.add_argument('--fault-rank', type=int, default=1)
    args, workload_args = parser.parse_known_args()
    if workload_args[:1] == ['--']:
        workload_args = workload_args[1:]
    world_size = int(os.environ['WORLD_SIZE'])
    if args.fault != 'none' and not 0 <= args.fault_rank < world_size:
        parser.error('--fault-rank must name an active training rank')

    stop = threading.Event()
    injector = None
    errors = []
    selected = args.fault != 'none' and int(os.environ['RANK']) == args.fault_rank
    if selected and not args.fault_marker.exists():
        iteration_file = args.checkpoint_dir / 'latest_checkpointed_iteration.txt'
        # Establish the baseline before starting Megatron or the injector thread. A
        # failed read of an existing tracker must not make an old save appear new.
        baseline = read_iteration(iteration_file, startup=True)

        def inject():
            try:
                inject_after_checkpoint(
                    iteration_file,
                    args.fault_marker,
                    args.fault,
                    stop,
                    baseline,
                )
            except Exception as exc:
                errors.append(exc)

        injector = threading.Thread(target=inject, daemon=True)
        injector.start()

    sys.path.insert(0, str(args.megatron_path))
    entrypoint = args.megatron_path / 'pretrain_gpt.py'
    sys.argv = [str(entrypoint), *workload_args]
    try:
        try:
            runpy.run_path(str(entrypoint), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        stop.set()
        if injector is not None:
            injector.join(timeout=2)

    if selected and (errors or not args.fault_marker.exists()):
        raise RuntimeError(
            'Requested demo fault was not injected. Check checkpoint completion and '
            f'increase TRAIN_ITERS if needed. Injector errors: {errors}'
        )


if __name__ == '__main__':
    main()
