import argparse
import fcntl
import functools
import os
import signal
import sys
import time


def _sigusr_handler(signum, frame, world_size_file):
    print(f"PID={os.getpid()} Signal {signum} received!", file=sys.stderr)
    decrement_world_size(world_size_file)
    sys.exit(signum)


def update_run_cnt(run_cnt_file):
    count = 0
    try:
        with open(run_cnt_file, "r+") as f:
            content = f.read().strip()
            count = int(content) if content else 0
    except FileNotFoundError:
        pass
    new_count = count + 1
    with open(run_cnt_file, "w") as f:
        f.write(f"{new_count}\n")
    print(f"UPDATED RUN COUNT {new_count}")


def increment_world_size(world_size_file):
    """
    Read an int counter from the file and write its value +1, lock the file so other processes would need to wait.
    If the file does not exist, create it and put 1 there.
    """
    with open(world_size_file, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file for exclusive access
        f.seek(0)
        content = f.read().strip()
        old_count = int(content) if content else 0
        new_count = old_count + 1
        f.seek(0)
        f.truncate()
        f.write(f"{new_count}\n")
        f.flush()
        print(f"UPDATED(post increment) WORLD SIZE: {new_count}")
        fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file


def decrement_world_size(world_size_file):
    """
    Read an int counter from the file and write its value -1, lock the file so other processes would need to wait.
    If the file does not exist, raise an error.
    """
    if not os.path.exists(world_size_file):
        raise FileNotFoundError(f"File {world_size_file} does not exist")
    with open(world_size_file, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file for exclusive access
        content = f.read().strip()
        old_count = int(content) if content else 0
        if old_count <= 0:
            print("Warning: World size already zero, no decrement performed.")
            return
        f.seek(0)
        f.truncate()
        new_count = old_count - 1
        f.write(f"{new_count}\n")
        f.flush()
        print(f"UPDATED(post decrement) WORLD SIZE: {new_count}")
        fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Workload Control Test Worker")
    parser.add_argument(
        "--max-time",
        type=int,
        required=False,
        default=600,
        help="Maximum execution time in seconds",
    )
    parser.add_argument(
        "--run-cnt-file",
        type=str,
        required=False,
        default="/tmp/_runs_cnt.txt",
        help="Path to the runs count file",
    )
    parser.add_argument(
        "--world-size-file",
        type=str,
        required=False,
        default="/tmp/_world_size.txt",
        help="Path to the world size file",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # Register signal handlers with explicit argument passing using functools.partial
    signal.signal(
        signal.SIGTERM, functools.partial(_sigusr_handler, world_size_file=args.world_size_file)
    )
    signal.signal(
        signal.SIGUSR1, functools.partial(_sigusr_handler, world_size_file=args.world_size_file)
    )
    signal.signal(
        signal.SIGUSR2, functools.partial(_sigusr_handler, world_size_file=args.world_size_file)
    )

    print(f"Initializing... PID={os.getpid()}")

    if int(os.environ.get('RANK', 0)) == 0:
        update_run_cnt(args.run_cnt_file)

    increment_world_size(args.world_size_file)

    print(f"PID={os.getpid()} Executing dummy loop...")

    start_time = time.monotonic()

    while True:
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > args.max_time:
            print(f"PID={os.getpid()} Time limit reached")
            break
        time.sleep(0.1)

    print(f"Done! PID={os.getpid()}")
    decrement_world_size(args.world_size_file)
    sys.exit(0)
