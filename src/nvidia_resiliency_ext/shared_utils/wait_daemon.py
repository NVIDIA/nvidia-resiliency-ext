#!/usr/bin/env python3

"""
SLURM by default terminates all user processes when the main job process is
finished. This also immediately terminates inprocess.MonitorProcess and
prevents it from submitting information to distributed store, and finalizing
iteration by waiting on termination barrier.

This script waits for all "python" processes launched by the current user to
finish before terminating the SLURM job, or waits for a given list of PIDs if provided in a file.
"""

import os
import sys
import time
import argparse
from typing import List


def wait_for_pids(pids: List[int]) -> None:
    """Wait for all specified PIDs to finish."""
    print(f"Monitoring {len(pids)} PIDs: {pids}")
    
    while pids:
        finished_pids = []
        
        for pid in pids:
            try:
                # Check if process exists by sending signal 0
                os.kill(pid, 0)
            except OSError:
                # Process has finished or doesn't exist
                finished_pids.append(pid)
        
        # Remove finished/invalid PIDs from the monitoring list
        for pid in finished_pids:
            pids.remove(pid)
            print(f"PID {pid} has finished or is invalid. {len(pids)} PIDs remaining: {pids}")
        
        if pids:
            time.sleep(1)


def read_pids_from_file(pidfile: str) -> List[int]:
    """Read PIDs from a file."""
    pids = []
    try:
        with open(pidfile, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    pids.append(int(line))
    except (IOError, ValueError) as e:
        print(f"Error reading PID file {pidfile}: {e}")
        return []
    
    return pids


def wait_daemon(pidfile: str) -> None:
    """Main function to wait for daemon processes."""
    # If PID file doesn't exist, exit immediately
    if not os.path.exists(pidfile):
        print(f"Error: PID file {pidfile} does not exist. Exiting.")
        sys.exit(1)
    
    pids = read_pids_from_file(pidfile)
    if not pids:
        print(f"No valid PIDs found in {pidfile}")
        return
    
    print(f"Waiting for PIDs from file: {pidfile}")
    wait_for_pids(pids)
    print("All monitored processes have exited.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wait for Python processes to finish before terminating SLURM job. "
                   "If PID file doesn't exist, exits immediately. Invalid PIDs are excluded from monitoring."
    )
    parser.add_argument(
        'pidfile', 
        help='File containing PIDs to wait for'
    )
    args = parser.parse_args()
    
    wait_daemon(args.pidfile)


if __name__ == "__main__":
    main() 