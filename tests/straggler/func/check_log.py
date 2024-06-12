# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import re


def cmd_num_reports(log_file_lines, args):
    report_lines = [ln for ln in log_file_lines if 'STRAGGLER REPORT' in ln]
    num_reports = len(report_lines)
    if not (args.min <= num_reports <= args.max):
        raise ValueError(
            f"Invalid number of reports: {num_reports}. Valid range is from {args.min} to {args.max}."
        )


def _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers):
    found_stragglers = set()
    for ln in log_file_lines:
        match = pattern.search(ln)
        if match:
            rank_value = int(match.group(1))
            found_stragglers.add(rank_value)
    if found_stragglers != expected_stragglers:
        raise ValueError(
            f"Invalid relative GPU stragglers. Found: {found_stragglers}. Expected: {expected_stragglers}."
        )


def cmd_relative_gpu_stragglers(log_file_lines, args):
    pattern = re.compile(r'DETECTED RELATIVE STRAGGLER GPU RANK=(\d+)')
    expected_stragglers = set(args.ranks)
    _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers)


def cmd_individual_gpu_stragglers(log_file_lines, args):
    pattern = re.compile(r'DETECTED INDIVIDUAL STRAGGLER GPU RANK=(\d+)')
    expected_stragglers = set(args.ranks)
    _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers)


def read_log_file(log_file):
    lines = []
    contains_done_entry = False
    with open(log_file, 'r') as f:
        for ln in f.readlines():
            lines.append(ln.strip())
            if 'DONE' in ln:
                contains_done_entry = True
    if not contains_done_entry:
        raise ValueError("Log file does not contain a 'DONE' entry.")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Verify a log file created by ddp_test.py.")
    parser.add_argument('--log', required=True, help='Path to the log file')

    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    subp1 = subparsers.add_parser('num_reports', help='Number of straggler reports.')
    subp1.add_argument('--min', type=int)
    subp1.add_argument('--max', type=int)
    subp1.set_defaults(func=cmd_num_reports)

    subp2 = subparsers.add_parser('relative_gpu_stragglers', help='Relative GPU stragglers.')
    subp2.add_argument('--ranks', type=int, nargs='*', default=set())
    subp2.set_defaults(func=cmd_relative_gpu_stragglers)

    subp3 = subparsers.add_parser('individual_gpu_stragglers', help='Individual GPU stragglers.')
    subp3.add_argument('--ranks', type=int, nargs='*', default=set())
    subp3.set_defaults(func=cmd_individual_gpu_stragglers)

    args = parser.parse_args()

    if args.command:
        lines = read_log_file(args.log)
        args.func(lines, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
