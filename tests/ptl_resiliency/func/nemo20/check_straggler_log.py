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


def cmd_check_status(log_file_lines, args):
    expected_status = "FAILED" if args.expect_failed else "SUCCEEDED"
    pattern = re.compile(r"Status: (SUCCEEDED|FAILED)")
    for ln in log_file_lines:
        match = pattern.search(ln)
        if match:
            detected_status = match.group(1)
            if detected_status != expected_status:
                raise ValueError(
                    f"Incorrect status: expected {expected_status} but found {detected_status}"
                )
            return


def cmd_num_reports(log_file_lines, args):
    report_lines = [ln for ln in log_file_lines if "GPU relative performance" in ln]
    num_reports = len(report_lines)
    if not (args.min <= num_reports <= args.max):
        raise ValueError(
            f"Invalid number of reports: {num_reports}. Valid range is from {args.min} to {args.max}."
        )


def _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers):
    rank_pattern = re.compile(r"StragglerId\(rank=(\d+),")
    found_stragglers = set()

    for ln in log_file_lines:
        pattern_found = pattern.search(ln)
        if pattern_found:
            pattern_found = pattern_found.group(1)

            # Extract the ranks from the matched stragglers section
            matches = rank_pattern.findall(pattern_found)
            for match in matches:
                rank_value = int(match)
                found_stragglers.add(rank_value)

    if found_stragglers != expected_stragglers:
        raise ValueError(
            f"Invalid relative GPU stragglers. Found: {found_stragglers}. Expected: {expected_stragglers}."
        )


def cmd_relative_gpu_stragglers(log_file_lines, args):
    pattern = re.compile(
        r"STRAGGLER DETECTION WARNING: Some GPUs have worse relative performance. Affected ranks: \{(.*?)\}"
    )
    expected_stragglers = set(args.ranks)
    _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers)


def cmd_individual_gpu_stragglers(log_file_lines, args):
    pattern = re.compile(
        r"STRAGGLER DETECTION WARNING: Some GPUs performance dropped. Affected ranks: \{(.*?)\}"
    )
    expected_stragglers = set(args.ranks)
    _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers)


def cmd_check_terminating(log_file_lines, args):
    terminate_pattern = re.compile(r"Detected stragglers. Terminating training...")
    last_ckpt_pattern = re.compile(
        r"Async checkpoint save for step \d+.*last.ckpt.*finalized successfully"
    )
    failed_pattern = re.compile(r"failed \(exitcode: 1\)")

    detected_stragglers = False
    last_ckpt_finalized = False
    failed_exit_code = False

    for ln in log_file_lines:
        if terminate_pattern.search(ln):
            detected_stragglers = True
            continue

        if detected_stragglers and last_ckpt_pattern.search(ln):
            last_ckpt_finalized = True
            continue

        if last_ckpt_finalized and failed_pattern.search(ln):
            failed_exit_code = True
            break

    if not detected_stragglers:
        raise ValueError("Straggler termination message not found.")
    if not last_ckpt_finalized:
        raise ValueError("Checkpoint for last.ckpt was not finalized successfully.")
    if not failed_exit_code:
        raise ValueError("Log file does not contain 'failed (exitcode: 1)'.")


def read_log_file(log_file):
    lines = []
    contains_done_entry = False
    with open(log_file, "r") as f:
        for ln in f.readlines():
            lines.append(ln.strip())
            if "Status:" in ln:
                contains_done_entry = True
    if not contains_done_entry:
        raise ValueError("Log file does not contain a 'Status:' entry.")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Verify a log file created by test_llama3.py."
    )
    parser.add_argument("--log", required=True, help="Path to the log file.")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    subp1 = subparsers.add_parser("num_reports", help="Number of straggler reports.")
    subp1.add_argument("--min", type=int)
    subp1.add_argument("--max", type=int)
    subp1.set_defaults(func=cmd_num_reports)

    subp2 = subparsers.add_parser(
        "relative_gpu_stragglers", help="Relative GPU stragglers."
    )
    subp2.add_argument("--ranks", type=int, nargs="*", default=set())
    subp2.set_defaults(func=cmd_relative_gpu_stragglers)

    subp3 = subparsers.add_parser(
        "individual_gpu_stragglers", help="Individual GPU stragglers."
    )
    subp3.add_argument("--ranks", type=int, nargs="*", default=set())
    subp3.set_defaults(func=cmd_individual_gpu_stragglers)

    subp4 = subparsers.add_parser(
        "check_terminating",
        help="Check if stragglers were terminated and last.ckpt was saved.",
    )
    subp4.set_defaults(func=cmd_check_terminating)

    subp5 = subparsers.add_parser(
        "check_status", help="Check if the status is failed or succeeded."
    )
    subp5.add_argument(
        "--expect-failed", action="store_true", help="Expect the status to be FAILED."
    )
    subp5.set_defaults(func=cmd_check_status)

    args = parser.parse_args()

    if args.command:
        lines = read_log_file(args.log)
        args.func(lines, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
