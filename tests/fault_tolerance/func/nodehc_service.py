#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA
# SPDX-License-Identifier: Apache-2.0
#
# Standalone NodeHealthCheck gRPC server for testing.
#
# This implements the HealthCheckService defined in
# nvidia_resiliency_ext.shared_utils.proto.nvhcd.proto
# and listens on a Unix domain socket (UDS), e.g. unix:///var/run/nvhcd.sock
#
# Example:
#   PYTHONPATH=src python tests/fault_tolerance/func/nodehc_service.py \
#       --socket /var/run/nvhcd.sock --success
#
#   PYTHONPATH=src python tests/fault_tolerance/func/nodehc_service.py \
#       --socket /tmp/nvhcd.sock --fail --exit-code 42 --output '{"detail":"failed"}'
#
#   # Simulate slow/timeout response (e.g., client timeout set to 60s, sleep 65s)
#   PYTHONPATH=src python tests/fault_tolerance/func/nodehc_service.py \
#       --socket /tmp/nvhcd.sock --success --sleep-seconds 65
#

import argparse
import json
import os
import signal
import sys
import time
from concurrent import futures
from importlib import import_module

import grpc

from nvidia_resiliency_ext.shared_utils.proto import nvhcd_pb2

sys.modules.setdefault("nvhcd_pb2", nvhcd_pb2)
nvhcd_pb2_grpc = import_module("nvidia_resiliency_ext.shared_utils.proto.nvhcd_pb2_grpc")


class HealthCheckService(nvhcd_pb2_grpc.HealthCheckServiceServicer):
    def __init__(
        self,
        success: bool,
        exit_code: int,
        output: str,
        error: str,
        echo_args: bool,
        sleep_seconds: int,
    ):
        self._success = success
        self._exit_code = exit_code
        self._output = output
        self._error = error
        self._echo_args = echo_args
        self._sleep_seconds = max(0, int(sleep_seconds or 0))

    def RunHealthCheck(self, request, context):
        print(f"Recvd request {request}")
        # Optional artificial delay to simulate slow/timeout responses
        if self._sleep_seconds > 0:
            time.sleep(self._sleep_seconds)
        # Optionally echo back args in the output for visibility
        output = self._output
        if self._echo_args:
            try:
                parsed = json.loads(output) if output else {}
            except Exception:
                parsed = {"output": output}
            parsed["received_args"] = list(request.args)
            output = json.dumps(parsed)
        return nvhcd_pb2.HealthCheckResponse(
            success=self._success,
            exit_code=self._exit_code,
            output=output,
            error=self._error,
        )


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _cleanup_socket(path: str) -> None:
    try:
        if os.path.exists(path) and not os.path.isdir(path):
            os.unlink(path)
    except FileNotFoundError:
        pass


def parse_args(argv):
    p = argparse.ArgumentParser(description="Standalone NodeHealthCheck (nvhcd) gRPC server (UDS).")
    p.add_argument(
        "--socket",
        default="/var/run/nvhcd.sock",
        help="Unix domain socket path to bind (default: /var/run/nvhcd.sock)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--success", action="store_true", help="Always return success (default).")
    mode.add_argument("--fail", action="store_true", help="Always return failure.")
    p.add_argument("--exit-code", type=int, default=0, help="Exit code to return (default: 0).")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output string/JSON to return (default: empty).",
    )
    p.add_argument("--error", type=str, default="", help="Error string to return (default: empty).")
    p.add_argument(
        "--echo-args",
        action="store_true",
        help="Include received request.args in the response JSON under 'received_args'.",
    )
    p.add_argument(
        "--sleep-seconds",
        type=int,
        default=0,
        help="Sleep for N seconds before responding (simulate slow/timeout). Default: 0 (no delay).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])

    # Determine success
    success = True
    if args.fail:
        success = False
    elif args.success:
        success = True

    # Prepare socket
    uds_path = args.socket
    _ensure_parent_dir(uds_path)
    _cleanup_socket(uds_path)

    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = HealthCheckService(
        success=success,
        exit_code=args.exit_code,
        output=args.output,
        error=args.error,
        echo_args=bool(args.echo_args),
        sleep_seconds=int(args.sleep_seconds or 0),
    )
    nvhcd_pb2_grpc.add_HealthCheckServiceServicer_to_server(servicer, server)

    target = f"unix://{uds_path}"
    server.add_insecure_port(target)

    # Graceful shutdown on SIGTERM/SIGINT
    stopped = {"flag": False}

    def handle_signal(signum, frame):
        if not stopped["flag"]:
            stopped["flag"] = True
            server.stop(grace=None)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    server.start()
    print(f"[nodehc_service] Listening on {target}")
    try:
        server.wait_for_termination()
    finally:
        _cleanup_socket(uds_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
