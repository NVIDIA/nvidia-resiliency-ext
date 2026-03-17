#!/usr/bin/env python3
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

"""Build-level cleanup: generated protos and build artifacts. Run from project root: python clean.py"""

import glob
import os
import shutil
import sys

# Project root = directory containing this script
ROOT = os.path.dirname(os.path.abspath(__file__))
PROTO_DIR = os.path.join(ROOT, "src", "nvidia_resiliency_ext", "shared_utils", "proto")


def _clean_proto(dirpath: str) -> int:
    """Remove generated protobuf files (*_pb2.py, *_pb2.pyi, *_pb2_grpc.py)."""
    patterns = ("*_pb2.py", "*_pb2.pyi", "*_pb2_grpc.py")
    removed = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(dirpath, pattern)):
            try:
                os.remove(path)
                print(f"Removed: {path}")
                removed += 1
            except OSError as e:
                print(f"Warning: could not remove {path}: {e}", file=sys.stderr)
    return removed


def _clean_build_artifacts(root: str) -> int:
    """Remove build/, dist/, *.egg-info."""
    removed = 0
    for name in ("build", "dist"):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed: {path}")
            removed += 1
    for path in glob.glob(os.path.join(root, "*.egg-info")):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed: {path}")
            removed += 1
    return removed


def main() -> None:
    total = 0
    if os.path.isdir(PROTO_DIR):
        n = _clean_proto(PROTO_DIR)
        total += n
    n = _clean_build_artifacts(ROOT)
    total += n
    if total == 0:
        print("Nothing to clean.")
    else:
        print(f"Done. Removed {total} item(s).")


if __name__ == "__main__":
    main()
