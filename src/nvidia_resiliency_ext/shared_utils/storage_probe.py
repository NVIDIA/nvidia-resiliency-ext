# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal storage path probe utility
# `python -m nvidia_resiliency_ext.shared_utils.storage_probe --storage-probe <paths...>`

import json
import logging
import os
import sys
from typing import Optional

# Module logger
logger = logging.getLogger(__name__)


def _storage_path_probe(paths: Optional[list[str]] = None) -> dict:
    """
    Probe a list of paths and return a dict with keys: invalid, missing, unreadable.
    This function is invoked inside a short-lived subprocess to avoid hangs on
    remote filesystems access.
    """
    invalid: list[str] = []
    missing: list[str] = []
    unreadable: list[str] = []

    if not paths:
        logger.debug("storage probe invoked with no paths; treating as success")
        return {"invalid": invalid, "missing": missing, "unreadable": unreadable}

    for p in paths:
        # Skip None and empty strings
        if not p:
            continue

        # Ensure p is a string (defensive check)
        if not isinstance(p, str):
            invalid.append(f"{p} (expected string path, got {type(p).__name__})")
            continue

        if not p.startswith("/"):
            invalid.append(f"{p} (not absolute; expected an absolute path starting with '/')")
            continue

        try:
            pnorm = os.path.normpath(p)
            if not os.path.exists(pnorm):
                missing.append(f"{pnorm} (missing)")
                continue

            if os.path.isfile(pnorm):
                try:
                    with open(pnorm, "rb") as fh:
                        fh.read(4096)
                except Exception as e:
                    unreadable.append(f"{pnorm} (read error: {str(e)})")
                continue

            if os.path.isdir(pnorm):
                try:
                    # list up to a few entries to ensure readability
                    _ = os.listdir(pnorm)[:5]
                except Exception as e:
                    unreadable.append(f"{pnorm} (list error: {str(e)})")
                continue

            # Other types: use os.stat to probe
            try:
                os.stat(pnorm)
            except Exception as e:
                unreadable.append(f"{pnorm} (stat error: {str(e)})")
                continue
        except Exception as e:
            unreadable.append(f"{p} (probe error: {str(e)})")

    # Log results: warn on any issues, debug on complete success
    if invalid or missing or unreadable:
        # Combine lists for readable warning
        parts = []
        if invalid:
            parts.append("invalid:" + ";".join(invalid))
        if missing:
            parts.append("missing:" + ";".join(missing))
        if unreadable:
            parts.append("unreadable:" + ";".join(unreadable))
        logger.warning("storage probe issues: %s" % (" | ".join(parts)))
    else:
        logger.debug("storage probe success: all paths accessible")

    return {"invalid": invalid, "missing": missing, "unreadable": unreadable}


if __name__ == "__main__":
    # Minimal CLI for module-level probe invocation
    import argparse

    parser = argparse.ArgumentParser(description="Storage path probe utility")
    parser.add_argument("paths", nargs="*", help="Paths to probe for storage health")
    args = parser.parse_args()

    if args.paths is not None:
        res = _storage_path_probe(args.paths)
        # Always print JSON to stdout for the caller to parse
        sys.stdout.write(json.dumps(res))
        # Exit non-zero if any issues found
        if res.get("invalid") or res.get("missing") or res.get("unreadable"):
            sys.exit(1)
        sys.exit(0)
