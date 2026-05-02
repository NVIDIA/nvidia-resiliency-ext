# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module entry point for nvidia_resiliency_ext.services.attrsvc."""

from nvidia_resiliency_ext.attribution._optional import reraise_if_missing_attribution_dependency


def main() -> None:
    """Run nvidia_resiliency_ext.services.attrsvc with a clear message when attribution extras are missing."""
    try:
        from .app import main as app_main
    except ModuleNotFoundError as exc:
        try:
            reraise_if_missing_attribution_dependency(exc, feature="nvrx-attrsvc")
        except ModuleNotFoundError as friendly_exc:
            raise SystemExit(str(friendly_exc)) from exc
        raise

    app_main()


if __name__ == "__main__":
    main()
