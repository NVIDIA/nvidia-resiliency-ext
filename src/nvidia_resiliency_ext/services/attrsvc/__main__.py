# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module entry point for nvidia_resiliency_ext.services.attrsvc."""


def main() -> None:
    """Run nvidia_resiliency_ext.services.attrsvc."""
    from .app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
