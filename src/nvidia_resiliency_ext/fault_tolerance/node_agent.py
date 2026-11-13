# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private entrypoint for the control host's worker-launching child."""

from .child_process import read_child_args


def main():
    args = read_child_args()
    if args is None:
        raise RuntimeError("node-agent must be started by ft_launcher")
    args._is_control_host_agent = True
    from .launcher import main as launcher_main

    launcher_main(args)


if __name__ == "__main__":
    main()
