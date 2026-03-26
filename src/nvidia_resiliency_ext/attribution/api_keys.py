# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load API tokens from environment and well-known file paths.

**NVIDIA API key** — Required for LogSage and LLM merge paths; callers that embed attribution
should fail startup or analysis if :func:`load_nvidia_api_key` returns empty.

**Slack bot token** — Optional; empty means notifications are disabled (postprocessing no-ops).
"""

from __future__ import annotations

import os


def _read_key_file(path: str) -> str:
    """Read and strip a secret file; return empty string on missing file or I/O error."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def load_nvidia_api_key() -> str:
    """Load NVIDIA API key from environment or file.

    Required for LLM-based attribution. Checks in order:

    1. ``NVIDIA_API_KEY`` environment variable
    2. ``NVIDIA_API_KEY_FILE`` environment variable (path to key file)
    3. ``~/.nvidia_api_key``
    4. ``~/.config/nvrx/nvidia_api_key``

    Returns:
        API key string, or empty string if not found or unreadable.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        return api_key.strip()

    key_file = os.getenv("NVIDIA_API_KEY_FILE")
    if key_file and os.path.isfile(key_file):
        return _read_key_file(key_file)

    home = os.path.expanduser("~")
    for path in (
        os.path.join(home, ".nvidia_api_key"),
        os.path.join(home, ".config", "nvrx", "nvidia_api_key"),
    ):
        if os.path.isfile(path):
            v = _read_key_file(path)
            if v:
                return v

    return ""


def load_slack_bot_token() -> str:
    """Load Slack bot token from environment or file (optional integrations).

    If empty, Slack notifications are skipped. Checks in order:

    1. ``SLACK_BOT_TOKEN`` environment variable
    2. ``SLACK_BOT_TOKEN_FILE`` environment variable (path to token file)
    3. ``~/.slack_bot_token``
    4. ``~/.slack_token`` (common alternate filename)
    5. ``~/.config/nvrx/slack_bot_token``

    Returns:
        Token string, or empty string if not found or unreadable.
    """
    token = os.getenv("SLACK_BOT_TOKEN")
    if token:
        return token.strip()

    key_file = os.getenv("SLACK_BOT_TOKEN_FILE")
    if key_file and os.path.isfile(key_file):
        v = _read_key_file(key_file)
        if v:
            return v

    home = os.path.expanduser("~")
    for path in (
        os.path.join(home, ".slack_bot_token"),
        os.path.join(home, ".slack_token"),
        os.path.join(home, ".config", "nvrx", "slack_bot_token"),
    ):
        if os.path.isfile(path):
            v = _read_key_file(path)
            if v:
                return v

    return ""
