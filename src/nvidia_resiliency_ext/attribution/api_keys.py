# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load API tokens from environment and well-known file paths.

**LLM API key** — Required for LogSage and LLM merge paths; callers that embed attribution
should fail startup or analysis if :func:`load_llm_api_key` returns empty.

**Slack bot token** — Optional; empty means notifications are disabled (postprocessing no-ops).
"""

from __future__ import annotations

import os

LLM_API_KEY_ENV_VAR = "LLM_API_KEY"
LLM_API_KEY_FILE_ENV_VAR = "LLM_API_KEY_FILE"
DEFAULT_LLM_API_KEY_PATHS = ("~/.llm_api_key", "~/.config/nvrx/llm_api_key")


def _read_key_file(path: str) -> str:
    """Read and strip a secret file; return empty string on missing file or I/O error."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def llm_api_key_help_text() -> str:
    """Return user-facing guidance for configuring the LLM API key."""
    default_paths = " or ".join(DEFAULT_LLM_API_KEY_PATHS)
    return f"Set {LLM_API_KEY_ENV_VAR} or {LLM_API_KEY_FILE_ENV_VAR}, or create {default_paths}."


def llm_api_key_missing_message(
    *,
    include_empty: bool = False,
    context: str = "",
    suffix: str = "",
) -> str:
    """Return the standard user-facing missing-key message."""
    status = "LLM API key not found or empty." if include_empty else "LLM API key not found."
    parts = [status]
    if context:
        parts.append(context.strip())
    parts.append(llm_api_key_help_text())
    if suffix:
        parts.append(suffix.strip())
    return " ".join(parts)


def load_llm_api_key() -> str:
    """Load LLM API key from environment or file.

    Used by the attribution service layer. Checks in order:

    1. ``LLM_API_KEY`` environment variable
    2. ``LLM_API_KEY_FILE`` environment variable (path to key file)
    3. ``~/.llm_api_key``
    4. ``~/.config/nvrx/llm_api_key``

    Returns:
        API key string, or empty string if not found or unreadable.
    """
    api_key = os.getenv(LLM_API_KEY_ENV_VAR)
    if api_key:
        return api_key.strip()

    key_file = os.getenv(LLM_API_KEY_FILE_ENV_VAR)
    if key_file and os.path.isfile(key_file):
        v = _read_key_file(key_file)
        if v:
            return v

    for path in (os.path.expanduser(path) for path in DEFAULT_LLM_API_KEY_PATHS):
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
