# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deployment env file the service finds on its own.

``--env-file`` is a client-side flag an image cannot default, and the systemd
and SLURM paths do not use it. Settings' own ``env_file`` would populate the
model but not the environment, leaving ``LLM_API_KEY_FILE`` unset -- the model
client reads that from ``os.environ`` -- so the service would start healthy and
fail on its first analysis.
"""

from __future__ import annotations

import os

import pytest

from nvidia_resiliency_ext.services.attrsvc.config import (
    DEFAULT_ENV_FILE,
    ENV_FILE_VAR,
    load_env_file,
)


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / "attrsvc.env"
    path.write_text(
        "# a comment\n"
        "\n"
        "NVRX_ATTRSVC_HOST=0.0.0.0\n"
        "LLM_API_KEY_FILE=/run/secrets/llm_api_key\n"
        "SLACK_CHANNEL=C123,C456\n"
        "not a valid assignment\n"
        "  NVRX_ATTRSVC_PORT=8123  \n",
        encoding="utf-8",
    )
    for key in ("NVRX_ATTRSVC_HOST", "LLM_API_KEY_FILE", "SLACK_CHANNEL", "NVRX_ATTRSVC_PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ENV_FILE_VAR, str(path))
    return path


def test_the_file_reaches_plain_environment_readers(env_file):
    """The point of doing this in the environment rather than in Settings:
    the model credential is read from os.environ, not through the model."""

    assert load_env_file() == str(env_file)
    assert os.environ["LLM_API_KEY_FILE"] == "/run/secrets/llm_api_key"
    assert os.environ["NVRX_ATTRSVC_HOST"] == "0.0.0.0"
    # Slack is read without the NVRX_ATTRSVC_ prefix the other settings use.
    assert os.environ["SLACK_CHANNEL"] == "C123,C456"


def test_comments_blanks_and_junk_are_skipped(env_file):
    load_env_file()
    assert os.environ["NVRX_ATTRSVC_PORT"] == "8123", "surrounding whitespace is trimmed"
    assert "not a valid assignment" not in os.environ


def test_an_explicit_value_outranks_the_file(env_file, monkeypatch):
    """Anything in the environment came from whoever started the service and
    must beat a file the deployment left on disk, or -e would stop working."""

    monkeypatch.setenv("NVRX_ATTRSVC_HOST", "127.0.0.1")
    load_env_file()
    assert os.environ["NVRX_ATTRSVC_HOST"] == "127.0.0.1"


def test_an_empty_setting_disables_the_file(env_file, monkeypatch):
    monkeypatch.setenv(ENV_FILE_VAR, "")
    assert load_env_file() is None
    assert "LLM_API_KEY_FILE" not in os.environ


def test_a_missing_file_is_not_fatal(tmp_path, monkeypatch):
    """Deployments that configure the service another way are normal, and an
    unreadable path must not stop it starting."""

    monkeypatch.setenv(ENV_FILE_VAR, str(tmp_path / "absent.env"))
    assert load_env_file() is None


def test_the_default_path_is_state_not_logs():
    """/var/log is swept by rotation and cleanup policy; configuration is state,
    and attrsvc already keeps its cache in /var/lib/nvrx."""

    assert DEFAULT_ENV_FILE == "/var/lib/nvrx/attrsvc.env"
