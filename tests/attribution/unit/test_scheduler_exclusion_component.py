# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvidia_resiliency_ext.attribution.scheduler_exclusions import PublisherConfig
from nvidia_resiliency_ext.attribution.scheduler_exclusions import manager as module


@pytest.mark.parametrize(
    "env",
    [
        {},
        {
            "SLURM_ARRAY_JOB_ID": "999",
            "SLURM_JOB_PARTITION": "other",
            "NVRX_SEGMENT_HEALTH_CHECK_DIR": "/other/path",
            "NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS": "1",
            "NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT": "1",
            "SEGMENT_HEALTH_POLL_INTERVAL": "7",
        },
    ],
)
def test_process_owner_supplies_job_config_without_launcher_context(tmp_path, env, monkeypatch):
    config = PublisherConfig("123", "batch", tmp_path / "decisions", 120, 15)
    original_env = dict(env)
    process = Mock(pid=12345)
    process.poll.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "killpg", Mock())
    manager = module.SegmentHealthPublisherManager(config, env=env)
    try:
        assert manager.start()
        popen.assert_called_once()
        child_env = popen.call_args.kwargs["env"]
        assert child_env["SLURM_ARRAY_JOB_ID"] == "123"
        assert child_env["SLURM_JOB_PARTITION"] == "batch"
        assert child_env["NVRX_SEGMENT_HEALTH_CHECK_DIR"] == str(config.directory)
        assert child_env["NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS"] == "120"
        assert child_env["NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT"] == "15"
        assert PublisherConfig.from_env(child_env) == config
        assert child_env.get("SEGMENT_HEALTH_POLL_INTERVAL") == env.get(
            "SEGMENT_HEALTH_POLL_INTERVAL"
        )
        assert env == original_env
        assert not config.directory.exists()
    finally:
        manager.stop()


def test_component_and_cli_import_without_optional_dependencies():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
import sys
from nvidia_resiliency_ext.attribution.scheduler_exclusions import (
    PublisherConfig, SchedulerExclusionPublisher, SegmentHealthPublisherManager,
)
from nvidia_resiliency_ext.attribution.scheduler_exclusions.runtime import parent_lifetime, serve
from nvidia_resiliency_ext.services.scheduler_exclusions.__main__ import main

for prefix in (
    'torch', 'fastapi', 'pydantic', 'langchain', 'openai', 'logsage',
    'nvidia_resiliency_ext.fault_tolerance',
    'nvidia_resiliency_ext.attribution.restart_agent',
    'nvidia_resiliency_ext.services.attrsvc',
):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
""",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
