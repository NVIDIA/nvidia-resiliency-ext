# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from nvidia_resiliency_ext.services.smonsvc import slurm
from nvidia_resiliency_ext.services.smonsvc.slurm import SlurmClient


def test_slurm_client_resolves_command_paths(monkeypatch):
    commands = []

    monkeypatch.setattr(slurm.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(cmd, *, timeout):
        commands.append(cmd)
        return SimpleNamespace(
            returncode=0,
            stdout="123|train|alice|batch|RUNNING\n",
            stderr="",
        )

    monkeypatch.setattr(slurm, "_run_slurm_command", fake_run)

    client = SlurmClient(partitions=["batch"])
    jobs = client.get_running_jobs({"123": ("/tmp/out.log", "/tmp/err.log")})

    assert commands[0][0] == "/usr/bin/squeue"
    assert jobs == {
        "123": {
            "name": "train",
            "user": "alice",
            "partition": "batch",
            "state": "RUNNING",
            "stdout_path": "/tmp/out.log",
            "stderr_path": "/tmp/err.log",
        }
    }


def test_slurm_available_false_when_squeue_missing(monkeypatch):
    monkeypatch.setattr(slurm.shutil, "which", lambda command: None)

    client = SlurmClient(partitions=["batch"])

    assert client.check_slurm_available() is False
