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


_SCONTROL_TWO_JOBS = """JobId=101 JobName=run_a
   WorkDir=/runs/a
   Command=/runs/a/run.sh
   StdOut=/runs/a/slurm_out/slurm-101.out
   StdErr=/runs/a/slurm_out/slurm-101.err

JobId=102 JobName=run_b
   WorkDir=/runs/b
   Command=/runs/b/run_mock.sh
   StdOut=/runs/b/slurm_out/slurm-102.out
   StdErr=/runs/b/slurm_out/slurm-102.err
"""


def _client_with_scontrol(monkeypatch, output):
    monkeypatch.setattr(slurm.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(cmd, *, timeout):
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(slurm, "_run_slurm_command", fake_run)
    return SlurmClient(partitions=["batch"])


def test_scontrol_returns_job_paths_for_every_job(monkeypatch):
    # Regression: the in-loop "save previous job" branch only runs when a batch
    # holds more than one job, so a single-job fixture missed that it still
    # produced a bare tuple and crashed the poll with AttributeError.
    client = _client_with_scontrol(monkeypatch, _SCONTROL_TWO_JOBS)

    paths = client.get_job_output_paths(["101", "102"])

    assert set(paths) == {"101", "102"}
    for job_id, info in paths.items():
        assert isinstance(info, slurm.JobPaths), f"{job_id} returned {type(info).__name__}"


def test_scontrol_captures_workdir_and_command(monkeypatch):
    client = _client_with_scontrol(monkeypatch, _SCONTROL_TWO_JOBS)

    paths = client.get_job_output_paths(["101", "102"])

    assert paths["101"].stdout == "/runs/a/slurm_out/slurm-101.out"
    assert paths["101"].work_dir == "/runs/a"
    assert paths["101"].command == "/runs/a/run.sh"
    # The second job is the one saved by the in-loop branch.
    assert paths["102"].work_dir == "/runs/b"
    assert paths["102"].command == "/runs/b/run_mock.sh"
