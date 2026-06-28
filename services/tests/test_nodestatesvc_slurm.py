# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nvidia_resiliency_ext.services.nodestatesvc import slurm
from nvidia_resiliency_ext.services.nodestatesvc.slurm import (
    SlurmNodeStateClient,
    SlurmQueryError,
    normalize_node_state,
)


def test_normalize_node_state_removes_slurm_decorations():
    assert normalize_node_state("drain*") == "drain"
    assert normalize_node_state("no respond") == "no_respond"
    assert normalize_node_state("DOWN~") == "down"


def test_slurm_node_state_client_queries_sinfo_in_batches(monkeypatch):
    commands = []

    monkeypatch.setattr(slurm.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(cmd, *, timeout):
        commands.append(cmd)
        nodes_arg = cmd[cmd.index("--nodes") + 1]
        lines = []
        for node in nodes_arg.split(","):
            state = "drain*" if node == "node-b" else "allocated"
            lines.append(f"{node}|{state}|test reason")
        return SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr="")

    monkeypatch.setattr(slurm, "_run_slurm_command", fake_run)

    client = SlurmNodeStateClient(batch_size=2)
    records = client.get_node_states(["node-a", "node-b", "node-c"])

    assert len(commands) == 2
    assert records["node-a"].bad is False
    assert records["node-b"].bad is True
    assert records["node-b"].state == "DRAIN"
    assert records["node-c"].bad is False


def test_missing_sinfo_node_is_bad_unknown(monkeypatch):
    monkeypatch.setattr(slurm.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(cmd, *, timeout):
        return SimpleNamespace(returncode=0, stdout="node-a|idle|\n", stderr="")

    monkeypatch.setattr(slurm, "_run_slurm_command", fake_run)

    client = SlurmNodeStateClient()
    records = client.get_node_states(["node-a", "node-missing"])

    assert records["node-missing"].bad is True
    assert records["node-missing"].slurm_visible is False


def test_check_available_handles_os_error(monkeypatch):
    monkeypatch.setattr(slurm.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(cmd, *, timeout):
        raise PermissionError("denied")

    monkeypatch.setattr(slurm, "_run_slurm_command", fake_run)

    client = SlurmNodeStateClient()
    available, error = client.check_available()

    assert available is False
    assert "sinfo could not be started" in error


def test_node_state_query_wraps_os_error(monkeypatch):
    monkeypatch.setattr(slurm.shutil, "which", lambda command: f"/usr/bin/{command}")

    def fake_run(cmd, *, timeout):
        raise PermissionError("denied")

    monkeypatch.setattr(slurm, "_run_slurm_command", fake_run)

    client = SlurmNodeStateClient()

    with pytest.raises(SlurmQueryError, match="sinfo could not be started"):
        client.get_node_states(["node-a"])
