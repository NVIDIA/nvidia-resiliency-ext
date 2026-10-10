# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the Slurm attrsvc process supervisor."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = REPO_ROOT / "services" / "attrsvc" / "deploy" / "supervise_attrsvc.sh"
SLURM_SCRIPT = REPO_ROOT / "services" / "attrsvc" / "deploy" / "slurm.sbatch"


def _write_fake_service(tmp_path: Path) -> Path:
    service = tmp_path / "fake_attrsvc.sh"
    service.write_text(
        """#!/bin/bash
set -u
count=0
if [[ -f \"${FAKE_COUNT_FILE}\" ]]; then
    count=$(cat \"${FAKE_COUNT_FILE}\")
fi
count=$((count + 1))
printf '%s\\n' \"${count}\" > \"${FAKE_COUNT_FILE}\"
if (( count <= FAKE_FAILURES )); then
    exit \"${FAKE_EXIT_STATUS:-23}\"
fi
trap 'sleep \"${FAKE_TERM_DELAY_SECONDS:-0}\"; printf terminated > \"${FAKE_SIGNAL_FILE}\"; exit 0' TERM INT USR1 USR2
printf '%s\\n' \"$$\" > \"${FAKE_PID_FILE}\"
while true; do sleep 1; done
""",
        encoding="utf-8",
    )
    service.chmod(0o755)
    return service


def _environment(tmp_path: Path, *, failures: int, backoff: str) -> dict[str, str]:
    return {
        **os.environ,
        "FAKE_COUNT_FILE": str(tmp_path / "count"),
        "FAKE_PID_FILE": str(tmp_path / "pid"),
        "FAKE_SIGNAL_FILE": str(tmp_path / "signal"),
        "FAKE_FAILURES": str(failures),
        "FAKE_EXIT_STATUS": "23",
        "NVRX_ATTRSVC_SUPERVISOR_BACKOFF_SECONDS": backoff,
        "NVRX_ATTRSVC_SUPERVISOR_STABLE_SECONDS": "300",
    }


def _wait_for_file(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def _write_fake_date(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    date = bin_dir / "date"
    date.write_text(
        """#!/bin/bash
if [[ "${1:-}" == "+%s" ]]; then
    count=0
    if [[ -f "${FAKE_DATE_COUNT_FILE}" ]]; then
        count=$(cat "${FAKE_DATE_COUNT_FILE}")
    fi
    count=$((count + 1))
    printf '%s\\n' "${count}" > "${FAKE_DATE_COUNT_FILE}"
    if (( count == ${FAKE_DATE_DELAY_CALL:-0} )); then
        : > "${FAKE_DATE_READY_FILE}"
        sleep 0.2
    fi
    printf '%s\\n' "$((100 + count))"
else
    printf '2026-10-09T00:00:00Z\\n'
fi
""",
        encoding="utf-8",
    )
    date.chmod(0o755)
    return bin_dir


def test_supervisor_restarts_transient_failures_and_forwards_shutdown(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=2, backoff="0 0 0 0 0")
    process = subprocess.Popen(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _wait_for_file(tmp_path / "pid")
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=5)

    assert process.returncode == 0
    assert (tmp_path / "count").read_text(encoding="utf-8").strip() == "3"
    assert (tmp_path / "signal").read_text(encoding="utf-8") == "terminated"
    assert "restart_attempt=1/5 delay_seconds=0" in output
    assert "restart_attempt=2/5 delay_seconds=0" in output
    assert "stopped for supervisor shutdown" in output


def test_supervisor_does_not_launch_child_when_shutdown_arrives_before_launch(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=0, backoff="0")
    bin_dir = _write_fake_date(tmp_path)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["FAKE_DATE_COUNT_FILE"] = str(tmp_path / "date-count")
    environment["FAKE_DATE_DELAY_CALL"] = "1"
    environment["FAKE_DATE_READY_FILE"] = str(tmp_path / "date-ready")
    process = subprocess.Popen(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _wait_for_file(tmp_path / "date-ready")
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=5)

    assert process.returncode == 0
    assert not (tmp_path / "count").exists()
    assert "stopped for supervisor shutdown" in output


def test_supervisor_exits_after_configured_restart_limit(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=99, backoff="0 0")

    completed = subprocess.run(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 23
    assert (tmp_path / "count").read_text(encoding="utf-8").strip() == "3"
    assert "restart_attempt=2/2 delay_seconds=0" in completed.stdout
    assert "restart limit reached after 2 restart attempts" in completed.stdout


def test_supervisor_resets_restart_count_after_stable_runtime(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=2, backoff="0 0 0 0 0")
    bin_dir = _write_fake_date(tmp_path)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["FAKE_DATE_COUNT_FILE"] = str(tmp_path / "date-count")
    environment["NVRX_ATTRSVC_SUPERVISOR_STABLE_SECONDS"] = "001"
    process = subprocess.Popen(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _wait_for_file(tmp_path / "pid")
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=5)

    assert process.returncode == 0
    assert output.count("restart_attempt=1/5 delay_seconds=0") == 2
    assert "restart_attempt=2/5" not in output
    assert output.count("resetting consecutive restart count") == 2


def test_supervisor_rejects_zero_stable_runtime(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=0, backoff="0")
    environment["NVRX_ATTRSVC_SUPERVISOR_STABLE_SECONDS"] = "000"

    completed = subprocess.run(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 2
    assert "must be greater than zero" in completed.stdout
    assert not (tmp_path / "count").exists()


def test_supervisor_reaps_child_across_repeated_shutdown_signals(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=0, backoff="0")
    environment["FAKE_TERM_DELAY_SECONDS"] = "0.5"
    process = subprocess.Popen(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _wait_for_file(tmp_path / "pid")
    child_pid = int((tmp_path / "pid").read_text(encoding="utf-8"))
    process.send_signal(signal.SIGTERM)
    time.sleep(0.05)
    process.send_signal(signal.SIGUSR1)
    output, _ = process.communicate(timeout=5)

    assert process.returncode == 0
    assert (tmp_path / "signal").read_text(encoding="utf-8") == "terminated"
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert "stopped for supervisor shutdown" in output


def test_supervisor_does_not_enter_backoff_when_shutdown_arrives_after_child_exit(
    tmp_path: Path,
):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=1, backoff="30")
    bin_dir = _write_fake_date(tmp_path)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["FAKE_DATE_COUNT_FILE"] = str(tmp_path / "date-count")
    environment["FAKE_DATE_DELAY_CALL"] = "2"
    environment["FAKE_DATE_READY_FILE"] = str(tmp_path / "date-ready")
    process = subprocess.Popen(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _wait_for_file(tmp_path / "date-ready")
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=5)

    assert process.returncode == 0
    assert (tmp_path / "count").read_text(encoding="utf-8").strip() == "1"
    assert "restart_attempt=" not in output
    assert "stopped for supervisor shutdown" in output


def test_supervisor_defaults_and_slurm_wiring_are_explicit(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=0, backoff="1 2 5 10 20")
    environment.pop("NVRX_ATTRSVC_SUPERVISOR_BACKOFF_SECONDS")
    process = subprocess.Popen(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _wait_for_file(tmp_path / "pid")
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=5)

    assert process.returncode == 0
    assert "restart_backoff_seconds=1,2,5,10,20 stable_reset_seconds=300" in output
    assert 'exec "${SCRIPT_DIR}/supervise_attrsvc.sh" nvrx-attrsvc' in SLURM_SCRIPT.read_text(
        encoding="utf-8"
    )


def test_supervisor_rejects_invalid_backoff_configuration(tmp_path: Path):
    service = _write_fake_service(tmp_path)
    environment = _environment(tmp_path, failures=0, backoff="1 soon 5")

    completed = subprocess.run(
        ["bash", str(SUPERVISOR), str(service)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 2
    assert "invalid attrsvc supervisor backoff value: soon" in completed.stdout
