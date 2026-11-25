# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence beside the log, and the order the sources are asked in.

A run can end for a reason the application never records. The launcher's own
log and the scheduler's batch output both sit beside it, and each answers a
different question: why the run stopped, versus that it was stopped.
"""

from __future__ import annotations

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.l1.categories import category_by_id
from nvidia_resiliency_ext.services.attrsvc.corroborating import (
    DEFAULT_PROBES,
    SECTION_TIMEOUT_CATEGORY_ID,
)
from nvidia_resiliency_ext.services.attrsvc.nvrx_launcher import launcher_log_path, section_timeout

TIMEOUT_LINE = (
    "2026-11-24 13:20:55,132 [WARNING] [node0001-T02] [workload:1(3) infra:1105(0)] "
    "rank_monitor_server.py:639 [Cycle 3] Section 'step' has been open for "
    "121.30 seconds. Timeout is 120.00 seconds\n"
)
KILLED_LINE = (
    "2026-11-24 13:20:56,355 [DEBUG] [node0001-T09] rank_monitor_server.py:239 "
    "Rank monitor sent SIGCONT then SIGKILL to the rank (PID=3385849)\n"
)


def _layout(tmp_path, job_id, launcher_text):
    logs = tmp_path / "exp" / "logs"
    nvrx = tmp_path / "exp" / "nvrx" / job_id / "logs"
    logs.mkdir(parents=True)
    nvrx.mkdir(parents=True)
    log = logs / f"run_{job_id}_date_x_cycle3.log"
    log.write_text("0: iteration 10/100\n")
    (nvrx / f"nvrx_{job_id}_date_x_time_y.log").write_text(launcher_text)
    return str(log)


def test_the_launcher_log_sits_beside_the_application_log():
    assert launcher_log_path("/p/exp/logs/run_cycle3.log", "777") == "/p/exp/nvrx/777/logs"


def test_a_section_timeout_is_reported(tmp_path):
    log = _layout(tmp_path, "777", TIMEOUT_LINE * 20 + KILLED_LINE)

    found = section_timeout(log, "777", 3)

    assert found is not None
    assert found.section == "step"
    assert found.timeout_seconds == "120.00"
    assert found.ranks_reporting == 20
    assert found.monitor_killed_ranks is True


def test_a_timeout_in_another_cycle_is_not_this_cycle(tmp_path):
    """One launcher log covers every cycle of a job. A timeout in cycle 0 says
    nothing about cycle 3, and the lines carry the cycle precisely so they can
    be told apart."""
    other = TIMEOUT_LINE.replace("[Cycle 3]", "[Cycle 0]")
    log = _layout(tmp_path, "777", other * 20 + KILLED_LINE)

    assert section_timeout(log, "777", 3) is None


def test_a_few_warnings_are_not_a_kill(tmp_path):
    """A section briefly over its limit logs a warning without the monitor
    acting. The monitor firing shows up as thousands of ranks at once."""
    log = _layout(tmp_path, "777", TIMEOUT_LINE * 3)

    assert section_timeout(log, "777", 3) is None


def test_grpc_transport_logs_are_not_read(tmp_path):
    """The leaf and root logs are transport chatter, not decisions."""
    job = "777"
    log = _layout(tmp_path, job, "nothing here\n")
    nvrx = tmp_path / "exp" / "nvrx" / job / "logs"
    (nvrx / f"nvrx_{job}_date_x_time_y_grpc_root.log").write_text(TIMEOUT_LINE * 40)

    assert section_timeout(log, job, 3) is None


def test_a_missing_launcher_directory_falls_through(tmp_path):
    logs = tmp_path / "exp" / "logs"
    logs.mkdir(parents=True)
    log = logs / "run_777_date_x_cycle3.log"
    log.write_text("0: iteration 10/100\n")

    assert section_timeout(str(log), "777", 3) is None


@pytest.mark.parametrize("job_id, cycle_id", [("", 3), (None, 3), ("777", None)])
def test_without_identity_nothing_is_read(tmp_path, job_id, cycle_id):
    log = _layout(tmp_path, "777", TIMEOUT_LINE * 40)
    assert section_timeout(log, job_id, cycle_id) is None


def test_a_directory_outside_the_allowed_root_is_refused(tmp_path):
    log = _layout(tmp_path, "777", TIMEOUT_LINE * 40)

    assert section_timeout(log, "777", 3, allowed_root=str(tmp_path / "elsewhere")) is None
    assert section_timeout(log, "777", 3, allowed_root=str(tmp_path)) is not None


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_category_describes_the_mechanism_not_the_cause():
    category = category_by_id(SECTION_TIMEOUT_CATEGORY_ID)

    assert category is not None and category.decision == "RESTART"
    assert "rank_monitor_server" in category.description
    # It must not claim to know what the step was waiting on.
    assert "Not cat 11" in category.description and "Not cat 1:" in category.description


def test_the_timeout_is_asked_before_the_cancellation():
    """A section timeout ends with the launcher exiting, which cancels the
    array, so both can fire on one run. The timeout names the mechanism; the
    cancellation is its consequence."""

    assert [p.__name__ for p in DEFAULT_PROBES] == [
        "probe_section_timeout",
        "probe_scheduler_cancellation",
    ]
