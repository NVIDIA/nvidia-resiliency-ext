# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3]
    / "examples/fault_tolerance/deployment/slurm/scheduler_segment_health.sh"
)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def poller_environment(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sinfo_output = tmp_path / "sinfo.out"
    squeue_output = tmp_path / "squeue.out"
    scontrol_map = tmp_path / "scontrol.map"
    sinfo_output.write_text("", encoding="utf-8")
    squeue_output.write_text("", encoding="utf-8")
    scontrol_map.write_text("", encoding="utf-8")

    _write_executable(
        fake_bin / "timeout",
        """#!/bin/bash
[[ "$1" == "--kill-after=5s" ]] || exit 98
shift
shift
exec "$@"
""",
    )
    _write_executable(
        fake_bin / "sinfo",
        """#!/bin/bash
if [[ -n "${FAKE_SINFO_FAIL:-}" ]]; then echo "fake sinfo error" >&2; exit 1; fi
printf '%s\n' "$*" >"${FAKE_SINFO_ARGS}"
cat "${FAKE_SINFO_OUTPUT}"
""",
    )
    _write_executable(
        fake_bin / "squeue",
        """#!/bin/bash
if [[ -n "${FAKE_SQUEUE_FAIL:-}" ]]; then echo "fake squeue error" >&2; exit 1; fi
printf '%s\n' "$*" >"${FAKE_SQUEUE_ARGS}"
cat "${FAKE_SQUEUE_OUTPUT}"
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
printf '%s\n' "$*" >>"${FAKE_SCONTROL_ARGS}"
hostlist=""
for argument in "$@"; do hostlist="${argument}"; done
while IFS='|' read -r expected nodes; do
    if [[ "${expected}" == "${hostlist}" ]]; then
        [[ "${nodes}" != "__FAIL__" ]] || exit 1
        printf '%s\n' "${nodes}" | tr ',' '\n'
        exit 0
    fi
done <"${FAKE_SCONTROL_MAP}"
echo "unknown fake hostlist: ${hostlist}" >&2
exit 1
""",
    )
    output_dir = tmp_path / "decisions"
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_ARRAY_JOB_ID": "123",
            "FAKE_SCONTROL_ARGS": str(tmp_path / "scontrol.args"),
            "FAKE_SCONTROL_MAP": str(scontrol_map),
            "FAKE_SINFO_ARGS": str(tmp_path / "sinfo.args"),
            "FAKE_SINFO_OUTPUT": str(sinfo_output),
            "FAKE_SQUEUE_ARGS": str(tmp_path / "squeue.args"),
            "FAKE_SQUEUE_OUTPUT": str(squeue_output),
            "NVRX_SEGMENT_HEALTH_CHECK_DIR": str(output_dir),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "SLURM_JOB_PARTITION": "batch",
            "TMPDIR": str(tmp_path),
        }
    )
    return environment, output_dir, sinfo_output, squeue_output, scontrol_map


def _run_poller(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(_SCRIPT)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )


def _run_two_sourced_polls(
    environment: dict[str, str], between_polls: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f'source "{_SCRIPT}"\n'
                "scheduler_segment_health_poll_once || true\n"
                f"{between_polls}\n"
                "scheduler_segment_health_poll_once"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )


def _current_path(output_dir: Path, task_id: str = "7") -> Path:
    return output_dir / f"segment_health_check.123.{task_id}"


def _inactive_path(output_dir: Path, task_id: str = "7") -> Path:
    return output_dir / f"segment_health_check.123.{task_id}.inactive"


def _ledger_path(output_dir: Path) -> Path:
    return output_dir / "segment_health_check_history.123.log"


def _read_ledger(output_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in _ledger_path(output_dir).read_text(encoding="utf-8").splitlines()
    ]


def _set_scontrol_map(path: Path, mappings: dict[str, list[str]]) -> None:
    path.write_text(
        "".join(f"{hostlist}|{','.join(nodes)}\n" for hostlist, nodes in mappings.items()),
        encoding="utf-8",
    )


def _configure_two_excluded_tasks(sinfo_output, squeue_output, scontrol_map):
    sinfo_output.write_text("node18\nnode03\n", encoding="utf-8")
    squeue_output.write_text("12|node[17-20]\n7|node[01-04]\n", encoding="utf-8")
    _set_scontrol_map(
        scontrol_map,
        {
            "node[01-04]": ["node01", "node02", "node03", "node04"],
            "node[17-20]": ["node17", "node18", "node19", "node20"],
        },
    )


def test_publishes_per_task_decisions_and_audit(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    _configure_two_excluded_tasks(sinfo_output, squeue_output, scontrol_map)

    result = _run_poller(environment)

    assert result.returncode == 0, result.stderr
    assert _current_path(output_dir, "7").read_text(encoding="utf-8") == "node03"
    assert _current_path(output_dir, "12").read_text(encoding="utf-8") == "node18"
    records = _read_ledger(output_dir)
    assert [(record["event"], record["task_id"], record["nodes"]) for record in records] == [
        ("excluded", 7, "node03"),
        ("excluded", 12, "node18"),
    ]
    assert all(record["observed_at"].endswith("Z") for record in records)

    sinfo_args = Path(environment["FAKE_SINFO_ARGS"]).read_text(encoding="utf-8")
    assert "--partition=batch" in sinfo_args
    assert "--states=drain,down,fail,no_respond" in sinfo_args
    assert "--format=%N" in sinfo_args
    squeue_args = Path(environment["FAKE_SQUEUE_ARGS"]).read_text(encoding="utf-8")
    assert "--jobs=123" in squeue_args
    assert "--states=RUNNING" in squeue_args
    assert "--array" in squeue_args
    assert "--format=%K|%N" in squeue_args
    assert "--nodelist" not in squeue_args


def test_timestamp_failure_retains_audit_event(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("node03\n", encoding="utf-8")
    squeue_output.write_text("7|node03\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node03": ["node03"]})

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f'source "{_SCRIPT}"\n'
                "_scheduler_segment_health_timestamp_utc() { return 1; }\n"
                "scheduler_segment_health_poll_once"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert _current_path(output_dir).read_text(encoding="utf-8") == "node03"
    assert _read_ledger(output_dir)[0]["observed_at"] is None
    assert "history timestamp unavailable task=7 event=excluded" in result.stderr


def test_trims_scheduler_field_padding(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("  node03  \n", encoding="utf-8")
    squeue_output.write_text(" 7 | node[01-04] \n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node[01-04]": ["node01", "node03"]})

    result = _run_poller(environment)

    assert result.returncode == 0, result.stderr
    assert _current_path(output_dir).read_text(encoding="utf-8") == "node03"


def test_clean_poll_without_prior_state_avoids_squeue_and_artifacts(poller_environment):
    environment, output_dir, _, _, _ = poller_environment

    result = _run_poller(environment)

    assert result.returncode == 0, result.stderr
    assert not Path(environment["FAKE_SQUEUE_ARGS"]).exists()
    assert not list(output_dir.glob("segment_health_check.*"))


def test_sourcing_defines_interface_without_polling(poller_environment):
    environment, output_dir, _, _, _ = poller_environment

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f'source "{_SCRIPT}" && '
                "declare -F scheduler_segment_health_configure >/dev/null && "
                "declare -F scheduler_segment_health_poll_once >/dev/null && "
                "declare -F scheduler_segment_health_cleanup >/dev/null"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not Path(environment["FAKE_SINFO_ARGS"]).exists()
    assert not output_dir.exists()


def test_configure_initializes_without_polling(poller_environment):
    environment, output_dir, _, _, _ = poller_environment

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            f'source "{_SCRIPT}" && scheduler_segment_health_configure',
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_dir.is_dir()
    assert not Path(environment["FAKE_SINFO_ARGS"]).exists()


def test_reuses_workspace_until_explicit_cleanup(poller_environment):
    environment, _, _, _, _ = poller_environment

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f'source "{_SCRIPT}"\n'
                "scheduler_segment_health_configure || exit $?\n"
                'work="${_SCHEDULER_SEGMENT_HEALTH_WORK}"\n'
                'test -d "${work}" || exit 91\n'
                "scheduler_segment_health_poll_once || exit $?\n"
                'test "${work}" = "${_SCHEDULER_SEGMENT_HEALTH_WORK}" || exit 92\n'
                "scheduler_segment_health_poll_once || exit $?\n"
                'test "${work}" = "${_SCHEDULER_SEGMENT_HEALTH_WORK}" || exit 93\n'
                "scheduler_segment_health_cleanup || exit $?\n"
                'test ! -e "${work}" || exit 94\n'
                'test -z "${_SCHEDULER_SEGMENT_HEALTH_WORK:-}" || exit 95'
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


def test_artifact_directory_loss_fails_before_scheduler_query(poller_environment):
    environment, output_dir, _, _, _ = poller_environment

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f'source "{_SCRIPT}"\n'
                "scheduler_segment_health_poll_once || exit $?\n"
                'rm -f "${FAKE_SINFO_ARGS}"\n'
                'rmdir "${NVRX_SEGMENT_HEALTH_CHECK_DIR}"\n'
                "scheduler_segment_health_poll_once"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert not output_dir.exists()
    assert "artifact directory is unavailable" in result.stderr
    assert not Path(environment["FAKE_SINFO_ARGS"]).exists()


def test_relative_artifact_directory_is_rejected(poller_environment):
    environment, _, _, _, _ = poller_environment
    environment["NVRX_SEGMENT_HEALTH_CHECK_DIR"] = "relative/decisions"

    result = _run_poller(environment)

    assert result.returncode == 2


def test_unavailable_nodes_outside_job_publish_no_exclusion(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("other-node\n", encoding="utf-8")
    squeue_output.write_text("7|node[01-02]\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node[01-02]": ["node01", "node02"]})

    result = _run_poller(environment)

    assert result.returncode == 0, result.stderr
    assert not _current_path(output_dir).exists()
    assert not _ledger_path(output_dir).exists()


def test_running_task_recovery_clears_current_file(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("node03\n", encoding="utf-8")
    squeue_output.write_text("7|node[01-04]\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node[01-04]": ["node01", "node03"]})
    first = _run_poller(environment)
    assert first.returncode == 0, first.stderr

    sinfo_output.write_text("", encoding="utf-8")
    second = _run_poller(environment)

    assert second.returncode == 0, second.stderr
    assert _current_path(output_dir).exists()
    assert _current_path(output_dir).stat().st_size == 0
    assert [record["event"] for record in _read_ledger(output_dir)] == ["excluded", "cleared"]


def test_each_poll_reconstructs_state_from_current_files(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("node03\n", encoding="utf-8")
    squeue_output.write_text("7|node03\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node03": ["node03"]})
    result = _run_two_sourced_polls(
        environment,
        f': >"{environment["FAKE_SINFO_OUTPUT"]}"',
    )

    assert result.returncode == 0, result.stderr
    assert _current_path(output_dir).exists()
    assert _current_path(output_dir).stat().st_size == 0
    assert [record["event"] for record in _read_ledger(output_dir)] == ["excluded", "cleared"]


def test_inactive_task_preserves_last_observation_outside_control(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("node03\n", encoding="utf-8")
    squeue_output.write_text("7|node[01-04]\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node[01-04]": ["node01", "node03"]})
    first = _run_poller(environment)
    assert first.returncode == 0, first.stderr

    sinfo_output.write_text("", encoding="utf-8")
    squeue_output.write_text("", encoding="utf-8")
    second = _run_two_sourced_polls(
        environment,
        (
            f'test -f "{_current_path(output_dir)}" '
            f'&& test ! -e "{_inactive_path(output_dir)}" || exit 91'
        ),
    )

    assert second.returncode == 0, second.stderr
    assert not _current_path(output_dir).exists()
    assert _inactive_path(output_dir).read_text(encoding="utf-8") == "node03"
    assert _read_ledger(output_dir)[-1]["reason"] == "task_inactive"


def test_transient_missing_task_does_not_move_current_file_inactive(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("node03\n", encoding="utf-8")
    squeue_output.write_text("7|node[01-04]\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node[01-04]": ["node01", "node03"]})
    first = _run_poller(environment)
    assert first.returncode == 0, first.stderr

    sinfo_output.write_text("", encoding="utf-8")
    squeue_output.write_text("", encoding="utf-8")
    result = _run_two_sourced_polls(
        environment,
        f'printf "%s\\n" "7|node[01-04]" >"{squeue_output}"',
    )

    assert result.returncode == 0, result.stderr
    assert _current_path(output_dir).exists()
    assert _current_path(output_dir).stat().st_size == 0
    assert not _inactive_path(output_dir).exists()
    records = _read_ledger(output_dir)
    assert [record["event"] for record in records] == ["excluded", "cleared"]
    assert "reason" not in records[-1]


def test_current_state_scan_ignores_inactive_and_nonmatching_files(poller_environment):
    environment, output_dir, _, _, _ = poller_environment
    output_dir.mkdir()
    inactive = _inactive_path(output_dir)
    inactive.write_text("node03", encoding="utf-8")
    unrelated = output_dir / "segment_health_check.123.7.temporary"
    unrelated.write_text("node04", encoding="utf-8")

    result = _run_poller(environment)

    assert result.returncode == 0, result.stderr
    assert not Path(environment["FAKE_SQUEUE_ARGS"]).exists()
    assert inactive.read_text(encoding="utf-8") == "node03"
    assert unrelated.read_text(encoding="utf-8") == "node04"


def test_changed_unavailable_nodes_update_state_and_audit(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    squeue_output.write_text("7|node[01-04]\n", encoding="utf-8")
    _set_scontrol_map(
        scontrol_map,
        {"node[01-04]": ["node01", "node03", "node04"]},
    )
    sinfo_output.write_text("node03\n", encoding="utf-8")
    first = _run_poller(environment)
    assert first.returncode == 0, first.stderr

    sinfo_output.write_text("node04\nnode03\n", encoding="utf-8")
    second = _run_poller(environment)

    assert second.returncode == 0, second.stderr
    assert _current_path(output_dir).read_text(encoding="utf-8") == "node03,node04"
    records = _read_ledger(output_dir)
    assert [record["event"] for record in records] == ["excluded", "updated"]
    assert records[-1]["nodes"] == "node03,node04"


def test_unchanged_exclusion_does_not_duplicate_audit(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    sinfo_output.write_text("node03\n", encoding="utf-8")
    squeue_output.write_text("7|node03\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node03": ["node03"]})

    assert _run_poller(environment).returncode == 0
    assert _run_poller(environment).returncode == 0

    assert [record["event"] for record in _read_ledger(output_dir)] == ["excluded"]


@pytest.mark.parametrize("failure", ["sinfo", "squeue", "scontrol"])
def test_failed_pass_preserves_previous_decision(poller_environment, failure):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    output_dir.mkdir()
    decision = _current_path(output_dir)
    decision.write_text("node03", encoding="utf-8")

    if failure == "sinfo":
        environment["FAKE_SINFO_FAIL"] = "1"
    else:
        sinfo_output.write_text("node03\n", encoding="utf-8")
        if failure == "squeue":
            environment["FAKE_SQUEUE_FAIL"] = "1"
        else:
            squeue_output.write_text("7|node[01-04]\n", encoding="utf-8")
            _set_scontrol_map(scontrol_map, {"node[01-04]": ["__FAIL__"]})

    result = _run_poller(environment)

    assert result.returncode != 0
    assert decision.read_text(encoding="utf-8") == "node03"


def test_one_publication_failure_does_not_block_other_tasks(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    _configure_two_excluded_tasks(sinfo_output, squeue_output, scontrol_map)
    output_dir.mkdir()
    failed_task = _current_path(output_dir, "7")
    successful_task = _current_path(output_dir, "12")
    failed_task.write_text("node01", encoding="utf-8")

    first = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f'source "{_SCRIPT}"\n'
                "eval \"$(declare -f _scheduler_segment_health_set_current_value "
                "| sed '1s/_scheduler_segment_health_set_current_value/'"
                "'_scheduler_segment_health_set_current_value_real/')\"\n"
                "_scheduler_segment_health_set_current_value() {\n"
                '    [[ "$1" != "7" ]] || return 1\n'
                '    _scheduler_segment_health_set_current_value_real "$@"\n'
                "}\n"
                "scheduler_segment_health_poll_once"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )

    assert first.returncode != 0
    assert failed_task.read_text(encoding="utf-8") == "node01"
    assert successful_task.read_text(encoding="utf-8") == "node18"
    assert [(record["event"], record["task_id"]) for record in _read_ledger(output_dir)] == [
        ("excluded", 12)
    ]

    second = _run_poller(environment)

    assert second.returncode == 0, second.stderr
    assert failed_task.read_text(encoding="utf-8") == "node03"
    assert successful_task.read_text(encoding="utf-8") == "node18"
    assert [(record["event"], record["task_id"]) for record in _read_ledger(output_dir)] == [
        ("excluded", 12),
        ("updated", 7),
    ]


def test_stale_poll_deadline_clears_exclusion(poller_environment):
    environment, output_dir, _, _, _ = poller_environment
    output_dir.mkdir()
    decision = _current_path(output_dir)
    decision.write_text("node03", encoding="utf-8")
    environment["FAKE_SINFO_FAIL"] = "1"
    environment["NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS"] = "1"

    result = _run_two_sourced_polls(environment, "/bin/sleep 1")

    assert result.returncode != 0
    assert decision.stat().st_size == 0
    record = _read_ledger(output_dir)[-1]
    assert record["event"] == "cleared"
    assert record["task_id"] == 7
    assert record["reason"] == "stale_decision_expired"


def test_noncanonical_current_value_cannot_inject_poll_snapshot_rows(poller_environment):
    environment, output_dir, _, squeue_output, scontrol_map = poller_environment
    output_dir.mkdir()
    decision = _current_path(output_dir)
    decision.write_text("node03\n999|K|rogue", encoding="utf-8")
    squeue_output.write_text("7|node03\n999|node99\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node03": ["node03"], "node99": ["node99"]})

    result = _run_poller(environment)

    assert result.returncode == 0, result.stderr
    assert decision.stat().st_size == 0
    assert not _current_path(output_dir, "999").exists()
    assert not _ledger_path(output_dir).exists()


def test_trailing_newline_is_not_normalized_as_canonical_state(poller_environment):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    output_dir.mkdir()
    decision = _current_path(output_dir)
    decision.write_text("node03\n", encoding="utf-8")
    sinfo_output.write_text("node03\n", encoding="utf-8")
    squeue_output.write_text("7|node03\n", encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node03": ["node03"]})

    result = _run_poller(environment)

    assert result.returncode == 0, result.stderr
    assert decision.read_bytes() == b"node03"
    assert [record["event"] for record in _read_ledger(output_dir)] == ["updated"]


@pytest.mark.parametrize(
    "taskmap",
    [
        "not-a-task|node01\n",
        "7|node01|extra\n",
        "7|node{01,02}\n",
        "7|node01\n7|node02\n",
    ],
)
def test_malformed_task_mapping_preserves_previous_decision(poller_environment, taskmap):
    environment, output_dir, sinfo_output, squeue_output, scontrol_map = poller_environment
    output_dir.mkdir()
    decision = _current_path(output_dir)
    decision.write_text("node03", encoding="utf-8")
    sinfo_output.write_text("node01\n", encoding="utf-8")
    squeue_output.write_text(taskmap, encoding="utf-8")
    _set_scontrol_map(scontrol_map, {"node01": ["node01"], "node02": ["node02"]})

    result = _run_poller(environment)

    assert result.returncode != 0
    assert decision.read_text(encoding="utf-8") == "node03"


def test_malformed_scheduler_node_preserves_previous_decision(poller_environment):
    environment, output_dir, sinfo_output, _, _ = poller_environment
    output_dir.mkdir()
    decision = _current_path(output_dir)
    decision.write_text("node03", encoding="utf-8")
    sinfo_output.write_text("node01|unexpected\n", encoding="utf-8")

    result = _run_poller(environment)

    assert result.returncode != 0
    assert decision.read_text(encoding="utf-8") == "node03"


def test_array_job_id_must_be_numeric(poller_environment):
    environment, output_dir, _, _, _ = poller_environment
    environment["SLURM_ARRAY_JOB_ID"] = "../../wrong"

    result = _run_poller(environment)

    assert result.returncode == 2
    assert not output_dir.exists()


def test_partition_is_required(poller_environment):
    environment, output_dir, _, _, _ = poller_environment
    del environment["SLURM_JOB_PARTITION"]

    result = _run_poller(environment)

    assert result.returncode != 0
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS", "0"),
        ("NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT", "0"),
    ],
)
def test_numeric_settings_must_be_positive_integers(poller_environment, name, value):
    environment, output_dir, _, _, _ = poller_environment
    environment[name] = value

    result = _run_poller(environment)

    assert result.returncode == 2
    assert f"{name} must be a positive integer: {value}" in result.stderr
    assert not output_dir.exists()
