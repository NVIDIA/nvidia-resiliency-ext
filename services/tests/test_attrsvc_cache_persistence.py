# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surviving a restart.

All dedup state used to be in memory, so restarting attrsvc re-analyzed every
complete cycle of every running job and L3 evaluated each one as a first
attempt. Job 4234179's cycle 0 was analyzed five times across one afternoon.
"""

import json
import os

import pytest

from nvidia_resiliency_ext.attribution.restart_agent.models import AnalysisResult
from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
    _CACHE_SCHEMA,
    _STATUS_COMPLETED,
    _AttemptExecution,
)


class _Control:
    def __init__(self):
        self.seeded = None
        self._records = ()

    def records(self, job_id=None):
        return self._records

    def seed(self, records, *, mode="replace"):
        self.seeded = (list(records), mode)


class _Runtime:
    def __init__(self):
        self.attempt_record_control = _Control()


class _Backend:
    """The persistence surface of the real backend, without its thread pools."""

    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        RestartAgentServiceBackend as _Real,
    )

    _file_identity = _Real._file_identity
    _save_cache = _Real._save_cache
    _load_cache = _Real._load_cache
    _restore_entry = _Real._restore_entry

    def __init__(self):
        from threading import RLock

        self._lock = RLock()
        self._entries = {}
        self._path_index = {}
        self._runtime = _Runtime()
        self._cache_file = ""


def _result():
    return AnalysisResult(decision="RESTART", decision_basis="general_retry_available")


def _log(tmp_path, text="boom\n"):
    path = tmp_path / "run_4234179_date_x_cycle0.log"
    path.write_text(text)
    return str(path)


def _completed(log_path, job_id="4234179", cycle_id=0):
    return _AttemptExecution(
        key=(job_id, cycle_id),
        log_path=log_path,
        user="dnarayanan",
        job_id=job_id,
        cycle_id=cycle_id,
        status=_STATUS_COMPLETED,
        best_result=_result(),
        final_result=_result(),
        best_source="l1_enriched",
    )


@pytest.fixture
def backend():
    return _Backend()


def test_a_completed_analysis_survives_a_restart(backend, tmp_path):
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    backend._entries[("4234179", 0)] = _completed(log)
    backend._path_index[log] = ("4234179", 0)

    assert backend._save_cache(cache) is True

    restarted = _Backend()
    assert restarted._load_cache(cache) == 1
    entry = restarted._entries[("4234179", 0)]
    assert entry.status == _STATUS_COMPLETED
    assert entry.final_result.decision == "RESTART"
    assert restarted._path_index[log] == ("4234179", 0)


def test_a_log_that_grew_since_analysis_is_re_analyzed(backend, tmp_path):
    # The newest cycle log is still being appended to. Serving its cached
    # verdict would attribute a failure from minutes ago.
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    backend._entries[("4234179", 0)] = _completed(log)
    backend._save_cache(cache)

    with open(log, "a", encoding="utf-8") as handle:
        handle.write("more output after the analysis\n")

    restarted = _Backend()
    assert restarted._load_cache(cache) == 0
    assert restarted._entries == {}


def test_a_deleted_log_is_dropped(backend, tmp_path):
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    backend._entries[("4234179", 0)] = _completed(log)
    backend._save_cache(cache)
    os.unlink(log)

    assert _Backend()._load_cache(cache) == 0


def test_attempt_history_is_restored(backend, tmp_path):
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    backend._entries[("4234179", 0)] = _completed(log)
    backend._runtime.attempt_record_control._records = ()
    backend._save_cache(cache)

    # Rewrite with a record present; the store round-trips payload dicts.
    payload = json.loads(open(cache, encoding="utf-8").read())
    payload["attempt_records"] = [{"job_id": "4234179", "cycle_id": 0}]
    open(cache, "w", encoding="utf-8").write(json.dumps(payload))

    restarted = _Backend()
    restarted._load_cache(cache)

    seeded, mode = restarted._runtime.attempt_record_control.seeded
    assert mode == "replace"
    assert seeded == [{"job_id": "4234179", "cycle_id": 0}]


def test_history_is_restored_even_when_the_analysis_is_stale(backend, tmp_path):
    # History is still true about what happened; only the verdict is suspect.
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    backend._entries[("4234179", 0)] = _completed(log)
    backend._save_cache(cache)
    payload = json.loads(open(cache, encoding="utf-8").read())
    payload["attempt_records"] = [{"job_id": "4234179", "cycle_id": 0}]
    open(cache, "w", encoding="utf-8").write(json.dumps(payload))
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("grew\n")

    restarted = _Backend()
    assert restarted._load_cache(cache) == 0
    assert restarted._runtime.attempt_record_control.seeded is not None


def test_an_unknown_schema_is_ignored_not_guessed_at(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"schema": "something.v9", "entries": [{"log_path": "/x"}]}))

    assert _Backend()._load_cache(str(cache)) == 0


def test_a_missing_or_unset_cache_file_is_not_an_error(tmp_path):
    backend = _Backend()
    assert backend._load_cache("") == 0
    assert backend._save_cache("") is False
    assert backend._load_cache(str(tmp_path / "absent.json")) == 0


def test_a_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text("{not json")

    assert _Backend()._load_cache(str(cache)) == 0


def test_only_completed_analyses_are_persisted(backend, tmp_path):
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    pending = _completed(log, cycle_id=1)
    pending.status = "analyzing"
    pending.final_result = None
    pending.best_result = None
    backend._entries[("4234179", 1)] = pending

    backend._save_cache(cache)

    assert json.loads(open(cache, encoding="utf-8").read())["entries"] == []


def test_the_schema_is_stamped(backend, tmp_path):
    cache = str(tmp_path / "cache.json")
    backend._save_cache(cache)
    assert json.loads(open(cache, encoding="utf-8").read())["schema"] == _CACHE_SCHEMA
