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


class _Config:
    """Just the fingerprint: a cached decision is only valid under its policy."""

    def __init__(self, fingerprint="cfg-a"):
        self.config_fingerprint = fingerprint


class _Backend:
    """The persistence surface of the real backend, without its thread pools."""

    from nvidia_resiliency_ext.services.attrsvc.restart_agent_backend import (
        RestartAgentServiceBackend as _Real,
    )

    _file_identity = _Real._file_identity
    _save_cache = _Real._save_cache
    _load_cache = _Real._load_cache
    _restore_entry = _Real._restore_entry

    def __init__(self, fingerprint="cfg-a"):
        from threading import Lock, RLock

        self._lock = RLock()
        self._save_lock = Lock()
        self._entries = {}
        self._path_index = {}
        self._runtime = _Runtime()
        self._config = _Config(fingerprint)
        self._cache_file = ""


def _result():
    return AnalysisResult(decision="RESTART", decision_basis="general_retry_available")


def _log(tmp_path, text="boom\n"):
    path = tmp_path / "run_4234179_date_x_cycle0.log"
    path.write_text(text)
    return str(path)


def _completed(log_path, job_id="4234179", cycle_id=0, identity=None):
    # The real backend stats the source when the analysis finishes, not when
    # the cache is written; the stand-in has to do the same or the tests would
    # not exercise what production stores.
    if identity is None:
        stat = os.stat(log_path)
        identity = {"mtime": stat.st_mtime, "size": stat.st_size}
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
        source_identity=identity,
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


def test_a_log_that_grew_between_analysis_and_save_is_not_restored(backend, tmp_path):
    # The identity has to describe the bytes that were analyzed. Stat'ing at
    # save time instead pairs an old verdict with the file's later size, and a
    # stale result then validates as byte-for-byte current.
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    backend._entries[("4234179", 0)] = _completed(log)

    # Another job finishes and triggers a save after this log has moved on.
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("appended before anyone saved\n")
    backend._save_cache(cache)

    restarted = _Backend()
    assert restarted._load_cache(cache) == 0
    assert restarted._entries == {}


def test_decisions_from_a_different_policy_are_not_restored(backend, tmp_path):
    # A completed entry is never re-analyzed, so restoring across a budget or
    # policy change would keep returning advice the current policy would not
    # give. History is still seeded: it records what happened, not a verdict.
    log = _log(tmp_path)
    cache = str(tmp_path / "cache.json")
    backend._entries[("4234179", 0)] = _completed(log)
    backend._runtime.attempt_record_control._records = ()
    backend._save_cache(cache)
    payload = json.loads(open(cache, encoding="utf-8").read())
    payload["attempt_records"] = [{"job_id": "4234179", "cycle_id": 0}]
    open(cache, "w", encoding="utf-8").write(json.dumps(payload))

    restarted = _Backend(fingerprint="cfg-b")
    assert restarted._load_cache(cache) == 0
    assert restarted._entries == {}
    assert restarted._runtime.attempt_record_control.seeded is not None


def test_the_config_fingerprint_is_stamped(backend, tmp_path):
    cache = str(tmp_path / "cache.json")
    backend._save_cache(cache)
    saved = json.loads(open(cache, encoding="utf-8").read())
    assert saved["config_fingerprint"] == "cfg-a"


def test_concurrent_saves_leave_readable_json(backend, tmp_path):
    # The analysis workers are threads in one process, so a pid-qualified temp
    # name is shared between them. Unserialized, one saver truncates the file
    # another is writing and the published cache is corrupt.
    from concurrent.futures import ThreadPoolExecutor

    cache = str(tmp_path / "cache.json")
    for cycle in range(40):
        log = tmp_path / f"run_4234179_date_x_cycle{cycle}.log"
        log.write_text("boom\n" * (cycle + 1))
        backend._entries[("4234179", cycle)] = _completed(str(log), cycle_id=cycle)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in [pool.submit(backend._save_cache, cache) for _ in range(24)]]
    assert all(results)

    payload = json.loads(open(cache, encoding="utf-8").read())
    assert payload["schema"] == _CACHE_SCHEMA
    assert len(payload["entries"]) == 40
    assert not list(tmp_path.glob("cache.json.tmp*"))
