# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the node-local-staging persist-remote flush + straggler guard (async_ckpt.flush).

These cover the pure filesystem/thread logic (no torch / distributed): env config parsing, the
no-hedge (destination) errno fail-fast, the happy-path atomic commit, the hedge-on-deadline path,
the transient-failure retry contract, and local-stage GC.
"""
import errno
import os
import stat

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt import flush


@pytest.fixture(autouse=True)
def _reset_test_injection_budgets():
    """The stall/fail injections consume a per-process one-shot budget (module globals); reset
    them around every test so cases don't leak into each other."""
    flush._stall_used = 0
    flush._fail_used = 0
    yield
    flush._stall_used = 0
    flush._fail_used = 0


def _stage_shard(stage_dir, rel_key, payload=b"x" * 4096):
    os.makedirs(stage_dir, exist_ok=True)
    p = os.path.join(stage_dir, rel_key)
    with open(p, "wb") as f:
        f.write(payload)
    return p


# --------------------------------------------------------------------------- config


def test_from_env_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith("NVRX_CKPT_FLUSH_"):
            monkeypatch.delenv(k, raising=False)
    c = flush.FlushConfig.from_env()
    assert c.shard_timeout_secs == 120.0
    assert c.max_retry == 3
    assert c.max_concurrent_hedges == 4
    assert c.retry_backoff_cap_secs == 600.0
    assert c.stall_secs == 0.0 and c.stall_count == 1 and c.stall_rank == -1
    assert c.fail_count == 0 and c.fail_rank == -1


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("NVRX_CKPT_FLUSH_SHARD_TIMEOUT_SECS", "45")
    monkeypatch.setenv("NVRX_CKPT_FLUSH_MAX_RETRY", "7")
    monkeypatch.setenv("NVRX_CKPT_FLUSH_MAX_CONCURRENT_HEDGES", "9")
    monkeypatch.setenv("NVRX_CKPT_FLUSH_RETRY_BACKOFF_CAP_SECS", "300")
    monkeypatch.setenv("NVRX_CKPT_FLUSH_TEST_STALL_SECS", "2.5")
    monkeypatch.setenv("NVRX_CKPT_FLUSH_TEST_STALL_COUNT", "3")
    monkeypatch.setenv("NVRX_CKPT_FLUSH_TEST_FAIL_COUNT", "2")
    monkeypatch.setenv("NVRX_CKPT_FLUSH_TEST_FAIL_RANK", "0")
    c = flush.FlushConfig.from_env()
    assert (c.shard_timeout_secs, c.max_retry, c.max_concurrent_hedges) == (45.0, 7, 9)
    assert c.retry_backoff_cap_secs == 300.0
    assert (c.stall_secs, c.stall_count) == (2.5, 3)
    assert (c.fail_count, c.fail_rank) == (2, 0)


def test_no_hedge_errnos_membership():
    # destination-level failures where a fresh OST can't help -> fail fast (no hedge)
    for e in (errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EPERM):
        assert e in flush._NO_HEDGE_ERRNOS
    # a transient I/O error is NOT in the set -> it is hedged, not failed fast
    assert errno.EIO not in flush._NO_HEDGE_ERRNOS


# --------------------------------------------------------------------------- happy path


def test_flush_call_commits_all_shards(tmp_path):
    stage, dst = tmp_path / "stage", tmp_path / "durable"
    payloads = {f"__{i}_0.distcp": os.urandom(2048) for i in range(3)}
    file_map = [(_stage_shard(stage, k, v), k) for k, v in payloads.items()]
    ok, reason = flush.flush_call(file_map, str(dst), rank=0, cfg=flush.FlushConfig())
    assert ok is True and reason is None
    for k, v in payloads.items():
        assert (dst / k).read_bytes() == v  # byte-identical durable copy
    assert not list((dst).glob("*.t*"))  # no temp left behind


# --------------------------------------------------------------------------- no-hedge fail-fast


def test_flush_call_failfast_on_destination_errno(tmp_path):
    stage = tmp_path / "stage"
    file_map = [(_stage_shard(stage, "__0_0.distcp"), "__0_0.distcp")]
    ro = tmp_path / "readonly"
    ro.mkdir()
    os.chmod(ro, stat.S_IREAD | stat.S_IEXEC)  # EACCES on write -> a no-hedge errno
    try:
        cfg = flush.FlushConfig(shard_timeout_secs=2.0, max_retry=1)
        ok, reason = flush.flush_call(file_map, str(ro), rank=0, cfg=cfg)
    finally:
        os.chmod(ro, stat.S_IRWXU)
    assert ok is False
    assert "destination errno" in reason and "__0_0.distcp" in reason
    assert (
        "permanent" not in reason.lower()
    )  # honest wording: not "permanent", it IS retried upstream


# --------------------------------------------------------------------------- transient retry contract


def test_flush_call_test_fail_injection_then_recovers(tmp_path):
    stage, dst = tmp_path / "stage", tmp_path / "durable"
    file_map = [(_stage_shard(stage, "__0_0.distcp", b"data"), "__0_0.distcp")]
    cfg = flush.FlushConfig(fail_count=2, fail_rank=0)
    # first two calls are the injected transient failure; the third really commits
    r1 = flush.flush_call(file_map, str(dst), rank=0, cfg=cfg)
    r2 = flush.flush_call(file_map, str(dst), rank=0, cfg=cfg)
    r3 = flush.flush_call(file_map, str(dst), rank=0, cfg=cfg)
    assert r1[0] is False and "TEST" in r1[1]
    assert r2[0] is False and "TEST" in r2[1]
    assert r3 == (True, None)
    assert (dst / "__0_0.distcp").read_bytes() == b"data"


def test_test_fail_injection_only_targets_fail_rank(tmp_path):
    stage, dst = tmp_path / "stage", tmp_path / "durable"
    file_map = [(_stage_shard(stage, "__5_0.distcp", b"data"), "__5_0.distcp")]
    cfg = flush.FlushConfig(fail_count=5, fail_rank=0)
    ok, reason = flush.flush_call(file_map, str(dst), rank=5, cfg=cfg)  # rank 5 != fail_rank 0
    assert ok is True and reason is None


# --------------------------------------------------------------------------- hedge on deadline


def test_hedge_wins_when_attempt0_stalls(tmp_path):
    """attempt-0 stalls past a short deadline (injected); a fresh hedge must win and commit."""
    stage, dst = tmp_path / "stage", tmp_path / "durable"
    file_map = [(_stage_shard(stage, "__0_0.distcp", b"payload"), "__0_0.distcp")]
    # stall_secs > shard_timeout_secs so attempt-0 blows the deadline -> the loop hedges a fresh copy
    cfg = flush.FlushConfig(
        shard_timeout_secs=0.3, max_retry=3, max_concurrent_hedges=4, stall_secs=1.5, stall_count=1
    )
    ok, reason = flush.flush_call(file_map, str(dst), rank=0, cfg=cfg)
    assert ok is True and reason is None
    assert (dst / "__0_0.distcp").read_bytes() == b"payload"


# --------------------------------------------------------------------------- GC


def test_gc_local_stage_removes_own_shards_and_dir(tmp_path):
    iterd = tmp_path / "iter_0000100"
    p0 = _stage_shard(iterd, "__0_0.distcp")
    p1 = _stage_shard(iterd, "__1_0.distcp")
    # gc only the shards it is told about; when the iter dir empties it is rmdir'd
    flush.gc_local_stage([(p0, "__0_0.distcp"), (p1, "__1_0.distcp")])
    assert not os.path.exists(p0) and not os.path.exists(p1)
    assert not iterd.exists()  # emptied dir removed


def test_gc_local_stage_keeps_dir_with_other_shards(tmp_path):
    iterd = tmp_path / "iter_0000100"
    p0 = _stage_shard(iterd, "__0_0.distcp")
    _stage_shard(iterd, "__1_0.distcp")  # another rank's shard, not gc'd
    flush.gc_local_stage([(p0, "__0_0.distcp")])
    assert not os.path.exists(p0)
    assert iterd.exists()  # dir kept because rank 1's shard remains
