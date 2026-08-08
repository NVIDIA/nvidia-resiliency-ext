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

"""persist-remote flush with a straggler guard (node-local checkpoint staging, design §8).

The flusher (a daemon thread inside the persistent async worker) copies staged shards from
the node-local stage dir to the durable dir (Lustre). A wedged Lustre write is
**uninterruptible** (``D`` state -- ``SIGKILL`` stays pending until the syscall returns via
Lustre's own timeout), so the guard is **hedge, not interrupt**: leave the stuck copy alone,
race a fresh copy to a temp, first-to-finish wins, atomic in-dir ``rename`` commits it, and the
abandoned attempt unlinks its own temp when it eventually returns.

Config is read from the environment (Megatron exports the ``--ckpt-flush-*`` knobs from its
args before the worker is spawned; the spawn context inherits them). No ``lfs`` dependency: each
hedge is a fresh temp and Lustre assigns its OST at creation via round-robin, so the retry itself
supplies OST diversity -- for a single ~1/350 straggler a hedge almost always lands on a healthy
OST. (A multi-OST fabric event is instead absorbed by skip-a-save backpressure upstream.)
"""

import errno
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Copy failures where HEDGING is futile -- a fresh copy to a different OST fails identically because
# the problem is the destination (quota/space/perm), not a slow OST. Skip the hedge storm and fail
# fast; the flusher still RETRIES the whole call with backoff, so a freed quota self-heals.
_NO_HEDGE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EPERM})


@dataclass
class FlushConfig:
    """Straggler-guard knobs (design §9), sourced from env so they reach the spawned worker."""

    shard_timeout_secs: float = 120.0  # hedge trigger: a copy this slow is treated as stuck
    max_retry: int = 3  # hedge attempts per shard before giving up this cycle
    max_concurrent_hedges: int = 4  # cap outstanding (incl. stuck) attempts per shard
    retry_backoff_cap_secs: float = 600.0  # max backoff between whole-call retries
    # --- test injection (verify the guard without a real slow/dead OST) ---
    stall_secs: float = 0.0  # make attempt-0 sleep this long, simulating a stuck write
    stall_count: int = 1  # stall only the first N shards per worker (clean, bounded)
    stall_rank: int = -1  # -1 = all ranks; else only this rank's worker stalls
    fail_count: int = 0  # fail the first N flush calls (transient) -> exercises retry
    fail_rank: int = -1  # -1 = all ranks; else only this rank's worker fails

    @classmethod
    def from_env(cls) -> "FlushConfig":
        g = os.environ.get

        def _f(k, d):
            v = g(k)
            return float(v) if v not in (None, "") else d

        def _i(k, d):
            v = g(k)
            return int(v) if v not in (None, "") else d

        return cls(
            shard_timeout_secs=_f("NVRX_CKPT_FLUSH_SHARD_TIMEOUT_SECS", 120.0),
            max_retry=_i("NVRX_CKPT_FLUSH_MAX_RETRY", 3),
            max_concurrent_hedges=_i("NVRX_CKPT_FLUSH_MAX_CONCURRENT_HEDGES", 4),
            retry_backoff_cap_secs=_f("NVRX_CKPT_FLUSH_RETRY_BACKOFF_CAP_SECS", 600.0),
            stall_secs=_f("NVRX_CKPT_FLUSH_TEST_STALL_SECS", 0.0),
            stall_count=_i("NVRX_CKPT_FLUSH_TEST_STALL_COUNT", 1),
            stall_rank=_i("NVRX_CKPT_FLUSH_TEST_STALL_RANK", -1),
            fail_count=_i("NVRX_CKPT_FLUSH_TEST_FAIL_COUNT", 0),
            fail_rank=_i("NVRX_CKPT_FLUSH_TEST_FAIL_RANK", -1),
        )


# One-shot test-stall budget, per worker process (module state is per-process under spawn).
_stall_lock = threading.Lock()
_stall_used = 0
# One-shot test-fail budget: fail the first fail_count flush calls (transient) to exercise the retry.
_fail_lock = threading.Lock()
_fail_used = 0


def _maybe_test_fail(cfg: "FlushConfig", rank: int) -> bool:
    """TEST: inject a transient flush failure for the first ``fail_count`` calls (on ``fail_rank``),
    so the flusher's retry / auto-recovery can be validated without a real FS fault."""
    if cfg.fail_count <= 0:
        return False
    if cfg.fail_rank >= 0 and rank != cfg.fail_rank:
        return False
    global _fail_used
    with _fail_lock:
        if _fail_used < cfg.fail_count:
            _fail_used += 1
            return True
    return False


def _maybe_test_stall(cfg: FlushConfig, rank: int) -> float:
    """Return the stall duration for the next attempt-0, consuming the one-shot budget."""
    if cfg.stall_secs <= 0:
        return 0.0
    if cfg.stall_rank >= 0 and rank != cfg.stall_rank:
        return 0.0
    global _stall_used
    with _stall_lock:
        if _stall_used < cfg.stall_count:
            _stall_used += 1
            return cfg.stall_secs
    return 0.0


def _copy_file(src: str, dst: str) -> None:
    with open(src, "rb") as _s, open(dst, "wb") as _d:
        shutil.copyfileobj(_s, _d, 16 * 1024 * 1024)
        _d.flush()
        os.fsync(_d.fileno())


def _fsync_dir(d: str) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _flush_one_shard(
    stage_path: str, dst: str, cfg: FlushConfig, rank: int
) -> Tuple[bool, Optional[str]]:
    """Hedged copy of ONE shard local->durable. Returns ``(committed, reason)`` -- ``(True, None)``
    once atomically committed, else ``(False, "<why>")`` with the failure cause (errno / deadline).

    Spawns attempt-0 (natural OST allocation); on a blown ``shard_timeout_secs`` deadline it hedges
    a fresh copy to a new temp (up to ``max_retry``, capped by ``max_concurrent_hedges`` outstanding
    -- a stuck attempt still counts as it sits in ``D``). First attempt to finish wins the ``winner``
    slot; the winner is renamed onto ``dst`` (atomic within a Lustre dir); every other attempt unlinks
    its own temp when it returns. No ALERT here -- the caller (flusher) logs one call-level alert with
    the stuck-duration and retries the whole call in place, so per-shard/per-retry spam is avoided.
    """
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(dst)

    lock = threading.Lock()
    winner = {"tmp": None, "k": None}
    nohedge = {
        "errno": None,
        "msg": None,
    }  # set when an attempt hits a no-hedge (destination) error
    attempts = []  # list of (k, tmp, thread)

    def copy_worker(k: int, tmp: str, stall: float) -> None:
        try:
            if stall > 0:
                time.sleep(stall)  # TEST: simulate a stuck-OST write
            _copy_file(stage_path, tmp)
        except OSError as e:  # a failed attempt just abandons its temp
            if e.errno in _NO_HEDGE_ERRNOS:
                with lock:
                    if nohedge["errno"] is None:
                        nohedge["errno"], nohedge["msg"] = e.errno, str(e)
            logger.debug(f"{rank}: flush attempt {k} of {base} failed: {e}")
            _safe_unlink(tmp)
            return
        except Exception as e:
            logger.debug(f"{rank}: flush attempt {k} of {base} failed: {e}")
            _safe_unlink(tmp)
            return
        with lock:
            won = winner["tmp"] is None
            if won:
                winner["tmp"], winner["k"] = tmp, k
        if not won:  # a slower attempt lost the race -> clean up after itself
            _safe_unlink(tmp)

    def start_attempt(k: int, stall: float) -> None:
        tmp = f"{dst}.t{k}"
        t = threading.Thread(
            target=copy_worker, args=(k, tmp, stall), name=f"flush-{rank}-{base}-t{k}", daemon=True
        )
        attempts.append((k, tmp, t))
        t.start()

    start_attempt(0, _maybe_test_stall(cfg, rank))
    k = 0
    # Stop hedging on a no-hedge error (nohedge set: quota/space/perm -- a destination problem, not
    # a slow OST): a fresh copy would fail identically, so hedging only burns attempts and floods the
    # log -- fail fast here; the flusher then retries the whole call with backoff (recovers on quota).
    while winner["tmp"] is None and nohedge["errno"] is None and k < cfg.max_retry:
        attempts[-1][2].join(timeout=cfg.shard_timeout_secs)
        if winner["tmp"] is not None or nohedge["errno"] is not None:
            break
        outstanding = sum(1 for _, _, t in attempts if t.is_alive())
        if outstanding < cfg.max_concurrent_hedges:
            k += 1
            # Distinguish a straggler (attempt still running past the deadline) from a transient
            # error (attempt already returned without a no-hedge errno) so the message is honest.
            why = (
                f"exceeded {cfg.shard_timeout_secs:.0f}s (attempt {k - 1} likely on a stuck OST)"
                if attempts[-1][2].is_alive()
                else f"attempt {k - 1} failed transiently"
            )
            # DEBUG, not WARNING: this fires per rank * per shard * per hedge, so at scale a
            # fabric-wide stall would emit thousands of lines. The operator-facing signal is the
            # single rank-0 ALERT in the flusher (core.py) + the rank-0 pending_remote line.
            logger.debug(f"{rank}: persist-remote of {base} {why}; hedging attempt {k}")
            start_attempt(k, 0.0)  # hedges never stall
        else:
            logger.debug(
                f"{rank}: persist-remote of {base} at hedge cap ({cfg.max_concurrent_hedges}); "
                f"waiting on outstanding attempts before spawning more"
            )
            # At the cap: keep waiting on whatever is outstanding (loop re-joins the newest).
            time.sleep(min(cfg.shard_timeout_secs, 5.0))

    # Retries exhausted but attempts may still be racing -> one bounded final wait for a winner
    # (skip when a no-hedge error already doomed this shard).
    if winner["tmp"] is None and nohedge["errno"] is None:
        for _, _, t in attempts:
            t.join(timeout=cfg.shard_timeout_secs)
            if winner["tmp"] is not None:
                break

    if winner["tmp"] is None:
        if nohedge["errno"] is not None:
            return (
                False,
                f"destination errno {nohedge['errno']} ({nohedge['msg']}) — free space/quota",
            )
        return (
            False,
            f"all {len(attempts)} attempt(s) blew the {cfg.shard_timeout_secs:.0f}s deadline",
        )

    os.rename(winner["tmp"], dst)  # atomic in-dir commit
    if dst_dir:
        _fsync_dir(dst_dir)
    if winner["k"] > 0:
        # DEBUG, not INFO: fires per rank * per shard whenever a hedge wins, i.e. exactly during a
        # wide straggler event when many ranks hedge at once. The rank-0 [ckpt-timing] line already
        # reflects the (hedged) persist-remote duration; keep this as per-rank detail only.
        logger.debug(
            f"{rank}: persist-remote of {base} committed via hedge (attempt {winner['k']} won; "
            f"{len(attempts)} attempt(s) spawned)"
        )
    return True, None


def gc_local_stage(file_map) -> None:
    """Drop this rank's node-local staged shards once their durable flush has committed (design
    §5.5). Load is Lustre-only, so a committed local copy is dead weight; deleting it here bounds
    local disk to |U| (staged-but-not-durable) <= K -- WITHOUT this the local stage dir grows one
    iter per save forever. Unlinks only THIS rank's shard files (race-free on a node-shared stage
    dir) then best-effort rmdir's the emptying iter dir (succeeds for the last local rank)."""
    dirs = set()
    for stage_path, _rel_key in file_map:
        _safe_unlink(stage_path)
        dirs.add(os.path.dirname(stage_path))
    for d in dirs:
        try:
            os.rmdir(d)  # only when every local rank's shard for this iter is gone
        except OSError:
            pass


def flush_call(file_map, flush_dst: str, rank: int, cfg: FlushConfig) -> Tuple[bool, Optional[str]]:
    """Flush every shard of one save (persist-remote). Returns ``(committed, reason)`` -- committed
    is True only if ALL shards committed; on failure reason names the first failing shard's cause.

    When not durable the caller withholds ``comp_q`` (finalize/.metadata/tracker never fire for a
    partial save) and retries the whole call, so the iter stays in ``|U|`` and skip-a-save throttles
    new saves until the backlog clears / the flush recovers.
    """
    if _maybe_test_fail(cfg, rank):
        return False, "TEST: injected transient failure"
    all_ok = True
    reason = None
    for stage_path, rel_key in file_map:
        dst = os.path.join(flush_dst, rel_key)
        ok, r = _flush_one_shard(stage_path, dst, cfg, rank)
        if not ok:
            all_ok = False
            if reason is None:  # keep the first failing shard's cause for the flusher's alert
                reason = f"{os.path.basename(dst)}: {r}"
    return all_ok, reason
