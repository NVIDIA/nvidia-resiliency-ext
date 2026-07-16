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
"""Tests for the bounded blocking finalize wait (graceful shutdown safety).

A node-local-staging save whose durable flush is wedged never posts ``comp_q``; an unbounded
blocking finalize would then hang shutdown before the bounded terminate path can run. The
``finalize_timeout`` bounds that wait so the caller (AsyncCallsQueue.close) can proceed to
force-terminate the worker.
"""

import time
from queue import Queue

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import PersistentAsyncCaller


def _caller(comp_q):
    # bypass the heavy __init__; only the completion-poll fields matter here
    c = PersistentAsyncCaller.__new__(PersistentAsyncCaller)
    c.process = object()  # truthy -> enter the completion-wait loop
    c.comp_q = comp_q
    c.cur_item = None
    c.rank = 0
    return c


def test_blocking_finalize_is_bounded_by_finalize_timeout():
    # empty comp_q simulates a wedged durable flush (completion never posted)
    c = _caller(Queue())
    t0 = time.time()
    done = c.is_current_async_call_done(blocking=True, no_dist=True, finalize_timeout=0.3)
    dt = time.time() - t0
    assert done is False  # timed out -> reported still-active so close() can escalate
    assert 0.3 <= dt < 3.0  # bounded -- did NOT hang


def test_blocking_finalize_returns_done_when_completion_posted():
    q = Queue()
    q.put(7)  # completion already available
    c = _caller(q)
    done = c.is_current_async_call_done(blocking=True, no_dist=True, finalize_timeout=0.3)
    assert done is True  # returns immediately, well within the timeout
