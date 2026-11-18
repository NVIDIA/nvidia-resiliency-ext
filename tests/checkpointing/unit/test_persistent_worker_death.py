# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A dead persistent checkpoint worker must not hang the trainer.

The requests use builtins so that the spawned worker can unpickle them without importing test
code: `abs` is a call the worker finishes, and `os._exit` is one it dies in.
"""

import functools
import multiprocessing.context
import os
from unittest import mock

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt import core


@pytest.fixture
def caller():
    caller = core.PersistentAsyncCaller(cpu_shm_mode=True)
    caller.rank = 0
    yield caller
    caller.close(abort=True)


def _request(call_idx, async_fn, *async_fn_args, preload_fn=None):
    return core.AsyncRequest(async_fn, async_fn_args, [], preload_fn=preload_fn, call_idx=call_idx)


def test_a_worker_dying_while_preloading_does_not_block_staging(caller):
    exit_ = functools.partial(os._exit, 1)
    save = caller.submit(_request(0, abs, 0, None, preload_fn=exit_))

    assert save.staged(blocking=True)
    assert save.finished(blocking=True)


def test_a_new_worker_takes_over_from_a_dead_one(caller):
    assert caller.submit(_request(0, os._exit, 1)).finished(blocking=True)

    assert caller.submit(_request(1, abs, 0)).finished(blocking=True)
    assert caller._debug_is_async_process_running()


def test_a_worker_that_failed_to_start_is_started_again(caller):
    with mock.patch.object(
        multiprocessing.context.SpawnProcess, "start", side_effect=OSError("no fork")
    ):
        with pytest.raises(OSError):
            caller.submit(_request(0, abs, 0))

    assert caller.submit(_request(1, abs, 0)).finished(blocking=True)
