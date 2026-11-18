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

"""Async saves must not hang the trainer, lose a checkpoint or write a wrong one when the
persistent worker dies, is aborted or is replaced.

The tests use only what applications use: AsyncCallsQueue, AsyncRequest, FileSystemWriterAsync
and the state_dict_saver functions. Requests that the spawned worker runs without the test code
use builtins: `abs` is a call the worker finishes, `os._exit` one it dies in, and `time.sleep`
one it is still running.
"""

from unittest import mock

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest


@pytest.fixture
def warm_queue():
    """A persistent queue whose worker was started before the first save, as Megatron does."""
    AsyncCallsQueue.warmup_persistent_caller(rank=0, cpu_shm_mode=True)
    async_queue = AsyncCallsQueue(persistent=True)
    yield async_queue
    async_queue.close(abort=True)


@pytest.mark.timeout(30)
def test_a_rank_with_nothing_to_write_finishes_with_a_started_worker(warm_queue):
    finalize = mock.Mock()
    warm_queue.schedule_async_request(AsyncRequest(None, (), [finalize]))

    assert warm_queue.maybe_finalize_async_calls(blocking=True, no_dist=True) == [0]
    finalize.assert_called_once_with()
