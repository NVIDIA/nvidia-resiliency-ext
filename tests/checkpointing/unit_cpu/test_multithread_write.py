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

import queue
import threading

from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync


def test_buckets_are_written_concurrently(tmp_path):
    """Every bucket file is open at once."""
    buckets = [(str(tmp_path / f"{i}.distcp"), "key", ([], [])) for i in range(3)]
    all_open = threading.Barrier(len(buckets), timeout=2)

    def open_file(path, mode):  # returns only once every bucket is being written
        all_open.wait()
        return open(path, mode)

    results = queue.Queue()
    FileSystemWriterAsync.write_preloaded_data_multithread(
        [], False, open_file, 0, buckets, results
    )
    result = results.get_nowait()
    assert not isinstance(
        result, threading.BrokenBarrierError
    ), "buckets were not written concurrently"
    assert result == {0: [], 1: [], 2: []}
