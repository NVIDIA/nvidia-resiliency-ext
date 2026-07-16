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
"""Tests for FileSystemWriterAsync.flush_dst -- the authoritative durable-destination the async
worker's flusher must copy staged shards to. This is the single source callers use for
AsyncRequest.flush_dst so the request's flush target can never diverge from where the writer
actually staged (a divergence would finalize a local-only, torn-on-load checkpoint)."""

from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync


def _writer(staging: bool, checkpoint_dir: str):
    # bypass the heavy torch FileSystemWriter __init__; exercise only the flush_dst property logic
    w = FileSystemWriterAsync.__new__(FileSystemWriterAsync)
    w.checkpoint_dir = checkpoint_dir
    w.staging = staging
    return w


def test_flush_dst_is_durable_dir_when_staging():
    w = _writer(staging=True, checkpoint_dir="/lustre/durable/iter_0000100")
    assert w.flush_dst == "/lustre/durable/iter_0000100"


def test_flush_dst_is_none_when_not_staging():
    w = _writer(staging=False, checkpoint_dir="/lustre/durable/iter_0000100")
    assert w.flush_dst is None
