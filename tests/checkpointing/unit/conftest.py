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

import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterator

import pytest
import torch

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue

from .test_utilities import Utils


@pytest.fixture(scope="session")
def _run_id() -> str:
    """A random name for this run's scratch directories, the same on every rank.

    Other runs on the machine, from another worktree or branch, get different names, so runs
    can't delete each other's files.
    """
    Utils.initialize_distributed()
    run_id = [uuid.uuid4().hex if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(run_id, src=0)
    return run_id[0]


def _make_shared_dir(path: Path) -> None:
    """Create a new directory that every rank uses.

    Rank 0 creates it and fails if it already exists, which would mean two users of one
    directory. The other ranks wait until it exists.
    """
    if torch.distributed.get_rank() == 0:
        path.mkdir(parents=True)
    torch.distributed.barrier()


def _scratch_root(parent: str, run_id: str) -> Iterator[Path]:
    root = Path(parent) / f"nvrx-test-{run_id}"
    _make_shared_dir(root)
    yield root
    if Utils.rank == 0:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def _tmp_root(_run_id: str) -> Iterator[Path]:
    yield from _scratch_root(tempfile.gettempdir(), _run_id)


@pytest.fixture(scope="session")
def _shm_root(_run_id: str) -> Iterator[Path]:
    yield from _scratch_root("/dev/shm", _run_id)


def _test_dir(root: Path, request: pytest.FixtureRequest) -> Path:
    """`root/<module>/<class>/<test>`, so tests never share a directory.

    Test names repeat across modules and classes, so the path includes both.
    """
    parts = [request.node.path.stem, request.cls.__name__ if request.cls else "", request.node.name]
    path = root.joinpath(*(re.sub(r"[^\w.-]", "_", part) for part in parts if part))
    _make_shared_dir(path)
    return path


@pytest.fixture
def tmp_path_dist_ckpt(_tmp_root: Path, request: pytest.FixtureRequest) -> Path:
    """Directory for this test's checkpoints, shared by the ranks of this run."""
    return _test_dir(_tmp_root, request)


@pytest.fixture
def shm_path_dist_ckpt(_shm_root: Path, request: pytest.FixtureRequest) -> Path:
    """Like `tmp_path_dist_ckpt`, but in RAM (/dev/shm)."""
    return _test_dir(_shm_root, request)


@pytest.fixture
def async_queue():
    async_queue = AsyncCallsQueue()
    yield async_queue
    async_queue.close()
