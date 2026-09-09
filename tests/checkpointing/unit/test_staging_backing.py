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
"""Tests for check_local_staging_backing -- the NVRx-owned guard that a checkpoint staging dir is
backed by node-local disk (not RAM / a networked FS, which would defeat node-local staging). This
is a denylist (reject known-bad backings, allow the rest) and fails open on an unknown fstype."""

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt import filesystem_async as fa


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    fa._checked_staging_dirs.clear()
    yield
    fa._checked_staging_dirs.clear()


@pytest.mark.parametrize("bad", ["tmpfs", "ramfs", "lustre", "nfs", "gpfs", "beegfs", "cephfs"])
def test_rejects_ram_and_networked_backings(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(fa, "_staging_fstype", lambda p: bad)
    with pytest.raises(ValueError, match="RAM or networked FS"):
        fa.check_local_staging_backing(str(tmp_path))


@pytest.mark.parametrize("good", ["ext4", "xfs", "btrfs", "f2fs", "zfs", "overlay"])
def test_allows_node_local_disk_formats(tmp_path, monkeypatch, good):
    monkeypatch.setattr(fa, "_staging_fstype", lambda p: good)
    fa.check_local_staging_backing(str(tmp_path))  # no raise
    assert str(tmp_path) in fa._checked_staging_dirs


def test_unknown_fstype_fails_open(tmp_path, monkeypatch):
    # '?' (df failed / undetectable) must NOT block a possibly-valid disk -> revert to baseline
    monkeypatch.setattr(fa, "_staging_fstype", lambda p: "?")
    fa.check_local_staging_backing(str(tmp_path))  # no raise


def test_probes_each_staging_root_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fa, "_staging_fstype", lambda p: (calls.append(p), "ext4")[1])
    fa.check_local_staging_backing(str(tmp_path))
    fa.check_local_staging_backing(str(tmp_path))
    assert len(calls) == 1  # fstype is stable per mount -> probed once, then cached


def test_noop_on_falsy_dir(monkeypatch):
    # staging off (None / "") -> pure no-op, never probes (baseline path stays byte-identical)
    monkeypatch.setattr(fa, "_staging_fstype", lambda p: pytest.fail("must not probe"))
    fa.check_local_staging_backing(None)
    fa.check_local_staging_backing("")
