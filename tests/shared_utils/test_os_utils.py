# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared_utils/os_utils.py path helpers."""

import os

import pytest

from nvidia_resiliency_ext.shared_utils.os_utils import (
    normalize_allowed_roots,
    resolve_under_allowed_roots,
)


@pytest.fixture
def log_root(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    # realpath: on some platforms the pytest tmp dir itself sits behind a symlink.
    return os.path.realpath(str(root))


@pytest.fixture
def outside_dir(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    return os.path.realpath(str(outside))


class TestResolveUnderAllowedRoots:
    """Unit tests for the shared confinement helper."""

    def test_accepts_path_inside_root(self, log_root):
        target = os.path.join(log_root, "train.log")
        assert resolve_under_allowed_roots(target, [log_root]) == target

    def test_accepts_not_yet_existing_nested_path(self, log_root):
        target = os.path.join(log_root, "nested", "deeper", "train.log")
        assert resolve_under_allowed_roots(target, [log_root]) == target

    def test_rejects_absolute_path_outside_root(self, log_root, outside_dir):
        with pytest.raises(ValueError, match="outside the allowed"):
            resolve_under_allowed_roots(os.path.join(outside_dir, "owned.txt"), [log_root])

    def test_rejects_dotdot_traversal(self, log_root, outside_dir):
        escape = os.path.join(log_root, "..", "outside", "owned.txt")
        with pytest.raises(ValueError, match="outside the allowed"):
            resolve_under_allowed_roots(escape, [log_root])

    def test_rejects_symlink_escape(self, log_root, outside_dir):
        os.symlink(outside_dir, os.path.join(log_root, "link"))
        with pytest.raises(ValueError, match="outside the allowed"):
            resolve_under_allowed_roots(os.path.join(log_root, "link", "owned.txt"), [log_root])

    def test_rejects_symlinked_file_pointing_outside(self, log_root, outside_dir):
        victim = os.path.join(outside_dir, "authorized_keys")
        open(victim, "w").close()
        link = os.path.join(log_root, "train.log")
        os.symlink(victim, link)
        with pytest.raises(ValueError, match="outside the allowed"):
            resolve_under_allowed_roots(link, [log_root])

    def test_rejects_sibling_root_prefix(self, tmp_path):
        """commonpath compares components: /a/logs must not match /a/logs_evil."""
        root = os.path.realpath(str(tmp_path / "logs"))
        os.makedirs(root)
        evil = os.path.realpath(str(tmp_path / "logs_evil"))
        os.makedirs(evil)
        with pytest.raises(ValueError, match="outside the allowed"):
            resolve_under_allowed_roots(os.path.join(evil, "x.log"), [root])

    def test_rejects_empty_path_and_missing_roots(self, log_root):
        with pytest.raises(ValueError):
            resolve_under_allowed_roots("", [log_root])
        with pytest.raises(ValueError, match="no allowed roots"):
            resolve_under_allowed_roots(os.path.join(log_root, "a.log"), [])

    def test_rejects_nul_byte(self, log_root):
        with pytest.raises(ValueError, match="NUL"):
            resolve_under_allowed_roots(os.path.join(log_root, "a\x00.log"), [log_root])

    def test_multiple_roots_accepts_either(self, log_root, outside_dir):
        roots = [log_root, outside_dir]
        assert resolve_under_allowed_roots(os.path.join(outside_dir, "b.log"), roots)
        assert resolve_under_allowed_roots(os.path.join(log_root, "a.log"), roots)

    def test_normalize_dedupes_and_absolutizes(self, log_root):
        assert normalize_allowed_roots([log_root, log_root, "", None]) == [log_root]
        assert normalize_allowed_roots(None) == []
