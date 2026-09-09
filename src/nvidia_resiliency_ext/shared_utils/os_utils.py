# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
OS utilities for safe file operations.

This module provides os utilities to prevent symlink attacks and ensure
safe file operations.
"""

import os
import stat
from typing import Iterable, List, Optional, Sequence


def validate_directory(dir_path: str) -> None:
    """
    Validate that a directory is safe for file operations.

    This function performs comprehensive security checks to ensure the directory
    is safe from symlink attacks and has appropriate permissions.

    Args:
        dir_path: Path to the directory to validate

    Raises:
        OSError: If the directory is unsafe or inaccessible
    """
    if not os.path.exists(dir_path):
        return

    # Check if it's actually a directory
    if not os.path.isdir(dir_path):
        raise OSError(f"Path is not a directory: {dir_path}")

    # Check if it's a symlink
    if os.path.islink(dir_path):
        raise OSError(f"Directory is a symlink: {dir_path}")

    # Get directory stats
    try:
        dir_stat = os.stat(dir_path)
    except OSError as e:
        raise OSError(f"Cannot access directory {dir_path}: {e}")

    # Check if it's a regular directory (not a special file)
    if not stat.S_ISDIR(dir_stat.st_mode):
        raise OSError(f"Path is not a regular directory: {dir_path}")

    # Check permissions - directory should not be world-writable unless sticky bit is set
    mode = dir_stat.st_mode
    if stat.S_IWOTH & mode:  # World writable
        if not (stat.S_ISVTX & mode):  # No sticky bit
            raise OSError(f"Directory is world-writable without sticky bit: {dir_path}")


def validate_filepath(file_path: str) -> None:
    """
    Validate that a file path is safe for file operations.

    This function checks that the file (if it exists) is not a symlink and is a regular file.

    Args:
        file_path: Path to the file to validate

    Raises:
        OSError: If the file is unsafe or inaccessible
    """
    if os.path.exists(file_path):
        validate_directory(os.path.dirname(file_path))
        # Check if it's a symlink - this must be done every time
        if os.path.islink(file_path):
            raise OSError(f"File is a symlink: {file_path}")

        # Validate it's a regular file
        try:
            file_stat = os.stat(file_path)  # Follow symlinks for stat
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError(f"Path is not a regular file: {file_path}")
        except OSError as e:
            raise OSError(f"Cannot access file {file_path}: {e}")
    # If file doesn't exist, that's fine - it will be created


def normalize_allowed_roots(allowed_roots: Optional[Iterable[str]]) -> List[str]:
    """
    Resolve a set of allowed-root directories to absolute, symlink-free paths.

    Roots are resolved once at startup so that per-path confinement checks are a cheap
    lexical comparison against already-resolved strings.

    Args:
        allowed_roots: Iterable of directory paths (may be relative or contain ``~``).

    Returns:
        Deduplicated list of resolved absolute roots, preserving input order. Empty
        entries are skipped; an empty/None input yields an empty list.
    """
    normalized: List[str] = []
    for root in allowed_roots or ():
        if not root:
            continue
        real = os.path.realpath(os.path.abspath(os.path.expanduser(root)))
        if real not in normalized:
            normalized.append(real)
    return normalized


def resolve_under_allowed_roots(file_path: str, allowed_roots: Sequence[str]) -> str:
    """
    Resolve an untrusted file path and require that it stay under an allowed root.

    Fully resolves ``file_path`` (``..`` segments and symlinks on every component) and
    requires the result to live under one of ``allowed_roots``. Callers should operate on
    the returned path rather than the original: the returned path contains no symlinks,
    so opening it cannot be redirected outside the root between check and use.

    Args:
        file_path: Untrusted path, e.g. supplied by a network peer.
        allowed_roots: Non-empty roots, already passed through
            :func:`normalize_allowed_roots`.

    Returns:
        The resolved absolute path, guaranteed to be under one of ``allowed_roots``.

    Raises:
        ValueError: If ``file_path`` is empty, cannot be resolved, or resolves outside
            every allowed root.
    """
    if not file_path:
        raise ValueError("file_path cannot be empty")
    if not allowed_roots:
        raise ValueError("no allowed roots configured; refusing to resolve untrusted file_path")
    if "\x00" in file_path:
        raise ValueError("file_path contains a NUL byte")

    try:
        real = os.path.realpath(os.path.abspath(file_path))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot resolve file_path {file_path!r}: {e}") from e

    for root in allowed_roots:
        try:
            # commonpath compares whole path components, so /lustre/logs does not
            # match /lustre/logs_evil the way a plain string prefix check would.
            if os.path.commonpath([real, root]) == root:
                return real
        except ValueError:
            # Different drives / mixed absolute-relative: treat as not under this root.
            continue

    raise ValueError(
        f"file_path {file_path!r} (resolved to {real!r}) is outside the allowed "
        f"root(s) {list(allowed_roots)!r}"
    )
