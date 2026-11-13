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

""" FS Reader with metadata cached support. """

import os
from typing import ClassVar

from torch.distributed.checkpoint import FileSystemReader, Metadata

from ..filesystem import CheckpointFileSystem, LocalFileSystem

_LOCAL_FILESYSTEM = LocalFileSystem()


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Metadata is shared across all reader instances that use the same checkpoint
    directory and filesystem object, since the loaded metadata is identical.

    Attributes:
        _metadata_cache: Metadata and filesystem references keyed by backend identity and path.
    """

    _metadata_cache: ClassVar[dict[tuple[int, str], tuple[CheckpointFileSystem, Metadata]]] = {}

    def __init__(
        self,
        path: str | os.PathLike,
        cache_metadata: bool = True,
        *,
        filesystem: CheckpointFileSystem | None = None,
    ) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
            cache_metadata (bool): If True (default), metadata is cached at the class level
                and shared across all instances pointing to the same checkpoint directory.
                If False, metadata is always read from disk on each call.
        """
        super().__init__(path=path)
        self.fs = filesystem if filesystem is not None else _LOCAL_FILESYSTEM
        self.path = self.fs.init_path(path)
        self._cache_metadata = cache_metadata
        self._update_cache_key()

    def _update_cache_key(self):
        cache_path = os.fspath(self.path)
        if self.fs is _LOCAL_FILESYSTEM:
            cache_path = os.path.abspath(cache_path)
        self._cache_key = (id(self.fs), cache_path) if self._cache_metadata else None

    def reset(self, checkpoint_id=None) -> None:
        super().reset(checkpoint_id)
        self._update_cache_key()

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory is the same.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cache_key is None:
            return super().read_metadata()
        if self._cache_key not in CachedMetadataFileSystemReader._metadata_cache:
            CachedMetadataFileSystemReader._metadata_cache[self._cache_key] = (
                self.fs,
                super().read_metadata(),
            )
        return CachedMetadataFileSystemReader._metadata_cache[self._cache_key][1]

    @classmethod
    def clear_metadata_cache(
        cls,
        path: str | os.PathLike | None = None,
        *,
        filesystem: CheckpointFileSystem | None = None,
    ):
        """
        Clear the metadata cache.

        Args:
            path (Union[str, os.PathLike, None]): If provided, only the cache entry for
                this specific checkpoint path is removed. If None (default), the entire
                cache is cleared.
        """
        if path is None and filesystem is None:
            cls._metadata_cache.clear()
        else:
            fs = filesystem if filesystem is not None else _LOCAL_FILESYSTEM
            cache_path = os.fspath(fs.init_path(path)) if path is not None else None
            if cache_path is not None and fs is _LOCAL_FILESYSTEM:
                cache_path = os.path.abspath(cache_path)
            for key in list(cls._metadata_cache):
                if key[0] == id(fs) and (cache_path is None or key[1] == cache_path):
                    del cls._metadata_cache[key]
