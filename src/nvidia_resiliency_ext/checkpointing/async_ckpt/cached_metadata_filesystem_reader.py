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

import dataclasses
import functools
import gc
import os
import warnings
from typing import Any, Union

from torch.distributed.checkpoint import FileSystemReader, Metadata


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Metadata is cached for the life of the process and shared by all readers that read the same
    checkpoint directory with the same arguments. The cache never checks the files again; call
    clear_metadata_cache once the cached checkpoints are no longer needed or have changed.
    """

    def __init__(self, path: Union[str, os.PathLike], cache_metadata: bool = True) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
            cache_metadata (bool): If True (default), metadata is cached and shared across all
                instances that read the same checkpoint directory.
                If False, metadata is always read from disk on each call.
        """
        super().__init__(path=path)
        self._cache_metadata = cache_metadata

    @staticmethod
    @functools.cache
    def _read_cached_metadata(path: str, *args: Any, **kwargs: Any) -> Metadata:
        # Unpickling allocates millions of objects, which triggers thousands of collections over
        # a growing heap: about half the read time. Restore the caller's collector state after.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            return FileSystemReader(path).read_metadata(*args, **kwargs)
        finally:
            if gc_was_enabled:
                gc.enable()

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory and arguments are the same.

        Returns:
            Metadata: Checkpoint metadata. A shallow copy of the cached metadata, so that its
                storage_meta carries this reader's load_id.
        """
        if not self._cache_metadata:
            return super().read_metadata(*args, **kwargs)
        metadata = self._read_cached_metadata(os.path.abspath(self.path), *args, **kwargs)
        storage_meta = dataclasses.replace(metadata.storage_meta, load_id=self.load_id)
        return dataclasses.replace(metadata, storage_meta=storage_meta)

    @classmethod
    def clear_metadata_cache(cls, path: Union[str, os.PathLike, None] = None) -> None:
        """
        Clear the metadata cache.

        Args:
            path (Union[str, os.PathLike, None]): Deprecated. The whole cache is cleared whether
                or not a path is given.
        """
        if path is not None:
            warnings.warn(
                "The path argument of clear_metadata_cache is deprecated; the whole cache is "
                "cleared.",
                DeprecationWarning,
                stacklevel=2,
            )
        cls._read_cached_metadata.cache_clear()
