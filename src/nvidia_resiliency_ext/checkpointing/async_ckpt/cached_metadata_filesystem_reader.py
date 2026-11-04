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

"""Share deserialized checkpoint metadata between filesystem readers."""

import copy
import gc
import os
from threading import Lock
from typing import Any, Union

from torch.distributed.checkpoint import FileSystemReader, Metadata

_FileIdentity = tuple[str, int, int, int, int, int]


class CachedMetadataFileSystemReader(FileSystemReader):
    """Cache the most recently read metadata file in each process.

    Successive readers of an unchanged checkpoint reuse one deserialization. File identity
    checks invalidate the entry after replacement between completed loads. A checkpoint must
    remain unchanged throughout a load; these checks do not provide snapshot isolation.

    Each read returns independent top-level and storage metadata, including the reader's load
    ID. Large metadata dictionaries remain shared and callers must treat them as read-only.
    Only one file is retained, and callers can release it with ``clear_metadata_cache`` when
    loading finishes. Metadata retained by a save strategy is unaffected by clearing the cache.

    This reader uses ordinary filesystem paths, including mounted network filesystems. Other
    storage backends must retain their own reader and identity semantics.
    """

    _cached_metadata: tuple[_FileIdentity, Metadata] | None = None
    _cache_lock = Lock()

    def __init__(self, path: Union[str, os.PathLike], cache_metadata: bool = True) -> None:
        """Initialize the reader; disabling caching forces a fresh read on every call."""
        super().__init__(path=path)
        self._cache_metadata = cache_metadata

    def _file_identity(self, rank: int | None) -> _FileIdentity | None:
        if hasattr(self, "_get_metadata_path"):
            path = self._get_metadata_path(rank)
        else:
            path = self.fs.concat_path(self.path, ".metadata")
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return (
            os.path.abspath(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        # Avoid repeatedly scanning the growing object graph during deserialization.
        gc_enabled = gc.isenabled()
        try:
            gc.disable()
            return super().read_metadata(*args, **kwargs)
        finally:
            if gc_enabled:
                gc.enable()

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        """Reuse unchanged metadata and restore the GC setting even when reading fails."""
        cls = CachedMetadataFileSystemReader
        with cls._cache_lock:
            if not self._cache_metadata:
                return self._read_metadata(*args, **kwargs)

            identity = self._file_identity(kwargs.get("rank"))
            entry = cls._cached_metadata
            if identity is not None and entry is not None and entry[0] == identity:
                metadata = entry[1]
            else:
                # Release the old checkpoint before allocating another large object graph.
                cls._cached_metadata = None
                del entry
                metadata = self._read_metadata(*args, **kwargs)
                if identity is not None and identity == self._file_identity(kwargs.get("rank")):
                    cls._cached_metadata = (identity, metadata)

            result = copy.copy(metadata)
            if getattr(metadata, "storage_meta", None) is not None:
                result.storage_meta = copy.copy(metadata.storage_meta)
                result.storage_meta.load_id = self.load_id
            return result

    @staticmethod
    def clear_metadata_cache(path: Union[str, os.PathLike, None] = None) -> None:
        """Release cached metadata, optionally only for the given checkpoint directory."""
        cls = CachedMetadataFileSystemReader
        with cls._cache_lock:
            entry = cls._cached_metadata
            if path is None or (
                entry is not None and os.path.dirname(entry[0][0]) == os.path.abspath(path)
            ):
                cls._cached_metadata = None
