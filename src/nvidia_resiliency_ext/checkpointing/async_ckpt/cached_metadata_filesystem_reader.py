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
from typing import Dict, Union

from torch.distributed.checkpoint import FileSystemReader, Metadata


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Metadata is shared across all reader instances that use the same checkpoint
    directory (same path), since the loaded metadata is identical.

    Attributes:
        _metadata_cache (Dict[str, Metadata]): Class-level cache keyed by checkpoint path.
    """

    _metadata_cache: Dict[str, Metadata] = {}

    def __init__(self, path: Union[str, os.PathLike], cache_metadata: bool = True) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cache_key = os.path.abspath(os.fspath(path)) if cache_metadata else None

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory is the same.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cache_key not in CachedMetadataFileSystemReader._metadata_cache:
            CachedMetadataFileSystemReader._metadata_cache[self._cache_key] = (
                super().read_metadata()
            )
        return CachedMetadataFileSystemReader._metadata_cache[self._cache_key]

    @classmethod
    def clear_metadata_cache(cls):
        """
        Clear the metadata cache.
        """
        cls._metadata_cache.clear()
