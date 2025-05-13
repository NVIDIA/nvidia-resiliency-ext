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
from typing import Union

from torch.distributed.checkpoint import FileSystemReader, Metadata


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Attributes:
        _cached_metadata (Metadata or None): Cached metadata from the file system.
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cached_metadata = None

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cached_metadata is None:
            self._cached_metadata = super().read_metadata()
        return self._cached_metadata
