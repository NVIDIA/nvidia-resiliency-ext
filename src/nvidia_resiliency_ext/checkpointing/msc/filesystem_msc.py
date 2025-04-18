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

import io
import logging
import os
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import IO, Dict, List, Union, cast

try:
    import multistorageclient as msc
except ModuleNotFoundError:
    msc = None

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import StorageReader
from torch.distributed.checkpoint.filesystem import Metadata, MetadataIndex, _StorageInfo
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner, ReadItem
from torch.distributed.checkpoint.storage import StorageMeta
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future

logger = logging.getLogger(__name__)

MAX_WORKERS = 64


def prefetch_file(checkpoint_dir: str, relative_path: str) -> None:
    """
    Prefetch the checkpoint files

    Args:
        checkpoint_dir (str): The checkpoint directory.
        relative_path (str): The relative path to the checkpoint file.
    """
    path = os.path.join(checkpoint_dir, relative_path)
    s = time()
    with msc.open(path, "rb") as _:
        pass
    e = time()
    logger.debug(f"dowload {path}, time: {e-s:.2f} seconds")


class MultiStorageFileSystemReader(StorageReader):
    """
    A file system based storage reader for loading data using MSC. This class is similar to
    the FileSystemReader, but it uses MSC to read the data.
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize the MultiStorageFileSystemReader.
        """
        super().__init__()
        self.path = path
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = {}
        self.load_id = str(uuid.uuid4())
        self._cached_metadata = None

    def _slice_file(self, file, sinfo: _StorageInfo) -> io.IOBase:
        return _create_file_view(file, sinfo.offset, sinfo.length)

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """
        Reset the MultiStorageFileSystemReader.
        """
        self.storage_data = {}
        self.load_id = str(uuid.uuid4())

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Read the data from the MultiStorageFileSystemReader.
        """
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        # prefetch objects
        if len(per_file) > 0:
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(per_file))) as executor:
                futures = [
                    executor.submit(prefetch_file, str(self.path), rel_path)
                    for rel_path, _ in per_file.items()
                ]
                for future in futures:
                    future.result()

        for relative_path, reqs in per_file.items():
            new_path = os.path.join(self.path, relative_path)
            with msc.open(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(file_slice.read(item_md.length))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        tensor = cast(
                            torch.Tensor,
                            torch.load(
                                cast(IO[bytes], file_slice), map_location="cpu", weights_only=True
                            ),
                        )
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req).detach()

                        assert target_tensor.size() == tensor.size(), f"req {req.storage_index} "
                        f"mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementing the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        """
        Read the metadata from the MultiStorageFileSystemReader.
        """
        if self._cached_metadata is None:
            path = os.path.join(self.path, ".metadata")
            with msc.open(path, "rb") as metadata_file:
                metadata = pickle.load(metadata_file)

            if getattr(metadata, "storage_meta", None) is None:
                metadata.storage_meta = StorageMeta()
            metadata.storage_meta.load_id = self.load_id

            self._cached_metadata = metadata
        return self._cached_metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """
        Set up the storage reader.
        """
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Prepare the local plan.
        """
        return plan

    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        """
        Prepare the global plan.
        """
        return plans

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to load the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Validate the checkpoint_id.
        """
        return True
