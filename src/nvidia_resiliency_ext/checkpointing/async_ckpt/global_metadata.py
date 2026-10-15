# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The global metadata of a save, as the metadata cache keeps it and the coordinator writes it."""

import copy
import dataclasses
from typing import IO, Optional, Union

import numpy as np
from torch.distributed.checkpoint.metadata import Metadata

from nvidia_resiliency_ext.shared_utils import semconv, telemetry

from . import _metadata_pickler, _metadata_reuse
from ._metadata_pickler import StateDictTable, state_table


class GlobalMetadata:
    """A save's global metadata, without its storage_data, which is written from each save's write
    results.

    With the fast ``.metadata`` writers (see ``_metadata_pickler.writer``), it is a state table
    (see ``_metadata_pickler.state_table``): a few arrays, which the garbage collector does not
    visit, instead of a Python object per chunk. Otherwise it is a ``Metadata``, as before.

    It is never modified: ``with_fields`` returns a new one.
    """

    def __init__(self, content: Union[StateDictTable, Metadata]):
        """Use from_metadata. content: a state table, or a Metadata without storage_data."""
        self._content = content

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> "GlobalMetadata":
        """The global metadata of metadata, without its storage_data, and with only Metadata's
        fields (not e.g. the local plans older versions attach).

        As a table, it holds no reference to metadata's objects. Otherwise it shares its
        state_dict_metadata, which must then not be modified.
        """
        metadata = dataclasses.replace(metadata, storage_data=None)
        with telemetry.span(
            semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.metadata_to_table"
        ):
            table = _metadata_pickler.to_table(metadata)
        return cls(table if table is not None else metadata)

    @property
    def table(self) -> Optional[StateDictTable]:
        """The state table, or None if this is held as a Metadata."""
        return self._content if isinstance(self._content, StateDictTable) else None

    def to_metadata(self, storage_data: Optional[dict] = None) -> Metadata:
        """A new Metadata, with this storage_data."""
        if isinstance(self._content, StateDictTable):
            return _metadata_pickler.to_metadata(self._content, storage_data)
        metadata = copy.copy(self._content)
        metadata.storage_data = storage_data
        return metadata

    def with_fields(self, **values) -> "GlobalMetadata":
        """This global metadata with these Metadata fields (e.g. version) set to these values."""
        if isinstance(self._content, StateDictTable):
            return GlobalMetadata(state_table.with_fields(self._content, **values))
        metadata = copy.copy(self._content)
        for name, value in values.items():
            setattr(metadata, name, value)
        return GlobalMetadata(metadata)

    def entries(self) -> _metadata_reuse.Entries:
        """Its entries, as the reuse check reads them."""
        if isinstance(self._content, StateDictTable):
            return _metadata_reuse.TableEntries(self._content)
        return _metadata_reuse.ObjectEntries(self._content)

    def dump(
        self,
        stream: IO[bytes],
        storage_rows: Optional[np.ndarray] = None,
        storage_data: Optional[dict] = None,
    ) -> None:
        """Write it as ``.metadata``, with storage_data from storage_rows (the gathered
        write-result tables, see ``_metadata_pickler.table``) if given, else this storage_data."""
        if isinstance(self._content, StateDictTable):
            _metadata_pickler.dump_metadata(self._content, stream, storage_rows, storage_data)
        else:
            _metadata_pickler.dump_metadata(self.to_metadata(storage_data), stream, storage_rows)
