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

"""The part of a save's global metadata that every save shares, as the metadata cache keeps it and
the coordinator writes it."""

import dataclasses
import functools
from typing import IO, Optional, Union

import numpy as np
from torch.distributed.checkpoint.metadata import Metadata

from nvidia_resiliency_ext.shared_utils import semconv, telemetry

from . import _metadata_pickler, _metadata_reuse
from ._metadata_pickler import StateDictTable
from ._metadata_pickler.write_result_table import FqnList


class MetadataTemplate:
    """A save's global metadata without the parts each save sets: storage_data, written from the
    save's write results, and version and storage_meta, set when writing (see with_fields).

    With the fast ``.metadata`` writers (see ``_metadata_pickler.writer``), it is a state dict table
    (see ``_metadata_pickler.state_dict_table``): a few arrays, which the garbage collector does not
    visit, instead of a Python object per chunk. Otherwise it is a ``Metadata``, as before.

    It is never modified: ``with_fields`` returns a new one.
    """

    def __init__(self, content: Union[StateDictTable, Metadata]):
        """Hold content: a state dict table, or a Metadata without storage_data.

        Use from_metadata to make one."""
        self._content = content

    @classmethod
    def from_metadata(cls, metadata: Metadata) -> "MetadataTemplate":
        """Make the template of metadata: without its storage_data, and with only Metadata's fields,
        so that attributes a planner or an older nvrx set (e.g. all_local_plans) aren't written.

        As a table, it shares only the fqn strings of metadata's objects. Otherwise it shares its
        state_dict_metadata, which must then not be modified.
        """
        metadata = dataclasses.replace(metadata, storage_data=None)
        with telemetry.span(
            semconv.SPAN_GROUP_CKPT_PROFILING, "nv.nvrx.ckpt.save.metadata_to_table"
        ):
            table = _metadata_pickler.state_dict_table_if_fast(metadata)
        return cls(table if table is not None else metadata)

    @functools.cached_property
    def fqn_list(self) -> Optional[FqnList]:
        """The fqns of state_dict_metadata, in order: the fqn list index tables refer to; None if
        an fqn isn't exactly a str."""
        if isinstance(self._content, StateDictTable):
            return self._content.fqn_list
        return FqnList.of(self._content.state_dict_metadata)

    def to_metadata(self, storage_data: Optional[dict] = None) -> Metadata:
        """Build a new Metadata from the template, with this storage_data."""
        if isinstance(self._content, StateDictTable):
            return self._content.to_metadata(storage_data)
        return dataclasses.replace(self._content, storage_data=storage_data)

    def with_fields(self, **values) -> "MetadataTemplate":
        """Return a copy of the template with these Metadata fields (e.g. version) set.

        Raise TypeError for state_dict_metadata and storage_data, and for a name that isn't a
        field of Metadata.
        """
        if isinstance(self._content, StateDictTable):
            return MetadataTemplate(self._content.with_fields(**values))
        written = {"state_dict_metadata", "storage_data"} & set(values)
        if written:
            raise TypeError(f"not Metadata fields with_fields sets: {sorted(written)}")
        return MetadataTemplate(dataclasses.replace(self._content, **values))

    def entries(self) -> _metadata_reuse.Entries:
        """Return its entries, as the reuse check reads them."""
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
        write-result tables, see ``_metadata_pickler.write_result_table``) if given, else this
        storage_data.
        """
        if isinstance(self._content, StateDictTable):
            _metadata_pickler.dump_metadata(self._content, stream, storage_rows, storage_data)
        else:
            _metadata_pickler.dump_metadata(self.to_metadata(storage_data), stream, storage_rows)
