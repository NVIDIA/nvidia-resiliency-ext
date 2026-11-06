/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The C++ versions of `StateDictTable.from_metadata` and of `pickler._MetadataPickler`. The first
// converts a torch DCP `Metadata` into a `StateDictTable`. The second writes the standard pickle
// opcodes of the Metadata from that table. Both give exactly what the Python versions give. The
// few small objects (TensorProperties, StorageMeta, ...) are left to the Python `small_pickle`
// callback.

#include <pybind11/pybind11.h>

#include "pickler.h"
#include "state_dict_table.h"

using namespace nvrx::checkpoint;

PYBIND11_MODULE(_native, m) {
    m.doc() = "Fast conversion and pickling of torch.distributed.checkpoint Metadata.";
    m.def("from_metadata",
          &from_metadata,
          py::arg("metadata"),
          py::arg("small_pickle"),
          py::arg("table_cls"),
          "Convert metadata, without its storage_data, to a table_cls (a StateDictTable).");
    m.def("dumps_table",
          &dumps_table,
          py::arg("table"),
          py::arg("small_pickle"),
          py::arg("storage_data") = py::none(),
          py::arg("storage_rows") = py::none(),
          "Pickle the Metadata a state dict table was converted from, and return the bytes.");
}
