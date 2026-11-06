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

#include "state_dict_table.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cpython.h"
#include "string_map.h"

using namespace nvrx::checkpoint;
using namespace nvrx::checkpoint::detail;

namespace {

// New reference to obj.name, throwing on error.
[[nodiscard]] auto getattr(PyObject* obj, PyObject* name) -> py::object {
    PyObject* const r = PyObject_GetAttr(obj, name);
    if (r == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(r);
}

// Return the name of obj's type, for error messages.
[[nodiscard]] auto type_name(PyObject* obj) -> std::string { return Py_TYPE(obj)->tp_name; }

// Reject a malformed state dict table, before anything reads outside its arrays.
[[noreturn]] void bad_state_dict_table(std::string_view what) {
    throw py::value_error("state dict table: " + std::string(what));
}

// Return the attribute name of table, which must be exactly a tuple, as a StateDictTable's
// sequences are.
[[nodiscard]] auto tuple_attr(py::handle table, const char* name) -> py::tuple {
    py::object value = table.attr(name);
    if (!PyTuple_CheckExact(value.ptr())) {
        bad_state_dict_table(std::string(name) + " is not a tuple");
    }
    return py::reinterpret_steal<py::tuple>(value.release());
}

// Return the attribute name of table, which must be a 1-D, C-contiguous, aligned array of exactly
// the element type T.
template <TableValue T>
[[nodiscard]] auto array_attr(py::handle table, const char* name) -> Array<T> {
    py::object value = table.attr(name);
    if (!Array<T>::check_(value) || py::reinterpret_borrow<py::array>(value).ndim() != 1 ||
        (py::reinterpret_borrow<py::array>(value).flags() &
         py::detail::npy_api::NPY_ARRAY_ALIGNED_) == 0) {
        bad_state_dict_table(std::string(name) + " is not a 1-D array of the expected type");
    }
    return py::reinterpret_steal<Array<T>>(value.release());
}

// Return the values of a 1-D array, valid while the array is held.
template <TableValue T>
[[nodiscard]] auto values(const Array<T>& a) -> std::span<const T> {
    return {a.data(), static_cast<size_t>(a.size())};
}

// Return whether values = ndim * (1 + 2 * chunk_count): a tensor's size, and the offsets and
// sizes of chunk_count chunks.
// All three are non-negative; divides first, so nothing can overflow.
[[nodiscard]] auto holds_chunks(int64_t values, int64_t ndim, int64_t chunk_count) -> bool {
    if (ndim == 0) {
        return values == 0;
    }
    const int64_t per_dim = values / ndim;  // 1 + 2 * chunk_count
    return values % ndim == 0 && per_dim % 2 == 1 && per_dim / 2 == chunk_count;
}

// Return a numpy array that owns the vector's values, without copying them.
template <TableValue T>
[[nodiscard]] auto to_array(std::vector<T>&& vector) -> py::array_t<T> {
    auto owned = std::make_unique<std::vector<T>>(std::move(vector));
    const py::capsule owner(owned.get(),
                            [](void* p) { delete static_cast<std::vector<T>*>(p); });
    const std::vector<T>& values = *owned.release();
    return py::array_t<T>({values.size()}, {sizeof(T)}, values.data(), owner);
}

// Converts a Metadata into a StateDictTable, exactly as StateDictTable.from_metadata does.
class StateDictTableBuilder {
   public:
    // Look up the metadata classes the conversion checks objects against.
    explicit StateDictTableBuilder(py::object small_pickle)
        : small_pickle_(std::move(small_pickle)) {
        py::module_ const meta = py::module_::import("torch.distributed.checkpoint.metadata");
        metadata_cls_ = meta.attr("Metadata");
        tensor_cls_ = meta.attr("TensorStorageMetadata");
        bytes_cls_ = meta.attr("BytesStorageMetadata");
        chunk_cls_ = meta.attr("ChunkStorageMetadata");
        size_cls_ = py::module_::import("torch").attr("Size");
    }

    // Convert metadata, without its storage_data, to an instance of table_cls (a StateDictTable).
    // Use once, on a temporary builder: it moves its arrays and lists into the table.
    [[nodiscard]] auto build(py::handle metadata, const py::object& table_cls) && -> py::object {
        if (!is(metadata.ptr(), metadata_cls_)) {
            throw py::type_error("expected Metadata, got " + type_name(metadata.ptr()));
        }
        const py::object state = instance_dict(metadata.ptr());
        const py::object entries = getattr(metadata.ptr(), a_state_dict_metadata_.ptr());
        if (!PyDict_CheckExact(entries.ptr())) {
            throw py::type_error("expected dict for state_dict_metadata, got " +
                                 type_name(entries.ptr()));
        }
        // Snapshots, which hold the items while small_pickle runs Python code that could change
        // the dicts.
        py::list fields;
        for (const py::handle item : items(state.ptr())) {
            PyObject* const name = PyTuple_GET_ITEM(item.ptr(), 0);
            const std::string_view field_name = utf8(name);
            const bool written_here =
                field_name == STATE_DICT_METADATA || field_name == STORAGE_DATA;
            const py::object opcodes = written_here
                                           ? py::object(py::none())
                                           : py::object(pickled(PyTuple_GET_ITEM(item.ptr(), 1)));
            fields.append(py::make_tuple(py::handle(name), opcodes));
        }
        const py::list entry_items = items(entries.ptr());
        const size_t n = entry_items.size();
        py::list const fqns(n);
        kind_.reserve(n);
        ref_.reserve(n);
        ndim_.reserve(n);
        for (auto* starts : {&chunk_start_, &dims_start_}) {
            starts->reserve(n + 1);
            starts->push_back(0);
        }
        for (size_t i = 0; i < n; ++i) {
            PyObject* const fqn = PyTuple_GET_ITEM(entry_items[i].ptr(), 0);
            PyObject* const entry = PyTuple_GET_ITEM(entry_items[i].ptr(), 1);
            if (!PyUnicode_CheckExact(fqn)) {
                throw py::type_error("expected str fqn, got " + type_name(fqn));
            }
            fqns[i] = py::handle(fqn);
            size_t ndim = 0;
            size_t chunks = 0;
            if (const std::optional<TensorEntry> tensor =
                    is(entry, tensor_cls_) ? tensor_entry(entry) : std::nullopt) {
                // After the chunks are counted: small_pickle runs Python code.
                py::bytes opcodes = pickled(tensor->properties.ptr());
                const std::string_view key = bytes_view(opcodes.ptr());
                auto it = property_ids_.find(key);
                if (it == property_ids_.end()) {
                    it = property_ids_.emplace(key, properties_.size()).first;
                    properties_.append(opcodes);
                }
                push(Kind::Tensor, static_cast<int64_t>(it->second));
                ndim = tensor->ndim;
                chunks = tensor->chunks;
            } else if (is(entry, bytes_cls_) && PyDict_Size(instance_dict(entry).ptr()) == 0) {
                push(Kind::Bytes, 0);
            } else {
                push(Kind::Other, static_cast<int64_t>(others_.size()));
                others_.append(pickled(entry));
            }
            ndim_.push_back(static_cast<int64_t>(ndim));
            chunk_start_.push_back(chunk_start_.back() + static_cast<int64_t>(chunks));
            dims_start_.push_back(static_cast<int64_t>(dims_.size()));
        }
        if (std::cmp_not_equal(PyDict_Size(entries.ptr()), n)) {
            throw std::runtime_error("state_dict_metadata changed size while being converted");
        }
        return table_cls(py::arg("fields") = fields, py::arg("fqns") = fqns,
                         py::arg("kind") = to_array(std::move(kind_)),
                         py::arg("ref") = to_array(std::move(ref_)),
                         py::arg("ndim") = to_array(std::move(ndim_)),
                         py::arg("chunk_start") = to_array(std::move(chunk_start_)),
                         py::arg("dims_start") = to_array(std::move(dims_start_)),
                         py::arg("dims") = to_array(std::move(dims_)),
                         py::arg("properties") = std::move(properties_),
                         py::arg("others") = std::move(others_));
    }

   private:
    // A TENSOR entry's properties and number of chunks.
    struct TensorEntry {
        py::object properties;
        size_t ndim;
        size_t chunks;
    };

    // Append a TENSOR entry's size and chunks to dims_ and return its properties and numbers of
    // dims and chunks; nullopt, with nothing appended, if it isn't one (see _tensor_entry in
    // state_dict_table.py).
    [[nodiscard]] auto tensor_entry(PyObject* entry) -> std::optional<TensorEntry> {
        const py::object state = instance_dict(entry);
        PyObject* const properties = dict_get(state.ptr(), a_properties_.ptr());
        PyObject* const size = dict_get(state.ptr(), a_size_.ptr());
        PyObject* const chunks = dict_get(state.ptr(), a_chunks_.ptr());
        if (PyDict_Size(state.ptr()) != 3 || (properties == nullptr) || (size == nullptr) || !is(size, size_cls_) ||
            (chunks == nullptr) || !PyList_CheckExact(chunks)) {
            return std::nullopt;
        }
        const Py_ssize_t ndim = PyTuple_GET_SIZE(size);
        const Py_ssize_t chunk_count = PyList_GET_SIZE(chunks);
        if (ndim == 0 && chunk_count > 1) {
            return std::nullopt;  // a scalar saved by several ranks without dedup
        }
        const size_t mark = dims_.size();
        if (!append_dims(size, dims_)) {
            dims_.resize(mark);
            return std::nullopt;
        }
        for (Py_ssize_t k = 0; k < chunk_count; ++k) {
            PyObject* const chunk = PyList_GET_ITEM(chunks, k);
            const py::object chunk_state =
                is(chunk, chunk_cls_) ? instance_dict(chunk) : py::object();
            PyObject* const offsets =
                chunk_state ? dict_get(chunk_state.ptr(), a_offsets_.ptr()) : nullptr;
            PyObject* const sizes =
                chunk_state ? dict_get(chunk_state.ptr(), a_sizes_.ptr()) : nullptr;
            if (!chunk_state || PyDict_Size(chunk_state.ptr()) != 2 || (offsets == nullptr) || (sizes == nullptr) ||
                !is(offsets, size_cls_) || !is(sizes, size_cls_) ||
                PyTuple_GET_SIZE(offsets) != ndim || PyTuple_GET_SIZE(sizes) != ndim ||
                !append_dims(offsets, dims_) || !append_dims(sizes, dims_)) {
                dims_.resize(mark);
                return std::nullopt;
            }
        }
        return TensorEntry{.properties=py::reinterpret_borrow<py::object>(properties),
                           .ndim=static_cast<size_t>(ndim),
                      .chunks=static_cast<size_t>(chunk_count)};
    }

    // Append a torch.Size's dims; return false unless each is an int within int64 (a torch.Size
    // may also hold numpy ints or larger ints, which only pickle writes exactly).
    [[nodiscard]] static auto append_dims(PyObject* dims, std::vector<int64_t>& out) -> bool {
        const Py_ssize_t n = PyTuple_GET_SIZE(dims);
        for (Py_ssize_t k = 0; k < n; ++k) {
            PyObject* const dim = PyTuple_GET_ITEM(dims, k);
            if (!PyLong_CheckExact(dim)) {
                return false;
            }
            const long long value = PyLong_AsLongLong(dim);
            if (value == -1 && (PyErr_Occurred() != nullptr)) {
                PyErr_Clear();  // beyond int64
                return false;
            }
            out.push_back(value);
        }
        return true;
    }

    void push(Kind kind, int64_t ref) {
        kind_.push_back(static_cast<uint8_t>(kind));
        ref_.push_back(ref);
    }

    // Return the stdlib pickler's opcodes for obj.
    [[nodiscard]] auto pickled(PyObject* obj) const -> py::bytes {
        py::object b = small_pickle_(py::handle(obj));
        if (!PyBytes_CheckExact(b.ptr())) {
            throw py::type_error("small_pickle returned no bytes");
        }
        return py::reinterpret_steal<py::bytes>(b.release());
    }

    // Return a snapshot of a dict's items, as a list of (key, value) tuples.
    [[nodiscard]] static auto items(PyObject* dict) -> py::list {
        PyObject* const r = PyDict_Items(dict);
        if (r == nullptr) {
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::list>(r);
    }

    [[nodiscard]] static auto is(PyObject* obj, const py::object& cls) -> bool {
        return reinterpret_cast<PyObject*>(Py_TYPE(obj)) == cls.ptr();
    }

    py::object small_pickle_;
    py::object metadata_cls_, tensor_cls_, bytes_cls_, chunk_cls_, size_cls_;
    py::str a_state_dict_metadata_{"state_dict_metadata"}, a_properties_{"properties"};
    py::str a_size_{"size"}, a_chunks_{"chunks"}, a_offsets_{"offsets"}, a_sizes_{"sizes"};
    std::vector<uint8_t> kind_;
    std::vector<int64_t> ref_, ndim_, chunk_start_, dims_start_, dims_;
    py::list properties_, others_;
    StringMap<size_t> property_ids_;  // by their opcodes
};

}  // namespace

StateDictTableView::StateDictTableView(py::handle table)
    : table_(py::reinterpret_borrow<py::object>(table)),
      fqns_(tuple_attr(table, "fqns")),
      properties_(tuple_attr(table, "properties")),
      others_(tuple_attr(table, "others")),
      kind_array_(array_attr<uint8_t>(table, "kind")),
      ref_array_(array_attr<int64_t>(table, "ref")),
      ndim_array_(array_attr<int64_t>(table, "ndim")),
      chunk_start_array_(array_attr<int64_t>(table, "chunk_start")),
      dims_start_array_(array_attr<int64_t>(table, "dims_start")),
      dims_array_(array_attr<int64_t>(table, "dims")),
      kind_(values(kind_array_)),
      ref_(values(ref_array_)),
      ndim_(values(ndim_array_)),
      chunk_start_(values(chunk_start_array_)),
      dims_start_(values(dims_start_array_)),
      dims_(values(dims_array_)) {
    const py::tuple fields = tuple_attr(table, "fields");
    for (const py::handle field : fields) {
        if (!PyTuple_CheckExact(field.ptr()) || PyTuple_GET_SIZE(field.ptr()) != 2) {
            bad_state_dict_table("a field is not a (name, opcodes) tuple");
        }
        PyObject* const name = PyTuple_GET_ITEM(field.ptr(), 0);
        PyObject* const opcodes = PyTuple_GET_ITEM(field.ptr(), 1);
        if (!PyUnicode_CheckExact(name)) {
            bad_state_dict_table("a field name is a " + type_name(name) + ", not a str");
        }
        const std::string_view field_name = utf8(name);
        const bool written_here =
            field_name == STATE_DICT_METADATA || field_name == STORAGE_DATA;
        if (written_here ? opcodes != Py_None : !PyBytes_CheckExact(opcodes)) {
            bad_state_dict_table("unexpected opcodes of field " + std::string(field_name));
        }
        fields_.push_back(
            {std::string(field_name), py::reinterpret_borrow<py::object>(opcodes)});
    }
    check();
}

auto StateDictTableView::fqns_fingerprint() const -> int64_t {
    if (!fqns_fingerprint_) {
        fqns_fingerprint_ = table_.attr("fqn_list").attr("fingerprint").cast<int64_t>();
    }
    return *fqns_fingerprint_;
}

void StateDictTableView::check() const {
    const size_t n = entries();
    if (static_cast<size_t>(PyTuple_GET_SIZE(fqns_.ptr())) != n || ref_.size() != n ||
        ndim_.size() != n) {
        bad_state_dict_table("arrays of different lengths");
    }
    const auto check_starts = [n](std::span<const int64_t> starts,
                                  std::optional<size_t> pool) {
        if (starts.size() != n + 1 || starts[0] != 0 ||
            !std::ranges::is_sorted(starts) ||
            (pool && std::cmp_not_equal(starts[n], *pool))) {
            bad_state_dict_table("start offsets out of order or range");
        }
    };
    check_starts(chunk_start_, std::nullopt);
    check_starts(dims_start_, dims_.size());
    const auto in = [](int64_t ref, const py::tuple& pickles) {
        return ref >= 0 && ref < PyTuple_GET_SIZE(pickles.ptr()) &&
               PyBytes_CheckExact(PyTuple_GET_ITEM(pickles.ptr(), ref));
    };
    for (size_t i = 0; i < n; ++i) {
        if (!PyUnicode_CheckExact(fqn(i))) {
            bad_state_dict_table("an fqn is not a str");
        }
        const int64_t ndim = ndim_[i];
        const int64_t chunk_count = chunk_start_[i + 1] - chunk_start_[i];
        const int64_t values = dims_start_[i + 1] - dims_start_[i];
        if (ndim < 0) {
            bad_state_dict_table("a negative number of dims");
        }
        switch (kind_[i]) {
            case static_cast<uint8_t>(Kind::Tensor):
                if (!in(ref_[i], properties_) || !holds_chunks(values, ndim, chunk_count)) {
                    bad_state_dict_table("a tensor entry's dims don't match its chunks");
                }
                // Its chunks have no dims, so nothing else bounds their number.
                if (ndim == 0 && chunk_count > 1) {
                    bad_state_dict_table("a 0-dim tensor entry with more than one chunk");
                }
                break;
            case static_cast<uint8_t>(Kind::Other):
                if (!in(ref_[i], others_)) {
                    bad_state_dict_table("an entry refers to no pickle");
                }
                [[fallthrough]];
            case static_cast<uint8_t>(Kind::Bytes):
                if (ndim != 0 || chunk_count != 0 || values != 0) {
                    bad_state_dict_table("a non-tensor entry has dims");
                }
                break;
            default:
                bad_state_dict_table("unknown entry kind");
        }
    }
}

auto nvrx::checkpoint::from_metadata(py::handle metadata, py::object small_pickle,
                                           const py::object& table_cls) -> py::object {
    return StateDictTableBuilder(std::move(small_pickle)).build(metadata, table_cls);
}
