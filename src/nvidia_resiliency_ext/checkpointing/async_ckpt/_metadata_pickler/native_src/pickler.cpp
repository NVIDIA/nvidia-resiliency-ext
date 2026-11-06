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

#include "pickler.h"

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "cpython.h"
#include "state_dict_table.h"
#include "string_map.h"
#include "write_result_table.h"

using namespace nvrx::checkpoint;
using namespace nvrx::checkpoint::detail;

namespace {

constexpr char PROTO = '\x80', STOP = '.', MARK = '(', POP = '0';
constexpr char EMPTY_DICT = '}', EMPTY_LIST = ']', EMPTY_TUPLE = ')';
constexpr char SETITEMS = 'u', APPENDS = 'e', BUILD = 'b', NEWOBJ = '\x81', REDUCE = 'R';
constexpr char STACK_GLOBAL = '\x93', MEMOIZE = '\x94', NONE = 'N';
constexpr char TUPLE = 't', TUPLE1 = '\x85', TUPLE2 = '\x86', TUPLE3 = '\x87';
constexpr size_t BATCH = 1000;  // same batching as the stdlib pickler

// Borrowed reference to dict[key], throwing if key is missing.
[[nodiscard]] auto dict_item(PyObject* dict, PyObject* key) -> PyObject* {
    PyObject* const r = dict_get(dict, key);
    if (r == nullptr) {
        throw py::type_error("missing attribute");
    }
    return r;
}

// The value of an int that fits int64. Like the other checks here, it accepts only the exact type:
// anything else (a bool, a tuple for a torch.Size, ...) would unpickle as a different type, so it
// raises and the caller falls back to pickle.dump.
[[nodiscard]] auto as_int(PyObject* o) -> int64_t {
    if (!PyLong_CheckExact(o)) {
        throw py::type_error("expected int");
    }
    long long const v = PyLong_AsLongLong(o);
    if (v == -1 && (PyErr_Occurred() != nullptr)) {
        throw py::error_already_set();
    }
    return v;
}

// Writes the pickle of one Metadata into out_; use once per dump.
class Writer {
   public:
    // Look up the classes the storage_data encoding checks objects against.
    explicit Writer(py::object small_pickle) : small_pickle_(std::move(small_pickle)) {
        py::module_ const meta = py::module_::import("torch.distributed.checkpoint.metadata");
        index_cls_ = meta.attr("MetadataIndex");
        info_cls_ =
            py::module_::import("torch.distributed.checkpoint.filesystem").attr("_StorageInfo");
        size_cls_ = py::module_::import("torch").attr("Size");
    }

    // Pickle the Metadata that table_view's table was converted from, and return the bytes: a
    // NEWOBJ of Metadata built from its fields, in order. storage_data is written from rows (the
    // gathered write-result tables) if given, else from the storage_data dict, else as None.
    [[nodiscard]] auto dumps(StateDictTableView const& table_view,
                             py::handle storage_data,
                             py::object const& rows) -> py::bytes {
        // Holds the rows the tables point into.
        Rows const rows_array = rows.is_none() ? Rows() : rows.cast<Rows>();
        std::vector<WriteResultTable> const tables =
            rows.is_none() ? std::vector<WriteResultTable>() : parse_tables(rows_array);
        size_t n_storage = 0;
        if (!rows.is_none()) {
            for (WriteResultTable const& w : tables) {
                n_storage += w.entries();
            }
        } else if (PyDict_Check(storage_data.ptr())) {
            n_storage = PyDict_Size(storage_data.ptr());
        }
        // About 98 bytes per storage entry (its MetadataIndex and _StorageInfo) and about 8 bytes
        // per chunk dim in state_dict_metadata.
        out_.reserve(4096 + (100 * n_storage) + (8 * table_view.dims_size()));

        prelude();
        put_get(metadata_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        for (StateDictTableView::Field const& field : table_view.fields()) {
            put_str(field.name);
            if (field.name == STATE_DICT_METADATA) {
                dump_state_dict_metadata(table_view);
            } else if (field.name != STORAGE_DATA) {
                append(bytes_view(field.opcodes.ptr()));
            } else if (!rows.is_none()) {
                dump_storage_tables(tables, table_view);
            } else if (!storage_data.is_none()) {
                dump_storage_data(storage_data.ptr());
            } else {
                put(NONE);
            }
        }
        put({SETITEMS, BUILD, STOP});
        return py::bytes(out_.data(), out_.size());
    }

   private:
    // Protocol 4 header, then the classes and dict keys the encoding refers to, each memoized once.
    void prelude() {
        put({PROTO, '\x04'});
        constexpr char const* META = "torch.distributed.checkpoint.metadata";
        size_ref_ = global_ref("torch", "Size");
        metadata_ref_ = global_ref(META, "Metadata");
        index_ref_ = global_ref(META, "MetadataIndex");
        info_ref_ = global_ref("torch.distributed.checkpoint.filesystem", "_StorageInfo");
        chunk_ref_ = global_ref(META, "ChunkStorageMetadata");
        tensor_ref_ = global_ref(META, "TensorStorageMetadata");
        bytes_ref_ = global_ref(META, "BytesStorageMetadata");
        k_fqn_ = key_ref("fqn");
        k_index_ = key_ref("index");
        k_offset_ = key_ref("offset");
        k_relative_path_ = key_ref("relative_path");
        k_length_ = key_ref("length");
        k_offsets_ = key_ref("offsets");
        k_sizes_ = key_ref("sizes");
        k_properties_ = key_ref("properties");
        k_size_ = key_ref("size");
        k_chunks_ = key_ref("chunks");
    }

    // Write Metadata.state_dict_metadata from the table's entries, in order.
    void dump_state_dict_metadata(StateDictTableView const& table_view) {
        put(EMPTY_DICT);
        size_t const n = table_view.entries();
        fqn_memo_.assign(n, std::nullopt);
        for (size_t i = 0; i < n; ++i) {
            begin_item(i);
            fqn_memo_[i] = string(table_view.fqn(i));
            switch (table_view.kind(i)) {
                case Kind::Bytes:
                    put_get(bytes_ref_);
                    put({EMPTY_TUPLE, NEWOBJ});
                    break;
                case Kind::Other:
                    append(table_view.other(i));
                    break;
                case Kind::Tensor: {
                    put_get(tensor_ref_);
                    put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                    put_get(k_properties_);
                    append(table_view.properties(i));
                    put_get(k_size_);
                    size(table_view.size(i));
                    put_get(k_chunks_);
                    put(EMPTY_LIST);
                    size_t const chunk_count = table_view.chunks(i);
                    size_t const ndim = table_view.ndim(i);
                    for (size_t k = 0; k < chunk_count; ++k) {
                        begin_item(k);
                        std::span<int64_t const> const chunk = table_view.chunk(i, k);
                        put_get(chunk_ref_);
                        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                        put_get(k_offsets_);
                        size(chunk.first(ndim));
                        put_get(k_sizes_);
                        size(chunk.last(ndim));
                        put({SETITEMS, BUILD});
                        end_item(k, chunk_count, APPENDS);
                    }
                    put({SETITEMS, BUILD});
                    break;
                }
            }
            end_item(i, n, SETITEMS);
        }
    }

    // Metadata.storage_data: MetadataIndex -> _StorageInfo.
    void dump_storage_data(PyObject* dict) {
        if (!PyDict_CheckExact(dict)) {
            throw py::type_error("expected dict");
        }
        put(EMPTY_DICT);
        Py_ssize_t const n = PyDict_Size(dict);
        Py_ssize_t pos = 0;
        PyObject* idx = nullptr;
        PyObject* info = nullptr;
        for (Py_ssize_t i = 0; PyDict_Next(dict, &pos, &idx, &info) != 0; ++i) {
            begin_item(i);
            // Owns info while small(idx) runs Python code, which could remove it from the dict.
            py::object keep_info;
            if (!is(idx, index_cls_)) {
                keep_info = py::reinterpret_borrow<py::object>(info);
                small(idx);
                check_size(dict, n);
            } else {
                // MetadataIndex sets offset only when it is given; pickle its __dict__.
                py::object const state = instance_dict(idx);
                PyObject* const fqn = dict_item(state.ptr(), a_fqn_.ptr());
                PyObject* const index = dict_item(state.ptr(), a_index_.ptr());
                PyObject* const offset = dict_get(state.ptr(), a_offset_.ptr());
                if (PyDict_Size(state.ptr()) != ((offset != nullptr) ? 3 : 2)) {
                    throw py::type_error("unexpected MetadataIndex attributes");
                }
                std::string_view const fqn_view = utf8(fqn);
                put_index_start([&] { string(fqn_view); },
                                index == Py_None ? std::nullopt : std::optional(as_int(index)));
                if (offset != nullptr) {
                    put_get(k_offset_);
                    if (offset == Py_None) {
                        put(NONE);
                    } else {
                        size(offset);
                    }
                }
                put({SETITEMS, BUILD});
            }

            // _StorageInfo pickles its __dict__ without None values. transform_descriptors exists
            // from PyTorch 2.8; a _StorageInfo with it set is left to the stdlib pickler.
            py::object const state = is(info, info_cls_) ? instance_dict(info) : py::object();
            PyObject* const transforms =
                state ? dict_get(state.ptr(), a_transform_.ptr()) : nullptr;
            if (!state || ((transforms != nullptr) && transforms != Py_None)) {
                small(info);
                check_size(dict, n);
            } else {
                if (PyDict_Size(state.ptr()) != ((transforms != nullptr) ? 4 : 3)) {
                    throw py::type_error("unexpected _StorageInfo attributes");
                }
                std::string_view const path = utf8(dict_item(state.ptr(), a_relative_path_.ptr()));
                put_storage_info([&] { string(path); },
                                 as_int(dict_item(state.ptr(), a_offset_.ptr())),
                                 as_int(dict_item(state.ptr(), a_length_.ptr())));
            }
            end_item(i, n, SETITEMS);
        }
    }

    // Metadata.storage_data from write-result tables, as dump_storage_data writes the dict that
    // finish builds from the same write results.
    void dump_storage_tables(std::vector<WriteResultTable> const& tables,
                             StateDictTableView const& table_view) {
        put(EMPTY_DICT);
        size_t total = 0;
        for (WriteResultTable const& t : tables) {
            total += t.entries();
        }
        fqn_memo_.resize(table_view.entries());
        size_t i = 0;
        std::vector<std::string_view> strings;             // the current table's
        std::vector<std::optional<uint32_t>> string_memo;  // their memo indices, once written
        for (WriteResultTable const& t : tables) {
            if (t.fqn_mode() == FqnMode::Indices &&
                t.fingerprint() != table_view.fqns_fingerprint()) {
                bad_table("an index table encoded against another fqn list");
            }
            t.split_strings(strings);
            string_memo.assign(strings.size(), std::nullopt);
            size_t pos = 0;  // into the table's offsets
            for (size_t r = 0; r < t.entries(); ++r, ++i) {
                begin_item(i);
                dump_table_entry(t, t.entry(r), strings, string_memo, table_view, pos);
                end_item(i, total, SETITEMS);
            }
            t.check_offsets_used(pos);
        }
    }

    // Write one table entry: its MetadataIndex and _StorageInfo. pos is where its offset dims
    // start in the table's offsets; it moves past them.
    void dump_table_entry(WriteResultTable const& t,
                          Int64s const& entry,
                          std::vector<std::string_view> const& strings,
                          std::vector<std::optional<uint32_t>>& string_memo,
                          StateDictTableView const& table_view,
                          size_t& pos) {
        int64_t const flags = entry[Column::Flags];
        if ((flags & ~(FLAG_INDEX | FLAG_OFFSET | FLAG_OFFSET_SIZE)) != 0) {
            bad_table("unknown flags");
        }
        if (((flags & FLAG_OFFSET_SIZE) != 0) && ((flags & FLAG_OFFSET) == 0)) {
            bad_table("an offset size without an offset");
        }
        std::optional<int64_t> const index =
            ((flags & FLAG_INDEX) != 0) ? std::optional(entry[Column::Index]) : std::nullopt;
        int64_t const fqn = entry[Column::Fqn];
        put_index_start(
            [&] {
                if (t.fqn_mode() == FqnMode::Indices) {
                    put_fqn_entry(table_view, fqn);
                } else {
                    put_table_string(strings, string_memo, fqn);
                }
            },
            index);
        if ((flags & FLAG_OFFSET) != 0) {
            put_get(k_offset_);
            if ((flags & FLAG_OFFSET_SIZE) != 0) {
                int64_t const ndim = entry[Column::OffsetNdim];
                size(t.offset_dims(pos, ndim));
                pos += static_cast<size_t>(ndim);
            } else {
                put(NONE);
            }
        }
        put({SETITEMS, BUILD});
        put_storage_info([&] { put_table_string(strings, string_memo, entry[Column::Path]); },
                         entry[Column::Offset],
                         entry[Column::Length]);
    }

    // Write a table's string id: written or fetched on first use, then fetched by its memo index.
    void put_table_string(std::vector<std::string_view> const& strings,
                          std::vector<std::optional<uint32_t>>& string_memo,
                          int64_t id) {
        std::string_view const value = string_at(strings, id);
        std::optional<uint32_t>& memo = string_memo[static_cast<size_t>(id)];
        if (memo) {
            put_get(*memo);
        } else {
            memo = string(value);
        }
    }

    // Write the fqn of entry id of the state dict table, by the memo index the
    // state_dict_metadata pass gave it.
    void put_fqn_entry(StateDictTableView const& table_view, int64_t id) {
        if (id < 0 || static_cast<uint64_t>(id) >= table_view.entries()) {
            bad_table("fqn entry out of range");
        }
        std::optional<uint32_t>& memo = fqn_memo_[static_cast<size_t>(id)];
        if (memo) {
            put_get(*memo);
        } else {
            memo = string(table_view.fqn(static_cast<size_t>(id)));
        }
    }

    // Return a string id of a table's strings, checked.
    [[nodiscard]] static auto string_at(std::vector<std::string_view> const& strings, int64_t id)
        -> std::string_view {
        if (id < 0 || static_cast<uint64_t>(id) >= strings.size()) {
            bad_table("string id out of range");
        }
        return strings[static_cast<size_t>(id)];
    }

    // Write a MetadataIndex up to its offset, its fqn by write_fqn: the caller writes the offset,
    // if it has one, and closes the index with put({SETITEMS, BUILD}).
    template <typename WriteFqn>
    void put_index_start(WriteFqn&& write_fqn, std::optional<int64_t> index) {
        put_get(index_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        put_get(k_fqn_);
        write_fqn();
        put_get(k_index_);
        if (index) {
            put_int(*index);
        } else {
            put(NONE);
        }
    }

    // Write a _StorageInfo without transform descriptors, its relative path by write_path.
    template <typename WritePath>
    void put_storage_info(WritePath&& write_path, int64_t offset, int64_t length) {
        put_get(info_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        put_get(k_relative_path_);
        write_path();
        put_get(k_offset_);
        put_int(offset);
        put_get(k_length_);
        put_int(length);
        put({SETITEMS, BUILD});
    }

    // Open a batch of up to BATCH items with MARK before item i, as the stdlib pickler batches.
    void begin_item(size_t i) {
        if (i % BATCH == 0) {
            put(MARK);
        }
    }

    // Close the batch with op (SETITEMS or APPENDS) after item i, at its end or after item n - 1.
    void end_item(size_t i, size_t n, char op) {
        if (i % BATCH == BATCH - 1 || i == n - 1) {
            put(op);
        }
    }

    // Raise if Python code run by small() changed the size of a dict or list being written: the
    // loops would read past its end or skip entries. Stock pickle raises for dicts too.
    static void check_size(PyObject* container, Py_ssize_t n) {
        if (PyObject_Size(container) != n) {
            throw std::runtime_error("metadata changed size while being pickled");
        }
    }

    // Whether obj is exactly of class cls. Objects of exactly the metadata classes are encoded
    // here; others, subclasses included, are left to the stdlib pickler and keep their class.
    [[nodiscard]] static auto is(PyObject* obj, py::object const& cls) -> bool {
        return static_cast<void const*>(Py_TYPE(obj)) == static_cast<void const*>(cls.ptr());
    }

    // Append raw opcode bytes.
    void put(char c) { out_.push_back(c); }
    void put(std::initializer_list<char> cs) { out_.insert(out_.end(), cs.begin(), cs.end()); }
    // Append bytes as they are.
    void append(std::string_view bytes) { out_.insert(out_.end(), bytes.begin(), bytes.end()); }

    // A str, with the opcode the stdlib pickler picks at protocol 4: SHORT_BINUNICODE,
    // BINUNICODE from 256 bytes on, BINUNICODE8 past 4 GiB.
    void put_str(std::string_view s) {
        if (s.size() < 256) {
            put('\x8c');
            put(static_cast<char>(s.size()));
        } else if (s.size() <= UINT32_MAX) {
            put('X');
            put_le(s.size(), 4);
        } else {
            put('\x8d');
            put_le(s.size(), 8);
        }
        append(s);
    }

    // Fetch memo entry idx: BINGET, or LONG_BINGET from 256 on.
    void put_get(uint32_t idx) {
        if (idx < 256) {
            put('h');
            put(static_cast<char>(idx));
        } else {
            put('j');
            put_le(idx, 4);
        }
    }

    // An int, in the shortest form pickle uses: BININT1, BININT2, BININT or LONG1.
    void put_int(int64_t i) {
        if (i >= 0 && i < 256) {
            put('K');
            put(static_cast<char>(i));
        } else if (i >= 0 && i < 65536) {
            put('M');
            put_le(static_cast<uint64_t>(i), 2);
        } else if (i >= INT32_MIN && i <= INT32_MAX) {
            put('J');
            put_le(static_cast<uint32_t>(static_cast<int32_t>(i)), 4);
        } else {
            // LONG1 with pickle.encode_long's bytes: the shortest little-endian two's complement.
            // Drop top bytes that only repeat the sign of the byte below them.
            auto const u = static_cast<uint64_t>(i);
            auto const byte = [u](int k) { return static_cast<uint8_t>(u >> (8 * k)); };
            int nbytes = 8;
            while (nbytes > 1 &&
                   (byte(nbytes - 1) == (((byte(nbytes - 2) & 0x80) != 0) ? 0xff : 0x00))) {
                --nbytes;
            }
            put('\x8a');
            put(static_cast<char>(nbytes));
            put_le(u, nbytes);
        }
    }

    // Push module.name once into the memo; return its memo index.
    [[nodiscard]] auto global_ref(std::string_view module, std::string_view name) -> uint32_t {
        put_str(module);
        put_str(name);
        put({STACK_GLOBAL, MEMOIZE, POP});
        return memo_len_++;
    }

    // Push a dict key once into the memo; return its memo index.
    [[nodiscard]] auto key_ref(std::string_view name) -> uint32_t {
        put_str(name);
        put({MEMOIZE, POP});
        return memo_len_++;
    }

    // First use writes the string and memoizes it; later uses fetch it from the memo. The memo owns
    // a copy of each distinct string and is looked up by view, so lookups do not allocate.
    // Return its memo index.
    auto string(PyObject* s) -> uint32_t { return string(utf8(s)); }

    // The same, for a string's UTF-8 bytes.
    auto string(std::string_view v) -> uint32_t {
        if (auto const it = strings_.find(v); it != strings_.end()) {
            put_get(it->second);
            return it->second;
        }
        strings_.emplace(v, memo_len_);
        put_str(v);
        put(MEMOIZE);
        return memo_len_++;
    }

    // A torch.Size, as pickle reduces it: torch.Size(tuple_of_ints).
    void size(PyObject* dims) {
        if (!is(dims, size_cls_)) {
            throw py::type_error("expected torch.Size");
        }
        put_size(static_cast<size_t>(PyTuple_GET_SIZE(dims)),
                 [dims](size_t j) { return as_int(PyTuple_GET_ITEM(dims, j)); });
    }

    // Write a torch.Size from a state dict table's dims, as size(PyObject*) does.
    void size(std::span<int64_t const> dims) {
        put_size(dims.size(), [dims](size_t j) { return dims[j]; });
    }

    // The same, for dims from a write-result table.
    void size(Int64s const& dims) {
        put_size(dims.size(), [&dims](size_t j) { return dims[j]; });
    }

    // Write torch.Size of n dims, the j-th of which dim(j) returns.
    template <typename Dim>
    void put_size(size_t const n, Dim dim) {
        put_get(size_ref_);
        if (n == 0) {
            put(EMPTY_TUPLE);
        } else {
            if (n > 3) {
                put(MARK);
            }
            for (size_t j = 0; j < n; ++j) {
                put_int(dim(j));
            }
            put(n == 1 ? TUPLE1 : n == 2 ? TUPLE2 : n == 3 ? TUPLE3 : TUPLE);
        }
        put({TUPLE1, REDUCE});
    }

    // An object left to the stdlib pickler, spliced in from small_pickle's opcodes.
    void small(PyObject* obj) {
        py::object const b = small_pickle_(py::handle(obj));
        char* p = nullptr;
        Py_ssize_t n = 0;
        if (PyBytes_AsStringAndSize(b.ptr(), &p, &n) != 0) {
            throw py::error_already_set();
        }
        append(std::string_view(p, static_cast<size_t>(n)));
    }

    // The low nbytes bytes of v, least significant first.
    void put_le(uint64_t v, int nbytes) {
        for (int k = 0; k < nbytes; ++k) {
            put(static_cast<char>((v >> (8 * k)) & 0xff));
        }
    }

    py::object small_pickle_;
    py::object index_cls_, info_cls_, size_cls_;
    py::str a_fqn_{"fqn"}, a_index_{"index"}, a_offset_{"offset"};
    py::str a_relative_path_{"relative_path"}, a_length_{"length"};
    py::str a_transform_{"transform_descriptors"};
    // Memo indices of the classes and dict keys written by prelude().
    uint32_t size_ref_ = 0, metadata_ref_ = 0, index_ref_ = 0, info_ref_ = 0, chunk_ref_ = 0;
    uint32_t tensor_ref_ = 0, bytes_ref_ = 0;
    uint32_t k_fqn_ = 0, k_index_ = 0, k_offset_ = 0, k_relative_path_ = 0, k_length_ = 0;
    uint32_t k_offsets_ = 0, k_sizes_ = 0, k_properties_ = 0, k_size_ = 0, k_chunks_ = 0;

    // Not a std::string: libstdc++ declares std::string's members as extern templates, compiled
    // into the shared library, so they are never inlined, and each byte appended was a function
    // call through the PLT. std::vector<char> is instantiated here and its push_back inlines.
    // Building with -fno-plt kept most of the cost: it is the call, more than the PLT.
    std::vector<char> out_;
    uint32_t memo_len_ = 0;
    // Each state dict table entry's fqn memo index, once written; index tables name fqns by entry.
    std::vector<std::optional<uint32_t>> fqn_memo_;
    StringMap<uint32_t> strings_;
};

}  // namespace

auto nvrx::checkpoint::dumps_table(py::handle table,
                                   py::object small_pickle,
                                   py::handle storage_data,
                                   py::object const& storage_rows) -> py::bytes {
    return Writer(std::move(small_pickle))
        .dumps(StateDictTableView(table), storage_data, storage_rows);
}
