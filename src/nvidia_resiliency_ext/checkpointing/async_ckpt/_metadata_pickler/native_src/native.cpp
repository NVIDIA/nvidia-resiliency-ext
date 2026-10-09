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

// Native version of `pickler._MetadataPickler`: writes the standard pickle opcodes for a
// torch DCP `Metadata` directly. Its output is byte-identical to the Python writer. The few small
// objects (TensorProperties, StorageMeta, ...) are encoded by the Python `small_pickle` callback.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

constexpr char PROTO = '\x80', STOP = '.', MARK = '(', POP = '0';
constexpr char EMPTY_DICT = '}', EMPTY_LIST = ']', EMPTY_TUPLE = ')';
constexpr char SETITEMS = 'u', APPENDS = 'e', BUILD = 'b', NEWOBJ = '\x81', REDUCE = 'R';
constexpr char STACK_GLOBAL = '\x93', MEMOIZE = '\x94', NONE = 'N';
constexpr char TUPLE = 't', TUPLE1 = '\x85', TUPLE2 = '\x86', TUPLE3 = '\x87';
constexpr size_t BATCH = 1000;  // same batching as the stdlib pickler

// Write-result tables, as table.py encodes them: little-endian int64s, header first.
static_assert(std::endian::native == std::endian::little, "write-result tables are little-endian");
constexpr int64_t TABLE_MAGIC = 0x4E56525854424C31, TABLE_VERSION = 1;
enum class Header : uint8_t { Magic, Version, Entries, Strings, StringBytes, Offsets };
constexpr size_t HEADER = 8;  // int64s
enum class Column : uint8_t { Fqn, Index, OffsetNdim, Path, Offset, Length, Flags };
constexpr size_t COLUMNS = 7;
constexpr int64_t FLAG_INDEX = 0x1, FLAG_OFFSET = 0x2, FLAG_OFFSET_SIZE = 0x4;
// The gathered tables: one zero-padded table per row.
using Rows = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// Reject a malformed table, before anything reads past its row.
[[noreturn]] void bad_table(std::string_view what) {
    throw py::value_error("write-result table: " + std::string(what));
}

// A run of little-endian int64s in a span of bytes, which need not be aligned.
class Int64s {
   public:
    // The int64s in bytes, whose size is a multiple of 8.
    explicit Int64s(std::span<const std::byte> bytes) : bytes_(bytes) {}

    // The number of int64s.
    [[nodiscard]] size_t size() const { return bytes_.size() / sizeof(int64_t); }

    // The i-th int64, i < size(). bit_cast needs an object to read, and there is none at an
    // arbitrary offset of the row, so the bytes are copied into one; this compiles to one load.
    [[nodiscard]] int64_t operator[](size_t i) const {
        std::array<std::byte, sizeof(int64_t)> v;
        std::ranges::copy(bytes_.subspan(sizeof(int64_t) * i).first<sizeof(int64_t)>(), v.begin());
        return std::bit_cast<int64_t>(v);
    }

    // The field of a record (a header or an entry) that an enum names.
    template <typename Field>
        requires std::is_enum_v<Field>
    [[nodiscard]] int64_t operator[](Field field) const {
        return (*this)[static_cast<size_t>(field)];
    }

    // The count int64s from pos, pos + count <= size().
    [[nodiscard]] Int64s subspan(size_t pos, size_t count) const {
        return Int64s(bytes_.subspan(sizeof(int64_t) * pos, sizeof(int64_t) * count));
    }

   private:
    std::span<const std::byte> bytes_;
};

// A write-result table within its row of the gathered buffer. parse checks that its sections lie
// within the row; the accessors check what its entries refer to.
class WriteResultTable {
   public:
    // The table in a row.
    [[nodiscard]] static WriteResultTable parse(std::span<const std::byte> row) {
        if (row.size() < sizeof(int64_t) * HEADER) {
            bad_table("truncated");
        }
        const Int64s header(row.first(sizeof(int64_t) * HEADER));
        if (header[Header::Magic] != TABLE_MAGIC || header[Header::Version] != TABLE_VERSION) {
            bad_table("not a table of this version");
        }
        std::span<const std::byte> rest = row.subspan(sizeof(int64_t) * HEADER);
        // The next section: as many items of item_size bytes as the header's count field says,
        // checked to fit in the rest of the row (comparing counts, so sizes cannot overflow).
        const auto take = [&](Header count, size_t item_size) {
            const int64_t n = header[count];
            if (n < 0 || static_cast<uint64_t>(n) > rest.size() / item_size) {
                bad_table("truncated");
            }
            const auto section = rest.first(static_cast<size_t>(n) * item_size);
            rest = rest.subspan(section.size());
            return section;
        };
        const Int64s string_len(take(Header::Strings, sizeof(int64_t)));
        const Int64s entries(take(Header::Entries, sizeof(int64_t) * COLUMNS));
        const Int64s offsets(take(Header::Offsets, sizeof(int64_t)));
        const std::span<const std::byte> blob = take(Header::StringBytes, 1);
        return WriteResultTable(string_len, entries, offsets,
                                {reinterpret_cast<const char*>(blob.data()), blob.size()});
    }

    // The number of entries.
    [[nodiscard]] size_t entries() const { return entries_.size() / COLUMNS; }

    // Entry r < entries(): its COLUMNS int64s, indexed by Column.
    [[nodiscard]] Int64s entry(size_t r) const { return entries_.subspan(COLUMNS * r, COLUMNS); }

    // Replace out's contents with the strings, as views into the row.
    void split_strings(std::vector<std::string_view>& out) const {
        out.clear();
        std::string_view rest = blob_;
        for (size_t k = 0; k < string_len_.size(); ++k) {
            const int64_t len = string_len_[k];
            if (len < 0 || static_cast<uint64_t>(len) > rest.size()) {
                bad_table("string lengths exceed the string bytes");
            }
            out.push_back(rest.substr(0, static_cast<size_t>(len)));
            rest.remove_prefix(static_cast<size_t>(len));
        }
        if (!rest.empty()) {
            bad_table("string lengths don't match the string bytes");
        }
    }

    // The ndim offset dims from position pos of the offsets, checked to lie within them.
    [[nodiscard]] Int64s offset_dims(size_t pos, int64_t ndim) const {
        if (ndim < 0 || pos > offsets_.size() ||
            static_cast<uint64_t>(ndim) > offsets_.size() - pos) {
            bad_table("offsets out of range");
        }
        return offsets_.subspan(pos, static_cast<size_t>(ndim));
    }

    // Check that the entries' offsets, used up to position pos, are all of them.
    void check_offsets_used(size_t pos) const {
        if (pos != offsets_.size()) {
            bad_table("offset dims don't match the offsets");
        }
    }

   private:
    // The sections parse found.
    WriteResultTable(Int64s string_len, Int64s entries, Int64s offsets, std::string_view blob)
        : string_len_(string_len), entries_(entries), offsets_(offsets), blob_(blob) {}

    Int64s string_len_;  // the byte length of each string
    Int64s entries_;     // entries() rows of COLUMNS
    Int64s offsets_;     // the entries' offset dims, concatenated
    std::string_view blob_;  // the strings, concatenated
};

// The tables in the rows of a 2-D uint8 array.
[[nodiscard]] std::vector<WriteResultTable> parse_tables(const Rows& rows) {
    if (rows.ndim() != 2) {
        throw py::value_error("expected a 2-D array of write-result tables");
    }
    const size_t width = static_cast<size_t>(rows.shape(1));
    const std::span<const std::byte> bytes =
        std::as_bytes(std::span(rows.data(), static_cast<size_t>(rows.size())));
    std::vector<WriteResultTable> tables;
    tables.reserve(rows.shape(0));
    for (size_t r = 0; r < static_cast<size_t>(rows.shape(0)); ++r) {
        tables.push_back(WriteResultTable::parse(bytes.subspan(width * r, width)));
    }
    return tables;
}

// New reference to obj.name, throwing on error.
[[nodiscard]] py::object getattr(PyObject* obj, PyObject* name) {
    PyObject* const r = PyObject_GetAttr(obj, name);
    if (!r) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(r);
}

// New reference to obj.__dict__, the state pickle writes for the metadata classes.
[[nodiscard]] py::object instance_dict(PyObject* obj) {
    static PyObject* const name = PyUnicode_InternFromString("__dict__");
    const py::object d = getattr(obj, name);
    if (!PyDict_CheckExact(d.ptr())) {
        throw py::type_error("expected a __dict__");
    }
    return d;
}

// Borrowed reference to dict[key], or nullptr if key is missing.
[[nodiscard]] PyObject* dict_get(PyObject* dict, PyObject* key) {
    PyObject* const r = PyDict_GetItemWithError(dict, key);
    if (!r && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return r;
}

// Borrowed reference to dict[key], throwing if key is missing.
[[nodiscard]] PyObject* dict_item(PyObject* dict, PyObject* key) {
    PyObject* const r = dict_get(dict, key);
    if (!r) {
        throw py::type_error("missing attribute");
    }
    return r;
}

// UTF-8 view of a str's characters, valid while s is alive.
[[nodiscard]] std::string_view utf8(PyObject* s) {
    if (!PyUnicode_CheckExact(s)) {
        throw py::type_error("expected str");
    }
    Py_ssize_t n;
    const char* const p = PyUnicode_AsUTF8AndSize(s, &n);
    if (!p) {
        throw py::error_already_set();
    }
    return {p, static_cast<size_t>(n)};
}

// The value of an int that fits int64. Like the other checks here, it accepts only the exact type:
// anything else (a bool, a tuple for a torch.Size, ...) would unpickle as a different type, so it
// raises and the caller falls back to pickle.dump.
[[nodiscard]] int64_t as_int(PyObject* o) {
    if (!PyLong_CheckExact(o)) {
        throw py::type_error("expected int");
    }
    const long long v = PyLong_AsLongLong(o);
    if (v == -1 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return v;
}

// Writes the pickle of one Metadata into out_; use once per dump.
class Writer {
   public:
    // Look up the metadata classes the encoding checks objects against.
    explicit Writer(py::object small_pickle) : small_pickle_(std::move(small_pickle)) {
        py::module_ meta = py::module_::import("torch.distributed.checkpoint.metadata");
        metadata_cls_ = meta.attr("Metadata");
        tensor_cls_ = meta.attr("TensorStorageMetadata");
        bytes_cls_ = meta.attr("BytesStorageMetadata");
        chunk_cls_ = meta.attr("ChunkStorageMetadata");
        index_cls_ = meta.attr("MetadataIndex");
        info_cls_ =
            py::module_::import("torch.distributed.checkpoint.filesystem").attr("_StorageInfo");
        size_cls_ = py::module_::import("torch").attr("Size");
    }

    // The pickle of md: a NEWOBJ of Metadata built from its __dict__, field by field. With rows
    // (the gathered write-result tables), storage_data is written from them.
    [[nodiscard]] py::bytes dumps(PyObject* md, const py::object& rows) {
        if (!is(md, metadata_cls_)) {
            throw py::type_error("expected Metadata");
        }
        const py::object fields = instance_dict(md);
        // Holds the rows the tables point into.
        const Rows rows_array = rows.is_none() ? Rows() : rows.cast<Rows>();
        const std::vector<WriteResultTable> tables =
            rows.is_none() ? std::vector<WriteResultTable>() : parse_tables(rows_array);
        size_t n_storage = 0;
        if (!rows.is_none()) {
            for (const WriteResultTable& t : tables) {
                n_storage += t.entries();
            }
        } else if (PyObject* const storage_data = PyDict_GetItemString(fields.ptr(), "storage_data");
                   storage_data && PyDict_Check(storage_data)) {
            n_storage = PyDict_Size(storage_data);
        }
        // About 98 bytes per storage entry: its MetadataIndex and
        // _StorageInfo, plus the matching chunk in state_dict_metadata.
        out_.reserve(4096 + 100 * n_storage);

        prelude();
        put_get(metadata_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        // A snapshot, which holds the fields while small() runs Python code that could change them.
        const auto items = py::reinterpret_steal<py::list>(PyDict_Items(fields.ptr()));
        if (!items) {
            throw py::error_already_set();
        }
        for (const py::handle item : items) {
            const std::string_view field = utf8(PyTuple_GET_ITEM(item.ptr(), 0));
            PyObject* const value = PyTuple_GET_ITEM(item.ptr(), 1);
            put_str(field);
            if (field == "state_dict_metadata") {
                dump_state_dict_metadata(value);
            } else if (field == "storage_data") {
                if (rows.is_none()) {
                    dump_storage_data(value);
                } else {
                    dump_storage_tables(tables);
                }
            } else {
                small(value);
            }
        }
        put({SETITEMS, BUILD, STOP});
        return py::bytes(out_);
    }

   private:
    // Protocol 4 header, then the classes and dict keys the encoding refers to, each memoized once.
    void prelude() {
        put({PROTO, '\x04'});
        constexpr const char* meta = "torch.distributed.checkpoint.metadata";
        size_ref_ = global_ref("torch", "Size");
        metadata_ref_ = global_ref(meta, "Metadata");
        index_ref_ = global_ref(meta, "MetadataIndex");
        info_ref_ = global_ref("torch.distributed.checkpoint.filesystem", "_StorageInfo");
        chunk_ref_ = global_ref(meta, "ChunkStorageMetadata");
        tensor_ref_ = global_ref(meta, "TensorStorageMetadata");
        bytes_ref_ = global_ref(meta, "BytesStorageMetadata");
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

    // Metadata.state_dict_metadata: fqn -> TensorStorageMetadata or BytesStorageMetadata.
    void dump_state_dict_metadata(PyObject* dict) {
        if (!PyDict_CheckExact(dict)) {
            throw py::type_error("expected dict");
        }
        put(EMPTY_DICT);
        const Py_ssize_t n = PyDict_Size(dict);
        Py_ssize_t pos = 0;
        PyObject *fqn, *v;
        for (Py_ssize_t i = 0; PyDict_Next(dict, &pos, &fqn, &v); ++i) {
            // small() runs Python code, which could change the dict: fqn and v are not used after
            // it, and the dict's size is checked after it.
            begin_item(i);
            string(fqn);
            if (is(v, bytes_cls_) && PyDict_Size(instance_dict(v).ptr()) == 0) {
                put_get(bytes_ref_);
                put({EMPTY_TUPLE, NEWOBJ});
            } else if (!is(v, tensor_cls_)) {
                small(v);
                check_size(dict, n);
            } else {
                const py::object state = instance_dict(v);
                const auto chunks =
                    py::reinterpret_borrow<py::object>(dict_item(state.ptr(), a_chunks_.ptr()));
                if (PyDict_Size(state.ptr()) != 3) {
                    throw py::type_error("unexpected TensorStorageMetadata attributes");
                }
                if (!PyList_CheckExact(chunks.ptr())) {
                    throw py::type_error("expected list of chunks");
                }
                put_get(tensor_ref_);
                put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                put_get(k_properties_);
                small(dict_item(state.ptr(), a_properties_.ptr()));
                put_get(k_size_);
                size(dict_item(state.ptr(), a_size_.ptr()));
                put_get(k_chunks_);
                put(EMPTY_LIST);
                const Py_ssize_t nc = PyList_GET_SIZE(chunks.ptr());
                for (Py_ssize_t j = 0; j < nc; ++j) {
                    begin_item(j);
                    PyObject* const c = PyList_GET_ITEM(chunks.ptr(), j);
                    if (!is(c, chunk_cls_)) {
                        small(c);
                        check_size(chunks.ptr(), nc);
                    } else {
                        const py::object chunk = instance_dict(c);
                        if (PyDict_Size(chunk.ptr()) != 2) {
                            throw py::type_error("unexpected ChunkStorageMetadata attributes");
                        }
                        put_get(chunk_ref_);
                        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                        put_get(k_offsets_);
                        size(dict_item(chunk.ptr(), a_offsets_.ptr()));
                        put_get(k_sizes_);
                        size(dict_item(chunk.ptr(), a_sizes_.ptr()));
                        put({SETITEMS, BUILD});
                    }
                    end_item(j, nc, APPENDS);
                }
                put({SETITEMS, BUILD});
                check_size(dict, n);  // after small() of the properties
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
        const Py_ssize_t n = PyDict_Size(dict);
        Py_ssize_t pos = 0;
        PyObject *idx, *info;
        for (Py_ssize_t i = 0; PyDict_Next(dict, &pos, &idx, &info); ++i) {
            begin_item(i);
            // Owns info while small(idx) runs Python code, which could remove it from the dict.
            py::object keep_info;
            if (!is(idx, index_cls_)) {
                keep_info = py::reinterpret_borrow<py::object>(info);
                small(idx);
                check_size(dict, n);
            } else {
                // MetadataIndex sets offset only when it is given; pickle its __dict__.
                const py::object state = instance_dict(idx);
                PyObject* const fqn = dict_item(state.ptr(), a_fqn_.ptr());
                PyObject* const index = dict_item(state.ptr(), a_index_.ptr());
                PyObject* const offset = dict_get(state.ptr(), a_offset_.ptr());
                if (PyDict_Size(state.ptr()) != (offset ? 3 : 2)) {
                    throw py::type_error("unexpected MetadataIndex attributes");
                }
                put_index_start(utf8(fqn), index == Py_None ? std::nullopt
                                                            : std::optional(as_int(index)));
                if (offset) {
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
            const py::object state = is(info, info_cls_) ? instance_dict(info) : py::object();
            PyObject* const transforms =
                state ? dict_get(state.ptr(), a_transform_.ptr()) : nullptr;
            if (!state || (transforms && transforms != Py_None)) {
                small(info);
                check_size(dict, n);
            } else {
                if (PyDict_Size(state.ptr()) != (transforms ? 4 : 3)) {
                    throw py::type_error("unexpected _StorageInfo attributes");
                }
                put_storage_info(utf8(dict_item(state.ptr(), a_relative_path_.ptr())),
                                 as_int(dict_item(state.ptr(), a_offset_.ptr())),
                                 as_int(dict_item(state.ptr(), a_length_.ptr())));
            }
            end_item(i, n, SETITEMS);
        }
    }

    // Metadata.storage_data from write-result tables, as dump_storage_data writes the dict that
    // finish builds from the same write results.
    void dump_storage_tables(const std::vector<WriteResultTable>& tables) {
        put(EMPTY_DICT);
        size_t total = 0;
        for (const WriteResultTable& t : tables) {
            total += t.entries();
        }
        size_t i = 0;
        std::vector<std::string_view> strings;  // the current table's
        for (const WriteResultTable& t : tables) {
            t.split_strings(strings);
            size_t pos = 0;  // into the table's offsets
            for (size_t r = 0; r < t.entries(); ++r, ++i) {
                begin_item(i);
                dump_table_entry(t, t.entry(r), strings, pos);
                end_item(i, total, SETITEMS);
            }
            t.check_offsets_used(pos);
        }
    }

    // One table entry: its MetadataIndex and _StorageInfo. pos is where its offset dims start in
    // the table's offsets; it moves past them.
    void dump_table_entry(const WriteResultTable& t, const Int64s& entry,
                          const std::vector<std::string_view>& strings, size_t& pos) {
        const int64_t flags = entry[Column::Flags];
        const std::optional<int64_t> index =
            flags & FLAG_INDEX ? std::optional(entry[Column::Index]) : std::nullopt;
        put_index_start(string_at(strings, entry[Column::Fqn]), index);
        if (flags & FLAG_OFFSET) {
            put_get(k_offset_);
            if (flags & FLAG_OFFSET_SIZE) {
                const int64_t ndim = entry[Column::OffsetNdim];
                size(t.offset_dims(pos, ndim));
                pos += static_cast<size_t>(ndim);
            } else {
                put(NONE);
            }
        }
        put({SETITEMS, BUILD});
        put_storage_info(string_at(strings, entry[Column::Path]), entry[Column::Offset],
                         entry[Column::Length]);
    }

    // String id of a table's strings, checked.
    [[nodiscard]] static std::string_view string_at(const std::vector<std::string_view>& strings,
                                                    int64_t id) {
        if (id < 0 || static_cast<uint64_t>(id) >= strings.size()) {
            bad_table("string id out of range");
        }
        return strings[static_cast<size_t>(id)];
    }

    // A MetadataIndex up to its offset: the caller writes the offset, if it has one, and closes the
    // index with put({SETITEMS, BUILD}).
    void put_index_start(std::string_view fqn, std::optional<int64_t> index) {
        put_get(index_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        put_get(k_fqn_);
        string(fqn);
        put_get(k_index_);
        if (index) {
            put_int(*index);
        } else {
            put(NONE);
        }
    }

    // A _StorageInfo without transform descriptors.
    void put_storage_info(std::string_view path, int64_t offset, int64_t length) {
        put_get(info_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        put_get(k_relative_path_);
        string(path);
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
    [[nodiscard]] static bool is(PyObject* obj, const py::object& cls) {
        return reinterpret_cast<PyObject*>(Py_TYPE(obj)) == cls.ptr();
    }

    // Append raw opcode bytes.
    void put(char c) { out_.push_back(c); }
    void put(std::initializer_list<char> cs) { out_.append(cs.begin(), cs.end()); }

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
        out_.append(s);
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
            const uint64_t u = static_cast<uint64_t>(i);
            const auto byte = [u](int k) { return static_cast<uint8_t>(u >> (8 * k)); };
            int nbytes = 8;
            while (nbytes > 1 && (byte(nbytes - 1) == (byte(nbytes - 2) & 0x80 ? 0xff : 0x00))) {
                --nbytes;
            }
            put('\x8a');
            put(static_cast<char>(nbytes));
            put_le(u, nbytes);
        }
    }

    // Push module.name once into the memo; return its memo index.
    [[nodiscard]] uint32_t global_ref(std::string_view module, std::string_view name) {
        put_str(module);
        put_str(name);
        put({STACK_GLOBAL, MEMOIZE, POP});
        return memo_len_++;
    }

    // Push a dict key once into the memo; return its memo index.
    [[nodiscard]] uint32_t key_ref(std::string_view name) {
        put_str(name);
        put({MEMOIZE, POP});
        return memo_len_++;
    }

    // First use writes the string and memoizes it; later uses fetch it from the memo. The memo owns
    // a copy of each distinct string and is looked up by view, so lookups do not allocate.
    void string(PyObject* s) { string(utf8(s)); }

    // The same, for a string's UTF-8 bytes.
    void string(std::string_view v) {
        if (const auto it = strings_.find(v); it != strings_.end()) {
            put_get(it->second);
            return;
        }
        strings_.emplace(v, memo_len_++);
        put_str(v);
        put(MEMOIZE);
    }

    // A torch.Size, as pickle reduces it: torch.Size(tuple_of_ints).
    void size(PyObject* dims) {
        if (!is(dims, size_cls_)) {
            throw py::type_error("expected torch.Size");
        }
        put_get(size_ref_);
        const Py_ssize_t n = PyTuple_GET_SIZE(dims);
        if (n == 0) {
            put(EMPTY_TUPLE);
        } else {
            if (n > 3) {
                put(MARK);
            }
            for (Py_ssize_t j = 0; j < n; ++j) put_int(as_int(PyTuple_GET_ITEM(dims, j)));
            put(n == 1 ? TUPLE1 : n == 2 ? TUPLE2 : n == 3 ? TUPLE3 : TUPLE);
        }
        put({TUPLE1, REDUCE});
    }

    // The same, for dims from a write-result table.
    void size(const Int64s& dims) {
        const size_t n = dims.size();
        put_get(size_ref_);
        if (n == 0) {
            put(EMPTY_TUPLE);
        } else {
            if (n > 3) {
                put(MARK);
            }
            for (size_t j = 0; j < n; ++j) put_int(dims[j]);
            put(n == 1 ? TUPLE1 : n == 2 ? TUPLE2 : n == 3 ? TUPLE3 : TUPLE);
        }
        put({TUPLE1, REDUCE});
    }

    // An object left to the stdlib pickler, spliced in from small_pickle's opcodes.
    void small(PyObject* obj) {
        const py::object b = small_pickle_(py::handle(obj));
        char* p;
        Py_ssize_t n;
        if (PyBytes_AsStringAndSize(b.ptr(), &p, &n) != 0) {
            throw py::error_already_set();
        }
        out_.append(p, n);
    }

    // The low nbytes bytes of v, least significant first.
    void put_le(uint64_t v, int nbytes) {
        for (int k = 0; k < nbytes; ++k) put(static_cast<char>((v >> (8 * k)) & 0xff));
    }

    // Transparent hash, so strings_ can be looked up by string_view without building a std::string
    // per lookup. std::hash<std::string> is not transparent; this hashes both key types the same.
    struct StringHash {
        using is_transparent = void;
        [[nodiscard]] size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };

    py::object small_pickle_;
    py::object metadata_cls_, tensor_cls_, bytes_cls_, chunk_cls_, index_cls_, info_cls_;
    py::object size_cls_;
    py::str a_properties_{"properties"}, a_size_{"size"}, a_chunks_{"chunks"};
    py::str a_offsets_{"offsets"}, a_sizes_{"sizes"};
    py::str a_fqn_{"fqn"}, a_index_{"index"}, a_offset_{"offset"};
    py::str a_relative_path_{"relative_path"}, a_length_{"length"};
    py::str a_transform_{"transform_descriptors"};
    // Memo indices of the classes and dict keys written by prelude().
    uint32_t size_ref_ = 0, metadata_ref_ = 0, index_ref_ = 0, info_ref_ = 0, chunk_ref_ = 0;
    uint32_t tensor_ref_ = 0, bytes_ref_ = 0;
    uint32_t k_fqn_ = 0, k_index_ = 0, k_offset_ = 0, k_relative_path_ = 0, k_length_ = 0;
    uint32_t k_offsets_ = 0, k_sizes_ = 0, k_properties_ = 0, k_size_ = 0, k_chunks_ = 0;

    std::string out_;
    uint32_t memo_len_ = 0;
    std::unordered_map<std::string, uint32_t, StringHash, std::equal_to<>> strings_;
};

// Return the pickle of md; small_pickle(obj) returns the opcodes for an object the writer leaves to
// the stdlib pickler. With storage_rows, storage_data is written from them (see Writer::dumps).
[[nodiscard]] py::bytes dumps(py::handle md, py::object small_pickle, py::object storage_rows) {
    return Writer(std::move(small_pickle)).dumps(md.ptr(), storage_rows);
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Fast pickling of torch.distributed.checkpoint Metadata.";
    m.def("dumps", &dumps, py::arg("metadata"), py::arg("small_pickle"),
          py::arg("storage_rows") = py::none(),
          "Return the pickle of a torch.distributed.checkpoint Metadata, as standard pickle "
          "bytes.");
}
