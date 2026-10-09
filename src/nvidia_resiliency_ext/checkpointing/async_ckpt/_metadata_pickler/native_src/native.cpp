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

// Native version of `writer._MetadataPickler`: writes the standard pickle opcodes for a
// torch DCP `Metadata` directly. Its output is byte-identical to the Python writer. The few small
// objects (TensorProperties, StorageMeta, ...) are encoded by the Python `small_pickle` callback.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
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

// Columns and flags of a write-result table's entries, as in table.py.
enum Column { FQN, INDEX, OFFSET_NDIM, PATH, OFFSET, LENGTH, FLAGS, COLUMNS };
constexpr int64_t FLAG_INDEX = 1, FLAG_OFFSET = 2, FLAG_OFFSET_SIZE = 4;
using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// New reference to obj.name, throwing on error.
[[nodiscard]] py::object getattr(PyObject* obj, PyObject* name) {
    PyObject* const r = PyObject_GetAttr(obj, name);
    if (!r) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(r);
}

// New reference to obj.__dict__, the state pickle writes for the metadata classes.
[[nodiscard]] py::object instance_dict(PyObject* obj) {
    static PyObject* const name = PyUnicode_InternFromString("__dict__");
    const py::object d = getattr(obj, name);
    if (!PyDict_CheckExact(d.ptr())) throw py::type_error("expected a __dict__");
    return d;
}

// Borrowed reference to dict[key], or nullptr if key is missing.
[[nodiscard]] PyObject* dict_get(PyObject* dict, PyObject* key) {
    PyObject* const r = PyDict_GetItemWithError(dict, key);
    if (!r && PyErr_Occurred()) throw py::error_already_set();
    return r;
}

// Borrowed reference to dict[key], throwing if key is missing.
[[nodiscard]] PyObject* dict_item(PyObject* dict, PyObject* key) {
    PyObject* const r = dict_get(dict, key);
    if (!r) throw py::type_error("missing attribute");
    return r;
}

// UTF-8 view of a str's characters, valid while s is alive.
[[nodiscard]] std::string_view utf8(PyObject* s) {
    if (!PyUnicode_CheckExact(s)) throw py::type_error("expected str");
    Py_ssize_t n;
    const char* const p = PyUnicode_AsUTF8AndSize(s, &n);
    if (!p) throw py::error_already_set();
    return {p, static_cast<size_t>(n)};
}

// The value of an int that fits int64. Like the other checks here, it accepts only the exact type:
// anything else (a bool, a tuple for a torch.Size, ...) would unpickle as a different type, so it
// raises and the caller falls back to pickle.dump.
[[nodiscard]] int64_t as_int(PyObject* o) {
    if (!PyLong_CheckExact(o)) throw py::type_error("expected int");
    const long long v = PyLong_AsLongLong(o);
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
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

    // The pickle of md: a NEWOBJ of Metadata built from its __dict__, field by field. With tables
    // (write-result tables: (entry, offsets, strings) each), storage_data is written from them.
    [[nodiscard]] py::bytes dumps(PyObject* md, const py::object& tables) {
        if (!is(md, metadata_cls_)) throw py::type_error("expected Metadata");
        const py::object fields = instance_dict(md);
        size_t n_storage = 0;
        if (!tables.is_none()) {
            for (const py::handle t : tables) {
                n_storage += t.cast<py::tuple>()[0].cast<Int64Array>().shape(0);
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
        if (!items) throw py::error_already_set();
        for (const py::handle item : items) {
            const std::string_view field = utf8(PyTuple_GET_ITEM(item.ptr(), 0));
            PyObject* const value = PyTuple_GET_ITEM(item.ptr(), 1);
            put_str(field);
            if (field == "state_dict_metadata") {
                dump_state_dict_metadata(value);
            } else if (field == "storage_data") {
                if (tables.is_none()) dump_storage_data(value); else dump_storage_tables(tables);
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
        if (!PyDict_CheckExact(dict)) throw py::type_error("expected dict");
        put(EMPTY_DICT);
        const Py_ssize_t n = PyDict_Size(dict);
        Py_ssize_t pos = 0;
        PyObject *fqn, *v;
        for (Py_ssize_t i = 0; PyDict_Next(dict, &pos, &fqn, &v); ++i) {
            // small() runs Python code, which could change the dict: fqn and v are not used after
            // it, and the dict's size is checked after it.
            if (i % BATCH == 0) put(MARK);
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
                    if (j % BATCH == 0) put(MARK);
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
                    if (j % BATCH == BATCH - 1 || j == nc - 1) put(APPENDS);
                }
                put({SETITEMS, BUILD});
                check_size(dict, n);  // after small() of the properties
            }
            if (i % BATCH == BATCH - 1 || i == n - 1) put(SETITEMS);
        }
    }

    // Metadata.storage_data: MetadataIndex -> _StorageInfo.
    void dump_storage_data(PyObject* dict) {
        if (!PyDict_CheckExact(dict)) throw py::type_error("expected dict");
        put(EMPTY_DICT);
        const Py_ssize_t n = PyDict_Size(dict);
        Py_ssize_t pos = 0;
        PyObject *idx, *info;
        for (Py_ssize_t i = 0; PyDict_Next(dict, &pos, &idx, &info); ++i) {
            if (i % BATCH == 0) put(MARK);
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
                put_get(index_ref_);
                put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                put_get(k_fqn_);
                string(fqn);
                put_get(k_index_);
                if (index == Py_None) put(NONE); else put_int(as_int(index));
                if (offset) {
                    put_get(k_offset_);
                    if (offset == Py_None) put(NONE); else size(offset);
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
                put_get(info_ref_);
                put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                put_get(k_relative_path_);
                string(dict_item(state.ptr(), a_relative_path_.ptr()));
                put_get(k_offset_);
                put_int(as_int(dict_item(state.ptr(), a_offset_.ptr())));
                put_get(k_length_);
                put_int(as_int(dict_item(state.ptr(), a_length_.ptr())));
                put({SETITEMS, BUILD});
            }
            if (i % BATCH == BATCH - 1 || i == n - 1) put(SETITEMS);
        }
    }

    // Metadata.storage_data from write-result tables, as dump_storage_data writes the dict that
    // finish builds from the same write results.
    void dump_storage_tables(const py::object& tables) {
        put(EMPTY_DICT);
        size_t total = 0;
        std::vector<py::tuple> parts;
        for (const py::handle t : tables) {
            parts.push_back(t.cast<py::tuple>());
            total += parts.back()[0].cast<Int64Array>().shape(0);
        }
        size_t i = 0;
        for (const py::tuple& part : parts) {
            const Int64Array entry = part[0].cast<Int64Array>();
            const Int64Array offsets = part[1].cast<Int64Array>();
            const py::list strings_list = part[2].cast<py::list>();
            if (entry.ndim() != 2 || entry.shape(1) != COLUMNS || offsets.ndim() != 1) {
                throw py::value_error("malformed write-result table");
            }
            // Views of the strs' UTF-8 buffers: strings_list holds the strs while they are used.
            std::vector<std::string_view> strings;
            strings.reserve(strings_list.size());
            for (const py::handle str : strings_list) strings.push_back(utf8(str.ptr()));
            const auto e = entry.unchecked<2>();
            const int64_t* const dims = offsets.data();
            const size_t n_dims = static_cast<size_t>(offsets.shape(0));
            size_t pos = 0;
            const auto string_at = [&strings](int64_t id) {
                if (id < 0 || static_cast<size_t>(id) >= strings.size()) {
                    throw py::value_error("write-result table: string id out of range");
                }
                return strings[id];
            };
            for (py::ssize_t row = 0; row < e.shape(0); ++row, ++i) {
                if (i % BATCH == 0) put(MARK);
                const int64_t flags = e(row, FLAGS);
                put_get(index_ref_);
                put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                put_get(k_fqn_);
                string(string_at(e(row, FQN)));
                put_get(k_index_);
                if (flags & FLAG_INDEX) put_int(e(row, INDEX)); else put(NONE);
                if (flags & FLAG_OFFSET) {
                    put_get(k_offset_);
                    if (flags & FLAG_OFFSET_SIZE) {
                        const int64_t ndim = e(row, OFFSET_NDIM);
                        if (ndim < 0 || pos + static_cast<size_t>(ndim) > n_dims) {
                            throw py::value_error("write-result table: offsets out of range");
                        }
                        size(dims + pos, static_cast<size_t>(ndim));
                        pos += static_cast<size_t>(ndim);
                    } else {
                        put(NONE);
                    }
                }
                put({SETITEMS, BUILD});
                put_get(info_ref_);
                put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                put_get(k_relative_path_);
                string(string_at(e(row, PATH)));
                put_get(k_offset_);
                put_int(e(row, OFFSET));
                put_get(k_length_);
                put_int(e(row, LENGTH));
                put({SETITEMS, BUILD});
                if (i % BATCH == BATCH - 1 || i == total - 1) put(SETITEMS);
            }
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
        if (!is(dims, size_cls_)) throw py::type_error("expected torch.Size");
        put_get(size_ref_);
        const Py_ssize_t n = PyTuple_GET_SIZE(dims);
        if (n == 0) {
            put(EMPTY_TUPLE);
        } else {
            if (n > 3) put(MARK);
            for (Py_ssize_t j = 0; j < n; ++j) put_int(as_int(PyTuple_GET_ITEM(dims, j)));
            put(n == 1 ? TUPLE1 : n == 2 ? TUPLE2 : n == 3 ? TUPLE3 : TUPLE);
        }
        put({TUPLE1, REDUCE});
    }

    // The same, for n dims from a write-result table.
    void size(const int64_t* dims, size_t n) {
        put_get(size_ref_);
        if (n == 0) {
            put(EMPTY_TUPLE);
        } else {
            if (n > 3) put(MARK);
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
        if (PyBytes_AsStringAndSize(b.ptr(), &p, &n) != 0) throw py::error_already_set();
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
// the stdlib pickler. With tables, storage_data is written from them (see Writer::dumps).
[[nodiscard]] py::bytes dumps(py::handle md, py::object small_pickle, py::object tables) {
    return Writer(std::move(small_pickle)).dumps(md.ptr(), tables);
}

}  // namespace

PYBIND11_MODULE(native, m) {
    m.doc() = "Fast pickling of torch.distributed.checkpoint Metadata.";
    m.def("dumps", &dumps, py::arg("metadata"), py::arg("small_pickle"),
          py::arg("tables") = py::none(),
          "Return the pickle of a torch.distributed.checkpoint Metadata, as standard pickle "
          "bytes.");
}
