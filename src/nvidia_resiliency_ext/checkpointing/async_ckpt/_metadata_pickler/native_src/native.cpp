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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace py = pybind11;

namespace {

constexpr char PROTO = '\x80', STOP = '.', MARK = '(', POP = '0';
constexpr char EMPTY_DICT = '}', EMPTY_LIST = ']', EMPTY_TUPLE = ')';
constexpr char SETITEMS = 'u', APPENDS = 'e', BUILD = 'b', NEWOBJ = '\x81', REDUCE = 'R';
constexpr char STACK_GLOBAL = '\x93', MEMOIZE = '\x94', NONE = 'N';
constexpr char TUPLE = 't', TUPLE1 = '\x85', TUPLE2 = '\x86', TUPLE3 = '\x87';
constexpr size_t BATCH = 1000;  // same batching as the stdlib pickler

// New reference to obj.name, throwing on error.
py::object getattr(PyObject* obj, PyObject* name) {
    PyObject* r = PyObject_GetAttr(obj, name);
    if (!r) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(r);
}

// New reference to obj.__dict__, the state pickle writes for the metadata classes.
py::object instance_dict(PyObject* obj) {
    static PyObject* name = PyUnicode_InternFromString("__dict__");
    py::object d = getattr(obj, name);
    if (!PyDict_CheckExact(d.ptr())) throw py::type_error("expected a __dict__");
    return d;
}

// Borrowed reference to dict[key], or nullptr if key is missing.
PyObject* dict_get(PyObject* dict, PyObject* key) {
    PyObject* r = PyDict_GetItemWithError(dict, key);
    if (!r && PyErr_Occurred()) throw py::error_already_set();
    return r;
}

// Borrowed reference to dict[key], throwing if key is missing.
PyObject* dict_item(PyObject* dict, PyObject* key) {
    PyObject* r = dict_get(dict, key);
    if (!r) throw py::type_error("missing attribute");
    return r;
}

// UTF-8 view of a str's characters, valid while s is alive.
std::string_view utf8(PyObject* s) {
    if (!PyUnicode_CheckExact(s)) throw py::type_error("expected str");
    Py_ssize_t n;
    const char* p = PyUnicode_AsUTF8AndSize(s, &n);
    if (!p) throw py::error_already_set();
    return {p, static_cast<size_t>(n)};
}

// The value of an int that fits int64. Like the other checks here, it accepts only the exact type:
// anything else (a bool, a tuple for a torch.Size, ...) would unpickle as a different type, so it
// raises and the caller falls back to pickle.dump.
int64_t as_int(PyObject* o) {
    if (!PyLong_CheckExact(o)) throw py::type_error("expected int");
    long long v = PyLong_AsLongLong(o);
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

// Writes the pickle of one Metadata into out_; use once per dump.
class Writer {
   public:
    // Look up the metadata classes the encoding checks objects against.
    explicit Writer(py::object small_pickle) : small_pickle_(std::move(small_pickle)) {
        py::module_ meta = py::module_::import("torch.distributed.checkpoint.metadata");
        tensor_cls_ = meta.attr("TensorStorageMetadata");
        bytes_cls_ = meta.attr("BytesStorageMetadata");
        chunk_cls_ = meta.attr("ChunkStorageMetadata");
        index_cls_ = meta.attr("MetadataIndex");
        info_cls_ = py::module_::import("torch.distributed.checkpoint.filesystem").attr("_StorageInfo");
        size_cls_ = py::module_::import("torch").attr("Size");
    }

    // The pickle of md: a NEWOBJ of Metadata built from its __dict__, field by field.
    py::bytes dumps(PyObject* md) {
        py::object fields = instance_dict(md);
        // About 98 bytes per storage entry: its MetadataIndex and
        // _StorageInfo, plus the matching chunk in state_dict_metadata.
        PyObject* storage_data = PyDict_GetItemString(fields.ptr(), "storage_data");
        size_t n_storage = storage_data && PyDict_Check(storage_data) ? PyDict_Size(storage_data) : 0;
        out_.reserve(4096 + 100 * n_storage);

        prelude();
        put_get(metadata_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        for (auto item : py::reinterpret_borrow<py::dict>(fields)) {
            std::string_view field = utf8(item.first.ptr());
            PyObject* value = item.second.ptr();
            put_str(field);
            if (field == "state_dict_metadata") {
                dump_state_dict_metadata(value);
            } else if (field == "storage_data") {
                dump_storage_data(value);
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
        const char* meta = "torch.distributed.checkpoint.metadata";
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
        Py_ssize_t pos = 0, n = PyDict_Size(dict);
        PyObject *fqn, *v;
        for (Py_ssize_t i = 0; PyDict_Next(dict, &pos, &fqn, &v); ++i) {
            if (i % BATCH == 0) put(MARK);
            string(fqn);
            if (is(v, bytes_cls_) && PyDict_Size(instance_dict(v).ptr()) == 0) {
                put_get(bytes_ref_);
                put({EMPTY_TUPLE, NEWOBJ});
            } else if (!is(v, tensor_cls_)) {
                small(v);
            } else {
                py::object state = instance_dict(v);
                PyObject* chunks = dict_item(state.ptr(), a_chunks_.ptr());
                if (PyDict_Size(state.ptr()) != 3) {
                    throw py::type_error("unexpected TensorStorageMetadata attributes");
                }
                if (!PyList_CheckExact(chunks)) throw py::type_error("expected list of chunks");
                put_get(tensor_ref_);
                put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                put_get(k_properties_);
                small(dict_item(state.ptr(), a_properties_.ptr()));
                put_get(k_size_);
                size(dict_item(state.ptr(), a_size_.ptr()));
                put_get(k_chunks_);
                put(EMPTY_LIST);
                Py_ssize_t nc = PyList_GET_SIZE(chunks);
                for (Py_ssize_t j = 0; j < nc; ++j) {
                    if (j % BATCH == 0) put(MARK);
                    PyObject* c = PyList_GET_ITEM(chunks, j);
                    if (!is(c, chunk_cls_)) {
                        small(c);
                    } else {
                        py::object chunk = instance_dict(c);
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
            }
            if (i % BATCH == BATCH - 1 || i == n - 1) put(SETITEMS);
        }
    }

    // Metadata.storage_data: MetadataIndex -> _StorageInfo.
    void dump_storage_data(PyObject* dict) {
        if (!PyDict_CheckExact(dict)) throw py::type_error("expected dict");
        put(EMPTY_DICT);
        Py_ssize_t pos = 0, n = PyDict_Size(dict);
        PyObject *idx, *info;
        for (Py_ssize_t i = 0; PyDict_Next(dict, &pos, &idx, &info); ++i) {
            if (i % BATCH == 0) put(MARK);
            if (!is(idx, index_cls_)) {
                small(idx);
            } else {
                // MetadataIndex sets offset only when it is given; pickle its __dict__.
                py::object state = instance_dict(idx);
                PyObject* fqn = dict_item(state.ptr(), a_fqn_.ptr());
                PyObject* index = dict_item(state.ptr(), a_index_.ptr());
                PyObject* offset = dict_get(state.ptr(), a_offset_.ptr());
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
            py::object state = is(info, info_cls_) ? instance_dict(info) : py::object();
            PyObject* transforms = state ? dict_get(state.ptr(), a_transform_.ptr()) : nullptr;
            if (!state || (transforms && transforms != Py_None)) {
                small(info);
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

    // Whether obj is exactly of class cls. Objects of exactly the metadata classes are encoded
    // here; others, subclasses included, are left to the stdlib pickler and keep their class.
    static bool is(PyObject* obj, const py::object& cls) {
        return reinterpret_cast<PyObject*>(Py_TYPE(obj)) == cls.ptr();
    }

    // Append raw opcode bytes.
    void put(char c) { out_.push_back(c); }
    void put(std::initializer_list<char> cs) { out_.append(cs.begin(), cs.end()); }

    // A str: SHORT_BINUNICODE, or BINUNICODE from 256 bytes on.
    void put_str(std::string_view s) {
        if (s.size() < 256) {
            put('\x8c');
            put(static_cast<char>(s.size()));
        } else {
            put('X');
            put_le(static_cast<uint32_t>(s.size()), 4);
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
            uint64_t u = static_cast<uint64_t>(i);
            auto byte = [u](int k) { return static_cast<uint8_t>(u >> (8 * k)); };
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
    uint32_t global_ref(std::string_view module, std::string_view name) {
        put_str(module);
        put_str(name);
        put({STACK_GLOBAL, MEMOIZE, POP});
        return memo_len_++;
    }

    // Push a dict key once into the memo; return its memo index.
    uint32_t key_ref(std::string_view name) {
        put_str(name);
        put({MEMOIZE, POP});
        return memo_len_++;
    }

    // First use writes the string and memoizes it; later uses fetch it from the memo. The memo owns
    // a copy of each distinct string and is looked up by view, so lookups do not allocate.
    void string(PyObject* s) {
        std::string_view v = utf8(s);
        if (auto it = strings_.find(v); it != strings_.end()) {
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
        Py_ssize_t n = PyTuple_GET_SIZE(dims);
        if (n == 0) {
            put(EMPTY_TUPLE);
        } else {
            if (n > 3) put(MARK);
            for (Py_ssize_t j = 0; j < n; ++j) put_int(as_int(PyTuple_GET_ITEM(dims, j)));
            put(n == 1 ? TUPLE1 : n == 2 ? TUPLE2 : n == 3 ? TUPLE3 : TUPLE);
        }
        put({TUPLE1, REDUCE});
    }

    // An object left to the stdlib pickler, spliced in from small_pickle's opcodes.
    void small(PyObject* obj) {
        py::object b = small_pickle_(py::handle(obj));
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
        size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };

    py::object small_pickle_;
    py::object tensor_cls_, bytes_cls_, chunk_cls_, index_cls_, info_cls_, size_cls_;
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
// the stdlib pickler.
py::bytes dumps(py::handle md, py::object small_pickle) {
    return Writer(std::move(small_pickle)).dumps(md.ptr());
}

}  // namespace

PYBIND11_MODULE(native, m) {
    m.doc() = "Fast pickling of torch.distributed.checkpoint Metadata.";
    m.def("dumps", &dumps, py::arg("metadata"), py::arg("small_pickle"),
          "Return the pickle of a torch.distributed.checkpoint Metadata, as standard pickle bytes.");
}
