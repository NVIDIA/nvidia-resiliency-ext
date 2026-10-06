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
#include <cstring>
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

std::string_view utf8(PyObject* s) {
    if (!PyUnicode_CheckExact(s)) throw py::type_error("expected str");
    Py_ssize_t n;
    const char* p = PyUnicode_AsUTF8AndSize(s, &n);
    if (!p) throw py::error_already_set();
    return {p, static_cast<size_t>(n)};
}

// The writers encode only exact types; anything else (a bool, a tuple for a torch.Size, ...) would
// unpickle as a different type, so it raises and the caller falls back to pickle.dump.
int64_t as_int(PyObject* o) {
    if (!PyLong_CheckExact(o)) throw py::type_error("expected int");
    long long v = PyLong_AsLongLong(o);
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return v;
}

class Writer {
   public:
    Writer(py::object small_pickle, size_t expected_size) : small_pickle_(std::move(small_pickle)) {
        out_.reserve(expected_size);
    }

    std::string out_;
    uint32_t size_ref = 0;
    PyObject* size_type = nullptr;  // torch.Size

    void put(char c) { out_.push_back(c); }
    void put(std::initializer_list<char> cs) { out_.append(cs.begin(), cs.end()); }

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

    void put_get(uint32_t idx) {
        if (idx < 256) {
            put('h');
            put(static_cast<char>(idx));
        } else {
            put('j');
            put_le(idx, 4);
        }
    }

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
            // pickle.encode_long: minimal little-endian two's complement.
            unsigned __int128 mag = i < 0 ? static_cast<unsigned __int128>(-static_cast<__int128>(i))
                                          : static_cast<unsigned __int128>(i);
            int bit_length = 0;
            while (mag) {
                ++bit_length;
                mag >>= 1;
            }
            int nbytes = (bit_length >> 3) + 1;
            unsigned char full[16];
            __int128 x = i;
            std::memcpy(full, &x, 16);  // little-endian
            if (i < 0 && nbytes > 1 && full[nbytes - 1] == 0xff && (full[nbytes - 2] & 0x80)) --nbytes;
            put('\x8a');
            put(static_cast<char>(nbytes));
            out_.append(reinterpret_cast<const char*>(full), nbytes);
        }
    }

    // Push module.name once into the memo; return its memo index.
    uint32_t global_ref(std::string_view module, std::string_view name) {
        put_str(module);
        put_str(name);
        put({STACK_GLOBAL, MEMOIZE, POP});
        return memo_len_++;
    }

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

    void size(PyObject* dims) {
        if (reinterpret_cast<PyObject*>(Py_TYPE(dims)) != size_type) {
            throw py::type_error("expected torch.Size");
        }
        put_get(size_ref);
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

    void small(py::handle obj) {
        py::object b = small_pickle_(obj);
        char* p;
        Py_ssize_t n;
        if (PyBytes_AsStringAndSize(b.ptr(), &p, &n) != 0) throw py::error_already_set();
        out_.append(p, n);
    }

   private:
    void put_le(uint64_t v, int nbytes) {
        for (int k = 0; k < nbytes; ++k) put(static_cast<char>((v >> (8 * k)) & 0xff));
    }

    // Hashes std::string keys and std::string_view lookups alike (C++20 heterogeneous lookup).
    struct StringHash {
        using is_transparent = void;
        size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };

    py::object small_pickle_;
    uint32_t memo_len_ = 0;
    std::unordered_map<std::string, uint32_t, StringHash, std::equal_to<>> strings_;
};

py::bytes dumps(py::handle md, py::object small_pickle) {
    py::module_ meta_mod = py::module_::import("torch.distributed.checkpoint.metadata");
    py::object tensor_cls = meta_mod.attr("TensorStorageMetadata");
    py::object bytes_cls = meta_mod.attr("BytesStorageMetadata");
    py::object chunk_cls = meta_mod.attr("ChunkStorageMetadata");
    py::object index_cls = meta_mod.attr("MetadataIndex");
    py::object info_cls = py::module_::import("torch.distributed.checkpoint.filesystem").attr("_StorageInfo");
    py::object size_cls = py::module_::import("torch").attr("Size");
    // Objects of exactly these classes are encoded here; others, subclasses included, are left to
    // the stdlib pickler, so that they unpickle as their own class.
    auto is = [](PyObject* obj, const py::object& cls) {
        return reinterpret_cast<PyObject*>(Py_TYPE(obj)) == cls.ptr();
    };

    py::str a_properties("properties"), a_size("size"), a_chunks("chunks");
    py::str a_offsets("offsets"), a_sizes("sizes"), a_fqn("fqn"), a_index("index"), a_offset("offset");
    py::str a_transform("transform_descriptors"), a_relative_path("relative_path"), a_length("length");

    py::object fields = instance_dict(md.ptr());
    // About 98 bytes per storage entry: its MetadataIndex and
    // _StorageInfo, plus the matching chunk in state_dict_metadata.
    PyObject* storage_data = PyDict_GetItemString(fields.ptr(), "storage_data");
    size_t n_storage = storage_data && PyDict_Check(storage_data) ? PyDict_Size(storage_data) : 0;
    Writer w(std::move(small_pickle), 4096 + 100 * n_storage);
    w.size_type = size_cls.ptr();
    w.put({PROTO, '\x04'});
    const char* meta = "torch.distributed.checkpoint.metadata";
    w.size_ref = w.global_ref("torch", "Size");
    uint32_t metadata_ref = w.global_ref(meta, "Metadata");
    uint32_t index_ref = w.global_ref(meta, "MetadataIndex");
    uint32_t info_ref = w.global_ref("torch.distributed.checkpoint.filesystem", "_StorageInfo");
    uint32_t chunk_ref = w.global_ref(meta, "ChunkStorageMetadata");
    uint32_t tensor_ref = w.global_ref(meta, "TensorStorageMetadata");
    uint32_t bytes_ref = w.global_ref(meta, "BytesStorageMetadata");
    uint32_t k_fqn = w.key_ref("fqn"), k_index = w.key_ref("index"), k_offset = w.key_ref("offset");
    uint32_t k_relative_path = w.key_ref("relative_path"), k_length = w.key_ref("length");
    uint32_t k_offsets = w.key_ref("offsets"), k_sizes = w.key_ref("sizes");
    uint32_t k_properties = w.key_ref("properties"), k_size = w.key_ref("size"), k_chunks = w.key_ref("chunks");

    w.put_get(metadata_ref);
    w.put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
    for (auto item : py::reinterpret_borrow<py::dict>(fields)) {
        std::string_view field = utf8(item.first.ptr());
        w.put_str(field);
        PyObject* value = item.second.ptr();
        if ((field == "state_dict_metadata" || field == "storage_data") && !PyDict_CheckExact(value)) {
            throw py::type_error("expected dict");
        }
        if (field == "state_dict_metadata") {
            w.put(EMPTY_DICT);
            Py_ssize_t pos = 0, n = PyDict_Size(value);
            PyObject *fqn, *v;
            for (Py_ssize_t i = 0; PyDict_Next(value, &pos, &fqn, &v); ++i) {
                if (i % BATCH == 0) w.put(MARK);
                w.string(fqn);
                if (is(v, bytes_cls) && PyDict_Size(instance_dict(v).ptr()) == 0) {
                    w.put_get(bytes_ref);
                    w.put({EMPTY_TUPLE, NEWOBJ});
                } else if (!is(v, tensor_cls)) {
                    w.small(v);
                } else {
                    py::object state = instance_dict(v);
                    PyObject* chunks = dict_item(state.ptr(), a_chunks.ptr());
                    if (PyDict_Size(state.ptr()) != 3) {
                        throw py::type_error("unexpected TensorStorageMetadata attributes");
                    }
                    if (!PyList_CheckExact(chunks)) throw py::type_error("expected list of chunks");
                    w.put_get(tensor_ref);
                    w.put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                    w.put_get(k_properties);
                    w.small(dict_item(state.ptr(), a_properties.ptr()));
                    w.put_get(k_size);
                    w.size(dict_item(state.ptr(), a_size.ptr()));
                    w.put_get(k_chunks);
                    w.put(EMPTY_LIST);
                    Py_ssize_t nc = PyList_GET_SIZE(chunks);
                    for (Py_ssize_t j = 0; j < nc; ++j) {
                        if (j % BATCH == 0) w.put(MARK);
                        PyObject* c = PyList_GET_ITEM(chunks, j);
                        if (!is(c, chunk_cls)) {
                            w.small(c);
                        } else {
                            py::object chunk = instance_dict(c);
                            if (PyDict_Size(chunk.ptr()) != 2) {
                                throw py::type_error("unexpected ChunkStorageMetadata attributes");
                            }
                            w.put_get(chunk_ref);
                            w.put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                            w.put_get(k_offsets);
                            w.size(dict_item(chunk.ptr(), a_offsets.ptr()));
                            w.put_get(k_sizes);
                            w.size(dict_item(chunk.ptr(), a_sizes.ptr()));
                            w.put({SETITEMS, BUILD});
                        }
                        if (j % BATCH == BATCH - 1 || j == nc - 1) w.put(APPENDS);
                    }
                    w.put({SETITEMS, BUILD});
                }
                if (i % BATCH == BATCH - 1 || i == n - 1) w.put(SETITEMS);
            }
        } else if (field == "storage_data") {
            w.put(EMPTY_DICT);
            Py_ssize_t pos = 0, n = PyDict_Size(value);
            PyObject *idx, *info;
            for (Py_ssize_t i = 0; PyDict_Next(value, &pos, &idx, &info); ++i) {
                if (i % BATCH == 0) w.put(MARK);
                if (!is(idx, index_cls)) {
                    w.small(idx);
                } else {
                    // MetadataIndex sets offset only when it is given; pickle its __dict__.
                    py::object state = instance_dict(idx);
                    PyObject* fqn = dict_item(state.ptr(), a_fqn.ptr());
                    PyObject* index = dict_item(state.ptr(), a_index.ptr());
                    PyObject* offset = dict_get(state.ptr(), a_offset.ptr());
                    if (PyDict_Size(state.ptr()) != (offset ? 3 : 2)) {
                        throw py::type_error("unexpected MetadataIndex attributes");
                    }
                    w.put_get(index_ref);
                    w.put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                    w.put_get(k_fqn);
                    w.string(fqn);
                    w.put_get(k_index);
                    if (index == Py_None) w.put(NONE); else w.put_int(as_int(index));
                    if (offset) {
                        w.put_get(k_offset);
                        if (offset == Py_None) w.put(NONE); else w.size(offset);
                    }
                    w.put({SETITEMS, BUILD});
                }

                // _StorageInfo pickles its __dict__ without None values. transform_descriptors
                // exists from PyTorch 2.8; a _StorageInfo with it set is left to the stdlib pickler.
                py::object state = is(info, info_cls) ? instance_dict(info) : py::object();
                PyObject* transforms = state ? dict_get(state.ptr(), a_transform.ptr()) : nullptr;
                if (!state || (transforms && transforms != Py_None)) {
                    w.small(info);
                } else {
                    if (PyDict_Size(state.ptr()) != (transforms ? 4 : 3)) {
                        throw py::type_error("unexpected _StorageInfo attributes");
                    }
                    w.put_get(info_ref);
                    w.put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                    w.put_get(k_relative_path);
                    w.string(dict_item(state.ptr(), a_relative_path.ptr()));
                    w.put_get(k_offset);
                    w.put_int(as_int(dict_item(state.ptr(), a_offset.ptr())));
                    w.put_get(k_length);
                    w.put_int(as_int(dict_item(state.ptr(), a_length.ptr())));
                    w.put({SETITEMS, BUILD});
                }
                if (i % BATCH == BATCH - 1 || i == n - 1) w.put(SETITEMS);
            }
        } else {
            w.small(value);
        }
    }
    w.put({SETITEMS, BUILD, STOP});
    return py::bytes(w.out_);
}

}  // namespace

PYBIND11_MODULE(native, m) {
    m.doc() = "Fast pickling of torch.distributed.checkpoint Metadata.";
    m.def("dumps", &dumps, py::arg("metadata"), py::arg("small_pickle"),
          "Return the pickle of a torch.distributed.checkpoint Metadata, as standard pickle bytes.");
}
