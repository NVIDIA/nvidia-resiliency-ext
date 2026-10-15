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

// Native versions of `state_table.from_metadata` and `pickler._MetadataPickler`: converts a torch DCP
// `Metadata` into a `state_table.StateDictTable`, and writes the standard pickle opcodes for the
// Metadata from that table directly. Both give exactly what the Python versions give. The few small
// objects (TensorProperties, StorageMeta, ...) are encoded by the Python `small_pickle` callback.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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
constexpr int64_t TABLE_MAGIC = 0x314C42545852564E;  // the bytes "NVRXTBL1", read little-endian
constexpr int64_t TABLE_VERSION = 1;
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

    // Replace out's contents with the strings, as views into the row. Their bytes are not checked
    // to be UTF-8: tables come from this job's ranks, which encoded them (see table.py).
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

// New reference to obj.__dict__, the state pickle writes for the metadata classes. Only objects of
// exactly those classes get here, whose __dict__ is the generic one, so it is fetched directly
// rather than looked up as an attribute.
[[nodiscard]] py::object instance_dict(PyObject* obj) {
    PyObject* const r = PyObject_GenericGetDict(obj, nullptr);
    if (!r) {
        throw py::error_already_set();
    }
    const auto d = py::reinterpret_steal<py::object>(r);
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

// Kinds of state_dict_metadata entries, numbered as in state_table.py.
enum class Kind : uint8_t { Bytes = 0, Tensor = 1, Other = 2 };
constexpr std::string_view STATE_DICT_METADATA = "state_dict_metadata", STORAGE_DATA = "storage_data";

// Reject a malformed state table, before anything reads outside its arrays.
[[noreturn]] void bad_state_table(std::string_view what) {
    throw py::value_error("state table: " + std::string(what));
}

template <typename T>
using Array = py::array_t<T, py::array::c_style>;

// A tuple attribute of t, of exactly the type tuple: a StateDictTable's sequences are tuples.
[[nodiscard]] py::tuple tuple_attr(py::handle t, const char* name) {
    py::object v = t.attr(name);
    if (!PyTuple_CheckExact(v.ptr())) {
        bad_state_table(std::string(name) + " is not a tuple");
    }
    return py::reinterpret_steal<py::tuple>(v.release());
}

// A 1-D, C-contiguous array attribute of t, of exactly the element type T.
template <typename T>
[[nodiscard]] Array<T> array_attr(py::handle t, const char* name) {
    py::object v = t.attr(name);
    if (!Array<T>::check_(v) || py::reinterpret_borrow<py::array>(v).ndim() != 1) {
        bad_state_table(std::string(name) + " is not a 1-D array of the expected type");
    }
    return py::reinterpret_steal<Array<T>>(v.release());
}

// The values of a 1-D array, valid while the array is held.
template <typename T>
[[nodiscard]] std::span<const T> values(const Array<T>& a) {
    return {a.data(), static_cast<size_t>(a.size())};
}

// The bytes of a bytes object, valid while it is alive.
[[nodiscard]] std::string_view bytes_view(PyObject* b) {
    return {PyBytes_AS_STRING(b), static_cast<size_t>(PyBytes_GET_SIZE(b))};
}

// A state_table.StateDictTable, checked as state_table.check checks it, so that the writer can
// read its arrays without further checks.
class TableView {
   public:
    // A field of the Metadata: its name, and its opcodes (None for the fields written from the
    // table or the write results).
    struct Field {
        std::string name;
        py::object opcodes;
    };

    explicit TableView(py::handle t)
        : fqns_(tuple_attr(t, "fqns")),
          properties_(tuple_attr(t, "properties")),
          others_(tuple_attr(t, "others")),
          kind_array_(array_attr<uint8_t>(t, "kind")),
          ref_array_(array_attr<int64_t>(t, "ref")),
          ndim_array_(array_attr<int64_t>(t, "ndim")),
          chunk_start_array_(array_attr<int64_t>(t, "chunk_start")),
          dims_start_array_(array_attr<int64_t>(t, "dims_start")),
          dims_array_(array_attr<int64_t>(t, "dims")),
          kind_(values(kind_array_)),
          ref_(values(ref_array_)),
          ndim_(values(ndim_array_)),
          chunk_start_(values(chunk_start_array_)),
          dims_start_(values(dims_start_array_)),
          dims_(values(dims_array_)) {
        const py::tuple fields = tuple_attr(t, "fields");
        for (const py::handle f : fields) {
            if (!PyTuple_CheckExact(f.ptr()) || PyTuple_GET_SIZE(f.ptr()) != 2) {
                bad_state_table("a field is not a (name, opcodes) tuple");
            }
            PyObject* const name = PyTuple_GET_ITEM(f.ptr(), 0);
            PyObject* const opcodes = PyTuple_GET_ITEM(f.ptr(), 1);
            const std::string_view n = utf8(name);
            const bool written_here = n == STATE_DICT_METADATA || n == STORAGE_DATA;
            if (written_here ? opcodes != Py_None : !PyBytes_CheckExact(opcodes)) {
                bad_state_table("unexpected opcodes of field " + std::string(n));
            }
            fields_.push_back({std::string(n), py::reinterpret_borrow<py::object>(opcodes)});
        }
        check();
    }

    [[nodiscard]] size_t entries() const { return kind_.size(); }
    [[nodiscard]] const std::vector<Field>& fields() const { return fields_; }
    [[nodiscard]] PyObject* fqn(size_t i) const { return PyTuple_GET_ITEM(fqns_.ptr(), i); }
    [[nodiscard]] Kind kind(size_t i) const { return static_cast<Kind>(kind_[i]); }
    [[nodiscard]] std::string_view properties(size_t i) const {
        return bytes_view(PyTuple_GET_ITEM(properties_.ptr(), ref_[i]));
    }
    [[nodiscard]] std::string_view other(size_t i) const {
        return bytes_view(PyTuple_GET_ITEM(others_.ptr(), ref_[i]));
    }
    [[nodiscard]] size_t ndim(size_t i) const { return static_cast<size_t>(ndim_[i]); }
    [[nodiscard]] std::span<const int64_t> size(size_t i) const {
        return dims_.subspan(static_cast<size_t>(dims_start_[i]), ndim(i));
    }
    [[nodiscard]] size_t chunks(size_t i) const {
        return static_cast<size_t>(chunk_start_[i + 1] - chunk_start_[i]);
    }
    // Chunk k of entry i: its offsets, then its sizes, after the entry's size.
    [[nodiscard]] std::span<const int64_t> chunk(size_t i, size_t k) const {
        const size_t width = 2 * ndim(i);
        return dims_.subspan(static_cast<size_t>(dims_start_[i]) + ndim(i) + width * k, width);
    }
    [[nodiscard]] size_t dims_size() const { return dims_.size(); }

   private:
    // Check that the arrays describe entries() entries consistently; see state_table.check.
    void check() const {
        const size_t n = entries();
        if (static_cast<size_t>(PyTuple_GET_SIZE(fqns_.ptr())) != n || ref_.size() != n ||
            ndim_.size() != n) {
            bad_state_table("arrays of different lengths");
        }
        const auto check_starts = [n](std::span<const int64_t> starts,
                                      std::optional<size_t> pool) {
            if (starts.size() != n + 1 || starts[0] != 0 ||
                !std::ranges::is_sorted(starts) ||
                (pool && static_cast<uint64_t>(starts[n]) != *pool)) {
                bad_state_table("start offsets out of order or range");
            }
        };
        check_starts(chunk_start_, std::nullopt);
        check_starts(dims_start_, dims_.size());
        const auto in = [](int64_t ref, const py::tuple& pickles) {
            return ref >= 0 && ref < PyTuple_GET_SIZE(pickles.ptr()) &&
                   PyBytes_CheckExact(PyTuple_GET_ITEM(pickles.ptr(), ref));
        };
        for (size_t i = 0; i < n; ++i) {
            const int64_t ndim = ndim_[i];
            const int64_t nc = chunk_start_[i + 1] - chunk_start_[i];
            const int64_t values = dims_start_[i + 1] - dims_start_[i];
            if (ndim < 0) {
                bad_state_table("a negative number of dims");
            }
            switch (kind_[i]) {
                case static_cast<uint8_t>(Kind::Tensor):
                    if (!in(ref_[i], properties_) || !holds_chunks(values, ndim, nc)) {
                        bad_state_table("a tensor entry's dims don't match its chunks");
                    }
                    break;
                case static_cast<uint8_t>(Kind::Other):
                    if (!in(ref_[i], others_)) {
                        bad_state_table("an entry refers to no pickle");
                    }
                    [[fallthrough]];
                case static_cast<uint8_t>(Kind::Bytes):
                    if (ndim != 0 || nc != 0 || values != 0) {
                        bad_state_table("a non-tensor entry has dims");
                    }
                    break;
                default:
                    bad_state_table("unknown entry kind");
            }
        }
    }

    // Whether values = ndim * (1 + 2 * nc): a tensor's size and nc chunks of offsets and sizes.
    // All three are non-negative; divides first, so nothing can overflow.
    [[nodiscard]] static bool holds_chunks(int64_t values, int64_t ndim, int64_t nc) {
        if (ndim == 0) {
            return values == 0;
        }
        const int64_t per_dim = values / ndim;  // 1 + 2 * nc
        return values % ndim == 0 && per_dim % 2 == 1 && per_dim / 2 == nc;
    }

    py::tuple fqns_, properties_, others_;
    Array<uint8_t> kind_array_;
    Array<int64_t> ref_array_, ndim_array_, chunk_start_array_, dims_start_array_, dims_array_;
    std::span<const uint8_t> kind_;
    std::span<const int64_t> ref_, ndim_, chunk_start_, dims_start_, dims_;
    std::vector<Field> fields_;
};

// A numpy array that owns v's values, without copying them.
template <typename T>
[[nodiscard]] py::array_t<T> to_array(std::vector<T>&& v) {
    auto owned = std::make_unique<std::vector<T>>(std::move(v));
    const py::capsule owner(owned.get(),
                            [](void* p) { delete static_cast<std::vector<T>*>(p); });
    const std::vector<T>& values = *owned.release();
    return py::array_t<T>({values.size()}, {sizeof(T)}, values.data(), owner);
}

// Converts a Metadata into a state_table.StateDictTable, exactly as state_table.from_metadata does.
class TableBuilder {
   public:
    // Look up the metadata classes the conversion checks objects against.
    explicit TableBuilder(py::object small_pickle) : small_pickle_(std::move(small_pickle)) {
        py::module_ meta = py::module_::import("torch.distributed.checkpoint.metadata");
        metadata_cls_ = meta.attr("Metadata");
        tensor_cls_ = meta.attr("TensorStorageMetadata");
        bytes_cls_ = meta.attr("BytesStorageMetadata");
        chunk_cls_ = meta.attr("ChunkStorageMetadata");
        size_cls_ = py::module_::import("torch").attr("Size");
    }

    // The table of md, an instance of table_cls (state_table.StateDictTable).
    [[nodiscard]] py::object build(py::handle md, const py::object& table_cls) {
        if (!is(md.ptr(), metadata_cls_)) {
            throw py::type_error("expected Metadata, got " + type_name(md.ptr()));
        }
        const py::object state = instance_dict(md.ptr());
        const py::object entries = getattr(md.ptr(), a_state_dict_metadata_.ptr());
        if (!PyDict_CheckExact(entries.ptr())) {
            throw py::type_error("expected dict for state_dict_metadata, got " +
                                 type_name(entries.ptr()));
        }
        // Snapshots, which hold the items while small_pickle runs Python code that could change
        // the dicts.
        py::list fields;
        for (const py::handle item : items(state.ptr())) {
            PyObject* const name = PyTuple_GET_ITEM(item.ptr(), 0);
            const std::string_view n = utf8(name);
            const py::object opcodes = n == STATE_DICT_METADATA || n == STORAGE_DATA
                                           ? py::object(py::none())
                                           : py::object(pickled(PyTuple_GET_ITEM(item.ptr(), 1)));
            fields.append(py::make_tuple(py::handle(name), opcodes));
        }
        const py::list entry_items = items(entries.ptr());
        const size_t n = entry_items.size();
        py::list fqns(n);
        py::dict index;
        kind_.reserve(n), ref_.reserve(n), ndim_.reserve(n);
        for (auto* starts : {&chunk_start_, &dims_start_}) {
            starts->reserve(n + 1);
            starts->push_back(0);
        }
        for (size_t i = 0; i < n; ++i) {
            PyObject* const fqn = PyTuple_GET_ITEM(entry_items[i].ptr(), 0);
            PyObject* const v = PyTuple_GET_ITEM(entry_items[i].ptr(), 1);
            if (!PyUnicode_CheckExact(fqn)) {
                throw py::type_error("expected str fqn, got " + type_name(fqn));
            }
            fqns[i] = py::handle(fqn);
            index[py::handle(fqn)] = py::int_(i);
            size_t ndim = 0, chunks = 0;
            if (const std::optional<Tensor> t = is(v, tensor_cls_) ? tensor(v) : std::nullopt) {
                // After the chunks are counted: small_pickle runs Python code.
                py::bytes opcodes = pickled(t->properties.ptr());
                const auto [it, added] = property_ids_.try_emplace(
                    std::string(bytes_view(opcodes.ptr())), properties_.size());
                if (added) {
                    properties_.append(opcodes);
                }
                push(Kind::Tensor, static_cast<int64_t>(it->second));
                ndim = t->ndim;
                chunks = t->chunks;
            } else if (is(v, bytes_cls_) && PyDict_Size(instance_dict(v).ptr()) == 0) {
                push(Kind::Bytes, 0);
            } else {
                push(Kind::Other, static_cast<int64_t>(others_.size()));
                others_.append(pickled(v));
            }
            ndim_.push_back(static_cast<int64_t>(ndim));
            chunk_start_.push_back(chunk_start_.back() + static_cast<int64_t>(chunks));
            dims_start_.push_back(static_cast<int64_t>(dims_.size()));
        }
        if (static_cast<size_t>(PyDict_Size(entries.ptr())) != n) {
            throw std::runtime_error("state_dict_metadata changed size while being converted");
        }
        return table_cls(py::arg("fields") = fields, py::arg("fqns") = fqns,
                         py::arg("kind") = to_array(std::move(kind_)),
                         py::arg("ref") = to_array(std::move(ref_)),
                         py::arg("ndim") = to_array(std::move(ndim_)),
                         py::arg("chunk_start") = to_array(std::move(chunk_start_)),
                         py::arg("dims_start") = to_array(std::move(dims_start_)),
                         py::arg("dims") = to_array(std::move(dims_)),
                         py::arg("properties") = properties_, py::arg("others") = others_,
                         py::arg("index") = index);
    }

   private:
    // A TENSOR entry's properties and number of chunks.
    struct Tensor {
        py::object properties;
        size_t ndim;
        size_t chunks;
    };

    // Append a TENSOR entry's size and chunks to dims_ and return its properties and numbers of
    // dims and chunks; nullopt, with nothing appended, if it isn't one (see state_table._tensor).
    [[nodiscard]] std::optional<Tensor> tensor(PyObject* v) {
        const py::object state = instance_dict(v);
        PyObject* const properties = dict_get(state.ptr(), a_properties_.ptr());
        PyObject* const size = dict_get(state.ptr(), a_size_.ptr());
        PyObject* const chunks = dict_get(state.ptr(), a_chunks_.ptr());
        if (PyDict_Size(state.ptr()) != 3 || !properties || !size || !is(size, size_cls_) ||
            !chunks || !PyList_CheckExact(chunks)) {
            return std::nullopt;
        }
        const Py_ssize_t ndim = PyTuple_GET_SIZE(size);
        const size_t mark = dims_.size();
        append_dims(size, dims_);
        const Py_ssize_t nc = PyList_GET_SIZE(chunks);
        for (Py_ssize_t k = 0; k < nc; ++k) {
            PyObject* const c = PyList_GET_ITEM(chunks, k);
            const py::object cstate = is(c, chunk_cls_) ? instance_dict(c) : py::object();
            PyObject* const offsets = cstate ? dict_get(cstate.ptr(), a_offsets_.ptr()) : nullptr;
            PyObject* const sizes = cstate ? dict_get(cstate.ptr(), a_sizes_.ptr()) : nullptr;
            if (!cstate || PyDict_Size(cstate.ptr()) != 2 || !offsets || !sizes ||
                !is(offsets, size_cls_) || !is(sizes, size_cls_) ||
                PyTuple_GET_SIZE(offsets) != ndim || PyTuple_GET_SIZE(sizes) != ndim) {
                dims_.resize(mark);
                return std::nullopt;
            }
            append_dims(offsets, dims_);
            append_dims(sizes, dims_);
        }
        return Tensor{py::reinterpret_borrow<py::object>(properties), static_cast<size_t>(ndim),
                      static_cast<size_t>(nc)};
    }

    // Append a torch.Size's dims; a torch.Size holds only ints within int64.
    static void append_dims(PyObject* dims, std::vector<int64_t>& out) {
        const Py_ssize_t n = PyTuple_GET_SIZE(dims);
        for (Py_ssize_t k = 0; k < n; ++k) out.push_back(as_int(PyTuple_GET_ITEM(dims, k)));
    }

    void push(Kind kind, int64_t ref) {
        kind_.push_back(static_cast<uint8_t>(kind));
        ref_.push_back(ref);
    }

    // The stdlib pickler's opcodes for obj.
    [[nodiscard]] py::bytes pickled(PyObject* obj) const {
        py::object b = small_pickle_(py::handle(obj));
        if (!PyBytes_CheckExact(b.ptr())) {
            throw py::type_error("small_pickle returned no bytes");
        }
        return py::reinterpret_steal<py::bytes>(b.release());
    }

    // A snapshot of a dict's items, as a list of (key, value) tuples.
    [[nodiscard]] static py::list items(PyObject* dict) {
        PyObject* const r = PyDict_Items(dict);
        if (!r) {
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::list>(r);
    }

    [[nodiscard]] static bool is(PyObject* obj, const py::object& cls) {
        return reinterpret_cast<PyObject*>(Py_TYPE(obj)) == cls.ptr();
    }

    [[nodiscard]] static std::string type_name(PyObject* obj) { return Py_TYPE(obj)->tp_name; }

    py::object small_pickle_;
    py::object metadata_cls_, tensor_cls_, bytes_cls_, chunk_cls_, size_cls_;
    py::str a_state_dict_metadata_{"state_dict_metadata"}, a_properties_{"properties"};
    py::str a_size_{"size"}, a_chunks_{"chunks"}, a_offsets_{"offsets"}, a_sizes_{"sizes"};
    std::vector<uint8_t> kind_;
    std::vector<int64_t> ref_, ndim_, chunk_start_, dims_start_, dims_;
    py::list properties_, others_;
    std::unordered_map<std::string, size_t> property_ids_;  // by their opcodes
};

// Writes the pickle of one Metadata into out_; use once per dump.
class Writer {
   public:
    // Look up the classes the storage_data encoding checks objects against.
    explicit Writer(py::object small_pickle) : small_pickle_(std::move(small_pickle)) {
        py::module_ meta = py::module_::import("torch.distributed.checkpoint.metadata");
        index_cls_ = meta.attr("MetadataIndex");
        info_cls_ =
            py::module_::import("torch.distributed.checkpoint.filesystem").attr("_StorageInfo");
        size_cls_ = py::module_::import("torch").attr("Size");
    }

    // The pickle of the Metadata t was converted from: a NEWOBJ of Metadata built from its fields,
    // in order. storage_data is written from rows (the gathered write-result tables) if given,
    // else from the storage_data dict, else as None.
    [[nodiscard]] py::bytes dumps(const TableView& t, py::handle storage_data,
                                  const py::object& rows) {
        // Holds the rows the tables point into.
        const Rows rows_array = rows.is_none() ? Rows() : rows.cast<Rows>();
        const std::vector<WriteResultTable> tables =
            rows.is_none() ? std::vector<WriteResultTable>() : parse_tables(rows_array);
        size_t n_storage = 0;
        if (!rows.is_none()) {
            for (const WriteResultTable& w : tables) {
                n_storage += w.entries();
            }
        } else if (PyDict_Check(storage_data.ptr())) {
            n_storage = PyDict_Size(storage_data.ptr());
        }
        // About 98 bytes per storage entry (its MetadataIndex and _StorageInfo) and about 8 bytes
        // per chunk dim in state_dict_metadata.
        out_.reserve(4096 + 100 * n_storage + 8 * t.dims_size());

        prelude();
        put_get(metadata_ref_);
        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
        for (const TableView::Field& f : t.fields()) {
            put_str(f.name);
            if (f.name == STATE_DICT_METADATA) {
                dump_state_dict_metadata(t);
            } else if (f.name != STORAGE_DATA) {
                out_.append(bytes_view(f.opcodes.ptr()));
            } else if (!rows.is_none()) {
                dump_storage_tables(tables);
            } else if (!storage_data.is_none()) {
                dump_storage_data(storage_data.ptr());
            } else {
                put(NONE);
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

    // Metadata.state_dict_metadata, from the table's entries in order.
    void dump_state_dict_metadata(const TableView& t) {
        put(EMPTY_DICT);
        const size_t n = t.entries();
        for (size_t i = 0; i < n; ++i) {
            begin_item(i);
            string(t.fqn(i));
            switch (t.kind(i)) {
                case Kind::Bytes:
                    put_get(bytes_ref_);
                    put({EMPTY_TUPLE, NEWOBJ});
                    break;
                case Kind::Other:
                    out_.append(t.other(i));
                    break;
                case Kind::Tensor: {
                    put_get(tensor_ref_);
                    put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                    put_get(k_properties_);
                    out_.append(t.properties(i));
                    put_get(k_size_);
                    size(t.size(i));
                    put_get(k_chunks_);
                    put(EMPTY_LIST);
                    const size_t nc = t.chunks(i), ndim = t.ndim(i);
                    for (size_t k = 0; k < nc; ++k) {
                        begin_item(k);
                        const std::span<const int64_t> c = t.chunk(i, k);
                        put_get(chunk_ref_);
                        put({EMPTY_TUPLE, NEWOBJ, EMPTY_DICT, MARK});
                        put_get(k_offsets_);
                        size(c.first(ndim));
                        put_get(k_sizes_);
                        size(c.last(ndim));
                        put({SETITEMS, BUILD});
                        end_item(k, nc, APPENDS);
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
        if (flags & ~(FLAG_INDEX | FLAG_OFFSET | FLAG_OFFSET_SIZE)) {
            bad_table("unknown flags");
        }
        if ((flags & FLAG_OFFSET_SIZE) && !(flags & FLAG_OFFSET)) {
            bad_table("an offset size without an offset");
        }
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

    // The same, for dims from a state table.
    void size(std::span<const int64_t> dims) {
        const size_t n = dims.size();
        put_get(size_ref_);
        if (n == 0) {
            put(EMPTY_TUPLE);
        } else {
            if (n > 3) {
                put(MARK);
            }
            for (const int64_t d : dims) put_int(d);
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
    py::object index_cls_, info_cls_, size_cls_;
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

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Fast conversion and pickling of torch.distributed.checkpoint Metadata.";
    m.def(
        "from_metadata",
        [](py::handle md, py::object small_pickle, const py::object& table_cls) {
            return TableBuilder(std::move(small_pickle)).build(md, table_cls);
        },
        py::arg("metadata"), py::arg("small_pickle"), py::arg("table_cls"),
        "Return the metadata without storage_data as a table_cls (state_table.StateDictTable).");
    m.def(
        "dumps_table",
        [](py::handle table, py::object small_pickle, py::handle storage_data,
           const py::object& storage_rows) {
            return Writer(std::move(small_pickle)).dumps(TableView(table), storage_data, storage_rows);
        },
        py::arg("table"), py::arg("small_pickle"), py::arg("storage_data") = py::none(),
        py::arg("storage_rows") = py::none(),
        "Return the pickle of the Metadata a state table was converted from, as standard pickle "
        "bytes.");
}
