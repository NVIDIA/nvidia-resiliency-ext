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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr char PROTO = '\x80', STOP = '.', MARK = '(', POP = '0';
constexpr char EMPTY_DICT = '}', EMPTY_LIST = ']', EMPTY_TUPLE = ')';
constexpr char SETITEMS = 'u', APPENDS = 'e', BUILD = 'b', NEWOBJ = '\x81', REDUCE = 'R';
constexpr char STACK_GLOBAL = '\x93', MEMOIZE = '\x94', NONE = 'N';
constexpr char TUPLE = 't', TUPLE1 = '\x85', TUPLE2 = '\x86', TUPLE3 = '\x87';
// The opcodes that build a tuple of 1, 2 or 3 items without a MARK.
constexpr std::array<char, 3> SMALL_TUPLE = {TUPLE1, TUPLE2, TUPLE3};
constexpr size_t BATCH = 1000;  // same batching as the stdlib pickler

// Write-result tables, as write_result_table.py encodes them: little-endian int64s, header first.
static_assert(std::endian::native == std::endian::little, "write-result tables are little-endian");
constexpr int64_t TABLE_MAGIC = 0x314C42545852564E;  // the bytes "NVRXTBL1", read little-endian
constexpr int64_t TABLE_VERSION = 2;
enum class Header : uint8_t {
    Magic,
    Version,
    Entries,
    Strings,
    StringBytes,
    Offsets,
    FqnMode,
    Fingerprint
};
// How a table names fqns: by string id, or by entry in the coordinator's fqn list.
enum class FqnMode : uint8_t { Strings = 0, Indices = 1 };
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
    explicit Int64s(std::span<std::byte const> bytes) : bytes_(bytes) {}

    // The number of int64s.
    [[nodiscard]] auto size() const -> size_t { return bytes_.size() / sizeof(int64_t); }

    // The i-th int64, i < size(). bit_cast needs an object to read, and there is none at an
    // arbitrary offset of the row, so the bytes are copied into one; this compiles to one load.
    [[nodiscard]] auto operator[](size_t i) const -> int64_t {
        std::array<std::byte, sizeof(int64_t)> v{};
        std::ranges::copy(bytes_.subspan(sizeof(int64_t) * i).first<sizeof(int64_t)>(), v.begin());
        return std::bit_cast<int64_t>(v);
    }

    // The field of a record (a header or an entry) that an enum names.
    template <typename Field>
        requires std::is_enum_v<Field>
    [[nodiscard]] auto operator[](Field field) const -> int64_t {
        return (*this)[static_cast<size_t>(field)];
    }

    // The count int64s from pos, pos + count <= size().
    [[nodiscard]] auto subspan(size_t pos, size_t count) const -> Int64s {
        return Int64s(bytes_.subspan(sizeof(int64_t) * pos, sizeof(int64_t) * count));
    }

   private:
    std::span<std::byte const> bytes_;
};

// A write-result table within its row of the gathered buffer. parse checks that its sections lie
// within the row; the accessors check what its entries refer to.
class WriteResultTable {
   public:
    // The table in a row.
    [[nodiscard]] static auto parse(std::span<std::byte const> row) -> WriteResultTable {
        if (row.size() < sizeof(int64_t) * HEADER) {
            bad_table("truncated");
        }
        Int64s const header(row.first(sizeof(int64_t) * HEADER));
        if (header[Header::Magic] != TABLE_MAGIC || header[Header::Version] != TABLE_VERSION) {
            bad_table("not a table of this version");
        }
        std::span<std::byte const> rest = row.subspan(sizeof(int64_t) * HEADER);
        // The next section: as many items of item_size bytes as the header's count field says,
        // checked to fit in the rest of the row (comparing counts, so sizes cannot overflow).
        auto const take = [&](Header count, size_t item_size) {
            int64_t const n = header[count];
            if (n < 0 || static_cast<uint64_t>(n) > rest.size() / item_size) {
                bad_table("truncated");
            }
            auto const section = rest.first(static_cast<size_t>(n) * item_size);
            rest = rest.subspan(section.size());
            return section;
        };
        Int64s const string_len(take(Header::Strings, sizeof(int64_t)));
        Int64s const entries(take(Header::Entries, sizeof(int64_t) * COLUMNS));
        Int64s const offsets(take(Header::Offsets, sizeof(int64_t)));
        std::span<std::byte const> const blob = take(Header::StringBytes, 1);
        int64_t const fqn_mode = header[Header::FqnMode];
        if (fqn_mode != static_cast<int64_t>(FqnMode::Strings) &&
            fqn_mode != static_cast<int64_t>(FqnMode::Indices)) {
            bad_table("unknown fqn mode");
        }
        return {string_len,
                entries,
                offsets,
                // char may alias any type.
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                {reinterpret_cast<char const*>(blob.data()), blob.size()},
                static_cast<FqnMode>(fqn_mode),
                header[Header::Fingerprint]};
    }

    // The number of entries.
    [[nodiscard]] auto entries() const -> size_t { return entries_.size() / COLUMNS; }

    // Entry r < entries(): its COLUMNS int64s, indexed by Column.
    [[nodiscard]] auto entry(size_t r) const -> Int64s {
        return entries_.subspan(COLUMNS * r, COLUMNS);
    }

    // Replace out's contents with the strings, as views into the row. Their bytes are not checked
    // to be UTF-8: tables come from this job's ranks, which encoded them (see
    // write_result_table.py).
    void split_strings(std::vector<std::string_view>& out) const {
        out.clear();
        std::string_view rest = blob_;
        for (size_t k = 0; k < string_len_.size(); ++k) {
            int64_t const len = string_len_[k];
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
    [[nodiscard]] auto offset_dims(size_t pos, int64_t ndim) const -> Int64s {
        if (ndim < 0 || pos > offsets_.size() ||
            static_cast<uint64_t>(ndim) > offsets_.size() - pos) {
            bad_table("offsets out of range");
        }
        return offsets_.subspan(pos, static_cast<size_t>(ndim));
    }

    // Return how the table names fqns.
    [[nodiscard]] auto fqn_mode() const -> FqnMode { return fqn_mode_; }

    // Return the fingerprint of the fqn list an index table was encoded against.
    [[nodiscard]] auto fingerprint() const -> int64_t { return fingerprint_; }

    // Return the number of strings.
    [[nodiscard]] auto strings() const -> size_t { return string_len_.size(); }

    // Check that the entries' offsets, used up to position pos, are all of them.
    void check_offsets_used(size_t pos) const {
        if (pos != offsets_.size()) {
            bad_table("offset dims don't match the offsets");
        }
    }

   private:
    // The sections parse found.
    WriteResultTable(Int64s string_len,
                     Int64s entries,
                     Int64s offsets,
                     std::string_view blob,
                     FqnMode fqn_mode,
                     int64_t fingerprint)
        : string_len_(string_len),
          entries_(entries),
          offsets_(offsets),
          blob_(blob),
          fqn_mode_(fqn_mode),
          fingerprint_(fingerprint) {}

    Int64s string_len_;      // the byte length of each string
    Int64s entries_;         // entries() rows of COLUMNS
    Int64s offsets_;         // the entries' offset dims, concatenated
    std::string_view blob_;  // the strings, concatenated
    FqnMode fqn_mode_;
    int64_t fingerprint_;
};

// The tables in the rows of a 2-D uint8 array.
[[nodiscard]] auto parse_tables(Rows const& rows) -> std::vector<WriteResultTable> {
    if (rows.ndim() != 2) {
        throw py::value_error("expected a 2-D array of write-result tables");
    }
    auto const width = static_cast<size_t>(rows.shape(1));
    std::span<std::byte const> const bytes =
        std::as_bytes(std::span(rows.data(), static_cast<size_t>(rows.size())));
    std::vector<WriteResultTable> tables;
    tables.reserve(rows.shape(0));
    for (size_t r = 0; std::cmp_less(r, rows.shape(0)); ++r) {
        tables.push_back(WriteResultTable::parse(bytes.subspan(width * r, width)));
    }
    return tables;
}

// New reference to obj.name, throwing on error.
[[nodiscard]] auto getattr(PyObject* obj, PyObject* name) -> py::object {
    PyObject* const r = PyObject_GetAttr(obj, name);
    if (r == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(r);
}

// Return a new reference to obj.__dict__, the state pickle writes for the metadata classes. Only
// objects of exactly those classes get here, whose __dict__ is the generic one, so it is fetched
// directly rather than looked up as an attribute.
[[nodiscard]] auto instance_dict(PyObject* obj) -> py::object {
    PyObject* const r = PyObject_GenericGetDict(obj, nullptr);
    if (r == nullptr) {
        throw py::error_already_set();
    }
    auto const d = py::reinterpret_steal<py::object>(r);
    if (!PyDict_CheckExact(d.ptr())) {
        throw py::type_error("expected a __dict__");
    }
    return d;
}

// Borrowed reference to dict[key], or nullptr if key is missing.
[[nodiscard]] auto dict_get(PyObject* dict, PyObject* key) -> PyObject* {
    PyObject* const r = PyDict_GetItemWithError(dict, key);
    if ((r == nullptr) && (PyErr_Occurred() != nullptr)) {
        throw py::error_already_set();
    }
    return r;
}

// Borrowed reference to dict[key], throwing if key is missing.
[[nodiscard]] auto dict_item(PyObject* dict, PyObject* key) -> PyObject* {
    PyObject* const r = dict_get(dict, key);
    if (r == nullptr) {
        throw py::type_error("missing attribute");
    }
    return r;
}

// Return the name of obj's type, for error messages.
[[nodiscard]] auto type_name(PyObject* obj) -> std::string {
    return Py_TYPE(obj)->tp_name;
}

// UTF-8 view of a str's characters, valid while s is alive.
[[nodiscard]] auto utf8(PyObject* s) -> std::string_view {
    if (!PyUnicode_CheckExact(s)) {
        throw py::type_error("expected str");
    }
    Py_ssize_t n = 0;
    char const* const p = PyUnicode_AsUTF8AndSize(s, &n);
    if (p == nullptr) {
        throw py::error_already_set();
    }
    return {p, static_cast<size_t>(n)};
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

// Kinds of state_dict_metadata entries, numbered as in state_dict_table.py.
enum class Kind : uint8_t { Bytes = 0, Tensor = 1, Other = 2 };
constexpr std::string_view STATE_DICT_METADATA = "state_dict_metadata";
constexpr std::string_view STORAGE_DATA = "storage_data";

// Reject a malformed state dict table, before anything reads outside its arrays.
[[noreturn]] void bad_state_dict_table(std::string_view what) {
    throw py::value_error("state dict table: " + std::string(what));
}

// The element types of a state dict table's arrays.
template <typename T>
concept TableValue = std::same_as<T, uint8_t> || std::same_as<T, int64_t>;

template <TableValue T>
using Array = py::array_t<T, py::array::c_style>;

// Return the attribute name of table, which must be exactly a tuple, as a StateDictTable's
// sequences are.
[[nodiscard]] auto tuple_attr(py::handle table, char const* name) -> py::tuple {
    py::object value = table.attr(name);
    if (!PyTuple_CheckExact(value.ptr())) {
        bad_state_dict_table(std::string(name) + " is not a tuple");
    }
    return py::reinterpret_steal<py::tuple>(value.release());
}

// Return the attribute name of table, which must be a 1-D, C-contiguous, aligned array of exactly
// the element type T.
template <TableValue T>
[[nodiscard]] auto array_attr(py::handle table, char const* name) -> Array<T> {
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
[[nodiscard]] auto values(Array<T> const& a) -> std::span<T const> {
    return {a.data(), static_cast<size_t>(a.size())};
}

// Return the bytes of a bytes object, valid while it is alive.
[[nodiscard]] auto bytes_view(PyObject* b) -> std::string_view {
    return {PyBytes_AS_STRING(b), static_cast<size_t>(PyBytes_GET_SIZE(b))};
}

// A StateDictTable, checked as StateDictTable._check checks it, so that the writer can read its
// arrays without further checks.
class StateDictTableView {
   public:
    // A field of the Metadata: its name, and its opcodes (None for the fields written from the
    // table or the write results).
    struct Field {
        std::string name;
        py::object opcodes;
    };

    explicit StateDictTableView(py::handle table)
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
        py::tuple const fields = tuple_attr(table, "fields");
        for (py::handle const field : fields) {
            if (!PyTuple_CheckExact(field.ptr()) || PyTuple_GET_SIZE(field.ptr()) != 2) {
                bad_state_dict_table("a field is not a (name, opcodes) tuple");
            }
            PyObject* const name = PyTuple_GET_ITEM(field.ptr(), 0);
            PyObject* const opcodes = PyTuple_GET_ITEM(field.ptr(), 1);
            if (!PyUnicode_CheckExact(name)) {
                bad_state_dict_table("a field name is a " + type_name(name) + ", not a str");
            }
            std::string_view const field_name = utf8(name);
            bool const written_here =
                field_name == STATE_DICT_METADATA || field_name == STORAGE_DATA;
            if (written_here ? opcodes != Py_None : !PyBytes_CheckExact(opcodes)) {
                bad_state_dict_table("unexpected opcodes of field " + std::string(field_name));
            }
            fields_.push_back(
                {std::string(field_name), py::reinterpret_borrow<py::object>(opcodes)});
        }
        check();
    }

    [[nodiscard]] auto entries() const -> size_t { return kind_.size(); }
    // Return the fingerprint of the table's fqn list, read once, on first use: only index tables
    // need it.
    [[nodiscard]] auto fqns_fingerprint() const -> int64_t {
        if (!fqns_fingerprint_) {
            fqns_fingerprint_ = table_.attr("fqn_list").attr("fingerprint").cast<int64_t>();
        }
        return *fqns_fingerprint_;
    }
    [[nodiscard]] auto fields() const -> std::vector<Field> const& { return fields_; }
    [[nodiscard]] auto fqn(size_t i) const -> PyObject* { return PyTuple_GET_ITEM(fqns_.ptr(), i); }
    [[nodiscard]] auto kind(size_t i) const -> Kind { return static_cast<Kind>(kind_[i]); }
    [[nodiscard]] auto properties(size_t i) const -> std::string_view {
        return bytes_view(PyTuple_GET_ITEM(properties_.ptr(), ref_[i]));
    }
    [[nodiscard]] auto other(size_t i) const -> std::string_view {
        return bytes_view(PyTuple_GET_ITEM(others_.ptr(), ref_[i]));
    }
    [[nodiscard]] auto ndim(size_t i) const -> size_t { return static_cast<size_t>(ndim_[i]); }
    [[nodiscard]] auto size(size_t i) const -> std::span<int64_t const> {
        return dims_.subspan(static_cast<size_t>(dims_start_[i]), ndim(i));
    }
    [[nodiscard]] auto chunks(size_t i) const -> size_t {
        return static_cast<size_t>(chunk_start_[i + 1] - chunk_start_[i]);
    }
    // Return chunk k of entry i: its offsets, then its sizes, after the entry's size.
    [[nodiscard]] auto chunk(size_t i, size_t k) const -> std::span<int64_t const> {
        size_t const width = 2 * ndim(i);
        return dims_.subspan(static_cast<size_t>(dims_start_[i]) + ndim(i) + (width * k), width);
    }
    [[nodiscard]] auto dims_size() const -> size_t { return dims_.size(); }

   private:
    // Check that the arrays describe entries() entries consistently; see StateDictTable._check.
    void check() const {
        size_t const n = entries();
        if (std::cmp_not_equal(PyTuple_GET_SIZE(fqns_.ptr()), n) || ref_.size() != n ||
            ndim_.size() != n) {
            bad_state_dict_table("arrays of different lengths");
        }
        auto const check_starts = [n](std::span<int64_t const> starts, std::optional<size_t> pool) {
            if (starts.size() != n + 1 || starts[0] != 0 || !std::ranges::is_sorted(starts) ||
                (pool && std::cmp_not_equal(starts[n], *pool))) {
                bad_state_dict_table("start offsets out of order or range");
            }
        };
        check_starts(chunk_start_, std::nullopt);
        check_starts(dims_start_, dims_.size());
        for (size_t i = 0; i < n; ++i) {
            check_entry(i);
        }
    }

    // Check entry i's fqn, kind, reference and dims, once check() has found the arrays consistent.
    void check_entry(size_t i) const {
        if (!PyUnicode_CheckExact(fqn(i))) {
            bad_state_dict_table("an fqn is not a str");
        }
        int64_t const ndim = ndim_[i];
        int64_t const chunk_count = chunk_start_[i + 1] - chunk_start_[i];
        int64_t const values = dims_start_[i + 1] - dims_start_[i];
        if (ndim < 0) {
            bad_state_dict_table("a negative number of dims");
        }
        switch (kind_[i]) {
            case static_cast<uint8_t>(Kind::Tensor):
                if (!holds_pickle(ref_[i], properties_) ||
                    !holds_chunks(values, ndim, chunk_count)) {
                    bad_state_dict_table("a tensor entry's dims don't match its chunks");
                }
                // Its chunks have no dims, so nothing else bounds their number.
                if (ndim == 0 && chunk_count > 1) {
                    bad_state_dict_table("a 0-dim tensor entry with more than one chunk");
                }
                break;
            case static_cast<uint8_t>(Kind::Other):
                if (!holds_pickle(ref_[i], others_)) {
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

    // Return whether values = ndim * (1 + 2 * chunk_count): a tensor's size, and the offsets and
    // sizes of chunk_count chunks.
    // All three are non-negative; divides first, so nothing can overflow.
    [[nodiscard]] static auto holds_chunks(int64_t values, int64_t ndim, int64_t chunk_count)
        -> bool {
        if (ndim == 0) {
            return values == 0;
        }
        int64_t const per_dim = values / ndim;  // 1 + 2 * chunk_count
        return values % ndim == 0 && per_dim % 2 == 1 && per_dim / 2 == chunk_count;
    }

    // Return whether ref is the index of a bytes object in pickles.
    [[nodiscard]] static auto holds_pickle(int64_t ref, py::tuple const& pickles) -> bool {
        return ref >= 0 && ref < PyTuple_GET_SIZE(pickles.ptr()) &&
               PyBytes_CheckExact(PyTuple_GET_ITEM(pickles.ptr(), ref));
    }

    py::object table_;
    mutable std::optional<int64_t> fqns_fingerprint_;
    py::tuple fqns_, properties_, others_;
    Array<uint8_t> kind_array_;
    Array<int64_t> ref_array_, ndim_array_, chunk_start_array_, dims_start_array_, dims_array_;
    std::span<uint8_t const> kind_;
    std::span<int64_t const> ref_, ndim_, chunk_start_, dims_start_, dims_;
    std::vector<Field> fields_;
};

// Return a numpy array that owns the vector's values, without copying them.
template <TableValue T>
[[nodiscard]] auto to_array(std::vector<T>&& vector) -> py::array_t<T> {
    auto owned = std::make_unique<std::vector<T>>(std::move(vector));
    py::capsule const owner(owned.get(), [](void* p) {
        std::unique_ptr<std::vector<T>> const reclaimed(static_cast<std::vector<T>*>(p));
    });
    std::vector<T> const& values = *owned.release();
    return py::array_t<T>({values.size()}, {sizeof(T)}, values.data(), owner);
}

// Transparent hash, so a map keyed by std::string can be looked up by string_view without building
// a std::string per lookup. std::hash<std::string> is not transparent; this hashes both key types
// the same.
struct StringHash {
    using is_transparent = void;  // NOLINT(readability-identifier-naming): the name std looks up
    [[nodiscard]] auto operator()(std::string_view s) const noexcept -> size_t {
        return std::hash<std::string_view>{}(s);
    }
};

// A map keyed by strings that can be looked up by string_view.
template <typename T>
using StringMap = std::unordered_map<std::string, T, StringHash, std::equal_to<>>;

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
    [[nodiscard]] auto build(py::handle metadata, py::object const& table_cls) && -> py::object {
        if (!is(metadata.ptr(), metadata_cls_)) {
            throw py::type_error("expected Metadata, got " + type_name(metadata.ptr()));
        }
        py::object const state = instance_dict(metadata.ptr());
        py::object const entries = getattr(metadata.ptr(), a_state_dict_metadata_.ptr());
        if (!PyDict_CheckExact(entries.ptr())) {
            throw py::type_error("expected dict for state_dict_metadata, got " +
                                 type_name(entries.ptr()));
        }
        // Snapshots, which hold the items while small_pickle runs Python code that could change
        // the dicts.
        py::list fields;
        for (py::handle const item : items(state.ptr())) {
            PyObject* const name = PyTuple_GET_ITEM(item.ptr(), 0);
            std::string_view const field_name = utf8(name);
            bool const written_here =
                field_name == STATE_DICT_METADATA || field_name == STORAGE_DATA;
            py::object const opcodes = written_here
                                           ? py::object(py::none())
                                           : py::object(pickled(PyTuple_GET_ITEM(item.ptr(), 1)));
            fields.append(py::make_tuple(py::handle(name), opcodes));
        }
        py::list const entry_items = items(entries.ptr());
        size_t const n = entry_items.size();
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
            if (std::optional<TensorEntry> const tensor =
                    is(entry, tensor_cls_) ? tensor_entry(entry) : std::nullopt) {
                // After the chunks are counted: small_pickle runs Python code.
                py::bytes opcodes = pickled(tensor->properties.ptr());
                std::string_view const key = bytes_view(opcodes.ptr());
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
        return table_cls(py::arg("fields") = fields,
                         py::arg("fqns") = fqns,
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
        py::object const state = instance_dict(entry);
        PyObject* const properties = dict_get(state.ptr(), a_properties_.ptr());
        PyObject* const size = dict_get(state.ptr(), a_size_.ptr());
        PyObject* const chunks = dict_get(state.ptr(), a_chunks_.ptr());
        if (PyDict_Size(state.ptr()) != 3 || (properties == nullptr) || (size == nullptr) ||
            !is(size, size_cls_) || (chunks == nullptr) || !PyList_CheckExact(chunks)) {
            return std::nullopt;
        }
        Py_ssize_t const ndim = PyTuple_GET_SIZE(size);
        Py_ssize_t const chunk_count = PyList_GET_SIZE(chunks);
        if (ndim == 0 && chunk_count > 1) {
            return std::nullopt;  // a scalar saved by several ranks without dedup
        }
        size_t const mark = dims_.size();
        if (!append_dims(size, dims_)) {
            dims_.resize(mark);
            return std::nullopt;
        }
        for (Py_ssize_t k = 0; k < chunk_count; ++k) {
            PyObject* const chunk = PyList_GET_ITEM(chunks, k);
            py::object const chunk_state =
                is(chunk, chunk_cls_) ? instance_dict(chunk) : py::object();
            PyObject* const offsets =
                chunk_state ? dict_get(chunk_state.ptr(), a_offsets_.ptr()) : nullptr;
            PyObject* const sizes =
                chunk_state ? dict_get(chunk_state.ptr(), a_sizes_.ptr()) : nullptr;
            if (!chunk_state || PyDict_Size(chunk_state.ptr()) != 2 || (offsets == nullptr) ||
                (sizes == nullptr) || !is(offsets, size_cls_) || !is(sizes, size_cls_) ||
                PyTuple_GET_SIZE(offsets) != ndim || PyTuple_GET_SIZE(sizes) != ndim ||
                !append_dims(offsets, dims_) || !append_dims(sizes, dims_)) {
                dims_.resize(mark);
                return std::nullopt;
            }
        }
        return TensorEntry{.properties = py::reinterpret_borrow<py::object>(properties),
                           .ndim = static_cast<size_t>(ndim),
                           .chunks = static_cast<size_t>(chunk_count)};
    }

    // Append a torch.Size's dims; return false unless each is an int within int64 (a torch.Size
    // may also hold numpy ints or larger ints, which only pickle writes exactly).
    [[nodiscard]] static auto append_dims(PyObject* dims, std::vector<int64_t>& out) -> bool {
        Py_ssize_t const n = PyTuple_GET_SIZE(dims);
        for (Py_ssize_t k = 0; k < n; ++k) {
            PyObject* const dim = PyTuple_GET_ITEM(dims, k);
            if (!PyLong_CheckExact(dim)) {
                return false;
            }
            long long const value = PyLong_AsLongLong(dim);
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

    [[nodiscard]] static auto is(PyObject* obj, py::object const& cls) -> bool {
        return static_cast<void const*>(Py_TYPE(obj)) == static_cast<void const*>(cls.ptr());
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
        return {out_.data(), out_.size()};
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
                dump_metadata_index(idx);
            }
            dump_storage_info(info, dict, n);
            end_item(i, n, SETITEMS);
        }
    }

    // MetadataIndex sets offset only when it is given; pickle its __dict__.
    void dump_metadata_index(PyObject* idx) {
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
    // from PyTorch 2.8; a _StorageInfo with it set is left to the stdlib pickler, after which
    // check_size checks that the Python code it ran left dict with its n items.
    void dump_storage_info(PyObject* info, PyObject* dict, Py_ssize_t n) {
        py::object const state = is(info, info_cls_) ? instance_dict(info) : py::object();
        PyObject* const transforms = state ? dict_get(state.ptr(), a_transform_.ptr()) : nullptr;
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
    void put_index_start(WriteFqn const& write_fqn, std::optional<int64_t> index) {
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
    void put_storage_info(WritePath const& write_path, int64_t offset, int64_t length) {
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
            if (n > SMALL_TUPLE.size()) {
                put(MARK);
            }
            for (size_t j = 0; j < n; ++j) {
                put_int(dim(j));
            }
            put(n <= SMALL_TUPLE.size() ? SMALL_TUPLE.at(n - 1) : TUPLE);
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

PYBIND11_MODULE(_native, m) {
    m.doc() = "Fast conversion and pickling of torch.distributed.checkpoint Metadata.";
    m.def(
        "from_metadata",
        [](py::handle metadata, py::object small_pickle, py::object const& table_cls) {
            return StateDictTableBuilder(std::move(small_pickle)).build(metadata, table_cls);
        },
        py::arg("metadata"),
        py::arg("small_pickle"),
        py::arg("table_cls"),
        "Convert metadata, without its storage_data, to a table_cls (a StateDictTable).");
    m.def(
        "dumps_table",
        [](py::handle table,
           py::object small_pickle,
           py::handle storage_data,
           py::object const& storage_rows) {
            return Writer(std::move(small_pickle))
                .dumps(StateDictTableView(table), storage_data, storage_rows);
        },
        py::arg("table"),
        py::arg("small_pickle"),
        py::arg("storage_data") = py::none(),
        py::arg("storage_rows") = py::none(),
        "Pickle the Metadata a state dict table was converted from, and return the bytes.");
}
