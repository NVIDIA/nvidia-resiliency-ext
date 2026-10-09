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

"""Fast pickling of the ``.metadata`` file of a torch distributed checkpoint.

``pickle.dump`` of a DCP ``Metadata`` runs the generic object protocol for each of its many
small objects and tracks each of them in its memo, and it writes every fqn string in full each time
a separate string object holds it. This module writes the same pickle opcodes directly: it knows
the layout of the metadata classes, skips the per-object machinery, and writes each fqn and file
name once and refers back to it. The result is an ordinary pickle that references only torch
classes, so stock ``torch.distributed.checkpoint`` loads it unchanged; it unpickles into a
``Metadata`` equal to the one that was written.

The writing loop is implemented twice with byte-identical output: in C++ (the optional
``native`` extension of this package) and in Python as a fallback.

The writers depend on how torch's metadata classes pickle, so they are used only when all of these
hold, and ``pickle.dump`` is used otherwise:

- torch is one of the versions they were checked against (``TESTED_TORCH_VERSIONS``);
- each class is importable from the path the writers reference, and pickles its state exactly as
  the writers encode it (``_layout_supported``);
- a sample ``Metadata`` written by the writer unpickles equal to itself (``_works``).

A metadata object that does not fit the writers at save time, for example a container of an
unexpected type, also falls back to ``pickle.dump`` for that save.

The environment variable ``NVRX_FAST_METADATA_PICKLE`` selects how ``.metadata`` is written:

- unset or ``1`` (default): the C++ writer if it is built, else the Python writer, subject to the
  checks above;
- ``python``: the Python writer, subject to the same checks;
- ``force``: as the default, but also on torch versions outside ``TESTED_TORCH_VERSIONS``, except
  those in ``INCOMPATIBLE_TORCH_VERSIONS`` (logged as an error);
- ``0``, ``false``, ``off`` or ``no``: torch's own ``FileSystemWriter.finish`` and ``pickle.dump``.

Other values act as the default. The variable is read once per process, at the first save that
writes ``.metadata``; changing it later in the process has no effect.
"""

import copyreg
import functools
import importlib
import io
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
import struct
from typing import IO, Callable, Optional

import torch
from packaging.version import InvalidVersion, Version
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)

from . import table

try:
    from . import native
except ImportError:  # not built, or failed to compile
    native = None

logger = logging.getLogger(__name__)

PROTO, STOP, MARK, POP = b"\x80", b".", b"(", b"0"
EMPTY_DICT, EMPTY_LIST, EMPTY_TUPLE = b"}", b"]", b")"
SETITEMS, APPENDS, BUILD, NEWOBJ, REDUCE = b"u", b"e", b"b", b"\x81", b"R"
STACK_GLOBAL, MEMOIZE, NONE = b"\x93", b"\x94", b"N"
TUPLE, TUPLE1, TUPLE2, TUPLE3 = b"t", b"\x85", b"\x86", b"\x87"
_BATCH = 1000  # same batching as the stdlib pickler
END = SETITEMS + BUILD  # end of an object's state dict
_ABSENT = object()

# First and last torch (major, minor) versions the writers and FileSystemWriterAsync.finish were
# checked against. Extend after checking a new torch release.
TESTED_TORCH_VERSIONS = ((2, 4), (2, 14))
# Torch (major, minor) versions known not to work with them, even with
# NVRX_FAST_METADATA_PICKLE=force. Torch 2.3's FileSystemWriter lacks metadata_path and
# storage_meta, which FileSystemWriterAsync.finish uses.
INCOMPATIBLE_TORCH_VERSIONS = frozenset({(2, 3)})

_META = "torch.distributed.checkpoint.metadata"
# The import path the writers emit for each class, as the stdlib pickler would.
_CLASS_PATHS = {
    torch.Size: ("torch", "Size"),
    Metadata: (_META, "Metadata"),
    MetadataIndex: (_META, "MetadataIndex"),
    _StorageInfo: ("torch.distributed.checkpoint.filesystem", "_StorageInfo"),
    ChunkStorageMetadata: (_META, "ChunkStorageMetadata"),
    TensorStorageMetadata: (_META, "TensorStorageMetadata"),
    BytesStorageMetadata: (_META, "BytesStorageMetadata"),
}
# The state each class is encoded with: these attributes, in this order. A MetadataIndex has no
# offset attribute when it was created without one, and its state then has no offset either. A
# _StorageInfo whose transform_descriptors (PyTorch 2.8+) is set is left to the stdlib pickler.
_STATE_KEYS = {
    MetadataIndex: ("fqn", "index", "offset"),
    _StorageInfo: ("relative_path", "offset", "length"),
    ChunkStorageMetadata: ("offsets", "sizes"),
    TensorStorageMetadata: ("properties", "size", "chunks"),
    BytesStorageMetadata: (),
}


def _small_pickle(obj) -> bytes:
    """Opcodes for a small object, from the stdlib pickler without memo, to splice into the stream.

    Protocol 3 emits no FRAME opcodes, and fast mode emits no memo PUT/GET, so the body can be
    embedded anywhere without disturbing the writer's memo numbering.
    """
    buf = io.BytesIO()
    p = pickle.Pickler(buf, protocol=3)
    p.fast = True
    p.dump(obj)
    data = buf.getvalue()
    assert data[:2] == b"\x80\x03" and data[-1:] == STOP
    return data[2:-1]


def _str_header(n: int) -> bytes:
    """The opcode and length of an n-byte str, as the stdlib pickler picks them at protocol 4:
    SHORT_BINUNICODE, BINUNICODE from 256 bytes on, BINUNICODE8 past 4 GiB."""
    if n < 256:
        return b"\x8c" + bytes((n,))
    if n <= 0xFFFFFFFF:
        return b"X" + struct.pack("<I", n)
    return b"\x8d" + struct.pack("<Q", n)


def _str(s: str) -> bytes:
    """A str."""
    raw = s.encode("utf-8", "surrogatepass")
    return _str_header(len(raw)) + raw


def _int(i: int) -> bytes:
    """An int, in the shortest form pickle uses: BININT1, BININT2, BININT or LONG1."""
    if 0 <= i < 256:
        return b"K" + bytes((i,))
    if 0 <= i < 65536:
        return b"M" + struct.pack("<H", i)
    if -(2**31) <= i < 2**31:
        return b"J" + struct.pack("<i", i)
    raw = pickle.encode_long(i)
    return b"\x8a" + bytes((len(raw),)) + raw


def _get(idx: int) -> bytes:
    """Fetch memo entry idx: BINGET, or LONG_BINGET from 256 on."""
    return b"h" + bytes((idx,)) if idx < 256 else b"j" + struct.pack("<I", idx)


class _MetadataPickler:
    """Pure-Python writer; the C++ extension implements the same loop with the same output."""

    def __init__(self):
        """An empty protocol 4 pickle; dumps writes one Metadata into it."""
        self.out: list = [PROTO + b"\x04"]
        self.memo_len = 0
        self.strings: dict = {}
        self.ints: dict = {}
        self.sizes: dict = {}
        self.props: dict = {}

    def _memoize_top(self) -> bytes:
        """Memo index of the object just memoized, as the opcode that fetches it."""
        ref = _get(self.memo_len)
        self.memo_len += 1
        return ref

    def global_ref(self, module: str, name: str) -> bytes:
        """Push module.name once into the memo; return the opcode that fetches it."""
        self.out.append(_str(module) + _str(name) + STACK_GLOBAL + MEMOIZE + POP)
        return self._memoize_top()

    def string(self, s: str) -> bytes:
        """First use writes the string and memoizes it; later uses fetch it from the memo."""
        if type(s) is not str:
            raise TypeError(f"expected str, got {type(s).__name__}")
        ref = self.strings.get(s)
        if ref is not None:
            return ref
        ref = self.strings[s] = _get(self.memo_len)
        self.memo_len += 1
        return _str(s) + MEMOIZE

    def int(self, i: int) -> bytes:
        """An int, encoded once per value.

        Like the other checks here, it accepts only the exact type: anything else (a bool, a tuple
        for a torch.Size, ...) would unpickle as a different type, so it raises and dump_metadata
        falls back to pickle.dump.
        """
        if type(i) is not int:
            raise TypeError(f"expected int, got {type(i).__name__}")
        b = self.ints.get(i)
        if b is None:
            b = self.ints[i] = _int(i)
        return b

    def size(self, dims) -> bytes:
        """A torch.Size, as pickle reduces it: torch.Size(tuple_of_ints). Encoded once per value."""
        if type(dims) is not torch.Size:
            raise TypeError(f"expected torch.Size, got {type(dims).__name__}")
        return self.size_of(tuple(dims))

    def size_of(self, key: tuple) -> bytes:
        """A torch.Size with the given dims (a tuple of ints). Encoded once per value."""
        b = self.sizes.get(key)
        if b is None:
            ints = b"".join(self.int(d) for d in key)
            n = len(key)
            if n == 0:
                tup = EMPTY_TUPLE
            elif n <= 3:
                tup = ints + (TUPLE1, TUPLE2, TUPLE3)[n - 1]
            else:
                tup = MARK + ints + TUPLE
            b = self.sizes[key] = self.SIZE + tup + TUPLE1 + REDUCE
        return b

    def properties(self, p) -> bytes:
        """A TensorProperties, from the stdlib pickler, once per distinct value."""
        # Values' types are part of the key: equal values of different types (True and 1) are
        # pickled differently.
        key = (type(p), tuple((name, type(v), v) for name, v in vars(p).items()))
        b = self.props.get(key)
        if b is None:
            b = self.props[key] = _small_pickle(p)
        return b

    def dumps(self, md: Metadata, storage_tables=None) -> bytes:
        """The pickle of md: a NEWOBJ of Metadata built from its __dict__, field by field.

        With storage_tables (decoded write-result tables, see table.py), storage_data is written
        from them instead of md.storage_data.
        """
        if type(md) is not Metadata:
            raise TypeError(f"expected Metadata, got {type(md).__name__}")
        self._prelude()
        out = self.out
        out.append(self.METADATA + EMPTY_TUPLE + NEWOBJ + EMPTY_DICT + MARK)
        for field, value in vars(md).items():
            out.append(_str(field))
            if field == "state_dict_metadata":
                self._state_dict_metadata(value)
            elif field == "storage_data":
                if storage_tables is None:
                    self._storage_data(value)
                else:
                    self._storage_data_tables(storage_tables)
            else:
                out.append(_small_pickle(value))
        out.append(SETITEMS + BUILD + STOP)
        return b"".join(out)

    def _prelude(self) -> None:
        """The classes and dict keys the encoding refers to, each memoized once."""
        self.SIZE = self.global_ref(*_CLASS_PATHS[torch.Size])
        self.METADATA = self.global_ref(*_CLASS_PATHS[Metadata])
        self.INDEX = self.global_ref(*_CLASS_PATHS[MetadataIndex])
        self.INFO = self.global_ref(*_CLASS_PATHS[_StorageInfo])
        self.CHUNK = self.global_ref(*_CLASS_PATHS[ChunkStorageMetadata])
        self.TENSOR = self.global_ref(*_CLASS_PATHS[TensorStorageMetadata])
        self.BYTES = self.global_ref(*_CLASS_PATHS[BytesStorageMetadata])
        self.k = {}
        for name in (
            "fqn",
            "index",
            "offset",
            "relative_path",
            "length",
            "offsets",
            "sizes",
            "properties",
            "size",
            "chunks",
        ):
            self.out.append(_str(name) + MEMOIZE + POP)
            self.k[name] = self._memoize_top()
        # The constant runs of opcodes around each object's variable parts.
        k, start = self.k, EMPTY_TUPLE + NEWOBJ + EMPTY_DICT + MARK
        self.TENSOR_START = self.TENSOR + start + k["properties"]
        self.TENSOR_CHUNKS = k["chunks"] + EMPTY_LIST
        self.CHUNK_START = self.CHUNK + start + k["offsets"]
        self.INDEX_START = self.INDEX + start + k["fqn"]
        self.INFO_START = self.INFO + start + k["relative_path"]

    def _state_dict_metadata(self, value: dict) -> None:
        """Metadata.state_dict_metadata: fqn -> TensorStorageMetadata or BytesStorageMetadata."""
        if type(value) is not dict:
            raise TypeError(f"expected dict for state_dict_metadata, got {type(value).__name__}")
        out, k, size, string = self.out, self.k, self.size, self.string
        BYTES_OBJ = self.BYTES + EMPTY_TUPLE + NEWOBJ
        TENSOR_START, K_SIZE, TENSOR_CHUNKS = self.TENSOR_START, k["size"], self.TENSOR_CHUNKS
        CHUNK_START, K_SIZES = self.CHUNK_START, k["sizes"]
        out.append(EMPTY_DICT)
        items = list(value.items())
        for start in range(0, len(items), _BATCH):
            out.append(MARK)
            for fqn, v in items[start : start + _BATCH]:
                out.append(string(fqn))
                if type(v) is BytesStorageMetadata and not vars(v):
                    out.append(BYTES_OBJ)
                    continue
                if type(v) is not TensorStorageMetadata:
                    out.append(_small_pickle(v))
                    continue
                state = vars(v)
                if len(state) != 3:
                    raise TypeError(f"unexpected TensorStorageMetadata attributes: {list(state)}")
                chunks = state["chunks"]
                if type(chunks) is not list:
                    raise TypeError(f"expected list of chunks, got {type(chunks).__name__}")
                out.append(
                    TENSOR_START
                    + self.properties(state["properties"])
                    + K_SIZE
                    + size(state["size"])
                    + TENSOR_CHUNKS
                )
                for cstart in range(0, len(chunks), _BATCH):
                    out.append(MARK)
                    for c in chunks[cstart : cstart + _BATCH]:
                        if type(c) is not ChunkStorageMetadata:
                            out.append(_small_pickle(c))
                            continue
                        cstate = vars(c)
                        if len(cstate) != 2:
                            raise TypeError(
                                f"unexpected ChunkStorageMetadata attributes: {list(cstate)}"
                            )
                        out.append(
                            CHUNK_START
                            + size(cstate["offsets"])
                            + K_SIZES
                            + size(cstate["sizes"])
                            + END
                        )
                    out.append(APPENDS)
                out.append(END)
            out.append(SETITEMS)

    def _storage_data(self, value: dict) -> None:
        """Metadata.storage_data: MetadataIndex -> _StorageInfo."""
        if type(value) is not dict:
            raise TypeError(f"expected dict for storage_data, got {type(value).__name__}")
        out, k, size, string, int_ = self.out, self.k, self.size, self.string, self.int
        INDEX_START, K_INDEX, K_OFFSET = self.INDEX_START, k["index"], k["offset"]
        INFO_START, K_LENGTH = self.INFO_START, k["length"]
        out.append(EMPTY_DICT)
        items = list(value.items())
        for start in range(0, len(items), _BATCH):
            out.append(MARK)
            for idx, info in items[start : start + _BATCH]:
                if type(idx) is not MetadataIndex:
                    out.append(_small_pickle(idx))
                else:
                    # MetadataIndex sets offset only when it is given; pickle its __dict__.
                    state = vars(idx)
                    offset = state.get("offset", _ABSENT)
                    if len(state) != (2 if offset is _ABSENT else 3):
                        raise TypeError(f"unexpected MetadataIndex attributes: {list(state)}")
                    index = state["index"]
                    out.append(
                        INDEX_START
                        + string(state["fqn"])
                        + K_INDEX
                        + (NONE if index is None else int_(index))
                        + (
                            b""
                            if offset is _ABSENT
                            else K_OFFSET + (NONE if offset is None else size(offset))
                        )
                        + END
                    )
                # _StorageInfo pickles its __dict__ without None values. transform_descriptors
                # exists from PyTorch 2.8; a _StorageInfo with it set is left to the stdlib
                # pickler.
                state = vars(info) if type(info) is _StorageInfo else None
                if state is None or state.get("transform_descriptors") is not None:
                    out.append(_small_pickle(info))
                    continue
                if len(state) != (4 if "transform_descriptors" in state else 3):
                    raise TypeError(f"unexpected _StorageInfo attributes: {list(state)}")
                out.append(
                    INFO_START
                    + string(state["relative_path"])
                    + K_OFFSET
                    + int_(state["offset"])
                    + K_LENGTH
                    + int_(state["length"])
                    + END
                )
            out.append(SETITEMS)

    def _storage_data_tables(self, tables) -> None:
        """Metadata.storage_data from write-result tables, encoded as _storage_data encodes the
        dict finish builds from the same write results."""
        out, k, string, int_ = self.out, self.k, self.string, self.int
        INDEX_START, K_INDEX, K_OFFSET = self.INDEX_START, k["index"], k["offset"]
        INFO_START, K_LENGTH = self.INFO_START, k["length"]
        out.append(EMPTY_DICT)
        count = 0
        for t in tables:
            strings, offsets, pos = t.strings, t.offsets.tolist(), 0
            for fqn, index, ndim, path, offset, length, flags in t.entry.tolist():
                if count % _BATCH == 0:
                    if count:
                        out.append(SETITEMS)
                    out.append(MARK)
                count += 1
                if flags & table.FLAG_OFFSET_SIZE:
                    off = K_OFFSET + self.size_of(tuple(offsets[pos : pos + ndim]))
                    pos += ndim
                elif flags & table.FLAG_OFFSET:
                    off = K_OFFSET + NONE
                else:
                    off = b""
                out.append(
                    INDEX_START
                    + string(strings[fqn])
                    + K_INDEX
                    + (int_(index) if flags & table.FLAG_INDEX else NONE)
                    + off
                    + END
                    + INFO_START
                    + string(strings[path])
                    + K_OFFSET
                    + int_(offset)
                    + K_LENGTH
                    + int_(length)
                    + END
                )
        if count:
            out.append(SETITEMS)


def _python_dumps(md: Metadata, storage_tables=None) -> bytes:
    """The pickle of md, from the Python writer."""
    return _MetadataPickler().dumps(md, storage_tables)


def _native_dumps(md: Metadata) -> bytes:
    """The pickle of md, from the native writer."""
    return native.dumps(md, _small_pickle)


def _mode() -> str:
    """The value of NVRX_FAST_METADATA_PICKLE, normalized."""
    return os.environ.get("NVRX_FAST_METADATA_PICKLE", "1").strip().lower()


@functools.cache
def fast_metadata_enabled() -> bool:
    """Whether nvrx may write ``.metadata`` with its own code instead of torch's.

    False if disabled by ``NVRX_FAST_METADATA_PICKLE=0``, if torch is in
    ``INCOMPATIBLE_TORCH_VERSIONS``, or if it is outside ``TESTED_TORCH_VERSIONS`` (unless
    ``NVRX_FAST_METADATA_PICKLE=force``). Gates both the fast writers and
    ``FileSystemWriterAsync.finish``. Evaluated once per process; see the module docstring for the
    values of ``NVRX_FAST_METADATA_PICKLE``.
    """
    mode = _mode()
    if mode in ("0", "false", "off", "no"):
        return False
    try:
        release = Version(torch.__version__).release[:2]
    except InvalidVersion:  # an unparseable version counts as untested
        release = None
    if release in INCOMPATIBLE_TORCH_VERSIONS:
        if mode == "force":
            logger.error(
                f"NVRX_FAST_METADATA_PICKLE=force ignored: torch {torch.__version__} is known not "
                "to work with nvrx's .metadata writer; using torch's own writer"
            )
        return False
    if mode == "force":
        return True
    first, last = TESTED_TORCH_VERSIONS
    if release is None or not first <= release <= last:
        logger.info(
            f"torch {torch.__version__} is outside the versions nvrx's .metadata writer was "
            f"checked against ({first[0]}.{first[1]} to {last[0]}.{last[1]}); "
            "using torch's own writer"
        )
        return False
    return True


def _sample_metadata() -> Metadata:
    """A small Metadata that exercises every encoding path of the fast writers."""
    props = TensorProperties(dtype=torch.bfloat16)
    tensors = {
        "w": TensorStorageMetadata(
            props,
            torch.Size([4, 2**40, 3, 2]),
            [
                ChunkStorageMetadata(torch.Size([0, 0, 0, 0]), torch.Size([2, 2**40, 3, 2])),
                ChunkStorageMetadata(torch.Size([2, 0, 0, 0]), torch.Size([2, 2**40, 3, 2])),
            ],
        ),
        "s": TensorStorageMetadata(
            TensorProperties(dtype=torch.float32),
            torch.Size([]),
            [ChunkStorageMetadata(torch.Size([]), torch.Size([]))],
        ),
    }
    # Enough distinct strings to need 4-byte memo references.
    blobs = {f"obj.{i}/shard_{i}": BytesStorageMetadata() for i in range(300)}
    storage = {
        MetadataIndex("w", torch.Size([0, 0, 0, 0]), 0): _StorageInfo("__0_0.distcp", 0, 70000),
        MetadataIndex("w", torch.Size([2, 0, 0, 0]), 1): _StorageInfo("__1_0.distcp", 2**33, 2**31),
        MetadataIndex("s", torch.Size([]), None): _StorageInfo("__0_0.distcp", 300, 2),
    }
    storage.update(
        {
            MetadataIndex(fqn): _StorageInfo("__0_1.distcp", i * 1000, 17)
            for i, fqn in enumerate(blobs)
        }
    )
    return Metadata(state_dict_metadata={**tensors, **blobs}, storage_data=storage)


def _encoded_state(obj) -> dict:
    """The state the writers encode obj with."""
    keys = _STATE_KEYS[type(obj)]
    if type(obj) is MetadataIndex:
        keys = [key for key in keys if key in vars(obj)]
    return {key: getattr(obj, key) for key in keys}


def _reduces_as_encoded(obj) -> bool:
    """Whether pickle reduces obj to a NEWOBJ of its class with the state the writers encode.

    The state must match in order too; an empty state means no state at all.
    """
    state = _encoded_state(obj)
    r = obj.__reduce_ex__(4)
    return (
        len(r) >= 3
        and r[0] is copyreg.__newobj__
        and r[1] == (type(obj),)
        and all(x is None for x in r[3:])
        and (list(r[2].items()) == list(state.items()) if state else not r[2])
    )


def _layout_supported() -> bool:
    """Whether torch's metadata classes pickle exactly as the fast writers encode them."""
    try:
        for cls, (module, name) in _CLASS_PATHS.items():
            if getattr(importlib.import_module(module), name, None) is not cls:
                return False
        md = _sample_metadata()
        r = md.__reduce_ex__(4)
        if r[:2] != (copyreg.__newobj__, (Metadata,)) or r[2] != vars(md):
            return False
        tensor = md.state_dict_metadata["w"]
        index, info = next(iter(md.storage_data.items()))
        samples = [
            tensor,
            tensor.chunks[0],
            md.state_dict_metadata["obj.0/shard_0"],
            index,
            MetadataIndex("obj.0/shard_0"),  # without an offset
            info,
        ]
        if not all(_reduces_as_encoded(obj) for obj in samples):
            return False
        return tensor.size.__reduce_ex__(4) == (torch.Size, (tuple(tensor.size),))
    except Exception:
        logger.debug("torch DCP metadata layout check failed", exc_info=True)
        return False


def _works(dumps: Callable[[Metadata], bytes]) -> bool:
    """Whether dumps writes the sample Metadata so that it unpickles equal to itself."""
    md = _sample_metadata()
    try:
        return pickle.loads(dumps(md)) == md  # nosec - our own bytes
    except Exception:
        logger.warning("fast .metadata pickling failed its self-check", exc_info=True)
        return False


@functools.cache
def _select_dumps() -> Optional[Callable[[Metadata], bytes]]:
    """The fastest writer that works here, or None to use pickle.dump."""
    if not fast_metadata_enabled():
        return None
    if not _layout_supported():
        logger.warning("Unexpected torch DCP metadata classes; writing .metadata with pickle.dump")
        return None
    if native is not None and _mode() != "python" and _works(_native_dumps):
        return _native_dumps
    if _works(_python_dumps):
        return _python_dumps
    return None


def dump_metadata(metadata: Metadata, stream: IO[bytes]) -> None:
    """Write metadata to stream as a pickle that ``pickle.load`` reads back as an equal Metadata."""
    dumps = _select_dumps()
    if dumps is not None:
        try:
            data = dumps(metadata)
        except Exception:
            logger.warning("fast .metadata pickling failed; using pickle.dump", exc_info=True)
        else:
            stream.write(data)
            return
    # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
    # Severity: Medium   Confidence: High
    # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
    # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
    pickle.dump(metadata, stream)  # nosec
