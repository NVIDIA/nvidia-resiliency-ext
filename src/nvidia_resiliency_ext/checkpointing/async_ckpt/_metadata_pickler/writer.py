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
- ``force``: as the default, but also on torch versions outside ``TESTED_TORCH_VERSIONS``;
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
from packaging.version import Version
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)

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
_ABSENT = object()

# First and last torch (major, minor) versions the writers and FileSystemWriterAsync.finish were
# checked against. Extend after checking a new torch release.
TESTED_TORCH_VERSIONS = ((2, 4), (2, 14))

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


def _str(s: str) -> bytes:
    raw = s.encode("utf-8", "surrogatepass")
    if len(raw) < 256:
        return b"\x8c" + bytes((len(raw),)) + raw
    return b"X" + struct.pack("<I", len(raw)) + raw


def _int(i: int) -> bytes:
    if 0 <= i < 256:
        return b"K" + bytes((i,))
    if 0 <= i < 65536:
        return b"M" + struct.pack("<H", i)
    if -(2**31) <= i < 2**31:
        return b"J" + struct.pack("<i", i)
    raw = pickle.encode_long(i)
    return b"\x8a" + bytes((len(raw),)) + raw


def _get(idx: int) -> bytes:
    return b"h" + bytes((idx,)) if idx < 256 else b"j" + struct.pack("<I", idx)


class _MetadataPickler:
    """Pure-Python writer; the C++ extension implements the same loop with the same output."""

    def __init__(self):
        self.out: list = [PROTO + b"\x04"]
        self.memo_len = 0
        self.strings: dict = {}
        self.ints: dict = {}
        self.sizes: dict = {}
        self.props: dict = {}

    def _memoize_top(self) -> bytes:
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

    # The writers encode only exact types; anything else (a bool, a tuple for a torch.Size, ...)
    # would unpickle as a different type, so it raises and dump_metadata falls back to pickle.dump.
    def int(self, i: int) -> bytes:
        if type(i) is not int:
            raise TypeError(f"expected int, got {type(i).__name__}")
        b = self.ints.get(i)
        if b is None:
            b = self.ints[i] = _int(i)
        return b

    def size(self, dims) -> bytes:
        if type(dims) is not torch.Size:
            raise TypeError(f"expected torch.Size, got {type(dims).__name__}")
        key = tuple(dims)
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
        key = (type(p), tuple(vars(p).items()))
        b = self.props.get(key)
        if b is None:
            b = self.props[key] = _small_pickle(p)
        return b

    def dumps(self, md: Metadata) -> bytes:
        self.SIZE = self.global_ref(*_CLASS_PATHS[torch.Size])
        METADATA = self.global_ref(*_CLASS_PATHS[Metadata])
        INDEX = self.global_ref(*_CLASS_PATHS[MetadataIndex])
        INFO = self.global_ref(*_CLASS_PATHS[_StorageInfo])
        CHUNK = self.global_ref(*_CLASS_PATHS[ChunkStorageMetadata])
        TENSOR = self.global_ref(*_CLASS_PATHS[TensorStorageMetadata])
        BYTES = self.global_ref(*_CLASS_PATHS[BytesStorageMetadata])
        k = {}
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
            k[name] = self._memoize_top()
        out = self.out
        size, string, int_ = self.size, self.string, self.int

        out.append(METADATA + EMPTY_TUPLE + NEWOBJ + EMPTY_DICT + MARK)
        for field, value in vars(md).items():
            out.append(_str(field))
            if field in ("state_dict_metadata", "storage_data") and type(value) is not dict:
                raise TypeError(f"expected dict for {field}, got {type(value).__name__}")
            if field == "state_dict_metadata":
                out.append(EMPTY_DICT)
                items = list(value.items())
                for start in range(0, len(items), _BATCH):
                    out.append(MARK)
                    for fqn, v in items[start : start + _BATCH]:
                        out.append(string(fqn))
                        if isinstance(v, BytesStorageMetadata) and not vars(v):
                            out.append(BYTES + EMPTY_TUPLE + NEWOBJ)
                            continue
                        if not isinstance(v, TensorStorageMetadata):
                            out.append(_small_pickle(v))
                            continue
                        out.append(
                            TENSOR
                            + EMPTY_TUPLE
                            + NEWOBJ
                            + EMPTY_DICT
                            + MARK
                            + k["properties"]
                            + self.properties(v.properties)
                            + k["size"]
                            + size(v.size)
                            + k["chunks"]
                            + EMPTY_LIST
                        )
                        chunks = v.chunks
                        if type(chunks) is not list:
                            raise TypeError(f"expected list of chunks, got {type(chunks).__name__}")
                        for cstart in range(0, len(chunks), _BATCH):
                            out.append(MARK)
                            out.extend(
                                CHUNK
                                + EMPTY_TUPLE
                                + NEWOBJ
                                + EMPTY_DICT
                                + MARK
                                + k["offsets"]
                                + size(c.offsets)
                                + k["sizes"]
                                + size(c.sizes)
                                + SETITEMS
                                + BUILD
                                for c in chunks[cstart : cstart + _BATCH]
                            )
                            out.append(APPENDS)
                        out.append(SETITEMS + BUILD)
                    out.append(SETITEMS)
            elif field == "storage_data":
                out.append(EMPTY_DICT)
                items = list(value.items())
                for start in range(0, len(items), _BATCH):
                    out.append(MARK)
                    for idx, info in items[start : start + _BATCH]:
                        # MetadataIndex sets offset only when it is given; pickle its __dict__.
                        state = vars(idx)
                        offset = state.get("offset", _ABSENT)
                        if len(state) != (2 if offset is _ABSENT else 3):
                            raise TypeError(f"unexpected MetadataIndex attributes: {list(state)}")
                        index = state["index"]
                        out.append(
                            INDEX
                            + EMPTY_TUPLE
                            + NEWOBJ
                            + EMPTY_DICT
                            + MARK
                            + k["fqn"]
                            + string(state["fqn"])
                            + k["index"]
                            + (NONE if index is None else int_(index))
                            + (
                                b""
                                if offset is _ABSENT
                                else k["offset"] + (NONE if offset is None else size(offset))
                            )
                            + SETITEMS
                            + BUILD
                        )
                        # transform_descriptors does not exist before PyTorch 2.8; when present
                        # and set, the whole _StorageInfo goes through the stdlib pickler.
                        if getattr(info, "transform_descriptors", None) is not None:
                            out.append(_small_pickle(info))
                            continue
                        out.append(
                            INFO
                            + EMPTY_TUPLE
                            + NEWOBJ
                            + EMPTY_DICT
                            + MARK
                            + k["relative_path"]
                            + string(info.relative_path)
                            + k["offset"]
                            + int_(info.offset)
                            + k["length"]
                            + int_(info.length)
                            + SETITEMS
                            + BUILD
                        )
                    out.append(SETITEMS)
            else:
                out.append(_small_pickle(value))
        out.append(SETITEMS + BUILD + STOP)
        return b"".join(out)


def _python_dumps(md: Metadata) -> bytes:
    return _MetadataPickler().dumps(md)


def _native_dumps(md: Metadata) -> bytes:
    return native.dumps(md, _small_pickle)


def _mode() -> str:
    return os.environ.get("NVRX_FAST_METADATA_PICKLE", "1").strip().lower()


@functools.lru_cache(maxsize=None)
def fast_metadata_enabled() -> bool:
    """Whether nvrx may write ``.metadata`` with its own code instead of torch's.

    False if disabled by ``NVRX_FAST_METADATA_PICKLE=0``, or if torch is outside
    ``TESTED_TORCH_VERSIONS`` (unless ``NVRX_FAST_METADATA_PICKLE=force``). Gates both the fast
    writers and ``FileSystemWriterAsync.finish``. Evaluated once per process; see the module
    docstring for the values of ``NVRX_FAST_METADATA_PICKLE``.
    """
    mode = _mode()
    if mode in ("0", "false", "off", "no"):
        return False
    if mode == "force":
        return True
    first, last = TESTED_TORCH_VERSIONS
    if not first <= Version(torch.__version__).release[:2] <= last:
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
    md = _sample_metadata()
    try:
        return pickle.loads(dumps(md)) == md  # nosec - our own bytes
    except Exception:
        logger.warning("fast .metadata pickling failed its self-check", exc_info=True)
        return False


@functools.lru_cache(maxsize=None)
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
