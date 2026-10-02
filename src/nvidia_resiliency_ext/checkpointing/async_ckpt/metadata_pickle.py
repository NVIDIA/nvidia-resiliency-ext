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
``nvrx_metadata_pickle`` extension) and in Python as a fallback. If the torch classes do not have
the expected layout, ``pickle.dump`` is used instead.

Set ``NVRX_FAST_METADATA_PICKLE=0`` to always use ``pickle.dump``, or ``=python`` to skip the C++
extension.
"""

import functools
import io
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
import struct
from dataclasses import fields, is_dataclass
from typing import IO, Callable, Optional

import torch
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
    import nvrx_metadata_pickle
except ImportError:
    nvrx_metadata_pickle = None

logger = logging.getLogger(__name__)

PROTO, STOP, MARK, POP = b"\x80", b".", b"(", b"0"
EMPTY_DICT, EMPTY_LIST, EMPTY_TUPLE = b"}", b"]", b")"
SETITEMS, APPENDS, BUILD, NEWOBJ, REDUCE = b"u", b"e", b"b", b"\x81", b"R"
STACK_GLOBAL, MEMOIZE, NONE = b"\x93", b"\x94", b"N"
TUPLE, TUPLE1, TUPLE2, TUPLE3 = b"t", b"\x85", b"\x86", b"\x87"
_BATCH = 1000  # same batching as the stdlib pickler


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
        ref = self.strings.get(s)
        if ref is not None:
            return ref
        ref = self.strings[s] = _get(self.memo_len)
        self.memo_len += 1
        return _str(s) + MEMOIZE

    def int(self, i: int) -> bytes:
        b = self.ints.get(i)
        if b is None:
            b = self.ints[i] = _int(i)
        return b

    def size(self, dims) -> bytes:
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
        key = (str(p.dtype), str(p.layout), p.requires_grad, str(p.memory_format), p.pin_memory)
        b = self.props.get(key)
        if b is None:
            b = self.props[key] = _small_pickle(p)
        return b

    def dumps(self, md: Metadata) -> bytes:
        meta = "torch.distributed.checkpoint.metadata"
        self.SIZE = self.global_ref("torch", "Size")
        METADATA = self.global_ref(meta, "Metadata")
        INDEX = self.global_ref(meta, "MetadataIndex")
        INFO = self.global_ref("torch.distributed.checkpoint.filesystem", "_StorageInfo")
        CHUNK = self.global_ref(meta, "ChunkStorageMetadata")
        TENSOR = self.global_ref(meta, "TensorStorageMetadata")
        BYTES = self.global_ref(meta, "BytesStorageMetadata")
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
                        out.append(
                            INDEX
                            + EMPTY_TUPLE
                            + NEWOBJ
                            + EMPTY_DICT
                            + MARK
                            + k["fqn"]
                            + string(idx.fqn)
                            + k["index"]
                            + (NONE if idx.index is None else int_(idx.index))
                            + k["offset"]
                            + (NONE if idx.offset is None else size(idx.offset))
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
    return nvrx_metadata_pickle.dumps(md, _small_pickle)


# The pickled state of these classes is their instance __dict__ (no __slots__ or custom reduce),
# with exactly these fields. _StorageInfo gained transform_descriptors in PyTorch 2.8, together
# with a __getstate__ that leaves it out when None.
_EXPECTED_FIELDS = {
    MetadataIndex: {"fqn", "offset", "index"},
    ChunkStorageMetadata: {"offsets", "sizes"},
    TensorStorageMetadata: {"properties", "size", "chunks"},
    BytesStorageMetadata: set(),
}
_STORAGE_INFO_FIELDS = {"relative_path", "offset", "length"}


def _layout_supported() -> bool:
    """Whether the torch metadata classes have the layout the fast writers assume."""
    for cls, expected in [*_EXPECTED_FIELDS.items(), (_StorageInfo, None)]:
        if not is_dataclass(cls) or "__slots__" in vars(cls):
            return False
        if any(name in vars(cls) for name in ("__reduce__", "__reduce_ex__", "__setstate__")):
            return False
        names = {f.name for f in fields(cls)}
        if cls is _StorageInfo:
            if names - {"transform_descriptors"} != _STORAGE_INFO_FIELDS:
                return False
        elif names != expected or "__getstate__" in vars(cls):
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
    mode = os.environ.get("NVRX_FAST_METADATA_PICKLE", "1").strip().lower()
    if mode in ("0", "false", "off", "no"):
        return None
    if not _layout_supported():
        logger.warning("Unexpected torch DCP metadata classes; writing .metadata with pickle.dump")
        return None
    if nvrx_metadata_pickle is not None and mode != "python" and _works(_native_dumps):
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
