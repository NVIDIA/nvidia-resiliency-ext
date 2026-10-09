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

"""Writing the ``.metadata`` file of a torch distributed checkpoint, fast.

``dump_metadata`` writes it with the fastest writer that works here: the C++ ``native`` extension
of this package if it is built, else the Python writer in pickler.py. Both write the same bytes,
much faster than ``pickle.dump``; see pickler.py.

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
- ``force``: as the default, but also on torch versions not in ``TESTED_TORCH_VERSIONS``, except
  those in ``INCOMPATIBLE_TORCH_VERSIONS`` (logged as an error);
- ``0``, ``false``, ``off`` or ``no``: torch's own ``FileSystemWriter.finish`` and ``pickle.dump``.

Other values act as the default. The variable is read once per process, at the first save that
writes ``.metadata``; changing it later in the process has no effect.
"""

import copyreg
import dataclasses
import functools
import importlib
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
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

from . import pickler, table

try:
    from . import native
except ImportError:  # not built, or failed to compile
    native = None

logger = logging.getLogger(__name__)

# The torch (major, minor) versions the writers and FileSystemWriterAsync.finish were checked
# against. Add a version after checking it.
TESTED_TORCH_VERSIONS = frozenset(
    {(2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14)}
)
# Torch (major, minor) versions known not to work with them, even with
# NVRX_FAST_METADATA_PICKLE=force; this wins over TESTED_TORCH_VERSIONS. Torch 2.3's
# FileSystemWriter lacks metadata_path and storage_meta, which FileSystemWriterAsync.finish uses.
INCOMPATIBLE_TORCH_VERSIONS = frozenset({(2, 3)})


def _native_dumps(md: Metadata, storage_rows=None) -> bytes:
    """The pickle of md, from the native writer, which reads storage_rows itself."""
    return native.dumps(md, pickler.small_pickle, storage_rows)


def _mode() -> str:
    """The value of NVRX_FAST_METADATA_PICKLE, normalized."""
    return os.environ.get("NVRX_FAST_METADATA_PICKLE", "1").strip().lower()


def _torch_release() -> Optional[tuple]:
    """torch's (major, minor) version, or None if unparseable."""
    try:
        return Version(torch.__version__).release[:2]
    except InvalidVersion:
        return None


def _tested_torch() -> bool:
    """Whether torch is one of TESTED_TORCH_VERSIONS and not one of INCOMPATIBLE_TORCH_VERSIONS
    (an unparseable version is untested)."""
    release = _torch_release()
    return release not in INCOMPATIBLE_TORCH_VERSIONS and release in TESTED_TORCH_VERSIONS


@functools.cache
def fast_metadata_enabled() -> bool:
    """Whether nvrx may write ``.metadata`` with its own code instead of torch's.

    False if disabled by ``NVRX_FAST_METADATA_PICKLE=0``, if torch is in
    ``INCOMPATIBLE_TORCH_VERSIONS``, or if it is not in ``TESTED_TORCH_VERSIONS`` (unless
    ``NVRX_FAST_METADATA_PICKLE=force``). Gates both the fast writers and
    ``FileSystemWriterAsync.finish``. Evaluated once per process; see the module docstring for the
    values of ``NVRX_FAST_METADATA_PICKLE``.
    """
    mode = _mode()
    if mode in ("0", "false", "off", "no"):
        return False
    release = _torch_release()
    if release in INCOMPATIBLE_TORCH_VERSIONS:
        if mode == "force":
            logger.error(
                f"NVRX_FAST_METADATA_PICKLE=force ignored: torch {torch.__version__} is known not "
                "to work with nvrx's .metadata writer; using torch's own writer"
            )
        return False
    if mode == "force":
        return True
    if not _tested_torch():
        tested = ", ".join(f"{major}.{minor}" for major, minor in sorted(TESTED_TORCH_VERSIONS))
        logger.info(
            f"torch {torch.__version__} is not one of the versions nvrx's .metadata writer was "
            f"checked against ({tested}); using torch's own writer"
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
    keys = pickler.STATE_KEYS[type(obj)]
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
        for cls, (module, name) in pickler.CLASS_PATHS.items():
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
    if _works(pickler.dumps):
        return pickler.dumps
    return None


def writes_tables() -> bool:
    """Whether to send write results as tables and write storage_data from them.

    Needs a fast writer and, even with ``NVRX_FAST_METADATA_PICKLE=force``, a torch version in
    ``TESTED_TORCH_VERSIONS``: the table encodes the write results of those versions only.
    """
    return _tested_torch() and _select_dumps() is not None


def dump_metadata(metadata: Metadata, stream: IO[bytes], storage_rows=None) -> None:
    """Write metadata to stream as a pickle that ``pickle.load`` reads back as an equal Metadata.

    With storage_rows (the gathered write-result tables, a 2-D uint8 array with one zero-padded
    table per row, see table.py), its storage_data is the one they describe instead of
    metadata.storage_data.
    """
    dumps = _select_dumps()
    if dumps is not None:
        try:
            data = dumps(metadata, storage_rows=storage_rows)
        except Exception:
            logger.warning("fast .metadata pickling failed; using pickle.dump", exc_info=True)
        else:
            stream.write(data)
            return
    if storage_rows is not None:
        storage_data = table.to_storage_data(table.decode_rows(storage_rows))
        metadata = dataclasses.replace(metadata, storage_data=storage_data)
    # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
    # Severity: Medium   Confidence: High
    # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
    # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
    pickle.dump(metadata, stream)  # nosec
