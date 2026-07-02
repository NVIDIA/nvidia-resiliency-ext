# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Byte-offset stability tests underpinning prepared-metadata reuse.

A reviewer raised that, because the DCP ``.metadata`` is pickled, the byte
offsets might change between checkpoints -- which would make reusing a prepared
``.metadata`` (the ``reuse_metadata_obj`` path) unsafe. These tests pin down
exactly when the offsets are / are not invariant. The conclusion: for a fixed
checkpoint structure they are invariant, so reuse is valid.

Facts established here (no GPU / no distributed init required -- DCP ``no_dist``):

* **Tensors** (weights, optimizer state) are written as raw bytes
  (``length = numel * element_size``) and ``_StorageInfo`` carries **no
  checksum**, so each shard's ``(offset, length)`` is independent of the tensor
  *values* -> identical across checkpoints of the same structure.
* The ``.metadata`` *file* bytes do differ between two saves, but **only** in
  ``storage_meta`` (a fresh per-save ``save_id`` UUID + the checkpoint path);
  the data-locating ``storage_data`` and ``state_dict_metadata`` are identical.
  (This is almost certainly what the reviewer observed -- a changing file hash.)
* TE ``_extra_state`` is a **uint8 tensor**, so it is raw-stored too and its
  serialized length is value-independent.
* The only thing that shifts offsets is a BYTE_IO *object* whose serialized
  *length* changes -- a structure/shape change, or a raw ``bytes`` payload whose
  pickle length is content-dependent. Both are excluded by the documented reuse
  invariant (same structure / world size / dist-ckpt-workers / FP8 recipe).
"""

import copy
import io
import os
import pickle

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter


def _save(state_dict, path):
    dcp.save(state_dict, storage_writer=FileSystemWriter(str(path)), no_dist=True)


def _storage_data(path):
    md = FileSystemReader(str(path)).read_metadata()
    return {str(k): (v.relative_path, v.offset, v.length) for k, v in md.storage_data.items()}


def _raw_metadata(path):
    with open(os.path.join(str(path), ".metadata"), "rb") as f:
        return pickle.load(f)


def _te_uint8_extra_state(amax_fill, scale_fill, hist_len=1024):
    """A uint8 ``_extra_state`` tensor shaped like TE's ``get_extra_state``."""
    buf = io.BytesIO()
    torch.save(
        {
            "scale_fwd": torch.full((1,), scale_fill),
            "amax_history_fwd": torch.full((hist_len,), amax_fill),
        },
        buf,
    )
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


def test_storage_offsets_are_value_independent(tmp_path):
    """Different weight/optimizer/_extra_state VALUES -> identical storage_data.

    This is the core refutation: a checkpoint saved at two different training
    steps has the same byte layout, so a prepared ``.metadata`` from step N
    locates step N+1's data correctly.
    """
    step1 = {
        "w": torch.zeros(4096),
        "b": torch.zeros(256),
        "x._extra_state": _te_uint8_extra_state(0.0, 1.0),
    }
    step2 = {
        "w": torch.randn(4096),
        "b": torch.randn(256),
        "x._extra_state": _te_uint8_extra_state(123.0, 5.0),
    }
    _save(step1, tmp_path / "s1")
    _save(step2, tmp_path / "s2")
    assert _storage_data(tmp_path / "s1") == _storage_data(tmp_path / "s2")


def test_metadata_file_differs_only_in_storage_meta(tmp_path):
    """The ``.metadata`` bytes differ across saves, but only in ``storage_meta``.

    Zeroing ``storage_meta`` (the fresh ``save_id`` UUID + checkpoint path) makes
    the two pickles byte-identical, so the difference a reviewer sees in the file
    hash is pure bookkeeping and does not affect read correctness.
    """
    sd1 = {"w": torch.zeros(1024), "x._extra_state": _te_uint8_extra_state(0.0, 1.0)}
    sd2 = {"w": torch.ones(1024), "x._extra_state": _te_uint8_extra_state(9.0, 2.0)}
    _save(sd1, tmp_path / "a")
    _save(sd2, tmp_path / "b")
    a, b = _raw_metadata(tmp_path / "a"), _raw_metadata(tmp_path / "b")

    assert a.storage_data == b.storage_data
    assert a.state_dict_metadata == b.state_dict_metadata
    assert a.storage_meta != b.storage_meta  # only this differs

    a2, b2 = copy.copy(a), copy.copy(b)
    a2.storage_meta = b2.storage_meta = None
    assert pickle.dumps(a2) == pickle.dumps(b2)


def test_uint8_extra_state_serialized_size_is_value_independent():
    """Real TE ``_extra_state`` is a uint8 tensor; raw storage -> length does not
    depend on values, so it never shifts following offsets."""

    def sz(t):
        buf = io.BytesIO()
        torch.save(t, buf)
        return len(buf.getvalue())

    s1 = _te_uint8_extra_state(0.0, 1.0)
    s2 = _te_uint8_extra_state(987.0, 6.0)
    assert s1.numel() == s2.numel()  # same shape across steps
    assert sz(s1) == sz(s2)  # and same serialized length


def test_byteio_payload_size_can_be_content_dependent():
    """The one real risk, documented: a raw ``bytes`` payload (not a tensor) has
    a content-dependent pickle length, so if ``_extra_state`` were stored that
    way (older TE) AND its content changed, offsets would shift. A uint8 tensor
    of the same bytes does not have this problem (raw storage)."""

    def sz(o):
        buf = io.BytesIO()
        torch.save(o, buf)
        return len(buf.getvalue())

    n = 8192
    low_entropy = b"\x00" * n
    high_entropy = os.urandom(n)
    # As uint8 tensors (the real TE representation): raw storage -> equal length.
    assert sz(torch.frombuffer(bytearray(low_entropy), dtype=torch.uint8)) == sz(
        torch.frombuffer(bytearray(high_entropy), dtype=torch.uint8)
    )
    # As bytes objects (pickle stream): content-dependent length.
    assert sz(low_entropy) != sz(high_entropy)
