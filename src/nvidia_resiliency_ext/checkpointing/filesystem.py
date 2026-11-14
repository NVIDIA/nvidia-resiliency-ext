# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Injectable checkpoint filesystems compatible with PyTorch DCP."""

import io
import ntpath
import os
import pickle
import posixpath
import stat
from abc import abstractmethod
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import BinaryIO, Literal

from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemBase

Path = str | os.PathLike
FileType = Literal["file", "directory", "missing"]

__all__ = [
    "ArrowFileSystem",
    "CheckpointFileSystem",
    "LocalFileSystem",
    "MultiStorageFileSystem",
]


def _child_path(suffix: str) -> str:
    suffix = os.fspath(suffix)
    if ntpath.isabs(suffix) or "://" in suffix or ".." in suffix.replace("\\", "/").split("/"):
        raise ValueError(f"Checkpoint child path must remain inside its directory: {suffix!r}")
    return suffix


def _join(path: Path, suffix: str) -> str:
    suffix = _child_path(suffix)
    root = os.fspath(path)
    return root.rstrip("/") + "/" + suffix if root else suffix


def _binary_mode(mode: str) -> None:
    if mode not in ("rb", "wb"):
        raise ValueError("Checkpoint filesystems support only 'rb' and 'wb' streams")


class CheckpointFileSystem(FileSystemBase):
    """DCP filesystem with directory inventory and backend-specific synchronization.

    Read streams must be seekable; output streams need write, tell, and flush, not fileno.
    Output context exit commits the file and propagates close errors. Implementations passed
    to async writers must be pickleable and must not share native clients across processes.
    """

    @abstractmethod
    def stat(self, path: Path) -> FileType:
        """Return the entry kind; only an absent entry returns ``missing``."""

    @abstractmethod
    def listdir(self, path: Path) -> Iterable[str]:
        """Return immediate child names."""

    @abstractmethod
    def sync(self, stream: BinaryIO) -> None:
        """Flush and request backend durability before the output stream closes."""


class LocalFileSystem(FileSystem, CheckpointFileSystem):
    """POSIX and mounted filesystem implementation."""

    @contextmanager
    def create_stream(self, path: Path, mode: str):
        _binary_mode(mode)
        with super().create_stream(path, mode) as stream:
            yield stream

    def concat_path(self, path: Path, suffix: str) -> Path:
        return super().concat_path(path, _child_path(suffix))

    def stat(self, path: Path) -> FileType:
        try:
            mode = os.stat(path).st_mode
        except FileNotFoundError:
            return "missing"
        return "directory" if stat.S_ISDIR(mode) else "file"

    def exists(self, path: Path) -> bool:
        return self.stat(path) != "missing"

    def listdir(self, path: Path) -> Iterable[str]:
        return os.listdir(path)

    def sync(self, stream: BinaryIO) -> None:
        stream.flush()
        os.fsync(stream.fileno())


class MultiStorageFileSystem(CheckpointFileSystem):
    """Multi-Storage Client adapter with lazy optional imports."""

    @staticmethod
    def _msc():
        import multistorageclient

        return multistorageclient

    def init_path(self, path: Path) -> str:
        return os.fspath(path)

    def concat_path(self, path: Path, suffix: str) -> str:
        return _join(path, suffix)

    @contextmanager
    def create_stream(self, path: Path, mode: str):
        _binary_mode(mode)
        with self._msc().open(os.fspath(path), mode) as stream:
            yield stream

    def mkdir(self, path: Path) -> None:
        self._msc().os.makedirs(os.fspath(path), exist_ok=True)

    def exists(self, path: Path) -> bool:
        return self.stat(path) != "missing"

    def rename(self, path: Path, new_path: Path) -> None:
        self._msc().Path(path).rename(new_path)

    def rm_file(self, path: Path) -> None:
        self._msc().Path(path).unlink()

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Path) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike))

    def stat(self, path: Path) -> FileType:
        entry = self._msc().Path(path)
        if entry.is_dir(strict=False):
            return "directory"
        try:
            mode = entry.stat().st_mode
        except FileNotFoundError:
            return "missing"
        return "directory" if stat.S_ISDIR(mode) else "file"

    def listdir(self, path: Path) -> Iterable[str]:
        return [entry.name for entry in self._msc().Path(path).iterdir()]

    def sync(self, stream: BinaryIO) -> None:
        stream.flush()
        stream.fsync()


class _ChunkedWriter:
    def __init__(self, stream, max_write_bytes: int):
        self._stream = stream
        self._max_write_bytes = max_write_bytes
        self._position = 0

    @property
    def closed(self):
        return self._stream.closed

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self._position

    def write(self, data):
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            chunk = view[offset : offset + self._max_write_bytes]
            written = self._stream.write(chunk)
            written = len(chunk) if written is None else written
            if written <= 0 or written > len(chunk):
                raise OSError("Arrow output stream did not make progress")
            offset += written
            self._position += written
            self._stream.flush()
        return offset

    def flush(self):
        self._stream.flush()

    def close(self):
        if not self.closed:
            self._stream.close()

    def fileno(self):
        raise io.UnsupportedOperation("Arrow streams have no file descriptor")


class ArrowFileSystem(CheckpointFileSystem):
    """Adapt a process-local native Arrow filesystem without payload staging.

    The pickleable factory must construct a fresh filesystem with blocking writes, for
    example ``pyarrow.fs.S3FileSystem(background_writes=False)``. ``max_write_bytes`` limits
    each native write; native filesystem buffering is controlled by the factory.
    """

    def __init__(
        self,
        filesystem_factory: Callable,
        *,
        uri_scheme: str | None = None,
        max_write_bytes: int = 8 << 20,
    ):
        if not callable(filesystem_factory):
            raise TypeError("filesystem_factory must be a pickleable top-level callable")
        if max_write_bytes <= 0:
            raise ValueError("max_write_bytes must be positive")
        if uri_scheme is not None and (not uri_scheme or ":" in uri_scheme or "/" in uri_scheme):
            raise ValueError("uri_scheme must be a scheme name without '://'")
        try:
            pickle.dumps(filesystem_factory)
        except (TypeError, AttributeError, pickle.PicklingError) as exc:
            raise TypeError("filesystem_factory must be a pickleable top-level callable") from exc
        self.filesystem_factory = filesystem_factory
        self.uri_scheme = uri_scheme
        self.max_write_bytes = max_write_bytes
        self._filesystem = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_filesystem"] = None
        state["_pid"] = None
        return state

    def _client(self):
        pid = os.getpid()
        if self._filesystem is None or self._pid != pid:
            self._filesystem = self.filesystem_factory()
            self._pid = pid
        return self._filesystem

    def _native_path(self, path: Path) -> str:
        path = os.fspath(path)
        if "://" in path:
            prefix = f"{self.uri_scheme}://"
            if self.uri_scheme is None or not path.startswith(prefix):
                raise ValueError(f"URI does not match Arrow scheme {self.uri_scheme!r}: {path!r}")
            return path[len(prefix) :]
        return path

    def init_path(self, path: Path) -> str:
        self._native_path(path)
        return os.fspath(path)

    def concat_path(self, path: Path, suffix: str) -> str:
        return _join(path, suffix)

    @contextmanager
    def create_stream(self, path: Path, mode: str):
        _binary_mode(mode)
        native = self._native_path(path)
        if mode == "rb":
            with io.BufferedReader(self._client().open_input_file(native)) as stream:
                yield stream
        else:
            stream = _ChunkedWriter(
                self._client().open_output_stream(native, compression=None, buffer_size=0),
                self.max_write_bytes,
            )
            try:
                yield stream
            finally:
                stream.close()

    def mkdir(self, path: Path) -> None:
        self._client().create_dir(self._native_path(path), recursive=True)

    def exists(self, path: Path) -> bool:
        return self.stat(path) != "missing"

    def rename(self, path: Path, new_path: Path) -> None:
        self._client().move(self._native_path(path), self._native_path(new_path))

    def rm_file(self, path: Path) -> None:
        self._client().delete_file(self._native_path(path))

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Path) -> bool:
        return isinstance(checkpoint_id, (str, os.PathLike))

    def stat(self, path: Path) -> FileType:
        from pyarrow.fs import FileType

        entry = self._client().get_file_info(self._native_path(path))
        if entry.type == FileType.NotFound:
            return "missing"
        if entry.type == FileType.Directory:
            return "directory"
        if entry.type == FileType.File:
            return "file"
        raise OSError(f"Arrow could not determine checkpoint entry type: {path!r}")

    def listdir(self, path: Path) -> Iterable[str]:
        from pyarrow.fs import FileSelector

        entries = self._client().get_file_info(FileSelector(self._native_path(path)))
        return [posixpath.basename(entry.path.rstrip("/")) for entry in entries]

    def sync(self, stream: BinaryIO) -> None:
        stream.flush()
