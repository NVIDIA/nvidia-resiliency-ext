# spdx-filecopyrighttext: copyright (c) 2024 nvidia corporation & affiliates. all rights reserved.
# spdx-license-identifier: apache-2.0
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
# http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import weakref
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Optional, Union

import torch.distributed as dist

from .test_utilities import Utils

rank = int(os.environ["LOCAL_RANK"])


def empty_dir(path: Path):
    if Utils.rank > 0:
        return
    for p in path.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()


class TempNamedDir(TemporaryDirectory):
    """TemporaryDirectory with a fully named directory. Empties the dir if not empty."""

    def __init__(self, name: Union[str, Path], sync=True, ignore_cleanup_errors=False) -> None:
        self.name = str(name)
        if Utils.rank == 0:
            os.makedirs(name, exist_ok=True)
            empty_dir(Path(name))
        if sync:
            import torch

            torch.distributed.barrier()
        else:
            os.makedirs(name, exist_ok=True)

        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
        )
        self.sync = sync

    def cleanup(self, override_sync: Optional[bool] = None) -> None:
        sync = self.sync if override_sync is None else override_sync
        if sync:
            import torch

            torch.distributed.barrier()

        if Utils.rank == 0:
            super().cleanup()

    def __enter__(self):
        path = Path(super().__enter__())
        if self.sync:
            import torch

            torch.distributed.barrier()
        return path

    def __exit__(self, exc_type, exc_val, exc_tb):
        raised = exc_type is not None
        if not raised:
            self.cleanup()
