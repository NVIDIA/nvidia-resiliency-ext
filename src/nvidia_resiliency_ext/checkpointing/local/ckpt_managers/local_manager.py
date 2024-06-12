# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

""" A basic manager for local checkpoints."""

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import torch

from ...utils import debug_time
from ..base_state_dict import TensorAwareStateDict
from ..replication.strategies import ReplicationStrategy
from .base_manager import (
    BaseCheckpointManager,
    CheckpointingException,
    CkptID,
    SameMachineReplicationException,
)

logger = logging.getLogger(__name__)


class LocalCheckpointManager(BaseCheckpointManager):
    """Local Checkpoint Manager designed for handling checkpoints on local storage devices
    like SSDs or RAM disks.

    Args:
        root_local_ckpt_dir (str, Path): root checkpoint directory on local storage.
            Checkpoints from different iterations can be saved within the same root directory,
            as each will have a unique name
        session_id (str, optional): adds additional identification opportunity for local
            checkpoints used in different training workloads. An example use case
            is the `root_local_ckpt_dir` being configured by the cluster administrator
            (e.g. /tmp/...) and `session_id` configured by the end user for
            differentiating different local checkpoints.
        repl_strategy (ReplicationStrategy, optional): strategy used to perform local checkpoint
            shards replication.
    """

    def __init__(
        self,
        root_local_ckpt_dir: Union[str, Path],
        session_id: str = '',
        repl_strategy: Optional[ReplicationStrategy] = None,
    ):
        super().__init__(session_id, repl_strategy)
        self.root_local_ckpt_dir = root_local_ckpt_dir
        self._dir_created = False
        self._local_ckpt_dir = None

    @property
    def local_ckpt_dir(self):
        if self._local_ckpt_dir is None:
            self._local_ckpt_dir = Path(self.root_local_ckpt_dir) / self.session_id / str(self.rank)
        return self._local_ckpt_dir

    def _ensure_dir(self):
        """Ensure the checkpoint directory exists, creating it if necessary."""
        if not self._dir_created:
            os.makedirs(self.local_ckpt_dir, exist_ok=True)
            self._dir_created = True

    def _my_ckpt_ids(self) -> Iterable[CkptID]:
        """Collect all locally available checkpoint IDs."""
        self._ensure_dir()
        my_files = [f.name for f in self.local_ckpt_dir.iterdir() if f.is_file()]
        pattern = self._filename_from_template('\\d+', '\\d+', '\\')
        return [
            self._filename_to_id(filename)
            for filename in my_files
            if re.fullmatch(pattern, filename)
        ]

    @debug_time('LocalCheckpointManager._load', logger)
    def _load(self, ckpt_id: CkptID) -> Tuple[TensorAwareStateDict, str]:
        """Load of the checkpoint identified by ckpt_id."""
        local_ckpt_path = self._local_ckpt_path_from_id(ckpt_id)
        try:
            # Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
            # Severity: Medium   Confidence: High
            # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
            # More Info: https://bandit.readthedocs.io/en/latest/plugins/b614_pytorch_load_save.html
            return torch.load(local_ckpt_path, weights_only=False)  # nosec
        except FileNotFoundError as e:
            err_msg = f"File {local_ckpt_path} does not exist!"
            logging.info(err_msg)
            ckpt_files = [f.name for f in self.local_ckpt_dir.iterdir()]
            logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
            raise CheckpointingException(err_msg) from e

    @debug_time('LocalCheckpointManager._save', logger)
    def _save(self, state_dict: TensorAwareStateDict, ckpt_id: CkptID):
        """Save of the tensor_aware_state_dict identified by ckpt_id."""
        self._ensure_dir()
        save_path = self._local_ckpt_path_from_id(ckpt_id, True)
        assert ".dirty" in save_path.suffixes
        try:
            logging.info(f"Saving to {str(save_path)}")
            # Open file for exclusive access.
            # Fail if already exists.
            with open(save_path, "bx") as save_file:
                # Issue: [B614:pytorch_load_save] Use of unsafe PyTorch load or save
                # Severity: Medium   Confidence: High
                # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
                # More Info: https://bandit.readthedocs.io/en/latest/plugins/b614_pytorch_load_save.html
                torch.save(state_dict, save_file)  # nosec
            final_path = self._local_ckpt_path_from_id(ckpt_id, False)
            logging.info(f"Renaming {str(save_path)} to {final_path}")
            save_path.rename(target=final_path)

        except FileExistsError as e:
            ckpt_files = [f.name for f in self.local_ckpt_dir.iterdir()]
            logger.debug(f'Checkpoint directory content: {ckpt_files}')
            raise SameMachineReplicationException(ckpt_id) from e

    @debug_time('LocalCheckpointManager._cleanup', logger)
    def _cleanup(self, iteration):
        """Removes outdated or invalid checkpoints after successfully saving the checkpoint
        for the specified iteration.

        Args:
            iteration : The global iteration number for which the checkpoint was successfully saved
        """
        ckpts = self.local_ckpt_dir.glob(self._filename_from_template('*', '*', '*'))
        rm_ckpts = [ckpt for ckpt in ckpts if self._filename_to_id(ckpt.name)[0] < iteration]
        for ckpt in rm_ckpts:
            logging.info(f"Removing {ckpt}")
            ckpt.unlink()

    @debug_time('LocalCheckpointManager._cleanup_failed_save', logger)
    def _cleanup_failed_save(self, iteration):
        """Removes invalid checkpoints that could not be saved due to a failure.

        Args:
            iteration : The global iteration number for which the checkpoint failed to save.
        """
        rm_ckpts = self.local_ckpt_dir.glob(self._filename_from_template(iteration, '*', '*'))
        for ckpt in rm_ckpts:
            logging.info(f"Removing {ckpt}")
            ckpt.unlink()

    def _filename_from_template(
        self, iteration: Union[int, str], rank: Union[int, str], extra_suffix: str = ""
    ):
        digits = 7
        iteration_string = str(iteration).zfill(digits) if isinstance(iteration, int) else iteration
        if iteration_string.isdigit():
            assert len(iteration_string) == digits
        file_name = f"iter_{iteration_string}_{rank}_local{extra_suffix}.pt"
        return file_name

    def _local_ckpt_path_from_id(self, ckpt_id, is_dirty=False):
        iteration, rank, session_id = ckpt_id
        assert session_id == self.session_id
        suffix = ".dirty" if is_dirty else ""
        file_name = self._filename_from_template(iteration, rank, suffix)
        return self.local_ckpt_dir / file_name

    def _filename_to_id(self, filename):
        _, iteration, rank, _ = filename.split('_', 3)
        return (int(iteration), int(rank), self.session_id)
