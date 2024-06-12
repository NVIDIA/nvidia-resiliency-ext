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

import logging
from abc import abstractmethod
from datetime import timedelta
from functools import partial
from typing import Any, Callable, Dict, NewType, Optional

from ..checkpointing.async_ckpt.core import AsyncRequest
from ._utils import is_module_available

if is_module_available("lightning"):
    import lightning.pytorch as pl
    from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
elif is_module_available("pytorch_lightning"):
    import pytorch_lightning as pl
    from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
else:
    raise ImportError("Could not find 'lightning' or 'pytorch_lightning' module")


from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.utilities.types import _PATH
from torch import Tensor

from nvidia_resiliency_ext.checkpointing.local.base_state_dict import TensorAwareStateDict
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.base_manager import (
    BaseCheckpointManager,
)

logger = logging.getLogger(__name__)

StateDict = NewType('StateDict', Any)

LOCAL_CKPT_OPTS_KEY = 'local_checkpoint_options'


class LocalCheckpointCallback(pl.callbacks.ModelCheckpoint):
    """ModelCheckpoint with basic functionality. Only train_batch_end simple save.

    Simple callback for initiating local checkpoint save in `on_train_batch_end` method.
    Since local checkpoints are ephemeral, they shouldn't be used for "major" checkpoint
    types like `on_train_epoch_end`.

    This callback must be used in conjunction with the HierarchicalCheckpointIO,
    since the only thing this callback really does is passing some options
    to `trainer.save_checkpoint` which can be captured with HierarchicalCheckpointIO.

    Args:
        every_n_train_steps (int, optional): controls local checkpointing interval in terms
            of train iterations. Same semantic as in PTL ModelCheckpoint.
        train_time_interval (int, optional): controls local checkpointing interval in terms
            of wall time. Same semantics as in PTL ModelCheckpoint.
    """

    def __init__(
        self,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
    ):
        super().__init__(
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Skips super functionality"""
        logger.info('Skipping on_train_epoch_end local ckpt save')

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Skips super functionality"""
        logger.info('Skipping on_validation_end local ckpt save')

    def _save_topk_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        """Skips super functionality"""
        logger.info('Skipping _save_topk_checkpoint local ckpt save')

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        """Simply saves a local checkpoint with appropriate storage_options."""
        local_ckpt_opts = dict(ckpt_type='local', iteration=trainer.global_step)
        trainer.save_checkpoint(None, storage_options={LOCAL_CKPT_OPTS_KEY: local_ckpt_opts})


class HierarchicalCheckpointIO(_WrappingCheckpointIO):
    """Wrapper for a global CheckpointIO enabling local checkpointing.

    Based on the presence of local checkpointing options in saving `storage_options`,
    routes the save to the original global CheckpointIO or the local checkpoint manager.

    Must be used in conjunction with LocalCheckpointCallback which *initiates*
    local checkpoint saving during training.

    Args:
        wrapped_checkpoint_io (CheckpointIO): global CheckpointIO to wrap
        local_ckpt_manager (BaseCheckpointManager): local manager to use for local checkpoints
        get_global_ckpt_iteration_fn (Callable[[_PATH], int]): a function that
            given a path to a global checkpoint, extracts the global step iteration from it
            (either from the path itself or by loading metadata from the checkpoint).
        async_save (bool, optional): enables asynchronous save. Passed down to the local checkpoint
            manager unless overriden with `local_ckpt_options` in `_save_local_checkpoint`.
    """

    def __init__(
        self,
        wrapped_checkpoint_io: CheckpointIO,
        local_ckpt_manager: BaseCheckpointManager,
        get_global_ckpt_iteration_fn: Callable[[_PATH], int],
        async_save: bool = False,
    ):
        super().__init__(wrapped_checkpoint_io)
        self.local_ckpt_manager = local_ckpt_manager
        self.get_global_ckpt_iteration_fn = get_global_ckpt_iteration_fn
        self.async_save = async_save

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None
    ) -> Optional[AsyncRequest]:
        """Save local or global checkpoint, depending on the presence of options."""
        if storage_options is None or LOCAL_CKPT_OPTS_KEY not in storage_options:
            return self.checkpoint_io.save_checkpoint(checkpoint, path, storage_options)
        if path is not None:
            raise ValueError(f'Path shouldn\'t be set for a local checkpoint, got: {path}.')
        return self._save_local_checkpoint(checkpoint, storage_options.get(LOCAL_CKPT_OPTS_KEY))

    def _save_local_checkpoint(
        self, checkpoint: Dict[str, Any], local_ckpt_options: dict
    ) -> Optional[AsyncRequest]:
        """Save local checkpoint."""
        return self.local_ckpt_manager.save(
            self.to_tensor_aware_state_dict(checkpoint),
            local_ckpt_options['iteration'],
            is_async=local_ckpt_options.get('is_async', self.async_save),
        )

    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Load the newer of local (if available) and global checkpoint."""
        latest_local_iteration = self.local_ckpt_manager.find_latest()
        if latest_local_iteration < 0:
            logger.debug('No local checkpoint available')
            return self.checkpoint_io.load_checkpoint(path, map_location=map_location, **kwargs)

        # There is a local ckpt available, but we don't know if it's newer than the global ckpt yet
        latest_global_iteration = self.get_global_ckpt_iteration_fn(path)
        if latest_local_iteration >= latest_global_iteration:
            logger.info(
                f'Local checkpoint interation {latest_local_iteration} greater than'
                f' global {latest_global_iteration}.'
                f' Resuming from a local checkpoint'
            )
            intermediate_state_dict, checkpoint_name = self.local_ckpt_manager.load()
            logger.debug(f'Loaded local checkpoint {checkpoint_name}')
            return self.from_tensor_aware_state_dict(intermediate_state_dict, **kwargs)

        else:
            logger.warning(
                f'Found available local checkpoint from interation {latest_local_iteration},'
                f' but global iteration {latest_global_iteration} is greater.'
                f' Resuming from a global checkpoint.'
            )
            return self.checkpoint_io.load_checkpoint(path, map_location=map_location, **kwargs)

    def remove_checkpoint(self, path: _PATH) -> None:
        """Checkpoint removal is handled independently by the LocalCkptManager."""
        return self.checkpoint_io.remove_checkpoint(path)

    @classmethod
    def get_partial_wrapper_constructor(
        cls,
        local_ckpt_manager: BaseCheckpointManager,
        get_global_ckpt_iteration_fn: Callable[[_PATH], int],
    ):
        """Allows to provide all arguments to the constructor except for the wrapped checkpoint io."""
        return partial(
            cls,
            local_ckpt_manager=local_ckpt_manager,
            get_global_ckpt_iteration_fn=get_global_ckpt_iteration_fn,
        )

    @abstractmethod
    def to_tensor_aware_state_dict(self, checkpoint: Dict[str, Any]) -> TensorAwareStateDict:
        raise NotImplementedError

    @abstractmethod
    def from_tensor_aware_state_dict(self, tensor_aware_checkpoint: TensorAwareStateDict, **kwargs):
        raise NotImplementedError
