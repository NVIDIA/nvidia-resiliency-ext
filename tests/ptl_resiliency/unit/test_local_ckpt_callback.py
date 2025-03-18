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
from datetime import timedelta
from typing import Any, Dict, Iterable

from nvidia_resiliency_ext.ptl_resiliency._utils import is_module_available

if is_module_available("lightning"):
    import lightning.pytorch as pl
elif is_module_available("pytorch_lightning"):
    import pytorch_lightning as pl
else:
    raise ImportError("Could not find 'lightning' or 'pytorch_lightning' module")

import torch

from nvidia_resiliency_ext.checkpointing.local.base_state_dict import TensorAwareStateDict
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.base_manager import (
    BaseCheckpointManager,
    CkptID,
)
from nvidia_resiliency_ext.fault_tolerance.dict_utils import dict_list_map_outplace
from nvidia_resiliency_ext.ptl_resiliency.local_checkpoint_callback import (
    HierarchicalCheckpointIO,
    LocalCheckpointCallback,
)

from .test_ft_callback_hb import SimpleModel


def _run_trainining(
    max_steps=100_000,
    max_epochs=3,
    max_time=None,
    val_check_interval=None,
    custom_callbacks=None,
    custom_checkpoint_io_fn=None,
):

    custom_callbacks = custom_callbacks if custom_callbacks else []

    trainer = pl.Trainer(
        strategy='ddp',
        devices=1,
        accelerator='gpu',
        logger=False,
        max_steps=max_steps,
        max_epochs=max_epochs,
        max_time=max_time,
        val_check_interval=val_check_interval,
        callbacks=custom_callbacks,
    )
    if custom_checkpoint_io_fn is not None:
        trainer.strategy.checkpoint_io = custom_checkpoint_io_fn(trainer.strategy.checkpoint_io)

    model = SimpleModel()
    trainer.fit(model, ckpt_path='last')
    return trainer


def inspect_checkpoints(dirpath, expected_paths: Iterable[str] = None, verbose: bool = True):
    actual_paths = set(d.name for d in dirpath.iterdir())
    if verbose:
        print('Actual paths', actual_paths)

    if expected_paths is not None:
        assert actual_paths == set(expected_paths)


class NonDistributedInMemoryCheckpointManager(BaseCheckpointManager):
    """Simple implementation for a non-distributed in-memory local checkpoint."""

    def __init__(self, session_id, repl_strategy=None):
        super().__init__(session_id, repl_strategy)
        self.last_ckpt_id = None
        self.last_state_dict = None

    def _my_ckpt_ids(self) -> Iterable[CkptID]:
        return [self.last_ckpt_id]

    def _load(self, ckpt_id: CkptID):
        assert self.last_state_dict is not None
        return self.last_state_dict

    def _save(self, state_dict: TensorAwareStateDict, ckpt_id: CkptID):
        self.last_state_dict = state_dict
        self.last_ckpt_id = ckpt_id

    def _cleanup(self, iteration):
        pass


class MockLocalOnlySaveCheckpointManager(NonDistributedInMemoryCheckpointManager):
    """Mock manager that allows to track save calls and control the available iterations.

    Assumes that self.latest_iteration is set externally.
    """

    def __init__(self, session_id='mock', latest_iteration=-1):
        super().__init__(session_id, repl_strategy=None)
        self.mock_save_calls = []
        self._latest_iteration = latest_iteration

    def find_latest(self):
        self.latest_iteration = self._latest_iteration
        return self.latest_iteration

    def _save(self, state_dict: TensorAwareStateDict, ckpt_id: CkptID):
        self.mock_save_calls.append(ckpt_id)
        self._latest_iteration = ckpt_id[0]
        super()._save(state_dict, ckpt_id)

    def _cleanup_failed_save(self, iteration):
        return self._cleanup(iteration)


class SimpleTensorAwareStateDict(TensorAwareStateDict):
    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.orig_device = None

    def pop_tensors(self):
        raise NotImplementedError

    @property
    def tensors(self):
        raise NotImplementedError

    def tensors_to_orig_device(self):
        raise NotImplementedError

    def is_hollow(self) -> bool:
        raise NotImplementedError

    def insert_tensors(self, tensor_data):
        raise NotImplementedError

    def init_tensors(self):
        raise NotImplementedError

    def copy_tensors_to_cpu(self, non_blocking=False):
        def _to_cpu(x):
            if isinstance(x, torch.Tensor):
                orig_device = x.device.type
                if self.orig_device is None:
                    self.orig_device = orig_device
                else:
                    if orig_device != self.orig_device:
                        print(
                            f'WARNING: Orig device set to {self.orig_device},'
                            f' but encountered {orig_device} tensor. Setting self.orig_device=cpu'
                        )
                        self.orig_device = 'cpu'
                x = x.to("cpu")
            return x

        dict_list_map_outplace(_to_cpu, self.state_dict)

    def restore_tensor_device(self, non_blocking=False):
        assert self.orig_device is not None

        def _to_orig_device(x):
            if isinstance(x, torch.Tensor):
                x = x.to(self.orig_device)
            return x

        dict_list_map_outplace(_to_orig_device, self.state_dict)

    def to_state_dict(self):
        return self.state_dict


class SimpleHierarchicalCheckpointIO(HierarchicalCheckpointIO):

    def to_tensor_aware_state_dict(self, checkpoint: Dict[str, Any]) -> TensorAwareStateDict:
        return SimpleTensorAwareStateDict(checkpoint)

    def from_tensor_aware_state_dict(self, tensor_aware_checkpoint: SimpleTensorAwareStateDict):
        return tensor_aware_checkpoint.to_state_dict()


def test_local_ckpt_callback_called_every_n_train_steps(tmp_path):
    """Test if a local checkpoint is saved every 4 iterations."""
    local_every_n_train_steps = 4
    global_every_n_train_steps = 32
    max_steps = 128

    # Global checkpoints
    global_ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        every_n_train_steps=global_every_n_train_steps,
    )

    # Local checkpoints
    local_ckpt_callback = LocalCheckpointCallback(
        every_n_train_steps=local_every_n_train_steps,
    )

    mock_local_ckpt_manager = MockLocalOnlySaveCheckpointManager(
        'test_local_ckpt_callback_every_n_train_steps'
    )
    hier_ckpt_io_fn = SimpleHierarchicalCheckpointIO.get_partial_wrapper_constructor(
        mock_local_ckpt_manager,
        lambda s: 1000,
    )

    _run_trainining(
        max_steps=max_steps,
        custom_callbacks=[global_ckpt_callback, local_ckpt_callback],
        custom_checkpoint_io_fn=hier_ckpt_io_fn,
    )

    # the intermediate checkpoints are erased
    inspect_checkpoints(
        tmp_path, expected_paths={'last.ckpt', 'epoch=0-step=128.ckpt'}, verbose=False
    )
    assert len(mock_local_ckpt_manager.mock_save_calls) == max_steps // local_every_n_train_steps


def test_local_ckpt_callback_called_every_time_interval(tmp_path):
    """Test if a local checkpoint is saved every 0.2 iterations."""
    local_train_time_interval = timedelta(seconds=0.2)
    global_train_time_interval = timedelta(seconds=2)
    max_time = {'seconds': 5}

    # Global checkpoints
    global_ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        train_time_interval=global_train_time_interval,
    )

    # Local checkpoints
    local_ckpt_callback = LocalCheckpointCallback(
        train_time_interval=local_train_time_interval,
    )

    mock_local_ckpt_manager = MockLocalOnlySaveCheckpointManager(
        'test_local_ckpt_callback_every_n_train_steps'
    )
    hier_ckpt_io_fn = SimpleHierarchicalCheckpointIO.get_partial_wrapper_constructor(
        mock_local_ckpt_manager,
        lambda s: 1000,
    )

    trainer = _run_trainining(
        max_time=max_time,
        custom_callbacks=[global_ckpt_callback, local_ckpt_callback],
        custom_checkpoint_io_fn=hier_ckpt_io_fn,
    )

    # In theory there should be 25 checkpoints, but in practice there are overheads
    # and the checkpoint interval is sligthly larger than 0.2s
    assert 10 < len(mock_local_ckpt_manager.mock_save_calls) < 30


def test_local_ckpt_restoration(tmp_path, caplog):
    """Test if appropriate checkpoint is used for restoration."""
    local_every_n_train_steps = 4
    global_every_n_train_steps = 32
    max_steps = 128

    # Global checkpoints
    global_ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
        every_n_train_steps=global_every_n_train_steps,
    )

    # Local checkpoints
    local_ckpt_callback = LocalCheckpointCallback(
        every_n_train_steps=local_every_n_train_steps,
    )

    mock_local_ckpt_manager = MockLocalOnlySaveCheckpointManager(
        'test_local_ckpt_callback_every_n_train_steps',
    )
    hier_ckpt_io_fn = SimpleHierarchicalCheckpointIO.get_partial_wrapper_constructor(
        mock_local_ckpt_manager,
        lambda s: 1000,
    )

    trainer = _run_trainining(
        max_steps=max_steps,
        custom_callbacks=[global_ckpt_callback, local_ckpt_callback],
        custom_checkpoint_io_fn=hier_ckpt_io_fn,
    )
    assert trainer.global_step == max_steps

    # the intermediate checkpoints are erased
    inspect_checkpoints(
        tmp_path, expected_paths={'last.ckpt', 'epoch=0-step=128.ckpt'}, verbose=False
    )
    assert len(mock_local_ckpt_manager.mock_save_calls) == max_steps // local_every_n_train_steps

    # Make sure we load from a global ckpt when global iter is 1000 (large)
    with caplog.at_level(logging.INFO):
        trainer = _run_trainining(
            max_steps=max_steps * 2,
            custom_callbacks=[global_ckpt_callback, local_ckpt_callback],
            custom_checkpoint_io_fn=hier_ckpt_io_fn,
        )
        assert trainer.global_step == max_steps * 2

    assert 'Resuming from a global checkpoint' in caplog.text

    # Make sure we load from a local ckpt when local iter is 256 (larger than 100)
    hier_ckpt_io_fn = SimpleHierarchicalCheckpointIO.get_partial_wrapper_constructor(
        mock_local_ckpt_manager,
        lambda s: 100,
    )
    with caplog.at_level(logging.INFO):
        trainer = _run_trainining(
            max_steps=max_steps * 3,
            custom_callbacks=[global_ckpt_callback, local_ckpt_callback],
            custom_checkpoint_io_fn=hier_ckpt_io_fn,
        )

        assert trainer.global_step == max_steps * 3

    assert 'Resuming from a local checkpoint' in caplog.text

    # Same iteration as global (384), still load from local ckpt
    hier_ckpt_io_fn = SimpleHierarchicalCheckpointIO.get_partial_wrapper_constructor(
        mock_local_ckpt_manager,
        lambda s: 384,
    )
    with caplog.at_level(logging.INFO):
        trainer = _run_trainining(
            max_steps=max_steps * 4,
            custom_callbacks=[global_ckpt_callback, local_ckpt_callback],
            custom_checkpoint_io_fn=hier_ckpt_io_fn,
        )

        assert trainer.global_step == max_steps * 4

    assert 'Resuming from a local checkpoint' in caplog.text


# TODO
# def test_async_request_is_returned_for_async_save():
#     pass

# TODO
# def test_wrapped_checkpoint_io_is_not_called():
#     pass
