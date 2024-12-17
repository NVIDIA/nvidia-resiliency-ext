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
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from .test_utilities import Utils
from . import TempNamedDir

from nvidia_resiliency_ext.checkpointing.local.base_state_dict import TensorAwareStateDict
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)


# from typing import Any, Callable, Tuple, Union


class SimpleTensorAwareStateDict(TensorAwareStateDict):
    def __init__(self, iteration):
        self._tensors = [torch.empty((1000, 1000), device='cuda').random_() for _ in range(100)]
        self.iteration = iteration

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
        for i, ten in enumerate(self._tensors):
            self._tensors[i] = ten.to("cpu")

    def restore_tensor_device(self, non_blocking=False):
        for i, ten in enumerate(self._tensors):
            self._tensors[i] = ten.to("cuda")

    def to_state_dict(self):
        raise NotImplementedError

    def __eq__(self, other):
        if len(self._tensors) != len(other._tensors):
            return False
        for self_ten, other_ten in zip(self._tensors, other._tensors):
            if not torch.equal(self_ten, other_ten):
                return False
        return self.iteration == other.iteration


class TestLocalCheckpointing:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        pass

    def _async_save(self, async_save_request, async_save):
        if async_save:
            async_save_request.execute_sync()
        else:
            assert async_save_request == None

    @pytest.mark.parametrize(('use_ramdisk'), [True, False])
    @pytest.mark.parametrize(('async_save'), [True, False])
    def test_basic_save_load_scenarios(self, tmp_path_dist_ckpt, use_ramdisk, async_save):
        if use_ramdisk:
            tmp_path_dist_ckpt = Path("/dev/shm")
        with TempNamedDir(tmp_path_dist_ckpt / "test_save_load") as local_ckpt_dir:
            local_ckpt_dir = local_ckpt_dir / "subdir"  # Test handling of non-existent directories

            # "Without restart"
            checkpoint_manager = LocalCheckpointManager(local_ckpt_dir)
            intermediete_state_dict = SimpleTensorAwareStateDict(iteration=1)
            # SAVE
            async_save_request = checkpoint_manager.save(intermediete_state_dict, 1, async_save)
            self._async_save(async_save_request, async_save)
            # LOAD
            iteration = checkpoint_manager.find_latest()
            assert iteration == 1
            loaded_state_dict, ckpt_id = checkpoint_manager.load()
            intermediete_state_dict.restore_tensor_device()
            assert loaded_state_dict == intermediete_state_dict
            assert ckpt_id == (1, dist.get_rank(), '')

            # "Succesfull load after restart"
            checkpoint_manager = LocalCheckpointManager(local_ckpt_dir)
            # LOAD
            iteration = checkpoint_manager.find_latest()
            assert iteration == 1
            loaded_state_dict, ckpt_id = checkpoint_manager.load()
            assert loaded_state_dict == intermediete_state_dict
            assert ckpt_id == (1, dist.get_rank(), '')

            # "Failed load after restart"
            checkpoint_manager = LocalCheckpointManager(local_ckpt_dir)
            dist.barrier()
            ckpt_id = checkpoint_manager._ckpt_id(iteration)
            first_ckpt_path = checkpoint_manager._local_ckpt_path_from_id(ckpt_id)
            os.remove(first_ckpt_path)
            # LOAD
            iteration = checkpoint_manager.find_latest()
            assert iteration == -1

            # "Multiple saves"
            intermediete_state_dict = SimpleTensorAwareStateDict(iteration=1)
            # SAVE
            async_save_request = checkpoint_manager.save(intermediete_state_dict, 1, async_save)
            self._async_save(async_save_request, async_save)
            intermediete_state_dict = SimpleTensorAwareStateDict(iteration=2)
            # SAVE
            async_save_request = checkpoint_manager.save(intermediete_state_dict, 2, async_save)
            self._async_save(async_save_request, async_save)
            assert not first_ckpt_path.exists()
            ckpt_id = checkpoint_manager._ckpt_id(2)
            second_ckpt_path = checkpoint_manager._local_ckpt_path_from_id(ckpt_id)
            assert second_ckpt_path.exists()
