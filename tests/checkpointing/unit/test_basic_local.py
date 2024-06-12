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
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch.distributed as dist

from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)

from . import TempNamedDir
from .test_utilities import SimpleTensorAwareStateDict, Utils


class TestLocalCheckpointing:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        pass

    def _async_save(self, async_save_request, async_save):
        if async_save:
            async_save_request.execute_sync()
        else:
            assert async_save_request is None

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
            assert first_ckpt_path.exists()
            intermediete_state_dict = SimpleTensorAwareStateDict(iteration=2)
            # SAVE
            async_save_request = checkpoint_manager.save(intermediete_state_dict, 2, async_save)
            self._async_save(async_save_request, async_save)
            time.sleep(0.4)
            assert not first_ckpt_path.exists()
            ckpt_id = checkpoint_manager._ckpt_id(2)
            second_ckpt_path = checkpoint_manager._local_ckpt_path_from_id(ckpt_id)
            assert second_ckpt_path.exists()

    @contextmanager
    def find_latest_skeleton(self, tmp_path_dist_ckpt, extra_suffix, repl_strategy=None):
        with TempNamedDir(
            name=tmp_path_dist_ckpt / "test_find_latest_disabled_repl"
        ) as root_local_ckpt_dir:
            checkpoint_manager = LocalCheckpointManager(
                root_local_ckpt_dir, repl_strategy=repl_strategy
            )
            my_local_ckpt_subdir = Path(checkpoint_manager.local_ckpt_dir)
            my_local_ckpt_subdir.mkdir(parents=True, exist_ok=True)

            ckpt_filenames = [
                checkpoint_manager._filename_from_template(10, i, extra_suffix)
                for i in range(dist.get_world_size())
            ]
            ckpt_files = [my_local_ckpt_subdir / filename for filename in ckpt_filenames]

            yield checkpoint_manager, ckpt_files

    @pytest.mark.parametrize(('extra_suffix'), ["", "some_suffix", "some suffix with spaces"])
    def test_find_latest_repl_disable(self, tmp_path_dist_ckpt, extra_suffix):
        assert (
            dist.get_world_size() >= 2
        ), f"This test needs world_size >= 2, got {dist.get_world_size()}"
        with self.find_latest_skeleton(tmp_path_dist_ckpt, extra_suffix, repl_strategy=None) as (
            checkpoint_manager,
            ckpt_files,
        ):
            my_rank = dist.get_rank()
            # rank 0: []
            # rank 1: [ckpt_0, ckpt_1]
            # rank i: [ckpt_i] for i >= 2
            if my_rank == 1:
                ckpt_files[0].touch()
            if my_rank != 0:
                ckpt_files[my_rank].touch()

            assert (
                checkpoint_manager.find_latest() == -1
            ), "It's impossible to retrieve ckpt 0 with replication disabled!"
