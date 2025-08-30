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
import torch

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import abort_nvrx_checkpoint
from nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt import TorchAsyncCheckpoint

from . import TempNamedDir
from .test_utilities import TestModel, Utils


class TestAsyncSave:
    def setup_method(self, method):
        Utils.set_world_size(1)

    def teardown_method(self, method):
        Utils.set_world_size()

    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt):
        Utils.initialize_distributed()
        model = TestModel((1024, 1024), 10)
        ckpt_impl = TorchAsyncCheckpoint()
        state_dict = model.state_dict()
        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_async') as async_ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_sync') as sync_ckpt_dir,
        ):
            # async
            ckpt_impl.async_save(state_dict, async_ckpt_dir / 'test')

            # sync
            ckpt_impl.save(state_dict, sync_ckpt_dir / 'test')

            # finalize async
            ckpt_impl.finalize_async_save(blocking=True)

            # load and compare
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            loaded_async_state_dict = torch.load(async_ckpt_dir / 'test', map_location=device)
            loaded_sync_state_dict = torch.load(sync_ckpt_dir / 'test', map_location=device)
            for k in loaded_sync_state_dict.keys():
                assert k in loaded_async_state_dict, f"{k} is not in loaded async state_dict"
                assert torch.equal(
                    loaded_async_state_dict[k], loaded_sync_state_dict[k]
                ), f"loaded_async_state_dict[{k}] != loaded_sync_state_dict[{k}]"
                assert torch.equal(
                    loaded_async_state_dict[k], state_dict[k]
                ), f"loaded_async_state_dict[{k}] != src_state_dict[{k}]"
        ckpt_impl.close()

    def test_persistent_async_cp_abort(self, tmp_path_dist_ckpt):
        Utils.initialize_distributed()
        model = TestModel((1024, 1024), 10)
        ckpt_impl = TorchAsyncCheckpoint(persistent_queue=True)
        state_dict = model.state_dict()

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_async') as async_ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_sync') as sync_ckpt_dir,
        ):
            # Save Sync CP state for reference
            ckpt_impl.save(state_dict, sync_ckpt_dir / 'test')

            # Save and finalize  async CP
            ckpt_impl.async_save(state_dict, async_ckpt_dir / 'test')
            ckpt_impl.finalize_async_save(blocking=True)

            # Validate that NVRx CP workers are initialized
            async_calls_queue = ckpt_impl._get_async_calls_queue()
            assert async_calls_queue is not None, "After saving async CP, we expect valid object"
            async_process = async_calls_queue._get_async_caller()
            assert (
                async_process is not None
            ), "After a valid CP save, we expect async process to be running"
            assert async_process._debug_is_async_process_running(), "Valid async process expected"

            # Abort the CP workers to mock the action of inprocess restarts
            abort_nvrx_checkpoint()

            # Validate clean-up of NVrx CP workers is done
            async_calls_queue = ckpt_impl._get_async_calls_queue()
            assert async_calls_queue is not None, "We expect a valid state of AsyncCallsQueue"
            async_process = async_calls_queue._get_async_caller()
            if async_process is not None:
                assert (
                    async_process._debug_is_async_process_running() is False
                ), "After abort async process stops"

            # Re-start CP process by doing another async CP state.
            ckpt_impl.async_save(state_dict, async_ckpt_dir / 'test')
            ckpt_impl.finalize_async_save(blocking=True)

            # Validate that NVRx CP workers are initialized
            async_calls_queue = ckpt_impl._get_async_calls_queue()
            assert async_calls_queue is not None, "After saving async CP, we expect valid object"
            async_process = async_calls_queue._get_async_caller()
            assert (
                async_process is not None
            ), "After a valid CP save, we expect async process to be running"
            assert async_process._debug_is_async_process_running(), "Valid async process expected"

            # load and compare the re-started async-cp state with the reference sync CP
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            loaded_async_state_dict = torch.load(async_ckpt_dir / 'test', map_location=device)
            loaded_sync_state_dict = torch.load(sync_ckpt_dir / 'test', map_location=device)
            for k in loaded_sync_state_dict.keys():
                assert k in loaded_async_state_dict, f"{k} is not in loaded async state_dict"
                assert torch.equal(
                    loaded_async_state_dict[k], loaded_sync_state_dict[k]
                ), f"loaded_async_state_dict[{k}] != loaded_sync_state_dict[{k}]"
                assert torch.equal(
                    loaded_async_state_dict[k], state_dict[k]
                ), f"loaded_async_state_dict[{k}] != src_state_dict[{k}]"
        ckpt_impl.close()

    def test_persistent_async_cp_abort_during_cp_ops(self, tmp_path_dist_ckpt):
        Utils.initialize_distributed()
        model = TestModel((1024, 1024), 10)
        ckpt_impl = TorchAsyncCheckpoint(persistent_queue=True)
        state_dict = model.state_dict()

        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_async') as async_ckpt_dir,
            TempNamedDir(tmp_path_dist_ckpt / 'test_equivalence_sync') as sync_ckpt_dir,
        ):
            # Save Sync CP state for reference
            ckpt_impl.save(state_dict, sync_ckpt_dir / 'test')

            # Save and finalize  async CP
            ckpt_impl.async_save(state_dict, async_ckpt_dir / 'test')

            # Validate that NVRx CP workers are initialized
            async_calls_queue = ckpt_impl._get_async_calls_queue()
            assert async_calls_queue is not None, "After saving async CP, we expect valid object"
            async_process = async_calls_queue._get_async_caller()
            assert (
                async_process is not None
            ), "After a valid CP save, we expect async process to be running"
            assert async_process._debug_is_async_process_running(), "Valid async process expected"

            # Abort the CP workers to mock the action of inprocess restarts
            # Note that the previous async CP operation has not been finalized
            # This is to test abort during ongoing async CP operations
            abort_nvrx_checkpoint()

            # Validate clean-up of NVrx CP workers is done
            async_calls_queue = ckpt_impl._get_async_calls_queue()
            assert async_calls_queue is not None, "We expect a valid state of AsyncCallsQueue"
            async_process = async_calls_queue._get_async_caller()
            if async_process is not None:
                assert (
                    async_process._debug_is_async_process_running() is False
                ), "After abort async process stops"

            # Re-start CP process by doing another async CP state.
            ckpt_impl.async_save(state_dict, async_ckpt_dir / 'test')
            ckpt_impl.finalize_async_save(blocking=True)

            # Validate that NVRx CP workers are initialized
            async_calls_queue = ckpt_impl._get_async_calls_queue()
            assert async_calls_queue is not None, "After saving async CP, we expect valid object"
            async_process = async_calls_queue._get_async_caller()
            assert (
                async_process is not None
            ), "After a valid CP save, we expect async process to be running"
            assert async_process._debug_is_async_process_running(), "Valid async process expected"

            # load and compare the re-started async-cp state with the reference sync CP
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            loaded_async_state_dict = torch.load(async_ckpt_dir / 'test', map_location=device)
            loaded_sync_state_dict = torch.load(sync_ckpt_dir / 'test', map_location=device)
            for k in loaded_sync_state_dict.keys():
                assert k in loaded_async_state_dict, f"{k} is not in loaded async state_dict"
                assert torch.equal(
                    loaded_async_state_dict[k], loaded_sync_state_dict[k]
                ), f"loaded_async_state_dict[{k}] != loaded_sync_state_dict[{k}]"
                assert torch.equal(
                    loaded_async_state_dict[k], state_dict[k]
                ), f"loaded_async_state_dict[{k}] != src_state_dict[{k}]"
        ckpt_impl.close()
