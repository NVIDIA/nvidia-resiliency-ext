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
from nvidia_resiliency_ext.device_utils import get_current_device
import torch
import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt import TorchAsyncCheckpoint

from .test_utilities import Utils, TestModel
from . import TempNamedDir

class TestAsyncSave:
    def setup_method(self, method):
        pass

    def test_async_is_equivalent_to_sync(self, tmp_path_dist_ckpt):
        Utils.initialize_distributed()
        model = TestModel((1024,1024), 10)
        ckpt_impl = TorchAsyncCheckpoint()
        state_dict = model.state_dict()
        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_equivalence_async'
        ) as async_ckpt_dir, TempNamedDir(
            tmp_path_dist_ckpt / 'test_equivalence_sync'
        ) as sync_ckpt_dir:
            # async
            ckpt_impl.async_save(state_dict, async_ckpt_dir/'test')

            # sync
            ckpt_impl.save(state_dict, sync_ckpt_dir/'test')

            # finalize async
            ckpt_impl.finalize_async_save(blocking=True)

            # load and compare
            device = get_current_device()
            loaded_async_state_dict = torch.load(async_ckpt_dir/'test', map_location=device)
            loaded_sync_state_dict = torch.load(sync_ckpt_dir/'test', map_location=device)
            for k in loaded_sync_state_dict.keys():
                assert k in loaded_async_state_dict, f"{k} is not in loaded async state_dict"
                assert torch.equal(loaded_async_state_dict[k], loaded_sync_state_dict[k]), f"loaded_async_state_dict[{k}] != loaded_sync_state_dict[{k}]"
                assert torch.equal(loaded_async_state_dict[k], state_dict[k]), f"loaded_async_state_dict[{k}] != src_state_dict[{k}]"
