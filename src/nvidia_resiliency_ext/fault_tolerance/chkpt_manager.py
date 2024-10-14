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

from enum import Enum
from typing import Dict, Sequence

import torch

from .dict_utils import dict_list_map_inplace, dict_list_map_outplace

"""
In-memory checkpointing assumptions
====================================
* Rank state is a dictionary containing built-in Python types, custom objects, and tensors.
* Tensors included in the state can be hosted on any device (CPU, GPU).
* Objects included in the state should be pickleable and should not contain tensors.
* Rank state snapshot contains:
    - Copy of all non-tensor items from the original state.
    - CPU copies of the tensors from the original state.
* A tensor snapshot can be shared with the monitor process via shared (CPU) memory.
* Non-tensors should be pickled and sent via IPC (e.g. pipe).
* Basic internal API for state sharing (rank->monitor) is `send_heartbeat(snapshot)`.
* Monitor process merges the incoming state snapshot with the existing one: 
    - If there are new keys in the incoming dict, they are added.
    - If both keys exist, the value is replaced.
  Otherwise no change is made.

Checkpointing async algorithm outline
====================================

Algorithm uses 3 state snapshots:
* Unfinished/dirty snapshot for the current iteration ITER.
* Completed snapshot for the iteration ITER-1.
* Completed snapshot for the iteration ITER-2.
The snapshots are organized and updated in a rolling buffer manner.
We also need to track iterations that are currently held in the snapshots buffer.

Let:
* `rb` be the snapshots rolling buffer, with the capacity = 3.
* `head_idx` be the index of the unfinished/dirty snapshot in `rb`.
* `iters_in_rb` be the map of indices in `rb` to the iterations.

Now, before the parameter update is issued (e.g. `optimizer.step()`):
* [if not the first iter] wait for any ongoing tensors snapshotting for ITER-1.
* `head_idx = (head_idx+1) % 3`.
* `rb_idx_to_iter[head_idx] = ITER`.
* [if not the first iter] send the heartbeat with the updated state for ITER-1.

After the parameter update is issued:
* Update `rb[head_idx]` with the current state snapshot:
    - All non-tensors should be copied.
    - Tensors snapshotting (D2H) should be started.

When a rank starts, it can query for the in-memory checkpoint:
* Get the state from the monitor process.
* [If non empty] 
    - Gather iterations with completed snapshots from all ranks.
    - Find the last iteration for which all ranks have the completed snapshot (`last_iter_rb_idx`).
    - Restore (move from CPU to the original device) all tensors from `rb[last_iter_rb_idx]`.
    - Return `rb[last_available_iter_idx_in_rb]` and use it to resume work.

Optimizations
=============
* There is no need to send CPU tensor snapshot twice, as these are held in shared memory.
* Tensor snapshotting (D2H) can be done asynchronously (`cudaMemcpyAsync`) and possibly overlap with the forward/backward pass. 
  NOTE: pinned CPU memory is required for async D2H. NOTE: we need to benchmark async D2H with “naive” synchronous D2H, as it is not obvious if there are any benefits.
"""


class CheckpointManagerType(Enum):
    NONE = 0
    ASYNC = 1


class _TensorCpuSnapshot:
    def __init__(self, t=None):
        self.orig_device = None
        self.cpu_tensor = None
        self.mark_for_send = None
        if t is not None:
            self.take_snapshot(t)

    @staticmethod
    def _pin_memory(cpu_tensor):
        # `cudaHostRegister` hack is used to pin the existing CPU tensor memory,
        # as described here https://github.com/pytorch/pytorch/issues/32167
        cudart = torch.cuda.cudart()
        cudart.cudaHostRegister(
            cpu_tensor.data_ptr(),
            cpu_tensor.numel() * cpu_tensor.element_size(),
            0,
        )

    @staticmethod
    def _unpin_memory(cpu_tensor):
        cudart = torch.cuda.cudart()
        cudart.cudaHostUnregister(cpu_tensor.data_ptr())

    def take_snapshot(self, t):
        """
        Copy tensor to shared and pinned CPU memory.
        If it is already on CPU, another CPU copy will be made.
        """
        if (
            self.cpu_tensor is None
            or self.cpu_tensor.shape != t.shape
            or self.cpu_tensor.dtype != t.dtype
        ):
            if self.cpu_tensor is not None and self.orig_device != torch.device('cpu'):
                # unpin memory if we had CUDA tensor snapshot
                self._unpin_memory(self.cpu_tensor)
            self.cpu_tensor = torch.empty_like(t, device='cpu')
            self.cpu_tensor.share_memory_()
            if t.device != torch.device('cpu'):
                # pin memory for async D2H
                self._pin_memory(self.cpu_tensor)
            self.mark_for_send = True
        self.orig_device = t.device
        self.cpu_tensor.copy_(t, non_blocking=True)

    def restore(self):
        """
        Copies the tensor back the original device
        """
        return self.cpu_tensor.to(device=self.orig_device, copy=True)


class InMemCheckpointManagerAsync:
    """
    Checkpointing manager that implements async D2H.
    `prepare_for_state_update`, `after_state_update_issued` are called before and after parameters update,
    and return state snapshot to be sent in the subsequent heartbeat. Hence there can be 2 heartbeats for each training iteration.
    `restore` returns state dict that can be used to resume the training.
    """

    def __init__(self):
        self.rb_capacity = 3
        self.rb_head_idx = -1
        self.rb_curr_len = 0
        # use dict (instead of list) for states rolling buffer,
        # because we want to be able to send just a part of the buffer in a heartbeat
        # e.g. {updated_state_idx: {...}}
        self.rb = {i: {} for i in range(self.rb_capacity)}
        self.iters_in_rb = [None] * self.rb_capacity
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _mark_as_sent_if_tensor_snapshot(self, x):
        if isinstance(x, _TensorCpuSnapshot):
            x.mark_for_send = False
        return x

    def _restore_if_tensor_snapshot(self, x):
        if isinstance(x, _TensorCpuSnapshot):
            x = x.restore()
        return x

    def _snapshot_if_tensor(self, x):
        if torch.is_tensor(x):
            return _TensorCpuSnapshot(x)
        return x

    def _snapshot_recursive(self, x):
        if isinstance(x, (dict, list)):
            return dict_list_map_outplace(self._snapshot_if_tensor, x)
        return x

    def _merge_state_dicts_inplace(self, current, incoming) -> None:
        """
        Merge `incoming` state dict with the given state dict (`current`) inplace.
        If there are new keys in `incoming`, they will be added to the `current`.
        If a key exist in `current` and `incoming` value in `current` is replaced.
        If a value is a dict or list, shallow copy is made recursively.
        If a value is a tensor, it is replaced with `_TensorCpuSnapshot(tensor)`
        """
        for key in incoming:
            if key in current:
                if isinstance(incoming[key], dict) and isinstance(current[key], dict):
                    self._merge_state_dicts_inplace(current[key], incoming[key])
                elif torch.is_tensor(incoming[key]):
                    assert isinstance(current[key], _TensorCpuSnapshot)
                    current[key].take_snapshot(incoming[key])
                else:
                    current[key] = self._snapshot_recursive(incoming[key])
            else:
                current[key] = self._snapshot_recursive(incoming[key])

    def _get_state_part_for_ipc(self, current):
        """
        Extract not-yet-sent tensor snapshots and plain data that needs to be included in the heartbeat.
        Tensors that were sent can be skipped, as they are in shared CPU memory.
        """
        assert isinstance(current, dict)
        res = {}
        for k in current:
            if isinstance(current[k], dict):
                res[k] = self._get_state_part_for_ipc(current[k])
            elif isinstance(current[k], _TensorCpuSnapshot):
                if current[k].mark_for_send:
                    res[k] = current[k]
            else:
                assert not torch.is_tensor(
                    current[k]
                ), "sanity check: there should be no bare tensors sent via IPC"
                res[k] = current[k]
        return res

    def prepare_for_state_update(self, iter) -> Dict:
        """
        Called before new parameter update is issued, e.g. before `optimizer.step()`
        """
        if torch.cuda.is_available():
            self.stream.synchronize()

        prev_rb_head_idx = self.rb_head_idx
        self.rb_head_idx = (self.rb_head_idx + 1) % self.rb_capacity
        if self.rb_curr_len < self.rb_capacity:
            self.rb_curr_len += 1
        self.iters_in_rb[self.rb_head_idx] = iter

        state_for_ipc = {}

        if self.rb_curr_len > 1:
            state_for_ipc = self._get_state_part_for_ipc(self.rb[prev_rb_head_idx])

            # We extracted tensor snapshots to be sent, so mark all tensor snapshots as sent
            dict_list_map_inplace(self._mark_as_sent_if_tensor_snapshot, state_for_ipc)

            # NOTE: in `states` we send just the recently finished state, not the whole `self.rb`
            # rank monitor will merge {prev_rb_head_idx: state_for_ipc} with its states buffer

            state_for_ipc = {
                'states': {prev_rb_head_idx: state_for_ipc},
                'rb_head_idx': self.rb_head_idx,
                'rb_capacity': self.rb_capacity,
                'iters_in_rb': self.iters_in_rb,
            }
        else:
            assert prev_rb_head_idx == -1

        return state_for_ipc

    def after_state_update_issued(self, state_dict) -> Dict:
        """
        Called after parameter update is issued, e.g. after `optimizer.step()`
        """
        if torch.cuda.is_available():
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._merge_state_dicts_inplace(self.rb[self.rb_head_idx], state_dict)
        else:
            self._merge_state_dicts_inplace(self.rb[self.rb_head_idx], state_dict)
        return None  # None means there is no heartbeat sent after `after_state_update_issued` call

    def _gather_last_complete_iter(self, iters_in_rb: Sequence[int]):
        """
        Find max iter available on all ranks
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, iters_in_rb)
            common_iters = set.intersection(*[set(it) for it in gathered])
        else:
            common_iters = set(iters_in_rb)
        return sorted(list(common_iters))[-1] if common_iters else None

    def restore(self, state_dict) -> Dict:
        """
        Return state dict with the last completely snapshotted iteration
        Tensors in the returned state dict will be allocated on the same device
        where they originally were, while taking the snapshot.
        """
        if not state_dict:
            return {}
        iters_in_rb = state_dict['iters_in_rb']
        iter2idx = {v: k for k, v in enumerate(iters_in_rb)}
        rb_head_idx = state_dict['rb_head_idx']
        del iters_in_rb[rb_head_idx]  # remove not completed iter
        last_complete_iter = self._gather_last_complete_iter(iters_in_rb)
        if last_complete_iter is None:
            return {}
        last_iter_rb_idx = iter2idx[last_complete_iter]
        res_state = state_dict['states'][last_iter_rb_idx]
        return dict_list_map_outplace(self._restore_if_tensor_snapshot, res_state)


class InMemCheckpointManagerNone:
    def prepare_for_state_update(self, *args, **kwargs) -> Dict:
        return None

    def after_state_update_issued(self, *args, **kwargs) -> Dict:
        return None

    def restore(self, *args, **kwargs) -> Dict:
        return None
