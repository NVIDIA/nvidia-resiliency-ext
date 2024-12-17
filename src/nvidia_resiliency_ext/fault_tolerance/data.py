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

import multiprocessing as mp
import os
import socket
from dataclasses import dataclass
from typing import Mapping, Optional

import torch.distributed as dist


@dataclass
class RankInfo:
    """
    Rank info used internally by the fault tolerance.
    """

    rank: int
    host: str
    pid: int

    @staticmethod
    def get_for_current_rank():
        """
        Get info on current rank
        """
        assert (
            dist.is_available() and dist.is_initialized()
        ), "Torch distributed should be initialized"

        rank = dist.get_rank()
        host = socket.gethostname()
        pid = os.getpid()

        return RankInfo(
            rank=rank,
            host=host,
            pid=pid,
        )


class AuthkeyMsg:
    """
    Sent (rank -> rank monitor) right after IPC connection is established.
    We need to set authkey in server to be able to receive pickled Manager, Tensors etc.
    """

    def __init__(self, authkey: Optional[bytes] = None):
        self.authkey = authkey
        if self.authkey is None:
            self.authkey = bytes(mp.current_process().authkey)


class InitMsg:
    """
    Send (rank -> rank monitor) to initialize new session
    """

    pass


class HeartbeatMsg:
    """
    Heartbeat message. (rank -> rank monitor)
    """

    def __init__(self, rank: int, state_dict_for_chkpt: Optional[Mapping] = None):
        self.rank = rank
        self.state_dict_for_chkpt = state_dict_for_chkpt


class UpdateConfigMsg:
    """
    Sent from rank -> rank monitor, when some config items are updated.
    Currently, only heartbeat timeouts can be updated.
    """

    def __init__(
        self,
        new_initial_heartbeat_timeout: float,
        new_heartbeat_timeout: float,
    ):
        self.new_initial_heartbeat_timeout = new_initial_heartbeat_timeout
        self.new_heartbeat_timeout = new_heartbeat_timeout


class OkMsg:
    """
    Heartbeat confirmation (rank monitor -> rank)
    Additional info from the monitor can be provided via kwargs,
    that will be attached as an attributes of the created object.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
