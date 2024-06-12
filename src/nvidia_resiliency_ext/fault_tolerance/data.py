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

import dataclasses
import multiprocessing as mp
import os
import socket
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Set

import torch.distributed as dist

"""Env variable name that stores the path of IPC socket connecting a rank with its monitor"""
FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR = "FT_RANK_MONITOR_IPC_SOCKET"

"""Env variable name that stores the path of IPC socket connecting a rank with its launcher"""
FT_LAUNCHER_IPC_SOCKET_ENV_VAR = "FT_LAUNCHER_IPC_SOCKET"


@dataclass
class RankInfo:
    """
    Rank info used internally by the fault tolerance.
    """

    global_rank: int
    local_rank: int
    host: str
    pid: int

    @staticmethod
    def get_for_current_rank():
        """
        Get info on current rank
        """
        global_rank = None
        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
        else:
            global_rank = os.getenv('RANK', None)
        if global_rank is None:
            raise RuntimeError(
                "Could not find the rank of the current process. "
                "Is it a part of a distributed workload?"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        host = socket.gethostname()
        pid = os.getpid()

        return RankInfo(
            global_rank=global_rank,
            local_rank=local_rank,
            host=host,
            pid=pid,
        )


@dataclass
class HeartbeatTimeouts:
    """
    Contains hearbeat related timeouts used by FT.
    - `initial` is the timeout for the first heartbeat.
    - `subsequent` is the timeout for the subsequent heartbeats.
    - `were_calculated` indicates whether the timeouts were calculated from observed intervals
        or defined in the config.

    Usually, the first heartbeat takes longer to be sent, hence there are 2 separate timeouts.
    """

    initial: Optional[float] = None
    subsequent: Optional[float] = None
    were_calculated: Optional[bool] = None

    @property
    def are_valid(self) -> bool:
        return self.initial is not None and self.subsequent is not None

    def __str__(self) -> str:
        ini = f"{self.initial:.2f}" if self.initial is not None else "None"
        subs = f"{self.subsequent:.2f}" if self.subsequent is not None else "None"
        return f"HeartbeatTimeouts(initial={ini}, subsequent={subs}, were_calculated={self.were_calculated})"


@dataclass
class SectionTimeouts:
    """
    Contains section timeouts used by FT.

    Some timeouts can be calculated from the observed intervals, while others can be defined in the config.
    None as a timeout value means that the timeout is not used (as if it was +INF).

    - `section` is a section name to the timeout mapping.
    - `out_of_section` is the timeout for implicitly used "default" section
    - `were_calculated` indicates whether all timeouts were calculated
    - `calculated_sections` is a set of sections for which timeouts were calculated.
    - `is_out_of_section_calculated` indicates whether out_of_section timeout was calculated.
    """

    section: Mapping[str, Optional[float]] = dataclasses.field(default_factory=dict)
    out_of_section: Optional[float] = None

    calculated_sections: Set[str] = dataclasses.field(default_factory=set)
    is_out_of_section_calculated: bool = False

    @property
    def were_calculated(self) -> bool:
        all_sections = set(self.section.keys())
        return (self.calculated_sections == all_sections) and self.is_out_of_section_calculated

    @property
    def are_valid(self) -> bool:
        return (
            all(to is not None for to in self.section.values()) and self.out_of_section is not None
        )

    def __str__(self) -> str:
        sections_desc = ""
        oos = f"{self.out_of_section:.2f}" if self.out_of_section is not None else "None"
        for nm, to in self.section.items():
            to_as_str = f"{to:.2f}" if to is not None else "None"
            sections_desc += f"{nm}={to_as_str}, "
        return f"SectionTimeouts(out_of_section={oos}, sections=[{sections_desc}], calculated_sections={self.calculated_sections}, is_out_of_section_calculated={self.is_out_of_section_calculated})"

    def __post_init__(self):
        self.calculated_sections = set(self.calculated_sections)


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

    def __init__(
        self,
        rank: int,
        state: Optional[Mapping] = None,
    ):
        self.rank = rank
        self.state = state


class SectionAction(Enum):
    OPEN = 1
    CLOSE = 2
    CLOSE_ALL = 3


class SectionMsg:
    """
    Section update message. (rank -> rank monitor)
    """

    def __init__(
        self,
        rank: int,
        section: str,
        action: SectionAction,
    ):
        self.rank = rank
        self.section = section
        self.action = action


class UpdateConfigMsg:
    """
    Sent from rank -> rank monitor, when some config items are updated.
    Currently, only timeouts can be updated.
    """

    def __init__(
        self,
        hb_timeouts: Optional[HeartbeatTimeouts],
        section_timeouts: Optional[SectionTimeouts],
    ):
        self.hb_timeouts = hb_timeouts
        self.section_timeouts = section_timeouts


class OkMsg:
    """
    Heartbeat confirmation (rank monitor -> rank)
    Additional info from the monitor can be provided via kwargs,
    that will be attached as an attributes of the created object.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ErrorMsg:
    """
    Error message (rank monitor -> rank)
    """

    def __init__(self, cause='<not set>'):
        self.cause = cause

    def __str__(self):
        return f"{self.__class__.__name__}({self.cause})"


class WorkloadAction(Enum):
    """
    Actions that can be requested from the laucher
    * Continue - proceed with the next rendezvous as usual
    * ExcludeThisNode - do not use current node in the next rendezvous
    * ShutdownWorkload - shutdown rendezvous, do not restart ranks
    """

    Continue = 0
    ExcludeThisNode = 1
    ShutdownWorkload = 2


class WorkloadControlRequest:
    """
    Specify the action that should be taken by the launcher, with some optional context message
    This is sent from a rank to the launcher
    """

    def __init__(self, action: WorkloadAction, description="<not set>"):
        self.action = action
        self.description = description

    def __str__(self):
        return f"WorkloadControlRequest(action={self.action}, description={self.description})"
