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
import logging
import os
import socket
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .data import AuthkeyMsg, HeartbeatMsg, InitMsg, OkMsg, RankInfo, UpdateConfigMsg
from .rank_monitor_server import RankMonitorServer
from .timeouts_calc import TimeoutsCalc
from .utils import read_obj_from_ipc_socket, write_object_to_ipc_socket


class RankMonitorClientError(Exception):
    pass


@dataclass
class HeartbeatTimeouts:
    """
    Contains hearbeat related timeouts used by FT.
    - `initial` is the timeout for the first heartbeat.
    - `subsequent` is the timeout for the subsequent heartbeats.

    Usually, the first heartbeat takes longer to be sent, hence there are 2 separate timeouts.
    - `were_calculated` indicates whether the timeouts were calculated from
        the observed heartbeat intervals or defined in the (YAML) config.
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


class RankMonitorClient:
    """
    `RankMonitorClient` is a client for `RankMonitorServer`.
    Its instances are created in each rank process. After creation,
    IPC connection can be established with `RankMonitorServer` using `.init_workload_monitoring`.
    The client should send heartbeats to the server, which monitor its health.
    Heartbeats are sent with `.send_heartbeat`.

    `RankMonitorServer` monitors time between heartbeats and can detect hangs.
    `RankMonitorClient` can estimate suitable timeouts for the heartbeats,
    that will be used instead of values provided in the FT config.
    If there are timeouts predefined in the FT config and timeouts calculated,
    the calculated timeouts always take precedence. Currently used timeouts can be read from
    `timeouts` field. New timeouts can be calculated and set with `.calculate_and_set_timeouts`.

    Stateful protocol (`.state_dict()` `.load_state_dict()`)
    is used to persist the state of the client, e.g. calculated timeouts.

    `RankMonitorClient` logger is used for logging.
    """

    CURRENT_TIMEOUTS_STATE_KEY = "current_timeouts"

    def __init__(self):
        """
        Basic initialization of RankMonitorClient instance.
        `.init_workload_monitoring()` and `.load_state_dict()` need to be called to fully initialize.
        Full FT configuration will be obtained from the server via IPC.
        """
        self.rank_info = None
        self.rank_monitor_socket = None
        self.is_initialized = False
        self.timeouts_calc = None
        self.timeouts = None
        self.loaded_timeouts = None
        self.chkpt_manager = None
        self.iter_idx = 0
        self.cfg = None
        self.logger = logging.getLogger("RankMonitorClient")

    def _ensure_is_ready(self):
        if not self.is_initialized:
            raise RankMonitorClientError("RankMonitorClient is not initialized")
        assert self.rank_monitor_socket is not None

    def _ensure_response_is_ok(self, sock):
        reply = read_obj_from_ipc_socket(sock)
        if not isinstance(reply, OkMsg):
            raise RankMonitorClientError(f"Unexpected reply: {reply}. Expected OkMsg")
        return reply

    def _set_calculated_timeouts(self, new_timeouts: HeartbeatTimeouts):
        """
        Send calculated timeouts to the server. They have an effect immediately after the call.
        """
        assert new_timeouts.are_valid and new_timeouts.were_calculated
        cfg_upd_msg = UpdateConfigMsg(
            new_initial_heartbeat_timeout=new_timeouts.initial,
            new_heartbeat_timeout=new_timeouts.subsequent,
        )
        write_object_to_ipc_socket(cfg_upd_msg, self.rank_monitor_socket)
        self._ensure_response_is_ok(self.rank_monitor_socket)
        self.timeouts = new_timeouts

    @staticmethod
    def _merge_timeouts(new_value: float, old_value: float, alpha: float = 0.75) -> float:
        """Merge computed timeout values with EMA"""
        assert 0 < alpha <= 1, "Alpha must be in the range (0, 1]."
        assert old_value > 0 and new_value > 0, "Timeout values must be non-negative."
        return alpha * new_value + (1 - alpha) * old_value

    def calculate_and_set_timeouts(self, skip_if_not_ready: bool = False) -> bool:
        """
        Calculates and sets the timeouts used for hang detection.

        NOTE: this call synchronizes the calculated timeouts across all ranks.
        NOTE: if calculated timeout value is smaller that currently used, the new value is ignored.

        Args:
            skip_if_not_ready (bool, optional): If True, silently skips the calculation if there
              is not enough data collected. Otherwise error will be raised. Defaults to False.

        Returns:
            bool: True if the timeouts were calculated and set successfully. False is returned only
                if calculation was not possible and `skip_if_not_ready` was `True`.
        """
        self._ensure_is_ready()
        self.timeouts_calc.synchronize_all()
        if self.timeouts_calc.can_get_timeouts():
            to = self.timeouts_calc.get_timeouts()
            new_initial, new_subsequent = to.initial, to.subsequent
            if self.timeouts.are_valid and self.timeouts.were_calculated:
                new_initial = RankMonitorClient._merge_timeouts(new_initial, self.timeouts.initial)
                new_subsequent = RankMonitorClient._merge_timeouts(
                    new_subsequent, self.timeouts.subsequent
                )
            new_timeouts = HeartbeatTimeouts(
                initial=new_initial,
                subsequent=new_subsequent,
                were_calculated=True,
            )
            self._set_calculated_timeouts(new_timeouts)
            return True
        else:
            if skip_if_not_ready:
                return False
            else:
                raise RankMonitorClientError("Not enough heartbeats to compute timeouts.")

    def _send_heartbeat_impl(self, state) -> None:
        """
        Implementation of heartbeat sending.

        Args:
            state (Mapping): The state information to be included in the heartbeat message.
        """
        self._ensure_is_ready()
        try:
            hb_msg = HeartbeatMsg(self.rank_info.rank, state)
            write_object_to_ipc_socket(hb_msg, self.rank_monitor_socket)
            self._ensure_response_is_ok(self.rank_monitor_socket)
            self.timeouts_calc.update()
        except Exception as e:
            raise RankMonitorClientError(
                f"RankMonitorClient could not send the heartbeat. Exception: {e}"
            )

    def _connect_to_rmon_server(self):
        assert self.rank_monitor_socket is None
        ipc_socket_path = RankMonitorServer.get_ipc_socket_path(self.rank_info.rank)
        self.rank_monitor_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.rank_monitor_socket.connect(ipc_socket_path)
        write_object_to_ipc_socket(AuthkeyMsg(), self.rank_monitor_socket)
        self._ensure_response_is_ok(self.rank_monitor_socket)
        init_msg = InitMsg()
        init_msg.rank_info = self.rank_info
        write_object_to_ipc_socket(init_msg, self.rank_monitor_socket)
        reply_for_init = read_obj_from_ipc_socket(self.rank_monitor_socket)
        if not isinstance(reply_for_init, OkMsg):
            raise RankMonitorClientError(
                f"Unexpected reply for init msg: {reply_for_init}. Expected OkMsg"
            )
        # we receive current FT config from the server
        self.cfg = reply_for_init.cfg

    def init_workload_monitoring(
        self,
    ) -> None:
        """
        Initializes the fault tolerance and connects to the RankMonitorServer.
        """
        if self.is_initialized:
            raise RankMonitorClientError("RankMonitorClient is already initialized")

        self.logger.debug(f"Initializing fault detection. Rank process PID={os.getpid()}")

        self.rank_info = RankInfo.get_for_current_rank()

        self._connect_to_rmon_server()

        self.timeouts_calc = TimeoutsCalc(safety_factor=self.cfg.safety_factor)

        if self.loaded_timeouts:
            # restore the timeouts that were calculated and set previously.
            # rank monitor server is only aware of the timeouts from the main config.
            # if timeouts were calculated by the client, we need to send it to the server.
            self._set_calculated_timeouts(self.loaded_timeouts)
            assert self.timeouts.are_valid and self.timeouts.were_calculated is True
        elif (
            self.cfg.initial_rank_heartbeat_timeout is not None
            and self.cfg.rank_heartbeat_timeout is not None
        ):
            # will use predefined timeouts from the config
            self.timeouts = HeartbeatTimeouts(
                self.cfg.initial_rank_heartbeat_timeout,
                self.cfg.rank_heartbeat_timeout,
                False,
            )
            assert self.timeouts.are_valid and self.timeouts.were_calculated is False
        else:
            self.timeouts = HeartbeatTimeouts()
            assert self.timeouts.are_valid is False

        self.is_initialized = True

    def shutdown_workload_monitoring(self):
        """
        Shutdown the workload monitoring and close the connection to the RankMonitorServer.
        """
        if self.is_initialized:
            self.rank_monitor_socket.shutdown(socket.SHUT_RDWR)
            self.rank_monitor_socket.close()
            self.rank_monitor_socket = None
            self.is_initialized = False

    def send_heartbeat(self) -> None:
        """
        Sends a empty (not containing a state) heartbeat message to the rank monitor server.
        """
        self._send_heartbeat_impl(state=dict())

    def state_dict(self) -> Mapping[str, Any]:
        """
        Returns the state dictionary of this RankMonitorClient object.

        NOTE: this method returns the same values on all ranks,
            there are no rank-specific values in `RankMonitorClient` state.

        Returns:
            Mapping[str, Any]: The state dictionary containing the current state.
        """
        state = {}
        if self.timeouts.are_valid and self.timeouts.were_calculated:
            # timeouts are sychronized across all ranks after calculation,
            # so `self.timeouts` is identical on all ranks.
            state[self.CURRENT_TIMEOUTS_STATE_KEY] = dataclasses.asdict(self.timeouts)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Loads the state of the RankMonitorClient from a dictionary.

        Can be called at any momemnt e.g. before `init_workload_monitoring` or after.

        Args:
            state: (Mapping[str, Any]): The state as returend from the `state_dict` method.
        """
        if self.CURRENT_TIMEOUTS_STATE_KEY in state:
            self.loaded_timeouts = HeartbeatTimeouts(**state[self.CURRENT_TIMEOUTS_STATE_KEY])
            if self.is_initialized:
                self._set_calculated_timeouts(self.loaded_timeouts)
            # else, the timeouts will be set in `init_workload_monitoring`
