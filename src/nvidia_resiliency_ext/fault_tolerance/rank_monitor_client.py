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
from typing import Any, Collection, Mapping, Optional

from .data import (
    FT_LAUNCHER_IPC_SOCKET_ENV_VAR,
    FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR,
    AuthkeyMsg,
    HeartbeatMsg,
    HeartbeatTimeouts,
    InitMsg,
    OkMsg,
    RankInfo,
    SectionAction,
    SectionMsg,
    SectionTimeouts,
    UpdateConfigMsg,
    WorkloadControlRequest,
)
from .ipc_connector import IpcConnector
from .timeouts_calc import TimeoutsCalc
from .utils import read_obj_from_ipc_socket, write_object_to_ipc_socket


class RankMonitorClientError(Exception):
    pass


class RankMonitorClient:
    """
    `RankMonitorClient` is a client for `RankMonitorServer`.
    Its instances are created in each rank process. After creation,
    IPC connection can be established with `RankMonitorServer` using `.init_workload_monitoring`.

    `RankMonitorClient` provides 2 independent functionalities that can be used for hang detection:

    1. Heartbeat mechanism:
    `.send_heartbeat` method sends a heartbeat to the server.
    `RankMonitorServer` monitors time elapsed between heartbeats and detects hangs, based on two timeouts:
    - initial timeout, is a maximum time between client initialization and the first heartbeat
    - subsequent timeout, is a maximum time between two consecutive heartbeats

    2. Section mechanism:
    `.start_section` and `.end_section` are used to wrap some sections of user code.
    Sections are identified by a user-provided name. User can configure timeouts for each section in the FT config.
    Hang is detected if a section is open for too long.
    To ensure that code that is not wrapped in any section is not running for too long (hung),
    there is additional "out-of-section" timeout, which is active while no section is open.

    `RankMonitorClient` can estimate suitable timeouts for the heartbeats and sections,
    that will be used instead of values provided in the FT config.
    If there are timeouts predefined in the FT config and timeouts calculated,
    the calculated timeouts always take precedence.

    If a timeout value is None it means it's not used (as if it was +inf).

    Currently used timeouts can be read from `.hb_timeouts` and `.section_timeouts` fields.
    New timeouts can be calculated and set with `.calculate_and_set_hb_timeouts` and `.calculate_and_set_section_timeouts`.

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
        self.hb_timeouts = None
        self.section_timeouts = None
        self.loaded_hb_timeouts = None
        self.loaded_section_timeouts = None
        self.chkpt_manager = None
        self.iter_idx = 0
        self.cfg = None
        self.logger = logging.getLogger("RankMonitorClient")
        self.launcher_connector = None
        launcher_ipc_socket_path = os.getenv(FT_LAUNCHER_IPC_SOCKET_ENV_VAR, None)
        if launcher_ipc_socket_path is not None:
            self.launcher_connector = IpcConnector(launcher_ipc_socket_path)
        else:
            self.logger.info(
                f"{FT_LAUNCHER_IPC_SOCKET_ENV_VAR} env varialble is not set. "
                "`.send_workload_control_request` wont work. This is normal if "
                "this rank was not started with ft_launcher"
            )

    def _ensure_is_ready(self):
        if not self.is_initialized:
            raise RankMonitorClientError("RankMonitorClient is not initialized")
        assert self.rank_monitor_socket is not None

    def _ensure_response_is_ok(self, sock):
        reply = read_obj_from_ipc_socket(sock)
        if not isinstance(reply, OkMsg):
            raise RankMonitorClientError(f"Unexpected reply: {reply}. Expected OkMsg")
        return reply

    def _set_calculated_timeouts(
        self,
        new_hb_timeouts: Optional[HeartbeatTimeouts],
        new_section_timeouts: Optional[SectionTimeouts],
    ):
        """
        Send calculated timeouts to the server. They have an effect immediately after the call.
        If heartbeat or section timeouts is None, old values are not updated.
        """
        cfg_upd_msg = UpdateConfigMsg(
            hb_timeouts=new_hb_timeouts, section_timeouts=new_section_timeouts
        )
        write_object_to_ipc_socket(cfg_upd_msg, self.rank_monitor_socket)
        self._ensure_response_is_ok(self.rank_monitor_socket)
        if new_hb_timeouts is not None:
            self.hb_timeouts = new_hb_timeouts
        if new_section_timeouts is not None:
            self.section_timeouts = new_section_timeouts

    def calculate_and_set_hb_timeouts(
        self,
        skip_if_not_ready: bool = False,
    ) -> bool:
        """
        Calculates and sets heartbeat timeouts used for hang detection.

        NOTE: this call synchronizes the calculated timeouts across all ranks.

        Args:
            skip_if_not_ready (bool, optional): If True, silently skips the calculation if there
              is not enough data collected. Otherwise error will be raised. Defaults to False.

        Returns:
            bool: True if the timeouts were calculated and set successfully. False is returned only
                if calculation was not possible and `skip_if_not_ready` was `True`.
        """
        self._ensure_is_ready()
        self.timeouts_calc.synchronize_all()
        if self.timeouts_calc.can_get_hb_timeouts():
            hb_timeouts = self.timeouts_calc.get_hb_timeouts(current=self.hb_timeouts)
            self._set_calculated_timeouts(new_hb_timeouts=hb_timeouts, new_section_timeouts=None)
            return True
        else:
            if skip_if_not_ready:
                return False
            else:
                raise RankMonitorClientError("Not enough data to compute timeouts.")

    def calculate_and_set_section_timeouts(
        self,
        selected_sections: Optional[Collection[str]] = None,
        calc_out_of_section: bool = True,
        skip_if_not_ready: bool = False,
    ) -> bool:
        """
        Calculates and sets section timeouts used for hang detection.

        NOTE: this call synchronizes the calculated timeouts across all ranks.

        Args:
            selected_sections (List[str], optional): List of sections which timeouts should be updated,
              If not provided (None) all section timeouts will be updated
            calc_out_of_section: (bool): Determines if "out of section" timeout should be updated.
              Defaults to True.
            skip_if_not_ready (bool, optional): If True, silently skips the calculation if there
              is not enough data collected. Otherwise error will be raised. Defaults to False.

        Returns:
            bool: True if the timeouts were calculated and set successfully. False is returned only
                if calculation was not possible and `skip_if_not_ready` was `True`.
        """
        self._ensure_is_ready()
        # if no section is opened, update out-of-section time.
        # this can be useful, e.g. if "calculate_and_set_section_timeouts" is called at the end of a training.
        # it can happen that the final "out-of-section" chunk is the longest one observed, so it would
        # underestimate the oos timeout if we didn't update it here.
        self.timeouts_calc.maybe_bump_oos_time()
        self.timeouts_calc.synchronize_all()
        if self.timeouts_calc.can_get_section_timeouts(
            selected_sections=selected_sections, calc_out_of_section=calc_out_of_section
        ):
            section_timeouts = self.timeouts_calc.get_section_timeouts(
                selected_sections=selected_sections,
                calc_out_of_section=calc_out_of_section,
                current=self.section_timeouts,
            )
            self._set_calculated_timeouts(
                new_hb_timeouts=None, new_section_timeouts=section_timeouts
            )
            return True
        else:
            if skip_if_not_ready:
                return False
            else:
                raise RankMonitorClientError("Not enough data to compute timeouts.")

    def _send_heartbeat_impl(self, state) -> None:
        """
        Implementation of heartbeat sending.

        Args:
            state (Mapping): The state information to be included in the heartbeat message.
        """
        self._ensure_is_ready()
        try:
            hb_msg = HeartbeatMsg(self.rank_info.global_rank, state)
            write_object_to_ipc_socket(hb_msg, self.rank_monitor_socket)
            self._ensure_response_is_ok(self.rank_monitor_socket)
            self.timeouts_calc.update_on_heartbeat()
        except Exception as e:
            raise RankMonitorClientError(
                f"RankMonitorClient could not send the heartbeat. Exception: {e}"
            )

    def _send_section_msg_impl(self, section: str, action: SectionAction) -> None:
        """
        Implementation of section related update sending.

        Args:
            section (str): Section name
            action (SectionAction): Section related action
        """
        self._ensure_is_ready()
        try:
            msg = SectionMsg(rank=self.rank_info.global_rank, section=section, action=action)
            write_object_to_ipc_socket(msg, self.rank_monitor_socket)
            self._ensure_response_is_ok(self.rank_monitor_socket)
            self.timeouts_calc.update_on_section_event(section=section, action=action)
        except Exception as e:
            raise RankMonitorClientError(
                f"RankMonitorClient could not send section update. Exception: {e}"
            )

    def _connect_to_rmon_server(self):
        assert self.rank_monitor_socket is None
        rmon_ipc_socket_path = os.getenv(FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR, None)
        if rmon_ipc_socket_path is None:
            raise RankMonitorClientError(
                f"{FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR} env variable should "
                "be set in a process that is running RankMonitorClient"
            )
        self.rank_monitor_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.rank_monitor_socket.connect(rmon_ipc_socket_path)
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

        sections = self.cfg.rank_section_timeouts.keys()
        self.timeouts_calc = TimeoutsCalc(sections=sections, safety_factor=self.cfg.safety_factor)

        # by default, use predefined timeouts from the config
        self.hb_timeouts = HeartbeatTimeouts(
            initial=self.cfg.initial_rank_heartbeat_timeout,
            subsequent=self.cfg.rank_heartbeat_timeout,
            were_calculated=False,
        )
        self.section_timeouts = SectionTimeouts(
            section=self.cfg.rank_section_timeouts,
            out_of_section=self.cfg.rank_out_of_section_timeout,
            calculated_sections={},
            is_out_of_section_calculated=False,
        )

        if self.loaded_hb_timeouts or self.loaded_section_timeouts:
            # restore the timeouts that were calculated and stored previously.
            # rank monitor server is only aware of the timeouts from the main config.
            # if timeouts were calculated by the client, we need to send it to the server.
            self._set_calculated_timeouts(
                new_hb_timeouts=self.loaded_hb_timeouts,
                new_section_timeouts=self.loaded_section_timeouts,
            )

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

    def start_section(self, section: str) -> None:
        """
        Starts a new timed section with the given name.

        NOTE: Different sections can be arbitraly aranged (nested, partially or fully overlapping).
            but the same section name cannot be opened twice without closing it first.

        Args:
            section (str): User defined name of the section.
        """
        self._send_section_msg_impl(section, SectionAction.OPEN)

    def end_section(self, section: str) -> None:
        """
        Close the section with the given name.

        NOTE: The section must be opened.

        Args:
            section (str): User defined name of the section.
        """
        self._send_section_msg_impl(section, SectionAction.CLOSE)

    def end_all_sections(self) -> None:
        """
        Closes all currently opened sections.
        Does nothing if there are no sections open.
        """
        self._send_section_msg_impl(None, SectionAction.CLOSE_ALL)

    def state_dict(self) -> Mapping[str, Any]:
        """
        Returns the state dictionary of this RankMonitorClient object.

        NOTE: this method returns the same values on all ranks,
            there are no rank-specific values in `RankMonitorClient` state.

        Returns:
            Mapping[str, Any]: The state dictionary containing the current state.
        """
        state = {self.CURRENT_TIMEOUTS_STATE_KEY: {}}
        if self.hb_timeouts.are_valid and self.hb_timeouts.were_calculated:
            state[self.CURRENT_TIMEOUTS_STATE_KEY]['heartbeat'] = dataclasses.asdict(
                self.hb_timeouts
            )
        has_calculated_section_timeouts = (
            self.section_timeouts.calculated_sections
            or self.section_timeouts.is_out_of_section_calculated
        )
        if has_calculated_section_timeouts:
            state[self.CURRENT_TIMEOUTS_STATE_KEY]['section'] = dataclasses.asdict(
                self.section_timeouts
            )
            # convert set to list for JSON serialization
            calc_sections_list = list(
                state[self.CURRENT_TIMEOUTS_STATE_KEY]['section']['calculated_sections']
            )
            calc_sections_list.sort()
            state[self.CURRENT_TIMEOUTS_STATE_KEY]['section'][
                'calculated_sections'
            ] = calc_sections_list
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """
        Loads the state of the RankMonitorClient from a dictionary.

        Can be called at any momemnt e.g. before `init_workload_monitoring` or after.

        Args:
            state: (Mapping[str, Any]): The state as returend from the `state_dict` method.
        """
        if self.CURRENT_TIMEOUTS_STATE_KEY in state:
            ft_state = state[self.CURRENT_TIMEOUTS_STATE_KEY]
            if 'heartbeat' in ft_state:
                self.loaded_hb_timeouts = HeartbeatTimeouts(**ft_state['heartbeat'])
            if 'section' in ft_state:
                self.loaded_section_timeouts = SectionTimeouts(**ft_state['section'])
            if self.is_initialized:
                self._set_calculated_timeouts(
                    new_hb_timeouts=self.loaded_hb_timeouts,
                    new_section_timeouts=self.loaded_section_timeouts,
                )
            # else,
            # the timeouts will be set in `init_workload_monitoring`

    def send_workload_control_request(self, req: WorkloadControlRequest):
        """
        Request an workload related action.
        It is sent to the ft_launcher and affects the subsequent rendezvous.

        Args:
            req (WorkloadControlRequest): request specification
        """
        if self.launcher_connector is None:
            raise RuntimeError("IPC connection with the launcher is not available.")
        self.launcher_connector.send(
            (
                self.rank_info.global_rank,
                req,
            )
        )
