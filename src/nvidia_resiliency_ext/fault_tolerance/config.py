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

import argparse
import contextlib
import dataclasses
import logging
import signal
from dataclasses import dataclass, fields
from typing import Mapping, Optional

import yaml


@dataclass
class FaultToleranceConfig:
    """
    Configuration of the fault tolerance

    * `workload_check_interval` [float] periodic rank check interval (in seconds) in rank monitors.
    * `initial_rank_heartbeat_timeout` [float|None] timeout (in seconds) for the first heartbeat from a rank.

    Usually, it takes a bit longer for the first heartbeat to be sent, as the rank needs to initialize.
    If rank does not send the first heartbeat within `initial_rank_heartbeat_timeout`, failure is detected.

    * `rank_heartbeat_timeout` [float|None] timeout (in seconds) for subsequent heartbeats from a rank.

    If no rank heartbeat is received within `rank_heartbeat_timeout`, failure is detected.

    * `safety_factor` [float] when deducing the timeouts, observed intervals are
      multiplied by this factor to obtain the timeouts.
    * `rank_termination_signal` signal used to terminate the rank when failure is detected.
    * `log_level` log level of fault tolerance components
    * `rank_section_timeouts` Mapping[str,float|None] timeouts for specific sections in user code.
      Only sections listed here will send IPC messages to the monitor server and collect timing data.
      Sections not in this mapping will have near-zero overhead (no IPC, no timing collection).
    * `rank_out_of_section_timeout` [float|None] the timeout used for implicit/default section,
      that spans code not wrapped in any other section.
    * `restart_check_interval` - interval between checks if restart is in progress, needed for layered restart protocol
    * `enable_nic_monitor` - Enable NIC health monitoring in training. Default: False.
    * `enable_nic_healthcheck` - Enable NIC link state health check before rendezvous. This checks if
      network interface ports (RDMA/InfiniBand and Ethernet) are in ACTIVE state and fails if any port transitioned from ACTIVE to non-ACTIVE.
      Unlike enable_nic_monitor (which periodically monitors link_downed counters), this performs a one-time
      state check during rendezvous. Can be used independently or together with enable_nic_monitor. Default: False.
    * `pci_topo_file` - PCI topo file that describes GPU and NIC topology.
    * `link_down_path_template` - Template path for NIC link down files. Should contain '{dev_name}'
      placeholder which will be replaced with actual NIC device name.
    * `link_state_path_template` - Template path for NIC link state files. Should contain '{nic}'
      placeholder which will be replaced with actual NIC device name. Default: /sys/class/infiniband/{nic}/ports/1/state
    * `enable_dist_storage_healthcheck` - Enable distributed storage health check (Lustre + NFS)
      before rendezvous. Checks Lustre health and reachability of Lustre/NFS mounts. Default: False.
    * `storage_healthcheck_path` - Comma-separated absolute paths to validate for existence/readability
      before rendezvous. Used by the storage path health check. Default: None.
    * `skip_section_response` - If True, section and heartbeat messages are sent without waiting
      for server response (unidirectional communication). This significantly reduces latency for
      high-frequency operations. Server logs errors instead of sending them back.
      Default: True (recommended for production). Set to False during development to catch errors immediately.
    * `segment` - Controls hot spare node behavior and rank assignment strategy:
      - None (default): Simple hot spare mode suitable for H100 and systems without NVLink domain segmentation.
        First min_nodes become active, extras become hot spares. No ClusterUUID parsing required.
      - N (integer): Segment-aware mode for NVSwitch-based systems (DGX H200, HGX B200).
        Minimum number of nodes required per NVLink domain (identified by GPU ClusterUUID via nvidia-smi).
        Domains with fewer nodes are excluded. From each valid domain, as many complete segments
        as possible are selected (e.g., 12 nodes with segment=4 → use 12 nodes = 3 segments).
        min_nodes must be divisible by segment. When set, ClusterUUID is automatically queried.
      Note: segment=None and segment=1 have similar behavior in rank assignment, but segment=1
      requires ClusterUUID while segment=None does not.
    * `numa_bind_strict` - If True, use strict NUMA binding with both CPU and memory bound to the
      same NUMA node (--cpunodebind=N --membind=N). If False (default), only bind CPU to NUMA node
      and allow local memory allocation (--cpunodebind=N --localalloc). Default: False.
    * `gpu_memory_reclaim_timeout` [float] timeout (in seconds) to wait for GPU memory to be reclaimed
      after worker shutdown before starting new workers. Default: 50.0.
    * `gpu_memory_tolerance_mb` [float] maximum allowed GPU memory usage (in MB) when checking if
      memory has been reclaimed. Default: 512.0.
    * `gpu_memory_poll_interval` [float] poll interval (in seconds) for checking GPU memory during
      reclaim process. Default: 2.0.
    * `check_remaining_processes` [bool] if True, check for and log any remaining worker processes
      after termination. Useful for debugging process cleanup issues. Default: False.
    * `install_exception_hook` [bool] if True, installs sys.excepthook to capture uncaught exceptions
      in training worker processes, format and log the traceback, and use os._exit() to exit the
      process reliably. Default: False.
    * Attribution service (optional):
      - `attrsvc_host` [str] hostname/IP of the attribution service
      - `attrsvc_port` [int] port of the attribution service

    If any timeout is None, it has no effect (as if it was +INF).
    All timeouts can be deduced and set during runtime.
    """

    workload_check_interval: float = 5.0
    initial_rank_heartbeat_timeout: Optional[float] = 60.0 * 60.0
    rank_heartbeat_timeout: Optional[float] = 45.0 * 60.0
    rank_section_timeouts: Mapping[str, Optional[float]] = dataclasses.field(default_factory=dict)
    rank_out_of_section_timeout: Optional[float] = None
    node_health_check_interval: float = 5.0
    safety_factor: float = 5.0
    rank_termination_signal: signal.Signals = signal.SIGKILL
    log_level: int = logging.INFO
    restart_check_interval: float = 60.0
    enable_nic_monitor: bool = False
    enable_nic_healthcheck: bool = False
    enable_dist_storage_healthcheck: bool = False
    storage_healthcheck_path: Optional[str] = None
    pci_topo_file: Optional[str] = None
    link_down_path_template: Optional[str] = None
    link_state_path_template: Optional[str] = None
    skip_section_response: bool = True
    segment: Optional[int] = None
    numa_bind_strict: bool = False
    gpu_memory_reclaim_timeout: float = 50.0
    gpu_memory_tolerance_mb: float = 512.0  # Maximum allowed GPU memory usage (in MB)
    gpu_memory_poll_interval: float = 2.0  # Poll interval for GPU memory check (in seconds)
    check_remaining_processes: bool = False
    # Progress tracking configuration (controlled by max_no_progress_restarts)
    max_no_progress_restarts: int = 3
    min_progress_iterations: int = 200
    progress_update_interval: float = 30.0  # Seconds between sending progress updates to launcher
    install_exception_hook: bool = False
    # Attribution service configuration (optional)
    attrsvc_host: Optional[str] = None
    attrsvc_port: Optional[int] = None

    @property
    def is_progress_tracking_enabled(self) -> bool:
        """Check if progress tracking is enabled (controlled by max_no_progress_restarts > 0)."""
        return self.max_no_progress_restarts > 0

    @staticmethod
    def from_kwargs(ignore_not_recognized: bool = True, **kwargs) -> 'FaultToleranceConfig':
        """
        Create a FaultToleranceConfig object from keyword arguments.

        Args:
            ignore_not_recognized (bool, optional): Whether to ignore unrecognized arguments. Defaults to True.
            **kwargs: Keyword arguments representing the fields of the FaultToleranceConfig object.

        Returns:
            FaultToleranceConfig: The created FaultToleranceConfig object.

        Raises:
            ValueError: If there are unrecognized arguments and ignore_not_recognized is False.
        """
        fields_set = {f.name for f in fields(FaultToleranceConfig) if f.init}
        matching_args = {k: v for k, v in kwargs.items() if k in fields_set}
        extra_args = {k: v for k, v in kwargs.items() if k not in fields_set}
        if extra_args and not ignore_not_recognized:
            raise ValueError(f"Not recognized args: {extra_args}")
        return FaultToleranceConfig(**matching_args)

    @staticmethod
    def from_yaml_file(cfg_path: str, ignore_not_recognized: bool = True) -> 'FaultToleranceConfig':
        """
        Load the fault tolerance configuration from a YAML file.

        YAML file should contain `fault_tolerance` section.
        `fault_tolerance` section can be at the top level or nested in any other section.

        Args:
            cfg_path (str): The path to the YAML configuration file.
            ignore_not_recognized (bool, optional): Whether to ignore unrecognized configuration options.
                Defaults to True.

        Returns:
            FaultToleranceConfig: The fault tolerance configuration object.

        Raises:
            ValueError: If the 'fault_tolerance' section is not found in the config file.
        """
        with open(cfg_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            ft_cfg = FaultToleranceConfig._find_fault_tol_section(yaml_data)
            if ft_cfg:
                return FaultToleranceConfig.from_kwargs(
                    **ft_cfg, ignore_not_recognized=ignore_not_recognized
                )
            else:
                raise ValueError(f"'fault_tolerance' section not found in config file {cfg_path}")

    @staticmethod
    def _parse_timeout_arg(timeout_arg: str) -> Optional[float]:
        """
        Parse a timeout CLI argument.
        Timeout can be a float or 'None'/'null'/'' to represent None.

        Args:
            timeout_arg (str): The timeout value as a string

        Returns:
            Optional[float]: The parsed timeout value or None
        """
        timeout_arg = timeout_arg.strip()
        if timeout_arg.lower() in ['none', 'null', '']:
            return None
        else:
            return float(timeout_arg)

    @staticmethod
    def _parse_section_timeouts_arg(section_timeouts_arg: str) -> Mapping[str, Optional[float]]:
        """
        Parse section timeouts CLI argument.
        Expected format: "section1:timeout1,section2:timeout2,..."
        Timeout can be a float or 'None'/'null'/'' to represent None.

        Args:
            section_timeouts_arg (str): The section timeouts string

        Returns:
            Mapping[str, Optional[float]]: Dictionary mapping section names to timeout values
        """
        section_timeouts_arg = section_timeouts_arg.strip()
        if not section_timeouts_arg:
            return {}
        section_timeout_pairs = section_timeouts_arg.split(",")
        res = {}
        for st in section_timeout_pairs:
            section, timeout = st.split(":")
            section = section.strip()
            timeout = timeout.strip()
            res[section] = FaultToleranceConfig._parse_timeout_arg(timeout)
        return res

    @staticmethod
    def from_args(args: argparse.Namespace):
        """
        Init FT config object from parsed CLI args.

        Implements the following logic:
        - Use default FT config as a base.
        - If there is a config file argument defined, first try to read the FT config from the file.
        - Update the FT config with FT args provided via CLI.
        - If can't read from file and there are no related args in CLI, raise an exception.

        Args:
            args (argparse.Namespace): Parsed arguments
        """
        # Start with default config
        ft_cfg = FaultToleranceConfig()
        is_read_from_file = False

        # Try to read from config file if specified
        if args.ft_cfg_path is not None:
            with contextlib.suppress(ValueError):
                ft_cfg = FaultToleranceConfig.from_yaml_file(args.ft_cfg_path)
                is_read_from_file = True

        # Extract FT args from CLI
        cli_ft_args = {}
        timeout_fields = [
            'initial_rank_heartbeat_timeout',
            'rank_heartbeat_timeout',
            'rank_out_of_section_timeout',
            'workload_check_interval',
            'node_health_check_interval',
            'safety_factor',
            'restart_check_interval',
            'gpu_memory_reclaim_timeout',
            'gpu_memory_tolerance_mb',
            'gpu_memory_poll_interval',
        ]
        for field in fields(FaultToleranceConfig):
            cli_field_name = f"ft_{field.name}"
            val = getattr(args, cli_field_name, None)
            if val is not None:
                if field.name == "rank_section_timeouts" and isinstance(val, str):
                    val = FaultToleranceConfig._parse_section_timeouts_arg(val)
                elif field.name in timeout_fields and isinstance(val, str):
                    val = FaultToleranceConfig._parse_timeout_arg(val)
                cli_ft_args[field.name] = val

        # Update config with CLI args
        for arg_name, arg_val in cli_ft_args.items():
            setattr(ft_cfg, arg_name, arg_val)

        # Fix any type issues
        ft_cfg._fix_log_level_type()
        ft_cfg._fix_rank_termination_signal_type()

        return ft_cfg

    def to_yaml_file(self, cfg_path: str) -> None:
        """
        Convert the configuration object to a YAML file and save it to the specified path.

        Args:
            cfg_path (str): The path to save the YAML file.

        Returns:
            None
        """
        # first, ensure that `rank_termination_signal` and `log_level` have their native types
        # this might not be the case, if the object was modified after creation
        self._fix_rank_termination_signal_type()
        self._fix_log_level_type()
        with open(cfg_path, 'w') as file:
            ft_cfg_dict = dataclasses.asdict(self)
            ft_cfg_dict['rank_termination_signal'] = self.rank_termination_signal.name
            ft_cfg_dict['log_level'] = self.log_level
            ft_cfg_dict = {'fault_tolerance': ft_cfg_dict}
            yaml.dump(ft_cfg_dict, file)

    @staticmethod
    def _find_fault_tol_section(yaml_data):
        if isinstance(yaml_data, dict):
            if "fault_tolerance" in yaml_data:
                return yaml_data["fault_tolerance"]
            else:
                for key, value in yaml_data.items():
                    sub_config = FaultToleranceConfig._find_fault_tol_section(value)
                    if sub_config:
                        return sub_config
        elif isinstance(yaml_data, list):
            for item in yaml_data:
                sub_config = FaultToleranceConfig._find_fault_tol_section(item)
                if sub_config:
                    return sub_config
        return None

    def _fix_rank_termination_signal_type(self):
        if isinstance(self.rank_termination_signal, int):
            self.rank_termination_signal = signal.Signals(self.rank_termination_signal)
        elif isinstance(self.rank_termination_signal, str):
            sig_str = self.rank_termination_signal.upper()
            if getattr(signal, sig_str, None) is None:
                raise ValueError(
                    f"Invalid rank_termination_signal string: {self.rank_termination_signal}"
                )
            self.rank_termination_signal = signal.Signals[sig_str]
        elif isinstance(self.rank_termination_signal, signal.Signals):
            self.rank_termination_signal = self.rank_termination_signal
        else:
            raise ValueError(
                f"Invalid value for rank_termination_signal: {self.rank_termination_signal}"
            )

    def _fix_log_level_type(self):
        if isinstance(self.log_level, int):
            if not (logging.DEBUG <= self.log_level <= logging.CRITICAL):
                raise ValueError(
                    f"Invalid log level value ({self.log_level}). Should be in [{logging.DEBUG} (DEBUG), {logging.FATAL} (CRITICAL)]"
                )
        elif isinstance(self.log_level, str):
            log_level_str = self.log_level.upper()
            if log_level_str in ['DEBUG', 'DBG']:
                self.log_level = logging.DEBUG
            elif log_level_str == 'INFO':
                self.log_level = logging.INFO
            elif log_level_str in ['WARNING', 'WARN']:
                self.log_level = logging.WARNING
            elif log_level_str == 'ERROR':
                self.log_level = logging.ERROR
            elif log_level_str == 'CRITICAL':
                self.log_level = logging.CRITICAL
            else:
                raise ValueError(f"Invalid log level string: {self.log_level}")
        else:
            raise ValueError(f"Invalid value for rank_termination_signal: {self.log_level}")

    def __post_init__(self):
        self._fix_rank_termination_signal_type()
        self._fix_log_level_type()
