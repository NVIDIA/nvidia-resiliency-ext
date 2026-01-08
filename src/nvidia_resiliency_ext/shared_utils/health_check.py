# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import os
import shlex
import subprocess
import sys
import threading
import traceback
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import quote_plus

import defusedxml.ElementTree as ET
import httpx
from pydantic import BaseModel

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

# Get the nvrx logger
logger = logging.getLogger(LogConfig.name)


# -----------------------------
# Local utility helpers
# -----------------------------
def _run_shell(cmd: str, timeout: int = 55) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        return -9, e.stdout or "", e.stderr or "timeout"
    except Exception as e:
        return -1, "", str(e)


# Adds basic thread safety, allowing to run health checks from multiple threads.
# This is needed for rendezvous unit tests. NOTE: It will work as long as each
# function/method that uses NVML performs NVML initialization and shutdown.
# Please follow this pattern when adding new code.
_nvml_lock = threading.RLock()


def with_pynvml_lock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _nvml_lock:
            return func(*args, **kwargs)

    return wrapper


class PynvmlMixin:
    def __init__(self):
        # Initialize pynvml to None
        self.pynvml = None

    def check_pynvml_availability(self) -> bool:
        try:
            import pynvml

            self.pynvml = pynvml
            return True
        except ImportError:
            logger.warning("Pynvml is not installed.")
            return False

    def is_gb200_platform(self) -> bool:
        """
        Detect if the current platform is GB200.

        Since all nodes are homogeneous on GPUs, we only need to check GPU 0.
        This allows for platforms with different numbers of GPUs (4, 8, etc.).

        Returns:
            bool: True if GB200 platform is detected, False otherwise.
        """
        if not self.pynvml:
            return False

        try:
            self.pynvml.nvmlInit()
            num_gpus = self.pynvml.nvmlDeviceGetCount()

            # Need at least one GPU to check
            if num_gpus == 0:
                return False

            # Check only GPU 0 since all nodes are homogeneous
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            name = self.pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # GB200 GPUs have names like "GB200" or "B200"
            return any(gpu_type in name.upper() for gpu_type in ["GB200", "B200"])

        except self.pynvml.NVMLError as e:
            logger.debug(f"NVML Error while detecting GB200: {e}")
            return False
        finally:
            try:
                self.pynvml.nvmlShutdown()
            except self.pynvml.NVMLError:
                pass

        return False

    def get_gb200_static_mapping(self) -> dict:
        """
        Get the static GPU to NIC mapping for GB200 platform.

        Returns:
            dict: Mapping from GPU rank to NIC name for GB200.
        """
        return {0: "mlx5_0", 1: "mlx5_1", 2: "mlx5_3", 3: "mlx5_4"}

    @with_pynvml_lock
    def get_gpu_pci_mapping(self):
        """
        Retrieve GPU local rank to PCI Bus mapping using pynvml.
        """
        assert (
            self.pynvml is not None
        ), "pynvml is not initialized. Ensure check_pynvml_availability() is called first."

        gpu_pci_map = {}
        try:
            self.pynvml.nvmlInit()
            num_gpus = self.pynvml.nvmlDeviceGetCount()

            for i in range(num_gpus):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                pci_info = self.pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id = pci_info.busId
                if isinstance(bus_id, bytes):
                    bus_id = bus_id.decode('utf-8')
                bus_id = bus_id.lower()
                # Extract the last 12 characters (standard PCI format)
                gpu_pci_map[i] = bus_id[-12:]

        except self.pynvml.NVMLError as e:
            logger.error(f"NVML Error: {e}\n{traceback.format_exc()}")

        finally:
            try:
                self.pynvml.nvmlShutdown()
            except self.pynvml.NVMLError as e:
                logger.error(f"Failed to shut down NVML: {e}")

        return gpu_pci_map


class PciMixin:
    """
    Mixin to interact with PCI devices.
    """

    def __init__(self):
        pass

    def get_pci_ancestor(self, pci_bus_id: str):
        """
        Walk up the PCI hierarchy and return a list of ancestors, from closest to root.

        Args:
          pci_bus_id (str): 12-byte PCI bus id (e.g. "0000:19:00.0")
        """
        path = os.path.join("/sys/bus/pci/devices", pci_bus_id)
        ancestors = []

        while os.path.exists(path):
            ancestors.append(pci_bus_id)
            parent_path = os.path.realpath(os.path.join(path, ".."))
            parent_bus_id = os.path.basename(parent_path)

            # Stop if we reach the root
            if parent_bus_id == pci_bus_id or not parent_bus_id.startswith("0000:"):
                break

            pci_bus_id = parent_bus_id
            path = parent_path

        return ancestors  # Closest device (self) to root

    def get_ib_pci_mapping(self):
        """
        Retrieve IB device to PCI bus mapping from /sys/class/infiniband
        """
        ib_pci_map = {}
        infiniband_path = "/sys/class/infiniband"

        try:
            for ib_device in sorted(os.listdir(infiniband_path)):
                ib_device_path = os.path.realpath(os.path.join(infiniband_path, ib_device))

                # Extract PCI bus ID (last 12 characters in standard format)
                pci_bus_id = ib_device_path.split("/")[-3][-12:]
                ib_pci_map[ib_device] = pci_bus_id
        except FileNotFoundError:
            logger.warning("No such file or directory: %s" % infiniband_path)

        return ib_pci_map


class GPUHealthCheck(PynvmlMixin):
    def __init__(
        self,
        device_index: Optional[int] = None,
        interval: int = 60,
        on_failure: Optional[Callable] = None,
    ):
        """
        Initializes the GPUHealthCheck class.

        Args:
            device_index (Optional[int]): GPU device index to check. If None, checks all GPUs.
            interval (int): Interval in seconds between asynchronous health checks.
            on_failure (Optional[Callable]): Callback function to handle health check failures.
        """
        super().__init__()
        logger = logging.getLogger(LogConfig.name)
        self.device_index = device_index
        self.interval = interval
        self.on_failure = on_failure
        self.pynvml_available = self.check_pynvml_availability()
        self.enabled = self._check_driver_version()

    @with_pynvml_lock
    def _check_driver_version(self) -> bool:
        """
        Checks if the GPU driver version supports health checks (version r570 or newer).

        Returns:
            bool: True if the driver supports health checks, False otherwise.
        """
        GPU_RECOVERY_API_MIN_DRIVER_VERSION = 570

        if not self.pynvml_available:
            logger.warning("GPU Health checks are disabled because pynvml is not available.")
            return False

        try:
            self.pynvml.nvmlInit()
            driver_version = self.pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')
            self.pynvml.nvmlShutdown()

            major_version = int(driver_version.split('.')[0])

            if major_version < GPU_RECOVERY_API_MIN_DRIVER_VERSION:
                logger.warning(
                    f"Health checks disabled: GPU driver version r{major_version} is older than "
                    f"required r{GPU_RECOVERY_API_MIN_DRIVER_VERSION} for the GPU Recovery API."
                )
                return False
            return True

        except Exception as e:
            logger.warning(
                f"GPU Health checks disabled: Unable to determine driver version due to: {e}"
            )
            return False

    async def async_check(self) -> None:
        """
        Asynchronous GPU health check that runs periodically.

        Periodically checks GPU health and handles any failures if they occur.
        """
        if not self.enabled:
            return

        while True:
            await asyncio.sleep(self.interval)
            result = await self._check_health()
            if not result and self.on_failure:
                await self.on_failure()

    async def _check_health(self) -> bool:
        """
        Performs the asynchronous GPU health check.

        Returns:
            bool: True if all GPUs are healthy, False if any GPU has an issue.
        """
        return self._perform_health_check()

    def __call__(self) -> Union[Optional[Exception], bool]:
        """
        Synchronous GPU health check callable.

        Returns:
            bool: Returns True if GPUs are healthy.
        """
        if not self.enabled:
            logger.warning("Health checks are disabled; skipping synchronous check.")
            return True

        result = self._perform_health_check()
        return result

    @with_pynvml_lock
    def _perform_health_check(self) -> bool:
        """
        Core method to perform GPU health check. Used by both sync and async checks.

        Checks the recovery action needed for the specified GPU device(s).

        Returns:
            bool: True if all specified GPUs are healthy (no recovery action needed), False otherwise.
        """
        try:
            self.pynvml.nvmlInit()

            # Determine which devices to check
            devices_to_check = (
                [self.device_index]
                if self.device_index is not None
                else range(self.pynvml.nvmlDeviceGetCount())
            )

            # Check all specified devices
            for device_id in devices_to_check:
                if not self._check_gpu_health(device_id):
                    return False

            return True

        except self.pynvml.NVMLError as e:
            logger.warning(f"NVML Error: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected Error: {str(e)}")
            return False
        finally:
            try:
                self.pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Error during NVML shutdown: {str(e)}")

    def _check_gpu_health(self, device_id: int) -> bool:
        """
        Check health for a specific GPU device.

        Args:
            device_id (int): GPU device index to check.

        Returns:
            bool: True if GPU is healthy, False otherwise.
        """
        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)

            if not hasattr(self.pynvml, "NVML_FI_DEV_GET_GPU_RECOVERY_ACTION"):
                # PyNVML is not available so we assume the GPU is healthy
                return True

            # Get the GPU recovery action status
            recovery_action = self.pynvml.nvmlDeviceGetFieldValues(
                handle, [self.pynvml.NVML_FI_DEV_GET_GPU_RECOVERY_ACTION]
            )[0].value.uiVal

            # Interpret the recovery action
            if recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_NONE:
                return True
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_GPU_RESET:
                logger.warning(
                    f"GPU {device_id}: Requires a reset to recover. Terminate GPU processes and reset the GPU."
                )
                return False
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_NODE_REBOOT:
                logger.warning(
                    f"GPU {device_id}: Requires a node reboot to recover. Reboot the system."
                )
                return False
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_DRAIN_P2P:
                logger.warning(
                    f"GPU {device_id}: Requires peer-to-peer traffic to be drained. Terminate related processes."
                )
                return False
            elif recovery_action == self.pynvml.NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET:
                logger.warning(
                    f"GPU {device_id}: Operating at reduced capacity. Drain existing work and reset the GPU."
                )
                return False
            else:
                logger.warning(
                    f"GPU {device_id}: Unknown recovery action status: {recovery_action}"
                )
                return False

        except self.pynvml.NVMLError as e:
            logger.warning(f"NVML Error: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected Error: {str(e)}")
            return False


class NicHealthCheck(PynvmlMixin, PciMixin):
    # Default path template for IB link down counter
    DEFAULT_LINK_DOWN_PATH = "/sys/class/infiniband/{nic}/ports/1/counters/link_downed"

    def __init__(
        self,
        device_index: Optional[int] = None,
        interval: int = 60,
        pci_topo_file: Optional[str] = None,
        link_down_path_template: Optional[str] = None,
        on_failure: Optional[Callable] = None,
    ):
        """
        Initializes the NicHealthCheck class. 'pci_topo_file' is required on some CSPs (e.g. Azure) to describe
        GPUs and NIC topology, where the PCIe Bus ID is obfusticated in the VM.

        Args:
            device_index (Optional[int]): GPU device index to monitor. If provided, automatically sets the NIC device
                                        and initializes the baseline counter during construction.
            interval (int): Interval in seconds between asynchronous health checks.
            pci_topo_file (Optional[str]): The topo file describes the hardware topology of a system,
                                            specifically how CPUs and PCI devices (like GPUs and NICs) are connected.
            link_down_path_template (Optional[str]): Template string for the link down counter path.
                                                    Use {nic} as placeholder for NIC name.
            on_failure (Optional[Callable]): Callback function to handle health check failures.
        """
        self.interval = interval
        self.pci_topo_file = pci_topo_file
        self.on_failure = on_failure
        self.nic_name = None

        # GPU local rank number.
        self._local_rank = None

        # NIC health-check checks on
        # "/sys/class/infiniband/{device}/ports/{port}/counters/link_downed" counter.
        # It declares a failure if the counter is incremented from last check.
        self._prev_link_downed = -1

        if self.check_pynvml_availability():
            self._get_gpu_ib_mapping()
        else:
            logger.warning("Failed to import pynvml. Nic health checks disabled.")
            self._gpu_ib_map = None

        self.link_down_path_template = link_down_path_template or self.DEFAULT_LINK_DOWN_PATH

        # If device_index is provided, automatically set the NIC device and initialize baseline
        if device_index is not None:
            self.set_nic_device(device_index)

    def _get_gpu_ib_mapping(self):
        """
        Find GPU local rank to closest IB/NIC mapping.

        Algorithm:
        1. Extract GPU PCI Bus IDs from nvidia-smi.
        2. Extract IB/NIC PCI Bus IDs from /sys/class/infiniband symlinks.
        3. Find the closest NIC for each GPU by identifying their nearest common ancestor in the PCI hierarchy.
        4. If there are multiple NICs having the same nearest common ancestor, favors the NIC that hasn't been assigned.
        5. If automatic derivation fails and platform is GB200, use static mapping table.
        """
        gpu_pci_map = self.get_gpu_pci_mapping()
        ib_pci_map = self.get_ib_pci_mapping()
        logger.info("gpu_pci_map: %s ib_pci_map: %s" % (gpu_pci_map, ib_pci_map))

        if self.pci_topo_file is not None:
            assignments = self._get_gpu_ib_assignments_from_topo(gpu_pci_map, ib_pci_map)
        else:
            assignments = self._get_gpu_ib_assignments_from_system(gpu_pci_map, ib_pci_map)

        gpu_ib_map = {}
        used_ib_devices = set()  # Track assigned IB devices
        # First pass: Greedy assignment ensuring uniqueness of IB devices for each GPU
        for gpu_rank, ib_dev, _ in assignments:
            if gpu_rank not in gpu_ib_map and ib_dev not in used_ib_devices:
                gpu_ib_map[gpu_rank] = ib_dev
                used_ib_devices.add(ib_dev)

        # Second pass: Handle remaining GPUs and assign them to already used IB devices
        for gpu_rank, ib_dev, _ in assignments:
            if gpu_rank not in gpu_ib_map:
                gpu_ib_map[gpu_rank] = ib_dev  # Assign even if it's the same IB device

        # If automatic derivation failed and we're on GB200 platform, use static mapping
        if not gpu_ib_map and self.is_gb200_platform():
            logger.info("Automatic GPU-IB mapping failed, using GB200 static mapping")
            gpu_ib_map = self.get_gb200_static_mapping()

        self._gpu_ib_map = gpu_ib_map or None
        logger.info("gpu_ib_map: %s" % self._gpu_ib_map)

    def _get_gpu_ib_assignments_from_system(self, gpu_pci_map: dict, ib_pci_map: dict):
        """
        Walk the system PCI device tree to find all GPU to IB device assignments.
        """
        # Collect all (gpu_rank, ib_device, depth) tuples
        assignments = []
        for gpu_rank, gpu_pci in gpu_pci_map.items():
            gpu_ancestors = self.get_pci_ancestor(gpu_pci)
            for ib_dev, ib_pci in ib_pci_map.items():
                nic_ancestors_list = self.get_pci_ancestor(ib_pci)
                # Find lowest common ancestor
                common_ancestor = next(
                    (node for node in gpu_ancestors if node in nic_ancestors_list), None
                )
                if common_ancestor:
                    # The total number of hops taken by both paths before they meet at the common ancestor
                    depth = gpu_ancestors.index(common_ancestor) + nic_ancestors_list.index(
                        common_ancestor
                    )
                    assignments.append((gpu_rank, ib_dev, depth))

        # Sort by depth (lower is better)
        assignments.sort(key=lambda x: x[2])
        return assignments

    def _get_gpu_ib_assignments_from_topo(self, gpu_pci_map: dict, ib_pci_map: dict):
        """
        Generate GPU to IB device assignments from the given topo file.
        """
        # Parse topology file.
        topo_map = self._parse_topo_file(self.pci_topo_file)

        # Reverse topo_map to find parent PCI bridge for each PCI device.
        device_to_parent = {
            dev: parent for parent, children in topo_map.items() for dev in children
        }

        # Generate all possible (GPU, IB) assignments.
        assignments = []
        for gpu_rank, gpu_pci in gpu_pci_map.items():
            parent_pci = device_to_parent.get(gpu_pci, None)
            if not parent_pci:
                logger.warning("Failed to find GPU pci_bus_id: %s in the topo file." % (gpu_pci))
                continue  # Skip if GPU is not found in the topo mapping

            # Find IB devices under the same parent PCI bridge
            ib_candidates = [
                ib for ib, ib_pci in ib_pci_map.items() if ib_pci in topo_map.get(parent_pci, [])
            ]

            for ib_dev in ib_candidates:
                assignments.append((gpu_rank, ib_dev, 0))
        return assignments

    def _parse_topo_file(self, file_path):
        """
        Parse topology XML file into a parent PCI bridge to child PCI device mapping.
        """
        pci_mapping = defaultdict(list)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            for pci_bridge in root.findall(".//pci"):
                parent_busid = pci_bridge.get("busid")
                child_devices = [child.get("busid") for child in pci_bridge.findall("pci")]

                if parent_busid and child_devices:
                    pci_mapping[parent_busid].extend(child_devices)

        except ET.ParseError as e:
            logger.error(f"XML Parsing error in {file_path}: {e}")
        except FileNotFoundError:
            logger.error(f"Topology file not found: {file_path}")
        except OSError as e:
            logger.error(f"Error opening topology file {file_path}: {e}")

        return dict(pci_mapping)

    def set_nic_device(self, local_rank: int):
        """
        Set the closest NIC/IB device to monitor for a given GPU.

        Args:
            local_rank (int): Local rank of the GPU.
        """
        if self._gpu_ib_map is None:
            logger.error(
                f"gpu_ib_map is empty. Disable NIC health check for local_rank: {local_rank}"
            )
            return

        self._local_rank = local_rank
        self.nic_name = self._gpu_ib_map.get(local_rank, None)
        if self.nic_name is None:
            logger.error(
                f"GPU missing in gpu_ib_map. Disable NIC health check for local_rank: {local_rank}"
            )
            return

        # Initialize the baseline link_downed counter for proper delta calculation
        self._perform_health_check()

        logger.info(
            "Local rank: %s Nic name: %s baseline counter: %s"
            % (self._local_rank, self.nic_name, self._prev_link_downed)
        )

    async def async_check(self) -> None:
        """
        Asynchronous NIC/IB health check that runs periodically.

        Periodically checks NIC/IB health and handles any failures if they occur.
        """
        while True:
            await asyncio.sleep(self.interval)
            if self.nic_name is None:
                continue
            result = await self._check_health()
            if not result and self.on_failure:
                await self.on_failure()

    async def _check_health(self) -> bool:
        """
        Performs the asynchronous NIC health check.

        Returns:
            bool: True if NIC/IB is healthy, False if it has an issue.
        """
        return self._perform_health_check()

    def __call__(self) -> Union[Optional[Exception], bool]:
        """
        Synchronous NIC health check callable.

        Returns:
            bool: Returns True if NIC is healthy.
        """
        if self.nic_name is None:
            logger.warning("NIC health check is disabled; skipping synchronous check.")
            return True

        result = self._perform_health_check()
        return result

    def _perform_health_check(self) -> bool:
        """
        Core method to perform NIC health check. Used by both sync and async checks.

        Returns:
            bool: True if NIC/IB is healthy, False otherwise.
        """
        link_downed_path = self.link_down_path_template.format(nic=self.nic_name)
        if not os.path.exists(link_downed_path):
            logger.warning(
                "NIC/IB: %s link_downed_path not exists: %s" % (self.nic_name, link_downed_path)
            )
            return True

        try:
            # Read the current counter value
            with open(link_downed_path, 'r') as ff:
                link_downed_value = int(ff.read().strip())

            # Check if the counter has been incremented
            if self._prev_link_downed >= 0 and link_downed_value > self._prev_link_downed:
                logger.warning(
                    "GPU %s NIC/IB %s link down counter has been incremented: %s -> %s "
                    % (self._local_rank, self.nic_name, self._prev_link_downed, link_downed_value)
                )
                return False
            self._prev_link_downed = link_downed_value
        except Exception:
            logger.warning(
                "Exception while reading link_downed counter: %s" % traceback.format_exc()
            )

        return True


class NicLinkStateHealthCheck(PciMixin):
    """Health check for NIC link state monitoring (RDMA and Ethernet).

    This class monitors the state of network interface ports (both RDMA/InfiniBand
    and Ethernet) to detect when ports transition from ACTIVE to non-ACTIVE states.
    It takes a baseline snapshot of all port states during initialization and compares
    against this baseline during health checks.
    """

    # Default path template for NIC link state
    DEFAULT_LINK_STATE_PATH = "/sys/class/infiniband/{nic}/ports/1/state"

    def __init__(
        self,
        link_state_path_template: Optional[str] = None,
        on_failure: Optional[Callable] = None,
    ):
        """
        Initializes the NicLinkStateHealthCheck class.

        Args:
            link_state_path_template (Optional[str]): Template string for the link state path.
                                                      Use {nic} as placeholder for NIC name.
            on_failure (Optional[Callable]): Callback function to handle health check failures.
        """
        super().__init__()
        self.link_state_path_template = link_state_path_template or self.DEFAULT_LINK_STATE_PATH
        self.on_failure = on_failure

        # Snapshot of initial port states: {nic_name: state_string}
        self._baseline_port_states: Dict[str, str] = {}

        # Initialize baseline by scanning all network devices
        self._initialize_baseline()

    def _initialize_baseline(self):
        """
        Initialize the baseline snapshot of all NIC port states.

        This scans all available network devices (RDMA/InfiniBand and Ethernet) and
        records their current port states as the baseline for future comparisons.
        """
        infiniband_path = "/sys/class/infiniband"

        if not os.path.exists(infiniband_path):
            logger.warning(f"InfiniBand path does not exist: {infiniband_path}")
            return

        try:
            for ib_device in sorted(os.listdir(infiniband_path)):
                state_path = self.link_state_path_template.format(nic=ib_device)

                if not os.path.exists(state_path):
                    logger.debug(f"NIC link state path not found: {state_path}")
                    continue

                try:
                    with open(state_path, 'r') as f:
                        state = f.read().strip()
                        # Parse state - format is usually like "4: ACTIVE" or just "ACTIVE"
                        # We want to extract the state name after any colon
                        if ':' in state:
                            state = state.split(':', 1)[1].strip()
                        self._baseline_port_states[ib_device] = state
                except Exception as e:
                    logger.warning(f"Failed to read baseline state for {ib_device}: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to scan network devices: {str(e)}")

        if not self._baseline_port_states:
            logger.warning("No network devices found for baseline initialization")
        else:
            # Log all baseline states in a single consolidated message
            devices_summary = ", ".join(
                [f"{dev}={state}" for dev, state in self._baseline_port_states.items()]
            )
            logger.debug(f"NIC baseline states: {devices_summary}")

    def __call__(self) -> bool:
        """
        Synchronous NIC link state health check callable.

        Returns:
            bool: Returns True if all ports remain in their baseline state (or ACTIVE),
                  False if any port has transitioned from ACTIVE to non-ACTIVE.
        """
        return self._perform_health_check()

    def _perform_health_check(self) -> bool:
        """
        Core method to perform NIC link state health check.

        Compares current port states against the baseline. If any port that was
        initially ACTIVE is now in a different (non-ACTIVE) state, this is
        considered a health check failure.

        Returns:
            bool: True if all ports are healthy, False otherwise.
        """
        if not self._baseline_port_states:
            # No baseline established, skip the check
            logger.debug("No baseline port states available, skipping NIC link state check")
            return True

        all_healthy = True

        for ib_device, baseline_state in self._baseline_port_states.items():
            state_path = self.link_state_path_template.format(nic=ib_device)

            if not os.path.exists(state_path):
                logger.debug(f"NIC link state path no longer exists: {state_path}")
                continue

            try:
                with open(state_path, 'r') as f:
                    current_state = f.read().strip()
                    # Parse state - format is usually like "4: ACTIVE" or just "ACTIVE"
                    if ':' in current_state:
                        current_state = current_state.split(':', 1)[1].strip()

                    # Check if port transitioned from ACTIVE to non-ACTIVE
                    if baseline_state == "ACTIVE" and current_state != "ACTIVE":
                        logger.error(
                            f"NIC device {ib_device}: Port state changed from "
                            f"{baseline_state} to {current_state}. Health check FAILED."
                        )
                        all_healthy = False
                    elif current_state != baseline_state:
                        logger.warning(
                            f"NIC device {ib_device}: Port state changed from "
                            f"{baseline_state} to {current_state}"
                        )
            except Exception as e:
                logger.warning(
                    f"Exception while reading link state for {ib_device}: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )

        return all_healthy


class NVLHealthCheck(PynvmlMixin):
    def __init__(
        self,
        device_index: Optional[int] = None,
        interval: int = 60,
        on_failure: Optional[Callable] = None,
    ):
        """
        Initializes the NVLHealthCheck class.

        Args:
            device_index (Optional[int]): GPU device index to check. If None, checks all GPUs.
            interval (int): Interval in seconds between asynchronous health checks.
            on_failure (Optional[Callable]): Callback function to handle health check failures.
        """
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.on_failure = on_failure
        self.pynvml_available = self.check_pynvml_availability()

    async def async_check(self) -> None:
        """
        Asynchronous NVL health check that runs periodically.

        Periodically checks NVL health and handles any failures if they occur.
        """
        while True:
            await asyncio.sleep(self.interval)
            result = await self._check_health()
            if not result and self.on_failure:
                await self.on_failure()

    async def _check_health(self) -> bool:
        """
        Performs the asynchronous NVL health check.

        Returns:
            bool: True if NVL links are healthy, False if any link has an issue.
        """
        return self._perform_health_check()

    def __call__(self) -> Union[Optional[Exception], bool]:
        """
        Synchronous NVL health check callable.

        Returns:
            bool: Returns True if NVL links are healthy.
        """
        result = self._perform_health_check()
        return result

    @with_pynvml_lock
    def _perform_health_check(self) -> bool:
        """
        Core method to perform NVL health check. Used by both sync and async checks.

        Checks the state of all NVL links for the specified GPU device(s).

        Returns:
            bool: True if all NVL links are healthy, False otherwise.
        """
        if not self.pynvml_available:
            return True

        try:
            self.pynvml.nvmlInit()

            # Determine which devices to check
            devices_to_check = (
                [self.device_index]
                if self.device_index is not None
                else range(self.pynvml.nvmlDeviceGetCount())
            )

            # Check all specified devices
            all_healthy = True
            for device_id in devices_to_check:
                if not self._check_nvl_links_for_device(device_id):
                    all_healthy = False

            return all_healthy

        finally:
            try:
                self.pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Error during NVML shutdown: {str(e)}")

    def _check_nvl_links_for_device(self, device_index: int) -> bool:
        """
        Check NVL links for a specific GPU device.

        Args:
            device_index (int): GPU device index to check.

        Returns:
            bool: True if all NVL links are healthy, False otherwise.
        """
        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # Check all NVL links for this device
            for link_id in range(self.pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    link_state = self.pynvml.nvmlDeviceGetNvLinkState(handle, link_id)

                    if link_state == self.pynvml.NVML_FEATURE_DISABLED:
                        logger.warning(
                            f"GPU {device_index}: NVL link {link_id} is in DISABLED state"
                        )
                        return False

                except self.pynvml.NVMLError as e:
                    # Link might not exist or be accessible, which is normal
                    # Only log if it's not a "not supported" error
                    if "not supported" not in str(e).lower():
                        logger.warning(
                            f"GPU {device_index}: NVL link {link_id} not accessible: {e}"
                        )
                    continue

            return True

        except self.pynvml.NVMLError as e:
            logger.warning(f"NVML Error checking GPU {device_index}: {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected Error checking GPU {device_index}: {str(e)}")
            return False


class NodeHealthCheck:
    """
    Node-level health check that queries the node health check service used by InJob via gRPC over a Unix domain socket (UDS).
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        endpoint: Optional[str] = None,
        args: Optional[list] = None,
        interval: int = 60,
        on_failure: Optional[Callable] = None,
        request_timeout_sec: Optional[float] = 60,
    ):
        self.socket_path = socket_path
        self.endpoint = endpoint
        self.args = args or []
        self.interval = interval
        self.on_failure = on_failure
        self.request_timeout_sec = request_timeout_sec
        self._channel_target: Optional[str] = None

        # Optional imports; health check becomes a no-op if gRPC/protos aren't available
        self._grpc = None
        self._pb2 = None
        self._pb2_grpc = None
        try:
            import sys as _sys

            import grpc as _grpc_mod

            # Relative import from package
            from .proto import nvhcd_pb2 as _pb2_mod

            _sys.modules.setdefault("nvhcd_pb2", _pb2_mod)
            from .proto import nvhcd_pb2_grpc as _pb2_grpc_mod

            self._grpc = _grpc_mod
            self._pb2 = _pb2_mod
            self._pb2_grpc = _pb2_grpc_mod
        except Exception as e:
            logger.debug(f"Node health check gRPC client unavailable: {e}")
        finally:
            # Pre-validate and cache a usable channel target (or None to skip)
            self._channel_target = self._validate_and_get_target()

    async def async_check(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            result = await self._check_health()
            if not result and self.on_failure:
                await self.on_failure()

    async def _check_health(self) -> bool:
        return self._perform_health_check()

    def __call__(self) -> Union[Optional[Exception], bool]:
        return self._perform_health_check()

    def _resolve_channel_target(self) -> Optional[str]:
        """
        Resolve endpoint into a gRPC channel target (UDS only).
        - If endpoint is a path (starts with '/' or './'), prefix with unix://
        - If endpoint already starts with 'unix://', use as is
        - If no endpoint provided, fall back to socket_path
        - Any non-UDS formats are ignored (return None)
        """
        target = self.endpoint
        if not target:
            # Backward compatibility: use socket_path
            if self.socket_path:
                if self.socket_path.startswith("unix://"):
                    target = self.socket_path
                else:
                    target = f"unix://{self.socket_path}"
        else:
            if target.startswith("unix://"):
                # Already UDS
                pass
            elif target.startswith("/") or target.startswith("./"):
                target = f"unix://{target}"
            else:
                # Non-UDS target not supported at this time
                logger.debug(
                    f"Health check endpoint '{target}' is not a UDS; skipping node health check."
                )
                return None
        return target

    def _perform_health_check(self) -> bool:
        """
        Query node health check service over UDS and interpret success as healthy.
        Behavior:
          - If gRPC/protos are unavailable, or UDS socket is missing, return True (non-fatal/optional check).
          - On gRPC connectivity errors, return False.
        """
        # Use pre-validated target computed during initialization
        target = self._channel_target
        if not target:
            return True

        try:
            with self._grpc.insecure_channel(target) as channel:
                stub = self._pb2_grpc.HealthCheckServiceStub(channel)
                request = self._pb2.HealthCheckRequest(args=["--no-slurm"])
                if self.request_timeout_sec is not None:
                    response = stub.RunHealthCheck(request, timeout=self.request_timeout_sec)
                else:
                    response = stub.RunHealthCheck(request)

                if not getattr(
                    response, "success", False
                ):  # If the health check failed, return False
                    exit_code = getattr(response, "exit_code", None)
                    output = getattr(response, "output", "")
                    error = getattr(response, "error", "")
                    msg = f"Node health check failed (exit_code={exit_code}). Output: {output}"
                    if error:
                        msg += f" Error: {error}"
                    logger.warning(msg)
                    return False

                logger.debug("Node health check: success")
                return True

        except Exception as e:
            logger.warning(
                f"Node health check gRPC connectivity error: {str(e)}; treating as healthy (skip)."
            )
            return True

    def _validate_and_get_target(self) -> Optional[str]:
        """
        Validate gRPC/proto availability and UDS target before making RPC call.
        Returns a usable channel target string or None to indicate skipping the check.
        """
        # If gRPC client cannot be constructed, return None (non-fatal/optional check)
        if self._grpc is None or self._pb2 is None or self._pb2_grpc is None:
            return None

        # Resolve endpoint/target
        target = self._resolve_channel_target()
        if not target:
            return None

        # If using UDS and socket does not exist, assume service not deployed and skip
        if target.startswith("unix://"):
            uds_path = target[len("unix://") :]
            if not os.path.exists(uds_path):
                logger.debug(
                    f"Node health check socket not found at {uds_path}; skipping node health check."
                )
                return None

        return target


# -----------------------------
# Distributed storage checks integration (Lustre + NFS)
# -----------------------------
"""
Distributed Storage Health + Mounts Check

1) Lustre health verification via /sys/fs/lustre/health_check
2) Filesystem mounts verification (Lustre and NFS)
"""


class DistributedStorageHealthCheck:
    def __init__(
        self,
        interval: int = 60,
        on_failure: Optional[Callable] = None,
    ):
        self.interval = interval
        self.on_failure = on_failure

    async def async_check(self) -> None:
        """
        Periodically runs distributed storage checks and invokes on_failure when a failure is detected.
        """
        while True:
            await asyncio.sleep(self.interval)
            result = await self._check_health()
            if not result and self.on_failure:
                await self.on_failure()

    async def _check_health(self) -> bool:
        """
        Asynchronous wrapper for the health check call.
        """
        return self._perform_health_check()

    def __call__(self) -> bool:
        """
        Synchronous distributed storage health check callable.
        """
        return self._perform_health_check()

    def _perform_health_check(self) -> bool:
        """
        Execute distributed storage checks; returns True if all selected checks pass.
        """
        try:
            return self.run_storage_checks()
        except Exception as e:
            logger.warning(f"distributed storage health checks execution error: {e}")
            return False

    def _check_lustre_health(self) -> bool:
        """
        Check Lustre health status file. Returns:
          - True on success (status 'healthy' or file missing)
          - False on failure (read error or status not 'healthy')
        """
        health_path = "/sys/fs/lustre/health_check"
        if os.path.exists(health_path):
            try:
                with open(health_path, "r") as fh:
                    status = fh.read().strip()
                if status == "healthy":
                    logger.debug("lustre health_check OK: healthy")
                    return True
                logger.warning(f"lustre health_check not healthy: '{status}'")
                return False
            except Exception as exc:
                logger.warning(f"error reading {health_path}: {exc}")
                return False
        # File missing is non-fatal since might not be container mounted
        logger.debug(f"{health_path} not found, skipping health status check")
        return True

    def run_storage_checks(self):
        """Run distributed storage health checks and mounts validation (Lustre + NFS)"""
        ok = True

        # 1) Lustre health; if it fails, skip mounts and return False
        if not self._check_lustre_health():
            return False

        # 2) Mounts reachability validation: discover mounts and verify their directories are reachable
        fs_types = ["lustre", "nfs", "nfs4"]
        types_str = ",".join(fs_types)
        # Discover mount targets within the current namespace (container-aware)
        rc, out, err = _run_shell(
            f"findmnt --noheadings --type={types_str} --output=target", timeout=55
        )
        if rc != 0:
            logger.warning(f"error running findmnt for types {types_str}: {err}")
            ok = False
            mount_dirs: list[str] = []
        else:
            # Extract target mount directories directly (container mount points)
            mount_dirs = [line.strip() for line in out.splitlines() if line.strip()]
            logger.debug(
                f"findmnt succeeded for types {types_str}: discovered {len(mount_dirs)} target(s)"
            )
            if not mount_dirs:
                logger.debug(f"no {types_str} mount directories discovered")

        if mount_dirs:
            # First, verify reachability of mount roots via timed shell ops
            unreachable = []
            for d in mount_dirs:
                q = shlex.quote(d)
                # Check directory type
                rc_d, _, _ = _run_shell(f"test -d {q}", timeout=15)
                if rc_d != 0:
                    unreachable.append(f"{d} (not a directory)")
                    continue
                # Attempt to list directory to ensure it is reachable
                rc_list, out_list, err_list = _run_shell(f"ls -1 -- {q}", timeout=15)
                if rc_list != 0:
                    msg = err_list or out_list or "list error"
                    unreachable.append(f"{d} ({msg.strip()})")
            if unreachable:
                logger.warning("unreachable mount directories:\n" + "\n".join(unreachable))
                ok = False
            else:
                logger.debug("all mount directories reachable:\n" + "\n".join(mount_dirs))

        return ok


# -----------------------------
# Storage path health checks
# -----------------------------


class StoragePathHealthCheck:
    """
    Validate a list of absolute paths by checking existence.
    - Input: list of absolute paths (e.g., ["/data/a", "/mnt/b"])
    - Behavior: logs invalid (non-absolute) and missing paths; returns False if any issue.
    """

    def __init__(self, paths: list[str], on_failure: Optional[Callable] = None):
        self.paths = paths
        self.on_failure = on_failure

    def __call__(self) -> bool:
        return self._perform_health_check()

    def _perform_health_check(self) -> bool:
        """
        Execute path probes in a separate Python subprocess to avoid hangs on remote
        filesystems.
        """
        args = [
            sys.executable,
            "-m",
            "nvidia_resiliency_ext.shared_utils.storage_probe",
        ] + self.paths

        # Timeout: 5s per path (conservative)
        timeout = 5 * max(1, len(self.paths))

        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("StoragePathHealthCheck probe timed out")
            return False
        except Exception as e:
            logger.warning(f"StoragePathHealthCheck probe failed: {e}")
            return False

        if proc.returncode != 0:
            # Probe subprocess signalled failure via non-zero exit code.
            stderr = (proc.stderr or "").strip()
            logger.warning(
                f"StoragePathHealthCheck probe subprocess failed (rc={proc.returncode}): {stderr}"
            )
            return False

        # Success: returncode == 0
        logger.debug("StoragePathHealthCheck probe subprocess exited with rc=0")
        if self.paths:
            logger.debug("all storage paths accessible:\n" + "\n".join(self.paths))
        return True


class AttrSvcResult(BaseModel):
    result: Any
    status: str = "completed"


class AttributionService:
    """
    Client that queries an external attribution service to analyze artifacts (e.g., logs).
    Behavior:
      - POSTs to submit for log analysis
      - GETs results by the last submitted log_path
    """

    def __init__(
        self,
        log_path: Optional[str],
        host: str,
        port: int,
    ):
        self.host = host
        self.port = port
        self.log_path = log_path
        # Track the most recent log_path we submitted
        self._last_submitted: Optional[str] = None

    def __call__(self, log_path: Optional[str] = None) -> None:
        """
        Fire-and-forget entrypoint.
        - If an event loop is running, schedules the request with create_task and returns the Task.
        - If no event loop is running, runs in a background thread and returns None.
        """
        effective_path = log_path if log_path else self.log_path
        if effective_path is None:
            raise ValueError("log_path is required (provide at init or call time)")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.get_attrsvc_result_async(effective_path))
        except RuntimeError:
            # No running event loop; run in a background thread without blocking
            threading.Thread(
                target=lambda: asyncio.run(self.get_attrsvc_result_async(effective_path)),
                daemon=True,
            ).start()
        return None

    async def get_attrsvc_result_async(self, log_path: str) -> None:
        """
        Async caller for _get_attrsvc_result.
        """
        await self._get_attrsvc_result(log_path)

    async def _get_attrsvc_result(self, log_path: str) -> None:
        """
        Internal async method that interacts with the external attribution service:
          - If a prior submission exists, GET results for the last submitted path
          - Then, POST the new log_path to submit for analysis
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                create_url = f"http://{self.host}:{self.port}/logs"

                # 1) If we previously submitted a path, GET its results
                if self._last_submitted:
                    q_last = quote_plus(self._last_submitted)
                    get_url = f"{create_url}?log_path={q_last}"
                    try:
                        resp = await client.get(get_url, headers={"accept": "application/json"})
                        if resp.status_code == 200:
                            payload = resp.json() if resp.text else {}
                            result = payload.get("result", payload)
                            status = payload.get("status", "completed")
                            attrsvc_result = AttrSvcResult(result=result, status=status)
                            logger.info("AttrSvcResult status=%s", attrsvc_result.status)
                            logger.info(
                                "AttrSvcResult result preview: %s", str(attrsvc_result.result)[:200]
                            )
                        else:
                            logger.warning(f"AttributionService GET returned {resp.status_code}")
                    except Exception as e:
                        logger.warning(f"AttributionService GET failed: {e}")

                # 2) Submit the new path for analysis
                await client.post(
                    create_url,
                    json={"log_path": log_path},
                    headers={"accept": "application/json"},
                )
                self._last_submitted = log_path
                logger.debug(f"AttributionService submitted: {log_path}")

        except Exception as e:
            # Logging is sufficient; do not propagate exceptions
            logger.warning(f"AttributionService request failed: {e}")
            return None
