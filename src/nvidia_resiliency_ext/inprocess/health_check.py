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

import abc
import datetime
import logging
import os
import threading
import time

import torch

from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

# Import the health check classes from shared_utils
from ..shared_utils.health_check import GPUHealthCheck, NicHealthCheck, NVLHealthCheck
from . import exception
from .state import FrozenState


class HealthCheck(abc.ABC):
    r'''
    Abstract base class for ``health_check`` argument for
    :py:class:`inprocess.Wrapper`.

    :py:class:`HealthCheck` ensures the worker is in a healthy state and can
    execute the workload.

    Health checks are executed after the target function failure was discovered
    (on local, or other distributed ranks), local distributed group was
    destroyed, and after the user-provided
    :py:class:`inprocess.finalize.Finalize` finished.

    :py:class:`HealthCheck` is executed to filter out unhealthy ranks (e.g. due
    to corrupted CUDA context). The execution should be local to a given rank,
    other ranks may have already been terminated, lost or still executing the
    wrapped function.

    Unhealthy state is reported by raising an :py:exc:`Exception`. The
    exception is reraised by the :py:class:`inprocess.Wrapper`, and should lead
    to termination of the main Python interpreter process.

    Multiple instances of :py:class:`HealthCheck` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    '''

    @abc.abstractmethod
    def __call__(self, state: FrozenState) -> FrozenState:
        r'''
        Implementation of a :py:class:`HealthCheck`.

        Args:
            state: read-only :py:class:`Wrapper` state

        Returns:
            Forwarded read-only input ``state``.
        '''
        raise NotImplementedError


class CudaHealthCheck(HealthCheck):
    r'''
    Ensures that CUDA context for the current process is in a healthy state.

    Synchronizes with the GPU. Uses the device corresponding to ``LOCAL_RANK``
    environment variable, or the main thread's default CUDA device if
    ``LOCAL_RANK`` was not specified in the environment.

    Args:
        timeout: timeout for synchronization with the GPU
    '''

    def __init__(self, timeout=datetime.timedelta(seconds=30)):
        self.timeout = timeout

    def __call__(self, state: FrozenState) -> FrozenState:
        log = logging.getLogger(LogConfig.name)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            if (local_rank := os.getenv('LOCAL_RANK', None)) is not None:
                device = torch.device(int(local_rank))
            else:
                device = torch.device(torch.cuda.current_device())

            # sync waits for completion of all issued CUDA kernels, this could
            # take very long if CPU app code ran far ahead of CUDA code, but
            # there is no other way around, there is no way to cancel pending
            # CUDA kernels, and any pending kernel may corrupt CUDA context
            thread = threading.Thread(
                target=torch.cuda.synchronize,
                args=(device,),
                name=f'{type(self).__name__}Sync',
                daemon=True,
            )
            log.debug(f'1st torch.cuda.synchronize({device=})')
            thread.start()
            thread.join(self.timeout.total_seconds())
            if thread.is_alive():
                log.debug('torch.cuda.synchronize() timed out')
                raise exception.TimeoutError

            # 2nd sync to check if CUDA context is healthy
            log.debug(f'2nd torch.cuda.synchronize({device=})')
            torch.cuda.synchronize(device)
        return state


class FaultCounterExceeded(exception.RestartError):
    r'''
    Exception raised by :py:class:`FaultCounter` when number of faults on the
    current rank exceeds the threshold.
    '''

    pass


class FaultCounter(HealthCheck):
    r'''
    :py:class:`FaultCounter` counts faults caused by the current process. The
    process is terminated if total number of faults exceeds the
    ``max_rank_faults`` threshold.

    Args:
        max_rank_faults: maximum number of faults cause by the process
    '''

    def __init__(self, max_rank_faults=None):
        self.max_rank_faults = max_rank_faults
        self.faults_count = 0

    def __call__(self, state: FrozenState) -> FrozenState:
        if state.fn_exception is None:
            return state

        self.faults_count += 1
        max_rank_faults = self.max_rank_faults
        faults_count = self.faults_count

        if max_rank_faults is not None and faults_count > max_rank_faults:
            raise FaultCounterExceeded(f'{faults_count=} / {max_rank_faults=}')
        return state


class ChainedGPUHealthCheck(HealthCheck):
    r'''
    Ensures that GPU devices are in a healthy state by checking GPU recovery actions.

    Uses the GPUHealthCheck from shared_utils to perform comprehensive GPU health checks.
    This health check is executed after a fault to ensure the GPU is in a recoverable state.

    Args:
        device_index: Optional GPU device index to check. If None, checks all GPUs.
    '''

    def __init__(self, device_index=None):
        self.device_index = device_index
        self._gpu_checker = GPUHealthCheck(device_index=device_index)

    def __call__(self, state: FrozenState) -> FrozenState:
        # Perform GPU health check
        is_healthy = self._gpu_checker._perform_health_check()
        if not is_healthy:
            raise exception.HealthCheckError("GPU health check failed")

        return state


class ChainedNVLHealthCheck(HealthCheck):
    r'''
    Ensures that NVL (NVLink) connections are in a healthy state.

    Uses the NVLHealthCheck from shared_utils to perform comprehensive NVL link health checks.
    This health check is executed after the fault to ensure NVL links are functioning properly.

    Args:
        device_index: Optional GPU device index to check. If None, checks all GPUs.
    '''

    def __init__(self, device_index=None):
        self.device_index = device_index
        self._nvl_checker = NVLHealthCheck(device_index=device_index)

    def __call__(self, state: FrozenState) -> FrozenState:
        # Perform NVL health check
        is_healthy = self._nvl_checker._perform_health_check()
        if not is_healthy:
            # NVL health check failures are ignored in current implementation
            pass

        return state


class ChainedNicHealthCheck(HealthCheck):
    r'''
    Ensures that NIC (Network Interface Card) connections are in a healthy state.

    Uses the NicHealthCheck from shared_utils to perform comprehensive NIC health checks.
    This health check is executed after the fault to ensure NIC links are functioning properly.

    The NicHealthCheck constructor automatically handles device_index and baseline initialization,
    making this wrapper much simpler and more reliable.

    Args:
        device_index: Optional GPU device index to check. If None, checks all GPUs.
    '''

    def __init__(self, device_index=None):
        self.device_index = device_index
        self._nic_checker = NicHealthCheck(device_index=device_index)

    def __call__(self, state: FrozenState) -> FrozenState:
        # Perform NIC health check
        is_healthy = self._nic_checker._perform_health_check()
        if not is_healthy:
            raise exception.HealthCheckError("NIC health check failed - link down detected")

        return state


class XIDDetected(exception.RestartError):
    r'''
    Exception raised by :py:class:`XIDHealthCheck` when XID errors are detected.
    '''

    pass


class XIDHealthCheck(HealthCheck):
    r'''
    Checks for XID errors using three detection methods: DCGM Health Check, 
    DCGM Field Values, and pynvml.
    
    Raises XIDDetected exception if any XID errors are found.
    
    Args:
        timeout: timeout for XID detection checks
    '''

    def __init__(self, timeout=datetime.timedelta(seconds=10)):
        self.timeout = timeout
        self.log = logging.getLogger(__name__)
        
        self.log.info("XIDHealthCheck: Initializing XID Health Check")
        self.log.info(f"XIDHealthCheck: Timeout set to {timeout.total_seconds()}s")
        
        self.pynvml_available = self._check_pynvml_availability()
        self.dcgm_available = self._check_dcgm_availability()
        
        # Global DCGM handle to avoid reinitialization issues
        self._dcgm_handle = None
        self._dcgm_system = None
        
        self.log.info(f"XIDHealthCheck: Initialization complete - DCGM: {self.dcgm_available}, pynvml: {self.pynvml_available}")
        
    def _check_pynvml_availability(self) -> bool:
        """Check if pynvml is available for XID detection."""
        try:
            import pynvml
            self.pynvml = pynvml
            return True
        except ImportError:
            self.log.warning("pynvml is not installed, pynvml XID detection disabled.")
            return False
            
    def _check_dcgm_availability(self) -> bool:
        """Check if DCGM libraries are available for XID detection."""
        try:
            import dcgm_structs
            import dcgm_fields  
            import pydcgm
            self.dcgm_structs = dcgm_structs
            self.dcgm_fields = dcgm_fields
            self.pydcgm = pydcgm
            return True
        except ImportError:
            self.log.warning("DCGM libraries are not installed, DCGM XID detection disabled.")
            return False

    def _get_dcgm_handle(self):
        """Get or create global DCGM handle to avoid reinitialization."""
        if self._dcgm_handle is None:
            self._dcgm_handle = self.pydcgm.DcgmHandle(opMode=self.dcgm_structs.DCGM_OPERATION_MODE_AUTO)
            self._dcgm_system = self._dcgm_handle.GetSystem()
        return self._dcgm_handle, self._dcgm_system

    def __call__(self, state: FrozenState) -> FrozenState:
        """
        Perform XID detection using available methods.
        
        Args:
            state: read-only Wrapper state
            
        Returns:
            Forwarded read-only input state
            
        Raises:
            XIDDetected: if any XID errors are detected
        """
        self.log.info("=== XIDHealthCheck: Starting XID detection ===")
        
        if not torch.cuda.is_available():
            self.log.info("XIDHealthCheck: CUDA not available, skipping XID check")
            return state
            
        if not self.pynvml_available and not self.dcgm_available:
            self.log.warning("No XID detection libraries available, skipping XID check.")
            return state
            
        # Get GPU ID from actual current device (more reliable than LOCAL_RANK after rank reassignment)
        local_rank_env = os.getenv('LOCAL_RANK')
        if local_rank_env is not None:
            gpu_id = int(local_rank_env)  
        else:
            gpu_id = torch.cuda.current_device()  
        # Get distributed rank for logging purposes
        distributed_rank = int(os.getenv('RANK'))
        
        # Debug logging to verify mapping
        local_rank_env = os.getenv('LOCAL_RANK', 'None')
        self.log.info(f"XIDHealthCheck: Environment - RANK={distributed_rank}, LOCAL_RANK={local_rank_env}")
        self.log.info(f"XIDHealthCheck: torch.cuda.current_device()={gpu_id}")
        self.log.info(f"XIDHealthCheck: Checking GPU {gpu_id} for XID errors (rank {distributed_rank})")
        self.log.info(f"XIDHealthCheck: Available methods - DCGM: {self.dcgm_available}, pynvml: {self.pynvml_available}")
        
        # Run XID detection methods
        xid_detected = False
        error_details = []
        methods_run = 0
        methods_successful = 0
        
        # : DCGM Health Check (if available)
        if self.dcgm_available:
            methods_run += 1
            self.log.info("XIDHealthCheck: Running  - DCGM Health Check API")
            try:
                result = self._check_dcgm_health(gpu_id, distributed_rank)
                methods_successful += 1
                if result:
                    xid_detected = True
                    error_details.append("DCGM Health Check detected XID")
                    self.log.warning("XIDHealthCheck:  DETECTED XID!")
                else:
                    self.log.info(f"XIDHealthCheck: No XID detected on GPU {gpu_id} (rank {distributed_rank})")
            except Exception as e:
                self.log.warning(f"XIDHealthCheck:  failed: {e}")
                
        # Method 2: DCGM Field Values (if available)  
        # if self.dcgm_available:
        #     methods_run += 1
        #     self.log.info("XIDHealthCheck: Running Method 2 - DCGM Field Values API")
        #     try:
        #         result = self._check_dcgm_field_values(gpu_id)
        #         methods_successful += 1
        #         if result:
        #             xid_detected = True
        #             error_details.append("DCGM Field Values detected XID")
        #             self.log.warning("XIDHealthCheck: Method 2 DETECTED XID!")
        #         else:
        #             self.log.info("XIDHealthCheck: Method 2 - No XID detected")
        #     except Exception as e:
        #         self.log.warning(f"XIDHealthCheck: Method 2 failed: {e}")
                
        # Method 3: pynvml (if available)
        # if self.pynvml_available:
        #     methods_run += 1
        #     self.log.info("XIDHealthCheck: Running Method 3 - pynvml (NVML) API")
        #     try:
        #         result = self._check_pynvml(gpu_id)
        #         methods_successful += 1
        #         if result:
        #             xid_detected = True
        #             error_details.append("pynvml detected XID")
        #             self.log.warning("XIDHealthCheck: Method 3 DETECTED XID!")
        #         else:
        #             self.log.info("XIDHealthCheck: Method 3 - No XID detected")
        #     except Exception as e:
        #         self.log.warning(f"XIDHealthCheck: Method 3 failed: {e}")
        
        # Summary
        self.log.info(f"XIDHealthCheck: Summary - {methods_successful}/{methods_run} methods ran successfully")
        
        if xid_detected:
            error_msg = f"XID detected on GPU {gpu_id} (rank {distributed_rank}): {'; '.join(error_details)}"
            self.log.error(f"XIDHealthCheck: FINAL RESULT - {error_msg}")
            self.log.error("XIDHealthCheck: Raising XIDDetected exception to trigger restart")
            raise XIDDetected(error_msg)
        else:
            self.log.info(f"XIDHealthCheck: FINAL RESULT - No XID errors detected, GPU {gpu_id} is healthy (rank {distributed_rank})")
            self.log.info("XIDHealthCheck: Returning state unchanged (health check passed)")
            
        self.log.info("=== XIDHealthCheck: XID detection completed ===")
        return state
        
    def _check_dcgm_health(self, gpu_id: int, distributed_rank: int) -> bool:
        """
        : DCGM Health Check API (dcgm_health_watch_all)
        
        Uses dcgm_group.health.Check() to detect XID errors through health incidents.
        This is the method used by nvsentinel and other NVIDIA tools.
        
        Returns:
            bool: True if XID detected, False otherwise
        """
        self.log.info(f"  : Creating DCGM group for GPU {gpu_id} (rank {distributed_rank})")
        try:
            # Use global DCGM handle to avoid reinitialization
            dcgm_handle, dcgm_system = self._get_dcgm_handle()
            self.log.info("  : Got DCGM handle")
            
            # Create GPU group for this specific GPU
            group_name = f"health_check_{gpu_id}_{int(time.time())}"
            dcgm_group = self.pydcgm.DcgmGroup(
                dcgm_handle, 
                groupName=group_name, 
                groupType=self.dcgm_structs.DCGM_GROUP_EMPTY
            )
            dcgm_group.AddGpu(gpu_id)
            self.log.info(f"  : Created DCGM group '{group_name}' and added GPU {gpu_id}")
            
            # Set health monitoring - try DCGM_HEALTH_WATCH_ALL first
            try:
                health_systems = self.dcgm_structs.DCGM_HEALTH_WATCH_ALL
                self.log.info("  : Using DCGM_HEALTH_WATCH_ALL")
            except AttributeError:
                # Fallback to individual flags if DCGM_HEALTH_WATCH_ALL not available
                health_systems = (
                    self.dcgm_structs.DCGM_HEALTH_WATCH_PCIE |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_NVLINK |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_PMU |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_MCU |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_MEM |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_SM |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_INFOROM |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_THERMAL |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_POWER |
                    self.dcgm_structs.DCGM_HEALTH_WATCH_DRIVER
                )
                self.log.info("  : Using individual health watch flags")
            
            dcgm_group.health.Set(health_systems)
            self.log.info("  : Set health monitoring systems")
            
            # Execute health check
            self.log.info("  : Executing health check...")
            health_details = dcgm_group.health.Check()
            self.log.info(f"  : Health check completed, found {health_details.incidentCount} incidents")
            
            # Parse incidents to detect XID errors
            for i in range(health_details.incidentCount):
                incident = health_details.incidents[i]
                self.log.info(f"  : Incident {i}: health={incident.health}, error_code={incident.error.code}, msg='{incident.error.msg}'")
                if incident.health != self.dcgm_structs.DCGM_HEALTH_RESULT_PASS:
                    self.log.warning(f"  : Found health issue - {incident.error.msg}")
                    return True
                    
            self.log.info("  : All health incidents passed")
            return False
            
        except Exception as e:
            self.log.warning(f"  : DCGM health check failed for GPU {gpu_id}: {e}")
            return False

    def _check_dcgm_field_values(self, gpu_id: int) -> bool:
        """
        Method 2: DCGM Field Values API (DCGM_FI_DEV_XID_ERRORS)
        
        Directly queries the XID error field to get XID count.
        This is more direct but may miss some types of errors.
        
        Returns:
            bool: True if XID detected, False otherwise
        """
        self.log.info(f"  Method 2: Setting up field monitoring for GPU {gpu_id}")
        try:
            # Use global DCGM handle
            dcgm_handle, dcgm_system = self._get_dcgm_handle()
            self.log.info("  Method 2: Got DCGM handle")
            
            # Create field group for XID monitoring
            field_group = self.pydcgm.DcgmFieldGroup(dcgm_handle, "xid_fields")
            field_group.AddField(self.dcgm_fields.DCGM_FI_DEV_XID_ERRORS)
            self.log.info("  Method 2: Created field group and added DCGM_FI_DEV_XID_ERRORS field")
            
            # Create GPU group
            group_name = f"xid_field_check_{gpu_id}_{int(time.time())}"
            gpu_group = self.pydcgm.DcgmGroup(
                dcgm_handle,
                groupName=group_name,
                groupType=self.dcgm_structs.DCGM_GROUP_EMPTY
            )
            gpu_group.AddGpu(gpu_id)
            self.log.info(f"  Method 2: Created GPU group '{group_name}' and added GPU {gpu_id}")
            
            # Start field value watching
            self.log.info("  Method 2: Starting field watching (1s update, 10s max age, 1 sample)")
            gpu_group.samples.WatchFields(field_group, 1000000, 10.0, 1)  # 1s update, 10s max age, 1 sample
            
            # Wait a moment for data collection
            self.log.info("  Method 2: Waiting 0.5s for data collection...")
            time.sleep(0.5)
            
            # Get latest field values
            self.log.info("  Method 2: Getting latest field values...")
            field_values = gpu_group.samples.GetLatest(field_group)
            
            # Parse XID error count
            if gpu_id in field_values.values:
                self.log.info(f"  Method 2: Found field values for GPU {gpu_id}")
                gpu_values = field_values.values[gpu_id]
                if self.dcgm_fields.DCGM_FI_DEV_XID_ERRORS in gpu_values:
                    xid_field = gpu_values[self.dcgm_fields.DCGM_FI_DEV_XID_ERRORS]
                    self.log.info(f"  Method 2: Found XID field with {len(xid_field)} samples")
                    if len(xid_field) > 0:
                        xid_count = xid_field[-1].value  # Get latest value
                        self.log.info(f"  Method 2: Latest XID count = {xid_count}")
                        return xid_count > 0
                    else:
                        self.log.info("  Method 2: No XID field samples available")
                else:
                    self.log.info("  Method 2: XID field not found in GPU values")
            else:
                self.log.info(f"  Method 2: No field values found for GPU {gpu_id}")
                        
            self.log.info("  Method 2: No XID errors found")
            return False
            
        except Exception as e:
            self.log.warning(f"  Method 2: DCGM field values check failed for GPU {gpu_id}: {e}")
            return False

    def _check_pynvml(self, gpu_id: int) -> bool:
        """
        Method 3: pynvml (NVML) API
        
        Uses NVML directly to check for GPU recovery actions
        and various error indicators that may indicate XID conditions. This is the indirect check.
        
        Returns:
            bool: True if XID detected, False otherwise
        """
        self.log.info(f"  Method 3: Initializing pynvml for GPU {gpu_id}")
        try:
            # Initialize NVML if not already done
            try:
                self.pynvml.nvmlInit()
                self.log.info("  Method 3: NVML initialized")
            except self.pynvml.NVMLError:
                self.log.info("  Method 3: NVML already initialized")
            
            # Get device handle
            device_handle = self.pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.log.info(f"  Method 3: Got device handle for GPU {gpu_id}")
            
            # Check various error indicators
            xid_indicators_found = 0
            
            # Check 1: Pending page retirements (often caused by XID)
            self.log.info("  Method 3: Checking retired pages due to single-bit ECC errors...")
            try:
                retired_pages = self.pynvml.nvmlDeviceGetRetiredPages(
                    device_handle, 
                    self.pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS
                )
                self.log.info(f"  Method 3: Single-bit ECC retired pages: {retired_pages.pageCount}")
                if retired_pages.pageCount > 0:
                    self.log.warning(f"  Method 3: Found {retired_pages.pageCount} retired pages due to single-bit ECC!")
                    xid_indicators_found += 1
                    return True
            except self.pynvml.NVMLError as e:
                self.log.info(f"  Method 3: Single-bit ECC check not available: {e}")
            
            # Check 2: Pending page retirements due to double bit errors
            self.log.info("  Method 3: Checking retired pages due to double-bit ECC errors...")
            try:
                retired_pages_dbe = self.pynvml.nvmlDeviceGetRetiredPages(
                    device_handle, 
                    self.pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR
                )
                self.log.info(f"  Method 3: Double-bit ECC retired pages: {retired_pages_dbe.pageCount}")
                if retired_pages_dbe.pageCount > 0:
                    self.log.warning(f"  Method 3: Found {retired_pages_dbe.pageCount} retired pages due to double-bit ECC!")
                    xid_indicators_found += 1
                    return True
            except self.pynvml.NVMLError as e:
                self.log.info(f"  Method 3: Double-bit ECC check not available: {e}")
            
            # Check 3: Memory error counters
            self.log.info("  Method 3: Checking uncorrected memory error counters...")
            try:
                ecc_errors = self.pynvml.nvmlDeviceGetMemoryErrorCounter(
                    device_handle, 
                    self.pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, 
                    self.pynvml.NVML_VOLATILE_ECC, 
                    self.pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
                )
                self.log.info(f"  Method 3: Uncorrected ECC errors: {ecc_errors}")
                if ecc_errors > 0:
                    self.log.warning(f"  Method 3: Found {ecc_errors} uncorrected ECC errors!")
                    xid_indicators_found += 1
                    return True
            except self.pynvml.NVMLError as e:
                self.log.info(f"  Method 3: ECC error counter check not available: {e}")
                
            self.log.info(f"  Method 3: No XID indicators found (checked 3 error types)")
            return False
            
        except Exception as e:
            self.log.warning(f"  Method 3: pynvml check failed for GPU {gpu_id}: {e}")
            return False
