# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extended GPU Health Check with XID Equivalent Detection

This module extends the standard nvidia-resiliency-ext GPUHealthCheck with
proactive XID equivalent detection using NVML APIs. Instead of waiting for
XIDs to appear in logs, it directly queries hardware status for equivalent
conditions.

Key Features:
- Detects XID equivalent conditions before they appear in logs
- Maps specific XID types to corresponding NVML function calls
- Maintains same interface as original GPUHealthCheck
"""

import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Import the base GPUHealthCheck from nvidia-resiliency-ext
import sys
import os
sys.path.append('/lustre/fs1/portfolios/coreai/users/weshen/nvidia-resiliency-ext/src')

from nvidia_resiliency_ext.shared_utils.health_check import GPUHealthCheck, with_pynvml_lock


class XIDType(Enum):
    """XID type enumeration for mapping to NVML checks"""
    GPU_OFF_BUS = 43           # GPU detached from bus
    ECC_DBE_ERROR = 48         # Double-bit ECC error
    THERMAL_VIOLATION = 63     # Thermal protection triggered
    GPU_DETACH = 79            # GPU detached
    SRAM_PARITY_ERROR = 94     # SRAM parity error
    ECC_SBE_ERROR = 95         # Single-bit ECC error
    GSP_TIMEOUT = 119          # GSP firmware timeout
    GSP_CRASH = 122            # GSP firmware crash
    MMU_FAULT = 31             # Memory management unit fault
    PAGE_FAULT = 13            # GPU page fault
    L2_CACHE_ERROR = 24        # L2 cache error


class XIDEquivalentCheck:
    """
    XID equivalent detection using NVML APIs
    Maps XID types to corresponding NVML function calls for proactive detection
    """
    
    def __init__(self, pynvml_module):
        self.pynvml = pynvml_module
        self.log = logging.getLogger(__name__)
        
        # Store previous counter values for trend detection
        self._previous_counters = {}
        
    def check_ecc_errors(self, handle: Any, gpu_id: int) -> List[Tuple[XIDType, str]]:
        """
        Check for ECC errors equivalent to XID 48 (DBE) and XID 95 (SBE)
        
        Args:
            handle: NVML device handle
            gpu_id: GPU index
            
        Returns:
            List of detected issues as (XIDType, description) tuples
        """
        issues = []
        
        try:
            # Check for uncorrected (double-bit) ECC errors - equivalent to XID 48
            dbe_count = self.pynvml.nvmlDeviceGetMemoryErrorCounter(
                handle,
                self.pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                self.pynvml.NVML_ECC_COUNTER_TYPE_AGGREGATE
            )
            
            # Check for corrected (single-bit) ECC errors - equivalent to XID 95
            sbe_count = self.pynvml.nvmlDeviceGetMemoryErrorCounter(
                handle,
                self.pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                self.pynvml.NVML_ECC_COUNTER_TYPE_AGGREGATE
            )
            
            # Track counter changes
            prev_dbe = self._previous_counters.get(f"gpu_{gpu_id}_dbe", 0)
            prev_sbe = self._previous_counters.get(f"gpu_{gpu_id}_sbe", 0)
            
            if dbe_count > prev_dbe:
                issues.append((
                    XIDType.ECC_DBE_ERROR,
                    f"GPU {gpu_id}: Double-bit ECC errors detected ({dbe_count - prev_dbe} new errors)"
                ))
                
            if sbe_count > prev_sbe:
                # Only report SBE if rate is concerning (>10 new errors)
                new_sbe_count = sbe_count - prev_sbe
                if new_sbe_count > 10:
                    issues.append((
                        XIDType.ECC_SBE_ERROR,
                        f"GPU {gpu_id}: High rate of single-bit ECC errors ({new_sbe_count} new errors)"
                    ))
            
            # Update counters
            self._previous_counters[f"gpu_{gpu_id}_dbe"] = dbe_count
            self._previous_counters[f"gpu_{gpu_id}_sbe"] = sbe_count
            
        except self.pynvml.NVMLError as e:
            self.log.debug(f"GPU {gpu_id}: ECC error check failed: {e}")
            
        return issues
    
    def check_gpu_presence(self, handle: Any, gpu_id: int) -> List[Tuple[XIDType, str]]:
        """
        Check for GPU presence issues equivalent to XID 43 (off-bus) and XID 79 (detach)
        
        Args:
            handle: NVML device handle
            gpu_id: GPU index
            
        Returns:
            List of detected issues as (XIDType, description) tuples
        """
        issues = []
        
        try:
            # Test basic device operations to detect off-bus condition
            pci_info = self.pynvml.nvmlDeviceGetPciInfo(handle)
            
            # Check PCIe link status
            try:
                link_gen = self.pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                link_width = self.pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
                
                # If link generation or width is 0, GPU may be detaching
                if link_gen == 0 or link_width == 0:
                    issues.append((
                        XIDType.GPU_DETACH,
                        f"GPU {gpu_id}: PCIe link degraded (Gen: {link_gen}, Width: {link_width})"
                    ))
                    
            except self.pynvml.NVMLError as e:
                # PCIe link query failure may indicate off-bus condition
                issues.append((
                    XIDType.GPU_OFF_BUS,
                    f"GPU {gpu_id}: PCIe link query failed, possible off-bus condition: {e}"
                ))
                
        except self.pynvml.NVMLError as e:
            # PCI info query failure indicates severe connectivity issue
            issues.append((
                XIDType.GPU_OFF_BUS,
                f"GPU {gpu_id}: Device off-bus or disconnected: {e}"
            ))
            
        return issues
    
    def check_gsp_health(self, handle: Any, gpu_id: int) -> List[Tuple[XIDType, str]]:
        """
        Check for GSP firmware issues equivalent to XID 119 (timeout) and XID 122 (crash)
        
        Args:
            handle: NVML device handle
            gpu_id: GPU index
            
        Returns:
            List of detected issues as (XIDType, description) tuples
        """
        issues = []
        
        try:
            # Test GPU responsiveness with basic queries
            # Timeout or failure may indicate GSP issues
            
            # Test 1: Performance state query
            try:
                perf_state = self.pynvml.nvmlDeviceGetPerformanceState(handle)
            except self.pynvml.NVMLError_Timeout:
                issues.append((
                    XIDType.GSP_TIMEOUT,
                    f"GPU {gpu_id}: GSP timeout detected during performance state query"
                ))
                return issues
            except self.pynvml.NVMLError as e:
                if "timeout" in str(e).lower():
                    issues.append((
                        XIDType.GSP_TIMEOUT,
                        f"GPU {gpu_id}: GSP timeout detected: {e}"
                    ))
                    return issues
                    
            # Test 2: Power state query
            try:
                power_state = self.pynvml.nvmlDeviceGetPowerState(handle)
            except self.pynvml.NVMLError_Timeout:
                issues.append((
                    XIDType.GSP_TIMEOUT,
                    f"GPU {gpu_id}: GSP timeout detected during power state query"
                ))
                return issues
                
            # Test 3: Basic device name query (lightweight operation)
            try:
                device_name = self.pynvml.nvmlDeviceGetName(handle)
            except self.pynvml.NVMLError_Timeout:
                issues.append((
                    XIDType.GSP_TIMEOUT,
                    f"GPU {gpu_id}: GSP timeout detected during device name query"
                ))
            except self.pynvml.NVMLError as e:
                if "uninitialized" in str(e).lower() or "unknown" in str(e).lower():
                    issues.append((
                        XIDType.GSP_CRASH,
                        f"GPU {gpu_id}: Possible GSP crash detected: {e}"
                    ))
                    
        except Exception as e:
            self.log.debug(f"GPU {gpu_id}: GSP health check failed: {e}")
            
        return issues
    
    def check_thermal_status(self, handle: Any, gpu_id: int) -> List[Tuple[XIDType, str]]:
        """
        Check for thermal issues equivalent to XID 63
        
        Args:
            handle: NVML device handle
            gpu_id: GPU index
            
        Returns:
            List of detected issues as (XIDType, description) tuples
        """
        issues = []
        
        try:
            # Get current GPU temperature
            temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            
            # Get thermal thresholds
            try:
                shutdown_threshold = self.pynvml.nvmlDeviceGetTemperatureThreshold(
                    handle, self.pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
                )
                
                # Check if temperature is approaching shutdown threshold
                if temp >= shutdown_threshold - 5:  # 5°C margin
                    issues.append((
                        XIDType.THERMAL_VIOLATION,
                        f"GPU {gpu_id}: Critical temperature {temp}°C (shutdown at {shutdown_threshold}°C)"
                    ))
                elif temp >= shutdown_threshold - 15:  # 15°C warning margin
                    issues.append((
                        XIDType.THERMAL_VIOLATION,
                        f"GPU {gpu_id}: High temperature {temp}°C (warning, shutdown at {shutdown_threshold}°C)"
                    ))
                    
            except self.pynvml.NVMLError:
                # Fallback to general high temperature check
                if temp > 85:  # General high temperature threshold
                    issues.append((
                        XIDType.THERMAL_VIOLATION,
                        f"GPU {gpu_id}: High temperature {temp}°C detected"
                    ))
                    
        except self.pynvml.NVMLError as e:
            self.log.debug(f"GPU {gpu_id}: Thermal check failed: {e}")
            
        return issues
    
    def check_memory_faults(self, handle: Any, gpu_id: int) -> List[Tuple[XIDType, str]]:
        """
        Check for memory faults equivalent to XID 31 (MMU fault) and XID 13 (page fault)
        
        Args:
            handle: NVML device handle
            gpu_id: GPU index
            
        Returns:
            List of detected issues as (XIDType, description) tuples
        """
        issues = []
        
        try:
            # Check memory usage patterns that might indicate faults
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Check BAR1 memory status
            try:
                bar1_info = self.pynvml.nvmlDeviceGetBAR1MemoryInfo(handle)
                
                # High BAR1 usage might indicate memory mapping issues
                if bar1_info.used / bar1_info.total > 0.95:
                    issues.append((
                        XIDType.MMU_FAULT,
                        f"GPU {gpu_id}: High BAR1 memory usage ({bar1_info.used}/{bar1_info.total} bytes)"
                    ))
                    
            except self.pynvml.NVMLError:
                pass  # BAR1 info not available on all GPUs
                
            # Check for running processes to detect potential page fault conditions
            try:
                compute_procs = self.pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                graphics_procs = self.pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                
                # If memory usage is very high but few processes, might indicate fault condition
                total_procs = len(compute_procs) + len(graphics_procs)
                if mem_info.used / mem_info.total > 0.98 and total_procs == 0:
                    issues.append((
                        XIDType.PAGE_FAULT,
                        f"GPU {gpu_id}: High memory usage without active processes, possible page fault"
                    ))
                    
            except self.pynvml.NVMLError:
                pass  # Process enumeration might fail in some cases
                
        except self.pynvml.NVMLError as e:
            self.log.debug(f"GPU {gpu_id}: Memory fault check failed: {e}")
            
        return issues
    
    def check_sram_errors(self, handle: Any, gpu_id: int) -> List[Tuple[XIDType, str]]:
        """
        Check for SRAM parity errors equivalent to XID 94 and XID 24
        
        Args:
            handle: NVML device handle
            gpu_id: GPU index
            
        Returns:
            List of detected issues as (XIDType, description) tuples
        """
        issues = []
        
        try:
            # Check SRAM-specific ECC counters if available
            try:
                sram_errors = self.pynvml.nvmlDeviceGetMemoryErrorCounter(
                    handle,
                    self.pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    self.pynvml.NVML_ECC_COUNTER_TYPE_VOLATILE_SRAM
                )
                
                prev_sram = self._previous_counters.get(f"gpu_{gpu_id}_sram", 0)
                if sram_errors > prev_sram:
                    issues.append((
                        XIDType.SRAM_PARITY_ERROR,
                        f"GPU {gpu_id}: SRAM parity errors detected ({sram_errors - prev_sram} new errors)"
                    ))
                    
                self._previous_counters[f"gpu_{gpu_id}_sram"] = sram_errors
                
            except self.pynvml.NVMLError:
                pass  # SRAM counters not available on all GPUs
                
            # Check retired pages information
            try:
                retired_pages = self.pynvml.nvmlDeviceGetRetiredPages(
                    handle, self.pynvml.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS
                )
                
                prev_retired = self._previous_counters.get(f"gpu_{gpu_id}_retired", 0)
                current_retired = len(retired_pages) if retired_pages else 0
                
                if current_retired > prev_retired:
                    issues.append((
                        XIDType.L2_CACHE_ERROR,
                        f"GPU {gpu_id}: New retired pages detected ({current_retired - prev_retired} pages)"
                    ))
                    
                self._previous_counters[f"gpu_{gpu_id}_retired"] = current_retired
                
            except self.pynvml.NVMLError:
                pass  # Retired pages info not available on all GPUs
                
        except Exception as e:
            self.log.debug(f"GPU {gpu_id}: SRAM error check failed: {e}")
            
        return issues


class ExtendedGPUHealthCheck(GPUHealthCheck):
    """
    Extended GPU Health Check with XID Equivalent Detection
    
    This class extends the standard GPUHealthCheck to include proactive
    XID equivalent detection using NVML APIs. It maintains the same interface
    while adding comprehensive hardware status monitoring.
    """
    
    def __init__(self, interval: int = 60, on_failure=None, enable_xid_checks: bool = True):
        """
        Initialize Extended GPU Health Check
        
        Args:
            interval: Health check interval in seconds
            on_failure: Callback function for failure handling
            enable_xid_checks: Whether to enable XID equivalent checks
        """
        super().__init__(interval=interval, on_failure=on_failure)
        
        self.enable_xid_checks = enable_xid_checks
        self.xid_checker = None
        
        if self.enable_xid_checks and self.pynvml_available:
            self.xid_checker = XIDEquivalentCheck(self.pynvml)
            self.log.info("XID equivalent detection enabled")
        else:
            self.log.info("XID equivalent detection disabled")
    
    @with_pynvml_lock
    def _perform_health_check(self) -> bool:
        """
        Extended health check that includes both standard recovery action check
        and XID equivalent detection
        
        Returns:
            bool: True if all checks pass, False if any issue detected
        """
        # First run the standard health check
        standard_result = super()._perform_health_check()
        
        # If standard check failed, no need to do additional checks
        if not standard_result:
            return False
            
        # If XID checks are disabled, return standard result
        if not self.enable_xid_checks or not self.xid_checker:
            return standard_result
            
        # Perform XID equivalent checks
        try:
            self.pynvml.nvmlInit()
            device_count = self.pynvml.nvmlDeviceGetCount()
            
            all_xid_checks_passed = True
            
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Run all XID equivalent checks
                all_issues = []
                all_issues.extend(self.xid_checker.check_ecc_errors(handle, i))
                all_issues.extend(self.xid_checker.check_gpu_presence(handle, i))
                all_issues.extend(self.xid_checker.check_gsp_health(handle, i))
                all_issues.extend(self.xid_checker.check_thermal_status(handle, i))
                all_issues.extend(self.xid_checker.check_memory_faults(handle, i))
                all_issues.extend(self.xid_checker.check_sram_errors(handle, i))
                
                # Log any detected issues
                for xid_type, description in all_issues:
                    self.log.warning(f"XID equivalent detected - {xid_type.name} (XID {xid_type.value}): {description}")
                    all_xid_checks_passed = False
                    
            return all_xid_checks_passed
            
        except self.pynvml.NVMLError as e:
            self.log.warning(f"XID equivalent check failed: {str(e)}")
            return True  # Don't fail on XID check errors, fall back to standard check
        except Exception as e:
            self.log.warning(f"Unexpected error during XID equivalent check: {str(e)}")
            return True  # Don't fail on unexpected errors
        finally:
            try:
                self.pynvml.nvmlShutdown()
            except Exception as e:
                self.log.warning(f"Error during NVML shutdown: {str(e)}")
    
    def get_xid_check_status(self) -> Dict[str, Any]:
        """
        Get status information about XID equivalent checks
        
        Returns:
            Dict containing XID check configuration and status
        """
        return {
            "xid_checks_enabled": self.enable_xid_checks,
            "xid_checker_available": self.xid_checker is not None,
            "pynvml_available": self.pynvml_available,
            "supported_xid_types": [xid.name for xid in XIDType] if self.xid_checker else []
        }


# Usage example and integration helper
def create_extended_health_check(interval: int = 60, enable_xid_checks: bool = True):
    """
    Factory function to create ExtendedGPUHealthCheck instance
    
    Args:
        interval: Health check interval in seconds
        enable_xid_checks: Whether to enable XID equivalent detection
        
    Returns:
        ExtendedGPUHealthCheck instance
    """
    return ExtendedGPUHealthCheck(interval=interval, enable_xid_checks=enable_xid_checks)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create extended health check instance
    health_check = create_extended_health_check(interval=30, enable_xid_checks=True)
    
    # Print status
    status = health_check.get_xid_check_status()
    print("XID Check Status:", status)
    
    # Perform a single health check
    result = health_check()
    print(f"Health check result: {result}")
    
    # For async usage:
    # import asyncio
    # asyncio.run(health_check.async_check())
