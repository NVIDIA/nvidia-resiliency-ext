import json
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
from abc import ABC, abstractmethod

import torch

from nvidia_resiliency_ext.attribution.utils import capture_logs
from nvidia_resiliency_ext.shared_utils.health_check import GPUHealthCheck, NicHealthCheck
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)


class TraceCollector(ABC):
    """
    Base class for trace analyzers that process runtime-collected traces.
    Derived classes must implement `collect()` to gather trace data at runtime.
    """

    def __init__(
        self,
        path: str,
    ):
        self.path = path

    @abstractmethod
    def collect(self):
        pass


class TorchFRTraceCollector(TraceCollector):
    """
    A utility class for dumping NCCL traces from PyTorch for collective analysis.

    Each rank writes its trace to a separate file at the specified path, allowing users to
    analyze collective communication patterns post-execution. The class manages an internal
    thread that periodically dumps traces after a specified timeout interval.

    ### Features:
    - Each rank writes its trace file independently.
    - Traces are stored at the user-specified path for collective analysis.

    ### Usage:
    1. Initialize the class with the desired output path.
    2. Call `collect()` to dump the trace data.

    This class is particularly useful for debugging and performance profiling of NCCL-based
    distributed training in PyTorch.
    """

    def __init__(
        self,
        path: str,
        json=True,
    ):
        super().__init__(path)
        self.rank = torch.distributed.get_rank()
        self.trace = None
        self.stack_trace = None
        self.dump_fn = torch._C._distributed_c10d._dump_nccl_trace
        self.json = json
        logger = logging.getLogger(LogConfig.name)
        logger.info(f"{self.rank} created TorchFRTraceCollector")

    def collect(self):
        """
        Dumps the collected trace data to a file.

        This method performs the following steps:
        - Creates a unique output path for the trace file
        - Opens the file in write mode
        - Writes the trace data to the file
        - Flushes the file to ensure all data is written
        """

        output_path = f"{self.path}/_dump_{self.rank}"
        self.trace = self.dump_fn(
            includeCollectives=True, includeStackTraces=False, onlyActive=True
        )
        mode = 'wb'
        if self.json:
            output_path = output_path + '.json'
            mode = 'w'
        with open(output_path, mode) as f:
            logger.info(f"{self.rank} is about to dump its trace to {output_path}")
            dumped_dict = pickle.loads(self.trace)
            local_rank = self.rank % torch.cuda.device_count()
            health_check_results = TorchFRTraceCollector.get_health_check_results(local_rank)
            dumped_dict['health_check_results'] = {
                'gpu': health_check_results['gpu_health_check'],
                'nic': health_check_results['nic_health_check'],
            }
            if self.json:
                json.dump(dumped_dict, f, indent=4)
            else:
                pickle.dump(dumped_dict, f)
            os.fsync(f.fileno())

    @staticmethod
    def get_health_check_results(local_rank: int):
        """
        Performs health checks on the GPU and NIC for a given local rank.

        This method performs the following steps:
        - Performs GPU health check
        - Performs NIC health check
        - Returns the bypassed output strings for GPU and NIC health checks
        """
        health_check_results = {}
        with capture_logs(LogConfig.name) as stderr_gpu:
            gpu_health_check = GPUHealthCheck(device_index=local_rank)
            gpu_health = gpu_health_check._perform_health_check()
        with capture_logs(LogConfig.name) as stderr_nic:
            nic_health_check = NicHealthCheck()
            nic_health_check.set_nic_device(local_rank)
            nic_health = nic_health_check._perform_health_check()

        logger.info(f"GPU Health Check: {stderr_gpu.getvalue()}")
        logger.info(f"NIC Health Check: {stderr_nic.getvalue()}")
        health_check_results['gpu_health_check'] = {
            'status': 'Healthy' if gpu_health else 'Failed',
            'output': stderr_gpu.getvalue(),
        }
        health_check_results['nic_health_check'] = {
            'status': 'Healthy' if nic_health else 'Failed',
            'output': stderr_nic.getvalue(),
        }
        return health_check_results
