import json
import logging
import os
import pickle
from abc import ABC

import torch

from nvidia_resiliency_ext.attribution.utils import capture_logs
from nvidia_resiliency_ext.shared_utils.health_check import GPUHealthCheck, NicHealthCheck

logger = logging.getLogger(__name__)


class TraceCollector(ABC):
    """
    Base class for trace analyzers that process runtime-collected traces.

    This class provides a framework for collecting, dumping, and analyzing traces.
    It optionally manages an internal thread (`timeout_thread`) to periodically dump traces
    at configurable intervals.

    Derived classes must implement:
    - `dump()`: Writes collected traces to a file.
    - `collect()`: Gathers trace data at runtime.
    - `analyze()`: Processes and extracts insights from traces.

    The `update_timeout(timeout: int)` method allows adjusting the trace collection interval.
    Subclasses should define specific logic for trace handling.
    """

    def __init__(
        self,
        path: str,
    ):
        self.path = path

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
        self.trace = self.dump_fn(includeCollectives=True, onlyActive=True)
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
        health_check_results = {}

        with capture_logs() as stderr_gpu:
            gpu_health_check = GPUHealthCheck()
            gpu_health = gpu_health_check._perform_health_check_single_gpu(local_rank)
        with capture_logs() as stderr_nic:
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
