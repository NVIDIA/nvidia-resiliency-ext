import json
import logging
import os
import pickle
from abc import ABC
import glob
import argparse
import torch
import re
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer
from nvidia_resiliency_ext.shared_utils.health_check import GPUHealthCheck, NicHealthCheck
from nvidia_resiliency_ext.attribution.utils import capture_logs
logger = logging.getLogger(__name__)


_trace_analyzer = None

def init_trace_analyzer(
    path, json=True, 
):
    """
    Initialize the trace analyzer with the specified configuration.
    
    Args:
        path: Path to store trace files.
        json: Whether to save traces in JSON format.

    Returns:
        The initialized trace analyzer instance.
    """
    global _trace_analyzer
    if _trace_analyzer is None:
        _trace_analyzer = TorchTraceAnalyzer(path, json)
    return _trace_analyzer


def get_trace_analyzer():
    global _trace_analyzer
    return _trace_analyzer


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

        output_path = f"{path}/_dump_{self.rank}"
        self.trace = self.dump_fn(includeCollectives=True, onlyActive=True)
        mode = 'wb'
        if self.json:
            output_path = output_path + '.json'
            mode = 'w'
        with open(output_path, mode) as f:
            logger.info(
                f"{self.rank} is about to dump its trace to {output_path}"
            )
            dumped_dict = pickle.loads(self.trace)
            local_rank = self.rank % torch.cuda.device_count()
            if health_check:
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

    def analyze(self, schedule_order: str = 'TP->PP->DP'):
        """
        Analyzes the collected trace data using the CollectiveAnalyzer.
        """

        args = {
            'paths': [self.path],
            'verbose': False,
            'time_spread': False,
            'health_check': True,
            'llm_analyze': self.llm_analyze,
            'model': 'nvdev/nvidia/llama-3.3-nemotron-super-49b-v1', #nvidia/llama-3.1-nemotron-51b-instruct',
            'scheduling_order': schedule_order,
        }
        dump_files = glob.glob(os.path.join(self.path, '_dump_*'))
        if not dump_files:
            logger.warning(f"No dump files found in {self.path}")
            return
        logger.info(f"Found {len(dump_files)} dump files in {self.path}")
        args['pattern'] = '*.json' if self.json else '*'
        logger.info(f"Analyzing traces with options as {args}")
        analyzer = CollectiveAnalyzer(argparse.Namespace(**args))
        logger.info(f"CollectiveAnalyzer created")
        with capture_logs() as stdout:
            analyzer.run_sync(args['paths'])
        analysis_output = stdout.getvalue()
        logger.info(f"Analysis output: {analysis_output}")
        hanging_ranks = re.search(r'.*hanging ranks: (.*)', analysis_output)
        if hanging_ranks:
            # Parse the hanging ranks from the analysis output
            hanging_ranks_list = list(map(int, hanging_ranks.group(1).split(',')))
            if self.failure_config:
            # Check if failure_ranks is a list or a single value
                failure_ranks_list = self.failure_ranks if isinstance(self.failure_ranks, list) else [self.failure_ranks]
                # Find matched and unmatched elements
                matched_ranks = [rank for rank in failure_ranks_list if rank in hanging_ranks_list]
                unmatched_ranks = [rank for rank in failure_ranks_list if rank not in hanging_ranks_list]
                logger.info(f"Matched ranks: {matched_ranks}, count: {len(matched_ranks)}")
                logger.info(f"Unmatched ranks: {unmatched_ranks}, count: {len(unmatched_ranks)}")
            else:
                logger.info(f"hanging_ranks: {hanging_ranks_list}")
        logger.info(f"Completed Analyzing traces in {self.path}")
