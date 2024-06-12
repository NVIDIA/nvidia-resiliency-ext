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

import collections
import dataclasses
import functools
import inspect
import socket
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import torch

from .cupti import CuptiManager
from .interval_tracker import ReportIntervalTracker
from .reporting import ReportGenerator
from .statistics import Statistic


@dataclasses.dataclass(frozen=True)
class CallableId:
    """Represents a unique identifier for a callable object.

    Attributes:
        obj (object): The object that contains the callable
        name (str): The name of the callable

    Methods:
        __str__(): Returns a string representation of the CallableId,
            which includes the name of the object and the name of the callable
    """

    obj: object
    name: str
    arg_filter_fn: Optional[Callable[[inspect.BoundArguments], bool]] = None
    extra_args_fn: Optional[Callable[[inspect.BoundArguments], dict]] = None
    ignored_args: Optional[tuple[str, ...]] = None

    def __str__(self):
        if inspect.ismodule(self.obj):
            obj_name = self.obj.__name__
        elif inspect.isclass(self.obj):
            obj_name = f'{self.obj.__module__}.{self.obj.__name__}'
        elif hasattr(self.obj, '__class__'):
            obj_name = getattr(self.obj.__class__, '__name__', self.obj)
        else:
            obj_name = getattr(self.obj, '__name__', self.obj)
        return f'{obj_name}.{self.name}'


@dataclasses.dataclass
class CustomSection:
    """`CustomSection` represents user defined section of code
    (`Detector.detection_section`).

    Each section has CPU execution time computed.

    If CUDA profiling is enabled for the section, kernels launched in
    the section will be profiled with CUPTI. All kernel profiling
    results are collected by the CUPTI extension.
    """

    name: str
    location: str
    total_entry_cnt: int = 0
    max_elapseds_len: int = 8 * 1024
    cpu_elapsed_times: Deque[float] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=CustomSection.max_elapseds_len)
    )


class Detector:
    """Main class for straggler detection. The Detector class uses class
    methods and is not intended to be instantiated.

    Attributes:
        initialized (bool): Class-level attribute to track initialization.
        scores_to_compute (list): List of scores to compute, can include 'relative_perf_scores', 'individual_perf_scores'.
        gather_on_rank0 (bool): If True, when .generate_report is called report on rank 0 includes results for all ranks, reports on other ranks are empty
            If False, .generate_report is called report on any rank contains just the results for that particular rank
        profiling_interval (int, optional): Profile each `profiling_interval`-th section entry. Defaults to 1.
        report_time_interval(float, optional): Interval in seconds for generate_report_if_interval_elapsed. Defaults to 60.
        custom_sections (dict): Dict for recording CustomSection objects.
        cupti_manager (CuptiManager): CuptiManager is used for usage and managing of CUPTI methods, timing statistics calculation.
        reporter (ReportGenerator): ReportGenerator is used with result parsing and performance scoring algorithms.
        report_interval_tracker (ReportIntervalTracker): ReportIntervalTracker is used to synchronize report_time_interval between ranks.
        original_callables (dict, optional) Dict for restoring callable objects after wrapping them for profiling.
    """

    initialized: bool = False
    scores_to_compute: Sequence[str]
    gather_on_rank0: bool
    profiling_interval: int
    report_time_interval: float
    custom_sections: Dict[str, CustomSection]
    cupti_manager: CuptiManager
    reporter: ReportGenerator
    report_interval_tracker: ReportIntervalTracker
    original_callables: Optional[Dict[CallableId, Any]]

    def __new__(cls):
        msg = f"class {cls.__name__} should not be instantiated"
        raise RuntimeError(msg)

    @classmethod
    def initialize(
        cls,
        scores_to_compute: Union[Sequence[str], str] = "all",
        gather_on_rank0: bool = True,
        profiling_interval: int = 1,
        report_time_interval: float = 60,
        node_name: Optional[str] = None,
    ):
        """
        Args:
            scores_to_compute (list|str, optional): List of scores to compute, can include 'relative_perf_scores', 'individual_perf_scores'. or string "all" meaning: "all scores should be computed".
            gather_on_rank0 (bool, optional): If True, when .generate_report is called report on rank 0 includes results for all ranks, reports on other ranks are empty
                                    If False, .generate_report is called report on any rank contains just the results for that particular rank
            profiling_interval (int, optional): Profile each `profiling_interval`-th section entry. Defaults to 1.
            report_time_interval (float, optional): Interval in seconds for generate_report_if_interval_elapsed. Defaults to 60.
            node_name: (str, optional): User-friendly name of the current node to be used in reports. If `None` `socket.gethostname` will be used.
        """
        assert not cls.initialized

        cls.scores_to_compute = (
            ['relative_perf_scores', 'individual_perf_scores']
            if str(scores_to_compute) == "all"
            else scores_to_compute
        )
        cls.gather_on_rank0 = gather_on_rank0
        cls.profiling_interval = profiling_interval
        cls.custom_sections = {}
        cls.cupti_manager = CuptiManager(statsMaxLenPerKernel=8 * 1024)
        cls.cupti_manager.initialize()
        cls.reporter = ReportGenerator(
            scores_to_compute=cls.scores_to_compute,
            gather_on_rank0=gather_on_rank0,
            node_name=(node_name if node_name else socket.gethostname()),
        )
        cls.report_interval_tracker = ReportIntervalTracker(
            time_interval=report_time_interval,
            profiling_interval=profiling_interval,
        )

        cls.initialized = True

        cls.original_callables = {}

    @classmethod
    def shutdown(cls):
        """Shutdown Detector."""
        cls.cupti_manager.shutdown()
        cls.restore_original_callables()
        cls.cupti_manager = None
        cls.initialized = False

    @classmethod
    def _get_section_summaries(cls):
        """Compute statistics related to each user defined section.

        Currently, only CPU elapsed times are collected and summarized for the sections.

        Returns:
            Section name to statistics mapping
        """
        section_summaries = {}
        for key, section in cls.custom_sections.items():
            assert key == section.name
            if len(section.cpu_elapsed_times) == 0:
                continue  # TODO log a warning
            cpu_elapseds = torch.tensor(section.cpu_elapsed_times, dtype=torch.float64)
            cpu_stats = {
                Statistic.MIN: torch.min(cpu_elapseds).item(),
                Statistic.MAX: torch.max(cpu_elapseds).item(),
                Statistic.MED: torch.median(cpu_elapseds).item(),
                Statistic.AVG: torch.mean(cpu_elapseds).item(),
                Statistic.STD: (
                    torch.std(cpu_elapseds).item() if len(cpu_elapseds) > 1 else float("nan")
                ),
                Statistic.NUM: len(cpu_elapseds),
            }
            section_summaries[key] = cpu_stats
        return section_summaries

    @classmethod
    def _get_kernel_summaries(cls):
        """Get statistics related to captured CUDA kernels.

        These are actually computed by the CUPTI extension.

        Returns:
            Composite kernel name (with grid and block size included) to the statistics mapping
        """
        kernel_summaries = {}
        prof_results = cls.cupti_manager.get_results()
        for key, kernel_stats in prof_results.items():
            stats = {
                Statistic.MIN: kernel_stats.min,
                Statistic.MAX: kernel_stats.max,
                Statistic.MED: kernel_stats.median,
                Statistic.AVG: kernel_stats.avg,
                Statistic.STD: kernel_stats.stddev,
                Statistic.NUM: kernel_stats.num_calls,
            }
            kernel_summaries[key] = stats
        return kernel_summaries

    @classmethod
    def _reset_sections_elapseds(cls):
        for section in cls.custom_sections.values():
            section.cpu_elapsed_times.clear()

    @classmethod
    def generate_report(cls):
        """Calls ReportGenerator.generate_report method, resets recorded
        results."""
        assert cls.initialized

        # ensure all CUDA kernels are completed
        torch.cuda.synchronize()

        section_summaries = cls._get_section_summaries()
        kernel_summaries = cls._get_kernel_summaries()

        report = cls.reporter.generate_report(section_summaries, kernel_summaries)

        cls._reset_sections_elapseds()
        cls.cupti_manager.reset_results()

        return report

    @classmethod
    def generate_report_if_interval_elapsed(cls):
        """Calls ReportGenerator.generate_report method, if reporting interval elapsed.
        Supposed to be called during each training iteration on every rank.
        Reporting interval elapsed is synchronized beetween ranks through ReportIntervalTracker.
        Returns None if interval has not elapsed. Othewise `generate_report` return value is returned.
        """
        assert cls.initialized

        cls.report_interval_tracker.iter_increase()
        if cls.report_interval_tracker.is_interval_elapsed():
            report = cls.generate_report()
        else:
            report = None

        return report

    @classmethod
    def is_interval_elapsed(cls) -> bool:
        """Returns boolean flag that is True if interval elapsed during previous
        `generate_report_if_interval_elapsed` call. False otherwise.
        """
        return cls.report_interval_tracker.is_interval_elapsed()

    @staticmethod
    def _get_this_context_block_location() -> str:
        # back to "@contextmanager" decorated func, then back to the "__enter__", finally
        # we get the frame in the user code. TODO: verify this works with various Python versions
        caller_frame = inspect.currentframe().f_back.f_back.f_back  # type: ignore
        file_name = caller_frame.f_code.co_filename  # type: ignore
        line_number = caller_frame.f_lineno  # type: ignore
        return f"{file_name}:{line_number}"

    @classmethod
    def _ensure_section_name_is_valid(cls, name, location):
        # Enforce that section names are unique in the user code
        if name in cls.custom_sections:
            section = cls.custom_sections[name]
            if location != section.location:
                raise ValueError(f"Section name '{name}' is already used at: {section.location}")

    @classmethod
    @contextmanager
    def detection_section(
        cls,
        name: Optional[str] = None,
        profile_cuda: bool = True,
    ):
        """Context manager for monitoring user defined sections of code.

        NOTE: `profiling_interval` `Detector` constructor parameter determines how frequently
            sections are monitored. If can be > 1 to reduce the profiling overhead.

        Args:
            name (str, optional): Section name used for the reporting. Must be unique per user code.
                Defaults to None, in which case the `detection_section` entry location (`with ...`)
                (file path and line) is used as a section name.
            profile_cuda (bool, optional): If true, CUDA kernels launched under this section will
                be captured and used to compute rank "GPU performance score". Defaults to True.
        """

        if not cls.initialized:
            raise RuntimeError("Detector is not initialized.")

        section_location = Detector._get_this_context_block_location()
        if name is None:
            name = section_location

        # TODO: uncomment after Cython issue is resolved and
        # - test_unique_name_is_enforced,
        # - test_default_names_are_unique
        # test cases passes.
        # cls._ensure_section_name_is_valid(name, section_location)

        if name in cls.custom_sections:
            section = cls.custom_sections[name]
        else:
            section = CustomSection(name=name, location=section_location)
            cls.custom_sections[name] = section

        profile_this_entry = (section.total_entry_cnt % cls.profiling_interval) == 0
        section.total_entry_cnt += 1

        if profile_this_entry:

            if profile_cuda:
                cls.cupti_manager.start_profiling()
            cpu_time_start = time.perf_counter_ns()

            try:
                yield
            except:
                if profile_cuda:
                    cls.cupti_manager.stop_profiling()
                raise
            cpu_time_stop = time.perf_counter_ns()
            cpu_elapsed_time = (cpu_time_stop - cpu_time_start) * 1e-6
            section.cpu_elapsed_times.append(cpu_elapsed_time)
            if profile_cuda:
                cls.cupti_manager.stop_profiling()
                # CUPTI resuls are collected by the C++ extension
        else:
            yield  # if no need to profile this entry

    @classmethod
    def _build_wrapper(
        cls,
        fn,
        callable_id,
        profile_cuda: bool = True,
    ):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with cls.detection_section(
                name=str(callable_id),
                profile_cuda=profile_cuda,
            ):
                return fn(*args, **kwargs)

        return wrapper

    @classmethod
    def wrap_callables(
        cls,
        callable_ids: List[CallableId],
        profile_cuda: bool = True,
    ):
        """
        Each time `fn` (where `fn = getattr(callable_id.obj, callable_id.name)`) is called, the following will occur:

        .. code-block:: python

            with straggler.Detector.detection_section(str(callable_id)):
                fn(*args, **kwargs)

        Args:
            callable_ids (List[CallableId]):
                A list of callables to wrap with the detection context.

            profile_cuda (bool, optional):
                If `True`, CUDA kernels launched under this section will be captured
                and used to compute the rank's "GPU performance score". Defaults to `True`.
        """

        cls.original_callables = {}
        for callable_id in callable_ids:
            original = getattr(callable_id.obj, callable_id.name)
            cls.original_callables[callable_id] = original
            wrapped = cls._build_wrapper(
                original,
                callable_id,
                profile_cuda=profile_cuda,
            )
            setattr(callable_id.obj, callable_id.name, wrapped)

    @classmethod
    def restore_original_callables(cls):
        """Restore callable objects after cls._build_wrapper method was called
        for wrapping profiled callables."""
        if cls.original_callables:
            for callable_id, original in cls.original_callables.items():
                setattr(callable_id.obj, callable_id.name, original)
