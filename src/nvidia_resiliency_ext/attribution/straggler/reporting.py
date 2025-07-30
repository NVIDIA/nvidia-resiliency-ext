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
import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from . import dist_utils
from .name_mapper import NameMapper
from .statistics import Statistic

_SummaryType = Mapping[Statistic, float]  # type alias for kernel or section statistics


@dataclasses.dataclass(frozen=True)
class StragglerId:
    """
    Straggler identity
    """

    rank: int
    node: str


@dataclasses.dataclass(frozen=True)
class Report:
    """
    There are two types of performance scores provided in this report:
    - Relative performance scores (0...1) that compare the performance of the current rank to the best performing rank.
    - Individual performance scores (0...1) that compare the performance of the current rank to the rank's past performance.

    Relative scores need inter-rank synchronization to be computed, while individual scores can be computed on each rank separately.
    All performance scores can be interpreted as a ratio of: current performance/reference performance.

    For example:
    - If the relative performance score is 0.5, it means that the current rank is 2x slower than the best performing rank.
    - If the individual performance score is 0.5, it means that the current rank is 2x slower than its best performance.

    If `gather_on_rank0=True`: `*_perf_scores` fields contain results for all ranks (only on rank0; otherwise undefined).
    If `gather_on_rank0=False`: `*_perf_scores` fields contain only current rank results.

    Containers can be empty if there are no results.

    Attributes:
    - `gpu_relative_perf_scores`: Rank -> GPU relative performance score.
    - `section_relative_perf_scores`: Section name -> (Rank -> Section relative performance score).
    - `gpu_individual_perf_scores`: Rank -> GPU individual performance score.
    - `section_individual_perf_scores`: Section name -> (Rank -> Section individual performance score).
    - `rank_to_node`: Rank -> Node name, an auxiliary mapping useful for results reporting.
    - `local_section_summaries`: Local (e.g., this rank) timing stats for each user-defined section/context.
    - `local_kernel_summaries`: Local (e.g., this rank) timing stats for each captured CUDA kernel.
    - `generate_report_elapsed_time`: How long it took to generate this report.
    - `gather_on_rank0`: Mode for results gathering.
    - `rank`: Rank of the current report. `None` if not corresponding to any rank (i.e., `gather_on_rank0=True`).
    """

    gpu_relative_perf_scores: Mapping[int, float]
    section_relative_perf_scores: Mapping[str, Mapping[int, float]]
    gpu_individual_perf_scores: Mapping[int, float]
    section_individual_perf_scores: Mapping[str, Mapping[int, float]]
    rank_to_node: Mapping[int, str]
    local_section_summaries: Mapping[str, Any]
    local_kernel_summaries: Mapping[str, Any]
    generate_report_elapsed_time: float
    gather_on_rank0: bool
    rank: Optional[int]

    def identify_stragglers(
        self,
        gpu_rel_threshold: float = 0.75,
        section_rel_threshold: float = 0.75,
        gpu_indiv_threshold: float = 0.75,
        section_indiv_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """Identify the ranks with straggler GPUs based on performance
        thresholds.

        Args:
            gpu_rel_threshold (float): The threshold for relative GPU performance scores to identify stragglers. Default is 0.75
            section_rel_threshold (float): The threshold for relative sections performance scores. Default is 0.75
            gpu_indiv_threshold (float): The threshold for individual GPU performance scores. Default is 0.75
            section_indiv_threshold (float): The threshold for individual section performance scores. Default is 0.75

        Returns:
            Dict with string keys:
                - 'straggler_gpus_relative' (Set[StragglerId]): Stragglers with relative GPU performance scores below the threshold
                - 'straggler_gpus_individual' (Set[StragglerId]): Stragglers with individual GPU performance scores below the threshold
                - 'straggler_sections_relative' (Dict[str, Set[StragglerId]]): Sections with ranks having relative performance scores below the threshold
                - 'straggler_sections_individual' (Dict[str, Set[StragglerId]]): Sections with ranks having individual performance scores below the threshold
        """

        stragglers: Dict[str, Any] = {
            'straggler_gpus_relative': (
                {
                    StragglerId(rank=r, node=self.rank_to_node[r])
                    for r, d in self.gpu_relative_perf_scores.items()
                    if d < gpu_rel_threshold
                }
            ),
            'straggler_gpus_individual': (
                {
                    StragglerId(rank=r, node=self.rank_to_node[r])
                    for r, d in self.gpu_individual_perf_scores.items()
                    if d < gpu_indiv_threshold
                }
            ),
            'straggler_sections_relative': {},
            'straggler_sections_individual': {},
        }

        if self.section_relative_perf_scores:
            for (
                section,
                perf_scores,
            ) in self.section_relative_perf_scores.items():
                if straggler_ranks := {
                    StragglerId(rank=r, node=self.rank_to_node[r])
                    for r, d in perf_scores.items()
                    if d < section_rel_threshold
                }:
                    stragglers['straggler_sections_relative'][section] = straggler_ranks

        if self.section_individual_perf_scores:
            for (
                section,
                perf_scores,
            ) in self.section_individual_perf_scores.items():
                if straggler_ranks := {
                    StragglerId(rank=r, node=self.rank_to_node[r])
                    for r, d in perf_scores.items()
                    if d < section_indiv_threshold
                }:
                    stragglers['straggler_sections_individual'][section] = straggler_ranks

        return stragglers


class ReportGenerator:
    """This class is responsible for generating the performance report, based
    on the section and kernel summaries (section/kernel name -> stats).

    It is guaranteed that `.generate_report` is called on all ranks,
    so any required data synchronization can be done here.
    """

    def __init__(
        self,
        scores_to_compute,
        gather_on_rank0=True,
        pg=None,
        node_name='<notset>',
    ) -> None:
        """
        Args:
            scores_to_compute (list): List of scores to compute, e.g. ['relative_perf_scores', 'individual_perf_scores']
            gather_on_rank0 (bool): If True, report on rank 0 includes results for all ranks, reports on other ranks are empty
                                    If False, report on any rank contains just the results for that particular rank
            pg: Process group for communication
            node_name: User-friendly name of the current node, that will be used in reports
        """
        self.is_computing_rel_scores = 'relative_perf_scores' in scores_to_compute
        self.is_computing_indiv_scores = 'individual_perf_scores' in scores_to_compute
        self.gather_on_rank0 = gather_on_rank0
        self.group = pg
        self.world_size = dist_utils.get_world_size(self.group)
        self.rank = dist_utils.get_rank(self.group)
        self.node_name = node_name

        # dicts of best median execution times, used for individual scores normalization
        self.min_local_kernel_times: Dict[str, float] = collections.defaultdict(
            lambda: float('inf')
        )
        self.min_local_section_times: Dict[str, float] = collections.defaultdict(
            lambda: float('inf')
        )

        self.name_mapper = NameMapper(pg=pg)
        self.rank_to_node: Dict[int, str] = collections.defaultdict(lambda: '<unk>')

    def _compute_sections_perf_scores(
        self,
        sections_summaries: Mapping[str, _SummaryType],
        reference: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Compute local rank performance scores (0...1) for each custom section."""

        # Section scoring algorithm outline:
        # for each section calculate ratio: reference median time / median time on this rank
        # where "reference median time" can be:
        # - min median time across all ranks (aka "workload min")
        # - min median time observed on this rank (aka "rank historical min")
        # I.e. if a rank median time for a section is 2x higher than median on the fastest rank,
        # and the reference is "workload min", the rank will get a score 0.5
        # I.e. if a rank median time for a section is 2x higher that the minimum ever observed
        # on this rank, and the reference is "rank historical min", the rank will get a score 0.5

        section_to_score = {}
        for section, summary in sections_summaries.items():
            score = reference[section] / summary[Statistic.MED]
            section_to_score[section] = score
        return section_to_score

    def _compute_gpu_perf_score(
        self,
        kernels_summaries: Mapping[str, _SummaryType],
        reference: Mapping[str, float],
    ) -> float:
        """Compute local rank GPU performance score (0...1) based on the gathered
        kernel summaries."""

        # GPU scoring algorithm outline:
        # for each kernel calculate ratio: reference median / median time on this rank
        # reference median can be:
        # - min of median times across all ranks (aka "workload min")
        # - min median ever observed on this rank (aka "rank historical min")
        # compute final result, as a weighted arithmetic average of all kernel scores,
        # using total time spent in the kernel as a weight.
        # Kernels thet were not executed on all ranks are skipped.
        # (TODO can we allow for different ranks to capture different kernels?).

        rank_score = float('nan')
        if kernels_summaries:
            weighted_scores_sum = 0.0
            weights_sum = 0.0
            num_common_kernels = 0
            for kernel, summary in kernels_summaries.items():
                ref = reference[kernel]
                if math.isnan(ref):
                    continue  # no reference for this kernel, some ranks have not executed it
                num_common_kernels += 1
                score = ref / summary[Statistic.MED]
                weight = summary[Statistic.NUM] * summary[Statistic.AVG]
                weighted_scores_sum += score * weight
                weights_sum += weight
            if num_common_kernels > 0:
                rank_score = weighted_scores_sum / weights_sum
        return rank_score

    def _all_reduce_times(
        self,
        kernel_summaries: Mapping[str, _SummaryType],
        section_summaries: Mapping[str, _SummaryType],
    ) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        """Get minimum (across all ranks) median times for kernels and sections

        Args:
            kernel_summaries: local kernel summaries
            section_summaries: local section summaries

        Returns:
            Tuple of dicts: kernel name -> min median time, section name -> min median time
        """
        # Pack local median times into a tensor, then all-reduce it
        num_kernels = self.name_mapper.kernel_counter
        num_sections = self.name_mapper.section_counter
        total_len = num_kernels + num_sections
        times_tensor = torch.full((total_len,), -1.0, device='cpu')
        for k in kernel_summaries:
            idx = self.name_mapper.get_kernel_id(k)
            times_tensor[idx] = kernel_summaries[k][Statistic.MED]
        for s in section_summaries:
            idx = num_kernels + self.name_mapper.get_section_id(s)
            times_tensor[idx] = section_summaries[s][Statistic.MED]
        times_tensor = times_tensor.to(dist_utils.get_device_for_backend(self.group))
        dist_utils.all_reduce(times_tensor, op=torch.distributed.ReduceOp.MIN, group=self.group)
        # Unpack reduced tensor into dicts of name->score
        # NOTE: -1 will be obtained if some rank(s) did not have stats for given kernel/section
        # convert these to NaNs, to simplify further processing.
        times_tensor = times_tensor.cpu()
        min_kernel_times = {}
        for kernel_id in range(num_kernels):
            kernel_name = self.name_mapper.get_kernel_name(kernel_id)
            val = times_tensor[kernel_id].item()
            min_kernel_times[kernel_name] = val if val >= 0.0 else float('nan')
        min_section_times = {}
        for section_id in range(num_sections):
            section_name = self.name_mapper.get_section_name(section_id)
            val = times_tensor[num_kernels + section_id].item()
            min_section_times[section_name] = val if val >= 0.0 else float('nan')
        return min_kernel_times, min_section_times

    def _update_local_min_times(
        self,
        kernel_summaries: Mapping[str, _SummaryType],
        section_summaries: Mapping[str, _SummaryType],
    ) -> None:
        """Update minimum median times observed on this rank

        Args:
            kernel_summaries: local kernel summaries
            section_summaries: local section summaries
        """
        for kernel, summary in kernel_summaries.items():
            new_val = min(self.min_local_kernel_times[kernel], summary[Statistic.MED])
            self.min_local_kernel_times[kernel] = new_val
        for section, summary in section_summaries.items():
            new_val = min(self.min_local_section_times[section], summary[Statistic.MED])
            self.min_local_section_times[section] = new_val

    def _maybe_gather_rank_to_node(self) -> None:
        """Gather auxiliary rank to node mapping if not gathered yet"""
        if not self.rank_to_node:
            if self.gather_on_rank0:
                gathered_rank_node = dist_utils.all_gather_object(
                    (self.rank, self.node_name), self.group
                )
                self.rank_to_node = dict(gathered_rank_node)
            else:
                # if results are not gathered on rank0,
                # just put the current rank and node into the mapping,
                # in this case each rank deals only with its own results
                self.rank_to_node[self.rank] = self.node_name

    def _filter_out_nccl_kernels(self, kernel_summaries) -> Mapping[str, _SummaryType]:
        # TODO: NCCL kernels are skipped due to huge differences in execution time between ranks observerd:
        # E.g. `*ncclDevKernel_AllReduce_Sum*` MED=256us on a "fast" rank VS MED=3369us on a "slow" rank; need to:
        # - Explain that behaviour
        # - Decide on NCCL kernels: should we skip them all? or just a subset?
        # - Is the kernel name reliable way to identify NCCL kernels?
        return {k: v for k, v in kernel_summaries.items() if "ncclDev" not in k}

    def _get_tensor_from_scores(
        self,
        gpu_individual_score: float,
        sections_individual_scores: Mapping[str, float],
        gpu_relative_score: float,
        sections_relative_scores: Mapping[str, float],
    ):
        """Pack local GPU and section scores into a flat/1D CPU tensor"""
        sections_individual_scores = collections.defaultdict(
            lambda: float('nan'), sections_individual_scores
        )
        sections_relative_scores = collections.defaultdict(
            lambda: float('nan'), sections_relative_scores
        )
        num_sections = self.name_mapper.section_counter
        num_scores_per_rank = 2 + 2 * num_sections
        scores_tensor = torch.full((num_scores_per_rank,), float('nan'), device='cpu')
        scores_tensor[0] = gpu_individual_score
        scores_tensor[1] = gpu_relative_score
        for section_id in range(num_sections):
            section_name = self.name_mapper.get_section_name(section_id)
            scores_tensor[2 + section_id] = sections_individual_scores[section_name]
            scores_tensor[2 + num_sections + section_id] = sections_relative_scores[section_name]
        return scores_tensor

    def _get_scores_from_tensor(self, tensor) -> Tuple[float, Mapping, float, Mapping]:
        """Unpack flat/1D tensor obtained with `_get_tensor_from_scores`"""
        gpu_individual_score = tensor[0].item()
        gpu_relative_score = tensor[1].item()
        num_sections = (len(tensor) - 2) // 2
        assert num_sections == self.name_mapper.section_counter
        sections_individual_scores = {}
        sections_relative_scores = {}
        for section_id in range(num_sections):
            section_name = self.name_mapper.get_section_name(section_id)
            sections_individual_scores[section_name] = tensor[2 + section_id].item()
            sections_relative_scores[section_name] = tensor[2 + num_sections + section_id].item()
        return (
            gpu_individual_score,
            sections_individual_scores,
            gpu_relative_score,
            sections_relative_scores,
        )

    def _gather_results_on_rank0(
        self,
        gpu_individual_score: float,
        sections_individual_scores: Mapping[str, float],
        gpu_relative_score: float,
        sections_relative_scores: Mapping[str, float],
    ) -> Tuple[Mapping, Mapping, Mapping, Mapping]:
        """Collect all perf scores on rank0 and return them in the expected output format"""
        scores_tensor = self._get_tensor_from_scores(
            gpu_individual_score,
            sections_individual_scores,
            gpu_relative_score,
            sections_relative_scores,
        )

        gather_list = dist_utils.gather_on_rank0(scores_tensor, group=self.group)

        res_gi: Dict[int, float] = {}  # rank -> GPU individual score
        res_gr: Dict[int, float] = {}  # rank -> GPU relative score
        res_si: Dict[str, Dict[int, float]] = collections.defaultdict(
            dict
        )  # section -> (rank->section individual score)
        res_sr: Mapping[str, Dict[int, float]] = collections.defaultdict(
            dict
        )  # section -> (rank->section relative score)
        if self.rank == 0:
            # convert gathered per-rank scores into the final multi-rank dicts
            for r in range(self.world_size):
                gi, si, gr, sr = self._get_scores_from_tensor(gather_list[r])
                if self.is_computing_indiv_scores:
                    res_gi[r] = gi
                    for s in si:
                        res_si[s].update({r: si[s]})
                if self.is_computing_rel_scores:
                    res_gr[r] = gr
                    for s in sr:
                        res_sr[s].update({r: sr[s]})
        return res_gi, dict(res_si), res_gr, dict(res_sr)

    def generate_report(
        self,
        section_summaries: Mapping[str, _SummaryType],
        kernel_summaries: Mapping[str, _SummaryType],
    ):
        """Generate a performance report based on the given summaries.

        All ranks need to call .generate_report, as there can be synchronization between ranks.

        If `gather_on_rank0` is True:
            - The report on rank 0 includes results for all ranks.
            - On all other ranks the return value is None.

        If `gather_on_rank0` is False:
            - The report on any rank contains just the results for that particular rank.

        Args:
            section_summaries (dict): Timing stats collected for user-defined sections on the current rank.
            kernel_summaries (dict): Timing stats collected for captured CUDA kernels on the current rank.

        Returns:
            A `Report` object containing the performance scores and summaries,
            or None if gather_on_rank0 is True and the current rank is not 0
        """

        report_start = time.perf_counter_ns()

        # update rank and world size in case they are not set
        self.world_size = dist_utils.get_world_size(self.group)
        self.rank = dist_utils.get_rank(self.group)

        kernel_summaries = self._filter_out_nccl_kernels(kernel_summaries)

        self._maybe_gather_rank_to_node()  # one time rank->node mapping synchronization

        # prepare name mapper, so we can map kernel/section names to integer IDs if needed.
        # name mapper is not used if there is no inter-rank synchronization e.g.
        # with individual scores only + no gather_on_rank0. NOTE: inter-rank names synchronization
        # is expected to happen just at the beginning, when there are new kernels/sections captured.
        need_name_mapper = self.is_computing_rel_scores or self.gather_on_rank0
        if need_name_mapper:
            self.name_mapper.gather_and_assign_ids(
                kernel_names=list(kernel_summaries.keys()),
                section_names=list(section_summaries.keys()),
            )

        gpu_individual_score: float = float('nan')
        sections_individual_scores: Mapping[str, float] = {}
        if self.is_computing_indiv_scores:
            self._update_local_min_times(kernel_summaries, section_summaries)
            gpu_individual_score = self._compute_gpu_perf_score(
                kernel_summaries,
                reference=self.min_local_kernel_times,
            )
            sections_individual_scores = self._compute_sections_perf_scores(
                section_summaries,
                reference=self.min_local_section_times,
            )

        gpu_relative_score: float = float('nan')
        sections_relative_scores: Mapping[str, float] = {}
        if self.is_computing_rel_scores:
            min_workload_kernel_times, min_workload_section_times = self._all_reduce_times(
                kernel_summaries, section_summaries
            )
            gpu_relative_score = self._compute_gpu_perf_score(
                kernel_summaries,
                reference=min_workload_kernel_times,
            )
            sections_relative_scores = self._compute_sections_perf_scores(
                section_summaries,
                reference=min_workload_section_times,
            )

        # now we have scores for the local rank, need to format them for the report
        res_gpu_indiv: Mapping[int, float] = {}  # rank->GPU perf score
        res_section_idiv: Mapping[str, Mapping[int, float]] = (
            {}
        )  # section->(rank->section perf score)
        res_gpu_rel: Mapping[int, float] = {}  # rank->GPU perf score
        res_section_rel: Mapping[str, Mapping[int, float]] = (
            {}
        )  # section->(rank->section perf score)

        if self.gather_on_rank0:
            gathered_scores = self._gather_results_on_rank0(
                gpu_individual_score,
                sections_individual_scores,
                gpu_relative_score,
                sections_relative_scores,
            )
            if self.rank == 0:
                (
                    res_gpu_indiv,
                    res_section_idiv,
                    res_gpu_rel,
                    res_section_rel,
                ) = gathered_scores
        else:
            # convert local rank scores to rank->score format
            # and local section scores from name->score to name->rank->score
            # for consitency with the gather_on_rank0=True case.
            if self.is_computing_indiv_scores:
                res_gpu_indiv = {self.rank: gpu_individual_score}
                res_section_idiv = {
                    k: {self.rank: v} for k, v in sections_individual_scores.items()
                }
            if self.is_computing_rel_scores:
                res_gpu_rel = {self.rank: gpu_relative_score}
                res_section_rel = {k: {self.rank: v} for k, v in sections_relative_scores.items()}

        report_stop = time.perf_counter_ns()
        report_elapsed = (report_stop - report_start) * 1e-6

        report_params = {
            'gpu_relative_perf_scores': res_gpu_rel,
            'section_relative_perf_scores': res_section_rel,
            'gpu_individual_perf_scores': res_gpu_indiv,
            'section_individual_perf_scores': res_section_idiv,
            'rank_to_node': dict(self.rank_to_node),
            'local_section_summaries': section_summaries,
            'local_kernel_summaries': kernel_summaries,
            'generate_report_elapsed_time': report_elapsed,
            'gather_on_rank0': self.gather_on_rank0,
            'rank': self.rank,
        }

        report = Report(**report_params)

        if self.gather_on_rank0:
            return report if self.rank == 0 else None

        # otherwise, report scores for each rank separately
        return report
