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

import logging
import sys
import time
from typing import Optional

import torch

from ._utils import is_module_available

if is_module_available("lightning"):
    from lightning.pytorch.callbacks import Callback
elif is_module_available("pytorch_lightning"):
    from pytorch_lightning.callbacks import Callback
else:
    raise ImportError("Could not find 'lightning' or 'pytorch_lightning' module")

import nvidia_resiliency_ext.straggler as straggler


class StragglerDetectionCallback(Callback):

    def __init__(
        self,
        report_time_interval: float,
        calc_relative_gpu_perf: bool,
        calc_individual_gpu_perf: bool,
        num_gpu_perf_scores_to_print: int,
        gpu_relative_perf_threshold: float,
        gpu_individual_perf_threshold: float,
        stop_if_detected: bool,
        enable_ptl_logging: bool,
        profiling_interval: int = 1,
        logger_name: Optional[str] = "nemo_logger.StragglerDetectionCallback",
    ):
        """
        Initialize straggler detection callback instance.

        Args:
            report_time_interval (float): Interval [seconds] of the straggler check
            calc_relative_gpu_perf (bool): Calculate relative GPU performance
            calc_individual_gpu_perf (bool): Calculate individual GPU performance
            num_gpu_perf_scores_to_print (int): How many best and worst perf scores to print (0 - does not print periodically, but only if stragglers are detected)
            gpu_relative_perf_threshold (float): Threshold for relative GPU performance scores
            gpu_individual_perf_threshold (float): Threshold for individual GPU performance scores
            stop_if_detected (bool): Set to True, to terminate the workload if stragglers are detected
            enable_ptl_logging (bool): Set to True, to log GPU performance scores to all PTL loggers enabled through trainer
            profiling_interval (int): `profiling_interval` passed to `straggler.Detector.initialize`. Defaults to 1.
            logger_name (Optional[str], optional): Defaults to "nemo_logger.StragglerDetectionCallback".

        Raises:
            ValueError: If invalid config was provided.
        """
        self.initialized: bool = False
        self.logger = logging.getLogger(logger_name)
        self.report_time_interval: float = report_time_interval
        self.calc_relative_gpu_perf: bool = calc_relative_gpu_perf
        self.calc_individual_gpu_perf: bool = calc_individual_gpu_perf
        self.num_gpu_perf_scores_to_print: int = num_gpu_perf_scores_to_print
        self.gpu_relative_perf_threshold: float = gpu_relative_perf_threshold
        self.gpu_individual_perf_threshold: float = gpu_individual_perf_threshold
        self.stop_if_detected: bool = stop_if_detected
        self.enable_ptl_logging: bool = enable_ptl_logging
        self.profiling_interval: int = profiling_interval
        self.scores_to_compute = []
        if self.calc_relative_gpu_perf:
            self.scores_to_compute += ['relative_perf_scores']
        if self.calc_individual_gpu_perf:
            self.scores_to_compute += ['individual_perf_scores']
        if not self.scores_to_compute:
            raise ValueError(
                "No straggler performance scores specified. Check if calc_relative_gpu_perf=True or calc_individual_gpu_perf=True"
            )
        self.interval_est_was_reset = False

    def _wrap_ptl_callables(self, trainer):
        assert getattr(
            trainer.strategy, 'training_step', None
        ), f"{type(trainer.strategy)} does not have 'training_step' method."
        straggler.Detector.wrap_callables(
            callable_ids=[straggler.CallableId(trainer.strategy, "training_step")]
        )

    def setup(self, trainer, pl_module, stage):
        if not self.initialized:
            straggler.Detector.initialize(
                scores_to_compute=self.scores_to_compute,
                gather_on_rank0=True,
                profiling_interval=self.profiling_interval,
                report_time_interval=self.report_time_interval,
            )
            self._wrap_ptl_callables(trainer)
            self.initialized = True

    def teardown(self, trainer, pl_module, stage):
        if self.initialized:
            straggler.Detector.shutdown()
            self.initialized = False

    def _print_stragglers(self, stragglers):
        if rel_stragglers := stragglers['straggler_gpus_relative']:
            self.logger.warning(
                f"STRAGGLER DETECTION WARNING: Some GPUs have worse relative performance. Affected ranks: {rel_stragglers}"
            )
        if indiv_stragglers := stragglers['straggler_gpus_individual']:
            self.logger.warning(
                f"STRAGGLER DETECTION WARNING: Some GPUs performance dropped. Affected ranks: {indiv_stragglers}"
            )

    @staticmethod
    def _format_gpu_scores(rank_to_score, rank_to_node, num_best=3, num_worst=3) -> str:
        num_ranks = len(rank_to_score)
        scores_and_ranks = [(s, r) for r, s in rank_to_score.items()]
        scores_and_ranks.sort(reverse=True)
        res = ""
        if num_ranks > (num_best + num_worst):
            res += f" Worst performing {num_worst}/{num_ranks} ranks:\n"
            for s, r in reversed(scores_and_ranks[-num_worst:]):
                res += f"  Rank={r} Node={rank_to_node[r]} Score={s:.2f}\n"
            res += f" Best performing {num_best}/{num_ranks} ranks:\n"
            for s, r in scores_and_ranks[:num_best]:
                res += f"  Rank={r} Node={rank_to_node[r]} Score={s:.2f}\n"
        else:
            # if the number of ranks is small enough, print them all
            for s, r in reversed(scores_and_ranks):
                res += f"  Rank={r} Node={rank_to_node[r]} Score={s:.2f}\n"
        return res

    def _print_gpu_scores(self, report):

        assert self.num_gpu_perf_scores_to_print > 0

        if self.calc_relative_gpu_perf:
            rel_perf_str = self._format_gpu_scores(
                report.gpu_relative_perf_scores,
                report.rank_to_node,
                num_best=self.num_gpu_perf_scores_to_print,
                num_worst=self.num_gpu_perf_scores_to_print,
            )
            self.logger.info(f"\nGPU relative performance:\n{rel_perf_str}")

        if self.calc_individual_gpu_perf:
            indiv_perf_str = self._format_gpu_scores(
                report.gpu_individual_perf_scores,
                report.rank_to_node,
                num_best=self.num_gpu_perf_scores_to_print,
                num_worst=self.num_gpu_perf_scores_to_print,
            )
            self.logger.info(f"\nGPU individual performance:\n{indiv_perf_str}")

    def _log_gpu_perf_scores(self, pl_module, rank_to_score, rank_to_node, score_prefix):
        """
        Logs GPU performance scores with rank and node information to all PTL loggers enabled through trainer.
        """
        scores_log = {}
        min_val = float('nan')
        med_val = float('nan')
        max_val = float('nan')
        scores = list(rank_to_score.values())
        if scores:
            scores = torch.tensor(scores, dtype=torch.float32)
            min_val = torch.min(scores).item()
            med_val = torch.median(scores).item()
            max_val = torch.max(scores).item()
        scores_log[f"{score_prefix}/min"] = min_val
        scores_log[f"{score_prefix}/median"] = med_val
        scores_log[f"{score_prefix}/max"] = max_val
        try:
            pl_module.log_dict(scores_log, logger=True, batch_size=1, rank_zero_only=True)
        except Exception as e:
            self.logger.error(f"Failed to log GPU performance scores: {e}")

    def _log_gpu_scores(self, pl_module, report):

        assert self.enable_ptl_logging is True

        if self.calc_relative_gpu_perf:
            self._log_gpu_perf_scores(
                pl_module,
                rank_to_score=report.gpu_relative_perf_scores,
                rank_to_node=report.rank_to_node,
                score_prefix="gpu_relative_perf",
            )

        if self.calc_individual_gpu_perf:
            self._log_gpu_perf_scores(
                pl_module,
                rank_to_score=report.gpu_individual_perf_scores,
                rank_to_node=report.rank_to_node,
                score_prefix="gpu_individual_perf",
            )

    def _handle_straggler_report(self, pl_module, report) -> bool:
        stragglers = report.identify_stragglers(
            gpu_rel_threshold=self.gpu_relative_perf_threshold,
            gpu_indiv_threshold=self.gpu_individual_perf_threshold,
        )
        stragglers_found = (
            stragglers['straggler_gpus_relative'] or stragglers['straggler_gpus_individual']
        )
        if stragglers_found:
            self._print_stragglers(stragglers)
        if self.num_gpu_perf_scores_to_print > 0:
            self._print_gpu_scores(report)
        if self.enable_ptl_logging:
            self._log_gpu_scores(pl_module, report)
        return stragglers_found

    def _gather_flag_from_rank0(self, flag):
        flag = torch.tensor(
            [1.0 if flag else 0], device=torch.cuda.current_device(), dtype=torch.float32
        )
        torch.distributed.broadcast(flag, 0)
        flag = bool(flag.item() > 0)
        return flag

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        time_started = time.monotonic()
        rank = trainer.global_rank
        report = straggler.Detector.generate_report_if_interval_elapsed()
        stragglers_found = False
        if rank == 0 and report:
            # gather_on_rank0 is True, so only rank 0 has the report
            stragglers_found = self._handle_straggler_report(pl_module, report)
        # check if the report was generated
        if straggler.Detector.is_interval_elapsed():
            # report was generated on the rank0
            if self.stop_if_detected and self._gather_flag_from_rank0(stragglers_found):
                self._stop_training(trainer)
            # log reporting time
            elapsed = time.monotonic() - time_started
            self.logger.info(f"Straggler report processing time: {elapsed:.3f} sec.")

    def _stop_training(self, trainer) -> None:
        self.logger.error("Detected stragglers. Terminating training...")
        trainer.should_stop = True
        if trainer.checkpoint_callback:
            monitor_candidates = trainer.checkpoint_callback._monitor_candidates(trainer)
            trainer.checkpoint_callback._save_last_checkpoint(trainer, monitor_candidates)
            if hasattr(trainer.strategy.checkpoint_io, 'maybe_finalize_save_checkpoint'):
                self.logger.info("Async checkpointing detected, waiting for it to complete...")
                trainer.strategy.checkpoint_io.maybe_finalize_save_checkpoint(blocking=True)
            sys.exit(1)
