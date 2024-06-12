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


import json
import logging
import os
import pathlib
from collections import Counter
from typing import Callable, Optional, Union

import torch
import torch.distributed

from ._utils import (
    SimulatedFaultParams,
    is_module_available,
    parse_simulated_fault_params,
    setup_simulated_fault,
)

if is_module_available("lightning"):
    from lightning.pytorch.callbacks import Callback
elif is_module_available("pytorch_lightning"):
    from pytorch_lightning.callbacks import Callback
else:
    raise ImportError("Could not find 'lightning' or 'pytorch_lightning' module")


import nvidia_resiliency_ext.fault_tolerance as ft


class FaultToleranceSectionsCallback(Callback):
    """
    FaultToleranceSectionsCallback is a Torch Lightning callback for integration with the Fault Tolerance package.

    ``FaultToleranceSectionsCallback`` uses the new section-based FT API.
    In this implementation, there are 3 sections:
    - 'setup' which covers rank initialization (from '.setup' to the first training or eval iter)
    - 'step' which covers training and eval steps
    - 'checkpointing' which covers checkpoint related operations

    Everything else goes into the "out-of-section" area.

    FT is only active during a 'fit' stage.
    Training should be run with 'ft_launcher' for the callback to work.
    """

    MIN_TR_ITERS_FOR_STEP_TIMEOUT_UPDATE = 16
    TIMEOUTS_FILENAME = "_ft_state_.json"
    FT_DIR_NAME = "ft_state"

    def __init__(
        self,
        autoresume: bool,
        calculate_timeouts: bool,
        simulated_fault_params: Union[SimulatedFaultParams, dict, None] = None,
        exp_dir: Union[str, pathlib.Path, None] = None,
        logger_name: Optional[str] = "nemo_logger.FaultToleranceCallback",
        exc_handling_policy: Optional[Callable[[Exception], ft.WorkloadControlRequest]] = None,
    ):
        """
        Initialize callback instance.

        This is a lightweight initialization. Most of the initialization is conducted in the 'setup' hook.

        Args:
            autoresume (bool): Set to `True` if the FT auto-resume feature is used (e.g., there are multiple training jobs to be run).
            calculate_timeouts (bool): Set to `True` if FT timeouts should be calculated based on observed heartbeat intervals.
                Calculated timeouts overwrite the timeouts from the FT config.
                Timeouts are computed at the end of a training job, if there was checkpoint loading and saving.
                For example, for training started from scratch, the timeouts are computed at the end of the second job.
            simulated_fault_params (SimulatedFaultParams, dict, DictConfig, None): Simulated fault spec. It's for debugging only. Defaults to None.
                Should be a `SimulatedFaultParams` instance or any object that can be used for SimulatedFaultParams initialization with `SimulatedFaultParams(**obj)`.
            exp_dir (Union[str, pathlib.Path, None], optional): Directory where the FT state should be saved.
                Must be available for all training jobs. NOTE: Beware that PTL can move files written to its `trainer.log_dir`.
                Defaults to None, in which case it defaults to `trainer.log_dir/ft_state`.
            logger_name (Optional[str], optional): Logger name to be used.
                Defaults to "nemo_logger.FaultToleranceCallback".
            exc_handling_policy (Optional[Callable[[Exception], ft.WorkloadControlRequest]]):
                A callback function that manages workload behavior when an exception occurs.
                By default, workers are restarted until `ft_launcher --max-restarts=...` is reached.
                Implementing this callback allows for halting training or excluding nodes
                from subsequent rendezvous.
        """

        self.logger = logging.getLogger(logger_name)
        self.fault_tol_client = None
        self.autoresume = autoresume
        self.calculate_timeouts = calculate_timeouts
        self.simulated_fault_params = parse_simulated_fault_params(simulated_fault_params)
        self.provided_exp_dir = exp_dir
        self.timeouts_file_path = None
        self.is_training_ended = False
        self.caught_exception = False
        self.is_stop_exception = False
        self.was_checkpoint_loaded = False
        self.uses_async_checkpointing = False
        self.section_entry_cnt = Counter()
        self.last_checkpoint_iter = 0
        self.last_timeouts_upd_iter = 0
        self.num_tr_iters_total = 0
        self.num_eval_iters_total = 0
        self.exc_handling_policy = exc_handling_policy

    @property
    def is_initialized(self):
        return self.fault_tol_client is not None

    def _start_section(self, section_name, error_if_opened=True):
        self.section_entry_cnt[section_name] += 1
        if self.section_entry_cnt[section_name] == 1:
            self.fault_tol_client.start_section(section_name)
        else:
            assert self.section_entry_cnt[section_name] > 1
            if error_if_opened:
                raise RuntimeError(f"Section {section_name} was already open")

    def _end_section(self, section_name, error_if_closed=True):
        if self.section_entry_cnt[section_name] > 0:
            self.section_entry_cnt[section_name] -= 1
            if self.section_entry_cnt[section_name] == 0:
                self.fault_tol_client.end_section(section_name)
        else:
            assert self.section_entry_cnt[section_name] == 0
            if error_if_closed:
                raise RuntimeError(f"Section {section_name} was already closed")

    def _wrap_into_section(self, obj, method_name, section_name, allow_reopen=False):
        original_method = getattr(obj, method_name)
        assert callable(
            original_method
        ), f"{method_name} is not a callable method on the object {obj}."

        def __patched_method(*args, **kwargs):
            if self.is_initialized:
                self._start_section(section_name, error_if_opened=not allow_reopen)
                result = original_method(*args, **kwargs)
                self._end_section(section_name)
                return result
            else:
                return original_method(*args, **kwargs)

        setattr(obj, method_name, __patched_method)

    def _wrap_callables(self, trainer):
        assert hasattr(trainer, 'strategy'), "Trainer instance should have .strategy attribute"
        assert hasattr(
            trainer.strategy, 'checkpoint_io'
        ), "Strategy instance should have .checkpoint_io attribute"
        self._wrap_into_section(trainer.strategy, 'training_step', 'step')
        self._wrap_into_section(trainer.strategy, 'validation_step', 'step')
        # checkpointing section can be nested, e.g. in NeMo `maybe_finalize_save_checkpoint` calls `remove_checkpoint`
        self._wrap_into_section(
            trainer.strategy, 'save_checkpoint', 'checkpointing', allow_reopen=True
        )
        self._wrap_into_section(
            trainer.strategy, 'remove_checkpoint', 'checkpointing', allow_reopen=True
        )
        if hasattr(trainer.strategy.checkpoint_io, 'maybe_finalize_save_checkpoint'):
            self.uses_async_checkpointing = True
            self._wrap_into_section(
                trainer.strategy.checkpoint_io,
                'maybe_finalize_save_checkpoint',
                'checkpointing',
                allow_reopen=True,
            )

    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            self._verify_env()
            self._setup_fault_tolerance(trainer)
            self._wrap_callables(trainer)
            self._start_section('setup', error_if_opened=True)
            assert self.is_initialized

    def teardown(self, trainer, pl_module, stage):
        if stage == "fit" and self.is_initialized:
            self.is_training_ended = True
            if self._is_rank0():
                if self.autoresume and self._is_training_completed(trainer):
                    self._create_finished_flag_file()
            # setup section might be open if there were no iterations in this run
            self._end_section('setup', error_if_closed=False)
            self._maybe_update_ft_timeouts(is_final=True)
            self._shutdown_fault_tolerance()

    def on_train_batch_start(self, *args, **kwargs):
        self._end_section('setup', error_if_closed=False)
        self.num_tr_iters_total += 1

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # update FT timeouts after each checkpoint save is completed
        if self.last_timeouts_upd_iter < self.last_checkpoint_iter:
            self._maybe_update_ft_timeouts(is_final=False)
            self.last_timeouts_upd_iter = trainer.global_step

    def on_validation_batch_start(self, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.num_eval_iters_total += 1
            self._end_section('setup', error_if_closed=False)

    def on_load_checkpoint(self, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            # TODO: is it a persistent checkpoint?
            # local/in-memory reads can be very fast
            # and we can overestimate 'setup' timeout
            self.was_checkpoint_loaded = True

    def on_save_checkpoint(self, trainer, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.last_checkpoint_iter = trainer.global_step

    def on_exception(self, trainer, pl_module, exc):
        # this can be called outside of `fit` stage
        # NOTE: this is called on a rank that raised the exception,
        # might not be called on other ranks or might be called due to a different exception
        if self.is_initialized:
            self.caught_exception = True
            # setup section might be open if there were no iterations in this run
            self._end_section('setup', error_if_closed=False)
            # check if `sys.exit(0)` was invoked, interpret that as a "clean exit".
            # it's used i.e. by the NeMo preemption callback to stop the training.
            # NOTE: _TunerExitException raised by NeMo StatelessTimer is NOT captured here,
            # but `teardown` hook is called when _TunerExitException is raised.
            self.is_stop_exception = isinstance(exc, SystemExit) and not exc.code
            self._maybe_update_ft_timeouts(is_final=True)
            # check if the workload should be modified (e.g. node removed or training halted)
            if self.exc_handling_policy is not None:
                req = self.exc_handling_policy(exc)
                self.logger.debug(f"Exception handling policy callback returned: {req}.")
                assert isinstance(
                    req, ft.WorkloadControlRequest
                ), f"Exception handling policy should return WorkloadControlRequest, it returned {req}"
                self.fault_tol_client.send_workload_control_request(req)
                if req.action == ft.WorkloadAction.ShutdownWorkload:
                    self.logger.warning(
                        "Shutdown workload requested, so creating finished flag file."
                    )
                    self._create_finished_flag_file()
            self._shutdown_fault_tolerance()

    def _is_rank0(self):
        return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

    def _log_info_on_rank0(self, msg):
        if self._is_rank0():
            self.logger.info("[FaultToleranceCallback@rank0] " + str(msg))

    def _verify_env(self):
        if self.autoresume and not os.environ.get('FAULT_TOL_FINISHED_FLAG_FILE', ''):
            raise RuntimeError(
                "'FAULT_TOL_FINISHED_FLAG_FILE' env variable is not set. Was this job launched with FT launcher?"
            )

    def _verify_ft_config(self):
        expected_sections = {'setup', 'step', 'checkpointing'}
        if set(self.fault_tol_client.cfg.rank_section_timeouts.keys()) != expected_sections:
            raise ValueError(f"rank_section_timeouts should contain {expected_sections} entries")

    def _update_ft_timeouts(self, selected_sections, calc_out_of_section):
        self._log_info_on_rank0(
            f'Updating FT timeouts for: {selected_sections} update out-of-section: {calc_out_of_section}'
        )
        self.fault_tol_client.calculate_and_set_section_timeouts(
            selected_sections=selected_sections, calc_out_of_section=calc_out_of_section
        )
        self._log_info_on_rank0(
            f'Updated FT timeouts. New values: {self.fault_tol_client.section_timeouts}'
        )
        if self._is_rank0():
            # FT state is the same on all ranks, so we can save it only on rank 0
            with open(self.timeouts_file_path, mode='w') as f:
                json.dump(self.fault_tol_client.state_dict(), f)

    def _maybe_load_ft_timeouts(self):
        if self.calculate_timeouts:
            # we load the timeouts only when calculate_timeouts=True
            loaded_ft_state_dict = {}
            if self.timeouts_file_path.exists():
                with open(self.timeouts_file_path, mode='r') as f:
                    loaded_ft_state_dict = json.load(f)
            if loaded_ft_state_dict:
                self.fault_tol_client.load_state_dict(loaded_ft_state_dict)
                ft_timeouts = self.fault_tol_client.section_timeouts
                self._log_info_on_rank0(f"Fault tolerance timeouts loaded: {ft_timeouts}")

    def _setup_fault_tolerance(self, trainer):

        assert not self.is_initialized, "Fault tolerance client already initialized."

        self.fault_tol_client = ft.RankMonitorClient()

        # Format timeouts file path
        if self.provided_exp_dir:
            ft_dir = pathlib.Path(self.provided_exp_dir)
        else:
            ft_dir = pathlib.Path(trainer.log_dir) / self.FT_DIR_NAME
            if self._is_rank0():
                ft_dir.mkdir(exist_ok=True)
            trainer.strategy.barrier()

        self._log_info_on_rank0(f"Fault tolerance dir: {ft_dir}")
        if not ft_dir.exists():
            raise ValueError(f"Fault tolerance save directory does not exist: {ft_dir}")

        self.timeouts_file_path = ft_dir / self.TIMEOUTS_FILENAME

        self.fault_tol_client.init_workload_monitoring()
        self._verify_ft_config()

        self._maybe_load_ft_timeouts()

        ft_timeouts = self.fault_tol_client.section_timeouts
        self._log_info_on_rank0(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")

        # Simulated fault for testing/debug purposes
        if self.simulated_fault_params:
            setup_simulated_fault(self.simulated_fault_params)

        assert self.is_initialized

    def _shutdown_fault_tolerance(self):
        if self.is_initialized:
            self.fault_tol_client.shutdown_workload_monitoring()
            self.fault_tol_client = None
        assert not self.is_initialized

    def _create_finished_flag_file(self):
        try:
            flag_file_path = pathlib.Path(os.environ["FAULT_TOL_FINISHED_FLAG_FILE"])
            flag_file_path.touch(exist_ok=True)
        except Exception as e:
            self.logger.error(f"_create_finished_flag_file exception: {e}")

    def _maybe_update_ft_timeouts(self, is_final=False):
        if not self.calculate_timeouts:
            return

        # clean exit means no exception, or a "stop exception" that is used by the NeMo/PTL callbacks for training stopping.
        is_clean_exit = not self.caught_exception or self.is_stop_exception
        if not is_clean_exit:
            self._log_info_on_rank0("Rank exits due to a failure, won't update the FT timeouts")
            return

        # check which sections can be updated
        selected_sections = []

        if self.was_checkpoint_loaded:
            # checkpoint loading makes the init time longer (TODO: was it local/in-memory?)
            selected_sections.append('setup')
        else:
            self._log_info_on_rank0(
                "Can't update the setup section timeout until persistent checkpoint is loaded"
            )

        if self.num_tr_iters_total > self.MIN_TR_ITERS_FOR_STEP_TIMEOUT_UPDATE:
            # need a few training inters before updating the step timeout
            selected_sections.append('step')
        else:
            self._log_info_on_rank0(
                "Need to see more training iterations to update the step section timeout"
            )

        was_checkpoint_saved = self.last_checkpoint_iter > 0
        if was_checkpoint_saved:
            if not self.uses_async_checkpointing:
                selected_sections.append("checkpointing")
            else:
                # There can be too much checkpointing section time variability
                # across runs with the async checkpointing, e.g. in some runs all checkpointing
                # work can be parallelized (=short checkpointing sections) while in others we can
                # hit a costly finalization.
                self._log_info_on_rank0(
                    "Can't update the checkpointing section timeout with async checkpointing"
                )
        else:
            self._log_info_on_rank0(
                "Checkpointing section is not updated until a checkpoint was saved"
            )

        update_oos = False
        if is_final:
            # Update the out-of-section if it was a complete run
            if {'setup', 'step'}.issubset(selected_sections) and was_checkpoint_saved:
                update_oos = True
            else:
                self._log_info_on_rank0(
                    "The out-of-section timeout won't be updated until all FT sections were seen"
                )

        if selected_sections or update_oos:
            self._update_ft_timeouts(
                selected_sections=selected_sections, calc_out_of_section=update_oos
            )

    def _is_training_completed(self, trainer=None) -> bool:
        """
        Returns True if training is finished sucessfuly.
        """
        # if exiting AND just 0 or 1 training iterations were made AND error is not set,
        # assume training has finished successfully and there is nothing else to do.
        # 1 iteration is made when we run a workload for which 'max_time' elapsed,
        # so need to handle that special case.
        has_failed = self.caught_exception and not self.is_stop_exception
        if self.is_training_ended and self.num_tr_iters_total <= 1 and not has_failed:
            return True

        if trainer is not None:
            # if iters limit is reached:
            if (
                isinstance(trainer.max_steps, int)
                and trainer.max_steps > 0
                and trainer.global_step >= trainer.max_steps
            ):
                return True
            # if epochs limit is reached
            if (
                isinstance(trainer.max_epochs, int)
                and trainer.max_epochs > 0
                and trainer.current_epoch >= trainer.max_epochs
            ):
                return True

        return False
