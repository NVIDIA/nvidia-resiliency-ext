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
from typing import Optional, Union

import torch

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


class _TrainingStateMachine:
    """
    This class encapsulates logic for determining when:
    - training is finished successfully (`.is_training_completed` method)
    - FT timeouts can be updated (`.can_update_timeouts` property)

    `on_ ...` methods update the state and should be called from the corresponding PTL callback methods.
    `on_ft_heartbeat_sent` should be called after each FT heartbeat.
    """

    MIN_ITERS_FOR_TIMEOUT_UPDATE = 2

    def __init__(self):
        self.num_tr_iters_total = 0
        self.num_hb_total = 0
        self.num_hb_at_last_save = None
        self.seen_checkpointing = False
        self.loaded_checkpoint = False
        self.caught_exception = False
        self.is_stop_exception = False
        self.training_ended = False
        self.timeouts_updated = False

    def on_setup(self):
        assert self.num_tr_iters_total == 0
        assert self.num_hb_total == 0

    def on_teardown(self):
        self.training_ended = True

    def on_load_checkpoint(self):
        self.loaded_checkpoint = True

    def on_save_checkpoint(self):
        self.num_hb_at_last_save = self.num_hb_total

    def on_train_start(self):
        pass

    def on_train_batch_end(self):
        self.num_tr_iters_total += 1

    def on_train_end(self):
        pass

    def on_validation_start(self):
        pass

    def on_validation_batch_end(self):
        pass

    def on_validation_end(self):
        pass

    def on_exception(self, exc=None):
        self.caught_exception = True
        # check if `sys.exit(0)` was invoked, interpret that as a "clean exit".
        # it's used i.e. by the NeMo preemption callback to stop the training.
        # NOTE: _TunerExitException raised by NeMo StatelessTimer is NOT captured here,
        # but `teardown` hook is called when _TunerExitException is raised.
        self.is_stop_exception = isinstance(exc, SystemExit) and not exc.code

    def on_ft_timeouts_updated(self):
        self.timeouts_updated = True

    def on_ft_heartbeat_sent(self):
        self.num_hb_total += 1
        if not self.seen_checkpointing and self.num_hb_at_last_save is not None:
            # detect checkpointing that makes hearbeat interval longer
            # NOTE: neeed at least one post-checkpointing heartbeat
            num_pre_save = self.num_hb_at_last_save
            num_post_save = self.num_hb_total - self.num_hb_at_last_save
            self.seen_checkpointing = num_pre_save > 0 and num_post_save > 0

    def is_training_completed(self, trainer=None) -> bool:
        """
        Returns True if training is finished sucessfuly.
        """
        # if exiting AND just 0 or 1 training iterations were made AND error is not set,
        # assume training has finished successfully and there is nothing else to do.
        # 1 iteration is made when we run a workload for which 'max_time' elapsed,
        # so need to handle that special case.
        has_failed = self.caught_exception and not self.is_stop_exception
        if self.training_ended and self.num_tr_iters_total <= 1 and not has_failed:
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

    @property
    def can_update_timeouts(self) -> bool:
        """
        Returns True if new timeouts can be computed.
        `.on_timeouts_updated()` resets this property back to False.
        """
        if self.timeouts_updated:
            # timeouts are updated at most once per training run
            return False
        if self.num_tr_iters_total < self.MIN_ITERS_FOR_TIMEOUT_UPDATE:
            # need a few training iters
            return False
        if self.caught_exception and not self.is_stop_exception:
            # if stopping due to an exception, and it isn't "graceful stop" exception
            return False
        # check if there was checkpoint loading and saving
        # this makes heartbeat iterval longer than usual.
        return self.loaded_checkpoint and self.seen_checkpointing


class FaultToleranceCallback(Callback):
    """
    FaultToleranceCallback is a Torch Lightning callback for integration with the Fault Tolerance package.

    FT is only active during a 'fit' stage.
    Training should be run with 'ft_launcher' for the callback to work.
    """

    TIMEOUTS_FILENAME = "_ft_state.json"
    FT_DIR_NAME = "ft_state"

    def __init__(
        self,
        autoresume: bool,
        calculate_timeouts: bool,
        simulated_fault_params: Union[SimulatedFaultParams, dict, None] = None,
        exp_dir: Union[str, pathlib.Path, None] = None,
        logger_name: Optional[str] = "nemo_logger.FaultToleranceCallback",
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
        """

        self.logger = logging.getLogger(logger_name)
        self.fault_tol_client = None
        self.autoresume = autoresume
        self.calculate_timeouts = calculate_timeouts
        self.simulated_fault_params = parse_simulated_fault_params(simulated_fault_params)
        self.state_machine = None
        self.provided_exp_dir = exp_dir
        self.timeouts_file_path = None

    @property
    def is_initialized(self):
        return self.fault_tol_client is not None

    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            self._verify_env()
            self.state_machine = _TrainingStateMachine()
            self.state_machine.on_setup()
            self._setup_fault_tolerance(trainer)

    def teardown(self, trainer, pl_module, stage):
        # FT might be already deinitialized due to an exception
        if stage == "fit" and self.is_initialized:
            self.state_machine.on_teardown()
            if self._is_rank0():
                if self.autoresume and self.state_machine.is_training_completed(trainer):
                    self._create_finished_flag_file()
            self._send_ft_heartbeat()
            self._maybe_update_ft_timeouts()
            self._shutdown_fault_tolerance()

    def on_train_start(self, *args, **kwargs):
        self.state_machine.on_train_start()
        self._send_ft_heartbeat()

    def on_train_batch_end(self, *args, **kwargs):
        self.state_machine.on_train_batch_end()
        self._send_ft_heartbeat()

    def on_train_end(self, *args, **kwargs):
        self.state_machine.on_train_end()
        self._send_ft_heartbeat()

    def on_validation_start(self, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.state_machine.on_validation_start()
            self._send_ft_heartbeat()

    def on_validation_batch_end(self, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.state_machine.on_validation_batch_end()
            self._send_ft_heartbeat()

    def on_validation_end(self, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.state_machine.on_validation_end()
            self._send_ft_heartbeat()

    def on_load_checkpoint(self, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.state_machine.on_load_checkpoint()

    def on_save_checkpoint(self, *args, **kwargs):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.state_machine.on_save_checkpoint()
            # in NeMo, it can happen that there are 2 checkpointing operations
            # one after another, without any training/eval iteration between.
            # send a heartbeat, so in such case we wont get unusually long interval.
            self._send_ft_heartbeat()

    def on_exception(self, trainer, pl_module, exception):
        # this can be called outside of `fit` stage
        if self.is_initialized:
            self.state_machine.on_exception(exception)
            self._send_ft_heartbeat()
            self._maybe_update_ft_timeouts()
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

    def _send_ft_heartbeat(self):
        self.fault_tol_client.send_heartbeat()
        self.state_machine.on_ft_heartbeat_sent()

    def _maybe_update_ft_timeouts(self):
        if self.calculate_timeouts and self.state_machine.can_update_timeouts:
            self._log_info_on_rank0('Updating FT timeouts...')
            self.fault_tol_client.calculate_and_set_hb_timeouts()
            self.state_machine.on_ft_timeouts_updated()
            self._log_info_on_rank0(
                f'Updated FT timeouts. New values: {self.fault_tol_client.hb_timeouts}'
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
                ft_timeouts = self.fault_tol_client.hb_timeouts
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
        self._maybe_load_ft_timeouts()

        ft_timeouts = self.fault_tol_client.hb_timeouts
        if ft_timeouts.are_valid:
            self._log_info_on_rank0(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")
        else:
            if self.calculate_timeouts:
                self._log_info_on_rank0(
                    "Fault tolerance client initialized. Timeouts: not calculated yet."
                )
            else:
                raise RuntimeError(
                    "Fault tolerance doesn't have valid timeouts set and 'calculate_timeouts' is False."
                )
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
            flag_file_path.touch()
        except Exception as e:
            self.logger.error(f"_create_finished_flag_file exception: {e}")
