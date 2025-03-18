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

import contextlib
import gc
import json
import logging
import multiprocessing as mp
import os
import pathlib
import shutil
import signal
import sys
import tempfile

from nvidia_resiliency_ext.fault_tolerance.data import FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR
from nvidia_resiliency_ext.fault_tolerance.ipc_connector import IpcConnector
from nvidia_resiliency_ext.ptl_resiliency._utils import is_module_available

if is_module_available("lightning"):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.utilities.exceptions import _TunerExitException
elif is_module_available("pytorch_lightning"):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.utilities.exceptions import _TunerExitException
else:
    raise ImportError("Could not find 'lightning' or 'pytorch_lightning' module")

import pytest
import torch
from torch import nn

import nvidia_resiliency_ext.fault_tolerance as fault_tolerance
from nvidia_resiliency_ext.ptl_resiliency import (
    FaultToleranceSectionsCallback,
    SimulatedFaultParams,
)

TEST_WORLD_SIZE = 1


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(32), torch.ones(10)

    def __len__(self):
        return self.__dataset_len


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        pl.seed_everything(1234)
        self.layer_1 = nn.Linear(32, 16)
        self.layer_2 = nn.Linear(16, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        dataset = OnesDataset(256)
        return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    def val_dataloader(self):
        dataset = OnesDataset(128)
        return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    def test_dataloader(self):
        dataset = OnesDataset(128)
        return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)


class StoppingPtlCallback(Callback):

    def __init__(self, after_steps, exc_cls=None, sys_exit_code=None, save_checkpoint=True):
        self.exc_cls = exc_cls
        self.sys_exit_code = sys_exit_code
        self.after_steps = after_steps
        self.save_checkpoint = save_checkpoint
        self.steps_made = 0

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.steps_made += 1
        if self.steps_made >= self.after_steps:
            if self.save_checkpoint:
                self._save_last_checkpoint(trainer)
            self._exit()

    def _exit(self):
        if self.exc_cls is not None:
            raise self.exc_cls
        if self.sys_exit_code is not None:
            sys.exit(self.sys_exit_code)
        assert False, "should not get here"

    def _save_last_checkpoint(self, trainer):
        chkpt_cb = [cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)]
        assert len(chkpt_cb) == 1
        chkpt_cb = chkpt_cb[0]
        monitor_candidates = chkpt_cb._monitor_candidates(trainer)
        chkpt_cb._save_last_checkpoint(trainer, monitor_candidates)


@pytest.fixture
def tmp_path():
    try:
        dirpath = tempfile.mkdtemp()
        yield pathlib.Path(dirpath)
    finally:
        shutil.rmtree(dirpath)


def _get_ft_test_config():
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = None
    ft_cfg.rank_heartbeat_timeout = None
    ft_cfg.rank_section_timeouts = {'setup': 20, 'step': 5, 'checkpointing': 15}
    ft_cfg.rank_out_of_section_timeout = 5.0
    ft_cfg.workload_check_interval = 0.25
    ft_cfg.rank_termination_signal = signal.SIGTERM
    ft_cfg.log_level = logging.DEBUG
    return ft_cfg


class _SetFtIpcSocketPathCallback(Callback):
    def setup(self, trainer, pl_module, stage):
        rank = trainer.global_rank
        ipc_sock_path = f"{tempfile.gettempdir()}/_rmon_r{rank}.socket"
        os.environ[FT_RANK_MONITOR_IPC_SOCKET_ENV_VAR] = ipc_sock_path


@pytest.fixture()
def run_rank_monitors(request):

    if hasattr(request, 'param'):
        assert isinstance(request.param, fault_tolerance.FaultToleranceConfig)
        ft_cfg = request.param
    else:
        ft_cfg = _get_ft_test_config()

    mp_ctx_spawn = mp.get_context("spawn")
    rank_monitors = []

    try:
        for rank in range(TEST_WORLD_SIZE):
            os.environ["RANK"] = str(rank)
            ipc_sock_path = f"{tempfile.gettempdir()}/_rmon_r{rank}.socket"
            p = fault_tolerance.RankMonitorServer.run_in_subprocess(
                cfg=ft_cfg,
                ipc_socket_path=ipc_sock_path,
                is_restarter_logger=False,
                mp_ctx=mp_ctx_spawn,
            )
            rank_monitors.append(p)

        yield

    finally:
        for p in rank_monitors:
            with contextlib.suppress(Exception):
                p.terminate()
                p.join(timeout=180)
        del os.environ["RANK"]


@pytest.fixture()
def ipc_connector(tmp_path):
    ft_launcher_socket = str(tmp_path / "ft_launcher.socket")
    os.environ['FT_LAUNCHER_IPC_SOCKET'] = ft_launcher_socket
    try:
        launcher_conn = IpcConnector(socket_path=ft_launcher_socket)
        launcher_conn.start_receiving()
        yield launcher_conn
    finally:
        launcher_conn.stop_receiving()


def _create_test_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _run_trainining(
    tmp_path,
    max_steps=100_000,
    max_epochs=4,
    max_time=None,
    val_check_interval=None,
    chkpt_save_interval=None,
    custom_callbacks=None,
    expects_fit_exception=False,
    ft_exc_handling_policy=None,
):

    fault_tol_cb = FaultToleranceSectionsCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
        exc_handling_policy=ft_exc_handling_policy,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        every_n_train_steps=chkpt_save_interval,
        save_last=True,
    )

    custom_callbacks = custom_callbacks if custom_callbacks else []

    trainer = pl.Trainer(
        strategy='ddp',
        devices=1,
        accelerator='gpu',
        logger=False,
        max_steps=max_steps,
        max_epochs=max_epochs,
        max_time=max_time,
        val_check_interval=val_check_interval,
        callbacks=[_SetFtIpcSocketPathCallback(), fault_tol_cb, checkpoint_callback]
        + custom_callbacks,
    )

    model = SimpleModel()

    fit_exception_caught = None
    try:
        trainer.fit(model, ckpt_path='last')
    except BaseException as e:
        print(f"Exception from trainer.fit: {e}")
        fit_exception_caught = e

    is_exc_expected = expects_fit_exception is not False
    is_exc_captured = fit_exception_caught is not None
    assert is_exc_captured == is_exc_expected
    if isinstance(expects_fit_exception, type) and issubclass(expects_fit_exception, BaseException):
        assert isinstance(fit_exception_caught, expects_fit_exception)


def _get_ft_state(exp_dir):
    ft_state_path = os.path.join(exp_dir, FaultToleranceSectionsCallback.TIMEOUTS_FILENAME)
    assert os.path.exists(ft_state_path)
    state = None
    with open(ft_state_path, mode='r') as f:
        state = json.load(f)
    timeouts = state[fault_tolerance.RankMonitorClient.CURRENT_TIMEOUTS_STATE_KEY]
    r0 = set(timeouts['section']['calculated_sections'])
    r1 = bool(timeouts['section']['is_out_of_section_calculated'])
    return r0, r1


def _get_timeout_updates_count(log_file_path):
    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    return log_content.count("Updating FT timeouts")


def _run_eval(tmp_path, which='not set'):

    fault_tol_cb = FaultToleranceSectionsCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        save_last=True,
    )

    trainer = pl.Trainer(
        strategy='ddp',
        devices=1,
        accelerator='gpu',
        logger=False,
        callbacks=[fault_tol_cb, checkpoint_callback],
    )

    model = SimpleModel()

    if which == 'validate':
        trainer.validate(model, ckpt_path='last')
    elif which == 'test':
        trainer.test(model, ckpt_path='last')
    else:
        raise ValueError(f"Invalid 'which' value: {which} should be 'validate' or 'test'")


def test_finished_fit_with_iter_limit(tmp_path, run_rank_monitors):

    # training is completed due to the iters num limit,
    # ensure that the finished flag is created

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time=None, max_steps=128)
    gc.collect()

    assert finished_flag_path.exists()

    # force empty run, that should re-create "finished flag"
    finished_flag_path.unlink()
    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time=None, max_steps=128)
    gc.collect()

    assert finished_flag_path.exists()


def test_eval_with_callback(tmp_path, run_rank_monitors):

    # ensure that FT callback does not interfere with evaluation and test stages

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    _run_eval(tmp_path, which='validate')
    gc.collect()

    _run_eval(tmp_path, which='test')
    gc.collect()

    assert not finished_flag_path.exists()


def test_finished_fit_after_all_read(tmp_path, run_rank_monitors):

    # training is completed due all data processed
    # ensure that the finished flag is created

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time=None, max_steps=-1, max_epochs=2)
    gc.collect()

    assert finished_flag_path.exists()

    # force empty run, that should re-create "finished flag"
    finished_flag_path.unlink()
    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time=None, max_steps=128)
    gc.collect()

    assert finished_flag_path.exists()


def test_finished_fit_with_time_limit(tmp_path, run_rank_monitors):

    # training is completed due to the time limit,
    # ensure that the finished flag is created

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    # with the current impl, flag is not created after the initial run
    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time={'seconds': 2}, max_steps=1000000)
    gc.collect()

    assert not finished_flag_path.exists()

    # empty run is needed to determine that time limit is reached
    _run_trainining(tmp_path, max_time={'seconds': 2}, max_steps=1000000)
    gc.collect()

    assert finished_flag_path.exists()


def test_timeouts_updated_when_graceful_stop(tmp_path, run_rank_monitors):

    print(f"TEMP DIR = {tmp_path}")

    log_file_path = tmp_path / "test.log"

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    _create_test_logger("test_logger", log_file_path)

    # initial full run, that exits due to the iterations limit
    # step and checkpointing sections should be computed at (50, 75, 100)
    _run_trainining(
        tmp_path, max_time=None, max_steps=100, chkpt_save_interval=50, val_check_interval=75
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 3
    updated_sections, updated_oos = _get_ft_state(tmp_path)
    assert updated_sections == set(['step', 'checkpointing'])
    assert not updated_oos

    _create_test_logger("test_logger", log_file_path)

    # resumed run, starts from iter 100, and exits due to the iters limit
    # setup, step, checkpointing timeouts should be computed at (150, 200)
    # all sections + out of section should be computed at 200
    _run_trainining(
        tmp_path, max_time=None, max_steps=200, chkpt_save_interval=50, val_check_interval=75
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 2
    updated_sections, updated_oos = _get_ft_state(tmp_path)
    assert updated_sections == set(['setup', 'step', 'checkpointing'])
    assert updated_oos


def test_timeouts_updated_when_exc(tmp_path, run_rank_monitors):

    log_file_path = tmp_path / "test.log"

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    # initial run that exits due to the iters limit
    # step and checkpointing sections timeouts should be computed at (25, 50, 75, 80, 100)
    _create_test_logger("test_logger", log_file_path)
    _run_trainining(
        tmp_path, max_time=None, max_steps=100, chkpt_save_interval=25, val_check_interval=80
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 5
    updated_sections, updated_oos = _get_ft_state(tmp_path)
    assert updated_sections == set(['step', 'checkpointing'])
    assert not updated_oos

    # resumed (from iter 100) incomplete run, due to an exception, timeouts are not updated
    _create_test_logger("test_logger", log_file_path)
    stop_cb = StoppingPtlCallback(after_steps=20, exc_cls=ValueError, save_checkpoint=False)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 0

    # resumed run (from iter 100)
    # timeouts should be computed, after post-eval checkpointing and at the end
    # this run exits due to a "graceful stop" exception
    # NOTE: PTL does not call "on_exception" hook when "_TunerExitException" is raised
    _create_test_logger("test_logger", log_file_path)
    stop_cb = StoppingPtlCallback(
        after_steps=100, exc_cls=_TunerExitException, save_checkpoint=True
    )
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=False,
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 2
    updated_sections, updated_oos = _get_ft_state(tmp_path)
    assert updated_sections == set(['setup', 'step', 'checkpointing'])
    assert updated_oos


def test_timeouts_updated_when_sys_exit(tmp_path, run_rank_monitors):

    log_file_path = tmp_path / "test.log"

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    # run stopped normally, due to the iters limit
    # should update the timeouts after post-eval checkpoint and at the end (iters 75, 100)
    # setup and out-of-section should not be updated, as there was not checkpoint loading
    _create_test_logger("test_logger", log_file_path)
    _run_trainining(tmp_path, max_time=None, max_steps=100, val_check_interval=75)
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 2
    updated_sections, updated_oos = _get_ft_state(tmp_path)
    assert updated_sections == set(['step', 'checkpointing'])
    assert not updated_oos

    # incomplete resumed run, starts from iter 100. stops at iter 200, due to a error exit (code=1)
    # should update timeouts (setup, step, checkpointing) at iter 150, 175 (post-eval)
    _create_test_logger("test_logger", log_file_path)
    stop_cb = StoppingPtlCallback(
        after_steps=100, exc_cls=None, sys_exit_code=1, save_checkpoint=False
    )
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 2
    updated_sections, updated_oos = _get_ft_state(tmp_path)
    assert updated_sections == set(['setup', 'step', 'checkpointing'])
    assert not updated_oos

    # this resumed run starts at 175, exits due to a "graceful exit" with code 0
    # timeouts should be computed at 195, 200 (when leaving)
    # NOTE: PTL trainer.fit exits with exception even when "sys.exit(0)" is called in a callback
    _create_test_logger("test_logger", log_file_path)
    stop_cb = StoppingPtlCallback(
        after_steps=25, exc_cls=None, sys_exit_code=0, save_checkpoint=True
    )
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=20,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 2
    updated_sections, updated_oos = _get_ft_state(tmp_path)
    assert updated_sections == set(['setup', 'step', 'checkpointing'])
    assert updated_oos


def test_simulated_fault(tmp_path):

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    sim_fault1 = SimulatedFaultParams(fault_type='rank_hung', base_delay=33.33)
    fault_tol_cb1 = FaultToleranceSectionsCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
        simulated_fault_params=sim_fault1,
    )
    assert isinstance(fault_tol_cb1.simulated_fault_params, SimulatedFaultParams)
    assert fault_tol_cb1.simulated_fault_params.fault_type == 'rank_hung'
    assert fault_tol_cb1.simulated_fault_params.base_delay == 33.33

    sim_fault2 = {'fault_type': 'random', 'base_delay': 123.0}
    fault_tol_cb2 = FaultToleranceSectionsCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
        simulated_fault_params=sim_fault2,
    )
    assert isinstance(fault_tol_cb2.simulated_fault_params, SimulatedFaultParams)
    assert fault_tol_cb2.simulated_fault_params.fault_type == 'random'
    assert fault_tol_cb2.simulated_fault_params.base_delay == 123.0

    fault_tol_cb3 = FaultToleranceSectionsCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
        simulated_fault_params=None,
    )
    assert fault_tol_cb3.simulated_fault_params is None


def _get_ft_test_config_no_step_section():
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = None
    ft_cfg.rank_heartbeat_timeout = None
    ft_cfg.rank_section_timeouts = dict()
    ft_cfg.rank_out_of_section_timeout = 4.0
    ft_cfg.workload_check_interval = 0.25
    ft_cfg.rank_termination_signal = signal.SIGTERM
    ft_cfg.log_level = logging.DEBUG
    return ft_cfg


def _get_ft_test_config_unknown_sections():
    ft_cfg = fault_tolerance.FaultToleranceConfig()
    ft_cfg.initial_rank_heartbeat_timeout = None
    ft_cfg.rank_heartbeat_timeout = None
    ft_cfg.rank_section_timeouts = {
        'setup': 20,
        'step': 5,
        'checkpointing': 15,
        'unexpected_section': 123.0,
    }
    ft_cfg.rank_out_of_section_timeout = 4.0
    ft_cfg.workload_check_interval = 0.25
    ft_cfg.rank_termination_signal = signal.SIGTERM
    ft_cfg.log_level = logging.DEBUG
    return ft_cfg


invalid_configs = [
    _get_ft_test_config_no_step_section(),
    _get_ft_test_config_unknown_sections(),
]


@pytest.mark.parametrize("run_rank_monitors", invalid_configs, indirect=True)
def test_invalid_config(tmp_path, run_rank_monitors):
    log_file_path = tmp_path / "test.log"
    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)
    assert not finished_flag_path.exists()
    _create_test_logger("test_logger", log_file_path)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=20,
        val_check_interval=10,
        expects_fit_exception=ValueError,
    )


def test_workload_ctrl_messages(tmp_path, run_rank_monitors, ipc_connector):

    # Check FT callback exception handling policy
    # On __TestExc00 we request for the node exclusion
    # On __TestExc01 we request for the workload shutdown
    # Check if messages are received via IPC
    # (in real world ft_launcher is receiving them)

    class __TestExc00(Exception):
        pass

    class __TestExc01(Exception):
        pass

    def __exc_handling_cb(e) -> fault_tolerance.WorkloadControlRequest:
        if isinstance(e, __TestExc00):
            return fault_tolerance.WorkloadControlRequest(
                fault_tolerance.WorkloadAction.ExcludeThisNode, "test desc1"
            )
        if isinstance(e, __TestExc01):
            return fault_tolerance.WorkloadControlRequest(
                fault_tolerance.WorkloadAction.ShutdownWorkload, "test desc2"
            )
        raise AssertionError("Unexpected exc type")

    log_file_path = tmp_path / "test.log"

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()
    assert len(ipc_connector.fetch_received()) == 0

    # incomplete run, due to an exception __TestExc00
    # check if the request is received via IPC, workload is not finished yet
    _create_test_logger("test_logger", log_file_path)
    stop_cb = StoppingPtlCallback(after_steps=20, exc_cls=__TestExc00, save_checkpoint=False)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
        ft_exc_handling_policy=__exc_handling_cb,
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 0
    assert len(ipc_connector.fetch_received()) == 1
    assert len(ipc_connector.fetch_received()) == 0
    assert not finished_flag_path.exists()

    # incomplete run, due to an exception __TestExc01
    # this time we requested for the workload shutdown,
    # so check if the request is received via IPC and the finished flag is set
    _create_test_logger("test_logger", log_file_path)
    stop_cb = StoppingPtlCallback(after_steps=20, exc_cls=__TestExc01, save_checkpoint=False)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
        ft_exc_handling_policy=__exc_handling_cb,
    )
    gc.collect()

    assert _get_timeout_updates_count(log_file_path) == 0
    assert len(ipc_connector.fetch_received()) == 1
    assert len(ipc_connector.fetch_received()) == 0
    assert finished_flag_path.exists()
