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
import logging
import multiprocessing as mp
import os
import pathlib
import shutil
import signal
import sys
import tempfile

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
from nvidia_resiliency_ext.ptl_resiliency import FaultToleranceCallback, SimulatedFaultParams

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

    def __init__(self, after_steps, exc_cls=None, sys_exit_code=None):
        self.exc_cls = exc_cls
        self.sys_exit_code = sys_exit_code
        self.after_steps = after_steps
        self.steps_made = 0

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.steps_made += 1
        if self.steps_made >= self.after_steps:
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
    ft_cfg.initial_rank_heartbeat_timeout = 4.0
    ft_cfg.rank_heartbeat_timeout = 2.0
    ft_cfg.workload_check_interval = 0.25
    ft_cfg.rank_termination_signal = signal.SIGTERM
    ft_cfg.log_level = logging.DEBUG
    return ft_cfg


@pytest.fixture()
def run_rank_monitors():
    ft_cfg = _get_ft_test_config()
    mp_ctx_spawn = mp.get_context("spawn")
    rank_monitors = []

    try:
        for rank in range(TEST_WORLD_SIZE):
            os.environ["RANK"] = str(rank)
            p = fault_tolerance.RankMonitorServer.run_in_subprocess(ft_cfg, rank, mp_ctx_spawn)
            rank_monitors.append(p)

        yield

    finally:
        for p in rank_monitors:
            with contextlib.suppress(Exception):
                p.terminate()
                p.join(timeout=180)
        del os.environ["RANK"]


def _create_test_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _run_trainining(
    tmp_path,
    max_steps=100_000,
    max_epochs=4,
    max_time=None,
    val_check_interval=None,
    custom_callbacks=None,
    expects_fit_exception=False,
):

    fault_tol_cb = FaultToleranceCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
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
        callbacks=[fault_tol_cb, checkpoint_callback] + custom_callbacks,
    )

    model = SimpleModel()

    fit_exception_caught = False
    try:
        trainer.fit(model, ckpt_path='last')
    except BaseException:
        fit_exception_caught = True

    assert fit_exception_caught == expects_fit_exception


def _run_eval(tmp_path, which='not set'):

    fault_tol_cb = FaultToleranceCallback(
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

    # training is completed due to iters num limit,
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

    # ensure that FT callback does not interfere with out of the training loop evaluation
    _run_eval(tmp_path, which='validate')
    gc.collect()

    _run_eval(tmp_path, which='test')
    gc.collect()


def test_finished_fit_after_all_read(tmp_path, run_rank_monitors):

    # training is completed due all data read
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

    # ensure that FT callback does not interfere with out of the training loop evaluation
    _run_eval(tmp_path, which='validate')
    gc.collect()

    _run_eval(tmp_path, which='test')
    gc.collect()


def test_finished_fit_with_time_limit(tmp_path, run_rank_monitors):

    # training is completed due to time limit,
    # ensure that the finished flag is created

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    # flag should not be created after initial run
    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time={'seconds': 2}, max_steps=1000000)
    gc.collect()

    assert not finished_flag_path.exists()

    # empty run is needed to determine that time limit is reached
    _run_trainining(tmp_path, max_time={'seconds': 2}, max_steps=1000000)
    gc.collect()

    assert finished_flag_path.exists()

    # ensure that FT callback does not interfere with out of the training loop evaluation
    _run_eval(tmp_path, which='validate')
    gc.collect()

    _run_eval(tmp_path, which='test')
    gc.collect()


def test_timeouts_updated_when_graceful_stop(tmp_path, run_rank_monitors):

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time=None, max_steps=100, val_check_interval=75)
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" not in log_content

    # timeouts should be computed after the second, resumed run
    _run_trainining(tmp_path, max_time=None, max_steps=200, val_check_interval=75)
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" in log_content


def test_timeouts_updated_when_exc(tmp_path, run_rank_monitors):

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time=None, max_steps=100, val_check_interval=75)
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" not in log_content

    # incomplete run, due to a unexpected exception
    stop_cb = StoppingPtlCallback(after_steps=20, exc_cls=ValueError)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
    )
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" not in log_content

    # timeouts should be computed, this run exits due to a "graceful stop" exception
    # NOTE: PTL does not call "on_exception" hook when "_TunerExitException" is raised
    stop_cb = StoppingPtlCallback(after_steps=20, exc_cls=_TunerExitException)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=False,
    )
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" in log_content


def test_timeouts_updated_when_sys_exit(tmp_path, run_rank_monitors):

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    assert not finished_flag_path.exists()

    _run_trainining(tmp_path, max_time=None, max_steps=100, val_check_interval=75)
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" not in log_content

    # incomplete run, due to a error exit (code=1)
    stop_cb = StoppingPtlCallback(after_steps=20, exc_cls=None, sys_exit_code=1)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
    )
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" not in log_content

    # timeouts should be computed, this run exits due to a "graceful exit" with code 0
    # NOTE: PTL trainer.fit exits with exception even when "sys.exit(0)" is called in a callback
    stop_cb = StoppingPtlCallback(after_steps=20, exc_cls=None, sys_exit_code=0)
    _run_trainining(
        tmp_path,
        max_time=None,
        max_steps=1000,
        val_check_interval=75,
        custom_callbacks=[stop_cb],
        expects_fit_exception=True,
    )
    gc.collect()

    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()
    assert "Updated FT timeouts" in log_content


def test_simulated_fault(tmp_path):

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)

    finished_flag_path = tmp_path / "finished.flag"
    os.environ['FAULT_TOL_FINISHED_FLAG_FILE'] = str(finished_flag_path)

    sim_fault1 = SimulatedFaultParams(fault_type='rank_hung', base_delay=33.33)
    fault_tol_cb1 = FaultToleranceCallback(
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
    fault_tol_cb2 = FaultToleranceCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
        simulated_fault_params=sim_fault2,
    )
    assert isinstance(fault_tol_cb2.simulated_fault_params, SimulatedFaultParams)
    assert fault_tol_cb2.simulated_fault_params.fault_type == 'random'
    assert fault_tol_cb2.simulated_fault_params.base_delay == 123.0

    fault_tol_cb3 = FaultToleranceCallback(
        autoresume=True,
        calculate_timeouts=True,
        logger_name="test_logger",
        exp_dir=tmp_path,
        simulated_fault_params=None,
    )
    assert fault_tol_cb3.simulated_fault_params is None
