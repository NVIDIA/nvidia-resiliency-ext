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
import os
import pathlib
import shutil
import tempfile

import pytest
import lightning.pytorch as pl
import torch
from torch import nn

from nvidia_resiliency_ext.ptl_resiliency import StragglerDetectionCallback

pytestmark = pytest.mark.gpu


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
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        dataset = OnesDataset(1024 * 1024)
        return torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)

    def val_dataloader(self):
        dataset = OnesDataset(128 * 1024)
        return torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)


@pytest.fixture
def tmp_path():
    try:
        dirpath = tempfile.mkdtemp()
        yield pathlib.Path(dirpath)
    finally:
        shutil.rmtree(dirpath)


def _create_test_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def test_prints_perf_scores_when_fitting(tmp_path):
    # Run dummy 1 rank DDP training
    # Training time is limited to 3 seconds and straggler reporting is set to 1 second
    # Check if there are straggler related logs in the captured log
    max_steps = 1_00_000

    cb_args = dict(
        report_time_interval=1.0,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        num_gpu_perf_scores_to_print=1,
        gpu_relative_perf_threshold=0.0,
        gpu_individual_perf_threshold=0.0,
        enable_ptl_logging=False,
        stop_if_detected=False,
        logger_name="test_logger",
    )

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)
    straggler_det_cb = StragglerDetectionCallback(**cb_args)

    ptl_logger = pl.loggers.CSVLogger(save_dir=tmp_path, name=None, flush_logs_every_n_steps=1)

    trainer = pl.Trainer(
        strategy='ddp',
        devices=1,
        accelerator='gpu',
        enable_checkpointing=False,
        logger=ptl_logger,
        max_steps=max_steps,
        max_time={"seconds": 4},
        callbacks=[straggler_det_cb],
    )

    model = SimpleModel()
    trainer.fit(model)

    # num_gpu_perf_scores_to_print=1 so straggler related logs are expected in the captured log
    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()

    assert "GPU relative" in log_content
    assert "GPU individual" in log_content

    # enable_ptl_logging=False so no straggler related logs are expected in CSV log
    csv_log_content = None
    csv_file_path = os.path.join(ptl_logger.log_dir, "metrics.csv")
    with open(csv_file_path) as f:
        csv_log_content = f.read()

    assert "gpu_relative" not in csv_log_content
    assert "gpu_individual" not in csv_log_content


def test_logs_perf_scores_when_fitting(tmp_path):
    # Run dummy 1 rank DDP training
    # Training time is limited to 3 seconds and straggler reporting is set to 1 second
    # Check if there are straggler related logs in the the CSV log file
    max_steps = 1_00_000

    cb_args = dict(
        report_time_interval=1.0,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        num_gpu_perf_scores_to_print=0,
        gpu_relative_perf_threshold=0.0,
        gpu_individual_perf_threshold=0.0,
        enable_ptl_logging=True,
        stop_if_detected=False,
        logger_name="test_logger",
    )

    log_file_path = tmp_path / "test.log"
    _create_test_logger("test_logger", log_file_path)
    straggler_det_cb = StragglerDetectionCallback(**cb_args)

    ptl_logger = pl.loggers.CSVLogger(save_dir=tmp_path, name=None, flush_logs_every_n_steps=1)

    trainer = pl.Trainer(
        strategy='ddp',
        devices=1,
        accelerator='gpu',
        enable_checkpointing=False,
        logger=ptl_logger,
        max_steps=max_steps,
        max_time={"seconds": 4},
        callbacks=[straggler_det_cb],
    )

    model = SimpleModel()
    trainer.fit(model)

    # num_gpu_perf_scores_to_print=0 so no straggler related logs are expected in the captured log
    log_content = None
    with open(log_file_path) as f:
        log_content = f.read()

    assert "GPU relative" not in log_content
    assert "GPU individual" not in log_content

    # enable_ptl_logging=True so straggler related logs are expected in CSV
    csv_log_content = None
    csv_file_path = os.path.join(ptl_logger.log_dir, "metrics.csv")
    with open(csv_file_path) as f:
        csv_log_content = f.read()

    assert "gpu_relative" in csv_log_content
    assert "gpu_individual" in csv_log_content
