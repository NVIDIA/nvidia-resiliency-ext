# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import logging
import os
import sys
from datetime import timedelta


import nvidia_resiliency_ext.inprocess as inprocess
import nvidia_resiliency_ext.inprocess.tools as tools

from . import app, common


def launch(fn, **kwargs):
    rank = int(os.environ["RANK"])

    if level := os.getenv("LOG", None):
        loglevel = getattr(logging, level.upper())
    else:
        loglevel = logging.CRITICAL + 1

    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"/tmp/example_{rank}.log",
        filemode="w",
        force=True,
    )
    rank_filter = app.RankFilter(rank, "***", False)
    console = logging.StreamHandler(sys.stderr)
    format = "%(asctime)s | %(levelname)s | %(name)s | %(rank)-3s | %(message)s"
    formatter = app.AdaptiveFormatter(format)
    console.setFormatter(formatter)
    console.addFilter(rank_filter)
    console.setLevel(loglevel)
    logging.getLogger().addHandler(console)

    namespace = argparse.Namespace(**kwargs)
    fn(namespace=namespace)


class TestInternal(common.MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_processes()
        self.skip_return_code_checks = [
            self.test_wo_exitcode_fault0.__wrapped__,
            self.test_wo_exitcode_fault1.__wrapped__,
        ]

    @staticmethod
    def wrapper_kwargs():
        return {
            "store_kwargs": {"port": int(os.environ["MASTER_PORT"]) + 1},
            "progress_watchdog_interval": timedelta(milliseconds=50),
            "monitor_thread_interval": timedelta(milliseconds=50),
            "monitor_process_interval": timedelta(milliseconds=50),
            "last_call_wait": timedelta(milliseconds=50),
        }

    @staticmethod
    def train_kwargs():
        return {
            "last_iteration": 4,
            "train_stall": 0.01,
            "train_sleep": 0.01,
        }

    @common.parametrize(
        "fault",
        [
            (app.Fault.EXC,),
            (app.Fault.EXIT,),
        ],
    )
    def test_w_exitcode(self, fault):
        wrapped = inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=5),
            **self.wrapper_kwargs(),
        )(app.train)
        launch(wrapped, fault=fault, **self.train_kwargs())

    @common.parametrize(
        "fault",
        [
            (app.Fault.KILL,),
            (app.Fault.TERM,),
        ],
    )
    def test_wo_exitcode(self, fault):
        wrapped = inprocess.Wrapper(
            initialize=inprocess.initialize.RetryController(max_iterations=5),
            **self.wrapper_kwargs(),
        )(app.train)
        launch(wrapped, fault=fault, **self.train_kwargs())


class TestExternal(common.MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_processes()
        self.skip_all_return_code_checks = True

    @staticmethod
    def wrapper_kwargs():
        return {
            "store_kwargs": {"port": int(os.environ["MASTER_PORT"]) + 1},
            "progress_watchdog_interval": timedelta(milliseconds=50),
            "monitor_thread_interval": timedelta(milliseconds=50),
            "monitor_process_interval": timedelta(milliseconds=50),
            "last_call_wait": timedelta(milliseconds=50),
        }

    @staticmethod
    def train_kwargs():
        return {
            "train_stall": 0.001,
            "train_sleep": 0.001,
            "ext_min_delay": 0.1,
            "ext_max_delay": 0.2,
        }

    @common.parametrize(
        "ext_fault",
        [
            (tools.inject_fault.Fault.KILL,),
            (tools.inject_fault.Fault.TERM,),
        ],
    )
    def test_wo_exitcode(self, ext_fault):
        wrapped = inprocess.Wrapper(**self.wrapper_kwargs())(app.train)
        launch(wrapped, ext_fault=ext_fault, **self.train_kwargs())


common.instantiate_parametrized_tests(TestInternal)
common.instantiate_parametrized_tests(TestExternal)
