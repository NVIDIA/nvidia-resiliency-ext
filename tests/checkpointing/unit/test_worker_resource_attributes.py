# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import nullcontext
from unittest import mock
from urllib.parse import unquote

import pytest

from nvidia_resiliency_ext.checkpointing.async_ckpt import core


def _parse(carrier):
    return {
        segment.split("=", 1)[0]: unquote(segment.split("=", 1)[1])
        for segment in carrier.split(",")
        if "=" in segment
    }


def _caller():
    caller = core.PersistentAsyncCaller.__new__(core.PersistentAsyncCaller)
    caller.process = None
    caller.rank = 3
    caller.queue = mock.sentinel.queue
    caller.preload_q = mock.sentinel.preload_q
    caller.comp_q = mock.sentinel.comp_q
    caller.background_worker_is_daemon = True
    caller.cpu_priority = 10
    caller.io_priority = None
    caller.cpu_shm_mode = False
    return caller


def _start_worker(carrier, start_error=None, observed=None):
    observed = {} if observed is None else observed

    class FakeProcess:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            observed["process_args"] = kwargs["args"]

        def start(self):
            observed["carrier"] = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
            if start_error is not None:
                raise start_error

    fake_context = mock.Mock()
    fake_context.Process = FakeProcess
    caller = _caller()
    try:
        with (
            mock.patch.dict("os.environ", {}, clear=False),
            mock.patch.object(core.mp, "get_context", return_value=fake_context),
            mock.patch.object(core.PersistentAsyncCaller, "_worker_restart_callbacks", []),
        ):
            if carrier is None:
                os.environ.pop("OTEL_RESOURCE_ATTRIBUTES", None)
            else:
                os.environ["OTEL_RESOURCE_ATTRIBUTES"] = carrier
            try:
                caller._start_worker(3)
            finally:
                observed["restored"] = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
                observed["restored_present"] = "OTEL_RESOURCE_ATTRIBUTES" in os.environ
    finally:
        caller.process = None
    return observed


@pytest.mark.skipif(not core.telemetry._AVAILABLE, reason="nemo-lens is not installed")
@pytest.mark.parametrize("trainer_rank", ["7", ""])
def test_worker_start_uses_live_trainer_resource_and_preserves_rank(trainer_rank):
    trainer = (
        f"example.attribute=trainer-value,nv.dl.job.uuid=job-1,nv.dl.rank={trainer_rank},"
        "nv.dl.role=trainer,service.instance.id=trainer-7"
    )
    observed = _start_worker(trainer)
    worker = _parse(observed["carrier"])

    assert worker["example.attribute"] == "trainer-value"
    assert worker["nv.dl.job.uuid"] == "job-1"
    assert worker["nv.dl.rank"] == trainer_rank
    assert worker["nv.dl.role"] == "ckpt_worker"
    assert worker["service.instance.id"] == "nvrx-ckpt3"
    assert observed["process_args"][0] == 3
    assert observed["restored"] == trainer


@pytest.mark.skipif(not core.telemetry._AVAILABLE, reason="nemo-lens is not installed")
def test_worker_start_fills_missing_rank_without_trainer_carrier():
    observed = _start_worker(None)
    worker = _parse(observed["carrier"])

    assert worker["nv.dl.rank"] == "3"
    assert worker["nv.dl.role"] == "ckpt_worker"
    assert observed["restored"] is None
    assert not observed["restored_present"]


@pytest.mark.skipif(not core.telemetry._AVAILABLE, reason="nemo-lens is not installed")
def test_worker_start_restores_trainer_resource_when_process_start_fails():
    trainer = "example.attribute=trainer-value,nv.dl.rank=3,nv.dl.role=trainer"
    observed = {}
    with pytest.raises(RuntimeError, match="Process.start failed"):
        _start_worker(trainer, RuntimeError("Process.start failed"), observed)

    assert observed["restored"] == trainer


@pytest.mark.parametrize("trainer", [None, "", "nv.dl.rank=7,nv.dl.role=trainer"])
@pytest.mark.parametrize("fail", [False, True])
def test_worker_start_without_lens_leaves_environment_untouched(trainer, fail):
    observed = {}
    error = RuntimeError("Process.start failed") if fail else None
    with (
        mock.patch.object(core.telemetry, "_AVAILABLE", False),
        mock.patch.object(core.telemetry, "get_otel_resource_attributes", return_value={}),
        mock.patch.object(core.telemetry, "compose_attributes", return_value={}),
        mock.patch.object(
            core.telemetry,
            "publish_otel_resource_attributes",
            side_effect=lambda attributes: nullcontext(),
        ),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="Process.start failed"):
                _start_worker(trainer, error, observed)
        else:
            _start_worker(trainer, observed=observed)
    assert observed["carrier"] == trainer
    assert observed["restored"] == trainer
    assert observed["restored_present"] == (trainer is not None)


@pytest.mark.skipif(not core.telemetry._AVAILABLE, reason="nemo-lens is not installed")
@pytest.mark.parametrize("trainer", [None, ""])
@pytest.mark.parametrize("fail", [False, True])
def test_worker_start_restores_missing_or_empty_environment(trainer, fail):
    observed = {}
    if fail:
        with pytest.raises(RuntimeError, match="Process.start failed"):
            _start_worker(trainer, RuntimeError("Process.start failed"), observed)
    else:
        _start_worker(trainer, observed=observed)
    assert _parse(observed["carrier"])["nv.dl.rank"] == "3"
    assert observed["restored"] == trainer
    assert observed["restored_present"] == (trainer is not None)


@pytest.mark.skipif(not core.telemetry._AVAILABLE, reason="nemo-lens is not installed")
def test_exported_worker_resource_restores_types_from_spawn_carrier():
    import json
    import subprocess
    import sys

    trainer = (
        "example.attribute=trainer-value,nv.dl.job.uuid=job-1,nv.dl.rank=7,"
        "nv.dl.world_size=8,nv.dl.local_rank=1,nv.dl.topology.size.tp=2,"
        "nv.dl.training.target.train_tokens=9007199254740993,"
        "nv.dl.role=trainer,service.instance.id=trainer-7,software.version=001"
    )
    observed = _start_worker(trainer)
    code = r"""
import json
from functools import partial
from nvidia_resiliency_ext.shared_utils import telemetry
from nemo.lens import setup_telemetry
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
exporter = InMemorySpanExporter()
telemetry._setup_telemetry = partial(setup_telemetry, span_exporter=exporter)
handle = telemetry.setup_telemetry('nvrx.ckpt_worker')
with trace.get_tracer('worker-test').start_as_current_span('worker'):
    pass
trace.get_tracer_provider().force_flush()
print(json.dumps(dict(exporter.get_finished_spans()[0].resource.attributes)))
handle.shutdown()
"""
    env = dict(os.environ)
    env.update(
        OTEL_RESOURCE_ATTRIBUTES=observed["carrier"],
        NEMO_LENS_ENABLED="true",
        NEMO_LENS_METRICS_ENABLED="false",
        NEMO_LENS_LOGS_ENABLED="false",
        NEMO_LENS_TRACES_ENABLED="true",
        NEMO_LENS_GPU_PROBE="false",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    resource = json.loads(result.stdout)
    for key, expected in {
        "nv.dl.rank": 7,
        "nv.dl.world_size": 8,
        "nv.dl.local_rank": 1,
        "nv.dl.topology.size.tp": 2,
        "nv.dl.training.target.train_tokens": 9007199254740993,
    }.items():
        assert type(resource[key]) is int
        assert resource[key] == expected
    assert resource["example.attribute"] == "trainer-value"
    assert resource["nv.dl.job.uuid"] == "job-1"
    assert resource["nv.dl.role"] == "ckpt_worker"
    assert resource["service.instance.id"] == "nvrx-ckpt3"
    assert resource["service.name"] == "nvrx.ckpt_worker"
    assert resource["software.version"] == "001"
    assert observed["restored"] == trainer
