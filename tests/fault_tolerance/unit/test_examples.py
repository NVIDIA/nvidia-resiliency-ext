# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery contracts of the documented examples, without GPU/Megatron dependencies."""

import importlib
import itertools
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[3] / 'examples' / 'fault_tolerance'


@pytest.fixture
def ddp_example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module('train_ddp_sections_api')


class Monitor:
    def start_section(self, name):
        pass

    def end_section(self, name):
        pass


def test_ddp_mid_epoch_resume_matches_uninterrupted_training(ddp_example, tmp_path):
    """A saved optimizer step must also restore the next sample, including epoch shuffle."""
    args = SimpleNamespace(batch=2, logging_interval=100, save_interval=-1)
    dataset = torch.arange(1, 25, dtype=torch.float32).reshape(-1, 1)

    def setup():
        torch.manual_seed(5)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sampler = ddp_example.dist_utils.ResumableDistributedSampler(
            dataset, num_replicas=1, rank=0
        )
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch)
        return model, optimizer, sampler, loader

    seen = []

    def train(model, optimizer, sampler, loader, progress):
        hook = model.register_forward_pre_hook(
            lambda module, inputs: seen.extend(inputs[0].tolist())
        )
        ddp_example.training_loop(
            Monitor(), model, model, optimizer, 'cpu', loader, sampler, progress, args
        )
        hook.remove()

    model, optimizer, sampler, loader = setup()
    train(model, optimizer, sampler, loader, {'epoch_idx': 3, 'iter_idx': 0})
    expected_model = model.state_dict()
    expected_optimizer = optimizer.state_dict()
    expected_samples = seen[:]
    seen.clear()

    model, optimizer, sampler, loader = setup()
    progress = {'epoch_idx': 3, 'iter_idx': 0}
    train(model, optimizer, sampler, itertools.islice(loader, 3), progress)
    ddp_example.save_checkpoint(progress, model, optimizer, Monitor(), tmp_path, 'checkpoint.pt')

    model, optimizer, sampler, loader = setup()
    state = ddp_example.load_checkpoint(tmp_path / 'checkpoint.pt')
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    train(model, optimizer, sampler, loader, state['progress'])

    assert seen == expected_samples
    assert state['progress'] == {'epoch_idx': 3, 'iter_idx': 12}
    torch.testing.assert_close(model.state_dict(), expected_model, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)


def test_interrupted_ddp_save_preserves_completed_checkpoint(ddp_example, tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    ddp_example.save_checkpoint(
        {'epoch_idx': 0, 'iter_idx': 1}, model, optimizer, Monitor(), tmp_path, 'checkpoint.pt'
    )

    def interrupted_save(state, path):
        Path(path).write_bytes(b'incomplete checkpoint')
        raise OSError('simulated interrupted write')

    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='interrupted write'):
        ddp_example.save_checkpoint(
            {'epoch_idx': 0, 'iter_idx': 2}, model, optimizer, Monitor(), tmp_path, 'checkpoint.pt'
        )
    assert ddp_example.load_checkpoint(tmp_path / 'checkpoint.pt')['progress']['iter_idx'] == 1


@pytest.fixture
def gpt_worker(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module('megatron_gpt_worker')


@pytest.mark.parametrize(
    'fault,expected_signal', [('kill', signal.SIGKILL), ('hang', signal.SIGSTOP)]
)
@pytest.mark.parametrize('fresh_run', [False, True])
def test_gpt_fault_waits_for_new_checkpoint_and_fires_once(
    gpt_worker, tmp_path, monkeypatch, fault, expected_signal, fresh_run
):
    iteration = tmp_path / 'iteration'
    marker = tmp_path / 'fault.json'
    if not fresh_run:
        iteration.write_text('10')
    baseline = gpt_worker.read_iteration(iteration, startup=True)
    assert baseline == (0 if fresh_run else 10)
    stop = threading.Event()
    baseline_read = threading.Event()
    injected = threading.Event()
    real_read = gpt_worker.read_iteration
    signals = []

    def read(path):
        value = real_read(path)
        baseline_read.set()
        return value

    def kill(pid, sig):
        signals.append((pid, sig))
        injected.set()

    monkeypatch.setattr(gpt_worker, 'read_iteration', read)
    monkeypatch.setattr(os, 'kill', kill)
    monkeypatch.setenv('RANK', '1')
    thread = threading.Thread(
        target=gpt_worker.inject_after_checkpoint, args=(iteration, marker, fault, stop, baseline)
    )
    thread.start()
    try:
        assert baseline_read.wait(2)
        assert not injected.wait(0.15)  # Existing checkpoint alone must not trigger injection.
        iteration.write_text('')  # A transient partial marker is not a completed save.
        assert not injected.wait(0.15)
        iteration.write_text('15')
        assert injected.wait(2)
    finally:
        stop.set()
        thread.join(2)
    assert signals == [(os.getpid(), expected_signal)]
    assert json.loads(marker.read_text())['checkpoint'] == 15
    # A replacement incarnation can see newer saves, but must never inject again.
    iteration.write_text('20')
    gpt_worker.inject_after_checkpoint(iteration, marker, fault, threading.Event(), 15)
    assert len(signals) == 1


@pytest.mark.parametrize('startup_problem', ['empty', 'invalid', 'permission', 'io'])
def test_gpt_wrapper_rejects_unknown_startup_checkpoint(
    gpt_worker, tmp_path, monkeypatch, startup_problem
):
    """An unreadable existing save must not be mistaken for progress from iteration zero."""
    iteration = tmp_path / 'latest_checkpointed_iteration.txt'
    iteration.write_text({'empty': '', 'invalid': 'not-an-iteration'}.get(startup_problem, '10'))
    real_read = Path.read_text
    first_read = True
    training_started = []
    signals = []

    def read(path, *args, **kwargs):
        nonlocal first_read
        if path == iteration and first_read:
            first_read = False
            if startup_problem == 'permission':
                raise PermissionError('temporarily inaccessible tracker')
            if startup_problem == 'io':
                raise OSError('transient tracker read failure')
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(
        gpt_worker.runpy, 'run_path', lambda *a, **kw: training_started.append(True)
    )
    monkeypatch.setattr(os, 'kill', lambda *args: signals.append(args))
    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'megatron_gpt_worker.py',
            '--megatron-path',
            str(tmp_path),
            '--checkpoint-dir',
            str(tmp_path),
            '--fault-marker',
            str(tmp_path / 'fault.json'),
        ],
    )
    with pytest.raises(RuntimeError, match='Cannot establish startup checkpoint iteration'):
        gpt_worker.main()
    assert not training_started
    assert not signals
    assert not (tmp_path / 'fault.json').exists()
    if startup_problem in ('permission', 'io'):
        # The following successful read of the unchanged checkpoint must be harmless.
        assert gpt_worker.read_iteration(iteration) == 10


@pytest.mark.parametrize('fault', ['none', 'kill', 'hang'])
@pytest.mark.parametrize('fault_rank', [1, -1])
def test_gpt_single_gpu_fault_rank_only_matters_for_injection(tmp_path, fault, fault_rank):
    """A baseline reaches training with the default rank; invalid injection ranks do not."""
    training_started = tmp_path / 'training_started'
    (tmp_path / 'pretrain_gpt.py').write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('training_started').touch()\n"
        'raise SystemExit(0)\n'
    )
    command = [
        sys.executable,
        str(EXAMPLES / 'megatron_gpt_worker.py'),
        '--megatron-path',
        str(tmp_path),
        '--checkpoint-dir',
        str(tmp_path),
        '--fault-marker',
        str(tmp_path / 'fault.json'),
        '--fault',
        fault,
    ]
    if fault_rank != 1:
        command.extend(['--fault-rank', str(fault_rank)])
    result = subprocess.run(
        command,
        env=dict(os.environ, RANK='0', WORLD_SIZE='1'),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if fault == 'none':
        assert result.returncode == 0, result.stderr
        assert training_started.exists()
    else:
        assert result.returncode == 2
        assert '--fault-rank must name an active training rank' in result.stderr
        assert not training_started.exists()
    assert not (tmp_path / 'fault.json').exists()


def test_gpt_wrapper_rejects_silent_uninterrupted_demo(tmp_path):
    """Finishing without a completed save/injection must not masquerade as recovery."""
    (tmp_path / 'pretrain_gpt.py').write_text('raise SystemExit(0)\n')
    env = dict(os.environ, RANK='1', WORLD_SIZE='2')
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / 'megatron_gpt_worker.py'),
            '--megatron-path',
            str(tmp_path),
            '--checkpoint-dir',
            str(tmp_path),
            '--fault-marker',
            str(tmp_path / 'fault.json'),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert 'Requested demo fault was not injected' in result.stderr


@pytest.mark.parametrize('nnodes,gpus,tp,pp', [(1, 2, 1, 1), (2, 4, 2, 2)])
def test_gpt_command_keeps_monitoring_and_async_checkpointing(tmp_path, nnodes, gpus, tp, pp):
    (tmp_path / 'pretrain_gpt.py').touch()
    env = dict(
        os.environ,
        MEGATRON_PATH=str(tmp_path),
        NVRX_WORK_DIR=str(tmp_path / 'work with spaces'),
        NVRX_DRY_RUN='1',
        NVRX_RUN_ID='recovery-test',
        RDZV_ENDPOINT='control-host:29400',
        NNODES=str(nnodes),
        GPUS_PER_NODE=str(gpus),
        TP=str(tp),
        PP=str(pp),
    )
    result = subprocess.run(
        ['bash', str(EXAMPLES / 'run_megatron_gpt.sh')],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    command = shlex.split(result.stdout.splitlines()[-1])
    assert '--async-save' in command
    assert '--use-persistent-ckpt-worker' in command
    assert '--ft-num-warmup-iters=0' in command
    assert '--max-restarts=3' in command
    assert '--calc-ft-timeouts' not in command
    assert f'--nnodes={nnodes}' in command
    assert f'--global-batch-size={nnodes * gpus}' in command


def test_gpt_launch_rejects_reusing_a_completed_injection(tmp_path):
    (tmp_path / 'pretrain_gpt.py').touch()
    run_dir = tmp_path / 'runs' / 'old-run'
    run_dir.mkdir(parents=True)
    (run_dir / 'fault_injected.json').write_text('{}')
    result = subprocess.run(
        ['bash', str(EXAMPLES / 'run_megatron_gpt.sh')],
        env=dict(
            os.environ,
            MEGATRON_PATH=str(tmp_path),
            NVRX_WORK_DIR=str(tmp_path),
            NVRX_RUN_ID='old-run',
            NVRX_DRY_RUN='1',
            NNODES='1',
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert 'use a new NVRX_RUN_ID' in result.stderr
