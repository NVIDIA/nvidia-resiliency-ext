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

"""
Demo of DDP training with fault tolerance, using FT package sections API

It should be run with `ft_launcher`. E.g.
`ft_launcher --nproc-per-node=2 --ft-param-cfg-path=./examples/fault_tolerance/fault_tol_cfg_sections.yaml examples/fault_tolerance/train_ddp_sections_api.py --device=cpu`

This example uses following custom FT sections
- 'init' - covers workload initialization
- 'step' - covers training/evaluation step (fwd/bwd, loss calculation etc)
- 'checkpoint' - covers checkpoint saving

Timeout for each section is calculated when enough data is collected.
FT "out-of-section" timeout is calculated when the training run ends normally.
FT state is saved in a JSON file.

This example allows to simulate a training fault:
- selected rank hung
- selected rank terminated
"""
import argparse
import json
import logging
import os
import random
import signal
import sys
import threading
import time

import dist_utils
import log_utils
import numpy as np
import torch
import torch.nn as nn

import nvidia_resiliency_ext.fault_tolerance as fault_tolerance


# Dummy dataset.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, size, hidden):
        self.size = size
        self.hidden = hidden

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = torch.full(
            (self.hidden,),
            fill_value=idx,
            dtype=torch.float32,
            device='cpu',
        )
        return data


# Dummy model
class Model(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


def parse_args():
    def fault_desc(strings):
        parts = strings.split(",")
        assert len(parts) == 2, "Fault description must be in format 'fault,delay'"
        return {'fault': parts[0], 'delay': float(parts[1])}

    parser = argparse.ArgumentParser(
        description='Example of PyTorch DDP training with the Fault Tolerance package',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    parser.add_argument('--hidden', type=int, default=4096,
                        help='Hidden size')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--train_dataset_size', type=int, default=1000000,
                        help='Train dataset size')
    parser.add_argument('--val_dataset_size', type=int, default=2000,
                        help='Validation dataset size')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device')
    parser.add_argument('--save_interval', type=int, default=-1,
                        help='Interval for saving periodic checkpoints.')
    parser.add_argument('--logging_interval', type=int, default=1,
                        help='Interval for log entries')
    parser.add_argument('--log_all_ranks', action='store_true',
                        help='Enable logging from all distributed ranks')
    parser.add_argument('--output_dir', type=str, default='results/output',
                        help='Output dir')
    parser.add_argument('--checkpoint_fname', type=str, default='checkpoint.pt',
                        help='Name of a checkpoint file')
    parser.add_argument('--local_rank', type=int,
                        default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--simulated_fault', type=fault_desc,
                        help='Description of a fault to be simulated')
    # fmt: on

    args = parser.parse_args()
    return args


def load_checkpoint(path):
    map_location = {
        'cpu': 'cpu',
    }
    if torch.cuda.is_available():
        map_location['cuda:0'] = f'cuda:{torch.cuda.current_device()}'

    logging.info(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    return checkpoint


def save_checkpoint(
    progress,
    model,
    optimizer,
    ft_client,
    output_dir,
    checkpoint_fname,
):
    # Checkpointing is wrapped into "checkpoint" FT section
    # NOTE: FT state is not stored in the checkpoint, but in a separate JSON file
    ft_client.start_section('checkpoint')

    state = {
        'progress': progress,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(output_dir, checkpoint_fname)

    with dist_utils.sync_workers() as rank:
        if rank == 0:
            logging.info(f'Saving checkpoint to {checkpoint_path}')
            torch.save(state, checkpoint_path)

    ft_client.end_section('checkpoint')


def maybe_load_ft_state(path):
    # Load FT state from JSON file
    if os.path.exists(path):
        logging.info(f'FT state loading from: {path}')
        with open(path, 'r') as f:
            return json.load(f)
    else:
        logging.info(f'FT state file not found at: {path}')
        return None


def save_ft_state(ft_client, path):
    # Save FT state into a JSON file
    with dist_utils.sync_workers() as rank:
        if rank == 0:
            logging.info(f'Saving FT state into: {path}')
            ft_state = ft_client.state_dict()
            with open(path, 'w') as f:
                json.dump(ft_state, f)


def update_ft_section_timeouts(ft_client, selected_sections, calc_out_of_section, ft_state_path):
    # Update FT timeouts and save the FT state
    logging.info(
        f'Updating FT section timeouts for: {selected_sections} will update out-of-section: {calc_out_of_section}'
    )
    ft_client.calculate_and_set_section_timeouts(
        selected_sections=selected_sections, calc_out_of_section=calc_out_of_section
    )
    save_ft_state(ft_client, ft_state_path)


def training_loop(
    ft_client,
    para_model,
    model,
    optimizer,
    device,
    dataloader,
    progress,
    args,
):
    # Training epoch implementation

    epoch_idx = progress['epoch_idx']

    para_model.train()

    last_log_time = time.monotonic()

    num_iters_made = 0

    for iter_idx, x in enumerate(dataloader, start=progress['iter_idx']):

        # fwd/bwd and optimizer step are wrapped into "step" FT section
        ft_client.start_section('step')

        optimizer.zero_grad()
        x = x.to(device)
        y = para_model(x)
        loss = y.mean()
        train_loss = loss.item()
        loss.backward()

        if iter_idx % args.logging_interval == 0:
            avg_train_loss = dist_utils.all_reduce_item(train_loss, op='mean')
            logging.info(
                f'CHECK TRAIN epoch: {epoch_idx:4d} '
                f'iter: {iter_idx:5d} '
                f'loss: {avg_train_loss} '
                f'input: {x[:, 0]}'
            )
            if iter_idx > 0:
                time_per_iter = (time.monotonic() - last_log_time) / args.logging_interval
                last_log_time = time.monotonic()
                logging.debug(f'Avg time per iter: {time_per_iter:.3f} [sec]')

        progress['iter_idx'] = iter_idx + 1

        optimizer.step()

        ft_client.end_section('step')

        # Whether to do a periodic checkpointing
        periodic_save = iter_idx % args.save_interval == args.save_interval - 1
        if periodic_save:
            save_checkpoint(
                progress=progress,
                model=model,
                optimizer=optimizer,
                ft_client=ft_client,
                output_dir=args.output_dir,
                checkpoint_fname=args.checkpoint_fname,
            )

        num_iters_made += 1

    return num_iters_made


def validation_loop(ft_client, model, val_dataloader, epoch_idx, device):

    # Validation epoch implementation

    total_val_loss = 0
    model.eval()

    for iter_idx, x in enumerate(val_dataloader):

        # fwd and loss are wrapped into "step" FT section
        # 'step' section is used for both: training and eval steps
        ft_client.start_section('step')

        x = x.to(device)
        y = model(x)
        loss = y.mean().item()
        total_val_loss += loss

        ft_client.end_section('step')

    logging.info(
        f'CHECK VAL SUMMARY: epoch: {epoch_idx:4d} ' f'loss: {total_val_loss / (iter_idx + 1)}'
    )


_sim_fault_canceled = False
_sim_fault_is_set = False


def _cancel_simulated_fault():
    global _sim_fault_canceled
    _sim_fault_canceled = True


def _setup_simulated_fault(fault_desc, device):

    global _sim_fault_is_set
    _sim_fault_is_set = True  # should be True on all ranks

    rng = random.Random()

    logging.info(f"Initializing simulated fault: {fault_desc}")

    fault_type = fault_desc['fault']
    if fault_type == 'random':
        fault_type = rng.choice(['rank_killed', 'rank_hung'])

    rank_to_fail = rng.randint(0, dist_utils.get_world_size() - 1)
    rank_to_fail = torch.tensor([rank_to_fail], device=device)
    dist_utils.broadcast(rank_to_fail, 0)
    rank_to_fail = int(rank_to_fail.item())

    rank = torch.distributed.get_rank()
    if rank != rank_to_fail:
        return

    if fault_type == 'rank_killed':
        target_pid = os.getpid()
        target_sig = signal.SIGKILL
    elif fault_type == 'rank_hung':
        target_pid = os.getpid()
        target_sig = signal.SIGSTOP
    else:
        raise Exception(f"Unknown fault type {fault_type}")

    delay = fault_desc['delay'] + 4.0 * rng.random()

    logging.info(
        f"Selected fault={fault_type}; target rank={rank_to_fail}; delay={delay}",
    )

    def __fault_thread():
        time.sleep(delay)
        if _sim_fault_canceled:
            return
        print(
            f"\n####\nSimulating fault: {fault_type}; rank to fail: {rank_to_fail}\n#####\n",
            file=sys.stderr,
        )
        os.kill(target_pid, target_sig)

    fault_sim_thread = threading.Thread(target=__fault_thread)
    fault_sim_thread.daemon = True
    fault_sim_thread.start()


_signal_received = False


def _sig_handler(*args, **kwargs):
    print("Signal received!", file=sys.stderr)
    global _signal_received
    _signal_received = True


def main():
    signal.signal(signal.SIGTERM, _sig_handler)

    args = parse_args()

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    if args.device == 'cuda':
        device = torch.device('cuda')
        torch.cuda.set_device(args.local_rank)
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise RuntimeError('Unknown device')

    os.makedirs(args.output_dir, exist_ok=True)

    dist_utils.init_distributed_with_tcp_store(device)
    rank = dist_utils.get_rank()

    if args.log_all_ranks:
        log_file_name = f'train_log_rank_{dist_utils.get_rank()}.log'
    else:
        log_file_name = 'train_log.log'
    log_file_path = os.path.join(args.output_dir, log_file_name)

    # NOTE: logging appends outputs to an existing log file if it already
    # exists. Results from a single training run (potentially with many
    # restarts from a checkpoint) are stored in a single log file.
    log_utils.setup_logging(args.log_all_ranks, filename=log_file_path, filemode='a')

    logging.info(args)
    logging.info(f"SLURM_JOB_ID={os.getenv('SLURM_JOB_ID','<none>')} RANK={rank} PID={os.getpid()}")

    # Dummy datasets
    train_dataset = Dataset(args.train_dataset_size, args.hidden)
    val_dataset = Dataset(args.val_dataset_size, args.hidden)

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        drop_last=True,
    )

    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset,
    )

    # A dummy model and an optimizer
    model = Model(args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initial value for start epoch - will be overwritten if training is resumed from a checkpoint
    progress = {
        'epoch_idx': 0,
        'iter_idx': 0,
    }

    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_fname)

    # Initialize fault tolerance.
    ft_client = fault_tolerance.RankMonitorClient()
    ft_client.init_workload_monitoring()

    # try to load FT state from a JSON file
    ft_state_path = os.path.join(args.output_dir, 'ft_state.json')
    ft_state = maybe_load_ft_state(ft_state_path)
    if ft_state:
        ft_client.load_state_dict(ft_state)

    # Open "init" FT section that covers workload initialization
    ft_client.start_section('init')

    is_checkpoint_loaded = False

    # try to load checkpoint from disk
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            logging.info(f'Checkpoint was loaded from file: {checkpoint_path}')
            is_checkpoint_loaded = True
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            progress.update(checkpoint['progress'])

    # Return with zero exit code if model is already fully trained.
    if progress['epoch_idx'] == args.epochs:
        ft_client.end_section('init')  # explicitly end "init" section, to avoid FT warning
        ft_client.shutdown_workload_monitoring()
        torch.distributed.destroy_process_group()
        logging.info('Training finished.')
        sys.exit(0)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        sampler=train_sampler,
        num_workers=4,
        persistent_workers=True,
        pin_memory=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch,
        sampler=val_sampler,
        num_workers=4,
    )

    # Regular DDP init
    # NOTE: for convenience code is keeping both wrapped and unwrapped model and
    # uses wrapped model for training and unwrapped model for saving the
    # checkpoint and validation. It doesn't increase memory consumption
    # since both models are holding references to the same parameters.
    # Additionally saved checkpoint is ready for inference and doesn't have to
    # be manually unwrapped by accessing the (undocumented) "module" attribute
    # of DDP-wrapped model.
    if device.type == 'cuda':
        device_ids = [args.local_rank]
        output_device = args.local_rank
    elif device.type == 'cpu':
        device_ids = None
        output_device = None
    else:
        raise RuntimeError('Unsupported device type')
    para_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, output_device=output_device
    )

    # "init" FT section ends here
    ft_client.end_section('init')

    if is_checkpoint_loaded:
        # init time can be longer if there was checkpoint loading
        # so we update "init" secton timeout if a checkpoint was loaded
        update_ft_section_timeouts(ft_client, ['init'], False, ft_state_path)

    # Iteration over epochs, notice that it starts from 'epoch_idx'
    # which was previously loaded from the checkpoint
    for epoch_idx in range(progress['epoch_idx'], args.epochs):

        num_tr_iters_made = training_loop(
            ft_client,
            para_model,
            model,
            optimizer,
            device,
            train_dataloader,
            progress,
            args,
        )

        # If there were some training iterations observed, update "step" section timeout
        if num_tr_iters_made > 0:
            update_ft_section_timeouts(ft_client, ['step'], False, ft_state_path)

        # epoch_idx is incremented because the current epoch is finished
        # and potential resume from this checkpoint should start a new training epoch.
        progress['epoch_idx'] += 1
        progress['iter_idx'] = 0

        validation_loop(ft_client, model, val_dataloader, epoch_idx, device)

        # Checkpoint contains everything needed for deterministic resume:
        # state of the model, optimizer and other components,
        save_checkpoint(
            progress=progress,
            model=model,
            optimizer=optimizer,
            ft_client=ft_client,
            output_dir=args.output_dir,
            checkpoint_fname=args.checkpoint_fname,
        )

        # Update checkpointing section timeout after checkpoint saving was seen
        update_ft_section_timeouts(ft_client, ['checkpoint'], False, ft_state_path)

        # NOTE: SIGTERM is used by SLURM to initiate graceful job termination
        # if _any_ rank received SIGTERM, we leave the main loop
        if dist_utils.is_true_on_any_rank(_signal_received):
            logging.info('Leaving the main loop, due to SIGTERM')
            break

        # Setup simulated fault
        if args.simulated_fault and not _sim_fault_is_set:
            _setup_simulated_fault(args.simulated_fault, device)

    _cancel_simulated_fault()

    # update "out-of-section" FT timeout when the training run ends normally
    update_ft_section_timeouts(ft_client, [], True, ft_state_path)
    ft_client.shutdown_workload_monitoring()
    torch.distributed.destroy_process_group()
    logging.info('Leaving main, ret_code=0')
    sys.exit(0)


if __name__ == "__main__":
    main()
