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

import argparse
from dataclasses import dataclass

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule

from nvidia_resiliency_ext.ptl_resiliency import FaultToleranceCallback
from nvidia_resiliency_ext.ptl_resiliency.fault_tolerance_callback import SimulatedFaultParams


@dataclass
class Llama3Config36M(Llama3Config):
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16


def get_ft_callback(args):
    simulated_fault = None
    if args.sim_fault_desc:
        fault_type, base_delay = args.sim_fault_desc.split(",")
        fault_type = fault_type.strip()
        base_delay = float(base_delay.strip())
        simulated_fault = SimulatedFaultParams(
            fault_type=fault_type,
            base_delay=base_delay,
        )
    ft_callback = FaultToleranceCallback(
        autoresume=False,
        calculate_timeouts=True,
        exp_dir=args.log_dir,
        simulated_fault_params=simulated_fault,
    )
    return ft_callback


def get_trainer(args, callbacks):
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=None,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        ckpt_async_save=True,
        ckpt_parallel_load=False,
        ddp=DistributedDataParallelConfig(),
    )
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        max_time={"seconds": args.max_runtime},
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        strategy=strategy,
    )
    return trainer


def print_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def get_parser():
    parser = argparse.ArgumentParser(description="Llama3 Pretraining on a local node")

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer.model",
        help="Path to the tokenizer model file",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="How many nodes to use",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Specify the number of GPUs per node",
    )
    parser.add_argument('--max-runtime', type=int, default=900)  # in seconds
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1_000_000,
        help="Number of steps to run the training for",
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=80,
        help="Checkpoint saving interval in steps",
    )
    parser.add_argument(
        '--val-check-interval',
        type=int,
        default=40,
        help="Validation check interval in steps",
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        default=10,
        help="How many batches to use for validation",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Output dir.",
        required=False,
        default="./nemo_llama3_fault_tol",
    )
    parser.add_argument(
        "--sim-fault-desc",
        type=str,
        help="Description of a fault to be simulated, format is: <fault_type>,<base_delay>",
        required=False,
        default="",
    )
    return parser


def main():
    args = get_parser().parse_args()

    mbs = 1
    gbs = mbs * args.num_gpus * args.num_nodes

    data = MockDataModule(
        seq_length=8192,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=SentencePieceTokenizer(model_path=args.tokenizer_path),
    )

    model = LlamaModel(config=Llama3Config36M())

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=1,
        every_n_train_steps=args.checkpoint_interval,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
        filename='{step}-{epoch}',
    )

    trainer = get_trainer(args, callbacks=[checkpoint_callback, get_ft_callback(args)])

    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        use_datetime_version=False,
        update_logger_directory=True,
        wandb=None,
        ckpt=checkpoint_callback,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-2,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        timers=None,
        bf16=True,
        use_distributed_optimizer=False,
    )
    optim = MegatronOptimizerModule(config=opt_config)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ),
        optim=optim,
        tokenizer="data",
    )


if __name__ == "__main__":
    main()
