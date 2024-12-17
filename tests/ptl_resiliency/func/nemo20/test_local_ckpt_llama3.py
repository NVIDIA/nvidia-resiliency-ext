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
This module exemplifies NeMo 2.0 integration with NVRx local checkpointing.
The key parts is the implementation of MCore specific HierarchicalCheckpointIO
(MCoreHierarchicalCheckpointIO) which can be plugged into PTL strategy.
"""

import argparse
import logging
import socket

from dataclasses import dataclass
from typing import Iterable, Dict, Any, Optional, Callable

import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.callbacks.dist_ckpt_io import (
    AsyncFinalizableCheckpointIO,
    AsyncCompatibleCheckpointIO,
)

from nvidia_resiliency_ext.checkpointing.local.base_state_dict import TensorAwareStateDict
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.base_manager import (
    BaseCheckpointManager,
)
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)
from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
    LazyCliqueReplicationStrategy,
)
from nvidia_resiliency_ext.fault_tolerance.dict_utils import dict_list_map_inplace
from nvidia_resiliency_ext.ptl_resiliency.local_checkpoint_callback import (
    LocalCheckpointCallback,
    HierarchicalCheckpointIO,
)

logger = logging.getLogger(__name__)


class MCoreHierarchicalCheckpointIO(HierarchicalCheckpointIO, AsyncCompatibleCheckpointIO):
    """HierarchicalCheckpointIO implementation compatible with MCore distributed checkpointing.

    Args:
        wrapped_checkpoint_io (CheckpointIO): previously used checkpoint_io (for global checkpoints).
        local_ckpt_manager (BaseCheckpointManager): local checkpoint manager used to store the local checkpoints
        get_global_ckpt_iteration_fn (Callable[[_PATH], int]): a function that retrieves the iteration of a global checkpoint
            that will be compared with local checkpoint iteration in order to decide which to resume from.
        local_ckpt_algo (str, optional): local checkpoint save algorithm. See MCoreTensorAwareStateDict for details.
            By default, uses a fully parallel save and load algorithm ('fully_parallel`).
        parallelization_group (ProcessGroup, optional): save/load parallelization group
        allow_cache (bool, optional): if True, subsequent checkpoint saves will reuse
            the cached parallelization metadata.
    """

    def __init__(
        self,
        wrapped_checkpoint_io: CheckpointIO,
        local_ckpt_manager: BaseCheckpointManager,
        get_global_ckpt_iteration_fn: Callable[[_PATH], int],
        local_ckpt_algo: str = 'fully_parallel',
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        allow_cache: bool = False,
    ):
        super().__init__(wrapped_checkpoint_io, local_ckpt_manager, get_global_ckpt_iteration_fn)
        self.local_ckpt_algo = local_ckpt_algo
        self.parallelization_group = parallelization_group
        self.cached_metadata = None
        self.allow_cache = allow_cache

    def to_tensor_aware_state_dict(self, checkpoint: Dict[str, Any]) -> TensorAwareStateDict:
        """Specialized implementation using MCoreTensorAwareStateDict.

        Wraps the state dict in MCoreTensorAwareStateDict and makes sure
        that "common" state dict doesn't have any CUDA tensors (this is an
        NVRx v0.2 limitation).
        """
        state_dict_for_save, cached_metadata = MCoreTensorAwareStateDict.from_state_dict(
            checkpoint,
            algo=self.local_ckpt_algo,
            parallelization_group=self.parallelization_group,
            cached_metadata=self.cached_metadata,
        )

        def to_cpu(x):
            if isinstance(x, torch.Tensor) and x.device.type != 'cpu':
                logger.debug(f'Moving CUDA tensor to CPU')
                x = x.to('cpu', non_blocking=True)
            return x

        dict_list_map_inplace(to_cpu, state_dict_for_save.common)
        if self.allow_cache:
            self.cached_metadata = None
        return state_dict_for_save

    def from_tensor_aware_state_dict(
        self, tensor_aware_checkpoint: TensorAwareStateDict, sharded_state_dict=None
    ):
        """Unwraps MCoreTensorAwareStateDict to a plain state dict."""
        assert isinstance(
            tensor_aware_checkpoint, MCoreTensorAwareStateDict
        ), f'Unexpected tensor aware state dict type: {type(tensor_aware_checkpoint)}'
        return tensor_aware_checkpoint.to_state_dict(
            sharded_state_dict,
            algo=self.local_ckpt_algo,
            parallelization_group=self.parallelization_group,
        )


@dataclass
class Llama3Config36M(Llama3Config):
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16


def get_trainer(args, callbacks, async_save=True):
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=None,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        ckpt_async_save=async_save,
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
        '--local-checkpoint-interval',
        type=int,
        default=None,
        help="Local checkpoint saving interval in steps",
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
    parser.add_argument(
        "--async-save",
        action="store_true",
        help="Async ckpt save",
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
    callbacks = [checkpoint_callback]

    use_local_ckpt = args.local_checkpoint_interval is not None
    if use_local_ckpt:
        callbacks.append(
            LocalCheckpointCallback(
                every_n_train_steps=args.local_checkpoint_interval,
                async_save=args.async_save,
            )
        )

    trainer = get_trainer(args, callbacks=callbacks, async_save=args.async_save)

    if use_local_ckpt:
        checkpoint_io = trainer.strategy.checkpoint_io

        if args.async_save:
            assert isinstance(trainer.strategy.checkpoint_io, AsyncFinalizableCheckpointIO), type(
                trainer.strategy.checkpoint_io
            )
            checkpoint_io = checkpoint_io.checkpoint_io

        if args.num_nodes > 1:
            repl_strategy = LazyCliqueReplicationStrategy()
        else:
            print_rank0('Single node run - replication wil be disabled.')
            repl_strategy = None

        local_ckpt_manager = LocalCheckpointManager(
            f'{args.log_dir}/local_ckpt/{socket.gethostname()}',
            repl_strategy=repl_strategy,
        )
        # NOTE: in this example we always assume that local ckpt is newer than a global checkpoint
        # by passing `lambda s: 0` global checkpoint iteration retrieval function.
        # In practice global iteration should be extracted from global ckpt path.
        hierarchical_checkpointing_io = MCoreHierarchicalCheckpointIO(
            checkpoint_io, local_ckpt_manager, lambda s: 0
        )

        if args.async_save:
            hierarchical_checkpointing_io = AsyncFinalizableCheckpointIO(
                hierarchical_checkpointing_io
            )

        trainer.strategy.checkpoint_io = hierarchical_checkpointing_io

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
