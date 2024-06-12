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
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import nemo_run as run
import torch
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import (
    Llama3Config,
    Llama3Config8B,
    Llama3Config70B,
    LlamaModel,
)
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.llm.utils import Config
from nemo.utils.exp_manager import TimingCallback

from nvidia_resiliency_ext.ptl_resiliency import StragglerDetectionCallback


@dataclass
class Llama3Config145M(Llama3Config):
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16


def straggler_det_callback(
    report_time_interval=20, stop_if_detected=False
) -> run.Config[StragglerDetectionCallback]:
    return run.Config(
        StragglerDetectionCallback,
        report_time_interval=report_time_interval,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        num_gpu_perf_scores_to_print=5,
        gpu_relative_perf_threshold=0.7,
        gpu_individual_perf_threshold=0.7,
        stop_if_detected=stop_if_detected,
        enable_ptl_logging=True,
    )


def step_timing_callback() -> run.Config[TimingCallback]:
    return run.Config(TimingCallback)


def local_executor(
    nodes=1, devices=8, ft=False, retries=0, container_image="", time=""
) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def print_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def get_parser():
    parser = argparse.ArgumentParser(description="Llama3 Pretraining on a local node")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dryrun and exit",
        default=False,
    )
    parser.add_argument(
        "--size",
        type=str,
        default="145m",  # customized config for smaller number of parameters
        help="Choose llama3 model size 70b/8b/145m",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer.model",
        help="Path to the tokenizer model file",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="Set the tensor parallelism size",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Set the pipeline parallelism size",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Specify the number of GPUs",
    )
    parser.add_argument('--max-runtime', type=int, default=900)  # in seconds
    parser.add_argument('--report-time-interval', type=int, default=30)
    parser.add_argument(
        "--test-terminate",
        action="store_true",
        help="Stop training if stragglers are detected.",
        default=False,
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for your experiment title which will be appended after the model/exp name.",
        required=False,
        default="",
    )
    return parser


def main():
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    MODEL_SIZE_MAPPING: dict[str, dict[str, Any]] = {
        "145m": {
            "exp_name": "llama3-145m",
            "local": {
                "model": Config(LlamaModel, config=Config(Llama3Config145M)),
                "trainer": partial(
                    llm.llama3_8b.trainer,
                    tensor_parallelism=args.tp_size,
                    pipeline_parallelism=args.pp_size,
                    pipeline_parallelism_type=None,
                    virtual_pipeline_parallelism=None,
                    context_parallelism=1,
                    sequence_parallelism=False,
                    num_nodes=1,
                    num_gpus_per_node=args.num_gpus,
                ),
            },
        },
        "8b": {
            "exp_name": "llama3-8b",
            "local": {
                "model": Config(LlamaModel, config=Config(Llama3Config8B)),
                "trainer": partial(
                    llm.llama3_8b.trainer,
                    tensor_parallelism=args.tp_size,
                    pipeline_parallelism=args.pp_size,
                    pipeline_parallelism_type=None,
                    virtual_pipeline_parallelism=None,
                    context_parallelism=1,
                    sequence_parallelism=False,
                    num_nodes=1,
                    num_gpus_per_node=args.num_gpus,
                ),
            },
        },
        "70b": {
            "exp_name": "llama3-70b",
            "local": {
                "model": Config(LlamaModel, config=Config(Llama3Config70B)),
                "trainer": partial(
                    llm.llama3_70b.trainer,
                    tensor_parallelism=args.tp_size,
                    pipeline_parallelism=args.pp_size,
                    pipeline_parallelism_type=None,
                    virtual_pipeline_parallelism=None,
                    context_parallelism=1,
                    sequence_parallelism=False,
                    num_nodes=1,
                    num_gpus_per_node=args.num_gpus,
                ),
            },
        },
    }

    mbs = 1
    gbs = mbs * args.num_gpus
    exp_name = MODEL_SIZE_MAPPING[args.size]["exp_name"]

    data = Config(
        MockDataModule,
        seq_length=8192,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=run.Config(SentencePieceTokenizer, model_path=args.tokenizer_path),
    )

    # Uses local configs in this repo
    llama3_model = MODEL_SIZE_MAPPING[args.size]["local"]["model"]

    llama3_trainer = MODEL_SIZE_MAPPING[args.size]["local"]["trainer"](
        callbacks=[
            straggler_det_callback(
                report_time_interval=args.report_time_interval, stop_if_detected=args.test_terminate
            ),
            step_timing_callback(),
        ],
    )
    executor = local_executor()

    pretrain = run.Partial(
        llm.train,
        model=llama3_model,
        data=data,
        trainer=llama3_trainer,
        log=llm.default_log(
            tensorboard_logger=tensorboard_logger(name=exp_name),
            name=exp_name,
        ),
        resume=llm.default_resume(resume_if_exists=False),
        optim=llm.adam.distributed_fused_adam_with_cosine_annealing(),
        tokenizer="data",
    )

    pretrain.trainer.limit_val_batches = 0
    pretrain.trainer.max_time = {"seconds": args.max_runtime}

    if not args.test_terminate:
        pretrain.log.ckpt.save_top_k = -1
        pretrain.log.ckpt.save_last = False

    # max_steps = tokens / ( sequence length * GBS )
    rp2_tokens = 840000000000
    max_steps = math.ceil(rp2_tokens / pretrain.data.seq_length / gbs)
    pretrain.broadcast(max_steps=max_steps)

    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        pretrain.log.log_dir = "./nemo_run/checkpoints"

        for i in range(1):
            exp.add(
                pretrain,
                executor=executor,
                name=exp_name,
                tail_logs=True,
            )

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run(sequential=True, detach=True)


if __name__ == "__main__":
    main()
