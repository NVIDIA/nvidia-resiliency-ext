#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling fault-tolerant training
# with in-process resiliency or otherwise documented as NVIDIA-proprietary are not
# a contribution and subject to the following terms and conditions:

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Fine-tuning Stable Diffusion for text2image with LoRA and Resiliency."""
# This example demontrate how to integrate inprocess.wrapper into a multimodal model.

# ==========================================================================================================
# INSTRUCTION FOR RUNNING THE SCRIPT
# ==========================================================================================================
# ✅ Install dependencies
# pip install datasets transformers peft torchvision
# install diffusers from source:
# pip install git+https://github.com/huggingface/diffusers
# ✅ This script has been tested with:
# nvidia_resiliency_ext    0.4.0
# torch                    2.7.1
# torchvision              0.22.1
# diffusers                0.35.0.dev0
# transformers             4.53.1
# peft                     0.16.0
# pip                      22.0.2
# datasets                 3.6.0
# ✅ Example command:
# torchrun --nproc_per_node=2 sdxl_lora.py   --fault_prob 0.01   --dataset_name lambdalabs/naruto-blip-captions
# --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base   --resolution 512   --train_batch_size 2
# --use_8bit_adam   --output_dir lora-naruto   --num_train_epochs 6   --checkpointing_steps 100

import argparse
import logging
import math
import os

os.environ["TORCH_CPP_LOG_LEVEL"] = "error"
import datetime
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.distributed import PrefixStore, TCPStore
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# NVIDIA modification: Added in-process restart resiliency
import nvidia_resiliency_ext.inprocess as inprocess

local_rank = int(os.environ.get("LOCAL_RANK", 0))

device = torch.cuda.set_device(local_rank)
raise_timestamp = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)

# NVIDIA modification: Removed the log validation function


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # NVIDIA modification: Added fault prob argument
    parser.add_argument(
        "--fault_prob",
        type=float,
        default=0.0,  # 0 = disabled unless you set it
        help="Probability of raising a synthetic fault on each optimizer step.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower()
            for f in dir(transforms.InterpolationMode)
            if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


# NVIDIA modification: Creating static components for static objects that do'nt need to get reinitialized.
def build_static(args):
    # models
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    txt = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    tok = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    noise_sched = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    txt.requires_grad_(False)

    # add LoRA adapter & cast (same lines you already had)
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    return dict(
        unet=unet,
        vae=vae,
        txt=txt,
        tok=tok,  # new
        noise_scheduler=noise_sched,  # new
        opt=optimizer,
    )


# NVIDIA modification: The following two functions are for the fault injection
FAULT_RECORD_PATH = "/tmp/injected_faults.txt"


def already_faulted(step):
    if not os.path.exists(FAULT_RECORD_PATH):
        return False
    with open(FAULT_RECORD_PATH, "r") as f:
        return str(step) in f.read().splitlines()


def record_fault(step):
    with open(FAULT_RECORD_PATH, "a") as f:
        f.write(f"{step}\n")


# NVIDIA modification: Added call wrapper so the function executed (and re-executed) by nvidia_resiliency_ext.inprocess.Wrapper
def train(
    base_store,
    static,
    args,
    call_wrapper: inprocess.CallWrapper | None = None,
):
    """
    One training "round".  `inprocess.Wrapper` calls this function, and will call
    it again after a fault.  Objects in *static* survive across restarts; all
    NCCL / DDP-bound state is rebuilt inside this function.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    weight_dtype = torch.float32

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    store = PrefixStore(str(call_wrapper.iteration), base_store)
    torch.distributed.init_process_group(
        backend="nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=150),
    )

    is_main_process = rank == 0

    logging_dir = Path(args.output_dir, args.logging_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    if is_main_process and call_wrapper.iteration == 0:
        if args.report_to == "wandb" and is_wandb_available():
            wandb.init(
                project="text2image-fine-tune",
                config=vars(args),
                name="lora_run",
                dir=str(logging_dir),
            )

    unet = static["unet"]
    vae = static["vae"]
    text_encoder = static["txt"]
    tokenizer = static["tok"]
    noise_sched = static["noise_scheduler"]
    optimizer = static["opt"]

    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    weight_dtype = torch.float32

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {"train": os.path.join(args.train_data_dir, "**")}
        dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=args.cache_dir)

    column_names = dataset["train"].column_names
    image_column = args.image_column if args.image_column in column_names else column_names[0]
    caption_column = args.caption_column if args.caption_column in column_names else column_names[1]

    def tokenize_captions(batch, is_train=True):
        captions = []
        for c in batch[caption_column]:
            if isinstance(c, str):
                captions.append(c)
            elif isinstance(c, (list, tuple)):
                captions.append(random.choice(c) if is_train else c[0])
            else:
                raise ValueError("Unsupported caption format")
        inputs = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        batch["input_ids"] = inputs.input_ids
        return batch

    interpolation = getattr(
        transforms.InterpolationMode,
        args.image_interpolation_mode.upper(),
        transforms.InterpolationMode.BICUBIC,
    )
    aug = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=interpolation),
            (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess(batch):
        batch = tokenize_captions(batch)
        images = [img.convert("RGB") for img in batch[image_column]]
        batch["pixel_values"] = [aug(im) for im in images]
        return batch

    if is_main_process:
        if args.max_train_samples:
            dataset["train"] = (
                dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            )
        train_ds = dataset["train"].with_transform(preprocess)

    if not is_main_process:
        if args.max_train_samples:
            dataset["train"] = (
                dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            )
        train_ds = dataset["train"].with_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = args.max_train_steps or updates_per_epoch * args.num_train_epochs

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=total_updates * world_size,
    )

    if world_size > 1:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )
    else:
        train_sampler = None

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        total_updates = args.num_train_epochs * num_update_steps_per_epoch
    else:
        total_updates = args.max_train_steps

    # ===================NVIDIA modification: CHECKPOINT LOADING ===================
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    ckpts.sort(key=lambda x: int(x.split("-")[1]), reverse=True)  # newest first

    global_step = 0
    start_epoch = 0
    steps_completed_in_current_epoch = 0

    for ckpt in ckpts:
        step = int(ckpt.split("-")[1])
        path = os.path.join(args.output_dir, ckpt)
        try:
            if rank == 0:
                print(f"🔄  trying {ckpt} …")

            # Load model state
            model_to_load = unet.module if hasattr(unet, 'module') else unet
            model_to_load.load_state_dict(
                torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device)
            )

            # Load optimizer state
            optimizer.load_state_dict(
                torch.load(os.path.join(path, "optimizer.bin"), map_location=device)
            )

            # Load scheduler state
            lr_scheduler.load_state_dict(
                torch.load(os.path.join(path, "scheduler.bin"), map_location=device)
            )

            # Load training state
            training_state = torch.load(
                os.path.join(path, "training_state.bin"), map_location=device
            )
            global_step = training_state["global_step"]

            if rank == 0:
                print(f"✅  resumed from {ckpt}")
            break
        except Exception as e:
            if rank == 0:
                print(f"⚠️  {ckpt} corrupt ({e}); deleting")
                shutil.rmtree(path)

    # Calculate starting epoch and steps
    if global_step > 0:
        steps_per_epoch = num_update_steps_per_epoch
        start_epoch = global_step // steps_per_epoch
        steps_completed_in_current_epoch = global_step % steps_per_epoch
    else:
        start_epoch = 0
        steps_completed_in_current_epoch = 0

    # Synchronization for distributed training
    if world_size > 1:
        state_list = [global_step, start_epoch, steps_completed_in_current_epoch]
        torch.distributed.broadcast_object_list(state_list, src=0)
        if rank != 0:
            global_step, start_epoch, steps_completed_in_current_epoch = state_list
        # torch.distributed.barrier()

    # NVIDIA modification: Added a logging when starting with no checkpoint
    if global_step == 0 and rank == 0:
        print("❌ no usable checkpoints, starting fresh")

    unet.train()
    progress = tqdm(
        range(total_updates),
        disable=(rank != 0),  # Only show on main process
        initial=global_step,
        desc="Steps",
    )

    # =================== TRAINING LOOP ===================
    for epoch in range(start_epoch, args.num_train_epochs):
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader):
            # NVIDIA modified: Skip already completed steps in current epoch
            current_step = (step + 1) // args.gradient_accumulation_steps
            if epoch == start_epoch and current_step <= steps_completed_in_current_epoch:
                continue

            batch["pixel_values"] = batch["pixel_values"].to(device)
            batch["input_ids"] = batch["input_ids"].to(device)

            # ----- forward -----
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            if args.noise_offset:
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )
            timesteps = torch.randint(
                0,
                noise_sched.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            noisy_latents = noise_sched.add_noise(latents, noise, timesteps)
            enc_states = text_encoder(batch["input_ids"], return_dict=False)[0]

            model_pred = unet(noisy_latents, timesteps, enc_states, return_dict=False)[0]

            if noise_sched.config.prediction_type == "epsilon":
                target = noise
            else:  # v-prediction
                target = noise_sched.get_velocity(latents, noise, timesteps)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # ----- backward -----
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ----- logging / checkpoint -----
            if (step + 1) % args.gradient_accumulation_steps == 0:
                progress.update(1)
                global_step += 1
                train_loss += loss.detach().float()
                step_count += 1

                # ───────────────────NVIDIA modification: Saved checkpoints with caall_wrapper.atomic───────────────────
                if global_step % args.checkpointing_steps == 0 and rank == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)

                    with call_wrapper.atomic():
                        # Save model state dict (unwrap DDP if needed)
                        model_to_save = unet.module if hasattr(unet, 'module') else unet
                        torch.save(
                            model_to_save.state_dict(), os.path.join(ckpt_path, "pytorch_model.bin")
                        )

                        # Save optimizer state
                        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.bin"))

                        # Save scheduler state
                        torch.save(
                            lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.bin")
                        )

                        # Save training state
                        torch.save(
                            {
                                "global_step": global_step,
                                "epoch": epoch,
                            },
                            os.path.join(ckpt_path, "training_state.bin"),
                        )

                    # Unwrap DDP model if needed
                    model_to_save = unet.module if hasattr(unet, 'module') else unet

                    unet_lora = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model_to_save)
                    )
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=ckpt_path,
                        unet_lora_layers=unet_lora,
                        safe_serialization=True,
                    )

            # -----------------------------------NVIDIA modification: FAULT INJECTION----------------------------------
            last_ckpt_step = (global_step // args.checkpointing_steps) * args.checkpointing_steps
            if (
                args.fault_prob
                and not already_faulted(last_ckpt_step)
                and random.random() < args.fault_prob
            ):

                record_fault(last_ckpt_step)
                raise RuntimeError(
                    f"Injected fault at step {global_step} (iteration {call_wrapper.iteration})"
                )

            if global_step >= total_updates:
                break

        # Print epoch loss (inside epoch loop, after step loop)
        if step_count > 0:
            avg_loss = train_loss / step_count
            if rank == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    # =================== FINAL CHECKPOINT SAVING ===================
    if rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        model_to_save = unet.module if hasattr(unet, 'module') else unet
        torch.save(model_to_save.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))

        unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(model_to_save))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=final_dir,
            unet_lora_layers=unet_lora,
            safe_serialization=True,
        )

        logger.info(f"🎉 Final LoRA saved at: {final_dir}")


def main():
    args = parse_args()

    static = build_static(args)

    # NVIDIA modification: The TCP store
    base_store = TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]) + 2,  # avoid port clash
        world_size=int(os.environ["WORLD_SIZE"]),
        is_master=int(os.environ["RANK"]) == 0,
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )

    # NVIDIA modification: Wrapping the training function with in-process wrapper
    wrapped_train = inprocess.Wrapper(
        store_kwargs={"port": int(os.environ["MASTER_PORT"]) + 1},
        health_check=inprocess.health_check.CudaHealthCheck(),
    )(train)

    wrapped_train(base_store, static, args)


if __name__ == "__main__":
    main()
