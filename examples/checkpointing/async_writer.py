import argparse
import logging
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import checkpoint
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner, FileSystemReader

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Async Checkpointing Example")
    parser.add_argument(
        "--ckpt_dir",
        default="/tmp/test_checkpointing/",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--thread_count",
        default=2,
        type=int,
        help="Threads to use during saving. Affects the number of files in the checkpoint (saving ranks * num_threads).",
    )
    parser.add_argument(
        '--persistent_queue',
        action='store_true',
        help="Enables a persistent version of AsyncCallsQueue.",
    )

    return parser.parse_args()


class SimpleModel(nn.Module):
    """A simple feedforward neural network for demonstration purposes."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)


def init_distributed_backend(backend="nccl"):
    """Initializes the distributed process group using the specified backend."""
    try:
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        logging.info(f"Process {rank} initialized with {backend} backend.")
    except Exception as e:
        logging.error(f"Failed to initialize distributed backend: {e}")
        raise


def get_save_and_finalize_callbacks(writer, save_state_dict_ret) -> AsyncRequest:
    """Creates an async save request with a finalize function."""
    save_fn, preload_fn, save_args = writer.get_save_function_and_args()

    def finalize_fn():
        """Finalizes async checkpointing and synchronizes processes."""
        save_state_dict_async_finalize(*save_state_dict_ret)
        dist.barrier()

    return AsyncRequest(save_fn, save_args, [finalize_fn], preload_fn=preload_fn)


def save_checkpoint(checkpoint_dir, async_queue, model, thread_count):
    """Asynchronously saves a model checkpoint."""
    state_dict = model.state_dict()
    planner = DefaultSavePlanner()
    writer = FileSystemWriterAsync(checkpoint_dir, thread_count=thread_count)
    coordinator_rank = 0

    save_state_dict_ret, *_ = save_state_dict_async_plan(
        state_dict, writer, None, coordinator_rank, planner=planner
    )
    save_request = get_save_and_finalize_callbacks(writer, save_state_dict_ret)
    async_queue.schedule_async_request(save_request)


def load_checkpoint(checkpoint_dir, model):
    """Loads a model checkpoint synchronously."""
    state_dict = model.state_dict()
    checkpoint.load(
        state_dict=state_dict,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=DefaultLoadPlanner(),
    )
    return state_dict


def main():
    args = parse_args()
    logging.info(f"Arguments: {args}")

    # Initialize distributed training
    init_distributed_backend(backend="nccl")

    # Initialize model and move to GPU
    model = SimpleModel().to("cuda")

    # Create an async queue for handling asynchronous operations
    async_queue = AsyncCallsQueue(persistent=args.persistent_queue)

    # Define checkpoint directory based on iteration number
    iteration = 123
    checkpoint_dir = f"{args.ckpt_dir}/iter_{iteration:07d}"

    # Save the model asynchronously
    save_checkpoint(checkpoint_dir, async_queue, model, args.thread_count)

    logging.info("Finalizing checkpoint save...")
    async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
    async_queue.close()  # Explicitly close queue (optional)

    # Synchronize processes before loading
    dist.barrier()

    # Load the checkpoint
    loaded_sd = load_checkpoint(checkpoint_dir, model)

    # Synchronize again to ensure all ranks have completed loading
    dist.barrier()

    # Clean up checkpoint directory (only on rank 0)
    if dist.get_rank() == 0:
        logging.info(f"Cleaning up checkpoint directory: {args.ckpt_dir}")
        shutil.rmtree(args.ckpt_dir)

    # Ensure NCCL process group is properly destroyed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
