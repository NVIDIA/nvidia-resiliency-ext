import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt import TorchAsyncCheckpoint

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Async Checkpointing Basic Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--ckpt_dir',
        default="/tmp/test_async_ckpt/",
        help="Checkpoint directory for async checkpoints",
    )
    parser.add_argument(
        '--persistent_queue',
        action='store_true',
        help="Enables a persistent version of AsyncCallsQueue.",
    )
    return parser.parse_args()


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Linear layer: input size 10, output size 5
        self.fc2 = nn.Linear(5, 2)  # Linear layer: input size 5, output size 2
        self.activation = nn.ReLU()  # Activation function: ReLU

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def init_distributed_backend(backend="nccl"):
    """
    Initialize the distributed process group for NCCL backend.
    Assumes the environment variables (CUDA_VISIBLE_DEVICES, etc.) are already set.
    """
    try:
        dist.init_process_group(
            backend=backend,  # Use NCCL backend
            init_method="env://",  # Use environment variables for initialization
        )
        logging.info(f"Rank {dist.get_rank()} initialized with {backend} backend.")

        # Ensure each process uses a different GPU
        torch.cuda.set_device(dist.get_rank())
    except Exception as e:
        logging.error(f"Error initializing the distributed backend: {e}")
        raise


def cleanup(ckpt_dir):
    if dist.get_rank() == 0:
        logging.info(f"Cleaning up checkpoint directory: {ckpt_dir}")
        for file_item in os.scandir(ckpt_dir):
            if file_item.is_file():
                os.remove(file_item.path)


def main():
    args = parse_args()
    logging.info(f'{args}')

    # Initialize the distributed backend
    init_distributed_backend(backend="nccl")

    # Instantiate the model and move to CUDA
    model = SimpleModel().to("cuda")
    org_sd = model.state_dict()
    # Define checkpoint directory and manager
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    logging.info(f"Created checkpoint directory: {ckpt_dir}")
    ckpt_file_name = os.path.join(ckpt_dir, f"ckpt_rank{torch.distributed.get_rank()}.pt")

    ckpt_impl = TorchAsyncCheckpoint(persistent_queue=args.persistent_queue)

    ckpt_impl.async_save(org_sd, ckpt_file_name)

    ckpt_impl.finalize_async_save(blocking=True, no_dist=True, terminate=True)

    loaded_sd = torch.load(ckpt_file_name, map_location="cuda")

    for k in loaded_sd.keys():
        assert torch.equal(loaded_sd[k], org_sd[k]), f"loaded_sd[{k}] != org_sd[{k}]"

    # Synchronize processes to ensure all have completed the loading
    dist.barrier()

    # Clean up checkpoint directory only on rank 0
    cleanup(ckpt_dir)

    # Ensure NCCL process group is properly destroyed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
