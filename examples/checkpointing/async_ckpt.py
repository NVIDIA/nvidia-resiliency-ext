import argparse
import logging
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn

from nvidia_resiliency_ext.checkpointing.async_ckpt.torch_ckpt import TorchAsyncCheckpoint

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Checkpointing Basic Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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


def main():
    args = parse_args()
    logging.info(f"{args}")

    # Initialize the distributed backend
    init_distributed_backend(backend="nccl")

    # Instantiate the model and move to CUDA
    model = SimpleModel().to("cuda")

    # Define checkpoint directory and manager
    ckpt_dir = "/tmp/test_local_checkpointing/ckpt.pt"

    ckpt_impl = TorchAsyncCheckpoint()

    ckpt_impl.async_save(model.state_dict(), ckpt_dir + "ckpt.pt")

    finalize_async_save(blocking=True, no_dist=True)

    torch.load(ckpt_dir, ckpt_dir)

    # Synchronize processes to ensure all have completed the loading
    dist.barrier()

    # Clean up checkpoint directory only on rank 0
    if dist.get_rank() == 0:
        logging.info(f"Cleaning up checkpoint directory: {ckpt_dir}")
        shutil.rmtree(ckpt_dir)


if __name__ == "__main__":
    main()
