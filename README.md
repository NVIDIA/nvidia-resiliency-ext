# NVIDIA Resiliency Extension

The NVIDIA Resiliency Extension (NVRx) integrates multiple resiliency-focused solutions for PyTorch-based workloads.

## Core Components and Capabilities

- **Fault Tolerance**
  - Detection of hung ranks.  
  - Restarting training in-job, without the need to reallocate SLURM nodes.

- **In-Process Restarting**
  - Detecting failures and enabling quick recovery.

- **Async Checkpointing**
  - Providing an efficient framework for asynchronous checkpointing.

- **Local Checkpointing**
  - Providing an efficient framework for local checkpointing.

- **Straggler Detection**
  - Monitoring GPU and CPU performance of ranks.  
  - Identifying slower ranks that may impede overall training efficiency.

- **PyTorch Lightning Callbacks**
  - Facilitating seamless NVRx integration with PyTorch Lightning.

## Installation

### From sources
- `git clone https://github.com/NVIDIA/nvidia-resiliency-ext`
- `cd nvidia-resiliency-ext`
- `pip install .`


### From PyPI wheel
- `pip install nvidia-resiliency-ext`

### Platform Support

| Category             | Supported Versions / Requirements                                          |
|----------------------|----------------------------------------------------------------------------|
| Architecture         | x86_64, arm64                                                              |
| Operating System     | Ubuntu 22.04, 24.04                                                        |
| Python Version       | >= 3.10, < 3.13                                                            |
| PyTorch Version      | >= 2.3.1 (injob & chkpt), 2.5.1 & 2.6.0 (inprocess)                        |
| CUDA & CUDA Toolkit  | >= 12.5 (12.8 required for GPU health check)                               |
| NVML Driver          | >= 535 (570 required for GPU health check)                                 |
| NCCL Version         | >= 2.21.5 (injob & chkpt), >= 2.21.5 and <= 2.22.3 or 2.26.2 (inprocess)   |

## Usage

For detailed documentation and usage information about each component, please refer to the https://nvidia.github.io/nvidia-resiliency-ext/.
