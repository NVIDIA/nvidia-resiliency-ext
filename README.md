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

| Category            | Supported Versions / Requirements            |
|---------------------|----------------------------------------------|
| Architecture         | x86_64                                      |
| Operating System     | Ubuntu 22.04                                |
| Python Version       | >= 3.10, < 3.13                             |
| PyTorch Version      | 2.3+                                        |
| CUDA & CUDA Toolkit  | 12.5+                                       |
| NVML Driver          | 550 or later                                |
| NCCL Version         | 2.21.5+                                     |

**Note**: The package is designed to support Python >= 3.10, CUDA >= 11.8, PyTorch >= 2.0 and Ubuntu 20.04, but the recommended and tested environment for production is Python >= 3.10, < 3.13, CUDA 12.5+, and Ubuntu 22.04.

## Usage

For detailed documentation and usage information about each component, please refer to the https://nvidia.github.io/nvidia-resiliency-ext/.
