# NVIDIA Resiliency Extension

The NVIDIA Resiliency Extension (NVRx) integrates multiple resiliency-focused solutions for PyTorch-based workloads. Users can modularly integrate NVRx capabilities into their own infrastructure to maximize AI training productivity at scale. NVRx maximizes goodput by enabling system-wide health checks, quickly detecting faults at runtime and resuming training automatically. NVRx minimizes loss of work by enabling fast and frequent checkpointing. 

For detailed documentation and usage information about each component, please refer to https://nvidia.github.io/nvidia-resiliency-ext/.

> ⚠️ NOTE: This project is still experimental and under active development. The code, features, and documentation are evolving rapidly. Please expect frequent updates and breaking changes. Contributions are welcome and we encourage you to watch for updates.

## Core Components and Capabilities

- **[Fault Tolerance](https://nvidia.github.io/nvidia-resiliency-ext/fault_tolerance/index.html)**
  - Section-based hang detection and coordinated worker restarts within a running job.
  - [Megatron GPT recovery walkthrough](docs/source/fault_tolerance/examples/megatron_gpt.rst) with fixed CLI timeouts and NVRx-backed async checkpoints.
  - [Small DDP example](docs/source/fault_tolerance/examples/train_ddp_sections.rst) with section monitoring and synchronous PyTorch checkpoints.

- **[Async Checkpointing](https://nvidia.github.io/nvidia-resiliency-ext/checkpointing/async/index.html)**
  - Providing an efficient framework for asynchronous checkpointing.

- **[Local Checkpointing](https://nvidia.github.io/nvidia-resiliency-ext/checkpointing/local/index.html)**
  - Providing an efficient framework for local checkpointing.

- **[Straggler Detection](https://nvidia.github.io/nvidia-resiliency-ext/straggler_det/index.html)**
  - Monitoring GPU and CPU performance of ranks.  
  - Identifying slower ranks that may impede overall training efficiency.

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
| Python Version       | Release wheels/CI: 3.12, 3.14                                             |
| PyTorch Version      | >= 2.8.0                                                                   |
| CUDA & CUDA Toolkit  | >= 12.8                                                                    |
| NVML Driver          | >= 535 (570 required for GPU health check)                                 |
| TE Version           | >= 2.5                                                                     |
