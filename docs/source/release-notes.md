# Release Notes

NVIDIA Resiliency Extension is a Python package for framework developers and users to implement fault-tolerant features. It improves effective training time by minimizing downtime due to failures and interruptions.

## NVIDIA Resiliency Extension v0.5.0

### Highlights

- In-job restarts
    - PRs ([185](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/185), [190](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/190), [201](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/201)) improve the scalability, profiling, and performance of in-job restarts through improvements to the rendezvous operation
    - Key scaling and fault-tolerance improvements:
         - **New barrier-based rendezvous operation** introduces a substantial redesign that addresses several limitations of the previous dynamic rendezvous implementation. This provides more predictable, stable, and scalable in-job behavior
    - Faster termination path:
        - The worker termination timeout (--workers-stop-timeout) has been reduced from 30 seconds to 15 seconds, improving failure recovery latency and overall job responsiveness
    - New Flag for Infra-Aligned Rank Assignment:
        - A new flag, --ft-use-infra-group-rank, allows in-job scaling to follow the infrastructure scheduler’s rank assignment, preserving topology-aware placement decisions
    - Migration Guidance:
        - While the previous dynamic rendezvous-based implementation (v1) remains supported, users are strongly encouraged to adopt barrier-based rendezvous (v2) for improved reliability, stability, and performance

- Enhanced GPU and NVLink health checks
    - PR [145](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/145) introduces several improvements to health check module including
        - Refactored `GPUHealthCheck` to support device-specific monitoring
        - New `NVLHealthCheck` class for NVLink health validation
        - Automatic health check chaining in `Wrapper` class `ChainedGPUHealthCheck` and `ChainedNVLHealthCheck` for in-process use
        - Single GPU health check API for individual device validation and updated trace collector to use new GPU health check API

- Checkpointing
    - PRs ([108](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/108), [138](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/138), [154](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/154), [169](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/169), [170](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/170), [193](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/193), [197](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/197), [199](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/199)) improve the stability of checkpointing by deprecating the use of fork in asynchronous checkpointing, simplifying error propagation and shutdown cleanup logic
        - Introduced the option to use **Multithread File IO Instead of Multiprocess** to simplify error propagation logic, improve shutdown cleanup and enhance overall stability
        - Made persistent async checkpoint worker default (except for local checkpointing) and fixed cross-call state pollution
        - Added ability to abort async checkpoint process

- Fault attribution (new module introduced in v0.5)
    - PR [141](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/141) introduces the base attribution class which can be used to define any attribution module. This provides asynchronous combining multiple modules directly.
    - PR [172](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/172) improves error attribution by dumping NCCL traces from PyTorch for collective analysis on hang or watchdog timeout
        - It is an experimental module to identify ranks interrupting workload progress by analyzing Flight Recorder traces. It detects GPU errors, host issues, and GIL locks
        - PyT’s watchdog is currently configured to include the training process’s stack trace when generating Flight Recorder traces. However, this can lead to a deadlock if the trainer fails inside a routine that performs collectives while holding the GIL, since capturing the stack trace requires reacquiring the GIL. A new environment variable, TORCH_INCLUDE_STACK_TRACE=False (Default: True), has been added to PyTorch main to avoid this issue. This change will be included in the NGC PyT 25.11 container.

### Known Issues & Limitations

- Spare-Node Support
    - Spare nodes are not supported by either dynamic rendezvous or barrier-based rendezvous in the current release.
    - The earlier dynamic rendezvous technically supported spare nodes, but only when infra group rank assignment was not used. That mode isn't viable in real deployments because bypassing the infrastructure topology-aware rank assignment leads to degraded performance and inconsistent scaling behavior. Because of this, spare-node support isn't available in this release.
    - With barrier-based rendezvous, we've aligned fully with infra-assigned ranks to ensure correctness and performance. Spare-node support for barrier-based rendezvous is planned for a future update.
- CUDA 12 and Ubuntu 22.04 users are advised to build from source, since PyPI wheel for v0.5 defaults to CUDA 13
- In-process restart requires NCCL < v2.28.3 OR >= 2.28.9 due to a segmentation fault issue


## NVIDIA Resiliency Extension v0.4.1

### Highlights

This hotfix release includes important bug fixes, performance improvements, and minor updates to enhance stability.

- Checkpointing
    - [PR 104](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/104), [PR 106](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/106), [PR 108](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/108), [PR 111](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/111) and [PR 116](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/116) fix the asynchronous checkpointing module to switch from temporal to using the persistent worker that uses `spawn` instead of `fork`.
    - The fix in this release is working toward an intermediate milestone of deprecating the use of `fork` and instead using a `spawn` for asynchronous checkpointing. The complete transition to using `spawn` has the following dependencies on `fork` that will be eliminated in upcoming release:
        - Local checkpointing must continue to use the `fork` based asynchronous checkpointing as clarified in the usage guide.
        - File IO operations with multiprocessing can still trigger a `fork`

- In-process restart
    - [PR 103](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/103) fixes a case where extra CUDA contexts were created on local rank 0 after restart, consuming extra GPU memory on local rank 0.
    - [PR 112](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/112) fixes the workload state leaks across the restart boundary. The fix addresses a case where objects created in the wrapped function could not be garbage collected after a restart, manifesting as a memory leak.

### Known Issues & Limitations

- In a future release, we will add changes to automatically terminate the persistent process when the main process terminates.
- Until this change is implemented, job schedulers must ensure proper termination of the persistent process and its child workers for a graceful shutdown.


## NVIDIA Resiliency Extension v0.4.0

### Highlights

- Checkpointing
    - [PR 29](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/29) - Support for storing checkpoints to cloud object stores
        - Leverage cloud storage provider’s multithreaded SDK for rapid loading and saving checkpoints to object stores such as AWS S3, Azure Blob
          Storage, Google Cloud Storage and more using NVIDIA Multi-storage Client
        - Provide scalable, reliable, cheaper, single source of truth across clouds/regions
        - Provide opt-out configuration when creating FileSystemWriterAsync class instance to allow users to passthrough to the filesystem
    - [PR 36](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/36) - Critical bug fix to enable async checkpoint loading without errors

- In-process & In-job restart
    - [PR 35](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/35) - Nested restarter updates for in-process restart to align with in-job
      restart, so users have a consistent experience across in-process and in-job restarts
    - Updates to in-process nested restart functionality provided by Python Wrapper class and existing callback infrastructure with additional
      callbacks and logging

### Known Issues & Limitations
- Dependencies:
    - In-process requires Pytorch, at least [version](https://github.com/orgs/pytorch/packages/container/pytorch-nightly/398218496?tag=2.8.0.dev20250418-cuda12.6-cudnn9-devel), that includes changes in [PR 150690](https://github.com/pytorch/pytorch/pull/150690) to avoid
      deadlock in NCCL P2P communications (used in pipeline parallel)
    - In-process requires Transformer Engine including at least [PR 1715](https://github.com/NVIDIA/TransformerEngine/pull/1715) (merged) and [PR
      1812](https://github.com/NVIDIA/TransformerEngine/pull/1812) (not yet merged) to reduce cross-restart memory leaks

## NVIDIA Resiliency Extension v0.3.0

### Highlights

- Support for Blackwell GPU
- ARM based host CPU support
- In-process & In-job restart
    - Hierarchical in-process and in-job restart support
    - Warm spare support
- Health checks
    - GPU health check based on NVML
    - NIC
- Checkpointing
    - Existing capabilities that used to be part of Megatron Core is refactored to be part of NVRx. The checkpointing feature will be maintained as part of NVRx, and Megatron Core and NeMo will use the code from NVRx in the future.
    - Added support for checkpoint metadata caching to improve performance for subsequent checkpoints.

### Known Issues & Limitations

- GPU health check requires NVML driver version >= 570
- Current checkpointing implementation doesn't support persistent queue with replication

## NVIDIA Resiliency Extension v0.2.1

### Highlights

This release includes important bug fixes, performance improvements, and minor updates to enhance stability.

- **Build Fixes & Code Improvements**
    - Fixed missing #include to ensure proper compilationv in pytorch:24.12-py3 container.
    - Lazy loading of cupti_module.
    - Fixed ForkingPickler to ensure proper installation in pytorch:25.01-py3 container.


## NVIDIA Resiliency Extension v0.2.0

### Highlights

We excited to introduce many new features in NVIDIA Resiliency Extension v0.2.0.

- **In-process restart**
    - Provides a mechanism to restart the training without killing the running process via a Python function wrapper.
    - Compared to a traditional scheduler-level restart, restarting within the same process removes overheads associated with launching a new scheduler job, starting a container, initializing a new Python interpreter, loading dependencies, and creating a new CUDA context.
- **Asynchronous checkpoint**
    - Provides core utilities to make checkpointing routines run in the background.
    - It uses torch.multiprocessing to fork a temporary process to initiate asynchronous checkpointing routine.
    - Application can check this asynchronous checkpoint save in a non-blocking manner and specify a user-defined finalization step when all ranks finish their background checkpoint saving.
- **Local checkpoint**
    - Provides a mechanism to create a checkpoint in local host memory.
    - The local checkpointing mechanism is implemented via the Python LocalCheckpointManager class, which operates on a TensorAwareStateDict wrapper.
    - This wrapper encapsulates the operations necessary for efficient replication and data transfers.

### Known Issues & Limitations

- **For in-process restart**
    - If there is hang, presence of SHARP raises an exception, which leads to triggering in-job restart and in-process restart.
    - Customer needs to disable SHARP for using in-process restart with current version.
    - Requires ENV VARs to be set as follows:
        - NCCL_NVLS_ENABLE=0 to disable SHARP.
        - NCCL_NET_PLUGIN="none" if NCCL version < 2.24.1 to avoid duplicate NCCL net plugin init.
- **In-process and in-job restart**
    - These work with PyTorch version 24.07, 24.08, 24.09, and 24.10 but not with 24.11 due to a known NCCL issue.


## NVIDIA Resiliency Extension v0.1.3

### Highlights

We are excited to announce the first release of NVIDIA Resiliency Extension v0.1.3!

- **Straggler Detection API**
    - Provides tools for user to mark the section of code and configure the threshold to detect slow running GPU.
- **Fault Tolerance API**
    - Provides the rank monitor server and client, and modified torchrun launcher based on TorchElastic to automatically detect hang and ability to in-job restart the training.
