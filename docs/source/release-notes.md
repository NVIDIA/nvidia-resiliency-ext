# Release Notes

NVIDIA Resiliency Extension is a Python package for framework developers and users to implement fault-tolerant features. It improves effective training time by minimizing downtime due to failures and interruptions.

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
    - Added support for checkpoint metadata caching to improve performance for subsequent checkpoints

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
