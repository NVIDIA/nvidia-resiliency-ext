# Release Notes

NVIDIA Resiliency Extension is a Python package for framework developers and users to implement fault-tolerant features. It improves effective training time by minimizing downtime due to failures and interruptions.

## NVIDIA Resiliency Extension v0.6.0

### Highlights

- **In-job restart**
    - **Barrier-based rendezvous (v2) is now the default** ([#214](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/214)). The legacy dynamic rendezvous (v1) is **deprecated** and will be removed in a future release ([#282](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/282)).
    - Rendezvous protocol hardening — round-scoped keys, round-fenced CAS to prevent stale slot writes, and cleaner handling of participants exiting mid-rendezvous ([#262](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/262), [#263](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/263), [#300](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/300)).
    - Robust startup and shutdown — wait for TCPStore on initial connection ([#264](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/264)), handle signals during rendezvous ([#246](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/246)), notify peers to abort current workers on failure ([#228](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/228)), fix `terminate_mp_processes` to cover failed workers ([#270](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/270)).
    - **Hot-spare node support** — closes the v0.5 spare-node gap. Hot-spare is always-on and works with `--max-restarts` ([#226](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/226), [#250](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/250), [#266](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/266)):
        - **Simple mode** (default, `--ft-segment=None`) for H100 / non-NVSwitch systems — first `min_nodes` from `--nnodes=min:max` become active, the rest become standbys with reserved ranks. No GPU ClusterUUID required.
        - **Segment-aware mode** (`--ft-segment=N`) for NVSwitch systems (DGX H200, HGX B200) — uses GPU ClusterUUID to identify NVLink domains; nodes in the same segment get contiguous group ranks for NVLink locality. Requires `min_nodes % segment == 0`.
        - Block-aware rank assignment ([#250](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/250)) and hot-spare exit-handling fix ([#266](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/266)).
    - **Progress-based early termination** for in-job restarts and progress-tracker enhancements ([#218](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/218), [#255](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/255)).
    - **External InJob control-plane (experimental)** — embed `ft_launcher` orchestration in a host control plane ([#321](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/321)). Not yet QA-validated; APIs may change.
    - Section-timeout fixes — out-of-section timeout now fires for section-less workloads, baseline iteration tracking corrected ([#261](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/261), [#299](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/299)).
    - `--max-restarts` now reflects job-level restart attempts ([#211](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/211)); `ft_launcher` runs with sensible defaults out of the box ([#205](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/205), [#271](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/271)).
    - **NUMA binding** support in `ft_launcher` for optimized memory affinity ([#209](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/209)).

- **Health checks**
    - **NIC link-state health check** ([#230](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/230)).
    - **Distributed Storage health check** ([#239](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/239)).
    - **DCA integration** for HealthCheck ([#235](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/235)).
    - Fail-count tracking in `NodeHealthCheck` ([#244](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/244)).

- **Checkpointing**
    - **CPU shared-memory D2H path** in `FileSystemWriterAsync` removes a redundant H2H copy and resolves the prior shm D2H race ([#298](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/298)).
    - **PersistentAsyncCaller upgrades**: QoS control, worker data cache, warmup, IPC-handle caching via `ConsistentDataIdentifier`, and class-level metadata cache in `CachedMetadataFileSystemReader` ([#273](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/273), [#274](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/274), [#275](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/275)).
    - Reliability fixes: SIGSEGV on SIGKILL with dangling CUDA IPC handles ([#284](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/284)), CUDA IPC handle errors in persistent worker ([#288](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/288)), premature GC of preloaded pinned host tensors in `TemporalAsyncCaller` ([#291](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/291)), MXFP8/TE quantized tensor handling in IPC cache ([#276](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/276)), spawned persistent worker CUDA-device init ([#238](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/238)).

- **Fault attribution — productized as standalone services (experimental)**

    > The attribution module — including the Attribution Service, Flight Recorder integration, LogSage, and MCP integration — remains **experimental** in v0.6. APIs, CLI flags, and service contracts may change in subsequent releases.

    - **NVRx Attribution Service (`attrsvc`)** and **NVRx Slurm Monitor Service (`smonsvc`)** introduced as FastAPI-based standalone services ([#242](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/242), [#248](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/248)).
    - **`ft_launcher`-managed `attrsvc`** for co-located deployment ([#318](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/318)); UDS endpoints for `attrsvc`/`smonsvc` ([#315](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/315)).
    - Attribution is now an **optional package** — install with `pip install nvidia-resiliency-ext[attribution]` ([#305](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/305)). Attribution internals refactored under a `svc` subpackage with a clear controller/runner boundary ([#295](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/295), [#313](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/313), [#316](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/316)).
    - **PyTorch Flight Recorder (experimental)** support in `attrsvc` ([#283](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/283)); FR ordering switched to window-based instead of PG description ([#210](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/210), [#216](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/216), [#219](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/219)).
    - **LogSage (experimental)** integrated as an attribution module — direct in-process API, configurable LLM model via env, LogSage v0.1.7 ([#224](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/224), [#249](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/249), [#267](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/267), [#289](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/289), [#297](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/297), [#308](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/308)).
    - Triggers and outputs: last-cycle attribution trigger ([#247](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/247)), job-completion handling ([#251](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/251)), Slack bot notifications ([#253](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/253)).
    - **MCP (Model Context Protocol) integration (experimental)** exposes the attribution module as a tool for the NVIDIA resiliency agent ([#215](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/215)); `nvrx-attr` Claude skill bundle for the attribution workflow ([#312](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/312)).

- **Log aggregation & observability**
    - **Two-level gRPC log aggregation** (N leaves + root) with auto-tier selection and end-to-end tests ([#280](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/280), [#307](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/307), [#309](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/309)).
    - Writer-thread + persistent-reader refactor ([#254](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/254)); pipe-based per-cycle log capture and split-log support ([#225](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/225), [#240](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/240)).
    - **`NVRxCycleInfo` protobuf** exposes per-cycle metadata over the new gRPC interface ([#258](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/258), [#292](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/292)).
    - **GPU memory logger** ([#206](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/206)).

- **Build, packaging & security**
    - `poetry-dynamic-versioning` — wheel versions are derived from git tags (`v0.6.0-rc1`, `v0.6.0`, etc.) ([#260](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/260)).
    - New optional extras: `[attribution]` for the attribution service stack (LogSage, MCP, FastAPI, Slack), `[dataflow]` for `nvdataflow` integration.
    - Wheel security-scan cleanups and Bandit findings (`ionice` subprocess, FR module) ([#219](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/219), [#294](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/294), [#306](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/306)).
    - Torch 2.10 and `langchain` lock updates for CVE compliance.

### Deprecations & Removals

- **Legacy dynamic rendezvous (v1)** is deprecated; barrier-based rendezvous (v2) is the default. Plan to remove v1 in a future release ([#282](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/282)).
- **`--ft-restart-policy` (min-healthy)** is deprecated ([#259](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/259)).
- **`ptl_resiliency` package is deprecated and removed** from the wheel ([#282](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/282), [#285](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/285)). PyTorch Lightning users should pin v0.5.x or migrate to the underlying `fault_tolerance` / `checkpointing` / `attribution.straggler` APIs directly.
- **OneLogger integration** removed ([#257](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/257)).
- **In-process restart** is no longer the focus of the fast-restart solution and is slated for deprecation in a future release; new deployments should use in-job restart via `ft_launcher`.

### Installation

```bash
# Core resiliency (fault tolerance, checkpointing, health checks)
pip install nvidia-resiliency-ext

# With attribution stack (Attribution Service, LogSage, MCP, Slack notifications)
pip install nvidia-resiliency-ext[attribution]

# With nvdataflow integration
pip install nvidia-resiliency-ext[dataflow]
```

### Known Issues & Limitations

- **CUDA 12 / Ubuntu 22.04** users are advised to build from source — PyPI wheels default to CUDA 13.
- **Attribution, Flight Recorder analysis, LogSage, and MCP integration are experimental.** APIs, CLI flags, and service contracts may change in subsequent releases.
- **External InJob control-plane is experimental** and not yet QA-validated; APIs may change in subsequent releases.


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

- Node health check service integration
    - `ft_launcher` can query a node-local health check service before rendezvous by passing `--ft-node-health-check-endpoint`.
    - Users can reuse NVIDIA Base Command Manager (BCM) Slurm prolog or epilog health checks behind the same service interface. For example, customers can deploy an `nvhcd`-compatible daemon that invokes an existing BCM health check script and converts the result to the NVRx health check response format.
    - The integration is service-compatible: users can provide any equivalent daemon that implements the NVRx `HealthCheckService.RunHealthCheck` gRPC API over a Unix domain socket and returns JSON output with `fail_count == 0` for healthy nodes.
    - Example: `--ft-node-health-check-endpoint unix:///var/run/nvhcd.sock`. If the endpoint is not configured, or the optional service is unavailable, NVRx skips the external node health check; explicit failures returned by the service mark the node unhealthy.

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
