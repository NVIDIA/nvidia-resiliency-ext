# Release Notes for NVIDIA Resiliency Extension v0.5.0

Release Date: November 13, 2025

This release includes major new features for fault tolerance, checkpointing, health monitoring, and attribution analysis, along with numerous improvements and bug fixes.

---

## 🎉 Major New Features

### 1. Barrier-Based Rendezvous (PR #201)
- Introduced alternative barrier-based rendezvous implementation for fault tolerance
- Supports libuv-based communication for improved performance
- Adds `close()` semantics for proper resource cleanup
- Includes early termination support when unhealthy node count exceeds threshold
- Comprehensive unit test coverage for barrier rendezvous

### 2. Flight Recorder Attribution Module (PR #172)
- New attribution module to identify ranks interrupting workload progress
- Analyzes Flight Recorder traces to detect GPU errors, host issues, and GIL locks
- Uses `max_enqueued_collective_seq_id` heuristic to detect host issues
- Handles PyTorch watchdog timeout scenarios
- Supports operation without LLM integration
- Includes reference outputs and comprehensive unit tests with sample FR traces
- Located in `nvidia_resiliency_ext.attribution.trace_analyzer`

### 3. In-Job Profiling Metrics (PR #185)
- Added comprehensive timing metrics for fault tolerance lifecycle
- New `ProfilingEvent` enum tracks key events:
  - FAILURE_DETECTED, WORKER_TERMINATED
  - RENDEZVOUS_STARTED, RENDEZVOUS_COMPLETED
  - WORKER_START_STARTED, WORKER_START_COMPLETED
- `FaultToleranceProfiler` class with thread-safe event recording
- OneLogger exporter integration for metrics reporting

### 4. Flight Recorder Trace Collection (PR #122)
- Automatic FR trace collection at abort when Flight Recorder and env vars are configured
- `TraceCollector` class for collecting FR traces from all ranks
- Integration with `AbortTorchDistributed` workflow step
- Controlled via `NVRX_FR_TRACE_PATH` environment variable
- Disables stacktrace in FR dump to avoid GIL contention
- Moved trace_collector to use NVRX logging infrastructure

### 5. Enhanced GPU and NVLink Health Checks (PR #145)
- Refactored `GPUHealthCheck` to support device-specific monitoring
- New `NVLHealthCheck` class for NVLink health validation
- Automatic health check chaining in `Wrapper` class
- `ChainedGPUHealthCheck` and `ChainedNVLHealthCheck` for in-process use
- Single GPU health check API for individual device validation
- Updated trace collector to use new GPU health check API

### 6. NIC Health Monitoring and GB200 Support (PR #152)
- `ChainedNicHealthCheck` extending HealthCheck base class
- Automatic NIC health checks when `LOCAL_RANK` environment variable present
- GB200 platform detection via pynvml device count
- Configurable via `enable_nic_monitor` parameter (default: False)
- Integrated into InProcess wrapper initialization

### 7. Infrastructure Rank Support (PR #196)
- New `use_infra_group_rank` config option to use infrastructure-provided ranks (e.g., SLURM_PROCID)
- Alternative to sorted participant-based rank assignment
- Selective section monitoring - only sends IPC for configured sections
- Unidirectional section communication via `skip_section_response` (default: True)
- Significantly reduces latency for high-frequency monitoring operations
- CLI flag: `--ft-use-infra-group-rank`
- Hot spare not supported when using infrastructure group rank

### 8. Attribution Base Module (PR #141)
- `NVRxAttribution` base class for modular attribution pipeline
- Enables extensible attribution analysis workflows
- Comprehensive documentation and unit tests

---

## 🚀 Improvements

### Fault Tolerance Enhancements

#### InJob Restart Improvements (PR #44)
- Enhanced InJob restart in RankMonitorServer and launcher
- Modified `read_obj_from_ipc_stream` to raise exceptions instead of returning None
- RankMonitorServer handles launcher IPC connections with explicit closure
- Added launcher IPC server to receive commands from launcher
- LocalElasticAgent sends close messages before shutting down workers
- RankMonitorStateMachine allows FINALIZED state transitions
- Proper cleanup of launcher IPC socket files
- RankMonitorServer subprocess now uses `fork` for ~10 second speedup (PR #47)

#### Rendezvous Improvements
- Updated InJob to use upstream `torch.distributed.elastic` (PR #132)
- Removed old PyTorch 2.3.1 compatibility code from InJob
- Moved GPU health check out of Rendezvous join operation (PR #100)
- `ensure_node_is_healthy` and `handle_control_requests_from_rank` moved to `FtRendezvousHandler.next_rendezvous()`
- Support for `--nnodes=<min>:<max>` hot spare usage (PR #185)
- Legacy rendezvous set as default with option to use barrier-based
- Added `set_worker_group` to legacy rendezvous (PR #172)

#### Configuration and CLI
- Removed ft-launcher* arguments for consistency (PR #64)
  - Removed unnecessary config ignore flags
  - Fixed config file argument handling
  - Removed redundant parameter prefixes
  - Updated documentation and examples
- Allow null values for timeout CLI arguments (PR #195)
- `--rdzv-endpoint` now properly supported with c10d backend (PR #194)
- Fixed `--max-restarts` to reflect job level restart attempts (PR #211)
- Allowed ft_launcher to run with default FaultToleranceConfig (PR #205)

#### Monitoring and Logging
- Changed default worker monitor interval to 0.1 second (PR #48)
- Set monitor-interval to 5 seconds for InProcess examples (PR #63)
- Monitor unhealthy nodes in the system (PR #203)
- Changed join default timeout to 300 seconds
- Changed monitor_interval default to 0.3 second
- Lower rank monitor log level to reduce normal log spam (PR #203)
- Cleanup logging - limit "periodic restart check" log to twice per cycle (PR #203)
- Improved interrupted ranks logging with smart formatting:
  - ≤16 ranks: Show all individual ranks
  - 17-32 ranks: Show first 3 and last 3 ranks
  - >32 ranks: Show first 5 and last 5 with total count
- Truncate exception chain logging to reduce verbosity (PR #136)
- Added timestamp to `log_exc` function
- Fixed rank monitor client logger initialization at class level (PR #192)
- Use previous logging file when available (PR #171, #174)
- Fixed microsecond truncation in log timestamps (PR #179, #180)
- Traceback indents preserved during aggregation (PR #181, #184)
- Configure RankMonitorServer to inherit env vars from launcher (PR #177, #178)
- PyPI installation usage instruction added to Log Aggregator (PR #162)
- Fixed logger format for regular stderr/stdout handler
- Named change: `NVRX_DIST_LOG_DIR` → `NVRX_NODE_LOCAL_TMPDIR` (PR #156)

#### Workload Management
- Added workload exception for fault injection testing (PR #42)
- Improved fault injection registration setup (PR #42)
- Add mechanism to force clear raise event (PR #81)
- Add timestamp to workload exception
- Memory leak fixes in restart handling (PR #112):
  - Explicitly remove exception references
  - Move GC after stage advancement
  - Prevents expensive workload resources from being held
- InJob NestedRestarter regression fixes (PR #162)
- Added `monitor_process_pidfile` parameter to store monitor process PID (PR #99)
- Updated InProcess usage guide with `wait_daemon` and `--kill-on-bad-exit=0` instructions

#### Rank Monitor Improvements
- Fixed launcher socket finding robustness (PR #191)
  - Uses server process PID for exact socket path
  - Eliminates race conditions in socket discovery
  - Improves test reliability
- Changed default worker stop timeout to 15 seconds
- Added logging to in-process rank assignment (PR #133)

### Checkpointing Improvements

#### Stability and Architecture (PR #197)
- **Multithread File IO Instead of Multiprocess**: Major architectural change
  - Simplifies error propagation logic
  - Improves shutdown cleanup
  - Enhances overall stability

#### Error Handling and Propagation
- Fixed error propagation in checkpoint saving (PR #138, #170)
  - Uses torch's DistWrapper to send exceptions to coordinator
  - Ensures necessary operations happen in finally blocks
  - Added comprehensive tests with error injection
- Fixed sync CP to preload tensors (PR #111)
- Training exit after InProcess abort ensures clean shutdown of persistent async workers (PR #199)
- Add defensive check for kwargs being None (PR #204, #207)

#### Async Checkpoint Enhancements
- Made persistent async checkpoint worker default (PR #108)
- Fixed cross-call state pollution in AsyncRequest (PR #193)
  - Security fix: AsyncRequest.async_fn_kwargs no longer uses mutable default
  - Each request gets isolated kwargs dict
  - Prevents unintended mutation between requests
  - **Security Advisory: NVBUGS 5504235**
- Allow multiple AsyncCallsQueue in a process (PR #169)
  - Enables different behavior for distributed vs non-distributed checkpointing
- Ability to abort async checkpoint process (PR #154)
- Extended documentation on AsyncCallsQueue (PR #157)
- Add missing variable `rank` in `PersistentAsyncCaller` (PR #71)
- LocalCheckpointManager: Update async usage to `persistence=False` (PR #116)
  - AsyncRequests incompatible with persistent queue due to unpickleable methods

#### Metadata and Caching
- Migrate changes from MCore for checkpoint implementation (PR #40)
  - Refactored metadata cache
  - Minor fixes for PyTorch DCP signature changes
  - Updated examples/tests for torch.FSDP compatibility
- Enable caching correctly in checkpoint examples (PR #73)

#### PyTorch Compatibility
- MSC changes compatible with PyTorch 2.3 (PR #87)
  - Uses `inspect` module for compatibility checks
- Fixed `device_ids` usage in checkpoint managers (PR #65)
- Fixed `no_dist` behavior for async ckpt on certain ranks (PR #88)

#### Examples and Documentation
- Skip missing directory in async_ckpt.py example (PR #62)
- Updated examples with improved functionality (PR #101):
  - async_ckpt.py with MSC support
  - async_writer.py with bug fixes
  - local_ckpt.py improvements
- Updated checkpoint docstrings and comments (PR #85, #90, #106)
- Added MSC usage documentation (PR #57)
- Checkpoint test utils: fix rank/world size (PR #151)

### Health Checks
- Ensure Transformer Engine releases process groups (PR #137)
- Automatic health check chaining for GPU, NVL, and NIC monitoring
- Device-specific health monitoring support

### Logging Infrastructure
- NVRX Logger improvements across all modules
- OneLogger integration for metrics and profiling
- Used nvrx logger throughout codebase (PR #190)
- Corrected license headers (PR #190)

### Documentation
- Clarify nested restarter usage (PR #94)
- Documentation building instructions added to CONTRIBUTING.md (PR #105)
- Fixed absolute GitHub URLs to relative internal links
- InProcess doc clarification (PR #119)
  - Renamed "optimal" example to "advanced" for clarity
  - Improved semantic wording throughout
- Updated usage guides with monitor-interval guidance (PR #83)
- Updated README.md with github.io links (PR #110)
- Updated CONTRIBUTING.md (PR #84)
- Multiple documentation updates for better user experience

---

## 🐛 Bug Fixes

### Security Fixes
- Security warning patches for various modules (PR #52, #60)
- Security risk in file symlink resolution fixed (PR #194)
- Fixed cross-call state pollution in AsyncRequest (PR #193) - **NVBUGS 5504235**
- Added security warning skip comments where appropriate (PR #220)

### PyTorch Compatibility
- Torch 2.3.1 compatibility fixes:
  - `RendezvousInfo` not available (PR #132)
  - `RendezvousStoreInfo.build` parameter differences
  - `DEFAULT_PORT` not defined in c10d rendezvous
  - `next_rendezvous` access patterns
- PyTorch 2.8 Flight Recorder env variable support (PR #163)
  - Considers `TORCH_NCCL_BUFFER_SIZE` (< 2.8) and `TORCH_FR_BUFFER_SIZE` (>=2.8)

### Launcher and Configuration
- Fixed `--max-restarts` to reflect job-level restart attempts
- Restored deprecated arguments for backward compatibility (PR #82)
- Correct rank assignment example (PR #66)
- Fixed RankMonitorServer subprocess creation for better performance

### Logging
- Fixed microsecond truncation to int seconds in timestamps (PR #179, #180)
- Traceback indents preserved during log aggregation (PR #181)
- Initialize rank monitor client logger at class level (PR #192)
- `world_local_tmp` can be None if not specified in ENV (PR #162)

### Tests and CI
- Simplified dynamic rendezvous unit tests (PR #147)
  - Removed redundant tests not introduced by InJob
  - Added timeouts to prevent hanging
- Race condition in CI test fixed (PR #201)
- Fixed broken setup after straggler moved (PR #130)
- Fixed launcher socket finding in rank monitor server test (PR #191)
- Check for invalid/0 timestamps in straggler detection (PR #115)
- Make logging exhaustive unit tests optional (PR #164)

### Other Fixes
- Removed NVTE_FUSED_ATTN for NeMo >= 25.02 (PR #81)
- Set torch.cuda.set_device to right device in InProcess (PR #103)
  - PyTorch maintains active CUDA device in thread-local storage
- Default InProcess examples to CUDA (PR #120)
- Fixed incorrect concatenation in examples

---

## 🔧 Build and Dependencies

### Dependency Updates
- pynvml deprecated, replaced with nvidia-ml-py (PR #187)
- Added nv-one-logger dependency (PR #141, #185)
- Generate poetry.lock with poetry 1.8.5 (PR #213)
- Added poetry.lock file to repository (PR #203)

### Build Improvements
- CUDA path detection improvements (PR #128)
  - Check for nvcc in $PATH if /usr/local/cuda or $CUDA_PATH not found
  - Better user experience on systems with conda/module-based CUDA
  - Follows CMake FindCUDA patterns
- Used canned builder image (PR #194)
- Fixed cupti_build.py after straggler module move

---

## 🔄 Deprecations and Removals

### Removed Deprecated Code (PR #146)
- Removed deprecated rank filter
- Removed deprecated arguments and tests
- Corrected deprecated references in tests

### Code Cleanup
- Removed unused patches and variables
- Removed blank lines and formatting improvements
- Lint fixes across the codebase
- Refactoring for better code organization

---

## 🧪 Testing Improvements

- Added comprehensive unit tests for:
  - Barrier rendezvous
  - Flight recorder attribution with reference outputs
  - In-job profiling metrics
  - Enhanced health checks
  - Checkpoint error propagation
  - Persistent async worker shutdown
- Sample FR traces added for testing attribution module
- Reference validation summaries for FR attribution tests
- Improved test utilities for checkpoint testing
- Made exhaustive logging tests optional for faster CI

---

## 📝 Breaking Changes

### Configuration Changes
1. **Environment Variable Rename**: `NVRX_DIST_LOG_DIR` → `NVRX_NODE_LOCAL_TMPDIR`
2. **Default Behavior Changes**:
   - Persistent async checkpoint worker is now default
   - Monitor interval defaults changed (0.1s for worker, 0.3s for general, 5s for InProcess)
   - Join timeout default changed to 300 seconds
   - `use_infra_group_rank` defaults to True
   - `enable_nic_monitor` defaults to False
   - `skip_section_response` defaults to True

### API Changes
1. **AsyncRequest**: `async_fn_kwargs` parameter handling changed to prevent state pollution
2. **Health Checks**: Refactored to support device-specific monitoring
3. **ft-launcher**: Removed CLI arguments (deprecated arguments have been removed)

### Module Reorganization
1. **Straggler → Attribution**: Straggler module moved to `attribution` directory
2. **Profiling**: Moved `profiling.py` to `shared_utils/`

---

## 📦 Migration Guide

### From v0.4.1 to v0.5.0

#### Environment Variables
```bash
# Old
export NVRX_DIST_LOG_DIR=/path/to/logs

# New
export NVRX_NODE_LOCAL_TMPDIR=/path/to/logs
```

#### Import Changes
```python
# Old
from nvidia_resiliency_ext.straggler import StragglerDetector

# New
from nvidia_resiliency_ext.attribution.straggler import StragglerDetector
```

#### Configuration Updates
```python
# Monitor intervals - new defaults
# Worker monitor: 0.1s (was variable)
# General monitor: 0.3s (was variable)
# InProcess: 5s (update your configs if using custom values)

# Join timeout
# Default: 300s (was lower - update if you rely on old default)

# Infrastructure rank usage
# Default: use_infra_group_rank=True (was False)
# If you want old behavior:
config.use_infra_group_rank = False
```

#### Checkpoint API
```python
# AsyncRequest - ensure kwargs isolation
# Old (vulnerable to cross-call pollution):
request = AsyncRequest(async_fn=my_fn, async_fn_kwargs={})

# New (automatic isolation):
request = AsyncRequest(async_fn=my_fn)  # kwargs=None by default
# Or explicitly pass fresh dict per call
request = AsyncRequest(async_fn=my_fn, async_fn_kwargs={'key': 'value'})
```

---

## 👥 Contributors

Special thanks to all contributors who made this release possible:

- Hexin Wang (@hexinw-nvidia)
- Seonmyeong Bak (@sbak5)
- Aarti Basant (@aartibasant)
- Russell Hewett (@rhewett-nv)
- Herman Sahota (@herman-ai)
- Diego Pontoriero (@diegs)
- Namit Dhameja (@namitdhameja)
- Shunjia Ding (@shunjiad)
- Jacek Bieniusiewicz (@jbieniusiewi)
- Pramod Kumbhar (@pramodk)
- Anjali Shah (@anjalibshah)
- Abhijit Paithankar (@apaithankar)

---

## 📊 Statistics

- **Total Commits**: 520+ commits
- **Pull Requests Merged**: 100+
- **Release Candidates**: 8 (v0.5.0-rc1 through v0.5.0-rc8)
- **Files Changed**: 200+
- **Major Features**: 8
- **Bug Fixes**: 50+

---

## 🔗 Resources

- [GitHub Repository](https://github.com/NVIDIA/nvidia-resiliency-ext)
- [Documentation](https://nvidia.github.io/nvidia-resiliency-ext/)
- [PyPI Package](https://pypi.org/project/nvidia-resiliency-ext/)
- [Issue Tracker](https://github.com/NVIDIA/nvidia-resiliency-ext/issues)

---

## 📅 Release Timeline

- **v0.5.0-rc1**: August 30, 2025
- **v0.5.0-rc2**: September 3, 2025
- **v0.5.0-rc3**: September 5, 2025
- **v0.5.0-rc4**: September 8, 2025
- **v0.5.0-rc5**: September 16, 2025
- **v0.5.0-rc6**: October 17, 2025
- **v0.5.0-rc7**: October 24, 2025
- **v0.5.0-rc8**: November 1, 2025
- **v0.5.0**: November 13, 2025

---

*For detailed commit-by-commit changes, please see the [full commit history](https://github.com/NVIDIA/nvidia-resiliency-ext/compare/v0.4.1...v0.5.0).*

