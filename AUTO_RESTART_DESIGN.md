# Auto-Restart Feature Design Document

## Overview

The Auto-Restart feature enhances the existing InProcess resilience framework to support training restart across process boundaries. While the original InProcess only allowed training restart within the same process, this enhancement enables seamless restart across different processes, making it suitable for long-running distributed training jobs that may encounter process-level failures.

## Key Features

1. **Cross-Process Restart**: Training can now restart across process boundaries, not just within the same process
2. **External TCPStore**: TCPStore is now hosted externally to persist across process restarts
3. **Unique Key Space Management**: Global iteration counter increments by 100 to ensure unique key spaces
4. **Job Restart Tracking**: Per-job restart counter tracks both in-process and across-process restarts
5. **Clean Exit Handling**: RestartAbort exits with code 130 to signal clean termination

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Auto-Restart System                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────-──┐ │
│  │   Parent Process│    │  Child Process  │    │ TCPStore     │ │
│  │   (Monitor)     │    │  (Training)     │    │ Service      │ │
│  │                 │    │                 │    │              │ │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌──────-───┐ │ │
│  │ │fork_and_    │ │    │ │InProcess    │ │    │ │External  │ │ │
│  │ │monitor()    │ │    │ │Wrapper      │ │    │ │TCPStore  │ │ │
│  │ │             │ │    │ │             │ │    │ │          │ │ │
│  │ │• Fork child │ │    │ │• Training   │ │    │ │• Hosted  │ │ │
│  │ │• Monitor    │ │    │ │• Resilience │ │    │ │  by      │ │ │
│  │ │• Restart    │ │    │ │• Restart    │ │    │ │  Rank 0  │ │ │
│  │ │  on failure │ │    │ │  logic      │ │    │ │• Persists│ │ │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ │  across  │ │ │
│  └─────────────────┘    └─────────────────┘    │ │  restarts│ │ │
│                                                │ └─────────-┘ │ │
└─────────────────────────────────────────────────────────────────┘
```

### Process Flow

```
1. Application starts
   ↓
2. fork_and_monitor() called (before any imports)
   ↓
3. Parent process forks child
   ↓
4. Child process continues with training
   ↓
5. If training fails → InProcess handles restart
   ↓
6. If InProcess exits with RestartAbort (code 130) → Clean exit
   ↓
7. If InProcess exits with other codes → Parent restarts child
   ↓
8. Parent waits for child completion or restart
```

## Core Components

### 1. fork_and_monitor() Function

**Location**: `src/nvidia_resiliency_ext/shared_utils/auto_restart.py`

**Purpose**: Main entry point for auto-restart functionality

**Key Features**:
- Must be called before any other imports (especially CUDA-related)
- Uses `os.fork()` to create child process
- Parent monitors child and restarts on failure
- Recognizes clean exit codes (default: 130)

**Usage Example**:
```python
# ✅ CORRECT - Call at the very beginning
from nvidia_resiliency_ext.shared_utils.auto_restart import fork_and_monitor
fork_and_monitor()

# Now import other modules
import torch
import numpy as np
# ... rest of your code
```

### 2. External TCPStore Service

**Location**: `src/nvidia_resiliency_ext/inprocess/tcpstore_service.py`

**Purpose**: Provides persistent distributed store across process restarts

**Key Features**:
- Runs independently of training processes
- Hosted by Rank 0
- Persists across rank restarts
- Solves barrier coordination issues

**Configuration**:
```python
# In training application
store = torch.distributed.TCPStore(
    host_name=os.environ['MASTER_ADDR'],
    port=int(os.environ['MASTER_PORT']) + 1,  # Different port
    world_size=int(os.getenv('WORLD_SIZE', '1')),
    is_master=False,  # Client mode
    timeout=timedelta(seconds=300),
    use_libuv=True,
    wait_for_workers=False,
)
```

### 3. Enhanced State Management

**Location**: `src/nvidia_resiliency_ext/inprocess/state.py`

**Key Changes**:
- `global_iteration_counter` increments by 100 for cross-process restarts
- `job_restart_counter` tracks total restarts (in-process + across-process)
- State persistence across process boundaries

**State Flow**:
```
Initial State → Training → Failure → Restart → New State
     ↓              ↓         ↓        ↓         ↓
  iteration=0   iteration=0  Restart  iteration=100  iteration=100
  job_restart=0 job_restart=0 Abort   job_restart=1  job_restart=1
```

### 4. RestartAbort Exception

**Location**: `src/nvidia_resiliency_ext/inprocess/exception.py`

**Purpose**: Signals clean termination to prevent unnecessary restarts

**Key Features**:
- Exit code 130 (SIGINT-like)
- Caught by wrapper and converted to `sys.exit(130)`
- Parent process recognizes this as clean exit

## Configuration Changes

### 1. InProcess Wrapper Configuration

**New Parameters**:
```python
inprocess.Wrapper(
    # ... existing parameters ...
    store_kwargs={
        'tcp_store_host_rank': -1,  # External TCPStore
        'timeout': timedelta(seconds=300),
        'port': int(os.environ['MASTER_PORT']) + 2,
    },
    active_world_size=args.inprocess_active_world_size,  # Hot spare support
)
```

### 2. Command Line Arguments

**Existing Arguments Enhanced by Auto-Restart**:
- `--inprocess-max-iterations`: Now refers to total restart limit (both in-process and across-process)
- `--inprocess-active-world-size`: Number of active ranks (enables hot spare functionality)

**New Arguments**:
- `--inprocess-tcp-store-host-rank`: Set to -1 for external TCPStore

## Key Implementation Details

### 1. Global Iteration Counter Management

```python
# When InProcess restarts across process boundary
if rank == 0:
    # Wait for all ranks to acknowledge
    new_iteration = self.increment_global_iteration_counter(100)
```

**Why 100?**: Ensures unique key space for keys prefixed by iteration number, preventing conflicts between old and new training sessions.

### 2. Job Restart Counter Logic

```python
def should_increment_job_restart_counter(self, active_world_size: int, ranks_count: int = None) -> bool:
    """Check if job restart counter should be incremented."""
    if ranks_count is None:
        ranks_count = self.get_ranks_restart_counter()
    return ranks_count >= active_world_size
```

**Purpose**: Tracks when all active ranks have completed at least one iteration, enabling coordinated job-level restart counting.

### 3. Clean Exit Handling

```python
def _handle_restart_abort(self, exc, context=""):
    """Handle RestartAbort by exiting with code 130."""
    if isinstance(exc, RestartAbort):
        log.info(f'RestartAbort detected in {context}, exiting with code {exc.exit_code}')
        sys.exit(exc.exit_code)
    return exc
```

**Flow**: RestartAbort → Wrapper catches → `sys.exit(130)` → Parent recognizes clean exit → No restart

## Usage Examples

### 1. Basic Training Application

```python
#!/usr/bin/env python3

# Enable auto-restart (MUST be first import)
if os.getenv('NVRX_ENABLE_FORK_AND_MONITOR', '1') == '1':
    from nvidia_resiliency_ext.shared_utils.auto_restart import fork_and_monitor
    fork_and_monitor()

# Now import other modules
import torch
import torch.distributed as dist
from megatron.training import inprocess_restart

def main():
    # Training logic here
    pass

if __name__ == "__main__":
    main()
```

### 2. With InProcess Integration

```python
# Wrap training function with InProcess
pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

# Use external TCPStore
if store:
    dist.init_process_group(backend='nccl', store=store)
```

## Benefits

### 1. **Enhanced Resilience**
- Training can survive process-level failures
- Automatic restart without manual intervention
- Maintains training state across restarts

### 2. **Improved Resource Utilization**
- Hot spare ranks can be configured
- Better fault tolerance in distributed environments
- Reduced training downtime

### 3. **Simplified Operations**
- No need to manually restart failed training jobs
- Consistent restart behavior across different failure modes
- Better integration with job schedulers

### 4. **Maintained Compatibility**
- Existing InProcess functionality preserved
- Gradual migration path available
- No breaking changes to existing APIs

## Limitations and Considerations

### 1. **CUDA Safety**
- `fork_and_monitor()` must be called before CUDA initialization
- Process forking with active CUDA context can cause issues
- Critical to follow import order guidelines

### 2. **Resource Management**
- External TCPStore service requires dedicated resources
- Memory usage may increase with persistent state
- Network port management needed

### 3. **Debugging Complexity**
- Process monitoring adds debugging complexity
- Exit code interpretation required
- Log analysis across multiple processes


## Conclusion

The Auto-Restart feature significantly enhances the InProcess resilience framework by enabling cross-process training restarts. This enhancement addresses the limitations of process-bound restarts while maintaining compatibility with existing functionality. The implementation provides a robust foundation for long-running distributed training jobs that require high availability and fault tolerance.

The key architectural decisions around external TCPStore, unique key space management, and clean exit handling ensure that the system can gracefully handle various failure scenarios while maintaining training continuity across process boundaries.
