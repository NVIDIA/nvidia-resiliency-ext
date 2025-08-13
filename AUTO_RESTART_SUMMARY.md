# Auto-Restart Feature - Executive Summary

## Background and Rationale

### Current State: Two Separate Restart Solutions

NVRx currently has two distinct restart solutions:

- **InProcess**: Training restart within the same process
- **InJob**: Training restart across process boundaries (similar to PyTorch Elastic)

### Why Not Integrate InJob with InProcess?

While integrating InJob with InProcess might seem simpler at first glance, this approach has significant drawbacks:

#### Overlapping Features Create Complexity
InJob and InProcess share several overlapping components:
1. **Rendezvous** - Process coordination and discovery
2. **Rank Assignment** - Worker rank management
3. **Job Monitor** - Health monitoring and failure detection
4. **Job Abort Workflow** - Termination handling
5. **TCPStore** - Distributed state management

Managing two overlapping software stacks increases complexity and maintenance overhead without providing optimal training resilience.

#### Integration Challenges
- **Tight Coupling**: InJob is tightly coupled with features that overlap with InProcess
- **Architectural Conflicts**: Different design philosophies and state management approaches
- **Maintenance Burden**: Two codebases to maintain and debug

### Our Chosen Path: Organic Evolution of InProcess

We've chosen to **organically enhance InProcess** rather than integrate with InJob. This approach:

✅ **Maximizes Training Resilience**: Single, unified restart framework  
✅ **Reduces Complexity**: One codebase to maintain and debug  
✅ **Enables Gradual Migration**: Existing InProcess users can adopt features incrementally  
✅ **Provides Clear Roadmap**: Clear path to convergence  

### Convergence Strategy

The auto-restart feature is the **first step** toward a unified restart framework. Our roadmap includes:

1. **✅ Cross-Process Restart** (Current: Auto-Restart)
2. **🔄 Dynamic World Size Support** (Next: Enhanced InProcess)
3. **🔄 Advanced Health Checks** (Future: Leverage InJob health check patterns)
4. **🔄 Unified Framework** (Goal: Single restart solution)

## What is Auto-Restart?

Auto-Restart is a feature that enhances the existing InProcess resilience framework to support **training restart across process boundaries**. Unlike the original InProcess which only allowed restart within the same process, this enhancement enables seamless restart across different processes.

## Key Benefits

✅ **Cross-Process Resilience**: Training survives process-level failures  
✅ **Automatic Recovery**: No manual intervention required for restarts  
✅ **State Persistence**: Training state maintained across restarts  
✅ **Backward Compatible**: Existing InProcess functionality preserved  

## How It Works

### 1. Process Forking
```python
# MUST be called before any other imports
from nvidia_resiliency_ext.shared_utils.auto_restart import fork_and_monitor
fork_and_monitor()
```

### 2. Parent-Child Architecture
- **Parent Process**: Monitors child and restarts on failure
- **Child Process**: Runs the actual training with InProcess wrapper
- **Exit Code 130**: Signals clean termination (no restart needed)

### 3. External TCPStore
- TCPStore service runs independently of training processes
- Persists across process restarts
- Solves barrier coordination issues

## Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `fork_and_monitor()` | Main entry point, process management | `shared_utils/auto_restart.py` |
| `TCPStoreService` | External distributed store | `inprocess/tcpstore_service.py` |
| `RestartAbort` | Clean termination exception | `inprocess/exception.py` |
| Enhanced State | Cross-process state management | `inprocess/state.py` |

## Configuration Changes

### New Parameters
```python
inprocess.Wrapper(
    store_kwargs={
        'tcp_store_host_rank': -1,  # External TCPStore
        'port': int(os.environ['MASTER_PORT']) + 2,
    },
    active_world_size=args.inprocess_active_world_size,  # Hot spare support
)
```

### Existing Arguments Enhanced by Auto-Restart
- `--inprocess-max-iterations`: Now refers to total restart limit (both in-process and across-process)
- `--inprocess-active-world-size`: Number of active ranks (enables hot spare functionality)

**Note**: These arguments already existed in InProcess. Auto-restart enhances their functionality to work across process boundaries.

## State Management

### Counters
- **`global_iteration_counter`**: Increments by 100 for cross-process restarts
- **`job_restart_counter`**: Tracks total restarts (both types)
- **`ranks_restart_counter`**: Per-iteration restart tracking

### Key Space Strategy
```
Iteration 0:   Keys: iteration_0_*
Iteration 100: Keys: iteration_100_*
Iteration 200: Keys: iteration_200_*
```

## Usage Example

```python
#!/usr/bin/env python3

# Enable auto-restart (MUST be first)
if os.getenv('NVRX_ENABLE_FORK_AND_MONITOR', '1') == '1':
    from nvidia_resiliency_ext.shared_utils.auto_restart import fork_and_monitor
    fork_and_monitor()

# Now import other modules
import torch
from megatron.training import inprocess_restart

# Wrap training function
pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

if __name__ == "__main__":
    main()
```

## Critical Requirements

⚠️ **CUDA Safety**: `fork_and_monitor()` must be called before any CUDA initialization  
⚠️ **Import Order**: Must be the very first import in your application  
⚠️ **Exit Codes**: RestartAbort uses exit code 130 for clean termination  

## Integration Points

### With Megatron-LM
```python
# In pretrain_mamba.py
from megatron.training import inprocess_restart
pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
```

### With External TCPStore
```python
# TCPStore service runs on Rank 0
# Training processes connect as clients
store = torch.distributed.TCPStore(
    host_name=os.environ['MASTER_ADDR'],
    port=int(os.environ['MASTER_PORT']) + 1,
    world_size=int(os.getenv('WORLD_SIZE', '1')),
    is_master=False,
)
```

## Monitoring and Debugging

### Log Messages
- Parent process logs child launches and exits
- Exit code 130 indicates clean termination
- Other exit codes trigger automatic restart

### Common Issues
1. **Import Order**: CUDA imported before `fork_and_monitor()`
2. **Port Conflicts**: TCPStore port conflicts with existing services
3. **State Corruption**: Incomplete state restoration after restart

## Strategic Advantages

### Why This Approach Over Alternatives?

#### vs. InJob Integration
- **Avoids Overlapping Stacks**: No duplicate rendezvous, rank assignment, or monitoring code
- **Reduces Complexity**: Single codebase instead of managing two overlapping frameworks
- **Better Maintainability**: One team can focus on one restart solution

#### vs. Dual Maintenance
- **Single Codebase**: Eliminates the need to maintain and debug two separate systems
- **Unified Testing**: All restart scenarios tested in one framework
- **Consistent Behavior**: Same restart logic for all failure modes

#### vs. Incomplete Solutions
- **Clear Roadmap**: Defined path to full convergence with InJob features
- **Incremental Adoption**: Users can adopt features as they become available
- **Future-Proof**: Built on extensible architecture for future enhancements

#### vs. Vendor Lock-in
- **Leverages Investment**: Builds on existing InProcess codebase and user base
- **Gradual Migration**: No need to rewrite existing training code
- **Risk Mitigation**: Proven InProcess foundation with incremental improvements

## Performance Considerations

- **Memory Overhead**: Minimal additional memory for monitoring
- **Network Latency**: TCPStore communication adds small overhead
- **Restart Time**: Process forking is fast, training restart depends on checkpoint size

## Future Enhancements

### Convergence Roadmap
- **Phase 1**: ✅ Cross-process restart (Current: Auto-Restart)
- **Phase 2**: 🔄 Dynamic world size support in InProcess
- **Phase 3**: 🔄 Advanced health checks (leveraging InJob patterns)
- **Phase 4**: 🎯 Unified restart framework

## Support and Documentation

- **Design Document**: `AUTO_RESTART_DESIGN.md`
- **Diagrams**: `AUTO_RESTART_DIAGRAMS.md`
- **Tests**: `tests/shared_utils/test_auto_restart.py`
- **Examples**: `pretrain_mamba.py` in Megatron-LM

---

*This feature significantly enhances training resilience by enabling cross-process restarts while maintaining compatibility with existing InProcess functionality.*
