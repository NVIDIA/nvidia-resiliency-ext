# Testing and Simulation Utilities

This directory contains testing and simulation tools that are separate from production code. These utilities help with development, testing, and debugging of the fault tolerance system.

## Purpose

- **Separation of Concerns**: Keep testing/simulation code separate from production code
- **Clean Production Code**: No testing utilities mixed with production logic
- **Easy Maintenance**: All testing tools in one place
- **Extensibility**: Easy to add new testing utilities in the future

## Current Utilities

### health_check_injector.py

GPU health check failure injection utility that uses monkey-patching to simulate GPU failures without modifying production code.

**Usage**: Set `NVRX_INJECT_GPU_FAILURE` environment variable to enable automatic injection.

See the module documentation for details:
```python
from nvidia_resiliency_ext.testing_utils import health_check_injector
help(health_check_injector)
```

## Adding New Utilities

When adding new testing/simulation tools:

1. Create the module in this directory
2. Use environment variables to enable/disable functionality
3. Use monkey-patching or other non-invasive techniques when possible
4. Document the utility clearly
5. Add examples if applicable

## Examples

### Using Health Check Injection

```bash
# Enable GPU failure injection
export NVRX_INJECT_GPU_FAILURE=1:0,3:1,5:2
export SLURM_PROCID=0

# Run your training - injection activates automatically
python -m nvidia_resiliency_ext.fault_tolerance.ptl_resiliency \
    --nnodes 4 \
    --nproc_per_node 8 \
    --max_restarts 10 \
    train.py
```

## Design Principles

1. **Minimal Production Impact**: Testing utilities should have zero impact when disabled
2. **Environment-Driven**: Enable/disable via environment variables
3. **Self-Contained**: Each utility should be independent
4. **Well-Documented**: Clear documentation and examples
5. **Type-Safe**: Use proper type hints and validation

## Future Utilities

Potential additions to this directory:

- Network latency simulation
- Memory pressure simulation
- Checkpoint corruption simulation
- Communication failure injection
- Performance profiling tools
- Debugging helpers
- Test scenario generators

## Notes

- These utilities are for **testing and debugging only**
- Do not use in production unless specifically testing failure scenarios
- Some utilities may have performance overhead when enabled
- Always test utilities in a safe environment first

