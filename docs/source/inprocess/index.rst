Inprocess Restart
=================

In-process restart mechanism is implemented via a Python function wrapper that
adds restart capabilities to an existing Python function implementing
distributed PyTorch workload. Upon a fault, the wrapped function is restarted
across all distributed ranks, within the same operating system process.
Invoking restart of the wrapped function excludes distributed ranks that are
terminated, missing, or deemed unhealthy. When a failure occurs on any worker,
the wrapper ensures the function restarts simultaneously on all healthy ranks.
This process continues until all ranks complete execution successfully or a
termination condition is met.

Compared to a traditional scheduler-level restart, restarting within the same
process removes overheads associated with launching a new scheduler job,
starting a container, initializing a new Python interpreter, loading
dependencies, and creating a new CUDA context.

Restarting in the same process also enables the reuse of pre-instantiated,
process-group- and rank-independent objects across restart attempts. This reuse
eliminates the overhead of repeated reinitialization and minimizes restart
latency.

Features:

- automatic deinitialization of PyTorch distributed process group, and restart
  of the wrapped function upon encountering an unhandled Python exception in
  any distributed rank
- timeout mechanism to detect and recover from deadlocks or livelocks, and a
  guarantee that the job is making meaningful forward progress
- modular and customizable rank reassignment and health check functions
- support for pre-allocated and pre-initialized reserve workers
- gradual engineering ramp up: integration with existing codebase may start
  from restarting the entire ``main()`` function, then gradually refactor
  process-group-independent initialization into a separate function call in
  order to maximally reuse Python objects between restarts and minimize fault
  recovery overhead

For a comprehensive description of this functionality, including detailed
requirements, restrictions, and usage examples, please refer to the :doc:`Usage
Guide <usage_guide>` and :doc:`Examples <examples>`.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage_guide
   api
   examples
