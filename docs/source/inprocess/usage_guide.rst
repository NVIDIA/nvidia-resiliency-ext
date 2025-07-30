Usage Guide
===============================================================================
The :py:class:`nvidia_resiliency_ext.inprocess.Wrapper` serves as the primary interface for accessing
in-process restart functionality. It provides various configuration options
through its arguments, enabling customization of the restart process and fault
monitoring capabilities. To ensure efficient and effective restarts, the
function being wrapped must meet specific requirements. This usage guide
outlines the requirements, features, and limitations of the in-process restart
functionality provided by the :py:class:`Wrapper`.

Requirements
------------
In-process restart functionality requires
`PyTorch <https://pypi.org/project/torch/>`_ v2.5.1 or higher
and
`NCCL <https://github.com/NVIDIA/nccl>`_ v2.26.2 or higher
For further limitations and compatibility details, refer to the :ref:`Known
issues <known_issues>` section.

Requirements for the wrapped function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The wrapped function should be designed to support restarts, meaning it
  should carefully manage any external (e.g., global) state and resources,
  avoid using functions that can only be called once per process, such as
  :py:func:`multiprocessing.set_start_method()` or ``MPI_Init``, to ensure that
  the function can be executed multiple times in the same process without
  issues.

    - The function will automatically retry on any failure, meaning it will be
      called again with the same set of input arguments; extra caution is
      needed if the function accepts mutable arguments that might be modified
      during its execution, as these changes could affect subsequent retries.

- All operations that wait on results from NCCL kernels, or synchronize with
  the GPU, need to release Python `Global Interpreter Lock
  <https://docs.python.org/3/glossary.html#term-global-interpreter-lock>`_
  (GIL).

    - If the Python GIL is not released when a fault occurs, the graceful
      restart procedure cannot proceed. This is because the procedure runs in a
      separate Python thread, which is blocked from execution due to the GIL
      being held. As a result, hung ranks must be forcibly terminated using the
      :ref:`hard timeout <hard_timeout>` mechanism (``SIGKILL``). These
      terminated ranks will not rejoin the distributed job upon restart.

- The function does not suppress :py:exc:`BaseException`. If the wrapped
  function catches a :py:exc:`BaseException`, it must re-raise it to ensure it
  propagates to the outer scope.

- The function is responsible for initialization of PyTorch distributed backend
  (:py:func:`torch.distributed.init_process_group()`); the initialization needs
  to read `standard PyTorch distributed variables
  <https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization>`_
  (``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT`` and
  ``LOCAL_RANK``) from the environment. Users can use torchrun to override the environment
  variables (--master_addr=127.0.0.1, --master_port=29500, etc.) depending on
  their cluster requirements and also to run the provided examples ``torchrun --nproc_per_node=8
  --nnodes=1 --node_rank=0 basic_example.py``. For other environment variables when running
  with torchrun, please refer to the `run_inprocess_injob_example.sh <https://github.com/NVIDIA/nvidia-resiliency-
  ext/blob/main/examples/fault_tolerance/run_inprocess_injob_example.sh>`_ example for the recommended
  default values (for example, --monitor-interval=5).

- it's heavily recommended for the wrapped function to load the state affected
  by distributed collectives from a checkpoint on every restart (e.g. load
  weights of a model); outputs of distributed collectives are likely to become
  corrupted or invalid if a fault happened while a collective was in-flight and
  distributed backend was terminated.

Requirements for the execution environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The PyTorch NCCL watchdog must either be disabled or configured with a
  timeout longer than the ``hard_timeout`` of the
  :py:class:`nvidia_resiliency_ext.inprocess.Wrapper`. If the NCCL watchdog is triggered, it forcibly
  terminates the process, preventing a restart. To adjust the NCCL watchdog
  timeout, use the ``timeout`` argument when calling
  :py:func:`torch.distributed.init_process_group()` with the ``backend``
  parameter set to ``"nccl"``

- The job scheduler must not terminate the entire job if a faulty rank exits
  early or if the main process is terminated; instead, it should wait until all
  user-launched processes have fully exited before ending the distributed job.

Restrictions
------------
- node failure on rank 0 causes termination of the entire job; by default, rank
  0 hosts internal :py:class:`torch.distributed.TCPStore` to allow
  communication between ranks, users may specify a different implementation of
  a distributed store by subclassing from
  :py:class:`nvidia_resiliency_ext.inprocess.store.StoreMixin` and passing the subclass as
  ``store_factory`` argument to the :py:class:`nvidia_resiliency_ext.inprocess.Wrapper`

- blocking calls issued by the main process are generally not recoverable if
  they hang, except for NCCL collectives or functions waiting on them; NCCL
  collectives are asynchronously aborted by a separate monitoring thread that
  calls :py:class:`nvidia_resiliency_ext.inprocess.abort.AbortTorchDistributed`; users can specify
  additional :py:class:`nvidia_resiliency_ext.inprocess.abort.Abort` subclasses to asynchronously
  abort blocking calls from other software components.

- when using :py:class:`nvidia_resiliency_ext.inprocess.abort.AbortTransformerEngine` composed with
  :py:class:`nvidia_resiliency_ext.inprocess.abort.AbortTorchDistributed`, the
  :py:class:`nvidia_resiliency_ext.inprocess.abort.AbortTorchDistributed` should be
  the first abort in the composition chain.  In :py:class:`nvidia_resiliency_ext.inprocess.compose.Compose`,
  the last callback in the chain is executed first, so the following composition is recommended:

  .. code-block:: python

    inprocess.Compose(
        inprocess.abort.AbortTransformerEngine(),
        inprocess.abort.AbortTorchDistributed(),
    )


Functionality overview
----------------------

Implementation overview
~~~~~~~~~~~~~~~~~~~~~~~
Below is a simplified pseudocode snippet that illustrates the order of
operations executed by :py:class:`nvidia_resiliency_ext.inprocess.Wrapper`, providing a high-level
overview of the workflow within this class. This code is for illustrative
purposes only and may omit certain implementation details.

.. code-block:: python

  distributed_store = store_factory(**store_kwargs)
  initial_barrier()
  rank_assignment()
  rank_filter()  # deprecated

  while True:
      initialize()
      health_check()
      try:
        if rank_is_active:
            wrapped_function()
        else:
            sleep()
        completion_barrier()
      except:
          abort()
          finalize()
          health_check()
          iteration_barrier()
          rank_assignment()
          rank_filter()  # deprecated
      else:
          break

  termination_barrier()

Distributed execution behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entering and exiting the :py:class:`Wrapper` act as distributed synchronization
points. Upon entry, all workers retrieve their initial rank assignments and the
total number of workers by reading the standard PyTorch distributed environment
variables (``RANK``, ``WORLD_SIZE``). Subsequently, all workers synchronize
through a ``initial_barrier`` using a user-defined ``barrier_timeout`` to
ensure consistent initialization.

Upon completion of the wrapped function, all ranks that finish enter a
``completion_barrier`` governed by a user-defined ``completion_timeout``. If
any rank fails to synchronize within the ``completion_timeout``, it is treated
as a rank failure, triggering a restart of the wrapped function on all
distributed ranks.

The restart :py:class:`Wrapper` incorporates additional distributed barriers to
ensure proper synchronization: ``iteration_barrier`` (executed before rank
reassignment and filtering), and ``termination_barrier`` (executed before
exiting from the wrapped scope). These barriers are designed to be transparent
to the user, requiring no modifications to the wrapped function or assumptions
about the execution environment. They operate seamlessly to maintain
distributed consistency and coordination.

Rank assignment and filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rank assignment
^^^^^^^^^^^^^^^
The :py:class:`Wrapper` needs to ensure that the wrapped function is restarted
with a consecutive sequence of integer rank indices, from ``0`` to
``WORLD_SIZE - 1``, as some of the ranks from previous iteration may have been
terminated or are in an unhealthy state. Rank reassignment and new world size
computation is performed by
:py:class:`nvidia_resiliency_ext.inprocess.rank_assignment.RankAssignment` instance passed as
``rank_assignment`` argument to the :py:class:`Wrapper`.

Multiple RankAssignments could be composed with :py:class:`nvidia_resiliency_ext.inprocess.Compose`
to achieve the desired behavior.

For example:

.. code-block:: python

    rank_assignment = inprocess.Compose(
        inprocess.rank_assignment.ActivateAllRanks(),
        inprocess.rank_assignment.ShiftRanks(),
        inprocess.rank_assignment.FilterCountGroupedByKey(
            key_or_fn=lambda state: state.rank // 8,
            condition=lambda count: count == 8,
        )
    )

ensures that all ranks within each non-overlapping group of 8 consecutive
ranks remain healthy. If any rank within a group of 8 is unhealthy or
terminated, the entire group is terminated. The remaining healthy ranks are
then reassigned by shifting left to close any gaps, forming a new sequence
of consecutive integers from ``0`` up to the updated ``world size``.

Rank filtering
^^^^^^^^^^^^^^
The :py:class:`Wrapper` categorizes distributed ranks into two groups:

1. active ranks, which are calling the wrapped function
2. inactive ranks, which are waiting idle, and could serve as a static,
   preallocated and preinitialized pool of reserve ranks; reserve ranks would
   be activated in a subsequent restart iteration if previously active ranks
   were terminated or became unhealthy

Rank filtering is a process of selecting active and inactive ranks within a
given restart iteration, and is performed by
:py:class:`nvidia_resiliency_ext.inprocess.rank_assignment.RankAssignment` instance passed as
``rank_assignment`` argument to the :py:class:`Wrapper`.

Multiple :py:class:`nvidia_resiliency_ext.inprocess.rank_assignment.RankFilter` or
:py:class:`nvidia_resiliency_ext.inprocess.rank_assignment.RankAssignment` instances can be composed
using :py:class:`nvidia_resiliency_ext.inprocess.Compose` to achieve the desired behavior. Typically,
all :py:class:`RankFilter` instances should follow any
:py:class:`RankAssignment` steps that recalculate rank indices or adjust the
world size. For example:

.. code-block:: python

    rank_assignment=inprocess.Compose(
        inprocess.rank_assignment.ActiveWorldSizeDivisibleBy(M),
        inprocess.rank_assignment.MaxActiveWorldSize(N),
        inprocess.rank_assignment.ShiftRanks(),
    ),

shifts all healthy ranks to the left to fill gaps created by terminated ranks,
and then ensures that the active world size visible to the wrapped function is
the largest multiple of ``M`` that is not greater than ``N``. The remaining
healthy ranks would be inactive and serve as a reserve.

Initialize
~~~~~~~~~~
The :py:class:`Wrapper` accepts an optional, user-provided
:py:class:`nvidia_resiliency_ext.inprocess.initialize.Initialize` class, which is executed at the
start of every restart iteration, including the first one.
:py:class:`Initialize` can raise exceptions (e.g., if specific preconditions
are not met). Raising a standard Python :py:exc:`Exception` triggers another
restart of the wrapped function, while raising a :py:exc:`BaseException`
terminates the :py:class:`Wrapper`. The included
:py:class:`nvidia_resiliency_ext.inprocess.initialize.RetryController` can be used to limit the
number of restart attempts or to halt execution if the number of healthy
workers drops below a specified threshold.

Multiple initializers could be composed with :py:class:`nvidia_resiliency_ext.inprocess.Compose`.
The composition order follows mathematical composition. Therefore, the last listed function is called first.
Consequently, when using nested restarters, the :py:class:`nvidia_resiliency_ext.inprocess.nested_restarter.NestedRestarterHandlingCompleted`
should be listed first, as handling a restart is not complete until the end of the `Initialize`.

Wrapped function termination mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When a fault or timeout occurs on any rank participating in the distributed
job, the :py:class:`Wrapper` waits for the ``last_call_wait`` interval to allow
all concurrent faults from other distributed ranks to be recorded. After this
waiting period, the :py:class:`Wrapper` initiates a termination and restart
procedure across all ranks to ensure a consistent recovery process:

- the :py:class:`Wrapper` calls an instance of
  :py:class:`nvidia_resiliency_ext.inprocess.abort.Abort` from a separate Python thread; by default,
  this operation is equivalent to calling
  :py:func:`torch.distributed.destroy_process_group()`,

- next the :py:class:`Wrapper` raises asynchronous Python exception within the
  wrapped function; this exception interrupts the execution of the wrapped
  function, allowing control to return to the :py:class:`Wrapper` which then
  handles the restart process

The termination mechanism respects regular Python exception propagation logic,
and gives the wrapped function an opportunity to properly clean up resources by
calling all encountered exception handlers, context managers' ``__exit__``
methods etc. The restart exception raised by the :py:class:`Wrapper` is a
direct subclass of Python :py:exc:`BaseException` and it is required that the
wrapped function propagates this exception to the outer function scope.

The termination procedure runs in a separate Python thread. In some cases, the
main thread - unblocked by the destruction of the distributed process group -
might execute a few additional Python bytecode instructions before the
asynchronous exception is received. In most cases, it should be harmless as the
wrapped function is about to be interrupted and restarted, but the wrapped
function must not execute any code that may corrupt persistent storage and
prevent correct execution after a restart (e.g. the function cannot write
checkpoint to persistent storage). To protect against this possible data
corruption, the :py:class:`Wrapper` offers
:py:meth:`inprocess.CallWrapper.atomic` context manager, which implements a
lock shared by the main thread and the thread performing the termination
procedure. The termination procedure won't be launched if the main thread is in
:py:meth:`inprocess.CallWrapper.atomic` code block, and the main thread won't
enter into :py:meth:`inprocess.CallWrapper.atomic` code block if termination
procedure is already in progress. The use of the
:py:meth:`inprocess.CallWrapper.atomic` context manager is optional, and may be
omitted if the workload already includes mechanisms to guarantee that the
restarted wrapped function does not resume execution from a corrupted or
incomplete persistent state (e.g., a compromised checkpoint).


Progress timeout
~~~~~~~~~~~~~~~~
The :py:class:`Wrapper` implements two types of timeout events:

Soft timeout
^^^^^^^^^^^^
Soft timeout is equivalent to a Python exception raised by one of the
ranks, and triggers an attempt to restart the wrapped function on all healthy
ranks.

.. _hard_timeout:

Hard timeout
^^^^^^^^^^^^
The hard timeout mechanism forcefully terminates the main Python interpreter
process by sending a sequence of signals to ensure proper shutdown.

Initially, the :py:class:`Wrapper` sends the signals (``SIGCONT``, ``SIGTERM``)
to allow for a graceful shutdown. If the process remains active after this
step, a second sequence of signals (``SIGCONT``, ``SIGTERM``, ``SIGKILL``) is
sent after a delay specified by the ``termination_grace_time`` parameter. This
guarantees termination of the process if it fails to respond to the initial
signals.

The ``termination_grace_time`` parameter, configurable via :py:class:`Wrapper`,
defines the time interval between the two signal sequences. If the workload
implements ``SIGTERM`` cleanup handlers and their execution is critical for
successfully restarting the wrapped function, ``termination_grace_time`` should
be adjusted to allow sufficient time for these handlers to complete.

For workloads that do not implement ``SIGTERM`` handlers, it is safe to set
``termination_grace_time`` to 0 seconds to enable faster termination in cases
where the process hangs. This minimizes restart latency while ensuring the
process is terminated promptly.

.. _reporting_progress:

Reporting progress
^^^^^^^^^^^^^^^^^^
Timeout events are triggered when the wrapped function didn't report progress
in the specified timeout interval.

There are two methods to record progress:

- Automatic heartbeat: the :py:class:`Wrapper` periodically checks if the main
  thread of the Python interpreter keeps executing new bytecode instructions;

  - this method is always active and protects against hangs in calls that block
    Python interpreter, even in case when a blocking call released GIL,

  - it doesn't protect against while-true-like livelocks, where the interpreter
    keeps executing new bytecode instructions but doesn't make meaningful
    forward progress

- Manual heartbeat (optional): the wrapped function can optionally report
  progress by periodically calling the :py:meth:`inprocess.CallWrapper.ping`
  method:

  - the :py:class:`nvidia_resiliency_ext.inprocess.Wrapper` inspects the signature of the wrapped
    function for an argument annotated with the type
    :py:class:`nvidia_resiliency_ext.inprocess.CallWrapper`,

  - if such an argument is present, the :py:class:`Wrapper` injects an instance
    of :py:class:`nvidia_resiliency_ext.inprocess.CallWrapper` into the function, enabling it to call
    :py:meth:`inprocess.CallWrapper.ping` within its scope,

  - the timeout for the manual heartbeat is activated after the first call to
    the :py:meth:`inprocess.CallWrapper.ping` method.

Timeout event is triggered if either of the active progress monitoring methods
didn't record a heartbeat in the specified time interval.

Finalize
~~~~~~~~
The :py:class:`Wrapper` accepts optional, user-provided
:py:class:`nvidia_resiliency_ext.inprocess.finalize.Finalize` class. :py:class:`Finalize` class is
executed after a fault was detected, distributed group was destroyed, but
before the :py:class:`HealthCheck` is performed. :py:class:`Finalize` should
bring the process into a state where a restart of the wrapped function may be
attempted, e.g.: deinitialize any global variables or synchronize with any
async work issued by the wrapped function that was not already performed by
exception handlers in the wrapped function. Any failure during the execution of
:py:class:`Finalize` should raise an exception, in this case the health check
is skipped, exception is reraised by the :py:class:`Wrapper`, and the exception
should cause termination of the main Python interpreter process.

Multiple finalizers could be composed with :py:class:`nvidia_resiliency_ext.inprocess.Compose`.

Health check
~~~~~~~~~~~~
The :py:class:`Wrapper` calls optional, user-provided
:py:class:`nvidia_resiliency_ext.inprocess.health_check.HealthCheck` class before the restart to
ensure that the worker is in a healthy state. :py:class:`HealthCheck` is
executed after the wrapped function failure was discovered (on local or remote
distributed rank), local distributed group was destroyed, and the optional
:py:class:`Finalize` finished execution. The execution of the health check is
local to each rank that could potentially participate in a job after restart,
and it is meant to filter out unhealthy ranks that cannot continue executing
the workload (e.g. corrupted CUDA context). The execution should be local to
the calling rank, other ranks may have already been terminated, lost or still
executing the wrapped function. An unhealthy state is reported to
:py:class:`nvidia_resiliency_ext.inprocess.Wrapper` by raising an exception from
:py:meth:`inprocess.health_check.HealthCheck.__call__` method. The exception is
then reraised by the :py:class:`Wrapper`, and should cause termination of the
main Python interpreter process on the local rank.

Multiple health checks could be composed with :py:class:`nvidia_resiliency_ext.inprocess.Compose`.

Monitoring capabilities
~~~~~~~~~~~~~~~~~~~~~~~
The :py:class:`Wrapper` provides several monitoring mechanisms to track the
workload's progress and enable rapid restart capabilities in the event of a
fault.

.. _monitor_thread:

Monitor Thread
^^^^^^^^^^^^^^
The Monitor Thread runs as a separate :py:class:`threading.Thread` and is
tasked with periodically checking the distributed store for any faults reported
by other distributed ranks. It also ensures that the local rank is
:ref:`reporting progress <reporting_progress>`. If a fault or a lack of
progress is detected, it triggers :py:class:`nvidia_resiliency_ext.inprocess.abort.Abort` and raises
asynchronous Python exception within the wrapped function.

The execution interval of the monitoring loop is governed by the
``monitor_thread_interval`` parameter of the :py:class:`Wrapper`. During each
loop iteration, the thread queries the distributed store by invoking
:py:meth:`torch.distributed.Store.get`. For workloads with a large number of
distributed workers, it may be necessary to increase the
``monitor_thread_interval`` to avoid creating a communication bottleneck in the
distributed store caused by concurrent queries from multiple workers.

Monitor Process
^^^^^^^^^^^^^^^
The Monitor Process operates as a separate daemon process created by the
:py:class:`Wrapper`. Its responsibilities include ensuring the main workload
process remains active, submitting heartbeat signals to the distributed store
for the local rank, monitoring heartbeat signals from remote ranks, and
terminating the main process if it becomes unresponsive and irrecoverable.

The timeout for receiving a heartbeat from other distributed ranks is
configured with ``heartbeat_timeout`` parameter of the :py:class:`Wrapper`. If
any of the distributed rank doesn't submit a heartbeat within
``heartbeat_timeout`` interval, the rank is considered unresponsive, and a
restart is triggered on all distributed ranks.

The execution interval of the monitoring loop is governed by the
``monitor_process_interval`` parameter of the :py:class:`Wrapper`. Similar to
the :ref:`Monitor Thread <monitor_thread>`, each iteration of the loop queries
the distributed store. To prevent communication bottlenecks in the distributed
store, the monitoring interval should scale proportionally with the number of
distributed workers to avoid creating a communication bottleneck.

Progress Watchdog
^^^^^^^^^^^^^^^^^
The Progress Watchdog runs as a separate :py:class:`threading.Thread` and is
responsible for issuing automatic heartbeats to check if the main thread of the
Python interpreter keeps executing new bytecode instructions and receiving,
optional, manual heartbeats from the workload to track its progress. Refer to
:ref:`Reporting progress <reporting_progress>` for more details about automatic
and manual heartbeats.

The execution interval is governed by the ``progress_watchdog_interval``
parameter of the :py:class:`Wrapper`. The execution involves only the
node-local inter-process communication, and the interval does not need to be
scaled with the number of distributed workers.

Logging
~~~~~~~
The :py:class:`Wrapper` leverages the Python logging module to output messages.
It does not adhere to the conventional methods of fully integrating with an
application's root logger. Instead, logging from :py:class:`Wrapper` within the
main process is managed through a :py:class:`logging.StreamHandler`, which is
defined by the first ancestor in the logger hierarchy. Notably, the logging in
:py:class:`Wrapper` is configured to not store logs in files, and to not
`propagate
<https://docs.python.org/3/library/logging.html#logging.Logger.propagate>`_
logging messages to the ancestor loggers' handlers.

Logging with :py:obj:`logging.DEBUG` level shows the location where the wrapped
function suppressed the :py:exc:`BaseException` raised asynchronously by the
:py:class:`Wrapper`. The restart logic requires that BaseExceptions are
propagated from the wrapped function to the outer scope. This feature helps to
find locations where this assumption is not met, and the restart flow is
interrupted.

For the monitoring daemon process, logging is handled differently; logs are
written only to a file. The location of this log file is configurable. Users
can specify a custom path by passing a string to the
``monitor_process_logfile`` argument. This string may include the ``{rank}``
placeholder, which allows for dynamic filename generation based on the initial
distributed rank of the calling process.

Restart latency
---------------
Restart latency refers to the time elapsed between a fault occurring on any
distributed rank and successfully relaunching the wrapped function across all
distributed ranks.

The following table summarizes the latencies of all major items contributing to
the total restart latency. Rows marked with ``(H)`` increase restart latency
only when the application hangs. These items are not included if the
application raises a Python exception on any distributed rank.

+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| Category  | Item                                                   | Latency                                                                      |
+===========+========================================================+==============================================================================+
| NCCL/PyT  | :py:func:`torch.distributed.destroy_process_group()`   | ~0.5s + 0.01s * num pending NCCL kernels                                     |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| CUDA/user | complete pending CUDA kernels                          | ~training iteration                                                          |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| Wrapper   | wait for concurrent faults on other ranks              | ``last_call_wait``                                                           |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| Wrapper   | execute ``rank_assignment``                            | ~0.5s                                                                        |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| Wrapper   | TCPStore-based barrier                                 | 0.5s @ 16k ranks                                                             |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| user      | execute user-provided ``initialize``                   | N/A                                                                          |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| user      | execute user-provided ``finalize``                     | N/A                                                                          |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| user      | execute user-provided ``health_check``                 | N/A                                                                          |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| Wrapper   | ``(H)`` detect GIL-released hang                       | ``soft_timeout`` + ``monitor_process_interval``                              |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+
| Wrapper   | ``(H)`` detect GIL-holding hang                        | ``hard_timeout`` + ``monitor_process_interval`` + ``termination_grace_time`` |
+-----------+--------------------------------------------------------+------------------------------------------------------------------------------+

The latency for executing :py:func:`torch.distributed.destroy_process_group`
assumes that NCCL collective kernel termination interval was optimized. See
:ref:`Known issues <known_issues>` for more details. The latency for completing
all pending CUDA kernels assumes that the training loop performs
synchronization with the GPU at least once per training iteration.

.. _known_issues:

Known issues
------------

PyTorch
~~~~~~~
#. :py:class:`torch.distributed.ProcessGroupGloo` doesn't offer ``_shutdown()``
   method to terminate pending Gloo collectives (`pytorch/#130345
   <https://github.com/pytorch/pytorch/issues/130345>`_); if a rank
   participating in a Gloo collective stops making forward progress, the
   remaining ranks would wait till :py:class:`ProcessGroupGloo` timeout is
   exceeded; a workaround is to specify a short timeout for the ``gloo``
   backend to enable faster restarts.

#. The :py:class:`nvidia_resiliency_ext.inprocess.Wrapper` class uses
   :py:meth:`torch.distributed.Store.wait` to detect events in the distributed
   key-value store within its monitoring loops. Because these loops often
   advance to the next iteration after an expected timeout, PyTorch emits a
   warning every time :py:meth:`wait` times out, cluttering the output. To
   suppress these warnings, set the ``TORCH_CPP_LOG_LEVEL`` environment
   variable to ``error`` or ``fatal`` before importing ``torch``.

#. :py:class:`nvidia_resiliency_ext.inprocess.Wrapper` is not fully compatible with
   :py:func:`torch.distributed.run`. :py:func:`torch.distributed.run`
   automatically terminates all worker processes if any one of them fails, in
   this case :py:class:`nvidia_resiliency_ext.inprocess.Wrapper` can only recover from transient
   faults that don't cause termination of worker processes.

#. By default, PyTorch NCCL Watchdog forcefully terminates the process if NCCL
   call returns an error, or if CUDA context was corrupted. Forceful
   termination of the worker process prevents :py:class:`nvidia_resiliency_ext.inprocess.Wrapper`
   from restarting the wrapper function. A workaround is to set
   ``TORCH_NCCL_RETHROW_CUDA_ERRORS`` environment variable to ``0``, to avoid
   rethrowing CUDA and NCCL errors in PyTorch NCCL Watchdog.

#. PyTorch pairwise distributed process groups for P2P communication using
   :py:func:`torch.distributed.send`, :py:func:`torch.distributed.recv` (and
   similar functions) need to be created and initialized explicitly at the
   Python level with :py:func:`torch.distributed.new_group`. Aborting PyTorch
   NCCL backend with implicitly created P2P communicators may lead to hangs if
   PyTorch doesn't contain the fix implemented in `pytorch/#150690
   <https://github.com/pytorch/pytorch/pull/150690>`_.

#. PyTorch may raise segmentation fault if distributed backend is aborted while
   the first iteration of a backward pass is in progress (`pytorch/#149418
   <https://github.com/pytorch/pytorch/issues/149418>`_).

NCCL
~~~~
#. Support for NVLink SHARP (NVLS) in NCCL must be disabled by setting the
   ``NCCL_NVLS_ENABLE`` environment variable to ``0``.

CUDA
~~~~
#. To perform a restart, the :py:class:`nvidia_resiliency_ext.inprocess.Wrapper` needs to wait for
   completion of all executing and pending CUDA kernels. This is implemented
   with a GPU synchronization, and is a part of
   :py:class:`nvidia_resiliency_ext.inprocess.health_check.CudaHealthCheck`. Waiting for CUDA kernels
   to complete could increase the restart latency if many CUDA kernels are
   pending execution. A workaround is to periodically synchronize with the GPU
   from the wrapped function to reduce the depth of pending kernels queue.
