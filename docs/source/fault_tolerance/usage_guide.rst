Usage guide
############

Terms
*****
* ``Fault Tolerance``, ``FT`` is the ``fault_tolerance`` package.
* ``FT callback``, ``FaultToleranceCallback`` is a PTL callback that integrates FT with PTL.
* ``ft_launcher`` is a launcher tool included in FT, which is based on ``torchrun``.
* ``heartbeat`` is a lightweight message sent from a rank to its rank monitor that indicates that a rank is alive.
* ``section`` is a user code segment with a custom name assigned.
* ``rank monitor`` is a special side process started by ``ft_launcher`` that monitors its rank.
* ``timeouts`` are time intervals used by a rank monitor to detect that a rank is not alive.
* ``launcher script`` is a bash script that invokes ``ft_launcher``.
* ``PTL`` is PyTorch Lightning.

Design Overview
***************

* Each node runs a single ``ft_launcher``.
* FT configuration is passed to ``ft_launcher`` and propagated to other FT components.
* ``ft_launcher`` spawns rank monitors (once).
* ``ft_launcher`` spawns ranks (can also respawn if ``--max-restarts`` is greater than 0).
* Each rank uses ``RankMonitorClient`` to connect to its monitor (``RankMonitorServer``).
* Each rank periodically sends updates to its rank monitor (e.g., during each training and evaluation step).
* In case of a hang, the rank monitor detects missing updates from its rank and terminates it.
* If any ranks disappear, ``ft_launcher`` detects that and terminates or restarts the workload.
* ``ft_launcher`` instances communicate via the ``torchrun`` "rendezvous" mechanism.
* Rank monitors do not communicate with each other.

.. code-block:: text

   # Processes structure on a single node.
   # NOTE: each rank has its own separate rank monitor.

   [Rank_N]----(IPC)----[Rank Monitor_N]
      |                      |
      |                      |
   (re/spawns)            (spawns)
      |                      |
      |                      |
   [ft_launcher]-------------


Usage Overview
**************

FT launcher
-----------

Fault tolerance includes a launcher tool called ``ft_launcher``, which is based on ``torchrun``
and supports most ``torchrun`` command-line parameters. FT configuration can be specified either
via a YAML file using ``--ft-cfg-path`` or through command-line parameters
using ``--ft-<parameter-name>``.

If ``--max-restarts`` is specified, the launcher restarts failed workers.
The restart behavior depends on the ``--ft-restart-policy`` parameter, which supports two modes:

* ``any-failed`` (default)  
  All workers are restarted if any worker fails.  

* ``min-healthy``
  Workers are restarted when the number of healthy nodes (nodes where all worker processes are running) 
  falls below the minimum specified in ``--nnodes``. This allows for some worker failures to be handled 
  without restarting remaining workers, e.g., with the :doc:`../inprocess/index`.
  For details on how ``min-healthy`` policy interacts with :doc:`../inprocess/index` see :doc:`integration/inprocess`.

Rank assignment
^^^^^^^^^^^^^^^

The ``ft_launcher`` assigns ranks to workers during the rendezvous process.

**Infrastructure-based assignment (default):**

By default (``--ft-use-infra-group-rank=True``), rank assignments **always** come from the infrastructure:

* The launcher first checks ``SLURM_PROCID`` (automatically set in SLURM environments)
* If not available, it falls back to ``GROUP_RANK`` (set by ``ft_launcher`` itself)

Infrastructure ranks are used for **every rendezvous**, including after failures/restarts. Previous 
rank assignments are ignored. This ensures consistency with the infrastructure's rank assignment,
which is important for static deployments and proper resource allocation.

.. note::
   Hot spare/redundancy is **NOT supported** with ``use_infra_group_rank=True`` because dynamic 
   rendezvous cannot guarantee that lower infrastructure ranks will join as participants first.

**Deterministic assignment (alternative):**

Set ``--ft-use-infra-group-rank=False`` (or ``use_infra_group_rank: false`` in config) to use 
deterministic sorted assignment based on node descriptors. In this mode:

* Previous rank assignments are preserved when possible
* New workers fill gaps left by failed workers
* Ranks are reassigned based on sorted node descriptors
  

Hang detection
--------------

The FT package provides two fully independent mechanisms for detecting hangs in user code.
Users can choose the API that is best suited for their needs, or use both APIs at the same time.

* Heartbeats API

The training script periodically sends `heartbeats` to the monitor. 
If no heartbeat arrives in a defined time, the workload is considered hung.
This API is the simplest to use but might require coarse timeouts 
that need to cover a wide range of possible intervals between heartbeats. 
Please find more details in :doc:`integration/heartbeats`.

* Sections API  

Some parts of the training scripts are wrapped in `sections`. 
If any section is opened for too long, the workload is considered hung.
The sections-based API requires more changes in the user code, but timeouts 
can be defined more precisely, and hangs can be detected quicker. 
Please find more details in :doc:`integration/sections`.

Workload control
----------------
In some cases, it might be useful to control the ``ft_launcher`` behavior based on a rank state. 
For example, if an irrecoverable error is encountered in a rank, it might be reasonable to break 
the launcher restarting loop and exit instead of restarting; for other exception types, one might 
want to exclude the current node from subsequent restart attempts. ``RankMonitorClient`` exposes the 
:meth:`nvidia_resiliency_ext.fault_tolerance.rank_monitor_client.RankMonitorClient.send_workload_control_request` 
API, which can be used to control the workload restarting logic implemented in the launcher.

.. note::
   Please note that only the ft_launcher behavior is affected by this call. 
   The fault tolerance package is job scheduler-agnostic, 
   i.e., it does not control underlying SLURM job allocations.

Non-retryable exceptions
^^^^^^^^^^^^^^^^^^^^^^^^

The ``--ft-non-retryable-exception-file`` option allows you to specify a file containing exception 
patterns that indicate errors that won't be fixed by retrying. When a worker fails with an exception 
matching any of these patterns, the node is immediately marked as unhealthy and will not be retried, 
preventing wasted time on errors that require manual intervention to fix.

**File format:**

The exception patterns file should contain one pattern per line:

* Each pattern is a case-sensitive substring that will be matched against the full exception text 
  (exception type, message, and traceback)
* Lines starting with ``#`` are treated as comments and ignored
* Empty lines are ignored
* Patterns are matched using simple substring matching (not regular expressions)

**Example patterns file:**

.. code-block:: text

   # Configuration errors that won't fix themselves
   insufficient shared memory (shm)
   Permission denied
   No such file or directory
   
   # Application errors
   CUDA out of memory
   ValueError: Invalid configuration
   
   # Resource exhaustion
   Cannot allocate memory
   Too many open files

**Usage example:**

.. code-block:: bash

   ft_launcher \
       --nnodes=2 \
       --nproc-per-node=8 \
       --max-restarts=5 \
       --ft-non-retryable-exception-file=/path/to/exception_patterns.txt \
       train.py

**Behavior:**

When a worker process fails:

1. The launcher checks the worker's error file against the configured exception patterns
2. If a match is found, the node is marked as unhealthy and the launcher exits on that node
3. The unhealthy count is incremented in the rendezvous store
4. Other nodes in the training job will be affected according to the configured restart policy

This feature is particularly useful for:

* **Configuration errors**: Insufficient shared memory, wrong permissions, missing files
* **Application errors**: Invalid parameters, unsupported configurations
* **Resource exhaustion**: Out of memory errors, file descriptor limits
* **Dependency issues**: Missing libraries, version incompatibilities

These types of errors require manual intervention (e.g., adjusting launch parameters, fixing 
configurations, or installing dependencies) and will not be resolved by simply restarting the worker.
