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
  

HTTP Status API
^^^^^^^^^^^^^^^

The ``ft_launcher`` provides a built-in HTTP server that exposes launcher status information via a REST API.
This feature is **enabled by default** and can be used to monitor launcher state during training.

**Configuration:**

* The HTTP server is enabled by default on port ``2025``
* To disable: use ``--no-http-server`` flag
* To change host: use ``--http-server-host`` (default: ``0.0.0.0``)
* To change port: use ``--http-server-port`` (default: ``2025``)

**Available endpoints:**

* ``GET /status`` - Returns launcher status information

  * ``start_time``: UTC timestamp when the launcher started (ISO 8601 format)
  * ``current_cycle``: Current failure detection and recovery cycle count

**Example usage:**

.. code-block:: bash

   # HTTP server enabled by default at http://0.0.0.0:2025/status
   ft_launcher --nnodes=1 --nproc-per-node=8 train.py

   # Query status
   curl http://localhost:2025/status

   # Disable HTTP server if not needed
   ft_launcher --no-http-server --nnodes=1 --nproc-per-node=8 train.py

   # Use custom port
   ft_launcher --http-server-port=8080 --nnodes=1 --nproc-per-node=8 train.py

**Example response:**

.. code-block:: json

   {
     "start_time": "2025-10-29T12:34:56.789012Z",
     "current_cycle": 0
   }

The ``current_cycle`` counter increments each time a failure is detected, making it useful for tracking
the number of fault recovery attempts during training.


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
