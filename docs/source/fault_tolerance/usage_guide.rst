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

GPU Memory Reclaim
^^^^^^^^^^^^^^^^^^

When ``--max-restarts`` is specified, ``ft_launcher`` can optionally wait for GPU memory to be
released before starting new workers after a restart. This helps ensure that GPU memory from
terminated workers has been fully reclaimed before starting new processes.

This feature is controlled by three parameters:

* ``--ft-gpu-memory-reclaim-timeout`` (default: 50.0 seconds)
  Timeout for waiting for GPU memory to drop below the tolerance threshold. Set to 0 to disable the feature.

* ``--ft-gpu-memory-tolerance-mb`` (default: 512.0 MB)
  Maximum allowed GPU memory usage. The launcher waits until GPU memory drops below this threshold.

* ``--ft-gpu-memory-poll-interval`` (default: 2.0 seconds)
  Poll interval for checking GPU memory usage during the reclaim process.

On restarts, the launcher periodically checks GPU memory usage and waits until it drops below
the tolerance threshold or the timeout is reached. Memory statistics for each GPU are collected
and logged after the reclaim process completes. If the timeout is reached, an error is logged but the
restart proceeds as a best effort.

Rank assignment
^^^^^^^^^^^^^^^

The ``ft_launcher`` assigns ranks to workers during the rendezvous process.

Rank assignments always use infrastructure-based ordering when available:

* The launcher first checks ``SLURM_PROCID`` (automatically set in SLURM environments)
* If not available, it falls back to ``GROUP_RANK`` (set by ``ft_launcher`` itself)
* If neither environment variable is set, ranks are assigned deterministically based on sorted node descriptors

This ensures consistency with the infrastructure's rank assignment, which is important 
for static deployments and proper resource allocation.

Hot Spare Nodes and Segment-Aware Rank Assignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ft_launcher`` supports hot spare nodes, which are standby nodes that can replace failed nodes
during restart. Hot spare functionality is always enabled and works with ``--max-restarts``.

By default (``--ft-segment=None``), the launcher uses **simple hot spare mode**, which is suitable 
for most deployments including H100-based systems where NVLink domain segmentation is not required:

* The first ``min_nodes`` (from ``--nnodes``) are assigned as active workers
* Any additional nodes beyond ``min_nodes`` become hot spares with standby ranks
* Hot spares do not require GPU ClusterUUID or NVLink domain awareness
* This mode effectively treats each node independently for rank assignment

For large-scale NVSwitch-based systems (e.g., DGX H200, HGX B200), you can enable 
**segment-aware hot spare mode** using ``--ft-segment=N``:

* ``N`` specifies the minimum number of nodes required per NVLink domain (identified by GPU ClusterUUID)
* Only domains with at least ``N`` nodes participate in training
* From each valid domain, as many complete segments as possible are selected
* Nodes in the same segment receive contiguous group ranks for optimal performance
* The ``min_nodes`` parameter (from ``--nnodes``) must be divisible by ``segment``
* GPU ClusterUUID is automatically queried via nvidia-smi to identify NVLink domains

**Key Differences:**

* ``--ft-segment=None`` (default): Simple mode without domain awareness, suitable for H100 systems
* ``--ft-segment=1``: Each node is a segment, similar to simple mode but requires ClusterUUID
* ``--ft-segment=4`` or higher: Multi-node segments for NVSwitch-based systems

Example for H100 deployment (8 nodes requested, 6 needed for training):

.. code-block:: bash

   ft_launcher --nnodes=6:8 --nproc-per-node=8 \
               --max-restarts=3 \
               training_script.py

   # Nodes 0-5: Active workers (ranks 0-47)
   # Nodes 6-7: Hot spares (standby ranks 48-63)

Example for NVSwitch deployment with segment=4 (12 nodes requested, 8 needed):

.. code-block:: bash

   ft_launcher --nnodes=8:12 --nproc-per-node=8 \
               --ft-segment=4 --max-restarts=3 \
               training_script.py

   # Requires domains with at least 4 nodes each
   # 8 active nodes = 2 complete segments
   # 4 hot spare nodes available for restart

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
