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

Details:

* ``--ft-node-health-check-endpoint`` (alias: ``--ft-node_health_check_endpoint``) sets the optional node health check service endpoint used by InJob.
  Accepts Unix domain socket (UDS): ``/var/run/nvhcd.sock`` or ``unix:///var/run/nvhcd.sock``.
  See `Node health check service`_ for the BCM-backed and compatible-service usage model.

If ``--max-restarts`` is specified, the launcher restarts failed workers.
The ``--ft-restart-policy`` parameter is deprecated; only ``any-failed`` is supported: all workers
are restarted if any worker fails (torchrun-style behavior). This option may be removed in a future release.

Node health check service
^^^^^^^^^^^^^^^^^^^^^^^^^

The launcher can query an optional node-local health check service before workers enter
rendezvous. A practical public deployment model is to reuse NVIDIA Base Command
Manager (BCM) Slurm prolog or epilog health checks behind an ``nvhcd``-compatible
daemon. The NVRx integration point is service-compatible: any equivalent daemon
can be used if it implements the expected gRPC API over a Unix domain socket
(UDS).

To enable the external node health check with BCM:

* Build and deploy an ``nvhcd``-compatible daemon on every allocated node.
* Configure the daemon to invoke a BCM health check script, or a wrapper around
  an existing BCM prolog or epilog health check.
* Ensure the wrapper translates the BCM result into JSON with ``fail_count == 0``
  for a healthy node and a nonzero ``fail_count`` for an unhealthy node.
* Make the daemon's UDS visible from the job environment or training container.
* Pass the socket path to ``ft_launcher`` with ``--ft-node-health-check-endpoint``
  (alias: ``--ft-node_health_check_endpoint``).

For protocol details, see the ``nvhcd`` protobuf schema at
``src/nvidia_resiliency_ext/shared_utils/proto/nvhcd.proto``. The functional test
server at ``tests/fault_tolerance/func/nodehc_service.py`` is a minimal example
of a UDS gRPC service that implements this API.

Example:

.. code-block:: bash

   ft_launcher \
     --ft-node-health-check-endpoint unix:///var/run/nvhcd.sock \
     train.py

Endpoint behavior:

* UDS endpoints are supported. The value can be a path such as ``/var/run/nvhcd.sock``
  or a ``unix://`` URI such as ``unix:///var/run/nvhcd.sock``.
* If the endpoint is omitted, NVRx skips the external node health check.
* If the gRPC client dependencies are unavailable, the UDS socket is missing, or a
  connectivity error occurs, NVRx treats the external check as unavailable and does
  not fail the job for that reason.
* Explicit failures reported by the service mark the node unhealthy.

Compatible service contract:

* The service must implement ``HealthCheckService.RunHealthCheck`` from the NVRx
  ``nvhcd`` protobuf API and listen on the configured UDS.
* NVRx calls the service with ``args=["--no-slurm"]``.
* The response must set ``success`` and return JSON in ``output``. A healthy node
  is reported with ``success=true`` and ``{"fail_count": 0}``.
* If ``success`` is false, ``fail_count`` is nonzero, or ``output`` cannot be parsed
  as JSON with a ``fail_count`` field, NVRx treats the node as unhealthy.

Example ``nvhcd`` configuration for BCM:

.. code-block:: yaml

   socket_path: /var/run/nvhcd.sock
   healthcheck_path: /usr/local/sbin/nvrx-bcm-healthcheck-wrapper.sh
   log_level: info
   timeout: 120

Start the daemon on each node with this configuration, for example as a node-level
service or directly with:

.. code-block:: bash

   nvhcd -config /etc/nvhcd/config.yaml

If the training job runs inside a container, bind mount the UDS path into the
container so that ``ft_launcher`` can reach the daemon.

The wrapper can call the same reusable health check entry point that BCM uses for
Slurm prolog or epilog validation, then normalize the result for NVRx. When using
``nvhcd``, the gRPC request ``args`` are forwarded to the configured
``healthcheck_path`` as command-line arguments, so NVRx's ``--no-slurm`` argument
will appear in the wrapper's ``"$@"``. For example:

.. code-block:: bash

   #!/usr/bin/env bash
   set -euo pipefail

   if /path/to/bcm-healthcheck "$@"; then
     printf '{"fail_count": 0, "failed_checks": []}\n'
     exit 0
   else
     printf '{"fail_count": 1, "failed_checks": ["bcm_healthcheck"]}\n'
     exit 1
   fi

Because NVRx invokes this check during rendezvous rather than during Slurm's
actual prolog or epilog phase, the wrapper should also provide any inputs that
the BCM script expects from the Slurm lifecycle environment, or call a reusable
health check entry point from the local BCM deployment that does not depend on
those lifecycle-only variables.

This lets NVRx run the same class of node validation during in-job restart
rendezvous that cluster administrators may already run at Slurm prolog or epilog
time.

Distributed storage health check (Lustre + NFS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The launcher can perform a distributed storage health check before rendezvous. 
By default it is disabled. When enabled (via CLI or YAML), it:

* Verifies Lustre health via ``/sys/fs/lustre/health_check`` (fails if not healthy).
* Discovers distributed mount targets and checks that each mount is reachable.

* ``--ft-enable-dist-storage-healthcheck`` (alias: ``--ft_enable_dist_storage_healthcheck``)
  - Accepts a boolean-like value only to enable the mount checks
    (e.g., ``--ft-enable-dist-storage-healthcheck true``).

Storage path health check
^^^^^^^^^^^^^^^^^^^^^^^^^

Validate specific absolute paths for existence and basic readability before rendezvous.

* CLI: ``--ft-storage-health-check-path`` (alias: ``--ft_storage_health_check_path``)
  - Accepts a comma-separated list of absolute paths (each starting with ``/``).
  - Example: ``--ft-storage-health-check-path '/data/checkpoints,/mnt/dataset'``
* YAML: ``storage_healthcheck_path`` under the ``fault_tolerance`` section

.. code-block:: yaml

   fault_tolerance:
     # Comma-separated absolute paths
     storage_healthcheck_path: "/data/checkpoints,/mnt/dataset"

Validation behavior:
  - Files: attempts to read a small block (up to 4KB)
  - Directories: lists directory contents
  - Other existing types (e.g., devices/symlinks): performs ``stat`` access


Attribution service integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-cycle application logs do not enable attribution by themselves. To enable attribution, set
``--ft-attribution-endpoint``. The endpoint value ``localhost`` makes ``ft_launcher`` run the
attribution service on the TCPStore host; other endpoints are treated as externally managed
attribution services.
External endpoints may use schemes such as ``http://``, ``grpc://``, or ``unix://``. The current
in-job attribution client submits logs over HTTP(S); non-HTTP endpoint strings are preserved but do
not add a new transport implementation.
If ``--ft-attribution-endpoint`` is set, ``--ft-per-cycle-applog-prefix`` is required because the
attribution service analyzes the per-cycle application logs.

The service code is included in the NVRx wheel, but the service dependencies are optional.
Install the wheel with the ``attribution`` extra before running a launcher-managed attribution
service:

.. code-block:: bash

   python -m pip install 'nvidia_resiliency_ext-<version>-<tags>.whl[attribution]'

Plain ``python -m pip install nvidia_resiliency_ext-*.whl`` does not install the attribution
service dependencies.

* CLI:

  - ``--ft-attribution-endpoint <ENDPOINT>`` (alias: ``--ft_attribution_endpoint``), default disabled
  - ``--ft-attribution-llm-api-key-file <PATH>`` (alias: ``--ft_attribution_llm_api_key_file``)
  - ``--ft-attribution-llm-base-url <URL>`` (alias: ``--ft_attribution_llm_base_url``)
  - ``--ft-attribution-llm-model <MODEL>`` (alias: ``--ft_attribution_llm_model``)
  - ``--ft-attribution-startup-timeout <SECONDS>`` (alias: ``--ft_attribution_startup_timeout``), default ``20``

  The managed attribution app-log directory is derived from
  ``dirname(realpath(--ft-per-cycle-applog-prefix))``. Its stdout/stderr log is derived
  from ``--ft-per-cycle-applog-prefix`` as ``*_attribution.log``. The managed service listens on
  ``127.0.0.1:50050`` and is exposed to the in-job client as ``http://localhost:50050``.

  The managed attribution API key must come from ``--ft-attribution-llm-api-key-file`` or inherited
  ``LLM_API_KEY_FILE``. If neither points to a readable file, the TCPStore-host launcher fails
  before starting the attribution service.

  Example:

  .. code-block:: bash

     ft_launcher \
       --ft-per-cycle-applog-prefix /lustre/job123/train.log \
       --ft-attribution-endpoint localhost \
       --ft-attribution-llm-api-key-file /secure/llm_api_key \
       --ft-attribution-llm-base-url https://integrate.api.nvidia.com/v1 \
       --ft-attribution-llm-model nvidia/nemotron-3-super-120b-a12b \
       train.py

  To use an externally managed attribution service instead, specify an explicit endpoint:

  .. code-block:: bash

     ft_launcher \
       --ft-per-cycle-applog-prefix /lustre/job123/train.log \
       --ft-attribution-endpoint http://attribution.service.internal:8000 \
       train.py

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

Per-cycle logging and gRPC log aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using consolidated per-cycle application logs (for example via ``--ft-per-cycle-applog-prefix``)
with optional gRPC log funneling (``--ft-enable-log-server``), worker and launcher output can be
merged through pipes and streamed to one or more aggregators on the rendezvous host before a single
writer appends to shared storage (for example Lustre).

.. important::

   **Best-effort semantics.** Per-cycle gRPC log aggregation is best-effort. Logs around failure
   and restart may be incomplete; crash stack traces are not guaranteed to appear there. For
   critical diagnostics, use rank monitor logs (launcher log) for failure/timeout correlation and
   OS-level core dumps for reliable crash post-mortem. Do not assume aggregated logs are complete
   or reliable.

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

NUMA binding
^^^^^^^^^^^^

The ``ft_launcher`` supports automatic NUMA node binding for workers through the ``NVRX_GPUS_PER_NUMA``
environment variable. When set, the launcher automatically wraps each worker process with ``numactl``
to bind it to the appropriate NUMA node based on its local rank.

.. important::
   **Prerequisites:** This feature requires the ``numactl`` command-line tool to be installed and
   available in the system PATH. The launcher will fail to start workers if ``numactl`` is not found.

   To install on common Linux distributions:

   * **Ubuntu/Debian:** ``sudo apt-get install numactl``
   * **RHEL/CentOS/Rocky:** ``sudo yum install numactl``

**How it works:**

* Set ``NVRX_GPUS_PER_NUMA`` to the number of GPUs per NUMA node on your system
* The launcher calculates the NUMA node as: ``numa_node = local_rank // gpus_per_numa``
* Each worker is automatically wrapped with: ``numactl --cpunodebind=<numa_node> --membind=<numa_node>``
* This applies only to binary/script entrypoints (not Python function entrypoints)

**Example usage:**

.. code-block:: bash

    # For a system with 4 GPUs per NUMA node (8 GPUs total, 2 NUMA nodes)
    export NVRX_GPUS_PER_NUMA=4
    ft_launcher --nproc-per-node=8 train.py

    # In this configuration:
    # - Ranks 0-3 will be bound to NUMA node 0
    # - Ranks 4-7 will be bound to NUMA node 1

**Benefits:**

Proper NUMA binding can significantly improve performance by ensuring memory locality
and reducing cross-NUMA memory access overhead, which is especially important for
multi-GPU training workloads.


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
