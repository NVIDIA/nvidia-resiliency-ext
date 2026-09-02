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


Segment Health Check
^^^^^^^^^^^^^^^^^^^^

Set ``--ft-segment-health-check-dir <ABSOLUTE_PATH>`` or
``fault_tolerance.segment_health_check_dir`` to consume segment health
decisions before rendezvous. Omitting the setting disables the check.

See the :doc:`Segment Health Check integration guide
<integration/segment_health_check>` for the consumer contract.


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

The service code and dependencies are included in the NVRx wheel. Install the wheel before running
a launcher-managed attribution service:

.. code-block:: bash

   python -m pip install nvidia_resiliency_ext-<version>-<tags>.whl

* CLI:

  - ``--ft-attribution-endpoint <ENDPOINT>`` (alias: ``--ft_attribution_endpoint``), default disabled
  - ``--ft-attribution-llm-api-key-file <PATH>`` (alias: ``--ft_attribution_llm_api_key_file``)
  - ``--ft-attribution-llm-base-url <URL>`` (alias: ``--ft_attribution_llm_base_url``)
  - ``--ft-attribution-llm-model <MODEL>`` (alias: ``--ft_attribution_llm_model``)
  - ``--ft-attribution-analysis-backend {lib}`` (alias: ``--ft_attribution_analysis_backend``)
  - ``--ft-attribution-stop-action {log,no-restart}`` (alias: ``--ft_attribution_stop_action``), default ``log``
  - ``--ft-attribution-startup-timeout <SECONDS>`` (alias: ``--ft_attribution_startup_timeout``), default ``20``
  - ``--ft-attribution-export-url <URL>`` (alias: ``--ft_attribution_export_url``)

  The managed attribution app-log directory is derived from
  ``dirname(realpath(--ft-per-cycle-applog-prefix))``. When ``--ft-nvrx-logfile`` is set, the
  managed service stdout/stderr log is derived from it as ``*_attribution.log`` and written
  beside it. Without ``--ft-nvrx-logfile``, the log is named from the application prefix and
  written under an ``nvrx/`` subdirectory of the application-log directory. The managed service
  listens on ``127.0.0.1:50050`` and is exposed to the in-job client as
  ``http://localhost:50050``.

  The managed attribution API key must come from ``--ft-attribution-llm-api-key-file`` or inherited
  ``LLM_API_KEY_FILE``. If neither points to a readable file, the TCPStore-host launcher fails
  before starting the attribution service.

  To export managed attribution results, pass ``--ft-attribution-export-url`` or
  set ``attribution_export_url`` in the fault tolerance YAML config.

  Launcher-managed attribution defaults to the ``lib`` backend, which runs the Restart Agent
  directly in the attrsvc process. The backend flag may be left unset to use that default or set
  explicitly to ``lib``. The optional MCP/LogSage analysis path is no longer selectable from
  launcher-managed attribution.

  Attribution never delays a restart. When a cycle ends, the rendezvous host requests
  terminal analysis and immediately closes the next round, so the workload restarts while
  analysis is still running. A background poller on the rendezvous host fetches the verdict
  and, if attribution recommends stopping, terminates the job at that point -- even if the
  workload has already advanced one or more cycles past the analyzed one. A verdict that
  never arrives leaves the job running.

  The stop decision is global, not per-cycle: the first STOP verdict observed ends the job
  regardless of which cycle produced it.

  Only one terminal analysis is in flight at a time. If further cycles fail while a verdict
  is still outstanding, their terminal analysis is skipped rather than replacing the one
  already running. This keeps a fast crash loop from repeatedly discarding an analysis
  before it can finish, which would otherwise leave the job with no verdict at all.

  **Acting on a STOP verdict is opt-in.** With the default
  ``--ft-attribution-stop-action log``, attribution runs in full and every verdict is
  polled, logged and recorded, but a STOP never terminates anything. Set
  ``--ft-attribution-stop-action no-restart`` to make a STOP end the job.

  The default is deliberate because the two kinds of attribution mistake are not equally
  expensive. A missed STOP is bounded: the job crash-loops until ``--max-restarts`` runs
  out, and the progress tracker independently catches the stuck case. An enforced false
  STOP is not bounded: the job ends with the no-restart exit code and stays down until a
  human resubmits it. Run in ``log`` mode first, review the recorded verdicts against what
  actually happened on your workloads, and enable ``no-restart`` once you trust the
  precision.

  In ``log`` mode every failed cycle is analyzed, not just the first one that produces a
  STOP, so the verdicts accumulate over the life of the job. Each unenforced STOP is logged
  at ``ERROR``, says explicitly that it is not being enforced, and carries a running count
  of how many STOP verdicts have been seen so far. Comparing that count against the number
  of failed cycles is the measurement that tells you whether to enable ``no-restart``.

  When enforcement is on, every launcher in the job exits with the no-restart exit code
  rather than the generic failure code ``1``. See :ref:`ft-no-restart` below.

  ``ft_launcher`` sends job metadata with each attribution submission: ``user`` is read from
  ``SLURM_JOB_USER`` or ``USER``, and ``job_id`` is read from ``SLURM_ARRAY_JOB_ID`` or
  ``SLURM_JOB_ID``. If no corresponding environment variable is set, that field is omitted from
  the submission payload.

  Example:

  .. code-block:: bash

     ft_launcher \
       --ft-per-cycle-applog-prefix /lustre/job123/train.log \
       --ft-attribution-endpoint localhost \
       --ft-attribution-llm-api-key-file /secure/llm_api_key \
       --ft-attribution-llm-base-url https://integrate.api.nvidia.com/v1 \
       --ft-attribution-llm-model nvidia/nemotron-3-super-120b-a12b \
       --ft-attribution-export-url https://dataflow.example.test/dataflow2/example-index/posting \
       train.py

  To use an externally managed attribution service instead, specify an explicit endpoint:

  .. code-block:: bash

     ft_launcher \
       --ft-per-cycle-applog-prefix /lustre/job123/train.log \
       --ft-attribution-endpoint http://attribution.service.internal:8000 \
       train.py

.. _ft-no-restart:

No restart: telling "do not requeue" apart from a failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two mechanisms let NVRx conclude that a job must not be restarted at all:

* an attribution ``STOP`` verdict, when ``--ft-attribution-stop-action no-restart`` is
  set (see the attribution section above; the default ``log`` never terminates), and
* the progress tracker finding no progress across restarts
  (``--ft-max-no-progress-cycles`` / ``max_no_progress_cycles``, default ``3``).

The progress tracker reads the iteration from ``--ft-checkpoint-iteration-file``; without
that file it stays inactive and logs a one-time warning. Raise the cycle threshold, or set
it to ``0`` to disable tracking, for workloads that deliberately replay an iteration
without advancing the checkpoint -- for example Megatron-LM's rerun state machine
re-running a suspect result on different hardware to tell silent data corruption apart from
a genuine data issue. Those replays look identical to a stuck job from the outside.

Both answer the same scheduler-facing question, so both report the same result: every
launcher in the job exits with ``--ft-no-restart-exit-code`` (default ``93``) instead of
the generic failure code ``1``. Downstream tooling therefore needs a single rule -- exit
code ``93`` means NVRx decided, do not requeue -- rather than one rule per cause.

The specific cause is recorded in the rendezvous store shutdown key as ``attribution_stop``
or ``no_progress``, and is named in the launcher logs, so the two remain distinguishable
for diagnosis without complicating the scheduler contract.

Running out of restarts is deliberately *not* one of these cases. Exhausting
``--max-restarts`` means this launcher used up its retry budget, not that the job is
unrecoverable, so it still exits ``1`` and leaves requeueing to your scheduler policy.
The distinction is scope: a no-restart decision is a verdict about the job, while the
restart budget is a limit on a single allocation.

* CLI:

  - ``--ft-no-restart-exit-code <CODE>`` (alias: ``--ft_no_restart_exit_code``),
    default ``93``

Note that SLURM reports the maximum exit code across tasks and that ``sacct`` derived exit
codes are lossy, so treat this as a best-effort signal.

Nodes other than the one that made the decision read the reason from the rendezvous store,
so the store must outlive them. Both the store-host launcher and ``nvrx-control`` hold it
open for a few seconds after shutting down, but a node that has not read the key by then
exits ``1`` rather than the no-restart code. The job still stops either way.


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

Two write paths are possible:

* **With log funneling** (``--ft-enable-log-server``, enabled automatically when
  ``--ft-nvrx-logfile`` is set): every node streams to an aggregator on the rendezvous host and
  a *single* process writes the shared file. No cross-client write guarantees are needed, so
  this works on any shared filesystem and is the recommended deployment for multi-node jobs.
* **Without log funneling** (direct write): every node opens the same
  ``<prefix>_cycleN.log`` and appends to it with ``O_APPEND``. This is a multi-writer
  deployment and is not recommended on any filesystem. On Lustre the appends are atomic, but
  all nodes serialise on the same extent lock, so it does not scale. On NFS- and VAST-style
  storage the append is not atomic across clients at all, so concurrent appends can interleave
  or be lost and nothing reports an error.

.. important::

   **Use log funneling for multi-node jobs.** The recommended configuration is to pass
   ``--ft-nvrx-logfile`` alongside ``--ft-per-cycle-applog-prefix``; that enables
   ``--ft-enable-log-server`` automatically. Setting ``--ft-per-cycle-applog-prefix`` on its own
   leaves the log server off and selects multi-writer direct append, which is not a supported
   multi-node deployment. If you use neither option, worker and launcher output go to
   stdout/stderr and are captured by ``srun --output`` / ``--error`` as usual.

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
