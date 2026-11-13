Multinode deployment
====================

Launch shape
-------------

Start one ``ft_launcher`` per node and let it spawn the local GPU workers.
Under Slurm use ``srun --ntasks=<nodes> --ntasks-per-node=1``; a task per GPU
would start multiple launchers on each node and is rejected by the launcher.
All nodes need the same reachable endpoint, run ID, environment, checkpoint
path, and active training shape. ``localhost:0`` is for single-node runs only.

FT options that take a boolean value accept ``true`` / ``false``, ``1`` / ``0``, and
``yes`` / ``no`` (case-insensitive). Invalid values fail argument parsing before
services start. Omitting an option preserves its existing configuration default.

Control-host process ownership
------------------------------

The public command remains ``ft_launcher`` on every node. Existing ``sbatch`` /
``srun`` commands do not need a wrapper or another task. On the rendezvous host,
``ft_launcher`` supervises separate control and compute processes:

.. code-block:: text

   Control host                          Other nodes
   ft_launcher (supervisor)              ft_launcher (runs the node agent directly)
   +-- job-controller                    +-- training workers
   +-- node-agent                        +-- rank monitors / local helpers
   |   +-- training workers
   |   +-- rank monitors / local helpers
   +-- gRPC root / leaves (if enabled)
   +-- managed attribution (if enabled)

The host is selected from the rendezvous endpoint, honoring the existing
``is_host`` override; it is not selected from the training rank. The control-host
agent joins the same compute rendezvous as agents on the other nodes. ``job-controller``
owns the TCPStore, rendezvous rounds, membership decisions, attribution client,
and existing cycle-info reporter. The agent owns worker restart cycles and
node-local helpers. Both roles use the effective ``--ft-segment`` setting.
Other nodes retain their existing process and logging path.

The job-controller initializes its own optional telemetry provider using inherited
resource attributes and a distinct controller identity. Its attribution poller
uses the control host's node identity. The poller stops before bounded telemetry
shutdown. Rank monitors and logging services retain their ordinary log output.

The supervisor starts its children without a readiness handshake. Agents wait
for the configured TCPStore endpoint and log clients retry their connections.
In a colocated deployment, both ``job-controller`` and ``node-agent`` must exit
zero for successful completion; either may finish first. The supervisor keeps
monitoring until both finish, a child fails, or cancellation is requested.
A failure in either child, including an unhealthy-node
exception, fails the supervisor. Node/rack loss remains a critical
colocated-control failure.

Until service auto-restart is implemented, any auxiliary service that exits
unexpectedly also fails the supervisor, including an unexpected zero exit from
a continuously running service. The supervisor returns the first observed
failure's exit code; a signal termination maps to ``128 + signal``. A signal
exception reaching a launcher's top-level handler uses this same mapping on
both control and compute hosts (for example, SIGTERM returns ``143``). The controller
returns the configured ``--ft-no-restart-exit-code`` (default ``93``) for a
no-restart decision. These results flow to the existing ``srun`` task and caller's
generation-cleanup logic. Ordinary training-worker restarts remain inside the
node agent. The intended singleton policy is to cancel only the current array
when the control host exits. Queued singleton successors remain eligible to
start after ``0``, ``93``, or another exit code. The launcher does not cancel
those successors. The example script currently adds chain cancellation through
``control/no_restart``; correcting that existing example policy is outside this
control-host refactor.

On SIGTERM, SIGINT, SIGHUP, or SIGQUIT, the supervisor disables restarts and signals
its critical children with SIGTERM. It waits for their cleanup, stops managed
attribution, then stops the gRPC aggregation funnel and finally closes the local
control log. The aggregators
remain alive while the node agent drains its final logs. Children exceeding
their shutdown deadline receive SIGKILL. The supervisor allows the node-agent
``--term-timeout`` plus 15 seconds for monitor cleanup and log draining. The
additional allowance is best effort and must fit within the scheduler's
termination window, together with aggregator and control-log shutdown.
The final child wait is also bounded;
survivors are logged and cleanup continues. Descendant discovery uses one local
PID/parent snapshot with a best-effort time budget, followed by an in-memory
walk; it does not repeatedly scan process trees. These checks cannot interrupt a
kernel call that itself stalls. If the supervisor itself receives
SIGKILL, it cannot perform cleanup; Slurm must terminate the remaining job-step
processes, including descendants that created new sessions.

Two differences from the previous colocated launcher are accepted:

* **Peer exit codes can differ.** The job-controller keeps TCPStore alive for a
  three-second grace after its rendezvous loop ends. A delayed peer can lose its
  store connection and exit ``1`` even when the control host returns ``93``.
  This does not change the intended singleton-generation handoff. Future per-cycle
  ``end_reason`` reporting will consolidate these outcomes; matching exit codes
  from every node are not required.
* **Attribution STOP remains active during successful completion.** The
  job-controller enforces a STOP until graceful rendezvous shutdown is recorded.
  A node-agent whose local workers have succeeded still waits in the existing
  exit barrier before recording that shutdown. A STOP during this interval can
  end the generation with ``93``, even if the local agent would otherwise finish
  successfully, and can stop peers that are still finishing. The previous
  colocated launcher stopped checking attribution after detecting local success.
  This is an accepted policy change; no additional completion marker is used.

``--standalone`` is a single-node development mode with embedded control and
local logging. It does not start a supervisor or gRPC aggregators, even if
``--ft-enable-log-server`` is set; configured application and launcher logs are
written locally. The supervised deployment requires a fixed,
nonzero rendezvous port. No lifecycle/status IPC or additional public launcher
command is required. The job-controller is an internal supervisor-owned child;
external control uses the same supervisor through ``--ft-control-only``. Service auto-restart, reconnection, and state recovery
remain separate work; a separate cycle-analyzer process is not started here.

External CPU control host
--------------------------

Run ``ft_launcher --ft-control-only`` on the external CPU host. It starts the
same supervisor, job-controller, configured gRPC aggregators, and managed
attribution service, without a node-agent, training workers, or rank monitors.
No training command is required. ``--nnodes`` counts only compute nodes; the
control host does not participate in the compute rendezvous. Compute nodes
continue to run ``ft_launcher`` with the control endpoint and ``is_host=false``.

.. code-block:: text

   External CPU host                     Compute nodes
   ft_launcher (supervisor)               ft_launcher (node agent)
   +-- job-controller                    +-- training workers
   +-- gRPC root / leaves (if enabled)    +-- rank monitors / local helpers
   +-- managed attribution (if enabled)

For example, with training job ID ``12345`` and a reachable CPU host named
``cpu-control``:

.. code-block:: bash

   # On the CPU control host; no training command or GPU worker configuration.
   ft_launcher --ft-control-only --nnodes=64 \
       --rdzv-endpoint=cpu-control:29500 --rdzv-id=training-12345 \
       --rdzv-conf=is_host=true \
       --ft-nvrx-logfile=/lustre/job/nvrx.log \
       --ft-per-cycle-applog-prefix=/lustre/job/train.log \
       --ft-cycle-info-dir=/lustre/job/cycle-info \
       --ft-cycle-info-job-id=12345

   # On each compute node, using its existing scheduler launch arrangement.
   ft_launcher --nnodes=64 --nproc-per-node=8 \
       --rdzv-endpoint=cpu-control:29500 --rdzv-id=training-12345 \
       --rdzv-conf=is_host=false \
       --ft-nvrx-logfile=/lustre/job/nvrx.log \
       --ft-per-cycle-applog-prefix=/lustre/job/train.log \
       --ft-cycle-info-dir=/lustre/job/cycle-info train.py

``--ft-control-only`` requires a configured endpoint owned by that invocation,
and cannot be combined with ``--standalone`` or a training command. No separate
controller CLI or service-startup branch is used. Successful completion requires
only the job-controller to exit zero; auxiliary-service failures remain fatal.

When cycle-info output is enabled, ``--ft-cycle-info-job-id`` is required on the
control-only invocation. Supply the compute allocation's effective job ID
(``SLURM_ARRAY_JOB_ID`` when present, otherwise ``SLURM_JOB_ID``), which compute
agents use for ``NVRX_CURRENT_CYCLE_INFO``. The external controller preserves the
previous attempt-index convention of ``0``, regardless of the CPU allocation's
``SLURM_RESTART_COUNT``. Matching rendezvous IDs alone does not align these file
names. Colocated deployment continues to derive job and attempt identity from
its environment.

The CPU host may belong to the same Slurm job as the compute nodes or a separate
allocation. The enclosing batch script or orchestrator remains responsible for
observing the supervisor's exit status and ending the training generation when
control fails. This launcher does not cancel a separate allocation automatically.
The existing shutdown grace allows remote compute logs to drain through the
aggregators before they stop. Existing colocated launch commands are unchanged.

Fixed training world
---------------------

``--nnodes MIN:MAX`` selects an active training world of ``MIN`` nodes and
admits additional joined nodes as standbys, up to ``MAX``. With eight GPUs per
node, ``6:8`` means 48 active workers, not a variable 48-to-64-worker model.
For Megatron, preserve TP/PP/CP/EP, optimizer/checkpoint compatibility, and a
global batch divisible by microbatch size times DP. Promotions replace capacity
without resizing the model. See :doc:`examples/megatron_gpt` for TP/PP commands.

Replacement groups and rank assignment
--------------------------------------

Array tasks can contain multiple nodes. A replacement group represents that
allocation unit: all required nodes in a group must be healthy/present before
it becomes active. The size defaults to ``SLURM_NNODES`` or
``SLURM_JOB_NUM_NODES`` in arrays and can be specified through rendezvous
``replacement_group_size``. ``NVRX_REPLACEMENT_GROUP_ID`` overrides the default
``SLURM_ARRAY_TASK_ID`` identity. Segment size and replacement-group size impose
different constraints: the first is NVLink locality, the second scheduler
allocation completeness.

Infrastructure ordering uses these sources in precedence order:

* Optional node-name ordering through ``NVRX_INFRA_RANK_FROM_NODENAME``.
* Explicit cross-job ordering through ``CROSS_SLURM_PROCID``.
* Block/node topology ordering when ``SLURM_TOPOLOGY_ADDR_PATTERN=block.node``.
* ``SLURM_PROCID``, incorporating array task ID and nodes per array task to
  obtain unique infrastructure ranks across the array.
* ``GROUP_RANK``, then deterministic sorted node descriptors as fallback.

Some allocation-boundary checks deliberately use array/node ordering rather
than the optional topology ordering. Infrastructure rank, replacement-group
identity, launcher group rank, and worker global rank are different values.
Do not assume a worker rank identifies the same physical GPU after restart.
Use the failed cycle's rank map for diagnosis.

Hot spares and NVLink segments
------------------------------

Simple hot-spare mode is the default: **omit** ``--ft-segment``. It does not
require GPU ClusterUUID. The first eligible ``MIN`` nodes form the active world;
additional joined nodes wait with no training workers until promotion.

On systems where ``nvidia-smi -q`` reports the required GPU ClusterUUID,
``--ft-segment N`` selects complete N-node segments within NVLink domains.
Only domains containing at least N eligible nodes can contribute a segment,
nodes in a segment get contiguous group ranks, and ``MIN % N`` must be zero.
``--ft-segment 1`` still requires ClusterUUID; it is not equivalent to omitting
the option on a system without that property. Do not pass the string ``None``
or use zero to disable an integer segment option.

For eight allocated nodes, six active with eight workers each, run the same
command on every node with a shared endpoint and run ID:

.. code-block:: bash

   ft_launcher --nnodes=6:8 --nproc-per-node=8 \
     --rdzv-endpoint="$RDZV_HOST:29400" --rdzv-id="$RUN_ID" \
     --max-restarts=3 training_script.py

For twelve allocated nodes, eight active in complete four-node NVLink segments:

.. code-block:: bash

   ft_launcher --nnodes=8:12 --nproc-per-node=8 \
     --rdzv-endpoint="$RDZV_HOST:29400" --rdzv-id="$RUN_ID" \
     --ft-segment=4 --max-restarts=3 training_script.py

These show allocation shape only. Add workload instrumentation, matching finite
timeouts, and checkpoint restoration as in :doc:`examples/megatron_gpt`.
The number of usable spares depends on complete segments/replacement groups,
not just the count of physically healthy nodes.

Scheduler continuation
-----------------------

The colocated design loses its rendezvous service when the store host dies.
The :doc:`examples/singleton_deployment` script chains independent Slurm array
generations, each with a new host. A singleton dependency serializes generations;
it does not reserve resources or promise immediate scheduling. The array throttle
is a concurrency ceiling, not all-or-none initial admission. Failed array units
can be released for repair while other units continue; a normal fixed allocation
does not provide the same independently releasable replacement behavior.


NUMA binding
------------

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
