Multinode deployment
====================

Launch shape and fixed training world
-------------------------------------

Start one ``ft_launcher`` per node and let it spawn the local GPU workers.
Under Slurm use ``srun --ntasks=<nodes> --ntasks-per-node=1``; a task per GPU
would start multiple launchers on each node and is rejected by the launcher.
All nodes need the same reachable endpoint, run ID, environment, checkpoint
path, and active training shape. ``localhost:0`` is for single-node runs only.

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

Scheduler continuation and external control
-------------------------------------------

The colocated design loses its rendezvous service when the store host dies.
The :doc:`examples/singleton_deployment` script chains independent Slurm array
generations, each with a new host. A singleton dependency serializes generations;
it does not reserve resources or promise immediate scheduling. The array throttle
is a concurrency ceiling, not all-or-none initial admission. Failed array units
can be released for repair while other units continue; a normal fixed allocation
does not provide the same independently releasable replacement behavior.

The experimental ``nvrx-control`` command can run the long-lived TCPStore and
control services outside compute launchers. Compute nodes connect to its endpoint
with rendezvous ``is_host=false``. The control process owns the service lifetime
and must survive compute failures. This changes the store-host failure boundary;
it does not make the control service itself highly available. This deployment
path is not yet QA-qualified and APIs may change. Keep it separate from the
recommended colocated walkthrough until qualified in your environment.

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
