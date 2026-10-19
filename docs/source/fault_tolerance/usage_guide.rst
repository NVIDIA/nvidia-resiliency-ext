InJob restart usage guide
=========================

InJob restart detects failed or hung training workers and launches a
new worker group within the running job. Use ``ft_launcher`` instead of
``torchrun`` and enable the workload's section-monitoring integration.

The recommended Megatron-LM deployment uses **fixed section timeouts supplied
on the command line** and **NVRx-backed asynchronous checkpointing through
Megatron**. Begin with :doc:`examples/megatron_gpt`. For a small CPU/GPU example
of the monitoring API, use :doc:`examples/train_ddp_sections`, which retains
synchronous PyTorch checkpointing.

Recovery responsibilities
-------------------------

If a worker fails after checkpoint 20 while training iteration 27, the next
worker group loads checkpoint 20 and repeats subsequent training work.
NVRx does not preserve the failed workers' model or optimizer memory.

.. code-block:: text

   worker failure or section timeout
       -> terminate the worker group
       -> health checks and barrier rendezvous
       -> assign healthy active nodes (promote spares if available)
       -> launch new training processes
       -> workload loads the latest completed checkpoint
       -> training commits another optimizer step

* **NVRx InJob** monitors workers and coordinates termination, rendezvous, and
  process replacement. All workers restart when any worker fails.
* **The workload** saves and restores model, optimizer, RNG, scheduler, and data
  progress as required by its training algorithm. A restart without checkpoint
  restoration starts training again.
* **The scheduler and deployment scripts** allocate resources, release failed
  allocation units, admit cold replacements, and arrange continuation across
  jobs. ``ft_launcher`` does not allocate or cancel Slurm jobs.

Checkpoint recovery and monitoring are separate choices. The DDP example uses
rank-0 ``torch.save`` / ``torch.load``. Megatron assembles distributed training
state and uses NVRx's asynchronous checkpoint queue and writer infrastructure.
See :doc:`integration/megatron` for completion and finalization semantics.

Processes and terminology
-------------------------

Each compute node runs one ``ft_launcher``. It starts training workers and a
separate rank-monitor process for each local worker. Workers connect to their
monitors using local Unix sockets and report named section boundaries. Monitors
do not communicate with one another; launchers coordinate through rendezvous.

.. code-block:: text

   training worker <-- local IPC --> rank monitor
          ^                              ^
          | starts/restarts               | starts
          +--------- ft_launcher --------+
                          |
                   rendezvous store

A **cycle** is one worker-group attempt: cycle 0 is the initial attempt, cycle 1
the first restart. A **generation** in the singleton example is a separate
Slurm array allocation containing multiple possible cycles. A **hot spare** is
an allocated, joined launcher with no training workers. A **cold spare** is a
pending scheduler allocation and is not yet available to rendezvous.

The rendezvous/TCPStore host is a service role, distinct from training rank 0.
The singleton example colocates these roles on array task 0. Training ranks can
change across restarts; use the rank-to-node map from the failed cycle when
investigating a failure.

Launch and monitor
------------------

Launcher options precede the training entrypoint; workload options follow it.
For a workload that opens Megatron's section names, the essential shape is:

.. code-block:: bash

   ft_launcher \
     --nnodes=1 --nproc-per-node=2 \
     --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
     --max-restarts=3 \
     --ft-rank-section-timeouts setup:900,step:120,checkpointing:600 \
     --ft-rank-out-of-section-timeout=180 \
     /path/to/Megatron-LM/pretrain_gpt.py \
     --enable-ft-package ...

This illustrates argument placement; use the complete runnable example in
:doc:`examples/megatron_gpt` for model, data, and checkpoint arguments.

``--max-restarts`` defaults to **0**. Set a positive budget to enable recovery.
It counts restarts across the rendezvous cycles, including for replacement
launchers joining an existing run. Only the ``any-failed`` policy is supported;
omit the deprecated ``--ft-restart-policy`` option.

Barrier rendezvous is the only supported InJob rendezvous implementation.
``--ft-rdzv-impl`` defaults to ``barrier``; the removed ``legacy`` value is
rejected. For multinode runs, all launchers need the same reachable rendezvous
endpoint, run ID, and training shape. Do not use ``localhost:0`` across nodes.
Under Slurm launch **one task per node**, with ``ft_launcher`` starting the
local GPU workers. See :doc:`deployment`.

Configure the sections the workload actually opens. Megatron uses ``setup``,
``step``, and ``checkpointing``; the DDP example uses ``init``, ``step``, and
``checkpoint``. A name absent from ``rank_section_timeouts`` is ignored by the
client, including timing collection. A configured ``None``/``null`` timeout
has no deadline. Neither heartbeat nor section hang detection has finite
timeouts by default. Launching a workload alone does not instrument it.

Timeouts must cover the deployed model, initialization and checkpoint loading,
evaluation, asynchronous checkpoint finalization, and observed storage delays.
The numbers in examples are starting points, not service guarantees. See
:doc:`configuration` for warmup, YAML, and experimental calibration details.

Recovery limits
---------------

InJob needs a live control/store service, enough healthy nodes for the fixed
training shape, a readable completed checkpoint, and a remaining restart budget.
It does not repair bad data, incompatible checkpoints, persistent storage
outages, or deterministic training errors. Spares replace nodes without changing
the active world size; they do not implement elastic Megatron parallelism.

Loss of the colocated rendezvous host ends that generation. The
:doc:`examples/singleton_deployment` guide explains scheduler continuation and
cleanup, including external intervention when a hard node loss prevents
the batch script's exit trap from running. The experimental external control
process is described in :doc:`deployment`.

Where to go next
----------------

* :doc:`integration/megatron`: section boundaries and async checkpoint recovery.
* :doc:`examples/megatron_gpt`: small GPT recovery and multinode TP/PP variants.
* :doc:`integration/sections`: integrating another training loop.
* :doc:`configuration`: CLI/YAML settings, defaults, and experimental features.
* :doc:`operations`: checkpoint progress, cycle logs, stop decisions, and diagnosis.
* :doc:`services`: optional health checks and attribution.
* :doc:`timeout_diagnostics`: optional timeout stack capture.
