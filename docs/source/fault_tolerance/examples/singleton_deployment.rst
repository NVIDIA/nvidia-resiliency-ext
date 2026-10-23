Singleton job-array deployment (SLURM)
======================================

After validating :doc:`megatron_gpt`, use this advanced reference deployment of
``ft_launcher`` for large-scale training: colocated InJob
restart with a spare pool, plus a singleton job-array chain for colocated
rendezvous-host failure. Other unrecoverable conditions still need operator action.

Array task 0 runs training workers and hosts the rendezvous service, so its
liveness is the
generation's liveness. Every other task can be replaced from the spare pool; task 0
cannot, because the endpoint every other task dials is its hostname. Submitting the
array with ``--dependency=singleton`` provides cross-job continuation for that loss: losing
task 0 makes the current generation unrecoverable. All remaining array tasks must exit
or be cancelled before the dependency is satisfied; the next generation is then eligible
to run when resources become available, rendezvous on a new task 0, and resume
from the last checkpoint.

The example uses NVRx-backed asynchronous checkpoints through Megatron. Pin and
qualify both revisions as described in :doc:`../integration/megatron`. It trains
public `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ on mock
data, so it needs no corpus and no tokenizer model.

Files (under ``examples/fault_tolerance/deployment/``):

``README.md``
   Restart models, the rank-0 failure gap, and how the design maps onto Kubernetes.

``slurm/nvrx_singleton_array.sbatch``
   One generation: rendezvous host publication, the peer gate, ``ft_launcher``, and the
   teardown that releases the spare pool when task 0 exits.

``slurm/submit_chain.sh``
   Computes the array shape from the training-task, hot-spare, and cold-spare settings,
   then enqueues K generations.

``slurm/README.md``
   Run modes, prerequisites, knobs, and hot vs cold spares.

``watch/``
   ``nvrx-watch``: an out-of-job watcher that reconciles the chain and detects restart
   anomalies -- restart storms, restarts that never advance the checkpoint iteration,
   and stalled cycles. Reads NVRx cycle-info files, so its anomaly detection is
   scheduler-independent. See ``watch/DESIGN.md``.

Quick start
-----------

``NVRX_WORK_DIR`` must be creatable on the submission host and visible at the same path
from every compute node. Then choose one of these workload environments:

* **Container mode** (the mode exercised by the singleton QA cases): set the image and
  mount the shared work directory. ``MEGATRON_PATH`` is a path inside the image.

  .. code-block:: bash

     export NVRX_WORK_DIR=/shared/$USER/nvrx-run
     export NVRX_CONTAINER_IMAGE=/path/to/pytorch+nvrx.sqsh
     export NVRX_CONTAINER_MOUNTS=/shared:/shared
     export MEGATRON_PATH=/workspace/megatron-lm

* **Bare-metal mode** (not currently exercised by the singleton QA cases): leave
  ``NVRX_CONTAINER_IMAGE`` unset. Megatron-LM must be available at ``MEGATRON_PATH``
  on the submission host and every compute node, and nvidia-resiliency-ext
  (``ft_launcher``) must be installed on every compute node.

  .. code-block:: bash

     export NVRX_WORK_DIR=/shared/$USER/nvrx-run
     unset NVRX_CONTAINER_IMAGE
     export MEGATRON_PATH=/shared/path/to/Megatron-LM

The example defaults to segment-aware mode (``NVRX_SEGMENT=1``), which requires
``nvidia-smi -q`` to report a GPU ClusterUUID. On systems without a ClusterUUID, select
simple hot-spare mode by setting ``NVRX_SEGMENT`` to an explicit empty value:

.. code-block:: bash

   export NVRX_SEGMENT=

Submit the chain:

.. code-block:: bash

   cd examples/fault_tolerance/deployment/slurm

   NVRX_DRY_RUN=1 ./submit_chain.sh    # print the sbatch commands only
   ./submit_chain.sh                   # default: no-restart (exit 93) demo
   NVRX_NO_RESTART_DEMO=0 ./submit_chain.sh   # hand-off demo instead

   # injection-free scaled mock training: injection off, scale up
   NVRX_NO_RESTART_DEMO=0 NVRX_FAULT_INJECT=0 NVRX_INJECT_GPU_FAILURE= \
   NVRX_MODEL_PROFILE=8b NVRX_TRAIN_TASKS=32 NVRX_HOT_SPARES=1 NVRX_COLD_SPARES=32 \
   NVRX_CHAIN_DEPTH=8 NVRX_TIME_LIMIT=02:00:00 ./submit_chain.sh

Then watch the run from a login node by its Slurm job id (name, owner and work dir
resolve from Slurm; observe-only by default, ``--act`` cancels pending cold-spare tasks
from an orphaned generation):

.. code-block:: bash

   cd ../watch
   python3 -m nvrx_watch <job_id>

The default is a no-progress shutdown demonstration, not a successful recovery
test. Section timeouts and async-save settings are fixed; automatic calibration
is experimental. See the Slurm README for timeout/warmup overrides and the
checkpoint-before-fault checks required for the hand-off mode.

Array concurrency is not atomic initial admission. The singleton dependency
serializes generations without reserving capacity or guaranteeing a startup
time. A hard loss of task 0 can prevent its EXIT trap from running; observe-only
watching does not itself clean up the generation. Owner-enabled ``--act`` releases
pending orphaned spares only. Confirm all old tasks have ended before expecting
the successor to start.
