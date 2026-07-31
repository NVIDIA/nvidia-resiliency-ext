Singleton job-array deployment (SLURM)
======================================

A reference deployment of ``ft_launcher`` for large-scale training: colocated in-job
restart with a spare pool, plus a singleton job-array chain that survives the one
failure in-job restart cannot repair.

Array task 0 is both a training rank and the rendezvous host, so its liveness is the
generation's liveness. Every other task can be replaced from the spare pool; task 0
cannot, because the endpoint every other task dials is its hostname. Submitting the
array with ``--dependency=singleton`` makes that loss cheap instead of fatal: losing
task 0 ends the current array, and the scheduler immediately starts the next one, which
rendezvouses on a new task 0 and resumes from the last checkpoint.

The example trains public `Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ on mock
data, so it needs no corpus and no tokenizer model.

Files (under ``examples/fault_tolerance/deployment/``):

``README.md``
   Restart models, the rank-0 failure gap, and how the design maps onto Kubernetes.

``slurm/nvrx_singleton_array.sbatch``
   One generation: rendezvous host publication, the peer gate, ``ft_launcher``, and the
   teardown that releases the spare pool when task 0 exits.

``slurm/submit_chain.sh``
   Computes the array shape from one setting and enqueues K generations.

``slurm/README.md``
   Knobs, and hot vs cold spares.

``watch/``
   ``nvrx-watch``: an out-of-job watcher that reconciles the chain and detects restart
   anomalies -- restart storms, restarts that never advance the checkpoint iteration,
   and stalled cycles. Reads NVRx cycle-info files, so its anomaly detection is
   scheduler-independent. See ``watch/DESIGN.md``.

Quick start
-----------

.. code-block:: bash

   cd examples/fault_tolerance/deployment/slurm

   export MEGATRON_PATH=/workspace/megatron-lm
   export NVRX_WORK_DIR=/shared/$USER/nvrx-run

   NVRX_DRY_RUN=1 ./submit_chain.sh    # print the sbatch commands only
   ./submit_chain.sh                   # default: no-restart (exit 93) demo
   NVRX_NO_RESTART_DEMO=0 ./submit_chain.sh   # hand-off demo instead

   # real training: injection off, scale up
   NVRX_NO_RESTART_DEMO=0 NVRX_FAULT_INJECT=0 NVRX_INJECT_GPU_FAILURE= \
   NVRX_MODEL_PROFILE=8b NVRX_TRAIN_TASKS=32 ./submit_chain.sh

Then watch the run from a login node by its Slurm job id (name, owner and work dir
resolve from Slurm; observe-only by default, ``--act`` to enable the one action):

.. code-block:: bash

   cd ../watch
   python3 -m nvrx_watch <job_id>
