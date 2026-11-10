GPT checkpoint recovery with Megatron-LM
========================================

This walkthrough trains a small GPT on mock data, completes an NVRx-backed
async checkpoint, injects one worker failure, and resumes from a completed
checkpoint. It uses fixed CLI section timeouts. There is no corpus or tokenizer
model to download.

Requirements and validation status
----------------------------------

Provide a GPU environment with compatible PyTorch, CUDA, NCCL, Transformer
Engine, Megatron-LM, and this NVRx checkout installed. All nodes must see the
same source paths and work directory. The example does not build a training
container or install Megatron's dependencies.

See :doc:`../integration/megatron` for the reviewed Megatron revision and version
contract. The wrapper and command construction can be tested without GPUs;
end-to-end GPU and Slurm execution must be qualified in the target environment.
A successful dry run or documentation build is not evidence of training recovery.

Single-node run
---------------

From the NVRx repository root, choose a fresh work directory:

.. code-block:: bash

   export MEGATRON_PATH=/workspace/Megatron-LM
   export NVRX_WORK_DIR=/shared/$USER/nvrx-gpt
   export NVRX_RUN_ID=gpt-recovery-001
   export GPUS_PER_NODE=2

   NVRX_DRY_RUN=1 bash examples/fault_tolerance/run_megatron_gpt.sh
   bash examples/fault_tolerance/run_megatron_gpt.sh

The default trains a four-layer, hidden-size-512 GPT for 100 iterations and saves
every 5 iterations. It uses BF16, a distributed optimizer, ``torch_dist``,
``--async-save``, and ``--use-persistent-ckpt-worker``. Training/evaluation
sections are enabled from the first step with ``--ft-num-warmup-iters=0``.
The fixed deadlines are ``setup:900,step:120,checkpointing:600`` and 180 seconds
out of section. Qualify these values for your stack.

The script prints the exact launcher/workload command in dry-run mode; it still
validates paths and creates the artifact directories. This is not a submission
script and does not allocate GPUs. For Slurm or containers, run it inside the
allocated training environment.

Fault injection and repeated runs
---------------------------------

``NVRX_DEMO_FAULT=kill`` (default) sends SIGKILL to training rank 1 after the
latest-checkpoint iteration file advances beyond its startup baseline.
``NVRX_DEMO_FAULT=hang`` uses SIGSTOP to exercise timeout detection instead.
``NVRX_DEMO_FAULT=none`` disables injection for an uninterrupted baseline.
For fault injection on one GPU, set ``NVRX_FAULT_RANK=0``; the default rank 1
needs at least two active GPUs. An uninterrupted baseline ignores the fault
rank, so ``NVRX_DEMO_FAULT=none GPUS_PER_NODE=1`` needs no rank override.

The wrapper runs Megatron's unmodified entrypoint. It claims
``runs/<run-id>/fault_injected.json`` exclusively on shared storage before
signalling itself, so replacement worker processes do not inject again. It does
not modify training state, select checkpoint shards, or implement async saves.
If the requested fault never occurs, the selected worker reports failure at
normal exit instead of silently presenting an uninterrupted run as recovery.

Use a **new run ID for every separate test**. The submission shell rejects an
old ID containing an injection marker; worker restarts use that marker to avoid
repeated injection. To resume an existing training
checkpoint into a new launch, also set
``NVRX_CHECKPOINT_DIR=/shared/path/to/checkpoints``. The injector then waits
for a new checkpoint beyond that directory's initial iteration. Ensure
the existing tracker is readable and contains an integer at startup: an empty,
invalid, or unreadable tracker stops the wrapper before Megatron starts, rather
than treating its contents as zero. Only an absent tracker establishes a fresh
run's zero baseline. Temporary read failures during subsequent polling are
ignored without changing that baseline. Ensure
``TRAIN_ITERS`` exceeds the restored iteration by enough to complete multiple
checkpoints. The fixed-timeout script rejects a directory containing calculated
``ft_state.json``; archive that timeout-state file deliberately if appropriate.

Verify recovery
---------------

Artifacts live under ``$NVRX_WORK_DIR/runs/$NVRX_RUN_ID`` by default:

.. code-block:: bash

   cat "$NVRX_WORK_DIR/runs/$NVRX_RUN_ID/fault_injected.json"
   ls "$NVRX_WORK_DIR/runs/$NVRX_RUN_ID/cycle_infos"
   cat "$NVRX_WORK_DIR/runs/$NVRX_RUN_ID/checkpoints/latest_checkpointed_iteration.txt"

Check all of the following, not just a launcher exit status:

1. A checkpoint completed before the injected failure; record its iteration
   from the injection marker and correlate it with Megatron's logs.
2. The launcher records the failed cycle and starts replacement worker processes.
   For the hang case, correlate the section-timeout message with the failed rank.
3. Megatron reports loading a completed training checkpoint, including optimizer
   state, rather than starting from iteration zero.
4. Training commits optimizer steps beyond the restored checkpoint, a later async
   checkpoint completes, and the intended final iteration is reached.
5. The run finishes successfully without repeated injection or a no-progress stop.

Application logs are ``logs/train_cycle<N>.log``; launcher/monitor logs are
routed through ``logs/launcher.log``. Aggregation is best effort, so use cycle
records and checkpoint state together. Measure failure-to-first-committed-step
latency separately from failure detection and process startup.

Multinode TP/PP example
-----------------------

First validate single-node recovery. For two nodes with four GPUs each, this
example uses TP=2, PP=2, DP=2; the active world stays eight GPUs after a restart.
Use a shared reachable endpoint and a common run ID:

.. code-block:: bash

   # Inside an allocation with two nodes and four GPUs per node:
   export NNODES=2 GPUS_PER_NODE=4 TP=2 PP=2
   export NVRX_RUN_ID=gpt-tp2-pp2-001
   export RDZV_ENDPOINT="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):29400"
   srun --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 \
     bash examples/fault_tolerance/run_megatron_gpt.sh

The script sets global batch size to the active GPU count and microbatch size
to one, satisfying divisibility by DP for these shapes. Four model layers must
divide by PP and eight attention heads by TP. Additional workload arguments can
be appended, but changes to batch/model/checkpoint shape need validation.

This example demonstrates worker recovery within a fixed allocation; a failed
physical node can invalidate a normal Slurm allocation. For independently
releasable replacement groups, hot/cold spares, and rendezvous-host loss, continue
with :doc:`singleton_deployment`. Do not substitute allocation shrinking for
the array deployment's replacement behavior.

Source
------

.. literalinclude:: ../../../../examples/fault_tolerance/run_megatron_gpt.sh
   :language: bash
   :linenos:

The once-only injection wrapper is
``examples/fault_tolerance/megatron_gpt_worker.py``.
