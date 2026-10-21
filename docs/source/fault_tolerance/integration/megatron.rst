Megatron-LM integration
=======================

Megatron-LM already integrates the NVRx sections API. Run its
``pretrain_gpt.py`` entrypoint with ``ft_launcher`` and enable
``--enable-ft-package``. Do not add a second monitoring client or another
checkpoint loop around Megatron.

The walkthrough in :doc:`../examples/megatron_gpt` uses fixed section timeouts
on the launcher CLI and NVRx-backed asynchronous distributed checkpoints.
``--calc-ft-timeouts`` is experimental and is deliberately omitted.

Version and environment contract
--------------------------------

The source contract for this guide was reviewed against public Megatron-LM
revision ``36731acc3f0dac859ee7369bdc3a6af58bac546c``:
`FT integration <https://github.com/NVIDIA/Megatron-LM/blob/36731acc3f0dac859ee7369bdc3a6af58bac546c/megatron/training/ft_integration.py>`_,
`async orchestration <https://github.com/NVIDIA/Megatron-LM/blob/36731acc3f0dac859ee7369bdc3a6af58bac546c/megatron/training/async_utils.py>`_,
and `distributed writer integration <https://github.com/NVIDIA/Megatron-LM/blob/36731acc3f0dac859ee7369bdc3a6af58bac546c/megatron/core/dist_checkpointing/strategies/torch.py>`_.

This is a source-reviewed recipe, not a GPU/Slurm qualification claim for that
revision. Pin both Megatron and NVRx in the deployed container, record CUDA,
PyTorch, Transformer Engine, and NCCL versions, and run the recovery checks in
the example before relying on a combination. Megatron's FT argument help still
labels the integration internal-use; availability in its public source does not
establish a supported combination across all releases. Older Megatron versions
may offer an ``async_strategy`` selector; do not copy a backend-selection flag
from another revision without checking its help and implementation.

Section boundaries
------------------

.. list-table::
   :header-rows: 1

   * - Section
     - Megatron behavior
     - Timeout sizing
   * - ``setup``
     - Opened before initialization; closed before the first train/eval iteration
     - Model construction, distributed initialization, compilation, checkpoint load
   * - ``step``
     - Training/evaluation steps after configured warmup
     - Full step, including communication and optimizer work
   * - ``checkpointing``
     - Save-related work and async finalization calls
     - Preparation plus worst observed finalization stalls
   * - Out of section
     - Work outside open monitored sections, subject to warmup suppression
     - Data preparation, logging, and other between-section work

The section names must match exactly. The DDP example's ``init`` and
``checkpoint`` names are not aliases. Configure finite timeouts before launch;
simply enabling ``--enable-ft-package`` does not give those sections deadlines.
See :doc:`../configuration` for first-step coverage and restored timeout state.

Async checkpointing and recovery
--------------------------------

Megatron owns the training state and checkpoint lifecycle. For the distributed
``torch_dist`` path in the reviewed revision, NVRx supplies
``AsyncCallsQueue``, ``AsyncRequest``, and ``FileSystemWriterAsync``.
The workload arguments used in the example are:

.. code-block:: bash

   --load /shared/run/checkpoints \
   --save /shared/run/checkpoints \
   --save-interval 5 \
   --ckpt-format torch_dist \
   --async-save \
   --use-persistent-ckpt-worker

NVRx InJob remains responsible for process recovery; this is a different
component from NVRx async checkpointing.

A save request stages a snapshot and schedules background I/O. Training may
continue before that save is recoverable. Finalization completes the distributed
checkpoint and publishes its latest-checkpoint tracker. If workers die during
an unfinished save, recovery must use the previous completed checkpoint.
Do not advance a progress marker merely because a save was submitted.

Megatron wraps save preparation and async finalization in ``checkpointing``
sections. The entire background save is not one continuously open section, and
a stalled background operation can instead surface when finalization waits.
A small checkpointing timeout copied from a synchronous example is unsuitable.

For the distributed optimizer, restore optimizer shards and scheduler/data/RNG
state as well as model weights. Keep the example's active world size and TP/PP
shape fixed through replacement. Standbys do not enlarge that world. Different
topologies require explicit checkpoint-format and resharding validation; InJob
does not implement live model-parallel resizing.

Pass ``--ft-checkpoint-iteration-file`` on the launcher CLI with the path to
``latest_checkpointed_iteration.txt``. This tracks completed checkpoint
progress, not the newest training iteration. Sparse checkpointing and deliberate
Megatron reruns can look like no progress across cycles; see
:doc:`../operations` for the stop threshold and unavailable-marker behavior.

Timeout state and shutdown
--------------------------

Megatron saves calculated timeouts to ``ft_state.json`` only when calibration
runs, but restores an existing file at startup. Use a fresh directory for the
fixed-timeout walkthrough. That file is distinct from launcher YAML and from
the distributed training checkpoint.

Let Megatron finalize outstanding async saves during normal shutdown. A forced
worker termination can interrupt an in-flight save; the most recent completed
checkpoint remains the recovery point. Qualification must cover failures during
training, save preparation, and background-save/finalization windows.
