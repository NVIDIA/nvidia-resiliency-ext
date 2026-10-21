Section monitoring and checkpoint recovery with DDP
===================================================

This small example teaches the sections API without Megatron dependencies.
It retains **synchronous rank-0 PyTorch checkpointing**. Fixed CLI timeouts are
the default; dynamic timeout calculation and ``ft_state.json`` require an
explicit experimental option.

Run from the NVRx repository root with NVRx and PyTorch installed:

.. code-block:: bash

   ft_launcher --nproc-per-node=2 --rdzv-endpoint=localhost:0 --max-restarts=3 \
     --ft-rank-section-timeouts init:120,step:30,checkpoint:120 \
     --ft-rank-out-of-section-timeout 60 \
     examples/fault_tolerance/train_ddp_sections_api.py \
     --device=cpu --hidden=32 --batch=8 --epochs=4 \
     --train_dataset_size=512 --val_dataset_size=64 \
     --save_interval=8 --output_dir=/tmp/nvrx-ddp-sections

Use a fresh output directory for a baseline. Replace ``--device=cpu`` with
``--device=cuda`` and set the local process count to the available GPUs for
GPU execution. The complete checkpoint is ``checkpoint.pt`` under the output
directory. Running again with the same directory restores it; a completed
training run exits without repeating training.

What is monitored
-----------------

``init`` covers checkpoint restoration and DDP setup after the client connects.
``step`` covers training and evaluation work. ``checkpoint`` covers save
preparation, synchronous rank-0 I/O, and peer synchronization. Distributed
initialization before the monitoring client connects is outside its coverage.

.. literalinclude:: ../../../../examples/fault_tolerance/train_ddp_sections_api.py
   :language: python
   :start-at:     # Initialize fault tolerance.
   :end-at:     is_checkpoint_loaded = False

Checkpoint and data progress
----------------------------

Every worker restores model state, Adam optimizer state, and epoch/iteration
progress with ``torch.load(..., weights_only=True)``. The resumable distributed
sampler sets the saved epoch and skips consumed samples within that epoch;
changing an enumeration index alone would not restore data position.

Rank 0 writes a temporary checkpoint and renames it over ``checkpoint.pt``
after saving returns, then workers synchronize. Interruption during the save
does not overwrite the previous completed checkpoint. This simple publication
scheme requires a filesystem with same-directory atomic rename semantics; it
is not a guarantee against storage-server failure. General stochastic workloads
must also preserve per-rank RNG and any other training-dependent state.

.. literalinclude:: ../../../../examples/fault_tolerance/train_ddp_sections_api.py
   :language: python
   :start-at: def save_checkpoint(
   :end-at:     ft_client.end_section('checkpoint')

Exercise a failure
------------------

On a longer run, stop or kill a **training worker PID**, not the launcher or
monitor. Correlate its ``RANK`` and PID in the application log, and inject only
after ``Saving checkpoint`` has completed. SIGSTOP exercises a section timeout;
SIGKILL exercises worker-failure detection. Check that all workers restart, load
the completed checkpoint, advance training, and save a newer checkpoint.

The workload also offers ``--simulated_fault=rank_hung,8`` or
``--simulated_fault=rank_killed,8``. It arms this after an epoch checkpoint and
may arm again in a replacement process; use a long enough workload to observe it.
The tiny command above can finish before the injection timer fires.

Existing GPU functional harnesses target hangs/crashes and eventual completion.
The example recovery unit tests additionally compare resumed model/optimizer
state and sample order against uninterrupted training.

For this guide update, two-worker CPU/Gloo runs with Python 3.12 and PyTorch 2.8
completed after both SIGKILL and SIGSTOP injection following a mid-epoch
checkpoint. Their final model and optimizer state matched uninterrupted training
exactly. These CPU checks do not establish GPU/Slurm qualification.

Optional configuration and experiments
--------------------------------------

The supplied YAML is equivalent to the documented fixed deadlines:

.. literalinclude:: ../../../../examples/fault_tolerance/fault_tol_cfg_sections.yaml
   :language: yaml

Use ``--ft-cfg-path`` instead of the timeout CLI options if desired.
``--calculate_timeouts`` enables experimental calibration and timeout JSON
persistence; it is not needed for normal execution and can override fixed values.
See :doc:`../configuration`.

.. warning::

   Load only checkpoints you trust. For PyTorch versions before 2.10.0,
   CVE-2026-24747 affects the ``weights_only`` unpickler. Use PyTorch 2.10.0
   or newer when checkpoint provenance is not fully trusted.

The complete source is
``examples/fault_tolerance/train_ddp_sections_api.py``. Continue with
:doc:`megatron_gpt` for NVRx-backed async checkpointing and model parallelism.
