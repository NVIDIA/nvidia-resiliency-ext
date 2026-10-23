Sections API integration
========================

Use named sections to give setup, training/evaluation, and checkpoint work
separate fixed deadlines. Megatron already provides this instrumentation; see
:doc:`megatron`. This page describes integrating another training loop.

Configure before launch
-----------------------

For the small DDP example:

.. code-block:: bash

   ft_launcher --nproc-per-node=2 --rdzv-endpoint=localhost:0 --max-restarts=3 \
     --ft-rank-section-timeouts init:120,step:30,checkpoint:120 \
     --ft-rank-out-of-section-timeout 60 \
     examples/fault_tolerance/train_ddp_sections_api.py --device=cpu

The section names are application-defined and must match configuration exactly.
An omitted name skips both monitoring messages and timing collection; a name
mapped to ``null`` sends events but has no finite deadline. Finite timeouts must
cover the slowest expected operation in each phase.

Sections can be nested or overlap, but a section cannot be opened twice without
being closed. Out-of-section time is checked only when no monitored section is
open, with the warmup rules described in :doc:`../configuration`. It bounds
worker execution after monitoring begins, not launcher/rendezvous startup time.
Work before the client connects is outside this monitor's coverage.

Instrument the workload
-----------------------

On every worker, initialize a ``RankMonitorClient``, then report section
boundaries at equivalent training phases. Shut monitoring down on normal exit.
For example, after distributed initialization:

.. code-block:: python

   from nvidia_resiliency_ext.fault_tolerance import RankMonitorClient

   monitor = RankMonitorClient()
   monitor.init_workload_monitoring()
   monitor.start_section("init")
   # construct model/optimizer and restore checkpoint
   monitor.end_section("init")

   monitor.start_section("step")
   # forward, backward, optimizer step
   monitor.end_section("step")

   monitor.start_section("checkpoint")
   # save training state; synchronize peers as required by the checkpoint backend
   monitor.end_section("checkpoint")
   monitor.shutdown_workload_monitoring()

Monitoring is independent from checkpoint storage. The DDP example keeps
synchronous rank-0 PyTorch save/load; Megatron uses NVRx async save/finalization
through its own checkpoint lifecycle. All ranks must participate consistently in
checkpoint collectives and any optional timeout-calculation collectives.

Do not blindly close a failed section and continue training after an exception;
the workload must either establish a valid recovery path or let workers fail so
the launcher can restart the group. For graceful scheduler shutdown, record the
signal locally and handle it at a synchronized safe point in the training loop,
rather than running distributed operations inside a signal handler.

Fixed configuration is the default. The experimental
``calculate_and_set_section_timeouts()`` and timeout-state persistence APIs
remain available but are not needed for this integration. See
:doc:`../configuration` before experimenting with them.

See :doc:`../examples/train_ddp_sections` for runnable commands and checkpoint
recovery checks. The optional YAML equivalent is:

.. literalinclude:: ../../../../examples/fault_tolerance/fault_tol_cfg_sections.yaml
   :language: yaml
