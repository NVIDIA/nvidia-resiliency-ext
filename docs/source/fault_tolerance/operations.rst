Operations and troubleshooting
==============================

Cycle records and checkpoint progress
-------------------------------------

Pass ``--ft-cycle-info-dir`` to retain per-cycle records and the current-cycle
symlink. Workers receive ``NVRX_CURRENT_CYCLE_INFO``. Keep each launch attempt
or singleton generation in its own namespace so cycle numbers starting at zero
do not overwrite an older attempt. Correlate cycle, hostname, worker rank, and
PID rather than reading the latest cycle's placement as an earlier map.

``--ft-checkpoint-iteration-file`` should point at the workload's **completed**
checkpoint marker, for Megatron ``latest_checkpointed_iteration.txt``. It does
not measure live optimizer iterations or async save requests. No new completed
checkpoint across several failing cycles can trigger a no-progress stop even
when workers repeatedly reach later training iterations.

The tracker seeds its baseline from a readable marker at startup. When tracking
is enabled, a missing configured marker counts as no progress for each failed
cycle, including before the first checkpoint. A readable marker containing zero
also counts as no progress. With the default threshold of three, failed cycles
0, 1, and 2 without a checkpoint cause termination before cycle 3 starts.
If a marker disappears after a checkpoint, the tracker preserves the last known
iteration while counting the failed cycle as no progress. An empty, malformed,
or otherwise unreadable marker instead skips that cycle's check without changing
the count or baseline. ``--ft-min-progress-iterations`` defaults to 1.

For recovery qualification, record detection, cleanup, rendezvous, initialization,
checkpoint restoration, and first committed optimizer-step timestamps. Process
startup alone is not recovered training. No fixed restart latency is guaranteed.

.. _ft-no-restart:

No restart: telling "do not requeue" apart from a failure
---------------------------------------------------------

Two built-in policies report a dedicated no-restart verdict:

* an attribution ``STOP`` verdict, when ``--ft-attribution-stop-action no-restart`` is
  set (see :doc:`services`; the default ``log`` never terminates), and
* the progress tracker finding no progress across restarts
  (``--ft-max-no-progress-cycles`` / ``max_no_progress_cycles``, default ``3``).

The progress tracker reads the iteration from ``--ft-checkpoint-iteration-file``;
if no path is configured and no iterations are otherwise reported, it stays
inactive and logs a one-time warning. A configured but missing file counts as
no progress. Raise the cycle threshold, or set it to ``0`` to disable tracking,
for workloads that deliberately replay an iteration
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
------------------

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
--------------------------------------------

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

Workload control and diagnosis
------------------------------

``RankMonitorClient.send_workload_control_request()`` lets a workload request
that the launcher stop restarting or exclude a node from subsequent attempts.
It does not release or cancel scheduler allocations. Coordinate the application
error path rather than continuing with potentially corrupted optimizer state.

When diagnosing a failed cycle:

* Inspect its application log, launcher/monitor log, and cycle rank map together.
* Distinguish an original timeout/crash from NCCL/TCPStore shutdown cascades.
* Check completed checkpoint progress and restored iteration, not just process
  liveness. Before the first checkpoint, progress can legitimately be unknown.
* For suspected host OOM or hardware failure, collect compute-node kernel/Xid/ECC
  evidence where available. Passing a later health check does not establish that
  the original node was healthy.
* Use :doc:`timeout_diagnostics` for optional bounded stack capture on selected
  section timeouts. Heartbeat/out-of-section timeouts do not trigger those captures.

For shared multi-node logs, configure both the application prefix and launcher
log path so funneling is enabled. The standalone examples in :doc:`services`
include the application and launcher log routes alongside attribution options.
