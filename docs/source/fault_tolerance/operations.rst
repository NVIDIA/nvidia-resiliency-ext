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

No restart: distinguishing a terminal decision from a failure
--------------------------------------------------------------

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

Both end further InJob recovery in the current generation and report the same
``--ft-no-restart-exit-code`` (default ``93``), rather than the generic failure
code ``1``. Peers that observe the shutdown reason also use that code; delayed
peers may instead lose their store connection and exit ``1``, as described below.
The code does not itself cancel queued jobs or configure Slurm requeue behavior.
The intended singleton policy is to cancel only the current array on control-host
exit and leave queued successors eligible to start, including after ``93`` or
successful completion with ``0``. The repository's example currently also uses
``control/no_restart`` to cancel queued successors. Correcting that existing
example policy is separate from this control-host refactor.

The specific cause is recorded in the rendezvous store shutdown key as ``attribution_stop``
or ``no_progress``, and is named in the launcher logs, so the two remain distinguishable
for diagnosis without complicating the scheduler contract.

Running out of restarts is deliberately *not* one of these cases. Exhausting
``--max-restarts`` means this launcher used up its retry budget, not that the job is
unrecoverable, so it still exits ``1`` and leaves requeueing to your scheduler policy.
The distinction is the cause: ``93`` identifies an explicit attribution or
no-progress decision, while ``1`` identifies exhaustion of the ordinary retry
budget. Both stop InJob recovery in the current generation.

* CLI:

  - ``--ft-no-restart-exit-code <CODE>`` (alias: ``--ft_no_restart_exit_code``),
    default ``93``

Note that SLURM reports the maximum exit code across tasks and that ``sacct`` derived exit
codes are lossy, so treat this as a best-effort signal.

Nodes other than the one that made the decision read the reason from the rendezvous store,
so the store must outlive them. Both the store-host launcher and ``job-controller`` hold it
open for a few seconds after shutting down, but a node that has not read the key by then
exits ``1`` rather than the no-restart code. The job still stops either way.

The separate job-controller also continues enforcing attribution STOP while a
node-agent waits in the success exit barrier. Until graceful rendezvous shutdown
is recorded, a STOP can produce ``93`` after local workers have succeeded and
can stop peers that are still finishing. This accepted change from embedded
control, together with differing peer exit codes, is described in
:doc:`deployment`. Future per-cycle ``end_reason`` reporting will consolidate the
terminal outcomes.


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

Control-host process logs
-------------------------

The control-host supervisor writes its own logs and the stdout/stderr of
``job-controller``, managed attribution, and the gRPC root/leaf processes into one
control log. For example, ``--ft-nvrx-logfile /lustre/job/nvrx.log`` produces:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``nvrx.log``
     - Supervisor, control, managed attribution, and gRPC servers' own logs;
       written locally by one writer, independently of the gRPC path.
   * - ``nvrx_agent.log``
     - Node-agents and rank monitors on all nodes, including the control-host agent;
       uses the existing compute gRPC path.

The configured path is used exactly for the control log. No extra launcher
option is required. For the shared agent log, a trailing ``.log`` is replaced
with ``_agent.log``; otherwise ``_agent.log`` is appended. Without
``--ft-nvrx-logfile``, these process logs go to the original console stream.
Existing file-name headers are retained, with no extra component field.
The supervisor does not create separate attribution, root, or leaf log files.
This also applies to an external CPU host using ``--ft-control-only``; compute
logs still arrive through gRPC from the GPU nodes. Aggregators' own logs always
use the supervisor's local capture; there is no separate root/leaf file path or
process-log prefix option. ``--standalone`` is for single-node development and
does not start gRPC aggregation. Its configured application and launcher logs
are written locally; the embedded agent uses the derived ``*_agent.log`` and
there is no separate supervisor control log. Tools that previously read
node-agent or rank-monitor messages from the configured path should now read
the derived agent file.

A reader thread drains each process pipe and a separate writer appends to the
control log. Process exit codes determine completion and failure; no lifecycle
channel or log parsing is used for supervision. During cleanup, the node agent
has its worker termination allowance plus 15 seconds for cleanup and log draining
before the gRPC servers stop. The local writer remains
alive until control-service and aggregator logs have drained.
Capture is best effort: the implementation limits pending log memory and falls
back to original stderr on write failure. Overflow uses stderr when it supports
nonblocking writes; otherwise excess chunks are dropped. A blocked console can
also lose fallback data. File writes never run in the supervision loop; shutdown
has a finite drain deadline.

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
* Use :doc:`timeout_dump` for optional bounded stack capture on selected
  section timeouts. Heartbeat/out-of-section timeouts do not trigger those captures.

For shared multi-node logs, configure both the application prefix and launcher
log path so funneling is enabled. The standalone examples in :doc:`services`
include the application and launcher log routes alongside attribution options.
