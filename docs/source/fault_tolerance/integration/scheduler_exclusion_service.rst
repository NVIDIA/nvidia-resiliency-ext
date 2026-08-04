Scheduler Exclusion
===================

Scheduler Exclusion prevents NVRx from reusing a Slurm array-task generation
after the scheduler marks one of its nodes unavailable. The host-side
``nvrx-scheduler-exclusion-service`` queries Slurm outside the workload and
publishes a decision on storage mounted by every workload node. Slurm binaries
and credentials are not required inside the workload environment.

Deployment
----------

Slurm runs each array task's batch script on that task's first allocated node.
Only the lowest array task starts the service, which monitors the parent
``SLURM_ARRAY_JOB_ID``. The service can also monitor ``SLURM_JOB_ID`` for a
regular job and publish node decisions, but the current FT consumer supports
array-task decisions only.

.. code-block:: text

   first allocated node of the lowest array task
     |-- supervise nvrx-scheduler-exclusion-service
     |     |-- squeue: task generation, partition, and nodelist
     |     |-- scontrol: expand task nodelists
     |     |-- sinfo: query cluster-wide scheduler-unavailable nodes
     |     |-- intersect unavailable nodes with the allocation in memory
     |     |-- GET /scheduler-exclusions: current in-memory decision
     |     `-- atomically replace <shared>/scheduler_exclusion.<job-id>.jsonl
     |
     `-- srun workload
           `-- array-task Node0 runs SegmentHealthCheck on the first JSONL decision

The batch script passes one shared directory to the service with
``--output-dir`` and to ``ft_launcher`` with
``--ft-scheduler-exclusion-dir``. NVRx derives the filename from
``SLURM_ARRAY_JOB_ID`` when present, otherwise ``SLURM_JOB_ID``. The directory
can also hold service logs and future diagnostic artifacts. The service process
is supervised with bounded backoff. Its last decision remains readable across
a process restart; the FT consumer fails open once the artifact is 30 minutes
old.

The batch script owns the supervisor lifecycle. It records the supervisor PID,
checks service readiness, sends ``SIGTERM`` on every exit path, and calls
``wait`` to reap it. ``SIGTERM`` lets the service stop its HTTP server and
scheduler monitor cleanly. Do not launch the service with an untracked trailing
``&``. Slurm process cleanup is a final safety net, not the primary lifecycle
mechanism.

``services/scheduler_exclusions/deploy/slurm_array.sbatch`` is the complete
reference. It starts the service only in the lowest array task, restarts it with
capped backoff, redirects service output to the shared artifact directory, and
passes the same directory to ``ft_launcher``. Supply the site-specific
allocation options and workload arguments when submitting it, for example:

.. code-block:: bash

   sbatch \
     --account=<account> \
     --partition=<partition> \
     --array=0-95 \
     --nodes=18 \
     --ntasks=18 \
     --ntasks-per-node=1 \
     --export=ALL,NVRX_SCHEDULER_EXCLUSION_DIR=/shared/nvrx/${USER}/scheduler-exclusions \
     services/scheduler_exclusions/deploy/slurm_array.sbatch \
       --nnodes=1728:1728 \
       --nproc-per-node=4 \
       <other-ft-launcher-options> \
       train.py <training-arguments>

Supported FT Shape
------------------

The current FT consumer supports Slurm job arrays with one ``ft_launcher``
process per allocated node. Only Node0 of each array task reads the decision.
The rendezvous replacement group must contain the complete array task, so
``replacement_group_size`` must equal ``SLURM_NNODES``. A smaller group is not
supported because non-Node0 peers could become eligible before Node0 publishes
the task exclusion. The array must use ``--no-requeue`` because the compact FT
decision is keyed only by array-task ID. The reference batch script sets it.

Launch the workload step with ``srun --kill-on-bad-exit=1``. When an excluded
Node0 exits nonzero, this terminates the rest of that array task's workload step;
the batch script then exits nonzero and Slurm can replace the complete task from
the spare pool. The reference batch script implements these requirements.

Decision Artifact
-----------------

The artifact is JSONL. Its first line is a compact array of excluded task IDs.
The following task decision, node decision, and observation records retain the
metadata behind that control record.

At a restart boundary, ``SegmentHealthCheck`` runs the scheduler decision
predicate through the existing pre-rendezvous health-check flow. Only Node0 of
each array task reads the first line. It searches for the fixed quoted token
``"<task_id>"`` without decoding the detailed records. Quoting prevents task
``7`` from matching task ``17``.
``SLURM_NODEID=0`` identifies that launcher; ``SLURM_PROCID=0`` is the fallback
for the supported one-launcher-per-node launch shape. A regular-job consumer
will read the detailed node decision after the task metadata and check its
scheduler-visible node name; the current ``ft_launcher`` integration consumes
array-task decisions only.

.. code-block:: json

   ["7"]
   {"type":"decision","schema_version":1,"job_id":"12345","generated_at":"2026-08-04T19:00:00Z","scope":"array_task","excluded_array_tasks":[{"task_id":"7","restart_count":0,"valid_until":"2026-08-04T19:30:00Z"}]}
   {"type":"decision","schema_version":1,"job_id":"12345","generated_at":"2026-08-04T19:00:00Z","scope":"node","excluded_nodes":[{"node":"node-a","valid_until":"2026-08-04T19:30:00Z"}]}
   {"type":"observation","node":"node-a","state":"DRAIN","reason":"GPU failure","observed_at":"2026-08-04T19:00:00Z","valid_until":"2026-08-04T19:30:00Z","array_tasks":[{"task_id":"7","restart_count":0}]}

The first line contains strings so the FT path can search for an exact quoted
task ID. The consumer validates the compact no-whitespace array grammar before
matching that token. The file name identifies the Slurm array job. The opened file's
modification time supplies freshness; an artifact older than 30 minutes fails
open. A complete poll with no exclusions writes ``[]``.

The detailed records identify an array task generation by
``(task_id, restart_count)``. The service obtains both values from the same
``squeue --array`` row using ``ArrayTaskID`` and ``RestartCnt``. These fields
remain useful for diagnostics even though the supported ``--no-requeue`` FT
consumer matches only the task ID.

If several unavailable nodes map to one task generation, ``valid_until`` is the
latest expiry among its currently valid node observations. The task remains
excluded while any contributing observation remains valid. Task generations
are deduplicated and sorted numerically in the decision record.

The node decision contains every currently unavailable node for both array and
regular jobs. Each node has its own ``valid_until`` derived from its observation
time. A regular job has an empty array-task decision because it has no task
generation mapping.

Publication uses a temporary file in the destination directory, followed by
``flush``, ``fsync``, and ``os.replace``. A reader that opened the old path keeps
reading the complete old inode; a later reader gets the complete new inode.
Readers never observe a partially written or mixed artifact.

A complete poll with no exclusions publishes ``[]`` followed by two empty
detailed decisions. Allocation discovery and the filtered scheduler-state query
form one all-or-nothing refresh: failure of either leaves the previous artifact
unchanged. Publication failure is logged and reported by ``GET /stats`` without
stopping scheduler polling.

Scheduler Query and Cache
-------------------------

The service refreshes every 10 minutes and retains observations for 30 minutes.
Each poll performs one ``squeue`` query for the current allocation and one
cluster-wide ``sinfo`` query filtered to scheduler-unavailable states. Local
``scontrol show hostnames`` expansion maps compressed task nodelists to nodes;
the service then intersects the two query results in memory. The scheduler RPC
count therefore remains two regardless of allocation size during ordinary
polls.

The ``sinfo`` filter requests ``DRAIN``, ``DOWN``, ``FAIL``, and ``NO_RESPOND``;
its returned forms ``DRAINED``, ``DRAINING``, and ``FAILING`` are also
unavailable. Slurm represents an unresponsive node by appending ``*`` to its
state; the service normalizes such states to ``NO_RESPOND``.

Absence from the filtered response is not sufficient evidence that a previously
unavailable node recovered. When at most 16 cached exclusions remain allocated
but disappear from that response, the service performs one conditional,
unfiltered ``sinfo --nodes`` query. An explicitly allocatable response clears
the exclusion; an explicitly unavailable response refreshes it. Missing rows or
a failed verification retain the previous observation without refreshing its
timestamp, so it expires after 30 minutes. More than 16 candidates indicate a
broad transition: the service skips the extra query and lets their existing
observations expire. Nodes that leave the current allocation are omitted from
the published decision immediately.

Empty or ``UNKNOWN`` states, allocatable rows in the filtered response,
incomplete running-allocation rows, and empty nodelist expansions invalidate the
refresh. The previous cache and artifact remain in effect, and no exclusion is
inferred from malformed or failed scheduler data. The next scheduled refresh
tries again.

HTTP Operations
---------------

The service exposes the same task and node exclusions through HTTP and shared
storage. The JSONL form starts with the compact task-ID control record and then
stores detailed decisions; the HTTP response combines both lists into one
cache-only response:

* ``GET /healthz`` reports process health and the Slurm job ID.
* ``GET /scheduler-exclusions`` returns ``excluded_array_tasks`` and
  ``excluded_nodes``. It returns ``503`` until a cache snapshot is available.
* ``GET /stats`` reports polling, cache, and decision-publication diagnostics.
* ``POST /refresh`` queues a refresh and returns ``202`` immediately.

These endpoints never perform scheduler I/O in the request handler. The worker
coalesces concurrent refresh requests. The NVRx file consumer does not depend
on HTTP availability at a restart boundary.

FT-Side Observability
---------------------

At each rendezvous boundary, every array-task Node0 emits one INFO record for
the synchronous FT check:

.. code-block:: text

   Scheduler Exclusion FT check job_id=123 round=4 task_id=7 restart_count=2 outcome=not_excluded elapsed_ms=0.420

``elapsed_ms`` starts after Node0 detection and covers reading the current task
generation from the environment, opening and reading the first JSONL record,
checking file freshness, and matching the quoted task token. It measures the
synchronous decision-check latency; it does not include the service's
background Slurm polling. Outcomes are ``excluded``, ``not_excluded``,
``expired``, ``missing``, ``invalid``, ``io_error``, and ``not_applicable``.
Non-Node0 launchers neither read the artifact nor emit this record.
Task leaders perform their checks concurrently. Post-run analysis therefore
uses the maximum ``elapsed_ms`` for each ``(job_id, round)`` rather than the sum
across tasks. That maximum represents the round only when every expected
array-task Node0 emitted a sample; with incomplete coverage it is a lower bound.
The full distribution remains useful for identifying shared-filesystem tail
latency.

The existing ``rendezvous_started`` and ``rendezvous_completed`` profiling
events measure total rendezvous time, including replacement or quorum wait.
That duration is an operational consequence of an exclusion, not synchronous
artifact-check overhead. Artifact ``generated_at`` measures decision age and
must not be interpreted as processing latency.

Configuration
-------------

Set the shared directory with ``--output-dir`` or
``NVRX_SCHEDULER_EXCLUSION_DIR``. Other CLI options have matching environment
variables under the ``NVRX_SCHEDULER_EXCLUSION_`` prefix.

.. list-table::
   :header-rows: 1

   * - Environment suffix
     - Default
   * - ``DIR``
     - disabled; required for decision publication
   * - ``HOST``
     - ``127.0.0.1``
   * - ``PORT``
     - ``18080``
   * - ``SLURM_BIN_DIR``
     - use ``PATH``
   * - ``SLURM_CONF``
     - use inherited environment
   * - ``REFRESH_INTERVAL_SECONDS``
     - ``600``
   * - ``CACHE_TTL_SECONDS``
     - ``1800``
   * - ``QUERY_TIMEOUT_SECONDS``
     - ``30``
``NVRX_SCHEDULER_EXCLUSION_DIR``,
``NVRX_SCHEDULER_EXCLUSION_SLURM_BIN_DIR``, and
``NVRX_SCHEDULER_EXCLUSION_SLURM_CONF`` must be absolute when set. The
scheduler-exclusion directory must be on a POSIX-compatible filesystem visible
from the service host and workload. The service owns filenames within it.

Host Artifact
-------------

The service can be built as a standalone ``.pyz`` and run with the service
host's Python 3.10-or-newer interpreter. NVRx does not need to be installed on
that host. Building the wheel does not build or publish the zipapp. The
deployment owner builds it from the same NVRx revision as the workload, assigns
it an immutable version or commit-based path, and stages it on storage visible
to the batch script. Build and deployment assets are under
``services/scheduler_exclusions``. Set
``NVRX_SCHEDULER_EXCLUSION_ARTIFACT`` to the staged zipapp path when using the
reference launcher.
