Segment Health Check
====================

The segment health check prevents NVRx from reusing a Slurm allocation unit
after it is marked unavailable. The unit is an array task for a job array or
the job itself for a regular Slurm job. The FT consumer reads a decision
artifact from shared storage before joining rendezvous; it does not require
Slurm access inside the workload environment. The decision publisher is
independent of this consumer as long as it satisfies the artifact contract
below.

Configuration
-------------

Pass the shared decision directory to ``ft_launcher``:

.. code-block:: bash

   ft_launcher \
     --ft-segment-health-check-dir /shared/job/segment-health-check \
     <other-options> train.py

The path must be absolute and accessible to every launcher. Omitting the option
disables consumption.

Supported Launch Shape
----------------------

The FT consumer supports Slurm job arrays and regular jobs. Both require one
``ft_launcher`` process per allocated node and ``srun --kill-on-bad-exit=1``, so
an excluded process 0 terminates the workload step. Job arrays additionally
require ``--no-requeue`` for the initial deployment artifact lifecycle.

Only the launcher with ``SLURM_PROCID=0`` in each allocation unit reads the
artifact.

Artifact Contract
-----------------

The consumer derives the following path in the configured directory:

.. code-block:: text

   segment_health_check.<job_id>.<task_id>

For an array, ``job_id`` is ``SLURM_ARRAY_JOB_ID`` and ``task_id`` is
``SLURM_ARRAY_TASK_ID``. For a regular job, both values are ``SLURM_JOB_ID``.

A non-empty regular file marks the allocation unit unhealthy. A missing or
zero-byte file proceeds normally. The file content is diagnostic context owned
by the producer; the consumer performs one metadata check and does not read or
parse it. Unreadable paths and unexpected file types emit a warning and fail
open.

Runtime Behavior
----------------

Only allocation-unit process 0 reads the artifact before rendezvous. A clean or
fail-open result joins normally. A match raises ``UnhealthyNodeException``
through the existing health-check path and exits nonzero. For an array, the
same path also marks its replacement group unhealthy; the rendezvous host
already ignores unhealthy replacement groups. A regular job has no array-task
replacement group, so process-0 failure terminates its workload step through
``srun --kill-on-bad-exit=1``.

Other participants from that array task may join before process 0 publishes the
unhealthy marker. Slurm then terminates the task after process 0 exits, which
can require one additional rendezvous before eligible spares are selected.

With enough eligible spares, rendezvous proceeds without the excluded task. If
quorum cannot be formed, the existing rendezvous timeout applies; NVRx does not
weaken the exclusion to create quorum.

A scheduler state such as Slurm ``DRAIN`` does not evict a task that is already
running. NVRx consumes the corresponding decision at the next rendezvous
health-check boundary.

Observability
-------------

An eligible allocation-unit process 0 logs once when the check is installed:

.. code-block:: text

   Segment health check installed directory=/shared/nvrx job_id=123 task_id=7

A missing decision fails open quietly because the producer may not be running.
An unreadable path or unexpected file type emits a warning. An exclusion is
reported by the existing health-check error path. Other launchers do not
install the segment check or inspect the artifact.

Scheduler Publisher
-------------------

For Slurm arrays, the launcher can manage the publisher without a separate
batch-script startup command:

.. code-block:: bash

   ft_launcher \
     --ft-segment-health-check-dir /shared/job/segment-health-check \
     --ft-enable-segment-health-check-publisher \
     <other-options> train.py

The enable flag takes no value. Publishing defaults to disabled; the explicit
``--ft-disable-segment-health-check-publisher`` flag overrides an enabled
parser default without disabling the consumer. The flags are mutually exclusive.
Programmatic callers can set ``enable_segment_health_check_publisher`` on
``FaultToleranceConfig``. Enabling requires an absolute shared directory.

Only the rendezvous/TCP-store-host launcher starts the child, using the same
ownership decision as launcher-managed ``attrsvc``. Its array task and Slurm
process indices need not be zero. It needs ``SLURM_ARRAY_JOB_ID`` and
``SLURM_JOB_PARTITION``. Provide Slurm clients/plugins, cluster configuration,
authentication, user resolution, and scheduler connectivity in its environment
(for example through cluster-specific container bind mounts).

The child is independent of training worker process groups. Slurm reads and
shared-storage operations do not run in the launcher thread. The launcher
creates the process directly without waiting for readiness or a first
publication; spawn failures are logged without terminating training.
Scheduler observation, publication, and process management belong to the reusable
``attribution.scheduler_exclusions`` component. The launcher only selects the
owner, supplies job configuration, and starts/stops the manager; the existing
consumer continues enforcing the shared markers.
There is no supervisor thread or automatic restart if the child exits.
The publisher persists across training restarts and stops when its owning
launcher exits, with bounded process-group cleanup and a parent-death watcher.
There is no automatic owner failover if the owning launcher is lost or its task
is excluded. Use a standalone publisher when that lifetime is insufficient,
or when the rendezvous store is external and no participating launcher is its
host. This ownership rule does not change the consumer's allocation-unit
process-0 check.

For Slurm arrays, ``nvrx-scheduler-exclusions`` provides an independent Python
publisher using the deployed Bash producer's artifact and history contract.
It requires ``SLURM_ARRAY_JOB_ID``, ``SLURM_JOB_PARTITION``, and
``NVRX_SEGMENT_HEALTH_CHECK_DIR``, plus Slurm read access. It polls immediately
and then every 300 seconds by default. ``--once`` performs a single poll.

The standalone command is an alternative to launcher-managed publishing;
``attrsvc`` does not start either mode.
Run one publisher per Slurm array job, monitoring all running array tasks
(segments) under its ``SLURM_ARRAY_JOB_ID``, not one publisher per segment.
Separately submitted array jobs belonging to the same workload are not
automatically discovered. Do not run multiple publisher modes against the same
decision directory for that array job. The consumer remains unchanged.
Deployment prerequisites, failure handling, and configuration are documented in
``services/scheduler_exclusions/README.md`` in the source repository.
