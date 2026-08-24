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
disables consumption. The equivalent YAML field is
``fault_tolerance.segment_health_check_dir``.

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
