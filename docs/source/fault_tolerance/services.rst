Optional health checks and attribution
======================================

These services extend the basic section-monitoring and restart workflow. Start
with :doc:`examples/megatron_gpt` before enabling site-specific services.

Node health check service
-------------------------

The launcher can query an optional node-local health check service before workers enter
rendezvous. A practical public deployment model is to reuse NVIDIA Base Command
Manager (BCM) Slurm prolog or epilog health checks behind an ``nvhcd``-compatible
daemon. The NVRx integration point is service-compatible: any equivalent daemon
can be used if it implements the expected gRPC API over a Unix domain socket
(UDS).

To enable the external node health check with BCM:

* Build and deploy an ``nvhcd``-compatible daemon on every allocated node.
* Configure the daemon to invoke a BCM health check script, or a wrapper around
  an existing BCM prolog or epilog health check.
* Ensure the wrapper translates the BCM result into JSON with an empty
  ``failed_checks`` array for a healthy node and a non-empty array containing
  the actionable check names for an unhealthy node.
* Make the daemon's UDS visible from the job environment or training container.
* Pass the socket path to ``ft_launcher`` with ``--ft-node-health-check-endpoint``
  (alias: ``--ft-node_health_check_endpoint``).

For protocol details, see the ``nvhcd`` protobuf schema at
``src/nvidia_resiliency_ext/shared_utils/proto/nvhcd.proto``. The functional test
server at ``tests/fault_tolerance/func/nodehc_service.py`` is a minimal example
of a UDS gRPC service that implements this API.

Example:

.. code-block:: bash

   ft_launcher \
     --ft-node-health-check-endpoint unix:///var/run/nvhcd.sock \
     train.py

Endpoint behavior:

* UDS endpoints are supported. The value can be a path such as ``/var/run/nvhcd.sock``
  or a ``unix://`` URI such as ``unix:///var/run/nvhcd.sock``.
* If the endpoint is omitted, NVRx skips the external node health check.
* If the gRPC client dependencies are unavailable, the UDS socket is missing, or a
  connectivity error occurs, NVRx treats the external check as unavailable and does
  not fail the job for that reason.
* Explicit failures reported by the service mark the node unhealthy.

Compatible service contract:

* The service must implement ``HealthCheckService.RunHealthCheck`` from the NVRx
  ``nvhcd`` protobuf API and listen on the configured UDS.
* NVRx calls the service with the configured health check arguments, which
  default to ``["--no-slurm", "--group", "prolog", "epilog", "logs", "gpu"]``.
* To report an unhealthy node, return ``success=false``. For successful
  responses, NVRx treats JSON in ``output`` with a non-empty ``failed_checks``
  array as unhealthy. ``fail_count`` is informational and may include checks
  moved to ``failed_checks_ignored`` by site policy.
* NVRx ignores unavailable or unusable health check signals. A missing endpoint,
  missing gRPC dependency, connectivity error, non-JSON ``output``, missing
  ``failed_checks``, or invalid ``failed_checks`` value in a successful response
  is logged and does not mark the node unhealthy.

Example ``nvhcd`` configuration for BCM:

.. code-block:: yaml

   socket_path: /var/run/nvhcd.sock
   healthcheck_path: /usr/local/sbin/nvrx-bcm-healthcheck-wrapper.sh
   log_level: info
   timeout: 120

Start the daemon on each node with this configuration, for example as a node-level
service or directly with:

.. code-block:: bash

   nvhcd -config /etc/nvhcd/config.yaml

If the training job runs inside a container, bind mount the UDS path into the
container so that ``ft_launcher`` can reach the daemon.

The wrapper can call the same reusable health check entry point that BCM uses for
Slurm prolog or epilog validation, then normalize the result for NVRx. When using
``nvhcd``, the gRPC request ``args`` are forwarded to the configured
``healthcheck_path`` as command-line arguments, so the configured NVRx health
check arguments will appear in the wrapper's ``"$@"``. For example:

.. code-block:: bash

   #!/usr/bin/env bash
   set -euo pipefail

   if /path/to/bcm-healthcheck "$@"; then
     printf '{"fail_count": 0, "failed_checks": []}\n'
     exit 0
   else
     printf '{"fail_count": 1, "failed_checks": ["bcm_healthcheck"]}\n'
     exit 1
   fi

Because NVRx invokes this check during rendezvous rather than during Slurm's
actual prolog or epilog phase, the wrapper should also provide any inputs that
the BCM script expects from the Slurm lifecycle environment, or call a reusable
health check entry point from the local BCM deployment that does not depend on
those lifecycle-only variables.

This lets NVRx run the same class of node validation during InJob restart
rendezvous that cluster administrators may already run at Slurm prolog or epilog
time.

Distributed storage health check (Lustre + NFS)
-----------------------------------------------

The launcher can perform a distributed storage health check before rendezvous.
By default it is disabled. When enabled (via CLI or YAML), it:

* Verifies Lustre health via ``/sys/fs/lustre/health_check`` (fails if not healthy).
* Discovers distributed mount targets and checks that each mount is reachable.

* ``--ft-enable-dist-storage-healthcheck`` (alias: ``--ft_enable_dist_storage_healthcheck``)

  - Accepts a boolean-like value only to enable the mount checks
    (e.g., ``--ft-enable-dist-storage-healthcheck true``).

Storage path health check
-------------------------

Validate specific absolute paths for existence and basic readability before rendezvous.

* CLI: ``--ft-storage-health-check-path`` (alias: ``--ft_storage_health_check_path``)
  - Accepts a comma-separated list of absolute paths (each starting with ``/``).
  - Example: ``--ft-storage-health-check-path '/data/checkpoints,/mnt/dataset'``
* YAML: ``storage_healthcheck_path`` under the ``fault_tolerance`` section

.. code-block:: yaml

   fault_tolerance:
     # Comma-separated absolute paths
     storage_healthcheck_path: "/data/checkpoints,/mnt/dataset"

Validation behavior:
  - Files: attempts to read a small block (up to 4KB)
  - Directories: lists directory contents
  - Other existing types (e.g., devices/symlinks): performs ``stat`` access


Segment Health Check
--------------------

Set ``--ft-segment-health-check-dir <ABSOLUTE_PATH>`` or
``fault_tolerance.segment_health_check_dir`` to consume segment health
decisions before rendezvous. Omitting the setting disables the check.

See the :doc:`Segment Health Check integration guide
<integration/segment_health_check>` for the consumer contract.


Attribution service integration
-------------------------------

Per-cycle application logs do not enable attribution by themselves. To enable attribution, set
``--ft-attribution-endpoint``. The endpoint value ``localhost`` makes ``ft_launcher`` run the
attribution service on the TCPStore host; other endpoints are treated as externally managed
attribution services.
External endpoints may use schemes such as ``http://``, ``grpc://``, or ``unix://``. The current
InJob attribution client submits logs over HTTP(S); non-HTTP endpoint strings are preserved but do
not add a new transport implementation.
If ``--ft-attribution-endpoint`` is set, ``--ft-per-cycle-applog-prefix`` is required because the
attribution service analyzes the per-cycle application logs.

The service code and dependencies are included in the NVRx wheel. Install the wheel before running
a launcher-managed attribution service:

.. code-block:: bash

   python -m pip install nvidia_resiliency_ext-<version>-<tags>.whl

* CLI:

  - ``--ft-attribution-endpoint <ENDPOINT>`` (alias: ``--ft_attribution_endpoint``), default disabled
  - ``--ft-attribution-llm-api-key-file <PATH>`` (alias: ``--ft_attribution_llm_api_key_file``)
  - ``--ft-attribution-llm-base-url <URL>`` (alias: ``--ft_attribution_llm_base_url``)
  - ``--ft-attribution-llm-model <MODEL>`` (alias: ``--ft_attribution_llm_model``)
  - ``--ft-attribution-analysis-backend {lib}`` (alias: ``--ft_attribution_analysis_backend``)
  - ``--ft-attribution-stop-action {log,no-restart}`` (alias: ``--ft_attribution_stop_action``), default ``log``
  - ``--ft-attribution-startup-timeout <SECONDS>`` (alias: ``--ft_attribution_startup_timeout``), default ``20``
  - ``--ft-attribution-export-url <URL>`` (alias: ``--ft_attribution_export_url``)

  The managed attribution app-log directory is derived from
  ``dirname(realpath(--ft-per-cycle-applog-prefix))``. When ``--ft-nvrx-logfile`` is set, the
  managed service stdout/stderr log is derived from it as ``*_attribution.log`` and written
  beside it. Without ``--ft-nvrx-logfile``, the log is named from the application prefix and
  written under an ``nvrx/`` subdirectory of the application-log directory. The managed service
  listens on ``127.0.0.1:50050`` and is exposed to the InJob client as
  ``http://localhost:50050``.

  The managed attribution API key must come from ``--ft-attribution-llm-api-key-file`` or inherited
  ``LLM_API_KEY_FILE``. If neither points to a readable file, the TCPStore-host launcher fails
  before starting the attribution service.

  To export managed attribution results, pass ``--ft-attribution-export-url`` or
  set ``attribution_export_url`` in the fault tolerance YAML config.

  Launcher-managed attribution defaults to the ``lib`` backend, which runs the Restart Agent
  directly in the attrsvc process. The backend flag may be left unset to use that default or set
  explicitly to ``lib``. The optional MCP/LogSage analysis path is no longer selectable from
  launcher-managed attribution.

  Attribution never delays a restart. When a cycle ends, the rendezvous host requests
  terminal analysis and immediately closes the next round, so the workload restarts while
  analysis is still running. A background poller on the rendezvous host fetches the verdict
  and, if attribution recommends stopping, terminates the job at that point -- even if the
  workload has already advanced one or more cycles past the analyzed one. A verdict that
  never arrives leaves the job running.

  The stop decision is global, not per-cycle: the first STOP verdict observed ends the job
  regardless of which cycle produced it.

  Only one terminal analysis is in flight at a time. If further cycles fail while a verdict
  is still outstanding, their terminal analysis is skipped rather than replacing the one
  already running. This keeps a fast crash loop from repeatedly discarding an analysis
  before it can finish, which would otherwise leave the job with no verdict at all.

  **Acting on a STOP verdict is opt-in.** With the default
  ``--ft-attribution-stop-action log``, attribution runs in full and every verdict is
  polled, logged and recorded, but a STOP never terminates anything. Set
  ``--ft-attribution-stop-action no-restart`` to make a STOP end the job.

  The default is deliberate because the two kinds of attribution mistake are not equally
  expensive. A missed STOP is bounded: the job crash-loops until ``--max-restarts`` runs
  out, and the progress tracker independently catches the stuck case. An enforced false
  STOP is not bounded: the job ends with the no-restart exit code and stays down until a
  human resubmits it. Run in ``log`` mode first, review the recorded verdicts against what
  actually happened on your workloads, and enable ``no-restart`` once you trust the
  precision.

  In ``log`` mode every failed cycle is analyzed, not just the first one that produces a
  STOP, so the verdicts accumulate over the life of the job. Each unenforced STOP is logged
  at ``ERROR``, says explicitly that it is not being enforced, and carries a running count
  of how many STOP verdicts have been seen so far. Comparing that count against the number
  of failed cycles is the measurement that tells you whether to enable ``no-restart``.

  When enforcement is on, every launcher in the job exits with the no-restart exit code
  rather than the generic failure code ``1``. See :ref:`ft-no-restart`.

  ``ft_launcher`` sends job metadata with each attribution submission: ``user`` is read from
  ``SLURM_JOB_USER`` or ``USER``, and ``job_id`` is read from ``SLURM_ARRAY_JOB_ID`` or
  ``SLURM_JOB_ID``. If no corresponding environment variable is set, that field is omitted from
  the submission payload.

  Example:

  .. code-block:: bash

     ft_launcher \
       --ft-per-cycle-applog-prefix /lustre/job123/train.log \
       --ft-nvrx-logfile /lustre/job123/launcher.log \
       --ft-attribution-endpoint localhost \
       --ft-attribution-llm-api-key-file /secure/llm_api_key \
       --ft-attribution-llm-base-url https://integrate.api.nvidia.com/v1 \
       --ft-attribution-llm-model nvidia/nemotron-3-super-120b-a12b \
       --ft-attribution-export-url https://dataflow.example.test/dataflow2/example-index/posting \
       train.py

  To use an externally managed attribution service instead, specify an explicit endpoint:

  .. code-block:: bash

     ft_launcher \
       --ft-per-cycle-applog-prefix /lustre/job123/train.log \
       --ft-nvrx-logfile /lustre/job123/launcher.log \
       --ft-attribution-endpoint http://attribution.service.internal:8000 \
       train.py
