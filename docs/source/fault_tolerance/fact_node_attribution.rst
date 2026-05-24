FACT dmesg evidence collection
==============================

Use this feature to let FACT inspect recent host ``dmesg`` after a failed
fault-tolerance cycle. FACT filters the logs for node-level symptoms such as
XIDs, NVIDIA driver messages, and kernel faults, then returns suspect or faulty
nodes.

``ft_launcher`` starts one local ``nvrx-fact-agent`` per node when
``--ft-fact-url`` is set. On a failed cycle, the launcher sends a local UDS
notification to the agent and continues after ACK. FACT output is node-level
evidence; it does not by itself stop the job or change placement.

``--ft-attribution-endpoint`` is the separate application-log path for
job-level restart recommendations such as ``STOP`` or ``RESTART``. It is not
used for FACT dmesg submission.

Configuration
-------------

The current-cycle FACT path is enabled by ``--ft-fact-url``. Artifact output is
optional:

* ``health_logging.prefix`` or ``--ft-health-log-prefix`` sets the absolute
  output prefix.
* ``health_logging.dmesg.enabled`` or ``--ft-enable-health-log-dmesg`` queues
  the collected dmesg window to a shared per-cycle dmesg file.
* ``health_logging.fact_result.enabled`` or
  ``--ft-enable-fact-result-artifact`` queues per-node FACT submission records
  and the store-host FACT result record to a shared JSONL file.

When either artifact is enabled, the launcher gRPC log funnel must also be
enabled with ``--ft-per-cycle-applog-prefix`` and
``--ft-enable-log-server true``. The root log server is the only writer for the
shared FACT artifacts.

YAML configuration:

.. code-block:: yaml

   fault_tolerance:
     fact_url: http://fact.example.internal:8001/latest
     health_logging:
       prefix: /lustre/logs/job_health.log
       dmesg:
         enabled: true
       fact_result:
         enabled: true

Equivalent launcher flags:

.. code-block:: bash

   ft_launcher \
     --ft-per-cycle-applog-prefix /lustre/logs/train.log \
     --ft-enable-log-server true \
     --ft-fact-url http://fact.example.internal:8001/latest \
     --ft-health-log-prefix /lustre/logs/job_health.log \
     --ft-enable-health-log-dmesg true \
     --ft-enable-fact-result-artifact true \
     ...

``--ft-fact-url`` accepts either the FACT service root or the FACT API root
(``/latest``).

Cycle Flow
----------

One failed-cycle notification maps to one evidence collection attempt:

* After workers are stopped, the launcher sends ``cycle_failed`` to the local
  ``nvrx-fact-agent`` over UDS with the failed cycle id and cycle start
  timestamp.
* The agent ACKs immediately, collects a bounded recent dmesg window, queues
  the optional dmesg artifact, and POSTs the collected text to FACT.
* The FACT workload ``job_start_time`` uses the actual cycle start timestamp.
  The dmesg observation window remains the recent collection window.
* TCPStore carries only ``attributor_id`` and completion count.
* Completion count means a node reached any terminal local outcome: successful
  FACT submission, empty dmesg, collection failure, or FACT POST failure. It
  does not mean FACT accepted evidence from that node.
* The store-host agent waits for completion count or a deadline, performs the
  FACT GET, and queues the optional result artifact.

The default dmesg window is 12 minutes so NCCL timeout cases, where the
interesting kernel event may be roughly 10 minutes old, are still in scope.

The UDS ACK only means the local agent accepted the request. Dmesg collection,
FACT POST/GET, TCPStore completion, and gRPC artifact drain are best-effort.
They may be missing or partial if the launcher, agent, store-host, or gRPC log
funnel exits before they finish. When FACT ingestion succeeds, FACT /
Elasticsearch is the durable observability source; local artifacts are
postmortem evidence.

Artifacts
---------

For cycle ``N``, artifact paths are derived from the health-log prefix:

.. code-block:: text

   /lustre/logs/job_health.log -> /lustre/logs/job_health_dmesg_cycleN.log
   /lustre/logs/job_health.log -> /lustre/logs/job_health_fact_cycleN.log

The dmesg artifact is one shared file per failed cycle. Production collection
prefixes each dmesg line with the source node name, so per-node inspection can
filter the shared file by node.

The result artifact is JSONL. Leaf agents queue one record for their local FACT
submission result, including ``observation_id`` when FACT returns one. The
store-host agent queues a record containing the full ``FactAttributionResult``
plus ``run_id``, ``cycle``, ``job_id``, expected/completed node counts, and the
history-policy ``avoid_nodes`` decision when available.

All artifact records and dmesg chunks go through the launcher gRPC log funnel.
Record/chunk order is not guaranteed, but one queued chunk should not interleave
with another. FACT submission does not read these files; the service receives
the collected contents directly.

FACT History and Node Reuse
---------------------------

The current-cycle FACT result answers: which nodes look suspect for this failed
cycle? FACT history answers: has the same node appeared as a suspect node in
recent FACT records?

The controls above enable current-cycle dmesg collection, FACT submission, and
optional artifacts. They do not, by themselves, make a job-level ``STOP`` or
``RESTART`` decision. For node reuse, NVRx only evaluates nodes that FACT marks
suspect in the current cycle. For those nodes, it combines a short-lived
in-memory NVRx hot cache with optional durable FACT history. For example, if the
current failed cycle implicates ``node-a`` and an earlier cycle in the same
NVRx run also implicated ``node-a``, NVRx may avoid assigning active ranks to
``node-a`` on the next retry when enough other joined nodes are available.

Durable FACT history is optional and extends the same decision with failures
that happened before this NVRx process started:

.. code-block:: text

   --ft-fact-history-es-url <url>
   --ft-fact-history-es-auth-file <path>

The behavioral defaults should normally be left unchanged. The concrete FACT
history index or backend collection is deployment-specific and should be
provided by the FACT deployment.

.. code-block:: text

   fact_history_lookback = 14d
   fact_history_max_candidate_nodes = 16
   fact_history_query_timeout = 30s
   fact_policy_ready_timeout = 60s
   min_repeat_count_for_avoid = 2
   max_attribution_avoids_per_cycle = 1

This is not a hard exclusion and it is not a job-level ``STOP`` decision. The
policy must fail open: if FACT attribution, FACT history, or the local policy
answer is unavailable, late, or ambiguous, rendezvous proceeds without avoiding
nodes based on FACT repeat history. Concrete health-check failures and nodes
that cannot rejoin remain the immediate hard exclusion inputs.

See Also
--------

* :doc:`usage_guide` for the rest of the launcher workflow.
* :ref:`fault-tolerance-attribution-service` for application-log restart
  recommendations.
* :doc:`api/config` for the ``FaultToleranceConfig`` schema.
* ``src/nvidia_resiliency_ext/attribution/fact/fact_integration_design.md`` for
  the internal FACT agent, history, and avoid-node design.
