Dmesg FACT node attribution
===========================

``nvrx-fact-agent`` can collect kernel ``dmesg`` output after a failed
fault-tolerance cycle, optionally write that same evidence to per-node files,
and submit it to FACT. ``ft_launcher`` only sends a local failed-cycle
notification to the agent and continues after ACK. The
existing application-log attribution service can also stop further restarts when
it returns a ``STOP`` recommendation.

There are two attribution paths with different contracts:

* ``--ft-attribution-endpoint`` is the application-log attribution path. It
  returns a job-level restart recommendation such as ``STOP`` or ``RESTART``.
* ``--ft-fact-url`` enables FACT node attribution from host evidence. FACT
  identifies suspect or faulty nodes; node reuse, avoid, or exclusion is a
  separate placement-policy decision.

Use this feature when worker failure logs alone are not enough to diagnose node
or GPU issues such as XIDs, kernel driver errors, or other host-level symptoms.

Enable dmesg evidence files
---------------------------

Dmesg evidence file output has three controls:

* ``--ft-fact-url`` enables the failed-cycle notification to
  ``nvrx-fact-agent``.
* ``health_logging.prefix`` or ``--ft-health-log-prefix`` sets the absolute
  output prefix.
* ``health_logging.dmesg.enabled`` or ``--ft-enable-health-log-dmesg`` asks
  ``nvrx-fact-agent`` to write the collected dmesg window to a file.

YAML configuration:

.. code-block:: yaml

   fault_tolerance:
     fact_url: http://fact.example.internal:8001/latest
     health_logging:
       prefix: /lustre/logs/job_health.log
       dmesg:
         enabled: true

Equivalent launcher flags:

.. code-block:: bash

   ft_launcher \
     --ft-fact-url http://fact.example.internal:8001/latest \
     --ft-health-log-prefix /lustre/logs/job_health.log \
     --ft-enable-health-log-dmesg true \
     ...

The prefix must be absolute. When ``--ft-fact-url`` is set, ``ft_launcher``
starts one local ``nvrx-fact-agent`` per node and talks to it over a private
UDS path.

Output files
------------

For cycle ``N``, ``ft_launcher`` derives a per-node dmesg evidence path from
the health-log prefix and passes it to the local ``nvrx-fact-agent``:

.. code-block:: text

   /lustre/logs/job_health.log -> /lustre/logs/job_health_dmesg_<node>_cycleN.log

The store-host ``nvrx-fact-agent`` writes the per-cycle FACT result beside the
dmesg evidence files using the ``fact`` source name:

.. code-block:: text

   /lustre/logs/job_health.log -> /lustre/logs/job_health_fact_cycleN.log

Each node writes only its own dmesg evidence file. FACT submission does not read
these files; the service POSTs the collected contents directly.

Cycle behavior
--------------

One failed-cycle notification maps to one evidence collection attempt:

* The launcher reaches restart-decision logic and records the failed cycle.
* After workers are stopped, the launcher sends ``cycle_failed`` to the local
  ``nvrx-fact-agent`` over UDS.
* The agent ACKs immediately, then collects a bounded recent dmesg window.
* The same collected text is written to the optional evidence file and POSTed to
  FACT.
* The agent publishes terminal status and ``observation_id`` to TCPStore.
* The store-host agent reads statuses from TCPStore and performs the FACT GET.

The default dmesg window is 12 minutes so NCCL timeout cases, where the
interesting kernel event may be roughly 10 minutes old, are still in scope.

FACT attribution
----------------

FACT attribution is separate from the application-log attribution service. The
launcher does not submit dmesg contents to FACT directly. Enable launcher
notification with ``--ft-fact-url``; the launcher starts the local
``nvrx-fact-agent`` process and passes that URL to it.

.. code-block:: bash

   ft_launcher \
     --ft-health-log-prefix /lustre/logs/job_health.log \
     --ft-enable-health-log-dmesg true \
     --ft-fact-url http://fact.example.internal:8001/latest \
     ...

``--ft-fact-url`` accepts either the service root or FACT API root
(``/latest``). The launcher sends a local UDS notification and continues after
ACK; the agent collects a bounded recent dmesg window, posts per-node
observations, and lets the store-host agent perform the FACT GET. FACT calls
are best-effort: failures are logged, but the launcher restart path continues.

Application-log restart recommendations
---------------------------------------

The existing application-log attribution service still uses
``--ft-attribution-endpoint`` and ``--ft-per-cycle-applog-prefix``. Its setup is
documented in :ref:`fault-tolerance-attribution-service`.

When that service returns a normalized recommendation:

* ``STOP`` blocks another restart for the failed cycle.
* ``RESTART`` and ``CONTINUE`` are logged but do not block the normal restart
  policy.
* Missing or unavailable results are treated as no recommendation.

This restart-policy behavior is independent of FACT. FACT produces node
attribution evidence from dmesg logs; NVRx policy can treat that evidence as a
suspect-node signal and escalate repeated matching suspicion to node exclusion.
Concrete health-check failures and no-show nodes remain the immediate exclusion
inputs, while the application-log attribution service decides whether the
launcher should stop retrying.

See also
--------

* :doc:`usage_guide` for the rest of the launcher workflow.
* :doc:`api/config` for the ``FaultToleranceConfig`` schema.
* ``src/nvidia_resiliency_ext/attribution/fact/fact_agent_design.md`` for the agent
  path that keeps dmesg evidence POSTs out of ``ft_launcher``.
