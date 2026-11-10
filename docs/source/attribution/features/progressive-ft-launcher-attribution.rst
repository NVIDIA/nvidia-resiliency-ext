Progressive FT Launcher Attribution
===================================

Summary
-------

The ``ft_launcher`` integration submits one cycle-unique application-log path
to attrsvc. Terminal analysis is the production default. Optional progressive
work can move L0A parsing earlier while the workload is running, but it does not
change the final evidence, policy, or response contract.

The Restart Agent's internal progressive state and terminal-equivalence
requirements are specified in
``docs/design/attribution/restart_agent/PROGRESSIVE.md``. The complete HTTP and
NVRx client lifecycle is specified in
``docs/design/attribution/restart_agent/ATTRSVC_INTEGRATION.md``.

Runtime Flow
------------

1. At cycle start, NVRx creates the expected per-cycle log path and sends
   ``POST /logs`` with ``analysis_intent="progressive"``.
2. Attrsvc registers the attempt. By default it performs no pre-end analysis.
   When progressive precomputation is enabled, it incrementally performs L0A
   work over complete log lines.
3. When the cycle fails, NVRx sends the same path with
   ``analysis_intent="terminal"``. Attrsvc schedules authoritative analysis in
   the background after a bounded log-convergence drain.
4. NVRx restarts the workload without waiting for the recommendation. Its
   background poller sends ``GET /logs?wait=false`` until attrsvc reports
   ``completed`` or the request is abandoned.
5. A completed STOP recommendation terminates whichever cycle is running then.
   Every other completed recommendation is non-stopping at the NVRx boundary.

``GET /logs?wait=false`` is a probe. It never starts analysis. Terminal work is
started only by an accepted terminal POST.

Analysis Intents
----------------

``track_only``
   Register the path without starting verdict-producing work.

``progressive``
   Register the path. If both
   ``NVRX_ATTRSVC_RESTART_AGENT_PROGRESSIVE_ENABLED=true`` and the service's
   progressive policy permit it, schedule non-authoritative L0A precomputation.
   Otherwise this has registration-only behavior.

``terminal``
   Start authoritative background analysis for the registered attempt. The
   terminal path reads from byte zero when no usable progressive state exists.

Terminal Equivalence
--------------------

Progressive and terminal-only execution use the same byte-range reader,
incremental line decoder, L0 observation index, and canonical L0A finalizer.
Progressive chunk boundaries and polling metadata must not change evidence,
identity, progress facts, or the final recommendation.

At terminal submission, attrsvc waits for the live log to converge within a
bounded window, ingests the unread tail, and finalizes one source boundary. A
checkpoint is reused only when it was built from that exact boundary. Missing,
stale, failed, or disabled progressive state falls back to the same terminal
path from byte zero.

Client Failure Behavior
-----------------------

POST
   NVRx makes one POST attempt with a two-second client timeout. It installs a
   pollable request only after terminal POST returns ``2xx``. If terminal POST
   is not accepted, that cycle is not polled, so a nonexistent request cannot
   pin the first-come-first-served slot.

   The immediately following optional progressive POST is also suppressed once.
   Since it normally follows the failed terminal POST at the next cycle start,
   this avoids a second likely failure and up to another two seconds on the FT
   path. Terminal POST is never suppressed.

GET
   ``pending`` and ``in_flight`` mean attrsvc still owns the accepted request.
   ``completed`` closes it. If attrsvc restarts and loses process-local request
   state, a later GET returns ``404``; NVRx abandons that cycle and releases the
   slot so another failed cycle can be analyzed.

   Permanent request errors are abandoned immediately. Availability failures,
   including transport errors, timeouts, malformed responses, and transient
   HTTP statuses, are abandoned after three consecutive failures total. A valid
   ``pending`` or ``in_flight`` response resets that count.

External Attrsvc
----------------

NVRx can use the same HTTP lifecycle with an attrsvc on another host as long as
both processes can access the same application-log path. The attrsvc Slurm
deployment uses a supervisor daemon as its batch payload so transient attrsvc
process crashes are restarted on the same allocated CPU node and endpoint.

The supervisor preserves endpoint reachability, not service memory. Active
request and Restart Agent history state are process-local and are lost when the
attrsvc process restarts. The GET request-loss behavior above prevents such a
lost request from blocking all later cycles.

Configuration
-------------

``NVRX_ATTRSVC_RESTART_AGENT_PROGRESSIVE_ENABLED``
   ``false`` by default. Set to ``true`` to allow pre-end L0A work.

``NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS``
   ``all_explicit`` permits explicit progressive requests; ``off`` disables
   them. This policy is subordinate to the Restart Agent progressive switch.

Terminal log convergence is configured by the attrsvc Restart Agent quiet,
maximum-wait, and polling settings. These service settings apply to attrsvc
terminal submissions. Direct CLI and library analysis of an already complete
file does not use the service convergence wait.

Verification
------------

Coverage must establish that:

* terminal-only and progressive execution produce equivalent canonical L0A and
  final results for the same bytes;
* terminal POST acceptance precedes installation of the NVRx poll slot;
* an unaccepted terminal POST suppresses only the next progressive POST;
* request loss, permanent rejection, and bounded GET availability failures
  release the slot for later cycles;
* ``pending`` and ``in_flight`` reset consecutive GET availability failures;
* a completed STOP is latched while other completed actions are non-stopping;
  and
* the Slurm supervisor preserves the allocated node and endpoint across bounded
  attrsvc process restarts and forwards Slurm shutdown signals without restart.
