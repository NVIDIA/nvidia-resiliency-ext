Scheduler Node-State Service
****************************

This design describes the slow-path scheduler-state integration for
``ft_launcher``. The goal is to avoid split-brain with the cluster control
plane: if the scheduler already considers a node unusable, NVRx stops treating
that node as an eligible restart/rendezvous candidate.

This is separate from local health checks. Local health checks answer whether
the current launcher process is allowed to join rendezvous. Scheduler node
state answers whether a set of candidate nodes remains eligible from the
cluster control-plane point of view.

Intent
======

NVRx runs jobs that restart in place instead of requesting a fresh scheduler
allocation. This reduces restart latency, but it also means NVRx must
independently honor node state changes that a fresh allocation would have
avoided.

The scheduler learns that a node is bad from several producers, such as a
health-check system, manual operator action, a fault-attribution workflow, or a
cluster remediation service. NVRx does not integrate with each producer
directly. The durable signal for this slow path is scheduler-visible node state.

Current Scope
=============

The node-state service integration has four steps:

1. At cycle start, the NVRx-side client sends the selected active node list to
   the node-state service.
2. The service records the list for that cycle.
3. At cycle end or failure handling time, the client tells the service to close
   the cycle.
4. The service queries the scheduler in backend-sized batches and caches the
   bad or unknown nodes for rendezvous placement to consume before the next
   cycle is closed.

Service Responsibilities
========================

``nvrx-nodestatesvc`` exposes scheduler-visible node state over HTTP. The
backend is Slurm ``sinfo``.

The service owns:

* registering the node list for a cycle
* querying scheduler state for a registered cycle at cycle end
* classifying scheduler-bad states
* treating nodes missing from scheduler output as unknown and bad
* returning compact decision data plus audit metadata for bad or unknown nodes

Cycle Start
===========

The rendezvous host registers the active node list when a cycle starts.

.. code-block:: http

   POST /v1/cycles/start

.. code-block:: json

   {
     "job_id": "job-123",
     "cycle_id": "3",
     "nodes": ["nvl72033-T18", "nvl72034-T18"]
   }

The service stores a deduplicated in-memory entry:

.. code-block:: text

   (job_id, cycle_id) -> nodes, started_at, ended_at

The service keys cycle state by ``job_id`` and ``cycle_id`` together.
``job_id`` is the NVRx job or run namespace. ``cycle_id`` is the NVRx cycle
number within that namespace.

Registration response:

.. code-block:: json

   {
     "job_id": "job-123",
     "cycle_id": "3",
     "registered_nodes": 2
   }

Cycle End
=========

At cycle end, the rendezvous host tells the service that the cycle has ended.
The service accepts the request, marks the cycle as materializing, and starts a
background query for the registered nodes. The Slurm backend performs this
query in batches, controlled by the service ``--slurm-batch-size`` setting, so
a cycle with thousands of nodes does not require one oversized ``sinfo``
request.

.. code-block:: http

   POST /v1/cycles/end

Response:

.. code-block:: json

   {
     "job_id": "job-123",
     "cycle_id": "3",
     "accepted": true,
     "materializing": true,
     "registered_nodes": 2
   }

The response does not include the scheduler-state decision. The client uses the
cycle-status ``GET`` to observe the materialized result.

Cycle Status
============

After cycle end, the client reads the cached scheduler state for the registered
cycle. This ``GET`` is read-only and does not query Slurm.

.. code-block:: http

   GET /v1/cycles/3/node-states?job_id=job-123

Response:

.. code-block:: json

   {
     "job_id": "job-123",
     "cycle_id": "3",
     "requested_nodes": 2,
     "bad_nodes": ["nvl72034-T18"],
     "unknown_nodes": [],
     "nodes": [
       {
         "node": "nvl72034-T18",
         "state": "DRAIN",
         "raw_state": "drain",
         "reason": "gres/gpu failure",
         "slurm_visible": true,
         "bad": true
       }
     ]
   }

``bad_nodes`` is the decision field. ``unknown_nodes`` is the subset of bad
nodes that the scheduler did not return. ``nodes`` contains scheduler metadata
only for bad or unknown nodes; healthy nodes are omitted from this response by
default.

If the cycle id is unknown, the service returns ``404``. If the cycle has been
registered but not ended yet, or the background query is still materializing,
the service returns ``409 cycle_status_not_ready``.

State Classification
====================

The bad-state set is:

.. code-block:: text

   DOWN
   DRAIN
   DRAINED
   DRAINING
   FAIL
   FAILING
   NO_RESPOND

Nodes requested by the client but not returned by the scheduler are reported
as:

.. code-block:: json

   {
     "state": "UNKNOWN",
     "raw_state": "UNKNOWN",
     "reason": "node was not returned by sinfo",
     "slurm_visible": false,
     "bad": true
   }

This is conservative. A fresh scheduler allocation would not normally rely on a
node that the scheduler cannot resolve.

Client Behavior
===============

NVRx uses a single FT-side client module:

.. code-block:: text

   nvidia_resiliency_ext/fault_tolerance/node_state.py

The module exposes the HTTP client for cycle registration and cycle status.
The user-visible configuration is the service URL:

.. code-block:: text

   --ft-node-state-url http://host:8000

If the URL is unset, the feature is disabled. The client uses internal timeout
and retry defaults.

The rendezvous host calls:

.. code-block:: text

   POST /v1/cycles/start after active ranks are assigned for cycle N
   POST /v1/cycles/end when cycle N ends and the worker group is restarting
   GET  /v1/cycles/<cycle_id>/node-states?job_id=<job_id> before cycle N+1 placement

``POST /v1/cycles/end`` enqueues the background Slurm query and returns quickly.
The rendezvous round-open path only sends this POST and returns. It does not
wait for scheduler state.

The follow-up ``GET`` reads the cached result for the ended cycle so the next
placement / rendezvous decision does not issue a second scheduler query. The
store-host close-round loop uses short HTTP probes within a bounded decision
budget. If the cycle-end request is still materializing, placement waits in
that close-round loop until the result is ready or the budget expires.

The node list comes from the rendezvous node address. The service queries Slurm,
so those names must match the scheduler-visible node names. On clusters where
``socket.getfqdn()`` returns a fully qualified domain name or another value that
does not match Slurm, use the existing ``ft_launcher`` ``--local-addr`` argument
on each launcher process, set to that node's Slurm name. For example, in an
``srun``-launched shell, pass ``--local-addr "$(hostname -s)"`` so each node
advertises its own short hostname.

Failure handling is fail-open: if the service is unavailable or a request times
out, NVRx logs the failure and does not avoid any nodes based on scheduler
state for that cycle.

Placement Integration
=====================

The rendezvous store host consumes the cycle status:

.. code-block:: text

   cycle N status -> bad_nodes + unknown_nodes
                  -> direct node hard avoids for round N+1
                  -> replacement-group hard avoids when prior membership maps the node
                  -> active-rank selection excludes avoided nodes/groups

Avoided participants that still join the next rendezvous are assigned standby
ranks instead of active ranks. If the remaining eligible participants cannot
satisfy the rendezvous constraints, the store host keeps waiting and eventually
uses the existing rendezvous timeout/failure path; it does not fall back to
scheduler-bad nodes.
