Restart Agent
====================

The restart agent is an experimental NVRx attribution analyzer that
turns one interleaved distributed-training log and optional compact restart
history into ``STOP`` or ``RESTART`` guidance.

The implementation lives in:

.. code-block:: text

   src/nvidia_resiliency_ext/attribution/restart_agent/

Quick Start
-----------

Run deterministic L0/history/L4 analysis without an LLM:

.. code-block:: bash

   python3 -m nvidia_resiliency_ext.attribution.restart_agent.cli \
     /absolute/path/to/cycle.log \
     --job-id job-123 \
     --cycle-id 1 \
     --result-json /absolute/path/to/result.json \
     --trace-json /absolute/path/to/trace.json \
     --summary

The command prints the versioned result JSON to stdout. ``--result-json``
stores the public result and ``--trace-json`` stores stage evidence, timing,
provider events, and policy provenance. A missing or empty log produces a
degraded restart-biased result rather than invoking a model.

The Python facade returns the result and invocation-owned artifacts together:

.. code-block:: python

   from nvidia_resiliency_ext.attribution.restart_agent import (
       RestartAgent,
       RestartAgentRequest,
   )

   run = RestartAgent().run(
       RestartAgentRequest(
           log_path="/absolute/path/to/cycle.log",
           job_id="job-123",
           cycle_id=1,
       )
   )
   print(run.result.decision)

Target Stateful Runtime
-----------------------

The next runtime increment adds a transport-independent
``RestartAgentRuntime`` around the stateless analysis pipeline. It will own
an ``AttemptRecordStore`` containing bounded in-memory history for the current
process, enabled by default with 10
attempts per exact job and 3,000 records total. Product configuration may
disable history or override either bound. The same neutral ``AttemptRecord``
type represents an attempt while current and when later selected in a
``PriorAttemptView``. Each record retains explicit progress facts, including first/last
iteration and checkpoint-save markers, so later policy can distinguish early
failure from an attempt that completed observable work. It also contains one
required deterministic failure block and a route-keyed list of completed L2
enriched blocks. MVP prior-attempt comparison uses deterministic blocks only.

The initial record is assembled from L0 and, when its deterministic identity is
eligible, committed before fallback publication. L2 may add or replace a
route-keyed enriched block while the record is open. Model output unfinished at the analysis deadline is abandoned and
cannot mutate the closed record. Reanalysis of the same job and cycle replaces
its record; an actual workload restart uses the next cycle id and appends a
record.

Library/unit tests will use ``AttemptRecordControl`` to seed, inspect, and clear
the same in-memory store used by runtime analysis. For manual testing, the CLI
may read a JSON-array fixture through ``--attempt-records-json-in`` and
explicitly export the resulting records through
``--attempt-records-json-out``. These are editable test fixtures, not automatic
persistence. MCP is a later thin adapter over this runtime and need not expose
history operations in its product API. These interfaces are specified in
``docs/design/attribution/restart_agent/RUNTIME.md`` and are not implemented by
the current terminal facade yet.

Model Routes
------------

Use ``--config`` for parallel model routes. Start from
``examples/attribution/restart_agent.json`` and set every route's
``credential_ref`` environment variable to a readable API-key file path. The
configuration owns route endpoints, request limits, tool advertisement,
reasoning controls, retries, and the whole-analysis deadline:

.. code-block:: bash

   export LLM_API_KEY_FILE=/secure/path/to/key
   export LLM_API_KEY_OLD_FILE=/secure/path/to/second-key
   python3 -m nvidia_resiliency_ext.attribution.restart_agent.cli \
     /absolute/path/to/cycle.log \
     --config examples/attribution/restart_agent.json \
     --result-json /absolute/path/to/restart_agent.result.json \
     --trace-json /absolute/path/to/restart_agent.trace.json \
     --fallback-json-out /absolute/path/to/fallback.json \
     --summary

Export-controlled workloads must use an authorized ECCN-compliant route on
the Regulated Inference Hub. Keys and resolved secret values are not emitted
in results or traces.

The canonical engineering specifications start at:

.. code-block:: text

   docs/design/attribution/restart_agent/DESIGN.md

The analyzer builds a deterministic evidence bundle before any model call. An
optional profile-declared LLM/tool stage returns structured current-log
evidence, while deterministic client policy owns history comparison, retry-rule
and retry-budget evaluation, and the final action. While L1 runs, the analyzer
also prepares a deterministic
L0/history fallback candidate before L1 begins. A configurable whole-analysis
deadline returns that candidate if enriched model evidence is not ready; a
service can use the same candidate at the NVRx-owned deadline. Progressive
analysis is a required but currently unimplemented follow-up that will shift
evidence work into the running cycle.

For local multi-model runs, callers can declare canonical output paths for the
deterministic fallback and each route result/trace. Each completed artifact is
published directly at its final path. A separate incremental directory contains
only an atomic lifecycle status snapshot and append-only event stream; the
canonical batch result remains the final artifact.
The complete L0 bundle and Decision Evidence can be written to canonical JSON
files as soon as L0 finishes. Model-backed execution can additionally publish
the L0B model view before model routes complete.

The separate eval corpus and N-model comparison harness live under
``tools/restart_agent_eval/`` when the harness development change is checked
out. They qualify product profiles but do not ship in the NVRx package or
provide a second runtime implementation.
