Progressive FT Launcher Attribution
===================================

Summary
-------

Progressive FT launcher attribution reduces restart-decision latency by starting
log analysis when a fault-tolerance cycle starts, while the workload is still
running. When the cycle ends and ``ft_launcher`` asks attrsvc for a stop/restart
decision, attrsvc should reuse the analysis work already completed for the same
per-cycle application log and only perform the final missing work needed to
return an authoritative decision.

The feature is aimed at the ``ft_launcher`` integration path. The cluster-wide
service path may expose progressive behavior for validation, but it is not a
user-facing requirement there because service-mode attribution is post-mortem
and is not latency sensitive.

Design conclusion: attrsvc should stay mostly as plumbing. The progressive
option is offered at the NVRx-owned loganalysis tool boundary, and that tool can
later decide whether to delegate progressive state to LogSage or drive multiple
LogSage calls over the file flow. Until the LogSage contract is available,
``POST /logs`` can carry the intent and the loganalysis tool can report
``unsupported`` without changing ``GET /logs`` behavior.

Problem
-------

Today, ``ft_launcher`` submits a per-cycle log path near cycle start with
``POST /logs`` and later requests attribution with ``GET /logs`` after the
workload has stopped. The ``POST`` path records/tracks the log, but the
``GET`` path triggers the full LogSage attribution pipeline. For large
application logs this makes ``ft_launcher`` wait for parsing and LLM analysis
after the workload has already died, increasing the time before the launcher can
act on a stop/restart decision.

The per-cycle log path is already a stable correlation key. Both calls refer to
the same path produced by the cycle log naming convention, for example
``<applog>_cycle<id>.log``.

Goals
-----

* Start attribution pre-work from the ``ft_launcher`` cycle-start ``POST``.
* Preserve the existing ``GET /logs`` decision contract for ``ft_launcher``.
* Make the final ``GET`` result equivalent to terminal analysis of the complete
  per-cycle log.
* Avoid enabling expensive progressive analysis by default for cluster-wide
  service-mode submissions.
* Keep progressive analysis an optimization: if pre-analysis is unavailable,
  failed, stale, or unsupported, ``GET`` must fall back to the existing terminal
  analysis behavior.

Non-Goals
---------

* Do not require progressive analysis for service-mode cluster scans.
* Do not introduce a user-facing live-attribution workflow outside
  ``ft_launcher``.
* Do not change the stop/restart policy consumed by ``ft_launcher``.
* Do not make ``POST /logs`` block on analysis completion.
* Do not make ``POST /logs`` generate or return an attribution result.

User Flows
----------

``ft_launcher`` mode
~~~~~~~~~~~~~~~~~~~~

1. At cycle start, the launcher computes the per-cycle log path and sends
   ``POST /logs`` for that path.
2. Attrsvc recognizes the submission as an ``ft_launcher`` progressive-analysis
   request and starts non-blocking pre-analysis for the growing file.
3. The workload runs and continues writing to the same log file.
4. At cycle end, ``ft_launcher`` sends ``GET /logs`` for the same path.
5. Attrsvc uses the progressive state to complete the normal stop/end analysis
   and returns the normal attribution payload with a normalized recommendation.
6. If progressive state is missing or unusable, attrsvc computes the result with
   the existing terminal pipeline.

Service mode
~~~~~~~~~~~~

Service mode may continue using ``POST /logs`` for job/file tracking and
``GET /logs`` for post-mortem attribution. Progressive analysis should remain
disabled unless explicitly requested by a test or diagnostic client.

Functional Requirements
-----------------------

* ``POST /logs`` must accept an explicit signal that the caller wants
  progressive analysis for a single growing log.
* The ft_launcher client must send that signal when submitting a per-cycle log.
* Attrsvc must forward progressive intent to the loganalysis tool boundary and
  return from ``POST`` without waiting for analyzer completion.
* The loganalysis tool may delegate progressive state to LogSage or implement
  the file-flow orchestration itself. Attrsvc should not need to know which
  model was chosen.
* Attrsvc must not infer progressive analysis only from the absence of
  ``job_id`` because non-``ft_launcher`` callers can also submit single files.
* ``GET /logs`` must continue to run the normal terminal analysis path before
  returning a decision. Once LogSage/tool support exists, that terminal call can
  reuse progressive work when available.
* ``GET /logs`` must fall back to the existing full terminal analysis path when
  progressive analysis is unsupported, incomplete, stale, failed, or disabled.
* Progressive state, if any, must be correlated by normalized log path because
  this is the stable key shared by ``POST`` and ``GET``.
* ``POST /logs`` must remain a notification/early-start path. ``GET /logs``
  remains the result-producing path.
* The existing result cache behavior does not change: ``POST`` does not populate
  the final analysis cache; ``GET`` remains the path that computes and records
  final attribution results.

API and Data Contract
---------------------

Handoff Points
~~~~~~~~~~~~~~

The feature crosses three handoff points. Each handoff should be documented
separately so NVRx-owned plumbing is not confused with the shared LogSage
capability contract.

Service HTTP boundary
   Owned by NVRx. ``POST /logs`` accepts the optional progressive intent and
   ``GET /logs`` remains the stop/end decision API. This boundary should stay
   backward compatible for existing attrsvc clients. The service does not own
   progressive parsing, tailing, or final-result caching.

MCP / loganalysis boundary
   Owned by NVRx. This is the product feature boundary: it exposes the
   progressive option regardless of whether the implementation is an MCP tool,
   an in-process adapter, repeated LogSage calls, or a future LogSage-native
   progressive session. This is where NVRx should hide implementation ownership
   from attrsvc.

LogSage API boundary
   Shared with LogSage. This contract defines whether LogSage can start work
   early, preserve progressive state, and let the terminal analysis call reuse
   that state while producing a result equivalent to full terminal analysis. It
   remains the main open design item.

HTTP API
~~~~~~~~

Extend ``POST /logs`` with optional fields:

``analysis_intent``
   Optional analysis behavior requested by the client. Proposed values are
   ``"track_only"`` and ``"progressive"``. Default is ``"track_only"`` for
   backward compatibility.

The existing ``log_path``, ``user``, and ``job_id`` fields stay compatible.
Existing clients that omit ``analysis_intent`` retain current behavior.

``GET /logs`` should keep the current response shape. It may optionally include
diagnostic metadata in the future, but ``ft_launcher`` must continue to consume
the normalized ``recommendation`` field without understanding progressive
internals.

Python API
~~~~~~~~~~

Extend the internal submit boundary from the HTTP adapter down to the analyzer
with an optional progressive intent. The NVRx-side lifecycle should remain
simple: ``POST`` starts early work when requested, and the existing
``GET``/``analyze`` path remains the stop/end activity that returns the final
decision.

* ``submit_log(..., analysis_intent="track_only")``
* analyzer-level delegation to
  ``LogAnalyzer.start_progressive_analysis(path, user, job_id)``
* loganalysis runner delegation to the selected lib/MCP tool adapter

The current plumbing does not change ``Analyzer.analyze`` or the ``GET`` result
path. When the LogSage/tool contract supports reuse, terminal analysis can add a
``use_progressive``-style option at the loganalysis layer while preserving the
existing caller-facing ``GET`` shape.

MCP / Loganalysis Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~

The MCP/loganalysis boundary is NVRx-owned and should mirror the in-process
library adapter. The boundary needs two concepts:

Progressive start
   A non-result-producing operation used by the ``POST /logs`` path when
   ``analysis_intent="progressive"``. The initial code exposes this as
   ``log_analyzer_progressive_start``. It accepts the normalized ``log_path``,
   ``is_per_cycle=True`` for ft_launcher cycle logs, optional observability
   fields, and any runtime settings needed by LogSage to bind a future
   progressive session. It returns status metadata such as
   accepted/unsupported/failed and, if useful, a handle or session id. It must
   not return a final attribution result or create a cached MCP result resource.

Terminal run with progressive reuse
   The existing ``GET /logs`` path should still invoke terminal log analysis and
   receive the current LogSage-shaped result. Once the LogSage API is settled,
   the loganalysis boundary needs a way to ask the backend to reuse progressive
   state for the same path, for example an optional ``use_progressive=True``
   argument on ``log_analyzer``. If the backend cannot reuse state, the call
   should fall back to normal terminal analysis. The current plumbing leaves
   terminal ``GET`` unchanged.

Flight-recorder analysis is unchanged by this feature. ``POST /logs`` does not
notify FR. On ``GET /logs``, attrsvc can continue to run log analysis and FR
analysis with the existing terminal orchestration; only the log-analysis call
needs a way to reuse progressive LogSage state when it exists.

Configuration
-------------

Add a service-side policy switch for progressive analysis. The default honors
explicit progressive requests because ft_launcher is the expected caller and the
POST path remains non-blocking even while the LogSage progressive API is being
implemented.

``NVRX_ATTRSVC_PROGRESSIVE_ANALYSIS``
   ``all_explicit`` by default. Honors explicit progressive ``POST /logs``
   requests. Set ``off`` to disable progressive start. A stricter
   ft_launcher-only policy would require a caller identity or another
   server-side way to identify the submitter.

No attrsvc polling, cache, or concurrency settings are required in the current
plumbing. If the loganalysis tool later owns repeated LogSage calls over a
growing file, those operational controls should live with that tool
implementation rather than in the HTTP service wrapper.

Observability
-------------

Expose enough state to tell whether the latency optimization is working:

* Count progressive ``POST`` requests and whether they were accepted, ignored,
  or rejected by policy.
* Count started, unsupported, failed, canceled, completed, and fallback
  progressive analyses.
* Track final ``GET`` latency and, where possible, time saved by progressive
  work.
* Include active progressive paths in status/debug output without dumping log
  contents.
* Log when ``GET`` falls back to terminal analysis, including the fallback
  reason.

Compatibility
-------------

Existing attrsvc callers must continue to work without sending new fields.
Service-mode ``POST`` calls with ``job_id`` continue to support splitlog
detection and tracking. Single-file ``POST`` calls from older clients remain
track-only.

The final ``GET`` result must remain semantically compatible with the current
terminal analysis result. A progressive implementation may improve latency, but
must not weaken final attribution correctness.

NVRx-Side Implementation Plan
-----------------------------

1. Add an explicit progressive intent field to the shared HTTP helper and
   attrsvc ``SubmitRequest``.
2. Update the attribution-owned in-job HTTP client to send
   ``analysis_intent="progressive"`` on its normal ``POST /logs`` notification.
   ``ft_launcher`` should continue using the existing attribution submit hook
   and should not own the HTTP payload detail.
3. Thread the intent through ``AttributionHttpAdapter``,
   ``AttributionController``, and ``Analyzer.submit``.
4. In attrsvc, keep default ``POST`` behavior as track-only. When progressive
   intent is requested and the service feature gate allows it, initiate
   progressive analysis through ``LogAnalyzer.start_progressive_analysis``.
5. In MCP mode, expose ``log_analyzer_progressive_start`` as a
   non-result-producing tool. Until LogSage support exists, it returns
   ``unsupported`` status metadata.
6. On success, return the existing response shape. ``GET /logs`` remains the
   path that produces and records final attribution results.
7. Once the LogSage contract exists, extend the terminal loganalysis call so it
   can request progressive reuse while keeping the existing ``GET`` response
   shape and fallback behavior.

Progressive Execution Model
---------------------------

Attrsvc owns request plumbing and policy. The NVRx loganalysis tool owns the
progressive option exposed to attrsvc. The implementation behind that tool is
still a design choice and should be settled with the LogSage implementation.

LogSage-owned progression
   The loganalysis tool calls a LogSage ``start`` operation and returns.
   LogSage owns any background watching, tailing, checkpointing, or incremental
   state. The normal ``GET`` path calls terminal analysis with progressive reuse
   enabled. Attrsvc only sees accepted/unsupported/failure status.

Tool-owned progression
   The NVRx loganalysis tool advances analysis as the file grows, possibly by
   calling LogSage multiple times or by using a smaller LogSage ``advance``
   primitive. Polling/tailing configuration, concurrency limits, lifecycle
   cleanup, and state tracking belong to that tool implementation.

The requirement does not choose between these models. It requires that
``POST`` can request an early start, ``GET`` remains the stop/end decision path,
and terminal correctness is preserved by fallback to the existing full analysis.

LogSage API Proposal
--------------------

The current LogSage-style API is effectively terminal:

* input: complete ``log_path``
* output: final LogSage-shaped result and recommendation

For this feature, the NVRx loganalysis tool needs a way for LogSage to preserve
and reuse work for a growing per-cycle log. The shared behavior can be expressed
with two operations:

``start(path, *, session_id=None, is_per_cycle=True, metadata=None)``
   Create or return progressive state for a log. This should be idempotent for
   the same normalized path or supplied session id.

``run(path, *, use_progressive=True, final=True)``
   Return the normal terminal LogSage result for the complete log, reusing any
   progressive state that was started for the same path. This is invoked from
   the existing ``GET /logs`` stop/end path.

LogSage may internally expose an ``advance`` or completion primitive if that is
the cleanest implementation, or the NVRx loganalysis tool may drive repeated
LogSage calls. Attrsvc does not need a separate public
``finalize_progressive`` operation. The important part is that the terminal
``run`` can consume any remaining tail bytes, validate the final log state, and
produce the same result shape used today.

``cancel(handle)``
   Release resources if the job ends without a final attribution request or the
   service is shutting down.

Minimum status metadata returned through the NVRx boundary:

``handle`` or ``session_id``
   Stable identifier for the progressive analysis.

``consumed_offset``
   Last byte or line offset included in progressive state.

``status``
   ``pending``, ``running``, ``ready_to_complete``, ``completed``, ``failed``,
   or ``unsupported``.

``error``
   Structured failure reason when status is ``failed`` or ``unsupported``.

The important semantic requirement is that the stop/end ``run`` must produce a
result equivalent to running terminal LogSage on the complete final file.
Attrsvc should not need to understand LogSage's internal summaries, LLM prompt
state, or whether reuse was LogSage-owned or tool-owned.

Validation
----------

* Unit-test that existing ``POST /logs`` requests remain track-only by default.
* Unit-test that ft_launcher ``POST`` sends progressive intent.
* Unit-test that ``Analyzer.submit`` delegates progressive start to the
  loganalysis boundary only when the feature gate is enabled.
* Unit-test that the MCP/loganalysis progressive-start operation returns status
  metadata and does not run terminal attribution.
* Once LogSage/tool reuse exists, unit-test that the terminal MCP/loganalysis
  call can request progressive reuse while preserving the existing result shape.
* Unit-test that service-mode ``POST`` with ``job_id`` does not initiate
  progressive analysis by default.
* Once reuse exists, unit-test ``GET`` fallback when progressive state is
  missing, unsupported, failed, stale, or cannot be used to complete analysis.
* Unit-test that ``POST`` does not return a completed attribution result.
* Integration-test the full ``ft_launcher`` flow with a fake progressive
  analyzer: POST starts state, log grows, GET returns a normal recommendation
  through the existing stop/end path.
* End-to-end validate latency improvement with real LogSage progressive support
  once available.

Open Questions
--------------

* What exact LogSage API should back ``log_analyzer_progressive_start`` and
  terminal reuse?
* Should progressive advancement be LogSage-owned after ``start``, or should the
  NVRx loganalysis tool drive repeated LogSage calls over the growing file?
* If the NVRx loganalysis tool owns advancement, what polling and concurrency
  policy should it use?
