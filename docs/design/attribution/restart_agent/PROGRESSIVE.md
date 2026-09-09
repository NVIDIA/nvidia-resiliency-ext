# Progressive Analysis

This document is canonical for the Restart Agent progressive lifecycle.
Progressive execution changes when L0A work runs. It does not introduce a
second evidence schema, semantic analysis, history comparison, or retry policy.
Terminal execution is the production default. Progressive pre-end polling is
disabled unless attrsvc explicitly enables it after measurement shows that
terminal post-end L0A latency warrants the additional state and polling.

Terminal and progressive execution use one byte-range reader, one incremental
line decoder, one append-only L0 observation index, and one canonical L0A
finalizer. Direct terminal execution supplies the captured boundary
immediately; attrsvc terminal execution may ingest additional chunks during its
bounded rank-drain wait. Progressive execution supplies available chunks, waits
when no new bytes exist, resumes from its saved byte offset, and finalizes after
cycle end.

## Lifecycle

The intended NVRx service sequence is:

```text
cycle start
  POST /logs analysis_intent=progressive
    -> validate/register the cycle-unique log path
    -> terminal default: retain registration only
    -> progressive opt-in: start bounded, non-authoritative L0A precomputation

cycle end or failure
  POST /logs analysis_intent=terminal
    -> combine retained state with the unread final log tail
    -> finalize canonical L0A evidence
    -> publish the deterministic L0A/L3/L4 recommendation
    -> by default, schedule L0B/L1/L2/L3/L4 enrichment

result probes
  GET /logs?wait=false
    -> return completed, in_flight, or pending immediately
```

`GET /logs` with omitted `wait` or `wait=true` may join analysis and block until
a result or service timeout. `track_only` is service-local registration; it does
not invoke verdict-producing Restart Agent work.

The normalized, cycle-unique `log_path` is the service correlation key. The
service records `job_id` and integer `cycle_id` when available, but terminal
signals need not repeat them.

## State Machine

```text
registered -> precomputing <-> idle
     |              |           |
     +--------------+-----------+-> finalizing -> completed
                                                 +-> failed
```

`finalizing` is monotonic. A terminal request atomically enters it, wakes the
poller, waits for any in-flight read for that attempt, and starts final drain.
Duplicate progressive and terminal requests return the existing state or future.
Each same-attempt replacement has a generation; stale poll or route callbacks
cannot mutate a newer generation.
`failed` means service execution ended without a usable result; a normal
log-unavailable result is `completed`.

## Start And End Authority

Progressive start may inspect newly appended bytes, update deterministic
occurrence groups, collect progress/checkpoint facts, build candidate summaries,
and retain bounded context windows. It must not emit `STOP`, `RESTART`,
`decision_basis`, or another final policy result. The first implementation of
progressive evidence work runs only L0A before cycle end. It does not run L0B,
L1, L2, L3, or L4 speculatively against a provisional failure.

Progressive end is authoritative. It must:

- merge retained state with the unread final log tail;
- allow for bounded post-end log drain and rank interleaving;
- finalize canonical `L0Bundle`, `DecisionEvidence`, progress facts, and
  deterministic failure facts over the combined evidence;
- select one immutable `PriorAttemptView` after L0A finalization and use it for
  the deterministic and enriched branches;
- publish the deterministic recommendation before starting L1 routes;
- return the same external result schema as terminal execution; and
- trace missing, stale, evicted, or unusable progressive state before falling
  back to terminal analysis.

Canonical equivalence is structural equality after removing fields explicitly
designated as timing or operational metadata. Ingestion chunk boundaries,
progressive read count, idle/resume events, and pre-end timestamps must not
change canonical object identity, ordering, selected evidence, progress facts,
or deterministic failure facts.

Excluded comparison fields are limited to stage/request timestamps and
durations, read/poll/idle counters, source-reset reread byte counts, and
progressive cache hit/eviction/fallback diagnostics. Final source identity and
size, line numbers, object IDs, evidence content and ordering, selection and
lossiness metadata, and all policy-relevant facts remain part of the canonical
comparison.

## Bounded Reconciliation

Progressive state is one L0A accumulator, not a sequence of independently
combined bundles. It retains source identity and boundary, the next unread byte
offset, the decoder's incomplete byte tail, a compact logical-line byte index,
one-time-classified observation channels, the latest canonical L0A checkpoint,
and operational poll/build metadata.

The MVP checks file metadata first. An unchanged source is not reread. Growth
reads only `[saved_offset, captured_boundary)`. Completed lines are appended
exactly once and classified into reusable observation channels; an incomplete
line remains only in the decoder until a later read completes it. Canonical L0A
reducers build checkpoints from those channels rather than rescanning or
redecoding the complete log boundary.

If the file did not exist at progressive start, the accumulator adopts its
source identity when the file first appears. An idle accumulator may resume when
growth is observed or when progressive end requests finalization; idle is a
resource-management state, not an attempt outcome.

At progressive end, a stat-only drain observer runs concurrently with the
initial unread-source ingestion. Each observed growth event resets the quiet
interval and wakes a separate reader that ingests the newly available byte
range without blocking metadata observation. When the source first becomes
quiet, L0A may speculatively reduce the current captured boundary while the
drain observer continues. Live NVRx/attrsvc finalization cannot establish
convergence before the internal minimum observation period has elapsed; its
quiet interval begins no earlier than that boundary. Later growth makes a
checkpoint stale, resets quiet observation, and the next quiet interval may
replace it. Once convergence is established, the observer
sends an explicit completion notification; finalization does not wait for
another polling interval. It performs one exact catch-up read and reuses the
checkpoint only when its source boundary is the final boundary, otherwise it
performs the canonical reduction then. This overlaps unavoidable drain time
with L0A work without freezing an early boundary. If maximum wait expires
before convergence, the active incomplete tail is excluded and counted.
Replacement, truncation, or same-size modification resets the byte cursor,
decoder, observations, and checkpoint, then replays the captured boundary from
byte zero.

A precomputed checkpoint records the exact source boundary from which it was
built. Finalization reuses it only when that boundary still matches; unseen
terminal growth forces a new reduction after catch-up. Object IDs derive from
canonical type and source position, not poll or chunk sequence.

Decoding uses UTF-8. Malformed byte sequences are replaced only in the affected
physical line and are reported by replacement-character and affected-line
counters; they do not trigger a second whole-file decoding pass. LF is the only
physical-line boundary. CRLF strips its CR, while bare CR remains content.
Split CRLF, bare CR, multibyte characters, and unterminated final lines are
invariant to chunk boundaries.

The supported source modes are:

- `chunked`: production default; read fixed-size byte ranges and resume from
  the saved offset;
- `single_snapshot`: parity/regression mode; supply the captured boundary as one
  chunk through the same decoder, observation index, reducers, and finalizer.

For the same captured bytes, mode and chunk size are operational metadata and
must not change canonical evidence.

## Ownership And Handoff

The service owns scheduling and lifecycle; the product owns evidence semantics:

```text
ProgressiveCoordinator
  -> ChunkedLogReader
  -> ProgressiveL0Accumulator
  -> FinalizedL0A
  -> RestartAgentRuntime
```

`ProgressiveL0Accumulator` exposes typed `refresh`, `state`, and `finalize`
operations that are independently testable without attrsvc. `FinalizedL0A`
contains canonical `L0Bundle`, `DecisionEvidence`, an indexed file-backed
`LogSnapshot`, and a fixed source identity/byte/line boundary. The runtime
derives route-independent deterministic facts from that evidence, accepts this
object directly, builds L0B, and does not rebuild L0A. Model tools inspect only
the finalized boundary through that source view.

The product also exposes one deterministic test projection that removes only
the operational fields listed above. Terminal and progressive fixtures compare
that projection byte-for-byte rather than maintaining separate equivalence
logic in the service or harness.

## Pre-End Polling

The first progressive implementation uses service-owned periodic polling rather
than filesystem notifications. `pre_end_poll_seconds` is configurable and
defaults to `180` seconds.

- Progressive start schedules an immediate metadata check.
- Thereafter, the service checks file existence, identity, size, and mtime once
  per interval.
- An absent or unchanged file is not scanned or reprocessed.
- A new file starts at byte zero. A grown file reads only the new byte range and
  refreshes the canonical L0A checkpoint from the updated observation index.
- Replacement, truncation, or same-size modification resets and replays the
  captured source.
- Progressive end interrupts the polling wait and immediately starts final
  log-drain convergence and L0A finalization.

The polling interval bounds how stale precomputed L0A state may be during a
running attempt. It does not affect correctness because progressive end always
reads the final tail before producing authoritative evidence.

## Retained State

Progressive state is analyzer state, not an independent bundle cache. The
current implementation retains:

- source identity, byte boundary, read offset, incomplete byte tail, and compact
  logical-line byte offsets;
- one-time-classified observation channels and the latest canonical checkpoint;
- normalized occurrence groups and deterministic candidate summaries;
- progress, checkpoint, failure-episode, and incident facts;
- bounded raw windows around recent tail, progress, checkpoint, first-fault, and
  top-candidate anchors;
- selection, truncation, eviction, and lossiness metadata; and
- request identity and anomalies required for finalization.

Decoded completed lines are classified once and released. `LogSnapshot`
preserves original line numbers and quotes through a boundary-limited source
handle and compact offset index. The handle remains available only through the
last active L1/L2 route or the Restart Agent analysis timeout.

MVP service retention may be local in memory and is bounded by:

- `active_idle_seconds`: release active parsing resources after no observed
  growth; idle records remain subject to the low-cost pre-end metadata poll, and
  becoming idle does not emit a final action;
- `max_active_states`: cap active progressive records; and
- `max_completed_results`: cap retained completed decisions.

The defaults are `active_idle_seconds=900`, `max_active_states=64`, and
`max_completed_results=3000`. The active bound counts retained non-finalized
accumulators, not registrations. At the bound, the service evicts the
least-recently-used idle accumulator, then the oldest non-finalizing
accumulator. If none is safe to evict, it registers the attempt without
precomputation. Registration remains valid, and an evicted attempt uses terminal
L0A at end. A finalizing state is never evicted. Completed-result eviction
removes the oldest service entry; a later GET returns the existing not-found
response.

## Read And State Failures

- A transient pre-end stat/read failure preserves prior state, records the
  error, and retries on the next poll.
- Missing files before end are normal. Missing, empty, or unreadable files at
  finalization use the existing terminal log-unavailable result.
- Replacement, truncation, same-size modification, encoding reset, corrupt
  state, or eviction triggers replay from byte zero when possible.
- If final drain reaches its maximum wait, the last observed size becomes the
  immutable source boundary, an incomplete active tail is excluded, and the
  trace records the bound and discarded-tail byte count.
- Shutdown cancels poll work without inventing a decision. A later terminal
  request may run a complete terminal analysis.

The conservative drain belongs only to live sources ended through attrsvc.
Direct CLI/library and evaluation calls over already-created files capture EOF
as a known-complete boundary and finalize immediately.

## Latency And Candidate Publication

The Restart Agent does not own or enforce the NVRx action window. NVRx may
consume the best candidate available under its own policy and move on. That
external action does not close the analysis.

The Restart Agent publishes the deterministic recommendation as soon as authoritative
L0A, L3, and L4 complete. L1 routes continue under the configured Restart Agent
analysis timeout, 240 seconds by default. A later usable L1/L2/L3/L4 result
updates the current
`AttemptRecord`, internal history, and service-visible completed analysis. It
does not retroactively change an action NVRx already took.

The deterministic recommendation and every configured model route use the same L0A
state, Decision Evidence, immutable `PriorAttemptView`, schemas, and policy.
Output arriving after the Restart Agent's own analysis timeout or a superseding
same-attempt generation is rejected and traced. Future priority selection and
enriched prior-record selection are outside the implemented `collect_all` mode.

Qualification measures separately:

- work completed before progressive end;
- progressive-end-to-deterministic-recommendation latency;
- progressive-end-to-enriched-result p50, p90, and p99 latency;
- analyzer timeout and route completion rates.

Terminal timing uses the terminal signal as a common monotonic origin and
reports drain completion, canonical L0A readiness, deterministic recommendation
readiness, first route readiness, and full analysis completion separately.
L0A additionally reports source decode, source index/classification, and
canonical reduction time. These subphase durations explain cost; they are not
summed with overlapping drain time.

## Service Logging

Progressive registration emits INFO once:

```text
event=restart_agent.progressive.registered
status=<scheduled|disabled|precompute_skipped_capacity>
pre_end_poll_seconds=<seconds> active_idle_seconds=<seconds>
```

Periodic pre-end polling is DEBUG to avoid one INFO record per poll:

```text
event=restart_agent.progressive.refresh.completed changed=<bool>
wall_clock_s=<seconds> phase=<phase> bytes_ingested=<count>
source_ingest_s=<seconds> l0a_reduction_s=<seconds>
l0a_build_count=<count> poll_count=<count> growth_count=<count>
```

Terminal drain emits INFO once:

```text
event=restart_agent.terminal.drain_completed converged=<bool>
completion_reason=<reason> wall_clock_s=<seconds> poll_count=<count>
growth_count=<count> incomplete_tail_included=<bool>
```

Capacity skips, max-wait expiry, and precompute failure use WARNING with the same identity and
timing fields. DEBUG may add source decode/classification, reread, reset,
replacement-character, and pending-tail counts; it must not log source text.

## Validation

Progressive qualification must verify:

- terminal and progressive executions produce structurally equal canonical
  `L0Bundle`, `DecisionEvidence`, progress facts, and deterministic failure
  facts for the same finalized source bytes after designated timing and
  operational fields are removed;
- terminal and progressive scans using different byte chunk sizes preserve
  physical lines and produce identical canonical evidence;
- `single_snapshot` and `chunked` modes produce identical canonical evidence;
- incomplete lines, split CRLF, bare CR content, split multibyte UTF-8, and an
  unterminated final line are neither duplicated nor omitted;
- progressive start performs an immediate metadata check, scheduled polls do not
  reread unchanged files, and progressive end interrupts the polling wait;
- duplicate requests and poll/end races produce one finalization;
- UTF-8 boundaries, malformed-byte replacement, truncation, and state-eviction
  cases preserve safe terminal fallback;
- post-end drain does not omit the initiating failure;
- replacement, truncation, state loss, or unusable state degrades to terminal
  L0A without changing policy semantics;
- terminal-versus-progressive divergence is reported;
- canonical comparison fixtures exercise multiple chunk schedules for the same
  source bytes;
- post-`progressive_end` deterministic and, when enabled, enriched-result latency
  measurements use
  the production lifecycle rather than terminal request-to-result timing;
- NVRx moving on does not prevent a later usable L1 result from updating
  internal history; and
- output rejected after the Restart Agent analysis timeout or superseding
  generation cannot mutate the current record.
