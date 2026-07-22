# Restart Agent Runtime And Attempt Record Spec

This document is canonical for runtime composition, current-lifetime attempt
record ownership, prior-attempt injection, and CLI/library execution.
`SCHEMA.md` owns the data shapes and `POLICY.md` owns how L3 interprets a
`PriorAttemptView`.

## Runtime Boundary

The restart-agent runtime is stateful orchestration around the stateless
analysis pipeline. Configuration parsing and concrete dependency construction
remain outside the runtime.

```text
JSON file / CLI arguments / environment
  -> configuration loader
  -> immutable RestartAgentConfig
  -> composition root
       -> credential and provider-client resolution
       -> model-route construction
       -> attempt-record-store construction
  -> RestartAgentRuntime
       -> request orchestration
       -> current-lifetime attempt records
       -> model-route concurrency and deadlines
  -> CLI, library, or future thin MCP adapter
```

The same runtime implementation MUST be used by CLI, library, and future MCP
entrypoints. A transport adapter MUST NOT implement a second attempt-record,
history-comparison, or policy path.

## Component Responsibilities

| Component | Responsibility |
| --- | --- |
| Configuration loader | Read JSON, validate closed schemas, apply defaults, and return immutable typed configuration. |
| Credential resolver | Resolve external credential references without placing secrets in effective configuration or traces. |
| Composition root | Interpret typed configuration and construct routes, clients, attempt-record storage, the record assembler, and the runtime. |
| `RestartAgentRuntime` | Coordinate requests, deadlines, route execution, candidate publication, and current-lifetime attempt records. |
| `AttemptRecordStore` | Store current and prior `AttemptRecord` objects without interpreting evidence or policy. |
| `AttemptRecordControl` | Seed, inspect, and clear in-memory attempt records; the CLI may serialize inspected records as an explicit fixture. |
| `AttemptRecordAssembler` | Create the initial immutable L0 record and produce immutable same-key replacements as enriched L2 facts become available. |
| CLI adapter | Convert arguments and files into typed configuration and analysis requests; optionally import/export manual history fixtures. |
| Future MCP adapter | Expose the required product operations as a thin view over the same runtime. |

Ownership and construction are distinct: the runtime owns the live record
lifecycle, but the composition root constructs and injects the concrete store.
This preserves a clean test seam and permits another store implementation
without changing runtime behavior.

## Configuration Flow

The runtime MUST NOT receive a configuration path, raw JSON dictionary,
credential environment-variable name, or API-key file path. The supported
flow is:

```python
config = load_restart_agent_config("restart_agent.json")
runtime = build_restart_agent_runtime(config)
result = runtime.analyze(request)
```

The loader performs parsing, validation, defaulting, and normalization. The
composition root resolves credentials and constructs dependencies. The runtime
receives ready-to-use dependencies plus non-secret effective-configuration
metadata for tracing.

Actual attempt records are runtime state, not configuration. Product
configuration controls only record retention and history-comparison behavior.
The supported configuration is:

```json
{
  "runtime": {
    "history": {
      "enabled": true,
      "max_attempts_per_job": 10,
      "max_total_records": 3000
    }
  }
}
```

History is enabled by default. Omitting `runtime.history` is equivalent to the
object above. `max_attempts_per_job` and `max_total_records` MUST be positive
integers and default to `10` and `3000`, respectively. Operators may override
either bound. When history is disabled, the composition root injects a null
record store and explicit seeding fails with a clear `history_disabled` error
rather than silently discarding records.

The checked-in `restart_agent.json` includes these defaults explicitly so the
runtime-owned state policy is visible to operators.

## Attempt Record Store Contract

The runtime depends on a store interface equivalent to:

```python
class AttemptRecordStore(Protocol):
    def get_prior_attempts(
        self,
        job_id: str,
        before_cycle_id: int,
    ) -> PriorAttemptView: ...

    def upsert_attempt(self, record: AttemptRecord) -> None: ...

    def records(
        self,
        job_id: str | None = None,
    ) -> tuple[AttemptRecord, ...]: ...

    def replace(self, records: Sequence[AttemptRecord]) -> None: ...

    def clear(self, job_id: str | None = None) -> None: ...
```

The MVP store is `InMemoryAttemptRecordStore`. It has these semantics:

- exact `job_id` matching;
- integer `cycle_id` ordering;
- current and future cycles excluded from `PriorAttemptView` results;
- immutable in-memory views supplied to an invocation;
- idempotent upsert by `(job_id, cycle_id)`;
- replacement rather than recurrence inflation when a cycle is reanalyzed;
- retention of one shared L0 progress summary, one required deterministic
  failure-facts block, and a compact enriched-facts list;
- at most one enriched entry per `route_id`; a repeated route update replaces
  that entry rather than appending a duplicate;
- deterministic facts remain the only history-comparison source in MVP;
- independent state for different jobs;
- thread-safe view and update operations;
- oldest-cycle eviction after `max_attempts_per_job` records for one job; and
- oldest-record eviction by internal insertion sequence when
  `max_total_records` is exceeded across all jobs.

The per-job bound is a cycle-ordered bounded queue. Replacing an existing
`(job_id, cycle_id)` does not change record count or insertion sequence. New
records are inserted under one store lock; per-job eviction runs first, followed
by total-record eviction. The internal insertion sequence is store metadata and
is not serialized in an `AttemptRecord` fixture.

`records()` returns deterministic `(job_id, cycle_id)` order. Because fixtures
do not serialize live insertion sequence, `seed()` first validates unique keys
and normalizes records to that same order. `replace` assigns fresh insertion
sequence in normalized order; `merge` applies normalized records through
idempotent upsert in that order. This makes fixture replay deterministic while
keeping live total-cap eviction based on actual insertion sequence.

The MVP store survives only for the lifetime of one runtime process. A runtime
restart starts with no records. Prior-view availability, record count, enriched
entry count, and eviction MUST be traced. Persistent, distributed, or
automatically serialized records are outside MVP scope.

## Attempt Record Control And Injection

The runtime exposes a transport-independent control surface:

```python
class AttemptRecordControl:
    def seed(
        self,
        records: Sequence[AttemptRecord],
        *,
        mode: Literal["replace", "merge"] = "replace",
    ) -> None: ...

    def records(
        self,
        job_id: str | None = None,
    ) -> tuple[AttemptRecord, ...]: ...

    def clear(self, job_id: str | None = None) -> None: ...
```

This is primarily a library/unit-test seam for state-based testing. The CLI may
adapt `--attempt-records-json-in` through `seed()` and write `records()` through
`--attempt-records-json-out`. These explicit fixture
operations are not a public analysis request, automatic persistence,
production state-transfer format, or MCP operation. A later hydration design
may reuse the typed store/control boundary, but that is not an MVP contract.

Attempt records are seeded as established runtime state before analysis. They
are not arbitrary fields on each public analysis request. This preserves one
state lifecycle while allowing exact in-memory state-based tests.

`replace` is the deterministic default for tests and delegates to the store's
atomic `replace()` operation. `merge` performs the same idempotent
`(job_id, cycle_id)` upsert used during live operation. Both modes validate
record shape, unique keys, and ordering. The store then applies the same
per-job and total eviction rules used during live operation.

## Prior-View And Record Eligibility

`job_id` remains a string and `cycle_id` remains an integer. Literal `0` is
valid only when it is the caller's real identifier; the runtime MUST NOT invent
`"unknown"`, `0`, a log-path hash, or another synthetic identity for missing
metadata. Synthetic identities could join unrelated jobs or replace unrelated
cycles.

Selecting a `PriorAttemptView` requires all of:

- history enabled;
- non-empty `job_id` supplied by the caller;
- integer `cycle_id` supplied by the caller.

The runtime can select prior records before L0 knows the current fingerprint.
L3 additionally requires a non-empty fingerprint in the deterministic or
selected enriched facts before it can compare roots. An `AttemptRecord` upsert
requires a non-empty deterministic L0 fingerprint because the deterministic
block is mandatory in MVP.

If store, request, or fingerprint eligibility is absent, analysis continues
without recurrence evidence. Trace state reports one of `history_disabled`,
`missing_job_id`, `missing_cycle_id`, or `missing_root_fingerprint`. Missing
identity is not an error in the public analysis request. Missing job/cycle
prevents prior selection and record upsert; missing deterministic fingerprint
prevents record upsert and makes L3 unavailable for that branch.

An eligible lookup that finds no prior records is available-but-empty, not
unavailable. L3 receives an empty immutable view with
`availability_reason=ready`; this distinguishes a first attempt from disabled
history or missing identity.

## Attempt Progress Construction

L0 constructs one immutable `AttemptProgressSummary` from deterministic log
facts. `AttemptRecordAssembler` places that summary at the top level of the
record so deterministic and enriched routes share it. Rank-duplicated marker
lines are deduplicated into logical observations before values or counts are
computed.

- Training progress uses completed application iteration/step markers. Setup,
  started-but-not-completed operations, traceback copies, and raw rank fanout do
  not count.
- `training_progress=observed` requires at least one completed marker.
  `not_observed` requires a fully scanned readable log and a recognized
  progress-marker dialect. Otherwise it is `unknown`.
- `first_completed_step` and `last_completed_step` are the minimum and maximum
  comparable completed values. `completed_step_delta` is `last - first`.
  `progress_marker_count` is the number of distinct logical completed-marker
  observations, not raw log lines.
- Checkpoint progress uses successfully completed checkpoint saves made by the
  current attempt. Checkpoint-save starts and failed saves do not count.
- `checkpoint_load_step` is the most recent successfully loaded resume point
  before the primary failure. It is never counted as a save by this attempt.
- `first_checkpoint_step`, `last_checkpoint_step`, `checkpoint_step_delta`, and
  `checkpoint_marker_count` are computed from deduplicated completed saves.
- `failure_position` is relative to the canonical L0 failure episode and is
  `before_observed_training_progress`, `after_observed_training_progress`, or
  `unknown`.
- `progress_after_failure` is `observed` when a completed compatible progress
  marker occurs after the canonical failure episode ends. In an interleaved log
  it proves job-level continuation, not recovery of the same rank or component.
  It is `not_observed` only with adequate readable tail coverage and a
  recognized progress dialect; otherwise it is `unknown`.

Checkpoint delta and marker count are intentionally distinct. For successful
saves at steps `100`, `200`, and `400`, `checkpoint_step_delta=300` while
`checkpoint_marker_count=3`. The former measures observed distance and the
latter measures observation frequency; only comparable step values participate
in L3 advancement.

## Invocation Lifecycle

For one request, the runtime performs:

```text
validate request
  -> take immutable PriorAttemptView for exact job and earlier cycles
  -> construct invocation context
  -> L0 builds progress and deterministic failure facts
  -> assemble current AttemptRecord(enriched=[]); upsert when eligible
  -> run deterministic L3/L4
  -> publish deterministic fallback candidate
  -> start configured model routes
  -> each completed L2 route atomically adds/replaces enriched[route_id]
  -> run that enriched route through L3/L4 using the same PriorAttemptView
  -> close record updates at completion or deadline
  -> return invocation-owned results and artifacts
```

The record has one contract throughout its lifetime. Immutable replacements
represent the current `AttemptRecord` during analysis; its final value appears
unchanged in a later cycle's `PriorAttemptView`. There is no conversion to a
differently named history record. L0 supplies shared progress and deterministic
facts, and L2 supplies each enriched fact block.
`AttemptRecordAssembler` creates the immutable replacements and the runtime
commits them. L3 and L4 read the selected block and prior view but do not mutate
the record.

The deterministic block is the sole policy-active history-comparison source
for MVP `collect_all`. Enriched entries are retained for observability and
future design but ignored by later L3 comparisons. Model-route completion order
therefore cannot affect MVP policy. Future route selection may choose enriched
facts only after its authority and migration semantics are specified.

Attempt progress is not route-selected semantic output. The runtime constructs
`AttemptProgressSummary` from deterministic current-log L0 facts and stores it
once alongside the deterministic failure block. This prevents two model routes
from creating different claims about whether the same cycle reached training,
checkpointed, or progressed after the fault.

An enriched entry contains only compact L2-grounded current-attempt failure
facts plus `route_id`. It does not contain model transcripts, citations, tool
payloads, token metrics, `HistorySummary`, or `L4PolicyOutcome`. Copying L3 into
the record would recursively embed prior history; copying L4 would make the
record depend on policy version and retry state rather than only this attempt's
observations.

At the absolute analysis deadline, the runtime stops waiting and returns the
best result already available. Pending futures are cancelled when possible.
An already-running HTTP worker may unwind after the deadline because Python
threads cannot be forcibly terminated, but its output is abandoned: it cannot
publish a route result, change the closed-cycle action, or mutate the closed
attempt record.
Provider-request timeouts remain clamped to the remaining analysis budget, and
no new retry, model turn, or tool call starts after the deadline.

Re-running analysis for the same workload attempt uses the same
`(job_id, cycle_id)` and atomically replaces its record, initially clearing the
old enriched list. An actual NVRx workload restart uses a new `cycle_id` and
appends a distinct record. Concurrent invocations for one `(job_id, cycle_id)`
are serialized by the runtime. Each invocation has an internal generation, and
an older or post-deadline callback cannot update the replacement record.

## Library And CLI Exercise Paths

The library supports direct stateful execution:

```python
config = load_restart_agent_config("restart_agent.json")
records = InMemoryAttemptRecordStore(
    max_attempts_per_job=10,
    max_total_records=3000,
)
runtime = build_restart_agent_runtime(config, attempt_record_store=records)

runtime.attempt_record_control.seed(initial_records)
for request in ordered_requests:
    result = runtime.analyze(request)

assert runtime.attempt_record_control.records("job-123") == expected_records
```

For manual testing, the CLI may seed one invocation from a plain JSON array of
`AttemptRecord` objects and explicitly export the resulting store:

```text
restart-agent cycle-4.log --job-id job-123 --cycle-id 4 \
  --attempt-records-json-in prior-attempts.json \
  --attempt-records-json-out attempts-through-cycle-4.json
```

Both flags are explicit manual-test operations. The output is the complete
post-upsert in-memory record set, deterministically ordered by `job_id` and
integer `cycle_id`, written atomically as the same plain JSON-array fixture
shape accepted by `--attempt-records-json-in`. The CLI chooses no default
record path,
does not write unless requested, and does not preserve state after the process
exits. A user may edit or branch the exported fixture to construct alternate
history scenarios. Ordered multi-cycle state behavior is also exercised
through library/unit tests that preserve one in-memory store across requests.

These APIs are implemented; `STATUS.md` records their current qualification
level and remaining production work.

## Future MCP And Attrsvc Integration

MCP is a later transport adapter over `RestartAgentRuntime`; it does not own
attempt-record or history-comparison semantics. The product MCP surface may
expose only analysis operations.
If attrsvc hydration is designed later, it may build on the same typed store
boundary. That later design must define transport and lifecycle semantics rather
than assuming the MVP unit-test control is a wire contract.

No attrsvc integration, MCP history endpoint, hydration protocol, or
restart-surviving persistence is required for MVP.

## Verification

The runtime/attempt-record implementation is complete only when tests cover:

- exact-job selection, integer ordering, and exclusion of current/future cycles;
- available-but-empty history versus disabled or identity-ineligible history;
- bounds, oldest-cycle eviction, and disabled history;
- total-record eviction across multiple jobs;
- duplicate-cycle idempotency and deterministic replacement;
- initial L0 record creation with empty enriched entries;
- route-keyed enriched add/replace without duplicate entries or lost updates;
- deterministic-only prior comparison despite stored enriched entries;
- missing job, cycle, and root identity without synthetic joins;
- completed-step/checkpoint-step comparison, fallback iteration comparison,
  and conflicting progress dimensions;
- progress-marker deduplication, observed/not-observed/unknown classification,
  checkpoint load-versus-save separation, and delta/count construction;
- L3 completed-step/checkpoint/failure-position comparison, including
  conflicting-marker fallback to `unknown`;
- immutable invocation views and concurrent access;
- same-key invocation generations, serialization, and rejection of stale or
  post-deadline record updates;
- in-memory seed, inspect, clear, and deterministic CLI fixture round trips;
- multiple jobs and multiple ordered cycles in one runtime;
- model timeout with the deterministic attempt record still available;
- deterministic record behavior independent of enriched route completion order;
- runtime restart with empty state; and
- repeated library invocations sharing one runtime-owned attempt-record store.
