# L0A Complete Evidence Assembly

L0A is the deterministic log-understanding stage. It converts one immutable log
snapshot into complete typed evidence, selects deterministic decision evidence,
and supplies the authoritative source facts from which the runtime constructs
route-independent current-attempt artifacts. Decision evidence distinguishes a
strict initiating primary from a weaker selected observation when the initiating
failure is absent from the log. L0A runs before any model call.

`SCHEMA.md` owns exact serialized shapes. `PATTERN_REGISTRY.md` owns executable
pattern intent. `L0B.md` owns the bounded model-facing projection.

L0A ends at deterministic evidence. Runtime attempt-record assembly derives
compact current-attempt artifacts from that evidence; model interpretation,
grounding, history comparison, and action policy belong to L1-L4.

## Purpose And Authority

L0A exists to turn large, noisy, interleaved training logs into auditable
structure without asking a model to discover basic chronology or repeatedly
rendered events.

```text
immutable log snapshot + product-defined L0 settings
  -> detection
  -> contextualization
  -> L0Bundle
  -> deterministic primary and observation selection
  -> DecisionEvidence

runtime assembly from L0A outputs
  -> AttemptProgressSummary + deterministic AttemptFailureFacts
  -> AttemptRecord
```

L0A is authoritative for:

- source line numbering and log metadata;
- deterministic registry matches and normalized occurrence grouping;
- progress, checkpoint, setup, recovery, and termination observations;
- candidate anchors, bounded context windows, failure episodes, and distributed
  incidents;
- deterministic primary selection, selected-observation selection, and their
  separate identities;
- root-observer ranks and unattributed root occurrences for the selected
  primary occurrence group;
- selection, cap, truncation, and lossiness accounting.

L0A produces structural, pre-semantic evidence. Downstream stages may enrich or
audit its interpretation, but they do not mutate the L0A bundle or deterministic
primary. L0A does not infer semantic ownership, compare prior attempts, consume
retry budgets, or choose an action.

## Inputs And Outputs

| Direction | Contract | Meaning |
| --- | --- | --- |
| Input | `LogSnapshot` | Immutable source byte boundary, UTF-8 logical-line access with localized malformed-byte replacement, and stable original line numbers. |
| Input | L0 settings and registries | Product-defined patterns, context limits, grouping rules, and evidence caps. |
| Output | `L0Bundle` | Complete typed evidence over the inspected snapshot. |
| Output | `DecisionEvidence` | Canonical policy-relevant facts and references selected from the bundle. |

The runtime derives, but L0A does not directly emit, two compact artifacts from
these outputs:

| Runtime-derived artifact | L0A source | Meaning |
| --- | --- | --- |
| `AttemptProgressSummary` | `L0Bundle` progress and coverage facts | Route-independent current-attempt progress stored once in `AttemptRecord`. |
| deterministic `AttemptFailureFacts` | `DecisionEvidence` | Optional root identity, optional observation-only identity, outcome, position, policy-neutral failure classifiers, root-observer facts, locality, and affected entity when deterministically available. |

`AttemptRecordAssembler` owns their construction and the runtime owns record
commit. This materialization does not reinterpret L0A evidence.

Complete does not mean a raw copy of every line. L0A may retain bounded windows
and representative occurrences when it records the source counts, samples,
omissions, caps, and lossiness needed to explain that reduction.

Context-window construction collects seed candidates using the
`episode_cause_signal_registry` rule, then de-duplicates them by source line.
Duplicate line numbers retain the first, highest-priority reason.
`eligible_seed_count` is the number of unique eligible seed lines after this
de-duplication and before the eight-window cap is applied. The ordered seed
categories are:

1. each preliminary failure episode's start and terminal line, with episodes
   and their two boundaries in source order;
2. explicit cause-confirmation lines in source order;
3. up to three high-signal lines from the deterministic sampler;
4. the first representative line for each distinct registry-id and normalized
   pattern group, in stable registry-match order.

Stable registry-match order means ascending source line. When more than one
registry row matches the same line, their order is the declaration order in the
registry. Registry compaction preserves that order for equal-line matches, and
context seeding retains the first occurrence of each
`(registry_id, normalized_pattern)` group. Registry declaration order is
therefore behaviorally significant and must not be changed as a formatting-only
edit.

The high-signal sampler starts with the first signal. It next considers signals
by high-signal priority and descending source line, suppressing candidates
within 25 lines of an already selected signal. High-signal priority is ordered
as follows; internal numeric weights are not part of the contract:

| Order | Signal class |
| ---: | --- |
| 1 | Traceback |
| 2 | Bare process-killed signal |
| 3 | Runtime error or exception |
| 4 | Fatal or critical message |
| 5 | Assert or bounds failure |
| 6 | Generic error |
| 7 | Timeout |
| 8 | Other high-signal text |

The sampler then considers remaining signals from the end of the log and
finally fills any open slots in source order. Its returned seeds are source
ordered. After category-level de-duplication, the first eight seeds become
context windows and the rest are counted as omitted. The bundle reports the
reduction explicitly:

```json
{
  "context_window_selection": {
    "rule": "episode_cause_signal_registry",
    "eligible_seed_count": 12,
    "selected_seed_count": 8,
    "omitted_seed_count": 4,
    "limit": 8,
    "cap_hit": true
  },
  "caps_hit": ["context_window_seeds"]
}
```

`rule` MUST be `episode_cause_signal_registry` for this contract.
`eligible_seed_count` MUST equal `selected_seed_count + omitted_seed_count`,
and `selected_seed_count` MUST equal the number of emitted context windows.
Bundle assembly and this accounting are one transaction: inconsistent
accounting fails L0A without emitting a partial `L0Bundle` or
`DecisionEvidence`.

L0A-omitted seeds do not enter `L0Bundle.context_windows`. L0B therefore starts
its own selection accounting from L0A's emitted windows, not from L0A's
original eligible-seed set.

The local-file implementation feeds fixed-size byte chunks through one
incremental line decoder. It retains an incomplete byte tail until a line is
complete, then adds that line exactly once to an append-only observation index.
Each completed line is classified once into compact channels such as progress,
registry matches, high-signal candidates, checkpoint operations, and teardown.
Cheap exact cues route a line only to relevant regex families; the cue gates are
an execution optimization and do not change the evidence definitions.
Registry hits first become lightweight observations containing source identity,
normalized shape, outcome, and locality. L0A aggregates repeated observations
and selects representative head/tail samples before constructing full
`FailureEvidence`, source context, and fingerprints. Counts, rank/node/GPU
spread, first occurrence, and sample lines still describe the complete group.
Canonical reducers consume these compact channels to build the bundle.
Selected source text is read through one temporary, reduction-scoped source
view shared by contextual reducers. Bounded selection projections preserve the
existing evidence caps, while repeated reads reuse the temporary source text.
The reuse cache may evict old entries, but eviction MUST NOT remove lines from
an in-progress batch lookup or from the current reduction's evidence.
The view is released as soon as the bundle is assembled; complete decoded log
content is not retained between reductions.

Production `chunked` mode does not retain a second decoded copy of the log.
Completed lines are classified once and discarded after their compact
observation channels are updated. A compact byte-offset index plus an open,
boundary-limited source handle supplies later random or streaming line access
to reducers, L1 tools, and L2 grounding. Byte chunks are released after decoder
feed; only the incomplete byte tail remains between reads.

L0A observes only completed physical lines. A chunk boundary never creates a
logical line boundary or causes an incomplete line to be decoded or classified.
Bytes after the last `LF` remain pending until a later chunk supplies `LF`, or
until terminal finalization closes the source boundary and emits the final
unterminated line. Consequently, chunk boundaries must not change logical-line
assignment, replacement-character placement for malformed encoded bytes, line
numbering, or canonical L0A output.

`single_snapshot` changes byte delivery to one chunk for parity tests; it does
not define another reducer path. Both modes must produce the same canonical
L0A payload for the same source boundary.

Read mode and source-view storage mode are independent:

| Dimension | Values | Meaning |
| --- | --- | --- |
| Byte read mode | `chunked`, `single_snapshot` | How source bytes are delivered to the incremental decoder. |
| Source-view storage mode | `indexed_file`, `memory` | How `LogSnapshot` retrieves decoded source lines after ingestion. |

Production normally combines `chunked` ingestion with `indexed_file` storage.
Parity tests may use `single_snapshot` ingestion with the same indexed-file
storage, while caller-provided snapshots may use `memory`. Therefore,
`single_snapshot` does not imply in-memory storage, and `chunked` does not mean
that evidence is retained as chunks.

## Evidence Objects

L0A uses a small set of related objects:

| Object | Meaning |
| --- | --- |
| Normalized occurrence group | Repeated observations with the same volatile-token-stripped shape. Preserves count, first occurrence, samples, locality spread, and registry role. |
| Candidate anchor | A high-signal source line selected for chronology or context assembly. It is a retrieval anchor, not a root-cause conclusion. |
| Bounded context window | Original source lines around one or more anchors, with line range, selection reason, and truncation metadata. |
| Failure episode | One local causal story: prior progress, initiating/terminal exception chain, downstream teardown, later progress or recovery, and final episode status. |
| Distributed mechanism incident | An inherently distributed failure such as a collective timeout. One observer is sufficient. |
| Distributed fanout incident | The same ordinary mechanism observed from at least two distinct ranks in one progress segment. |
| Selected observed failure | One canonical terminal failure surface retained when no initiating primary is supportable. It is a correlation and policy-availability anchor, not a causal conclusion. |

A single-rank ordinary exception produces an episode but not a distributed
incident. A single-observer collective timeout may produce both an episode and
a distributed-mechanism incident.

### Fault Lifecycle Example: NCCL RDMA Port Events

L0A treats a correlated NCCL RDMA port lifecycle as one failure episode.
Repeated rank renderings first collapse into occurrence groups; the episode
then joins the initiating port error to any same-port recovery evidence and
compatible downstream failure.

```text
terminal episode
  completed iteration
  -> port error
  -> no matching port active
  -> no later completed progress
  -> NCCL collective timeout

recovered episode
  port error
  -> client reregistration
  -> port active
  -> later completed iteration
```

For the terminal form, the port error may become the episode identity and the
NCCL timeout remains downstream failure evidence. For the recovered form, the
episode status is `recovered`; a later CUDA, non-finite, or unrelated terminal
failure starts a separate episode. Multiple port-error/recovery sequences in
one log remain separate episodes.

The correlation key is node plus device and port. Rank, PID, timestamp, and
event code remain provenance. `NCCL WARN NET/IB` is retained as source dialect
`nccl_net_ib`; it does not by itself establish whether the physical network is
InfiniBand or RoCE. A port-active event without an earlier matching port error
is retained only as lifecycle context.

The run-level `progress_after_failure_episode` fact is scoped to the episode
containing the selected deterministic primary. Progress after an earlier
recovered episode remains visible on that episode and does not imply progress
after the terminal failure.

`rank_spread` remains local to the evidence object that contains it. For
example, a distributed incident's spread may include ranks that emitted the
initiating event, downstream copies, cascades, or teardown. It is not a
root-only observation.

For the selected primary, L0A separately derives two generic facts from its
root occurrence group:

- `root_observer_ranks`: distinct parseable ranks that directly emitted the
  initiating root occurrence; and
- `unattributed_root_occurrence_count`: root occurrences for which L0A could
  not parse a rank.

Downstream copies, cascades, diagnostics, and teardown do not contribute to
either field. `root_observer_ranks` is unavailable when the selected primary
cannot be associated unambiguously with a root occurrence group.

Availability is explicit:

| Root group state | `root_observer_ranks` | `unattributed_root_occurrence_count` |
| --- | --- | --- |
| Associated, attributed ranks parsed | Non-empty array | Non-negative integer, commonly zero |
| Associated, no attributed rank parsed | Empty array | Non-negative integer |
| Association unavailable | `null` | `null` |

The two fields are available or unavailable together. L0A computes them from
the complete lightweight occurrence aggregation before representative
sampling, so L0B selection and projection caps do not affect the counts.
`DecisionEvidence.coverage_lossiness.root_observer_facts` records whether the
association was complete and why it was unavailable. Any future L0A cap that
could make these counts incomplete must emit both values as `null` rather than
present a partial count as complete.

These facts are preserved unchanged in deterministic and enriched
`AttemptFailureFacts`; L1 and L2 do not reinterpret them. Rank identities remain
useful for audit, but cross-attempt policy compares the observer count and does
not require the same rank identity after restart.

Root-observer association is identity-anchor-specific. L0A uses only the
occurrence group whose first or retained sample lines contain the selected
identity anchor, restricted to the primary registry id when one exists. Other
same-registry groups may remain in `selected_evidence_references` as narrative
support, but their rank spread and unattributed occurrences do not become root
locality. When no anchored group is available, both root-observer fields are
`null`; L0A does not infer locality from a generic registry-id match.

## Overlapping Observations And Causal Narrative

Initialization activity, progress, fault candidates, recovery observations,
primary failure, cascade/fanout, and teardown are analytical roles, not
contiguous phases of the source log. Rank skew, asynchronous execution, and
buffered writes may place observations with several roles next to one another
or make their file order differ from their causal order. Different ranks may
also be initializing, progressing, cascading, and tearing down at the same
time.

L0A therefore classifies individual observations, associates them using
available rank, operation, artifact, timestamp, and progress-segment facts,
and reconstructs one or more failure episodes. A fault candidate followed by
compatible recovery or continued progress remains non-terminal evidence. The
selected primary is the best-supported initiating observation in the terminal
episode. When that initiating observation is not available, L0A may separately
select one terminal failure surface without promoting it to primary. Cascade and
teardown observations remain consequences or observations of unknown relation.
This logical narrative does not partition every source line into exactly one
phase.

A ranked traceback is one rank-local sequence even when thousands of other
ranks render frames between its lines. L0A closes that traceback at the first
explicit exception summary from the same rank, unless that rank starts another
traceback first. The traceback header remains the episode start; the explicit
exception is its terminal identity. Unranked tracebacks retain a bounded
physical-line scan because no rank key is available for safe correlation.

The reconstructed story has this general shape:

```text
initialization / progress
  -> fault candidate
       -> recovery or continued progress
       -> progress continues                    (may repeat)
  -> terminal primary failure
       -> cascade / distributed fanout
       -> teardown
```

This is a logical causal narrative reconstructed from overlapping
observations, not a sequence of contiguous source-log regions.

## Assembly Algorithm

The implementation follows three deterministic transformations:

1. **Detect.** Decode completed source lines incrementally and classify each
   line once. Parse rank and node fields, progress markers, checkpoint/setup
   markers, registry matches, retry-lifecycle state, diagnostic context, generic
   traceback/fatal/process-termination anchors, path-access facts, and
   high-signal structural lines.
2. **Contextualize.** Aggregate repeated lightweight observations before full
   evidence enrichment, build context windows, attach progress before and after
   candidate faults, group exception chains into episodes, group distributed
   events into incidents, distinguish causes from cascades/teardown, and attach
   explicit cause confirmations.
3. **Assemble and select.** Collect eligible registry roots, terminal episode
   identities, distributed-incident identities, and explicit cause
   confirmations. Reject retry-pending, recovered, teardown, cascade,
   diagnostic, and generic announcement candidates unless stronger causal
   structure reclassifies them.
   Apply the deterministic primary ordering below. If no primary is supportable,
   apply the separate selected-observation ordering. Attach alternatives and
   related evidence, freeze `L0Bundle`, and then freeze `DecisionEvidence`.

The source log is not rescanned while selecting `DecisionEvidence`.

## Progress And Outcome

L0A separates:

- `application_progress`: a comparable monotonic training marker advanced;
- `checkpoint_progress`: a checkpoint save completed successfully;
- `setup_progress`: initialization advanced, without proving forward training;
- `recovery_evidence`: the same operation later succeeded, retried
  successfully, or explicitly skipped/quarantined a bad input;
- `failure_iteration`: a position attached to failure, not proof that the
  iteration completed.

Repeated output, liveness, teardown, retry counters, rank ids, timestamps,
checkpoint starts, and failed checkpoints are not progress by themselves.

Progress is conservative when a workload dialect is not recognized:

- `ProgressFacts.training_progress_dialect_recognized` records whether L0A
  recognized a compatible completed-training-progress marker dialect;
- `ProgressFacts.checkpoint_progress_dialect_recognized` records whether L0A
  recognized a compatible completed-checkpoint marker dialect;
- marker absence is `not_observed` only when the corresponding dialect field is
  true and no completed marker was found;
- otherwise the corresponding status is `unknown`;
- `progress_after_failure=not_observed` requires a finalized, fully scanned
  source boundary, `training_progress_dialect_recognized=true`, and no
  compatible progress marker after the selected failure; the selected failure
  may be the final source line;
- a later interleaved marker proves job continuation, not recovery of a
  particular rank, node, GPU, or component.

Candidate outcome is one of:

- `terminal`: tied to termination with no later recovery/progress;
- `recovered`: the same operation later succeeded;
- `progressed_after`: compatible application or checkpoint progress followed;
- `retry_pending`: the observed failure belongs to an explicitly incomplete
  retry sequence;
- `unresolved`: the failure was observed, but neither recovery nor terminality
  was established. This is an operational outcome, not uncertainty about
  whether the failure occurred or what caused it.

Only `terminal` or terminal-linked `unresolved` observations are eligible for
the deterministic primary. Eligibility does not imply `STOP`.

Retry lifecycle is separate from failure identity. L0A recognizes explicit
attempt counters and retry-transition language as `pending`, `succeeded`, or
`exhausted`, retaining the attempt and maximum-attempt values when present. A
pending or succeeded retry remains evidence but cannot become the deterministic
primary or seed history. An explicitly exhausted retry may be primary-eligible
when its normal outcome and causal-role checks also pass. Unrecognized or
ambiguous wording does not manufacture a retry state.

## Noise And Attention

Noisy faults followed by compatible progress remain compressed context rather
than consuming top-anchor and context-window budget repeatedly. Examples include
filesystem warnings, network disturbances, port warnings, repeated path-access
warnings, and downstream rank fanout.

Such evidence is promoted only when it is:

- temporally tied to the terminal episode;
- the earliest unrecovered high-signal candidate;
- an explicit cause confirmation; or
- the strongest available terminal observation.

Stable CUDA/PyTorch diagnostic boilerplate, cleanup messages, stack-trace
warnings, and scheduler cancellation are retained with diagnostic, teardown,
or cascade roles. They cannot become a primary merely because their text is
high severity.

## Deterministic Primary And Observation Identity

Primary selection uses chronology, causal role, episode status, explicit cause
confirmation, and available progress context. Registry class alone is not
decisive. Only `terminal` and terminal-linked `unresolved` episodes participate;
`retry_pending`, `recovered`, and `progressed_after` candidates remain evidence
but cannot be the deterministic primary.

L0A selects the deterministic primary using this three-way flow:

1. If eligible episodes exist, select the earliest eligible initiating episode
   and record `primary_episode_selection_basis` as
   `earliest_eligible_initiating_episode`.
2. Otherwise, if an eligible terminal registry root exists outside an episode,
   select that root, leave `primary_episode_id` unset, and record the basis as
   `eligible_registry_root_without_episode`.
3. Otherwise, retain no primary, leave `primary_episode_id` unset, and record
   the basis as `not_available`.

For episode selection, L0A removes `retry_pending`, recovered,
`progressed_after`, cascade-only, teardown-only, diagnostic-only, and
unsupported bare-termination episodes. It
orders the remaining terminal or terminal-linked unresolved episodes by
initiating identity line, then episode start line, terminal line, and stable
episode id. The first episode in that source ordering is the primary episode.
This makes the earliest eligible initiating episode after the last applicable
progress marker authoritative; the presence of a registry match in a later
episode does not displace it.

Within the selected episode, identity precedence is:

| Order | Identity | Selection rule |
| ---: | --- | --- |
| 1 | Concrete initiating failure | Preserve an eligible registry root or a causally linked specific precursor. |
| 2 | Explicit cause confirmation | Promote a later scheduler, kernel, or runtime confirmation only when the existing episode identity is a cause-unknown process termination. |
| 3 | Normalized initiating terminal identity | When no registry root exists, derive a stable root fingerprint only from a structurally specific terminal exception that the episode supports as initiating. |
| 4 | No primary | Retain no primary when only teardown, cascade, diagnostic, generic announcement, or unsupported bare termination remains. |

For a traceback episode, the header is structural context rather than a
concrete initiating identity. When the episode contains a linked explicit
terminal exception, primary selection uses that exception and its normalized
fingerprint, regardless of the number of interleaved physical lines.

Cause confirmations are always retained as supporting evidence when they link
to the nearest process-termination episode and no compatible progress
intervenes. They do not replace an already concrete initiating failure. Within
the same precedence tier, source line orders candidates. Exact-line ties first
use the preferred registry role: `cause_confirmation` when the episode identity
was promoted from an explicit cause confirmation, otherwise `root_candidate`.
The role `either` ranks second, followed by the remaining eligible registry
role. Further ties order non-generic before generic observations, then ascending
`registry_id`, `failure_class`, signature, and root fingerprint. Insertion order
never resolves a tie.

For this exact-line registry tie-break only, `registry_id=observed_exception`
is generic and every other registry match is non-generic. Genericity is not
inferred from registry names or signature text and is not a general taxonomy
for other L0 evidence objects.

For example, `100: Killed` followed without intervening progress by
`110: slurmstepd ... oom_kill` selects line 110 because line 100 does not name a
cause. If line 100 instead contains `RuntimeError: CUDA out of memory`, line 100
remains primary and line 110 is retained as confirmation evidence.

Generic error announcements and outer wrappers do not replace a concrete
exception merely because they occur earlier. `selection_summary` records the
selected episode id, episode-selection basis, line, and identity basis so corpus
review can distinguish episode ordering, registry, episode-derived, and
cause-confirmation choices.

The deterministic branch produces two independent history values when source
evidence supports them:

- `root_fingerprint`: stable observed failure mechanism;
- `affected_entity`: exact operation-associated `artifact`.

`root_fingerprint` is unavailable when the initiating primary is unavailable.
Null is absence of comparable root identity, not a shared `unknown` value; two
null roots never match.

Rank, node, GPU, timestamp, cycle id, and source line remain occurrence
metadata. They do not enter the root fingerprint. Exact artifact paths are
retained in the affected-entity identity because they distinguish otherwise
similar failures.

For an operation-associated artifact, L0A uses the strongest unambiguous
identity observed for the failing operation. A checkpoint load normally uses
the normalized checkpoint path plus checkpoint iteration; an explicit
shard/file/object identity refines it when available. Buffer-local details such
as a Unicode decoder position, offending byte, rank, or traceback line remain
diagnostic observations. They do not enter the affected-entity identity because
the same artifact may be buffered or sharded differently on replay.

Operation lifecycle constrains this association. When an operation has a
completion followed by observable progress before the selected primary, it is
historical comparison evidence and cannot inherit that later failure. A mixed
cross-rank success/failure remains eligible when no intervening progress proves
continuation beyond the operation. If more than one eligible operation supplies
different artifact identities, L0A leaves `affected_entity` unavailable.

After primary selection, L0A may also attach one explicitly failed filesystem
path from the selected primary line. The path must be absolute, source-grounded,
and unique on that line. Path extraction enriches the selected primary only; it
does not promote incidental, retry-pending, recovered, cascade, or teardown
path messages.

### Selected Observation When Primary Is Absent

L0A preserves every typed fault observation whether or not a primary can be
established. It additionally selects one canonical observation only when that
selection is deterministic and useful for describing the terminal failure
surface.

Selection is ordered and non-causal:

1. Collapse repeated rank copies into normalized occurrence groups and
   distributed fanout incidents.
2. Exclude retry-pending, recovered, progressed-after, successful-operation,
   diagnostic-frame, and generic-boilerplate groups.
3. Restrict remaining candidates to the terminal episode after the last
   applicable durable progress marker, with no compatible later recovery or
   progress.
4. Prefer an exhausted retry, then a terminal observation, then a
   terminal-linked unresolved observation. Within that outcome tier, prefer
   `unknown` causal role over known cascade and teardown roles.
5. Use the normalized group's first source occurrence as its canonical line;
   per-rank fanout and later copies remain samples of the same group.
6. If independent groups remain tied at the highest eligible tier and L0A
   cannot establish a relationship between them, retain no selected
   observation. Do not force an identity merely to make history available.

When a deterministic primary exists, `selected_observed_failure` is null and
the observation-only fallback is not used. Otherwise, the selected observation
produces an `observation_fingerprint` with `identity_kind=observation_only`.
The fingerprint normalizes volatile rank, node, GPU, timestamp, cycle, source
line, address, and descriptor values. It records that the same failure surface
was observed; it does not establish the same initiating cause and cannot feed a
root-scoped history ledger.

For example, after completed iteration 640001, repeated TCPStore connection
losses may form one selected observation while a later `Attempt 1/4 ... re-try`
file warning remains retry-pending evidence. The result may therefore contain a
TCPStore observation fingerprint with a null primary and null root fingerprint.

## Decision Evidence

`DecisionEvidence` is a standalone immutable type built from `L0Bundle`. It
contains:

- `deterministic_primary_candidate`;
- `selected_observed_failure` and its observation-only fingerprint, when one
  can be selected deterministically;
- `canonical_observed_identity`;
- references to selected anchors, windows, occurrence groups, episodes, and
  incidents;
- failure position and outcome;
- progress and checkpoint state;
- operation and artifact facts;
- later-progress and recovery observations;
- locality;
- coverage, lossiness, and provenance.

`DecisionEvidence` is intrinsically model-safe because the deterministic and
enriched runtime branches share the exact same object. It MUST NOT contain the request
source `log_path`, basename, parent-directory components, eval labels, case ids,
or any identity inferred from that source location. Its provenance is closed to
`source`, `log_line_count`, `log_byte_size`, `log_rescanned`, and `model_used`.
Full source identity remains private in `L0Bundle` and trace. Paths observed in
log content that identify workload artifacts, datasets, configuration, or
sockets remain valid evidence and are not source-location metadata.

It selects canonical policy-relevant facts and references. It does not copy all
L0A objects, create model prose, compare history, or emit an action. The exact
same object feeds deterministic and enriched runtime processing and remains
available to L2 and trace. L0B creates a compact model-facing projection of it;
L0B does not mutate this canonical object or expose exhaustive locality member
lists merely because they are retained internally.

## Degraded Behavior

- Missing, unreadable, or empty logs produce the public log-unavailable result
  outside normal L0A evidence assembly.
- Malformed UTF-8 bytes are replaced only in the affected physical line and
  counted; they do not fail analysis or trigger a whole-file replay.
- Unknown progress remains `unknown`; absence is not converted into evidence.
- Incomplete source-boundary coverage or an unrecognized progress dialect
  serializes `progress_after_failure=unknown` and records lossiness. The word
  `unresolved` is reserved for `FaultOutcome`; it is not a progress-observation
  state.
- No deterministic primary is a valid outcome. When one terminal observation
  is still selectable, L4 may apply root-independent general retry and
  same-job progress accounting. When neither exists, L4 returns
  `no_primary_failure`. L0A never invents a root to unlock root-scoped policy.

## Tracing

The trace or its lossless artifact references must preserve:

- exact source identity and line-numbering convention;
- the complete `L0Bundle`;
- exact `DecisionEvidence`;
- deterministic progress and failure facts;
- registry and L0 contract versions;
- counts, caps, omissions, truncation, and anomalies;
- deterministic payload hashes.

## Service Logging

L0A emits one INFO completion event per finalized source boundary:

```text
event=restart_agent.l0a.completed status=<status> wall_clock_s=<seconds>
source_ingest_s=<cumulative-seconds>
evidence_assembly_s=<cumulative-seconds>
decision_evidence_s=<cumulative-seconds>
cumulative_compute_s=<cumulative-seconds>
reused=<bool> source_bytes=<count> source_lines=<count>
occurrence_groups=<count> candidate_anchors=<count>
failure_episodes=<count> distributed_incidents=<count>
primary_line=<line|unknown> primary_class=<class|unknown>
selected_observation_line=<line|unknown>
identity_kind=<root|observation_only|none>
root_fingerprint_ready=<bool> observation_fingerprint_ready=<bool>
```

Decision Evidence selection emits its own INFO substage event:

```text
event=restart_agent.decision_evidence.completed wall_clock_s=<seconds>
cumulative_selection_s=<cumulative-seconds>
primary_line=<line|unknown> selected_reference_groups=<count>
selected_observation_line=<line|unknown>
identity_kind=<root|observation_only|none>
```

`wall_clock_s` is work performed by the final preparation call and may be zero
when that call reuses a precomputed boundary. The cumulative fields preserve
the source-ingestion, structured-evidence, and Decision Evidence work performed
before finalization; `cumulative_compute_s` is their sum and excludes idle poll
time and terminal-drain waiting.

When source-ingestion metrics are available, DEBUG emits
`restart_agent.l0a.detail` with decode and index/classification timing, chunk
and byte counts, malformed-byte counts, resets, rereads, caps, truncations, and
anomaly counts. It does not emit source text or complete evidence objects.

## KPIs

Quality KPIs require reviewer-owned gold. Missing labels produce
`not_available`, not failure.

| Quality KPI | Example |
| --- | --- |
| Primary evidence coverage | Gold accepts lines 100-110; at least one L0A candidate/window covers that range. |
| Selected primary accuracy | L0A selects line 104, which is inside the accepted gold range. |
| Selected-observation accuracy | Gold says the initiating cause is absent but accepts the TCPStore reset at line 30368 as the terminal visible failure surface; L0A selects that group without promoting it to primary. |
| Primary-abstention accuracy | A log containing only cascade/teardown evidence retains `primary=null` instead of manufacturing an initiating cause. |
| Progress/checkpoint precision and recall | All labeled completed-iteration markers are found and failed checkpoint starts are not counted as saves. |
| Typed-event accuracy | A teardown line is typed `teardown`, not `primary_failure`. |
| Episode/incident construction accuracy | Repeated rank renderings form one episode/fanout incident rather than independent roots. |
| Outcome accuracy | A fault followed by compatible progress is `progressed_after`. |
| Retry-lifecycle accuracy | `Attempt 1/4 ... then re-try` is `pending`, while an explicitly exhausted final attempt is `exhausted`; only the latter remains primary-eligible. |
| Identity accuracy and stability | The expected root/entity or observation-only identity is present and stable under rank/timestamp reordering; observation identity never populates the root field. |
| Lossiness correctness | Every omitted or truncated collection is represented in metadata. |

Operational metrics include:

- L0A wall time and scan throughput;
- source bytes, decoded lines, chunks, read/storage modes, and encoding;
- pending-tail, source-reset, and encoding-replay counts;
- occurrence-group, window, anchor, episode, and incident counts;
- serialized bundle size;
- cap, truncation, omission, and anomaly counts;
- bundle reuse status;
- deterministic replay/hash consistency;
- Decision Evidence selection time and referenced-object counts.

## Example

```text
1000 [rank 7] iteration 418 completed
1012 [rank 7] RuntimeError: CUDA out of memory
1013 [rank 9] RuntimeError: CUDA out of memory
1030 [rank 7] destroy_process_group() called during shutdown
```

L0A creates:

- one progress marker at line 1000;
- one normalized occurrence group for the repeated OOM observation;
- a bounded window around lines 1012-1030;
- one terminal failure episode;
- one distributed-fanout incident because two ranks reported the same ordinary
  mechanism;
- a teardown role at line 1030;
- a deterministic primary and root identity anchored at line 1012;
- `progress_after_failure=not_observed` because the finalized source boundary
  contains no later iteration marker and the iteration dialect is recognized.

`DecisionEvidence` selects that episode and its supporting references. L0A
still does not decide whether a restart can recover.
