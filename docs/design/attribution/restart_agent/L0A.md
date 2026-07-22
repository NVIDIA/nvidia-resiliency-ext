# L0A Complete Evidence Assembly

L0A is the deterministic log-understanding stage. It converts one immutable log
snapshot into complete typed evidence, selects deterministic decision evidence,
and constructs the route-independent current-attempt facts used by the
deterministic path. It runs before any model call.

`SCHEMA.md` owns exact serialized shapes. `PATTERN_REGISTRY.md` owns executable
pattern intent. `L0B.md` owns the bounded model-facing projection.

L0A ends at deterministic evidence and current-attempt facts; model
interpretation, grounding, history comparison, and action policy belong to
L1-L4.

## Purpose And Authority

L0A exists to turn large, noisy, interleaved training logs into auditable
structure without asking a model to discover basic chronology or repeatedly
rendered events.

```text
immutable log snapshot + product-defined L0 settings
  -> detection
  -> contextualization
  -> L0Bundle
  -> deterministic selection
  -> DecisionEvidence
  -> AttemptProgressSummary + deterministic AttemptFailureFacts
```

L0A is authoritative for:

- source line numbering and log metadata;
- deterministic registry matches and normalized occurrence grouping;
- progress, checkpoint, setup, recovery, and termination observations;
- candidate anchors, bounded context windows, failure episodes, and distributed
  incidents;
- deterministic primary selection and observed failure identity;
- selection, cap, truncation, and lossiness accounting.

L0A observations are provisional evidence. L0A does not infer semantic
ownership, compare prior attempts, consume retry budgets, or choose an action.

## Inputs And Outputs

| Direction | Contract | Meaning |
| --- | --- | --- |
| Input | `LogSnapshot` | Immutable source byte boundary, UTF-8 logical-line access with localized malformed-byte replacement, and stable original line numbers. |
| Input | L0 settings and registries | Product-defined patterns, context limits, grouping rules, and evidence caps. |
| Output | `L0Bundle` | Complete typed evidence over the inspected snapshot. |
| Output | `DecisionEvidence` | Canonical policy-relevant facts and references selected from the bundle. |
| Output | `AttemptProgressSummary` | Route-independent current-attempt progress facts. |
| Output | deterministic `AttemptFailureFacts` | Root fingerprint, outcome, position, locality, and affected entity when deterministically available. |

Complete does not mean a raw copy of every line. L0A may retain bounded windows
and representative occurrences when it records the source counts, samples,
omissions, caps, and lossiness needed to explain that reduction.

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

`single_snapshot` changes byte delivery to one chunk for parity tests; it does
not define another reducer path. Both modes must produce the same canonical
L0A payload for the same source boundary.

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

A single-rank ordinary exception produces an episode but not a distributed
incident. A single-observer collective timeout may produce both an episode and
a distributed-mechanism incident.

## Assembly Algorithm

The implementation follows three deterministic transformations:

1. **Detect.** Decode completed source lines incrementally and classify each
   line once. Parse rank and node fields, progress markers, checkpoint/setup
   markers, registry matches, diagnostic context, generic
   traceback/fatal/process-termination anchors, path-access facts, and
   high-signal structural lines.
2. **Contextualize.** Aggregate repeated lightweight observations before full
   evidence enrichment, build context windows, attach progress before and after
   candidate faults, group exception chains into episodes, group distributed
   events into incidents, distinguish causes from cascades/teardown, and attach
   explicit cause confirmations.
3. **Assemble and select.** Collect eligible registry roots, terminal episode
   identities, distributed-incident identities, and explicit cause
   confirmations. Reject recovered, teardown, cascade, diagnostic, and generic
   announcement candidates unless stronger causal structure reclassifies them.
   Select the earliest supported initiating observation in the terminal
   episode, attach alternatives and related evidence, freeze `L0Bundle`, and
   then freeze `DecisionEvidence`.

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

- `not_observed` is valid only when L0A recognized a compatible progress or
  checkpoint dialect and found no completed marker;
- otherwise the corresponding status is `unknown`;
- `progress_after_failure=not_observed` also requires source content after the
  selected failure;
- a later interleaved marker proves job continuation, not recovery of a
  particular rank, node, GPU, or component.

Candidate outcome is one of:

- `terminal`: tied to termination with no later recovery/progress;
- `recovered`: the same operation later succeeded;
- `progressed_after`: compatible application or checkpoint progress followed;
- `unresolved`: neither terminality nor recovery was established.

Only `terminal` or terminal-linked `unresolved` observations are eligible for
the deterministic primary. Eligibility does not imply `STOP`.

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

## Deterministic Primary And Identity

Primary selection uses chronology, causal role, episode status, explicit cause
confirmation, and available progress context. Registry class alone is not
decisive.

The deterministic ordering is:

1. preserve an eligible registry root when terminal episode context confirms
   it;
2. prefer a more specific initiating precursor linked to that episode;
3. use the terminal episode identity when no registry root covers the episode;
4. use an explicit cause confirmation linked to an otherwise cause-unknown
   termination;
5. retain no primary when only teardown, cascade, diagnostic, or unexplained
   process termination remains.

Generic error announcements and outer wrappers do not replace a concrete
exception merely because they occur earlier. `selection_summary` records the
selected line and basis so corpus review can distinguish registry,
episode-derived, and cause-confirmation choices.

The deterministic branch produces two independent history values when source
evidence supports them:

- `root_fingerprint`: stable observed failure mechanism;
- `affected_entity`: exact `artifact` or `data_position`.

Rank, node, GPU, timestamp, cycle id, and source line remain occurrence
metadata. They do not enter the root fingerprint. Exact artifact paths and data
positions are retained in the affected-entity identity because they distinguish
otherwise similar failures.

For an operation-associated artifact, L0A uses the strongest unambiguous
identity observed for the failing operation. A checkpoint load normally uses
the normalized checkpoint path plus checkpoint iteration; an explicit
shard/file/object identity refines it when available. Buffer-local details such
as a Unicode decoder position, offending byte, rank, or traceback line remain
diagnostic observations. They do not enter the affected-entity identity because
the same artifact may be buffered or sharded differently on replay.

## Decision Evidence

`DecisionEvidence` is a standalone immutable type built from `L0Bundle`. It
contains:

- `deterministic_primary_candidate`;
- `canonical_observed_identity`;
- references to selected anchors, windows, occurrence groups, episodes, and
  incidents;
- failure position and outcome;
- progress and checkpoint state;
- operation and artifact facts;
- later-progress and recovery observations;
- locality;
- coverage, lossiness, and provenance.

It selects canonical policy-relevant facts and references. It does not copy all
L0A objects, create model prose, compare history, or emit an action. The exact
same object feeds the deterministic recommendation and is embedded in L0B.

## Degraded Behavior

- Missing, unreadable, or empty logs produce the public log-unavailable result
  outside normal L0A evidence assembly.
- Malformed UTF-8 bytes are replaced only in the affected physical line and
  counted; they do not fail analysis or trigger a whole-file replay.
- Unknown progress remains `unknown`; absence is not converted into evidence.
- A cap or unavailable after-context leaves the affected fact unresolved and
  records lossiness.
- No deterministic primary is a valid outcome. L4 returns the degraded
  `no_primary_failure` result without recurrence accounting; L0A does not
  invent a root.

## Tracing

The trace or its lossless artifact references must preserve:

- exact source identity and line-numbering convention;
- the complete `L0Bundle`;
- exact `DecisionEvidence`;
- deterministic progress and failure facts;
- registry and L0 contract versions;
- counts, caps, omissions, truncation, and anomalies;
- deterministic payload hashes.

## KPIs

Quality KPIs require reviewer-owned gold. Missing labels produce
`not_available`, not failure.

| Quality KPI | Example |
| --- | --- |
| Primary evidence coverage | Gold accepts lines 100-110; at least one L0A candidate/window covers that range. |
| Selected primary accuracy | L0A selects line 104, which is inside the accepted gold range. |
| Progress/checkpoint precision and recall | All labeled completed-iteration markers are found and failed checkpoint starts are not counted as saves. |
| Typed-event accuracy | A teardown line is typed `teardown`, not `primary_failure`. |
| Episode/incident construction accuracy | Repeated rank renderings form one episode/fanout incident rather than independent roots. |
| Outcome accuracy | A fault followed by compatible progress is `progressed_after`. |
| Identity accuracy and stability | The expected root/entity are present and identical under rank/timestamp reordering. |
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
- `progress_after_failure=not_observed` only if after-context is available and
  the iteration dialect is recognized.

`DecisionEvidence` selects that episode and its supporting references. L0A
still does not decide whether a restart can recover.
