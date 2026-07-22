# Evidence Bundle Spec

L0 is the production analyzer's deterministic context assembly layer. L0A
builds the complete structured evidence bundle; L0B projects it into the
bounded, attention-efficient view initially supplied to the model. Both are
built before any model call.

This is not a separate product. It is part of the runtime analyzer. The eval
harness measures whether the bundle, optional tools, model profile, and policy
produce acceptable decisions.

## Purpose

L0A and L0B together exist to:

- avoid blindly inlining large logs into prompts while preserving a complete
  structured analyzer record;
- expose likely root-cause candidates without losing raw-line traceability;
- summarize routine training output while giving the model bounded original-log
  excerpts around the most relevant candidate anchors;
- deduplicate repeated symptoms while preserving ordering;
- separate candidate extraction from terminality and final policy;
- preserve progress, checkpoint, recovery, and cascade context;
- make progressive analysis useful by persisting structured candidate summaries
  before raw windows scroll out;
- let optional model tools inspect raw context only when the bundle is
  insufficient.

## Boundary

The L0A bundle and L0B projection MUST be built before L1 model evidence
extraction. They MAY use
deterministic parsers, regexes, stable pattern normalization, profile-declared
signature registries, bounded caches, and caller-provided runtime metadata.

Neither L0 sub-stage may:

- emit `STOP`, `RESTART`, scores, or `decision_basis`;
- compare prior attempts or decide same-root recurrence;
- rely on model-authored fields for ordering, progress, checkpoint, recovery,
  or terminality facts;
- treat an error-only filtered line as terminal evidence without original-log
  context or an explicit terminal marker.

Optional L1 tools are a secondary context-access mechanism. They can read the
retained L0A evidence or raw log context when L0B is ambiguous, but they do not
replace L0 generation.

## Bundle, Model, and Tool Split

LogSage is most useful as inspiration for deterministic bundle construction,
not as a model-facing tool list.

The intended split is:

1. L0A builds the complete structured evidence bundle first. This absorbs
   LogSage lessons
   such as template normalization, pattern clustering/dedupe, candidate error extraction,
   progress-after-fault detection, checkpoint facts, terminality/recovered/app
   done signals, bounded raw excerpts with line numbers, history-ready
   fingerprints, and selection/lossiness accounting.
2. L0B deterministically selects, deduplicates, bounds, and serializes the
   attention-efficient model-facing view. L1 consumes that typed view instead
   of starting from the whole raw log.
3. Optional L1 tools fill gaps. When the bundle is ambiguous, a profile may
   allow generic read-only tools such as `overview`, `grep_log`, and
   `read_window` so the model can inspect raw context around specific clues.

This keeps production and eval comparable: L0A assembly and L0B projection are
deterministic and separately measurable, while tool availability and budget
policy are profile-declared.
Tool calls are evaluated for whether they improve accuracy or expose missing
bundle context, and numeric caps are tuned by eval/profile data rather than
fixed by this bundle spec.

## Typed L0 Boundary

```text
raw log + L0A profile
  -> L0A Complete Evidence Assembly
  -> L0Bundle
  -> DecisionEvidence
  -> L0B Attention-Efficient Projection + L0B profile
  -> L0ModelFacingView
  -> L1
```

`L0Bundle` is the immutable complete structured result over all inspected log
bytes. Complete does not mean a raw copy of every line: bounded windows and
representatives are allowed only when the bundle records the corresponding
counts, caps, omissions, and lossiness.

`DecisionEvidence` is the immutable, versioned deterministic selection built
once from `L0Bundle`. It contains canonical policy-relevant facts and references
to L0A structures; it does not copy bounded windows, rescan the log, add history,
or emit an action. The fallback branch consumes it directly, and L0B includes
the exact same payload in the initial model view.

`L0ModelFacingView` is a versioned projection containing the exact structured
evidence and current-attempt execution context placed in the initial L1 user
message. It is not reconstructed by L1. The retained L0A bundle remains the
source for L0B support, L2 grounding/identity/audit, trace, replay, and optional read-only
tools. Decision Evidence is the current-attempt functional input to fallback
record assembly: L0 derives the shared `AttemptProgressSummary` and deterministic
`AttemptFailureFacts` from it. The runtime stores those in the initial
`AttemptRecord`, and fallback L3/L4 consume that record rather than reconstructing
facts from the bundle.

## Decision Evidence Contract

`DecisionEvidence` MUST contain:

- `deterministic_primary_candidate`: L0's provisional observed failure, or
  `null`; this is evidence, not a fallback recommendation;
- `canonical_observed_identity`: stable identity anchor, deterministic
  fingerprint/source, registry id, class, and signature;
- `selected_evidence_references`: provenance-only source lines and ids of
  selected anchors, windows, episodes, incidents, and normalized occurrence
  groups in L0A. Object ids are model-resolvable only when the route advertises
  `get_evidence_objects`;
- `failure_position`: primary/identity lines, iteration, phase, outcome, causal
  role, data position, and terminal-incident position;
- `progress_checkpoint_state`: completed progress, runtime, checkpoint load/save
  position, replay distance, and progress-after-failure facts;
- `operation_artifact_facts`, `later_progress_recovery`, and `locality`;
- `coverage_lossiness` and provenance proving deterministic construction without
  an LLM or a second log scan.

The high-level algorithm is:

```text
L0A collections
  -> select deterministic primary observation
  -> resolve its stable episode/incident identity anchor
  -> reference related anchors, windows, occurrence groups, episodes, incidents
  -> attach current-log progress/checkpoint/operation/recovery/locality facts
  -> attach coverage, lossiness, and provenance
  -> freeze the versioned DecisionEvidence defined by SCHEMA.md
```

L0A contains the complete structured evidence collections. Decision Evidence
selects their canonical policy-relevant facts and references. L0B combines
model-visible Decision Evidence with selected L0A evidence, bounded and
prioritized for attention efficiency.

L0B avoids repeating Decision Evidence facts in adjacent context sections.
`attempt_execution_context` adds only current-log scope and terminal timing;
progress, checkpoint, operation/artifact, later-progress, and locality facts
remain authoritative in Decision Evidence. This keeps the projection smaller
without removing any model-visible fact used by L2 grounding.

## Required L0A Bundle Content

The bundle MUST include:

- internal file metadata for traceability: normalized path, basename, parent
  path hints, byte size, and line count;
- rank metadata when rank prefixes are present;
- bounded path-access facts for configured read, write, and cache paths and for
  failed accesses, including operation intent and path namespace when
  parseable. Namespace differences are observational and do not prove the
  effective user, owner, mode, or ACL;
- node and GPU metadata only when explicit fields or validated rank-to-GPU
  mapping are available;
- normalized occurrence groups for candidate error, progress, checkpoint,
  scheduler, and cascade lines;
- bounded original-log excerpt windows around top candidate anchors, with
  original line numbers and truncation metadata;
- candidate anchors/event timeline entries for representative high-signal
  lines, first-fault/root candidates, and downstream registry or cascade
  matches;
- generic structural failure anchors for traceback terminal exceptions,
  assertion occurrences, fatal records, and process termination without
  assigning semantic cause or policy;
- bounded explicit cause-confirmation records from scheduler, kernel, or
  runtime streams embedded in the analyzed log. A bare `Killed` record proves
  termination only; a later explicit confirmation such as an OOM-kill report
  may establish its cause. Repeated confirmations MUST retain aggregate counts
  while bounding excerpts and representatives;
- diagnostic line roles for common CUDA/PyTorch debugging advice. Diagnostic
  lines remain in raw excerpts but are excluded from failure-anchor and root-
  fingerprint selection;
- first-fault candidates sorted by original line/time;
- internal registry matches from `TAXONOMY.md`, including line numbers,
  signature id, class, policy/role, fingerprint, and candidate outcome when
  known. Exact aggregate occurrence count and locality spread MUST be computed
  before repeated rank-equivalent matches are compacted. The stored bundle may
  retain bounded first/last representatives per normalized pattern and MUST
  record retained/dropped counts and the cap hit. The model-facing view MUST
  expose deduplicated provisional registry groups without authoritative policy
  or root-role labels. Rank fanout for one distributed process group, sequence,
  and operation is one observed event and MUST NOT be described as independent
  recurrence or persistence evidence;
- application progress markers such as iteration, step, epoch, or configured
  monotonic workload markers, with marker type, value, line, and completion
  state when parseable;
- checkpoint markers, including completed checkpoint saves and checkpoint step
  when parseable; started, partial, failed, or ambiguous checkpoint writes MUST
  be represented separately from completed checkpoint progress;
- operation/artifact comparisons that group observations by operation, logical
  artifact, and physical unit/shard when available; record prior completed
  values and source lines; declare comparison strength; and describe the latest
  attempt as completed, started-not-completed, failed, or unresolved. The
  associated terminal incident line SHOULD be retained when available;
- setup markers for attempted checkpoint load, checkpoint metadata load,
  resharding, and optimizer initialization when explicit. A checkpoint-load
  start records the selected checkpoint iteration and line but MUST NOT count
  as completed checkpoint progress;
- recovery markers after candidate faults, such as retry success, later path
  access, resumed iteration, or checkpoint completion;
- workload recovery hints such as bad-token/token-window retry, skip, or
  quarantine status when visible;
- scheduler and external-policy markers such as time limit, preemption, or
  cancellation;
- progress-centered failure episodes that bind recent compatible progress,
  observed precursor candidates, traceback start, terminal exception line,
  conservative cleanup/teardown hint, cancellation/abort/downstream fallout,
  and first compatible progress after the episode when present. When a timed
  terminal operation reports its configured timeout, an earlier high-signal
  event whose timestamp aligns with the operation start MAY be recorded as a
  precursor and stable episode identity anchor. The timing relation is an
  observed correlation, not proof of root cause. A nearby high-signal error
  immediately before a traceback or generic wrapper, with no intervening
  compatible progress, MAY likewise become the episode identity anchor. The
  complete chain remains attached to the episode;
- distributed-mechanism incidents for terminal timeout waves, including a wave
  observed by only one reporter. They retain exact event count, bounded
  representative lines, operation/process-group diversity, and locality, and
  collapse later rank/process-group fanout. The earliest observed terminal
  timeout is the failure mechanism; later reports in the same timeout epoch are
  fanout, not independent failures or recurrence;
- distributed exception-fanout incidents that collapse the same normalized
  terminal exception emitted by multiple ranks in one attempt. These retain
  bounded member lines and rank counts but MUST NOT be interpreted as
  cross-cycle persistence;
- deterministic terminal timing facts when parseable: last progress line and
  timestamp, configured timeout, first detection timestamp, elapsed seconds
  from progress to detection, and detection lag beyond the configured timeout;
- progress segments that group nearby candidate faults and mark status as
  `terminal`, `recovered`, `progressed_after`, or `unresolved`;
- per-episode post-fault summaries containing later progress, same-normalized-
  exception recurrence, later high-signal extent, teardown/cancellation
  anchors, and the first downstream cascade. These are observations, not
  persistence or policy conclusions;
- cascade groups for repeated downstream symptoms after a prior candidate;
- root fingerprint candidates with volatile tokens stripped;
- cascade fingerprints for repeated downstream symptoms;
- selection/lossiness metadata for filters, dedupe, sampling, truncation,
  context windows, progressive eviction, and configured caps;
- progressive state metadata when running in progressive mode.

`terminal` in an L0 episode means that the observed episode ended without later
compatible progress and may include teardown/cascade evidence. It does not mean
the policy action is `STOP`. If no prior progress marker was observed, L0 MUST
say that explicitly rather than describing the exception as occurring after
prior progress.

Repeated rank renderings of the same timed operation (`SeqNum` and `OpType`),
different process-group operations detected in the same terminal timeout wave,
serialized duplicate exceptions, and teardown wrappers belong to one failure
episode when no compatible progress separates them. They remain traceable as
duplicate, fanout, or wrapper lines, but MUST NOT consume independent
high-signal prompt slots or count as recurrence/persistence evidence.
An exception emitted from an observed finalizer, cleanup, shutdown, or process-
group teardown stack is classified structurally as a teardown cascade. Its
temporary pathname or exception type alone does not establish that role.
Repeated structurally classified teardown cascades are grouped by their stable
L0 identity before prompt rendering. Volatile path variants remain in bounded
representative quotes and aggregate counts, but do not create separate patterns
or failure episodes.

Comparable-operation history is current-log execution evidence, not L3
cross-cycle history. It answers questions such as whether earlier checkpoint
saves to the same artifact completed before the latest save attempt failed.
It does not prove that a particular inner write, append, flush, or metadata
update completed, and prior success supports retry plausibility without proving
that the latest failure is transient. The MVP checkpoint-save extractor is one
producer of this generic structure; additional operation parsers may be added
only when they preserve the same observational contract.

For eval or multi-model replay, a completed L0A bundle MAY be serialized once
and reused only while source path, byte size, modification identity, and bundle
schema still match. The L0B view MUST also be constructed once per declared
projection profile. Every model in one comparison MUST receive the same
byte-identical L0B view. Reuse changes execution cost, not semantics.

The L0B view MUST redact path-derived source metadata before
L1 sees it. Absolute paths, basenames, parent directories, eval case
directories, observed labels, and path/run-name hints may encode the expected
answer and must remain trace-only/debug metadata.

## Line Numbering

All bundle line fields are 1-based logical log lines produced by the analyzer's
text reader after universal-newline splitting on LF, CR, or CRLF. This is the
line-number contract for candidate anchors, excerpts, progress markers, failure
episodes, and citations.

The bundle SHOULD include `anomalies.line_numbering` metadata that names the
scheme and warns when shell tools may display different line numbers. This can
happen when logs contain embedded carriage returns or rank prefixes such as
`3260:` that look like line numbers but are part of the log text.

## LogSage Lessons To Absorb

The bundle should absorb these LogSage lessons as deterministic or
client-derived analyzer behavior:

- normalize noisy log lines into stable patterns before expensive analysis;
- group repeated patterns and keep representative raw lines;
- treat exact normalized-template grouping as the MVP default; Drain, MinHash, or
  other near-duplicate algorithms are optional extensions only if eval shows
  exact templates fragment too much;
- separate candidate extraction from final restart policy;
- track progress and checkpoint facts because faults can recover or training can
  continue after them;
- send bounded original-log excerpts around the top candidate anchors, including
  before/after context, so recovered/progressed-after faults do not become false
  roots;
- expose representative high-signal anchors even when no taxonomy row matched
  the exact line, including late terminal-looking exceptions that appear after
  earlier noisy/recovered warnings, and connect each anchor to nearby progress
  and downstream matches;
- record terminality separately from the existence of an error string;
- preserve candidate summaries during streaming/progressive analysis;
- persist history-ready fingerprints rather than raw prior logs.

Prompt-only policy lessons from LogSage should be converted into registry rows,
policy rules, or eval cases before they can affect production decisions.

## Normalized Occurrence Grouping And Excerpts

The bundle MUST use deterministic normalization to collapse repeated rank/node
noise before model calls. The MVP grouping algorithm is exact match on normalized
templates that strip volatile paths, numbers, hex values, quoted blobs, long
opaque tokens, timestamps, PIDs, ranks, and similar fields while preserving
meaningful error words and operation names.

`NormalizedOccurrenceGroup` is the preferred name for the runtime aggregate of
lines sharing one normalized text shape. It is not a registry pattern. A group
may reference the registry detector that produced its observations, or it may
come from deterministic progress, checkpoint, recovery, lifecycle, or
diagnostic detection with no signature-registry id. Registry identity remains
part of the grouping key, so overlapping generic and specific matches may
produce separate groups over the same source lines.

An occurrence group records its id, normalized shape, first line, count,
bounded sample lines, locality spread, optional registry id, and deterministic
classification. It does not prove that its members belong to one causal
incident, one failure episode, or cross-cycle recurrence. Identical shapes may
appear at unrelated times; distributed fanout requires separate incident
evidence.

Normalized occurrence groups summarize ordinary or repeated log output. They
are not enough for root-cause reasoning by themselves. The model-facing bundle SHOULD include
bounded original-log excerpts for the highest-value candidate windows, selected
by first-fault candidates, primary/root candidates, downstream cascade anchors,
and progress/checkpoint anchors near faults. Overlapping candidate windows SHOULD
be merged, preserving original order and line numbers.

Candidate anchors are the compact event timeline that explains why an excerpt
or candidate was selected. They SHOULD include representative high-signal lines
even when the line is not a registry/taxonomy match. Sampling MUST NOT be limited
to the first few high-signal lines in the file; it should cover early signals,
terminal-looking exception/traceback lines, and late high-signal endpoints so
earlier noisy warnings cannot starve the decisive failure window. Structured
logging prefixes MUST be interpreted by explicit severity and message payload;
logger or module names such as `hypercorn.error` MUST NOT become failure
evidence when the explicit severity is informational. For each
anchor, the bundle SHOULD record:

- original line number, quote, and source tags such as `high_signal`,
  `terminal_exception`, `cause_confirmation`, `registry_candidate`, or
  `registry_selection`;
- whether the anchor has a provisional taxonomy/registry hint;
- a conservative `causal_role_hint`, which may identify cleanup/teardown stack
  context but MUST otherwise remain `unknown` rather than asserting a root;
- nearby observed application progress markers when present, labeled as
  ordering context rather than recovery/progressed-after proof;
- rank relation between the anchor and any later observed progress marker when
  ranks are available;
- first downstream registry match and first downstream cascade when present;
- the context window ids that cover the anchor;
- whether the prompt-facing excerpt included the anchor line.

A candidate anchor is a location proposed for inspection, not a root-cause
conclusion. For example, if line 1012 contains the first terminal-looking
exception after progress at line 1000, L0 may emit an anchor at 1012 with
`sources=[terminal_exception, high_signal]`, a prior-progress reference to
1000, a downstream teardown reference to 1030, and a context-window id. The
anchor says "inspect here and preserve these ordering relationships"; it does
not say that line 1012 is necessarily the root.

A bounded context window is the original-log evidence around one or more seed
anchors. For the anchor above, a profile may retain lines 972-1152, record
`seed_lines=[1012]` and `selected_by=terminal_exception`, and preserve original
line numbers, raw text, occurrence-group ids, and truncation metadata. The
window gives L1 coherent before/after context without making the entire log the
initial prompt. L0B may narrow or merge L0A windows for attention efficiency;
read-only tools may expand beyond them when the initial view is insufficient.

Structural extraction SHOULD recognize exception syntax generically rather
than enumerate error messages. For example, `RuntimeError`,
`UnicodeDecodeError`, `torch.AcceleratorError`, and framework-specific
`*Exception` lines use the same `observed_exception` anchor role. The complete
message and traceback are preserved for L1; text such as `index out of bounds`
is evidence content, not an L0 policy taxonomy row.

Terminal failure episodes are not limited to Python exceptions. A structurally
terminal operation timeout, fatal record, assertion, or equivalent observed
failure may anchor an episode when it is followed by termination/cancellation
and no compatible progress. Rank-rendered copies of the same timed operation
and different operations detected in one watchdog timeout epoch form one
episode and one distributed incident; their exact count and locality spread
remain aggregate facts.

A process-group watchdog line that explicitly says its thread terminated with a
CUDA/NCCL exception is also a terminal episode anchor. So is a CUDA runtime
status such as `cudaError*:` when it appears in a terminal error record. Repeated
`what()` or wrapper renderings of the same normalized exception payload belong
to one episode. Concatenated records are parsed by semantic markers rather than
assuming the fatal-process text starts a new line. For peer-GPU memory access
failures, the episode records the observed mechanism without asserting whether
the cause was fabric, remote GPU, or invalid client-selected peer memory.

A bare shell or launcher `Killed` record is an abrupt-process-termination
anchor, not an OOM classification. L0 scans the terminal remainder for explicit
cause confirmations and associates bounded representatives with the most
specific preceding termination episode when no compatible progress intervenes.
This relation keeps later scheduler output available to L1 without inventing a
cause when that output is absent. The episode timeline remains anchored at the
observed termination, while an explicit confirmation becomes the stable
identity anchor so equivalent model line choices converge for history.

The following stable CUDA/PyTorch boilerplate roles are diagnostic context:
asynchronous error-reporting/possibly incorrect stacktrace warnings,
`CUDA_LAUNCH_BLOCKING` suggestions, and `TORCH_USE_CUDA_DSA` compilation
suggestions. These lines MUST NOT become primary candidates. A diagnostic
suggestion does not prove that its named condition occurred.

Noisy warning treatment is progress-centered. Filesystem warnings, network or
port-flap warnings, transient NCCL warnings, and repeated path-access warnings
that are followed by compatible application iteration, checkpoint completion, or
explicit recovery should remain compressed occurrence-group context. The bundle may
report their normalized pattern, first line, count, sample lines, rank/node
spread, and `progressed_after` or `recovered` status, but it should not spend
top candidate-anchor or excerpt budget on every occurrence. Such warnings become
top anchors only when they are temporally tied to the terminal episode, are the
first unrecovered high-signal root candidate, or no later stronger terminal
candidate exists.

Each excerpt window SHOULD include:

- selected seed line numbers and why the window was selected;
- raw lines before and after the candidate, up to profile-configured line/byte
  caps;
- nearby observed progress or checkpoint markers when present;
- first downstream candidate or cascade line when present;
- source line count, included line count, and truncation/lossiness flags.

Optional tools are for missing or ambiguous context after this deterministic
assembly. A routine `read_window` call around a top candidate should be treated as
an eval signal that the bundle excerpt was too thin.

Registry repetition MUST be represented as one provisional group with count,
first line, sample lines, and representative quote. Repeated identical matches
must not consume additional candidate-anchor or context-window budget merely
because many ranks emitted the same downstream or cleanup symptom.

## L0 Review KPIs

L0 has separate quality KPIs and operational metrics. The eval bench MUST keep
them separate instead of treating small bundles, short latency, or zero tool
calls as proof of evidence quality.

### L0A Complete Evidence Assembly

Quality KPIs require reviewer-owned gold facts. When gold is absent or only
partially labeled, the corresponding KPI is `not_available`; it is not a
failure. The L0A quality scorecard includes:

- `primary_evidence_coverage`: whether any accepted gold primary line or window
  appears in the L0A candidate evidence set;
- `selected_primary_accuracy`: whether the deterministic primary singleton is
  within an accepted gold line or tolerance;
- progress and checkpoint detection precision, recall, and F1 when markers are
  exhaustively labeled, or recall only when labels are partial;
- typed-event precision, recall, and F1 by event type when that type is
  exhaustively labeled;
- failure-episode and distributed-incident construction accuracy;
- later-progress/recovery fact accuracy; and
- coverage and lossiness reporting correctness.

L0A operational metrics include:

- `l0a_wall_clock_s`, source lines/bytes, and scan throughput;
- complete-bundle serialized size and whether a prebuilt bundle was reused;
- window, anchor, occurrence-group, episode, and incident counts;
- caps, truncation, and lossiness counts; and
- deterministic replay/hash consistency.

### Decision Evidence Selection

- `decision_evidence_wall_clock_s` and schema version;
- deterministic-primary and canonical-identity availability;
- referenced source-line and L0A-object counts;
- stable-fingerprint availability and exact payload equality between the trace,
  fallback branch, and L0B request.

### L0B Initial Model Evidence View

L0B quality KPIs require gold or controlled corpus comparisons:

- required gold line/reference coverage in the initial model view;
- deterministic-primary retention, conditional on the primary being available
  in L0A;
- relevant progress, checkpoint, and supporting-context coverage;
- compaction safety: required evidence is not silently omitted; and
- first-turn completion, tool-call rate, decision-relevant context found only
  through tools, and bundled evidence reread through tools across models and
  controlled profile ablations.

The downstream model signals are attribution evidence, not automatic L0B
failures. A tool call that finds missing decision-relevant context suggests an
L0B gap. A call that rereads the initial view suggests model/profile tool-use
inefficiency.

L0B operational metrics include:

- `l0b_wall_clock_s` and model-view schema version;
- `view_size`: compact JSON characters, estimated tokens, and estimation ratio;
- `budget_utilization`: used, limit, and utilization for each declared section
  or excerpt limit;
- `selection_counts`: available, selected, omitted, and configured limit for
  each projected evidence collection;
- `compaction_counts`: merged/omitted windows, source/model-facing lines, and
  explicit truncation counts; and
- `projection_integrity`: payload serialization, resolvable references, valid
  line ranges, selection accounting, limit compliance, lossiness accounting,
  and deterministic payload hash.

Aggregate `l0_wall_clock_s` is the sum of L0A, Decision Evidence selection, and
L0B wall time for compatibility
and end-to-end accounting. A tool call that finds new decision-relevant evidence
may identify an L0B omission. A call that only rereads L0B evidence is an L1
model/tool-efficiency finding, not an L0A coverage defect.

For loose review logs without a human-approved label, analyzer-selected-primary
visibility is an operational diagnostic, not a quality score. Scored eval cases
SHOULD add gold primary, progress, checkpoint, and required-view evidence once a
reviewer approves them.

Tool calls are optimization signals, not automatic failures. If a tool call
adds new relevant line context, L0 likely missed useful context. If it only
re-reads lines already in the prompt excerpts, the issue is more likely prompt
wording or model tool-use behavior. If every model asks for the same missing
context, treat that as a high-priority bundle coverage gap.

## Job And Run Metadata

The bundle SHOULD expose compact job-shape and run-progress metadata when it can
be parsed deterministically. This metadata helps L1 understand scale and
progress depth without deriving those facts from scattered raw lines.

Job metadata SHOULD include:

- explicit `world_size` and source line when a config/log line provides it;
- observed rank minimum, maximum, and distinct count;
- `inferred_world_size_lower_bound = max(observed_rank) + 1` when ranks are
  seen;
- `world_size_source`, distinguishing `explicit`,
  `observed_rank_lower_bound`, and `not_found`;
- observed node count when nodes are explicit;
- whether rank-to-GPU mapping is available.

An explicit `world_size` is authoritative for job scale. Rank-derived world
size is only a lower bound and MUST be labeled as such. Same-rank observations
MUST NOT imply same GPU unless a validated rank-to-GPU mapping is present.

Run progress summary SHOULD include first/last parsed iteration, iteration
delta, total iterations when present, first/last consumed samples, consumed
sample delta, successful-runtime duration when timestamps are parseable,
progress/checkpoint marker counts, last checkpoint iteration, iterations since
that checkpoint, selected checkpoint-load iteration/line when a load started,
and whether any failure episode had compatible progress after it. These are
current-attempt summary facts over marker records; the marker records and raw
line citations remain the auditable evidence.

`iterations_since_checkpoint` is failure proximity to the latest resumable
state. It estimates how much work a restart may replay and whether a
deterministic sample, token window, or phase-boundary failure may be encountered
again quickly. It does not classify the failure or prove that retry will or will
not recover.

## Progress Detection

Progress is a current-attempt fact extracted into the bundle before policy
evaluation. `POLICY.md` owns how progress affects STOP/RESTART and history; this
section owns how the bundle identifies progress from the current log.

The bundle MUST separate these concepts:

- `application_progress`: a monotonic workload marker advances. Examples are
  completed or observed training iteration, train step, global step, epoch, or
  profile-configured monotonic counters such as consumed samples/tokens when the
  workload logs them reliably.
- `checkpoint_progress`: a checkpoint save completed successfully, with a
  checkpoint step/iteration when parseable.
- `setup_progress`: a bounded, deduplicated initialization milestone such as a
  checkpoint-load start, checkpoint metadata load, resharding observation,
  optimizer setup, successfully loaded checkpoint, or completed CUDA-graph
  build. Each marker carries `started`, `observed`, or `completed` state. Setup
  markers explain how far initialization got, but are not forward training
  progress, checkpoint-save progress, post-fault recovery, or history
  advancement.
- `recovery_evidence`: a fault-specific operation later succeeds, retries
  successfully, path access succeeds, or the workload explicitly skips or
  quarantines a bad token/sample/window. This marks a candidate as `recovered`
  unless application or checkpoint progress is also observed.
- `latest_observed_failure_iteration`: an iteration explicitly attached to a
  terminal failure line. It locates the failure and may establish phase or
  checkpoint-to-failure distance, but it is not a completed progress marker.

L0 SHOULD also summarize `later_progress_after_fault_observations`: normalized
fault-like observed events for which later compatible job progress exists in the
current log. This
is direct evidence that the job continued after an observed disturbance. It is
not a generic rule that infrastructure failures recover, and in an interleaved
distributed log it does not prove that the same rank, node, network path, or
component recovered. Each summary MUST therefore retain event and later-progress
lines, state `ordering_basis=log_order`, describe the interpretation as
`job_progress_observed_after_event`, and set `component_recovery_proven=false`
unless a future source supplies explicit component-local recovery evidence.

The following MUST NOT be treated as progress by themselves:

- `cycle_id` or NVRx restart count;
- repeated log emission, heartbeats, throughput-only lines, process liveness,
  rendezvous/teardown chatter, or unchanged metric values;
- scheduler allocation, node state, or service polling;
- checkpoint started, checkpoint in progress, partial checkpoint, failed
  checkpoint, or ambiguous checkpoint messages;
- retry counters, rank IDs, worker IDs, PIDs, timestamps, durations, or memory
  counters;
- a lower, equal, or marker-incompatible progress value.

Progress marker records SHOULD include:

- `marker_type`: for example `iteration`, `step`, `global_step`, `epoch`,
  `checkpoint_step`, or a profile-configured monotonic marker name;
- `value`: parsed integer/string value;
- `state`: `completed`, `observed`, `started`, `failed`, or `ambiguous`;
- `line`, timestamp when available, raw quote, source pattern id, and parse
  confidence/source;
- rank, node, and GPU only when explicit or validated from runtime metadata.

Only marker-compatible values are comparable. For L3 history comparison,
`POLICY.md` compares completed application steps and successful checkpoint-save
steps first. It uses an explicit observed failure iteration only when neither
positive-progress dimension is comparable. An observed failure iteration is
comparable position, not evidence that the iteration completed. For
candidate outcome inside one attempt, a later greater compatible application
marker after a fault is evidence for `progressed_after`; a later completed
checkpoint is also evidence for `progressed_after`. A later retry/path
success/skip/quarantine without monotonic advancement is recovery evidence, not
necessarily application progress.

Bundle construction MUST identify progress deterministically:

1. Classify candidate lines into normalized occurrence groups including `progress`,
   `checkpoint`, `setup_progress`, `recovery`, `error`, `scheduler`, and
   `cascade`.
2. Parse marker records from progress/checkpoint/recovery groups using
   deterministic registry patterns before any model call.
3. Normalize marker names and values, strip volatile fields, and keep original
   line numbers and representative quotes.
4. Track the highest comparable application marker and highest completed
   checkpoint marker observed so far, plus any iteration explicitly attached
   to the terminal failure.
5. For every candidate fault, inspect later original-log context and persisted
   progressive summaries for greater compatible progress, completed checkpoint,
   or recovery evidence.
6. Mark candidate outcome as `progressed_after`, `recovered`, `terminal`, or
   `unresolved` using the rules below.
7. Record coverage and lossiness separately for `application_progress`,
   `checkpoint_progress`, and `progress_segments`.

## Failure Episodes

Failure episodes are the preferred MVP structure for giving L1 useful context
when error strings vary. They are built around progress boundaries, not around a
specific taxonomy match.

For each terminal-looking exception after recent progress, the bundle SHOULD
emit a failure episode with:

- `last_progress_before`: the nearest prior compatible application or completed
  checkpoint marker;
- `start_line`: the traceback or first terminal-looking line that starts the
  episode;
- `terminal_exception_line`: the summary exception line when present;
- `exception_chain_lines`: causal exception summaries that belong to the same
  serialized exception chain;
- `duplicate_rendering_lines`: repeated renderings of the same exception in
  that chain;
- `wrapper_exception_lines`: outer exceptions emitted while handling an inner
  exception;
- parsed locality and iteration fields from the terminal exception when
  explicit;
- `first_teardown_line`: first cleanup/teardown marker such as
  `destroy_process_group`, program-exit cleanup, or equivalent;
- `first_process_termination_line`: first explicit process termination marker
  when visible;
- `first_scheduler_cancel_line`: first scheduler/job cancellation marker such
  as `CANCELLED AT` when visible;
- first downstream cascade/cleanup marker when visible;
- bounded `cause_confirmations` carrying the original line, quote, and
  confirmation kind when an explicit later scheduler/kernel/runtime record
  explains the termination;
- `first_progress_after`: the first later compatible application or checkpoint
  marker, or `null`;
- `status`: `terminal`, `progressed_after`, `recovered`, or `unresolved`;
- context window ids that cover the episode.

Progress after the episode is the strongest current-log recovery signal. If
compatible progress advances after an exception, the episode should normally be
`progressed_after` and treated as context. If the episode has recent progress
before, a terminal-looking exception, cancellation/downstream fallout, and no
later compatible progress, it should normally be `terminal` even when the
specific error string has no taxonomy row.

Because distributed logs are buffered and interleaved, later progress-like text
MUST be validated as compatible progress before it can change episode status.
Rank-local ordering observations may be preserved as metadata, but a later line
from another rank is not by itself proof of recovery unless the marker advances
a comparable workload counter.

Serialized duplicate, inner-cause, and outer-wrapper exception renderings SHOULD
form one failure episode rather than independent roots. L0 SHOULD select the
most causal observed mechanism as the canonical identity anchor while retaining
every chain line for audit. This is normally the terminal exception. A
high-signal event whose timestamp aligns with the start of a configured timed
operation MAY be retained as an earlier precursor and identity anchor. A nearby
high-signal error immediately preceding a traceback or generic wrapper MAY also
be retained when no compatible progress intervenes. A model may cite any
credible member of the episode, but history identity MUST remain independent of
that line choice. When a distributed wrapper fans out after the anchor, its
event count, bounded sample lines, and locality spread remain cascade metadata.
Conditional diagnostic suffixes remain in the raw quote and model evidence, but
MUST be excluded from the client-observed message signature used for history
identity. This prevents phrases such as "it is possible" or "may be caused by"
from turning an unconfirmed explanation into a fingerprint field.

### Distributed Failure Incidents

A failure episode explains local causal ordering and may contain an ordinary
failure reported by one rank. A distributed failure incident is an additional,
parallel representation with an explicit `incident_kind`:

- `distributed_mechanism` records an inherently distributed mechanism, such as
  a collective timeout. It may have one observed event or reporter; that does
  not imply multi-rank corroboration.
- `distributed_fanout` groups the same ordinary failure mechanism observed from
  at least two distinct ranks in the same attempt and progress segment.

A single-rank ordinary exception therefore produces an episode but no
distributed incident. A single-observer collective timeout may produce both an
episode and a `distributed_mechanism` incident. At scale, later reports from
many ranks and process groups are folded into that same mechanism incident when
they share a configured timeout epoch and no compatible progress separates
them.

For a distributed collective-timeout incident, L0 MUST retain:

- `incident_kind=distributed_mechanism`;
- earliest observed terminal line and quote;
- all member timeout line numbers for exact grounding, plus bounded sample
  lines for prompt use;
- exact event count, unique operation count, operation types/signatures,
  bounded rank samples, observed-rank count, and process-group types when
  explicit;
- status, phase, configured timeout, last progress, first detection time,
  elapsed time from progress to detection, and detection lag;
- `root_cause_status=unknown` unless separate direct cause-confirmation evidence
  exists.

The incident history fingerprint is client-derived from the observed incident
kind and validated phase. It MUST exclude rank, node, GPU, process-group id,
sequence number, operation type, tensor size, timestamp, and which report
appeared first. Those fields remain diagnostic detail. Reordering ranks or
collectives in a later cycle must not split an otherwise equivalent history
incident. A matching incident fingerprint is still only a recurrence input;
it does not establish the underlying cause by itself.

In progressive mode, each poll SHOULD update progress marker records and the
current maxima. If raw lines age out, the service MUST retain structured marker
records, source line numbers, offsets, source pattern ids, and lossiness
metadata. A retained marker record can prove progress; it cannot replace a raw
line quote when external output cites evidence.

## Candidate Outcome

When evidence is available, the analyzer MUST classify candidate outcome as:

- `terminal`: no later recovery/progress evidence and the candidate is tied to
  job termination;
- `recovered`: later log evidence shows the same operation retried or succeeded;
- `progressed_after`: later application iteration, step, or checkpoint progress
  is observed after the candidate;
- `unresolved`: no clear terminal or recovery/progress evidence.

Only `terminal` or explicitly terminal-linked `unresolved` candidates can become
the deterministic primary represented in `AttemptFailureFacts` and supplied to
L3/L4. This eligibility does not itself select `STOP`; L4 still requires qualified current-attempt semantics or an
exhausted same-root retry budget. Recovered and progressed-after faults remain
context.

## Progressive Behavior

The future progressive path incrementally fills the same L0A bundle and records
eviction as lossiness; it does not define another bundle type. Raw-window
eviction must preserve structured candidate summaries but must never fabricate a
raw quote. `PROGRESSIVE.md` owns the lifecycle and retention contract.

## Interaction With L1 Tools

The initial L1 model message receives the L0B model-facing view plus the
declared semantic schema and tool list. It does not receive the full L0A bundle
or raw attempt history. Tools may expose bounded source context backed by L0A.

The MVP production tool set is `overview`, `grep_log`, and `read_window`:

- `overview` is a read-only view over file metadata and bundle orientation;
- `grep_log` searches the original log and returns line-numbered matches;
- `read_window` returns original raw lines around selected evidence.

When raw lines requested by `read_window` are unavailable because of progressive
window eviction, file truncation, overwrite, or configured caps, the tool MUST
return a deterministic unavailable/truncated result with any relevant retained
candidate-summary references. It MUST NOT crash, loop indefinitely, or invent
raw log lines.

Tool results can support model evidence extraction, but policy-critical facts
must be client-grounded against the bundle or original tool-returned lines.

## Trace And Eval

The decision trace MUST preserve the L0A bundle, exact Decision Evidence, and
exact L0B view, or
lossless references to them. Every cited evidence line in external output MUST
be traceable to L0A, L0B, or a read-only tool result.

When an L1 model is called, the trace MUST include the exact versioned L0B
snapshot supplied to the model or a lossless artifact reference plus hash. This
lets operators compare what L0A retained, what Decision Evidence selected,
what L0B selected, what the model saw,
what tools it requested, and how the final policy used or rejected the model
evidence.

Eval should measure:

- L0A coverage of labeled key evidence;
- deterministic Decision Evidence identity/reference completeness;
- L0B initial visibility of labeled key evidence;
- candidate ordering correctness;
- progress/checkpoint/recovery correctness;
- context-window coverage and truncation;
- selection/lossiness accounting completeness;
- progressive candidate-summary retention;
- cases where optional tools found necessary context missing from the bundle.
