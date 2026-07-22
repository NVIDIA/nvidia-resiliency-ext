# Restart Log Analyzer Stage Inputs And Outputs

Status: non-normative presentation-review document. Do not implement from this
file or treat its examples, versions, limits, or defaults as contracts. Its
stage, registry, evidence, prompt, and KPI examples support the current deck.

Canonical product contracts remain in:

- [DESIGN.md](../../../../docs/design/attribution/restart_agent/DESIGN.md)
- [EVIDENCE_BUNDLE.md](../../../../docs/design/attribution/restart_agent/EVIDENCE_BUNDLE.md)
- [SCHEMA.md](../../../../docs/design/attribution/restart_agent/SCHEMA.md)
- [POLICY.md](../../../../docs/design/attribution/restart_agent/POLICY.md)
- [PATTERN_REGISTRY.md](../../../../docs/design/attribution/restart_agent/PATTERN_REGISTRY.md)
- [eval REQUIREMENTS.md](../REQUIREMENTS.md)

## Purpose

This document makes every runtime handoff explicit for design, development, and
QA review. Each stage is described by its functional inputs, work, functional
outputs, telemetry, degraded behavior, and ownership boundary.

The product and eval harness are separate systems. L0-L4 are runtime product
stages. The harness observes and scores their artifacts offline; it is not an
additional runtime stage.

## End-To-End View

```text
Runtime request + analysis profile
  log_path, job_id, cycle_id, optional attempt_history
                         |
                         v
L0A Complete Evidence Assembly -> immutable complete structured bundle
                         |
                         v
                 Decision Evidence
            compact canonical decision inputs
                         |
             +-----------+-----------------------------+
             |                                         |
             v                                         v
Fallback L3 -> L4                  L0B Initial Model Evidence View
             |                          + selected L0A context
             |                                         |
             |                                         v
             |                              initial model evidence
             |                                         |
             |                                         v
             |                              L1 Semantic Analysis
             |                                         |
             |                                         v
             |                    L2 Grounding, Identity, And Audit
             |                                         |
             |                                         v
             |                                   L3 History
             |                                         |
             |                                         v
             |                                   L4 Policy
             |                                         |
             +----------------------+------------------+
                                    v
                          Best available candidate
                                    |
                                    v
 retry rule + budget state + STOP/RESTART + provenance + trace
```

L0 has two deterministic sub-stages. L0A creates the complete structured
evidence bundle. It then derives `Decision Evidence`, the compact canonical
facts needed by history and policy. L0B combines a model-visible projection of
that decision evidence with selected supporting evidence from L0A to produce
the bounded, attention-efficient initial view presented to the model.

Decision Evidence is a shared typed handoff artifact, not a third numbered
stage. Its derivation is deterministic L0 work and should be negligible beside
the L0A scan; it exists to prevent the fallback and model branches from
independently selecting the same policy-relevant facts.

The fallback and enriched branches therefore share the same deterministic
facts. As soon as Decision Evidence is frozen, fallback L3/L4 starts without
waiting for L0B or L1. It does not rescan or independently analyze the log. The
enriched branch starts from the same Decision Evidence and proceeds through
L0B/L1/L2 before its own L3/L4 evaluation. The earlier fallback remains usable
if L1 is late or unavailable. L0B is the model's initial evidence, not its
evidence ceiling: optional L1 tools may retrieve additional bounded evidence
from retained L0A state or the source log.

Stateful orchestration sits outside these stages. `RestartAgentRuntime` selects
one immutable in-memory prior-attempt view before either branch starts, then upserts
the selected current-attempt record after analysis. Configuration parsing and
dependency construction remain outside the runtime. Library/unit tests seed,
inspect, and clear that same in-memory store. The CLI may explicitly import and
export deterministic JSON-array fixtures for manual scenario construction, but
it does not maintain history automatically. MCP is a later thin adapter over
the runtime.

### Handoff Summary

| Boundary | Functional input | Functional output | Next consumer |
| --- | --- | --- | --- |
| L0A Complete Evidence Assembly | Current-cycle log plus deterministic registries and bundle profile | Immutable complete structured evidence bundle | Decision Evidence, L0B, L2 grounding/identity/audit, trace, and tools |
| Decision Evidence shared artifact | Canonical L0A facts required by downstream decision logic | Compact deterministic identity, failure, progress, recovery, and placement facts | Fallback L3/L4 and model-visible L0B projection |
| L0B Initial Model Evidence View | Model-visible Decision Evidence, selected L0A support, and L0B evidence-selection rules and size limits | Bounded initial model evidence payload | L1 initial request |
| L1 Semantic Analysis | L0B view, prompt/schema, model profile, and optional tools backed by L0A/raw logs | Raw semantic assessment; no action | L2 |
| L2 Grounding, Identity, And Audit | Raw L1 output, L0A bundle, L0B view, tool evidence, source log | Minimally grounded `CurrentFailureFacts`, mechanical reference repairs, and separate audit findings/unapplied semantic suggestions | L3 current facts and observability; L4 receives unchanged L1 semantics |
| L3 History Enrichment | Client-derived identity, current progress, job/cycle identity, immutable runtime-selected prior attempts | Ordered exact-root comparisons, typed progress relations/deltas, exact-position and streak counts | L4 |
| L4 Policy Decision | Unchanged L1 assessment or fallback facts, observational L3 history, policy profile | Selected retry rule, allowed/count/exhausted state, decision basis, action, and provenance | Candidate selector/NVRx |
| Candidate selector | Fallback and enriched candidates plus readiness time | Best candidate available before the caller deadline | NVRx |
| Eval harness | Product result/trace plus optional human gold | Reviews, comparisons, scores, and profile recommendations | Developer/QA review only |

## Terminology: Registries, Evidence Structures, And Profiles

### What Is A Registry?

A registry is a versioned set of generic, deterministic detectors used by L0
to turn raw log lines into structured observations. A registry entry can
describe:

- what text shape to recognize;
- the stable detector id;
- the line's structural role, such as root candidate, cascade candidate,
  cause confirmation, diagnostic context, progress, or checkpoint;
- an initial ambiguity/class hint or known recovery behavior.

The registry does not establish root cause and does not choose `STOP` or
`RESTART`. Its output is candidate evidence for the bundle. L1 interprets that
evidence, L2 audits its grounding, L3 adds history, and L4 owns the action.

Registry entries must remain generic. They must not encode a particular case
path, job id, rank, timestamp, test-injection message, or expected action.

The current product has two closely related groups of deterministic detectors:

| Registry/detector group | Examples of facts produced |
| --- | --- |
| Failure/signature registry | Observed exception, permission denial, OOM indication, distributed timeout, teardown cascade, cause confirmation |
| Progress/checkpoint detectors | Iteration/step reached, checkpoint save/load started or completed, setup milestone, recovery or continuation after a fault |

An illustrative signature-registry entry is:

```yaml
registry_id: observed_exception
pattern: "<exception class ending in Error or Exception>: <message>"
role: root_candidate
initial_class_hint: ambiguous
```

This entry means "retain this as a possible initiating failure." It does not
mean "this is a user failure" or "stop the workload."

### What Is A Normalized Occurrence Group?

`NormalizedOccurrenceGroup` is the runtime aggregation of observed log lines
with the same normalized text shape. It is not a registry definition and it
does not establish that its members belong to one causal failure.

Normalization removes volatile values that should not split otherwise similar
observations, such as timestamps, rank prefixes, counters, and addresses. The
group then records the aggregate instead of retaining every repeated rendering:

```yaml
occurrence_group_id: og-1
normalized_shape: "runtimeerror_cuda_out_of_memory"
first_line: 1012
count: 2
sample_lines: [1012, 1013]
rank_spread: ["7", "9"]
registry_id: cuda_oom
classification: error
```

The implemented type and persisted schema use `NormalizedOccurrenceGroup`,
`occurrence_group_id`, and `normalized_shape`. This keeps occurrence aggregation
distinct from registry/progress detector `pattern_id` values.

Occurrence groups may be created from:

- registry matches, in which case `registry_id` identifies the detector that
  produced the observations;
- deterministic progress, checkpoint, setup, recovery, lifecycle, or
  diagnostic detectors, which may have no signature-registry id.

The relationship is:

```text
registry entry                         deterministic structural detector
     |                                              |
     v                                              v
registry matches                              observed markers
     |                                              |
     +---------------- normalize and group ---------+
                              |
                              v
                  Normalized Occurrence Group
```

An occurrence group can contain one or many lines. It can also span multiple
ranks, nodes, or distinct moments in the log. Its count and spread describe
where a normalized shape appeared; they do not prove same-incident fanout,
causal ordering, recovery, or recurrence.

Registry identity is part of the grouping key. If one line is retained under
both a generic `observed_exception` detector and a specific `cuda_oom`
detector, L0 may produce separate occurrence groups for those registry views
even though they reference the same source lines.

### Example: Failure Episode With An Associated Distributed Incident

Consider this synthetic log sequence:

The number before `|` is the source line number, not part of the log text:

```text
1000 | 7: [2026-07-15 10:00:00] iteration 418 / 1000 | consumed samples: 2568192 |
1012 | [rank 7] RuntimeError: CUDA out of memory
1013 | [rank 9] RuntimeError: CUDA out of memory
1030 | [rank 7] destroy_process_group() called during shutdown
```

L0 can represent it as three related structures:

```text
normalized occurrence group og-1
  normalized_shape: "runtimeerror_cuda_out_of_memory"
  first_line: 1012
  count: 2
  sample_lines: [1012, 1013]
  rank_spread: [7, 9]
  registry_id: cuda_oom

failure episode fe-1
  start_line: 1012
  end_line: 1030
  terminal_exception: observed exception/OOM at line 1012
  last_progress_before: iteration 418 at line 1000
  first_progress_after: none
  first_teardown_line: 1030
  status: terminal

distributed incident di-1
  incident_kind: distributed_fanout
  type: distributed_exception_fanout
  primary_observed_line: 1012
  member_event_lines: [1012, 1013]
  rank_spread: [7, 9]
  interpretation: same-attempt fanout, not cross-cycle recurrence
```

This does not imply one signature-registry entry for each displayed line:

- the iteration line is recognized by a dedicated progress detector and emits
  a typed progress marker; it is not a failure-signature registry match;
- the OOM line can match both the generic `observed_exception` row and the more
  specific `cuda_oom` row; L0 groups repeated rank copies into one incident
  rather than treating each matching line as an independent failure;
- the rank is parsed as line metadata, so rank numbers are not encoded in the
  registry pattern;
- `destroy_process_group` is recognized by a structural teardown detector. It
  helps L0 classify later activity as cleanup/cascade context and is not, by
  itself, a root-failure registry item.

Thus a raw line can produce no registry match, one match, multiple candidate
matches, or a non-registry structural fact. The L0A bundle combines those facts
into progress markers, normalized occurrence groups, incidents, and failure
episodes.

Those facts give L1 a compact failure episode without asserting whether the OOM
will repeat or whether retry can recover.

### Relationship Among Occurrence Groups, Episodes, And Incidents

These are different projections over some of the same observed lines, not a
strict parent-child hierarchy:

- a **normalized occurrence group** deduplicates a recurring text shape and
  records counts and placement spread without asserting a shared cause or time
  boundary;
- a **failure episode** explains local causal and temporal ordering around a
  terminal-looking failure: prior progress, exception chain, later progress or
  recovery, downstream fallout, and teardown;
- a **distributed incident** records either an inherently distributed mechanism
  or same-attempt multi-rank fanout. `distributed_mechanism` may have one
  observer; `distributed_fanout` requires at least two distinct ranks so fanout
  is not mistaken for independent failures or cross-cycle recurrence.

An occurrence group does not automatically become an incident: identical text
may appear in separate failure episodes or at unrelated times. An episode is
also not defined as a list of incidents. A single-rank ordinary failure is an
episode with no distributed incident. An inherently distributed mechanism
reported by one rank, such as a collective timeout, may be both an episode and
a `distributed_mechanism` incident. Multi-rank copies of one ordinary failure
may add a `distributed_fanout` incident and cause several initially parsed
per-rank exception candidates to be consolidated into one final failure
episode.

Examples:

| Log shape | Episode representation | Incident representation |
| --- | --- | --- |
| One rank raises a Python exception and the job exits | One episode containing the exception chain and teardown | None; there is no distributed fanout to group |
| One rank reports a collective timeout and the job exits | One episode containing timeout and teardown | One `distributed_mechanism` incident with `event_count=1`; no multi-rank corroboration is implied |
| The same OOM exception is rendered by 200 ranks before shutdown | One consolidated episode with prior progress and terminal outcome | One exception-fanout incident with 200 events and bounded rank samples |
| Hundreds of NCCL watchdog reports describe the same timeout epoch | One consolidated terminal episode around the earliest observed timeout and later cancellation | One collective-timeout incident grouping ranks, operations, and process groups |
| A data-reader error is followed by a distributed NCCL timeout wave | The reader failure is the possible initiating episode; the timeout is retained as downstream fallout or a separately linked episode, depending on the observed chain | The timeout wave remains its own incident; it does not become the proven root cause |
| Training advances after an error and later fails for a different reason | Separate episodes divided by compatible progress | Separate incidents only where either failure has distributed fanout |

A broader failure sequence can therefore involve multiple incidents, but they
remain separate records connected through ordering, shared episode lines, and
downstream/cascade relationships. They are not stored as child incidents
inside one episode merely because they are close together in the log.

### How The Harness Evaluates Registry Candidates

Every reviewed log is an opportunity to measure L0 coverage, but it is not a
reason to add a registry entry. The harness should collect evidence of gaps and
determine which tunable actually owns the problem.

Tool usage is one useful diagnostic signal:

| Observation across model runs | Likely interpretation |
| --- | --- |
| Several models request the same missing context and the tool returns new decision-relevant lines | Strong L0 coverage-gap signal |
| One model requests context that other models did not need | Possible model or model-profile behavior |
| A tool rereads lines already present in the model-facing bundle | Prompt/model/tool-efficiency issue, not a registry gap |
| A tool returns no new relevant evidence | Unproductive tool behavior or an insufficient tool request |
| Tool-provided evidence changes the primary, attribution, or operational assessment | High-priority investigation of the initial bundle |

A confirmed L0 coverage gap still does not automatically mean "add a registry
row." The harness should assign the gap to the narrowest correct owner:

| Gap | Candidate improvement |
| --- | --- |
| A generic deterministic line role was not recognized | Pattern, progress, checkpoint, recovery, or structural detector |
| L0A recognized the fact, but the L0B view omitted it | Projection selection, excerpt, deduplication, or budget profile |
| The relevant evidence was visible but the model misunderstood its role | Prompt or model-profile tuning |
| The model repeatedly reread visible evidence | Model/tool profile, not L0 |
| Semantics were sound but the action was wrong | L4 policy evaluation |

The harness may generate a registry candidate when the missing fact can be
recognized generically and deterministically. Multiple models finding the same
gap strengthens the case, as does recurrence across logs. A single log can
start an investigation, especially when human gold proves that L0 missed the
primary or progress evidence, but it normally should not directly change the
production registry.

The registry-candidate workflow is:

```text
run every log with a recorded profile
  -> inspect L0 coverage and tool-discovered evidence
  -> classify the owner of each gap
  -> cluster similar gaps across logs/models
  -> propose a generic detector when appropriate
  -> compare the existing and candidate registry profiles
  -> test affected cases and unrelated holdouts
  -> require human review
  -> publish a new versioned registry/profile
```

The harness is therefore a registry-candidate generator and validator, not an
automatic production-registry editor. Case paths, exact ranks, timestamps,
test names, and literal error-to-action mappings are prohibited candidates.

### What Is An Analysis Profile?

**Presentation scope:** retain this definition in the Markdown review artifact,
but omit a dedicated Analysis Profile slide from the PowerPoint deck. The deck
may refer to a versioned profile as the unit promoted by the harness without
enumerating this full configuration surface.

An analysis profile is the complete, non-secret, versioned configuration that
makes one analyzer run reproducible. It selects how every stage operates:

| Profile component | Controls |
| --- | --- |
| L0A | Registry versions, complete evidence assembly, grouping, and structured retention budgets |
| L0B | Evidence-selection rules and size limits for model-facing sections, excerpts, deduplication, context, and lossiness |
| L1 | Prompt/schema version, model and endpoint route, reasoning settings, token/context limits, advertised tools, tool rounds, retries, and timeout |
| L2 | Grounding/audit contract and mechanically permitted adjustments |
| L3 | Exact job/root history comparison, typed progress relations, and consecutive no-observed-advance counts |
| L4 | Retry-policy version, rule selection, retry budgets, action, and candidate priority |
| Observability | Trace schema, redaction, and artifact settings |

Credentials are not profile data. The profile may name an approved credential
slot or route, while the actual key remains in the runtime environment.

Conceptually, a profile looks like this:

```yaml
profile_id: restart-log-analyzer.qwen235b.v1
l0a:
  registry_version: builtin-mvp-v1
  bundle_strategy: progress-centered-episodes-v1
l0b:
  evidence_view_profile: attention-efficient-v1
l1:
  prompt_version: semantic-contract-v2
  model: nvidia/qwen/eccn-qwen-235b
  tools: [overview, grep_log, read_window]
  thinking_mode: auto
  max_tool_rounds: 2
l2:
  grounding_contract: evidence-audit-v1
l3:
  history_policy: exact-job-config-root-v1
l4:
  policy_version: restart-policy-v1
```

This example explains the intended configuration boundary; it is not an
accepted MVP profile-file format yet. The prototype currently derives the
effective profile from product defaults, environment variables, CLI options,
and harness target settings. Production qualification requires recording the
effective values and fingerprint so eval and production can run the same
declared profile.

The relationship is therefore:

```text
registry = one deterministic L0 input
prompt   = one semantic L1 input
policy   = one deterministic L4 input
profile  = the versioned configuration that selects all of them
```

## External Product Contract

### Runtime Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| `log_path` | yes | One cycle's interleaved workload log. The path is validated locally and is not exposed to L1 as a label. |
| `job_id` | production history | Stable NVRx job identity used for compatible-history comparison. |
| `cycle_id` | production history | Integer restart-cycle identity. Repeated captures of one cycle must not count as separate history attempts. |
| `attempt_history` | no | Compact prior-cycle results and progress facts. Raw prior logs are not required. |
| `analysis_mode` | no | `terminal`, `progressive_start`, or `progressive_end`; defaults to `terminal`. Terminal library/CLI execution is implemented; the progressive service lifecycle remains design work. |
| analysis profile | yes | Versioned L0A registries/assembly, L0B evidence-selection rules and size limits, L1 model/prompt/tool settings, history parameters, L4 policy, endpoint, and tracing configuration. |

Eval labels, directory names, and human gold are never runtime or model inputs.

### Final Output

```text
AnalysisResult
  decision: STOP | RESTART
  decision_basis: deterministic reason for the action
  primary_failure: grounded selected failure
  root_cause_assessment: preserved L1 assessment when available
  model_recovery_assessment: preserved L1 recovery assessment
  retry_policy: L4 rule, allowed retries, matched priors, and exhaustion state
  secondary_failures, cascades, evidence, evidence_coverage
  result_provenance:
    evidence_source
    model_contribution
    history_contribution
    result_quality
    nvrx_use
    L1 execution status/issues
```

The result is accompanied by a credential-free decision trace containing raw
stage outputs, transformations, timings, model/tool calls, retries, token use,
and candidate-selection details.

The harness consumes that trace through a versioned `ProductTrace` adapter.
The adapter validates the CLI envelope and selects `analysis_result` or
`collect_all_result` according to the declared schema; it does not infer an
unknown product version or reinterpret product-owned stage payloads.

## L0: Evidence Assembly And Initial Model View

L0 contains two deterministic sub-stages with separate outputs and quality
signals:

```text
L0A: raw log -> complete structured evidence bundle
Decision Evidence: complete bundle -> compact canonical decision facts
L0B: model-visible decision evidence + selected L0A support
     -> bounded initial model evidence view
```

"Complete" means the canonical structured L0 result for all inspected log
bytes, including explicit cap and lossiness accounting. It does not mean that
every raw log line is copied into the bundle. The source log remains available
to bounded read-only tools.

### L0A: Complete Evidence Assembly

#### Functional Inputs

- current-cycle log bytes visible at the time of analysis;
- generic pattern, progress, checkpoint, setup, cascade, and cause-confirmation
  registry;
- L0A scan, grouping, structured-retention, and lossiness budgets;
- prior progressive state when progressive execution is used.

#### Work

L0A deterministically scans and structures the log. It prioritizes stable
progress/checkpoint signals and failure episodes over a growing list of
error-specific rules. Repeated rank copies and downstream teardown are grouped
before excerpt selection.

#### Functional Output: `L0Bundle`

```text
source identity and byte/line accounting
job metadata
  explicit/observed world size, ranks, nodes, mapping availability
progress facts
  iterations, checkpoints, setup milestones, recovery markers
run progress summary
  completed progress, checkpoint load, observed failure position,
  replay distance, terminal timing
normalized occurrence groups and deterministic registry matches
candidate anchors and bounded context windows
failure episodes and distributed failure incidents
comparable-operation history within the current log
later-progress observations after fault-like events in the current log
post-fault summaries, cascades, and cause confirmations
stable observed fingerprints and experimental identities
evidence coverage, selection summary, caps, and lossiness
```

The immutable L0A bundle is retained for Decision Evidence construction, L0B
selection, L2 grounding/identity/audit, tracing, tools, and deterministic replay.

#### Candidate Anchor

A candidate anchor is a provisional location that tells downstream analysis
where to inspect and why. It is not an L0 root-cause conclusion. An anchor can
come from structural exception detection, a registry match, an explicit cause
confirmation, a terminal-looking line, or another high-signal observation.
It links the location to nearby progress, downstream fallout, and the bounded
window containing its original text.

Example:

```yaml
anchor_id: ca-1
line: 1012
quote: "[rank 7] RuntimeError: CUDA out of memory"
sources: [terminal_exception, high_signal]
causal_role_hint: unknown
prior_observed_progress_line: 1000
later_observed_progress_line: null
first_downstream_cascade:
  line: 1030
  fine_class: process_teardown
context_window_ids: [w-1]
```

This says line 1012 deserves inspection after progress at line 1000 and before
teardown at line 1030. It does not assert that the exception is the root cause
or that its policy is STOP.

#### Bounded Context Window

A bounded context window is an original-log slice around one or more candidate
anchor seed lines. It preserves coherent before/after evidence while preventing
the initial model view from being dominated by the full noisy log. Its bounds,
selection reason, original line numbers, occurrence-group links, and truncation
state are explicit.

Example:

```yaml
window_id: w-1
selected_by: terminal_exception
seed_lines: [1012]
start_line: 972
end_line: 1152
occurrence_group_ids: [og-3]
truncated: false
lines:
  - {line: 1000, text: "[rank 7] iteration 418 completed"}
  - {line: 1012, text: "[rank 7] RuntimeError: CUDA out of memory"}
  - {line: 1030, text: "[rank 7] destroy_process_group() called during shutdown"}
```

L0A retains the complete bounded window. L0B may merge or narrow windows for the
attention-efficient initial model view. An enabled read-only tool can inspect
additional raw context, so the L0B window is the best initial evidence selection,
not an evidence ceiling.

### Shared Artifact: Decision Evidence

#### Functional Inputs

- immutable L0A evidence bundle;
- request identity needed by downstream history and policy;
- deterministic identity and policy-evidence schema.

#### Work

Decision Evidence selects the compact canonical facts that L3 and L4 need. It
does not rescan the log, repeat L0A grouping, or perform semantic attribution.
The selection is built once from L0A and shared by both the fallback and model
branches. `EVIDENCE_BUNDLE.md` owns the normative selection algorithm.

#### Functional Output: `DecisionEvidence`

```text
canonical observed failure identity and stable fingerprint
failure position, iteration, operation, artifact, and data-position facts
fault outcome and later-progress/recovery observations from the current log
current progress and checkpoint state
rank, node, GPU, process-group, and mapping observations when available
coverage/lossiness and evidence provenance
```

Decision Evidence excludes prior-attempt history, model semantics, retry-policy
selection, and the final action. L3 adds history; L4 owns policy. The implementation
uses the immutable, schema-versioned `DecisionEvidence` type and constructs it once from
`L0Bundle` without rescanning the log.

The implemented deterministic selection is:

```text
L0A complete collections
  -> take L0A deterministic_primary_candidate, when present
  -> resolve its canonical failure-episode identity anchor
  -> prefer a matching distributed-incident fingerprint, then an identity-line
     registry fingerprint, then the deterministic primary fingerprint
  -> reference matching anchors, windows, episodes, incidents, occurrence groups,
     and source lines
  -> attach progress/checkpoint, comparable-operation, later-progress/recovery,
     locality, coverage/lossiness, and construction provenance
  -> freeze the schema-versioned DecisionEvidence
```

For example, if L0A selects an exception at line 1012 inside episode `fe-1`,
Decision Evidence may retain line 1012 as both the primary and identity anchor,
reference `ca-1`, `w-1`, `fe-1`, `di-1`, and `og-1`, and attach the last completed
iteration at line 1000 plus `progress_after_failure_episode=false`. It refers to
the complete window and incident in L0A rather than duplicating their raw data.

In presentation terms: L0A is complete structured evidence, Decision Evidence
is its canonical policy-relevant selection, and L0B is the bounded initial model
view. The exact contracts remain in the product evidence and schema specs.

### L0B: Initial Model Evidence View

#### Functional Inputs

- model-visible projection of Decision Evidence;
- immutable L0A evidence bundle for supporting-context selection;
- deterministic evidence-selection rules and section, excerpt, and context size
  limits;
- model-evidence schema version and lossiness requirements.

An illustrative L0B configuration is:

```yaml
l0b:
  max_context_windows: 4
  max_lines_per_window: 240
  max_chars_per_window: 50000
  max_chars_per_line: 360
  max_occurrence_groups: 30
  merge_windows_within_lines: 5
```

These values describe the current prototype's effective constants. They are a
target profile contract, not yet a fully configurable profile-file schema.

#### Work

L0B selects and serializes the evidence initially presented to L1. It combines
model-visible Decision Evidence with decision-relevant structure and bounded
original quotes selected from L0A, while reducing attention spent on routine
progress, repeated rank output, duplicate stack renderings, and downstream
teardown.

The goal is attention-efficient evidence delivery, not minimum byte count. A
larger view is justified when it adds necessary evidence; a smaller view is a
regression when it forces tools to recover omitted context.

#### Selection When Evidence Exceeds The Initial-View Limits

Context-window selection is deterministic. The current implementation:

1. reserves the earliest window containing a high-signal line, so late terminal
   output does not erase the earliest high-signal observation;
2. reserves a window containing the deterministic primary candidate when one
   exists;
3. ranks the remaining windows by whether they contain the primary candidate,
   their number of seed anchors, and their bounded count of high-signal lines;
4. breaks score ties by original start/end line order;
5. merges overlapping windows and windows separated by at most five lines;
6. retains at most four merged windows.

The current score is `1000` for containing the primary, plus `10` per seed
anchor, plus `20` per high-signal line up to ten such lines. The two reserved
windows are selected before this score fills the remaining slots.

Within each retained window, original line order is preserved. The initial view
keeps at most 240 lines, 50,000 characters, and 360 characters per line, and
marks truncation explicitly. Omitted L0A windows remain available to trace,
L2, and enabled read-only L1 tools.

Candidate anchors currently have a different boundary. L0A deterministically
retains at most 16 anchors by combining sampled high-signal lines, failure
episode starts and terminal lines, cause confirmations, the registry-selected
primary, and one representative per registry occurrence shape. High-signal
sampling preserves the earliest signal and favors traceback, process-kill,
exception, fatal/critical, assertion, error, and timeout lines while avoiding
nearby duplicates. The retained anchors are then ordered by source line, and
L0B currently includes all of them; it does not perform a second anchor ranking.
An anchor whose raw window was not selected remains visible with
`covered_by_excerpt=false`, allowing the model to request bounded context.

This distinction is intentional documentation of the prototype, not the final
tuning claim. Corpus evaluation should determine whether anchor selection also
needs an independent L0B priority and size limit. Tool requests for omitted
anchor context, gold-evidence coverage, semantic quality, route latency, and token use
provide the relevant measurements.

#### Functional Output: `L0ModelFacingView`

```text
compact run/job/progress overview
selected failure episodes and distributed incidents
candidate anchors and bounded coherent excerpts
deduplicated registry/pattern representatives with exact aggregate counts
post-fault progress, recovery, cascade, and teardown facts
coverage, omission, truncation, and lossiness metadata
```

`L0ModelFacingView` is the implemented type for the L0B Initial Model Evidence
View. Product code constructs it in the L0-owned projection module before
invoking L1, and the trace records its schema, exact payload, and projection
metrics independently from L1 model quality.

L0B is a sub-stage of L0 because it performs deterministic evidence selection
and no model reasoning. L0B ends when the final initial evidence payload is
ready. L1 begins when that payload is combined with model instructions and
submitted to the model. Prompt policy, model execution, and tool-driven
evidence expansion belong to L1.

The word `initial` is deliberate. L0B is intended to maximize the chance of a
correct one-turn assessment, but an enabled L1 tool profile may inspect bounded
additional L0A or raw-log context when the initial evidence is insufficient.
Attention efficiency is L0B's measurable objective, not a claim that its view
is exhaustive or that it controls the model's internal attention mechanism.

The trace contract preserves the complete L0A bundle, Decision Evidence, the
exact L0B Initial Model Evidence View, and later tool results as distinct
artifacts. It also preserves the L0 deterministic, raw L1 semantic, and L2
grounded primary selections independently.

### Quality KPIs And Operational Metrics

Quality and operations are separate scorecards. Runtime efficiency does not
prove evidence quality, and a missing gold label is reported as unavailable
rather than failed.

#### L0A Quality KPIs

- primary evidence coverage: an accepted gold primary line/window is present in
  the L0A candidate evidence set;
- selected primary accuracy: the deterministic primary singleton matches an
  accepted gold line within declared tolerance;
- progress/checkpoint detection precision, recall, and F1 with exhaustive gold,
  or recall only with partial gold;
- typed-event precision, recall, and F1 by exhaustively labeled event type;
- failure-episode and distributed-incident construction accuracy;
- later-progress/recovery fact accuracy; and
- coverage/lossiness reporting correctness.

#### L0A Operational Metrics

- assembly latency, source lines/bytes, and scan throughput;
- complete-bundle serialized size and reuse;
- counts of windows, anchors, episodes, incidents, and normalized occurrence
  groups;
- caps, truncation, and lossiness counts; and
- deterministic replay/hash consistency.

#### L0A Worked KPI Example

Assume the human gold for one log is:

```text
11800 iteration 418 completed
12000 loading checkpoint metadata
12083 UnicodeDecodeError in tensor_to_object
12135 FileNotFoundError during cleanup
```

Gold marks line 12083 as the initiating checkpoint-load failure, line 12135 as
teardown/cascade, iteration 418 as the last completed progress marker, and the
root cause as unresolved between durable checkpoint corruption and a transient
read failure. If L0A retains both 12083 and its context as primary candidates,
selects 12083 as the deterministic primary, records iteration 418, and assigns
12135 to the same failure episode as teardown, this case scores:

```text
primary evidence coverage:          pass (accepted line 12083 retained)
selected primary accuracy:          pass (singleton is line 12083)
progress detection:                 1/1 gold marker found
failure-role accuracy:              2/2 labeled failure lines correct
failure-episode construction:       pass (12083 root, 12135 teardown)
unsupported root-cause assertions:  0 (L0A does not claim corruption)
```

If L0A retains line 12083 but selects line 12135, primary evidence coverage
still passes while selected primary accuracy fails. The two KPIs deliberately
measure candidate-set recall and singleton selection separately.

#### Decision Evidence Signals

- deterministic projection latency;
- required identity/progress/recovery field completeness;
- stable-fingerprint availability and consistency;
- exact reuse by fallback and model branches;
- human-gold primary/progress retained from L0A.

#### L0B Quality KPIs

- required gold evidence and selected-reference coverage in the initial view;
- primary retention, conditional on L0A containing the accepted primary;
- relevant progress/checkpoint/supporting-context coverage;
- compaction safety: required evidence is not silently omitted; and
- first-turn completion, tool-call rate, context found only through tools, and
  bundled evidence reread through tools across models and profile ablations.

#### L0B Operational Metrics

- projection latency;
- model-view characters and estimated tokens;
- per-section and per-excerpt budget utilization;
- available/selected/omitted evidence counts plus merge/truncation counts; and
- projection integrity: serializable payload, resolvable references, valid line
  ranges, consistent selection accounting, limit compliance, explicit
  lossiness, and deterministic payload hash.

#### L0B Worked KPI Example

For the same checkpoint case, suppose L0A contains nine context windows and 19
rank-equivalent exception copies. L0B selects five windows, compacts the copies
to one representative plus a count, and preserves lines 11800, 12000, 12083,
and 12135. An illustrative score is:

```text
required evidence coverage:         4/4 gold lines/facts represented
primary retention:                  pass (12083 remains visible)
supporting-context coverage:        pass (load, prior progress, teardown visible)
compaction safety:                  pass (19 copies -> 1 + exact count)
tool-only required evidence:        0
projection latency:                 0.04 s
initial-view size:                  24,800 estimated tokens
configured budget utilization:      78%
window selection:                   9 available -> 5 selected
projection integrity:               pass
```

The numeric operational values are illustrative, not measurements from this
example log. Quality is judged against gold and downstream behavior; a smaller
view is not automatically better.

The downstream signals are influenced by model behavior, so the harness should
compare them across models, logs, and controlled L0B profile ablations before
assigning a projection defect.

### Degraded Behavior

- missing, unreadable, empty, or unusable terminal log produces an explicit
  fallback result;
- capped or sampled evidence remains usable only with lossiness recorded;
- inability to identify a semantic root is represented as candidate evidence,
  not a fabricated L0 STOP.

### L0 Does Not Own

- semantic root-cause attribution;
- user versus infrastructure classification;
- cross-cycle recurrence;
- `STOP` or `RESTART`.

## L1: Semantic Analysis

### Functional Inputs

- bounded L0B Initial Model Evidence View derived from Decision Evidence and
  selected supporting L0A context;
- versioned system prompt and structured output schema;
- model/endpoint and reasoning settings from the analysis profile;
- optional advertised read-only tools: `overview`, `grep_log`, and
  `read_window`; bounded `get_evidence_objects` is implemented but advertised
  only when its profile boolean is enabled;
- prior assistant/tool turns replayed by the stateless chat client.
- deterministic `restart_environment_context` from L0B: unchanged workload,
  recreated process state, normal restart delay, and hardware allocation or
  mutable external-service state that may change.

### Work

L1 asks the configured model to interpret the current log evidence. Optional
tools allow bounded L0A/raw-log inspection when the initial view is insufficient.
One bounded contract-repair turn is permitted for malformed required structure.

### Functional Output: `L1RawSemanticOutput`

```text
primary_failure
  source line, causal role, structured operation/mechanism/component/artifact
root_cause_assessment
  summary, status, plausible causes, missing evidence
recovery_assessment
  failure_domain {value, status, confidence}
  retry_outlook_without_workload_change {value, status, confidence}
  rationale
related_failures
claim-tagged evidence citations
```

L1 is asked for two related outputs, neither of which is an action:

1. **Observed failure and root-cause assessment.** Identify the primary
   mechanism, its causal role and source evidence, then state whether the root
   cause is established, supported but unconfirmed, hypothesis-only, or
   unknown. Preserve plausible alternatives and the evidence still missing.
2. **Current-attempt recovery assessment.** Describe what the current log says
   about the next unchanged NVRx attempt after normal teardown and restart
   delay.

The recovery assessment has exactly two claims:

| L1 field | Question answered | Downstream use |
| --- | --- | --- |
| `failure_domain` | Is the observed failure in the workload, infrastructure, or unknown domain? | Closed value plus independent evidence status and confidence; grounded L4 input. |
| `retry_outlook_without_workload_change` | After the declared restart transition, can the unchanged workload recover? | `cannot_recover`, `may_recover`, or `unknown`, plus independent status and confidence; used by L4 rule selection. |

L2 audits the cited evidence and mechanically checkable credibility of this raw
assessment without substituting its own semantic judgment. It preserves the raw
L1 object and emits any credibility concern as an unapplied audit suggestion.
L3 exclusively adds compatible cross-cycle recurrence and progress facts. L4 then applies
the versioned deterministic retry-budget policy to unchanged L1 fields plus L3
history and is the only stage that emits `STOP` or `RESTART`.

The exact raw response, parsed response, and interaction transcript are
immutable trace artifacts.

### Telemetry And Quality Signals

L1/model-route measurement is organized into four views:

- **Semantic quality:** semantic-primary accuracy against accepted gold lines
  or episodes; operation/mechanism/exception/causal-role accuracy; root-cause
  status; unsupported claims; recovery-field accuracy; and corpus calibration.
- **Behavioral efficiency:** first-turn usable result, structured-output
  usability, conversational/tool/repair turns, tool-context yield and
  redundancy, and input/output/reasoning/total tokens.
- **Endpoint reliability:** provider attempts, successful/failed calls,
  retries, timeouts, HTTP/provider failures, and failed-call time.
- **Route outcome:** model-enriched versus fallback contribution, final result
  quality/NVRx usability, end-to-end decision latency, and deadline outcome.

Semantic quality is conditional on a delivered response; an endpoint failure
with no response is `not_observed`. Provider retries do not increase model-turn
counts. Each model call records client-observed wall time. When the proxy also
returns timing headers, L1 retains the reported downstream-LLM-API span and
proxy pre-processing, post-processing, and message-copy spans. The downstream
span is not model compute time: it may include backend transport, queueing,
prefill, decode, and response delivery. Missing provider spans are omitted, not
inferred from client wall time.

#### L1 Worked KPI Example

Using the checkpoint case above, gold accepts line 12083 as primary and expects:

```text
operation: checkpoint_load
mechanism: metadata_deserialization
exception: UnicodeDecodeError
root_cause_status: unknown or supported_but_unconfirmed
failure_domain.value: unknown
failure_domain.status: unknown
retry_outlook_without_workload_change.value: may_recover or unknown
retry_outlook_without_workload_change.status: supported_but_unconfirmed or unknown
```

Example model outputs score differently:

| Model output | Semantic primary | Mechanism | Recovery fields | Root-cause safety |
| --- | --- | --- | --- | --- |
| `checkpoint_metadata_deserialization` at 12083 with the fields above | pass | operation/mechanism/exception all pass | 4/4 value/status fields accepted | pass |
| `python_user_exception` at 12083, same recovery fields | pass | operation/mechanism missing; exception pass | 4/4 accepted | pass but underspecified |
| `FileNotFoundError` at 12135 with established unrecoverability | fail: cleanup selected | fail | claim support fails | fail if it claims proven checkpoint corruption |

Operationally, a one-turn usable result in 8 seconds with no tools is reported
separately from semantic quality. A four-turn result that reaches the same
answer in 35 seconds may be equally accurate but behaviorally less efficient.
For example, a call may report client wall `0.609 s`, downstream LLM API
`0.157 s`, proxy pre-processing `1.839 ms`, and proxy post-processing
`0.690 ms`. This distinguishes the proxy's downstream span but does not label
the remaining time as gateway delay or split downstream time into queue,
prefill, and decode. Providers without these headers retain only client wall
time.

### Degraded Behavior

- provider or timeout failure leaves the deterministic candidate available;
- malformed required structure may trigger one repair turn;
- unusable final structure means L2 is not run and L3/L4 use the fallback path;
- a recovered provider retry is recorded as degraded L1 execution even when the
  accepted semantic output is usable.

### L1 Does Not Own

- trusted citations or stable history identity;
- compatible-history matching;
- retry-rule selection, retry-budget exhaustion, or final action.

The model is not allowed to decide `STOP` or `RESTART`.

L0 already owns the fallback-path root fingerprint. A model may propose a
diagnostic label, but that proposal cannot enter history.

## L2: Evidence Grounding, Identity, And Audit

### Functional Inputs

- immutable raw and parsed L1 output;
- the same immutable L0A bundle used to construct the L0B request view;
- exact model-visible initial excerpts and tool-result lines;
- source log for bounded citation verification.

### Work

L2 first performs the minimum source grounding required to construct the
enriched-path history identity. It then checks whether L1's references and
mechanically checkable claims are credible. It does not replace the model with
a second semantic opinion.

### Functional Outputs: `L2Result`, `CurrentFailureFacts`, And Grounded Semantics

```text
citation audit
  raw exact, rendered exact, abbreviated exact, nearby resolved, ungrounded
field findings
  field, code, message, severity, policy_material
ignored prohibited fields
audited related-failure roles
mechanical reference repairs with before/after/reason
stable identity anchor and deterministic root fingerprint
normalized CurrentFailureFacts for the L3 boundary
experimental family/concrete/client identities
grounded primary and related failures
preserved root-cause and model-policy assessments
policy-field audits with L1 value, support, suggestion, reason, and applied=false
```

Raw L1 remains unchanged. An exact source line or unique bounded quote match is
required before L2 can create a history fingerprint. If that grounding is not
available, current-cycle L1 semantics remain observable and usable, but
`CurrentFailureFacts.history_identity_ready=false`, so L3 cannot treat the model
signature as recurrence evidence. L2 may resolve an evidence line. When a semantic
field appears weak, L2 records an independent suggestion but cannot apply it.
For example, same-attempt rank fanout or checkpoint-to-failure position does not
by itself prove next-attempt persistence. Exact-object success, same-logical-
artifact success with shard uncertainty, and different-artifact success are
reported as different comparison strengths. Checkpoint comparisons retain the
logical checkpoint plus file/object/shard identity; dataloader comparisons
retain the dataset file plus shard/region when those facts are observable.

Findings are separated as:

| Finding | Effect |
| --- | --- |
| advisory | Visible only; does not degrade the result. |
| non-material credibility | Visible audit concern; does not alter policy inputs. |
| policy-material audit | Highlights a policy-relevant concern and may degrade result quality; the suggestion remains unapplied. |

### Telemetry And Quality Signals

- L2 latency and audit status;
- exact/rendered/nearby/unresolved citation counts;
- findings by severity and material-finding count;
- mechanical-reference-repair and policy-audit-observation counts;
- audit false-positive and false-negative rates under audit-focused gold;
- stable-identity completeness and cross-model agreement.
- enriched root-fingerprint owner/value/availability/source/history-readiness,
  relation to the L0 fallback fingerprint, and derivation latency;
- gold fingerprint accuracy plus corpus false-merge and false-split rates.

L2 owns this enriched-path fingerprint. L0 owns the fallback-path fingerprint.
Both paths are normalized to the same current-log-only `CurrentFailureFacts`
contract. L3 receives that object plus prior `attempt_history` as a separate
input and does not rewrite the fingerprint.

### Degraded Behavior

- structurally unusable L1 means `audit_status=not_run`;
- unresolved material findings preserve L1 provenance but mark the result
  degraded;
- non-material findings remain visible without changing NVRx eligibility.

### L2 Does Not Own

- whether the L1 semantic conclusion is correct;
- model, proxy, or endpoint timing and reliability;
- history recurrence;
- final policy or action.

## L3: History Enrichment

### Functional Inputs

- selected-path `CurrentFailureFacts` from L2 grounding or the deterministic L0 fallback;
- `job_id` and current `cycle_id`;
- compact prior `attempt_history` from distinct cycles, selected by
  `RestartAgentRuntime` before fallback and enriched branches run. This is an
  immutable in-memory view, not a serialized snapshot artifact.

Each `AttemptHistoryRecord` includes the stable root identity and a deterministic
`attempt_progress` summary. The summary records `observed`, `not_observed`, or
`unknown` training/checkpoint progress; first and last iteration and checkpoint
markers; their deltas/counts; failure position relative to observed training
progress; and progress after failure. This makes early-failure versus
progressed-attempt evidence explicit instead of asking L4 to infer it from one
last-step value.

### Work

L3 performs deterministic exact matching after stable structure exists. It does
not use an LLM to decide whether two differently worded model summaries mean the
same thing.

### Functional Output: `HistorySummary`

```text
history available
same-job and exact-root prior-attempt counts
per-cycle marker type, prior/current values, delta, and relation
per-cycle absolute attempt progress and early/after-progress failure position
observed-advance, same, regressed, unknown, and no-observed-advance counts
matching-root observed-progress, pre-progress-failure, and unknown-progress counts
exact failure-iteration, data-position, and artifact matches
consecutive same-root no-observed-advance count
same/cross node, GPU, and rank locality observations
rank-to-GPU mapping availability
```

History comparison requires exact `job_id`, distinct earlier cycles, and the
same grounded root fingerprint. L3 compares failure iteration only with failure
iteration, completed step only with completed step, and checkpoint step only
with checkpoint step. It emits `advanced`, `same`, `regressed`, or `unknown`.
A one-step positive delta is observed advancement. Because progress logging may
be sparse, equal/lower markers mean no **observed** advance, not proof that the
workload performed no unlogged work. Rank equality does not imply the same GPU
unless a mapping is available.

Relative advancement and absolute attempt progress answer different questions.
`advanced` asks whether this attempt went farther than a comparable prior one.
`attempt_progress` asks whether the prior attempt failed before any observed
training progress or after completing observable work. L3 preserves both. The
current MVP L4 rule uses advancement and the no-observed-advance streak; the
absolute summary is traced and available for a separately reviewed policy rule.

Runtime history is enabled by default with 3,000 attempts per exact job and may
be disabled or resized by product configuration. Upsert is idempotent by
`(job_id, cycle_id)`. State-based library/unit tests seed, inspect, and clear
the in-memory store through `RestartHistoryControl`.

### Telemetry And Quality Signals

- history availability and L3 latency;
- same-job/exact-root counts and per-cycle progress relations;
- progress-relation and delta accuracy against L3 gold;
- attempt-progress classification and marker-summary accuracy against L3 gold;
- exact failure/data/artifact match accuracy;
- consecutive no-observed-advance count accuracy;
- fingerprint false-merge and false-split rates;
- unknown-comparison rate.
- history-store enabled state, configured bound, seed provenance, evictions,
  and selected-record/replacement reason.
- manual fixture round-trip determinism and stable record ordering.

### Degraded Behavior

- absent history emits `available=false`; it is not evidence of non-recurrence;
- malformed records are counted and ignored;
- unknown or marker-incompatible progress remains visible and does not become
  no-observed-advance evidence.

### L3 Does Not Own

- semantic attribution;
- retry rules and retry-budget thresholds;
- `STOP` or `RESTART`.

## L4: Policy Decision

### Functional Inputs

- unchanged current-cycle L1 primary and recovery assessment, or the
  conservative Decision Evidence fallback candidate;
- L3 `HistorySummary`;
- versioned retry-budget and action policy.

Policy-active current-cycle fields are:

```text
failure_domain {value, status}
retry_outlook_without_workload_change {value, status}
```

Both claim confidences remain observable but are not policy inputs.

### Work

L4 combines the current-cycle recovery assessment with L3 observations,
selects a versioned retry rule, records the allowed-retry budget and matching
prior count, determines whether the budget is exhausted, and emits the action
with a deterministic decision basis.

Initial retry rules are:

| Rule | Selection | Allowed retries after the first failure |
| --- | --- |
| `workload_unrecoverable` | Policy-grounded workload domain and cannot-recover outlook, both `established_by_current_log` | `0` |
| `bounded_retry` | Policy-grounded `may_recover` outlook with established or supported status | configurable, default `1` |
| `general_retry` | Unknown, degraded, or otherwise unqualified current-attempt semantics | configurable, default `3` |

Only distinct prior cycles with exact `job_id`, exact root fingerprint, and no
observed advance count toward exhaustion. Unknown progress does not count, and
observed advance prevents exhaustion for the current decision.

### Functional Outputs: `L4DecisionOutput` And `AnalysisResult`

```text
retry_policy
  policy_version, rule, allowed_retries, matching_prior_failures,
  retry_budget_exhausted, current_evidence_qualified, observed_advance
decision_basis
decision: STOP | RESTART
result provenance and NVRx usability
selected primary, assessments, evidence, independent secondary failures,
and downstream effects with explicit cascade/teardown roles
```

Final assembly preserves L0 structural roles and L2-grounded relationship
rationales. Downstream cleanup and fanout are emitted under `cascades` with
`causal_role=cascade|teardown`, rather than being flattened into independent
secondary failures. This output assembly does not feed L4 policy.

The first current-cycle occurrence remains restart-biased unless evidence
affirmatively supports a required intervention before retry. Compatible
recurrence without progress can strengthen a later `STOP`.

### Telemetry And Quality Signals

- retry-rule, allowed-retry, matched-prior, and exhaustion accuracy;
- deterministic replay;
- action and decision-basis accuracy;
- `STOP` precision and false-`STOP` rate;
- bounded/general retry and ambiguous-case correctness;
- result quality, NVRx eligibility, and L4 latency.

### Degraded Behavior

- missing/unusable L1 uses conservative Decision Evidence plus history;
- model unavailable or pending is explicit in result provenance;
- absent primary produces fallback-only quality and NVRx default handling;
- L4 never presents a degraded result as normal.

### L4 Does Not Own

- raw-log interpretation;
- citation repair;
- history identity construction.

## Progressive Lifecycle And L0A Readiness

Progressive execution moves deterministic L0A work earlier, but it does not
equate workload end with complete shared-log evidence:

```text
cycle start       workload running       progressive_end     log converges
    |----------- L0A tail + precompute ---------|<-- final L0A overlap -->|
                                                |<-- NVRx decision window -->
```

- `progressive_start` creates the active attempt state and begins L0A tailing
  and precomputation while the workload is still running.
- `progressive_end` reports workload-cycle completion and starts the
  NVRx-owned decision window. It does not freeze L0A.
- Buffered and interleaved ranks may continue appending during the post-end
  log-drain overlap. L0A keeps ingesting and updating its structured evidence.
- Log convergence is the readiness signal that the available bytes needed for
  final L0A assembly have arrived. The analyzer then freezes `L0Bundle` and
  constructs the shared `DecisionEvidence` object.
- Freezing Decision Evidence starts fallback L3/L4 and the L0B/L1 enriched path
  from the same canonical facts. Candidate selection remains deadline-aware.

Progressive replay must measure cycle-end time, convergence time, L0A-ready
time, fallback-ready time, enriched-ready time, and the final candidate chosen
before the caller deadline. Terminal/manual runs do not establish those
production latency distributions.

## Candidate Selection And Deadline

| Candidate | Inputs | Ready condition | Purpose |
| --- | --- | --- | --- |
| `deterministic_fallback` | Decision Evidence + L3 + L4 | As soon as deterministic assembly/history/policy finish | Deadline-safe recommendation when L1 is late or unavailable. |
| `l1_enriched` | Decision Evidence + L0B + L1 + L2 + L3 + L4 | After a usable L1 response and downstream stages finish | Preferred recommendation when ready before the caller deadline. |

NVRx owns the exact decision deadline. The analyzer exposes candidate readiness
and provenance; it does not extend the caller's window. A late enriched result
may enter history for a later cycle but cannot revise a cycle NVRx has closed.

Current implementation note: terminal library/CLI analysis waits for the
enriched result. The fallback-ready callback exists. Full progressive
start/end/readiness service integration and log-drain qualification remain open.

## Trace Contract At Every Stage

Each stage exposes the same three planes:

| Plane | Contents | Downstream policy input? |
| --- | --- | --- |
| functional output | Typed facts intentionally passed to the next stage | yes |
| telemetry | timing, counts, calls, retries, tokens, coverage, findings | no |
| trace artifact | exact bundle, transcript, model output, transformations | required by review harness |
| process diagnostics | unexpected warnings, uncaught errors, and nonzero process exits | failure-only `stderr.log`; handled model/endpoint errors stay in trace telemetry |

This distinction prevents observability values or harness labels from silently
becoming policy inputs.

## Harness Boundary

The offline harness consumes product outputs and traces plus optional
human-approved gold. It emits per-model reviews, panel comparisons, stage
scores, endpoint/latency measurements, and profile recommendations.

The harness does not:

- provide evidence to a production model;
- modify a live result;
- choose a production profile without external approval;
- ship in the NVRx installable package.

## Proposed Deck Translation After Review

Once this document is approved, use it to revise the deck as follows:

1. one architecture slide showing external input, L0-L4 handoffs, both
   candidates, and final output;
2. one product-contract slide for L0-L2 inputs and outputs;
3. one product-contract slide for L3-L4, candidate selection, and the NVRx
   deadline;
4. one observability slide showing functional output versus telemetry/debug;
5. keep harness inputs/outputs in the separate Tune section.
