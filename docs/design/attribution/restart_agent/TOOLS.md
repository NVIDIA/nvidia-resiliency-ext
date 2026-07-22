# Restart Agent Tools And Pipeline Spec

This file is canonical for the L1 read-only tool loop, tool contracts,
prompt/client behavior, and provider/model fallback. `EVIDENCE_BUNDLE.md` is
canonical for deterministic bundle generation and large-log context assembly.
`DESIGN.md` is the index and global contract.

## Implemented Defaults

The route-profile settings in the first table MAY be overridden by
`restart_agent_config.v1`. The tool response limits in the second table are
fixed implementation bounds in the current schema. Changes to either set
SHOULD be measured against the eval corpus.

Raw-log tools operate on the immutable source snapshot created by preparation;
they do not reopen the source path per call. The evidence-tool factory is a
runtime dependency, so another read-only source can preserve the same
`EvidenceTools` contract without changing L1 or provider code.

### Library Fallback Without A Config

These are constructor fallbacks for direct callers that do not supply
`restart_agent_config.v1`; they are not promoted production-profile values.
`PROFILE.md` and the resolved effective config own route behavior.

| Setting | Default |
|---|---:|
| `routing.timeout_seconds` | 600 |
| `request.timeout_seconds` | 120 |
| `request.max_output_tokens` | 64000 |
| `request.context_window_tokens` | provider/model profile value; Qwen 235B known default 200000 |
| `request.context_safety_tokens` | 4096 |
| `request.temperature` | `0.2` |
| `request.top_p` | `0.7` |
| `tools.enabled` | true |
| `tools.advertisement` | `overview`, `grep_log`, `read_window`; object-read disabled |
| `tools.max_rounds` | 8 |
| `reliability.max_retries` | 1 |
| `reliability.retry_backoff_seconds` | 0.5 |
| `reasoning.thinking_mode` | `auto` |

### Fixed Tool Response Limits

| Setting | Default |
|---|---:|
| `overview.head_lines` | 40 |
| `overview.tail_lines` | 80 |
| `overview.max_chars` | 12000 |
| `grep_log.max_matches` | 50 |
| `grep_log.max_matches_hard_limit` | 200 |
| `read_window.before` | 20 |
| `read_window.after` | 80 |
| `read_window.max_lines` | 240 |
| `read_window.max_chars` | 50000 |
| `get_evidence_objects.max_refs` | 8 |
| `get_evidence_objects.max_chars` | 50000 |

## Large-Log Handling

The runtime analyzer MUST NOT depend on blindly inlining the full log.

L0A MUST first build the deterministic complete evidence bundle. When an L1
route is scheduled, L0B MUST then build the versioned, bounded model-facing
projection before L1 begins. Deterministic-only and log-unavailable executions
do not need to construct L0B. L0A and the log are exposed through read-only
tools for additional inspection. Tool outputs MUST include original line numbers and
truncation metadata. Model-facing prompts and tool outputs MUST NOT expose the
absolute log path, basename, parent directory names, eval case directory, or
path-derived hints because those values may contain human or LLM-generated
labels.

`EVIDENCE_BUNDLE.md` owns the detailed bundle-generation contract. This file
defines how the model sees the bundle and how it may use read-only tools for
additional context.

The initial L1 user message MUST include:

- the required `response_schema`;
- selected `decision_evidence`;
- current-log `attempt_execution_context`;
- declared `restart_environment_context`; and
- the supplied evidence projection under the model-facing `evidence_bundle`
  key, including byte size and line count.

The initial conversation trace MUST record these five sections as the exact
parsed `model_visible_payload` sent to the provider. L2 line and quote visibility
checks consume that payload in full, plus later tool results; they MUST NOT
approximate visibility from only the `evidence_bundle` subsection.

The response schema MUST be generated from the same executable contract used by
client validation. The advertised and enforced field sets, enums, limits,
support tags, confidence bounds, and non-primary semantics therefore cannot
drift independently.

For canonical L1 evidence, visibility includes the paired source line and text
or quote, not merely a provenance line number. Exact source-log content that was
absent from both the initial payload and tool results MUST NOT become grounded
claim support. L2 MAY repair a small line-number error only when the cited quote
was visible at one unique nearby line.

Advertised read-only tool schemas are carried by the provider request's `tools`
field, not duplicated in the user-message JSON. Tool-loop limits and routing
details are client execution metadata and belong in the trace/profile rather
than the model prompt.

Decision Evidence object ids are provenance references. They become resolvable
model inputs only when `get_evidence_objects` is among those advertised tools;
the default profile continues to leave that tool unadvertised.

The initial L1 user message MUST NOT include path-derived source metadata such
as log path, basename, parent directory names, case id, observed label directory,
or run-name/path hints. Traces may retain those values for reproducibility
outside the model conversation.

The initial L1 user message MUST NOT include `AttemptRecord` values or the
runtime-selected `PriorAttemptView` in MVP. L3 applies history after L1/L2. The initial user message SHOULD NOT
include large raw log content. Small previews belong in `overview`.

Policy invariant: ordering, progress-before-fault, coverage, and history inputs
used by L3/L4 come from L0, tool-call accounting, fallback context assembly,
the validated request, effective configuration, and runtime-selected
`PriorAttemptView`.
They MUST NOT come from model-authored fields. L1 may select or normalize the
current-log primary failure, but L2 MUST ground that selection against L0
ordering and model-visible cited evidence before it becomes eligible for
history comparison or policy.

The analyzer MUST NOT make a terminality decision from an error-only extracted
line set. Error-only filtering can surface candidates, but any evidence used to
prove terminal failure, recovery, or progress MUST include original-log context
around the candidate or a deterministic marker from the original log.

## Occurrence Normalization And Cache

L0 SHOULD normalize every candidate line into a volatile-token-stripped shape
before expensive classification. The normalized shape is an implementation
cache key and grouping key; it is not automatically the root fingerprint.

Normalized occurrence groups SHOULD preserve:

- normalized shape;
- first line;
- count;
- sample lines;
- rank, node, and GPU spread when parseable;
- deterministic registry match, when any;
- classification source: `deterministic`, `cache`, `l1`, or `unknown`.

The model-facing registry view MUST collapse repeated matches into provisional
groups and omit authoritative `policy_class` and root-role labels. L0 may retain
those fields internally for fallback and observability, but L1 must establish
causal position from traceback episodes, progress, and bounded raw context.

A normalization cache MAY store stable classifications such as `error`, `benign`,
`progress`, `checkpoint`, and `registry_id`. Cache hits are latency/cost
optimizations only. They MUST NOT bypass evidence coverage, original line
references, ordering checks, or final policy rules. Persistent cache is optional
for MVP, but cache hit/miss counts MUST be recorded in the trace.

## Context Assembly

L0 SHOULD assemble context windows by selecting representative seed lines from
normalized occurrence groups and expanding around those seed lines in the original log. This
is the preferred fallback context strategy, ahead of naive head/tail truncation.

Context windows MUST preserve original line numbers, seed line numbers,
truncation state, and why the window was selected. They SHOULD include enough
after-context to detect successful retry, continued iteration, checkpoint
completion, or clean application exit after a candidate fault.

Fallback prompts MUST use the L0 summary plus bounded representative context
windows. Head/tail previews are useful for orientation, but MUST NOT be the only
large-log strategy when candidate fault lines are known.

## Progressive Context Assembly

The future progressive path may run the same context assembly while the workload
is active and finalize it after cycle end. Tool availability and budgets remain
profile-owned. `PROGRESSIVE.md` owns the lifecycle, state, deadline, and
terminal-equivalence rules.

## Selection And Lossiness Accounting

Any stage that filters, deduplicates, samples, truncates, summarizes, or
otherwise drops log content MUST record what happened in the decision trace.
This includes deterministic filters, pattern deduplication, cache lookup,
bounded context assembly, progressive window eviction, L1 tool budget/cap
events when configured, and fallback pre-curation.

At minimum, the trace SHOULD record:

- raw line count and byte size;
- candidate line count after deterministic filters;
- unique normalized pattern count;
- count of lines or patterns dropped by benign/noise filters;
- count of candidate lines represented only by pattern samples;
- context windows selected and whether they were truncated;
- progressive raw-window lines evicted and durable candidate summaries retained;
- whether after-context was available for the selected primary candidate;
- whether any cited evidence came from an error-only view;
- whether any configured cap or max-output limit was hit.

Selection accounting is diagnostic and for coverage evaluation. It MUST NOT be
used as positive evidence for either policy class. If filtering or truncation
prevents inspection of context after a candidate fault, that candidate remains
`unresolved` unless the candidate line itself is explicit terminal user/config
evidence.

## Tool Contracts

The agent loop exposes tools bound to exactly one log file. The available tool
set is declared by the selected analysis profile and advertised by the analyzer
client to the model. The model can choose among advertised tools, but it cannot
create new executable tools.

Production MUST NOT dynamically create arbitrary executable tools from model
requests. Unknown tool names, unsupported tool requests, malformed arguments, or
requests outside the current profile MUST be rejected by the client and recorded
in the trace. Eval SHOULD report unsupported tool requests because they are a
signal that the generic tool interface may be too weak.

The default MVP production profile advertises exactly three generic
log-inspection primitives: `overview`, `grep_log`, and `read_window`. Tools
SHOULD remain generic, not
narrow failure-specific helpers. Future tools must be similarly general
inspection primitives and must be introduced through a new
analysis-profile/tool-interface version rather than dynamically created from
model requests. Prefer this shape over helpers like `find_device_side_assert` or
`find_nccl_timeout`.

The client also implements bounded `get_evidence_objects` lookup for future
profile-based evaluation. It is not in the default advertised-tool allowlist.
Therefore its schema is absent from default model requests and a model-authored
call to it is rejected as `tool_not_advertised`. A declared profile may opt in
by setting `tools.advertisement.get_evidence_objects=true`; implementation
availability alone never makes a tool callable. Single-model compatibility
configuration may still use `NVRX_LLM_ADVERTISED_TOOLS`, but a supplied
restart-agent config is authoritative.

### `overview`

Input: none.

Output:

```json
{
  "line_count": 1234,
  "byte_size": 456789,
  "head": [{"line": 1, "text": "..."}],
  "tail": [{"line": 1200, "text": "..."}],
  "deterministic_summary": {
    "progress_lines": [],
    "registry_candidate_groups": [],
    "registry_candidate": null,
    "candidate_anchors": [],
    "failure_episodes": [],
    "cause_confirmations": [],
    "cascade_groups": [],
    "termination_candidates": []
  },
  "truncated": false
}
```

`overview` MUST include line count, byte size, and enough head/tail preview to
orient the model. It MUST NOT include path-derived labels or hints. L0 runs once
per analysis; `overview` MUST be a read-only view over the same L0 bundle, not
an independent recomputation.

### `grep_log`

Input:

```json
{
  "pattern": "raising GPU error|device-side assert|Traceback",
  "ignore_case": true,
  "max_matches": 50
}
```

Output:

```json
{
  "pattern": "...",
  "matches": [{"line": 1174, "text": "..."}],
  "total_matches": 1,
  "truncated": false
}
```

`grep_log` MUST use the original line numbers. It MAY truncate matches, but MUST
report truncation.

### `read_window`

Input:

```json
{
  "center_line": 1174,
  "before": 20,
  "after": 80
}
```

Output:

```json
{
  "start_line": 1154,
  "end_line": 1254,
  "lines": [{"line": 1174, "text": "..."}],
  "truncated": false
}
```

The implementation MUST bound `before`, `after`, total returned lines, and total
characters. Defaults SHOULD favor enough context to see progress before the
fault and cascade after it.

If the requested raw range is unavailable because the file was truncated,
overwritten, evicted from a bounded progressive window, or capped by the
profile, `read_window` MUST return a deterministic non-crashing result:

```json
{
  "start_line": 1154,
  "end_line": 1254,
  "lines": [],
  "truncated": true,
  "error": "window_unavailable",
  "unavailable_reason": "progressive_window_evicted",
  "candidate_summary_refs": ["cand-17"]
}
```

Retained candidate summaries may be referenced to orient the model, but they
MUST NOT be rendered as raw log lines.

### `get_evidence_objects` (not advertised by default)

Input:

```json
{
  "refs": ["fe-1", "w-3", "og-4"]
}
```

Output:

```json
{
  "schema_version": "restart_agent_evidence_objects.v1",
  "requested_refs": ["fe-1", "w-3", "og-4"],
  "objects": [
    {
      "ref": "fe-1",
      "object_type": "failure_episode",
      "payload": {"episode_id": "fe-1", "start_line": 1000},
      "truncated": false
    }
  ],
  "missing_refs": [],
  "invalid_refs": [],
  "omitted_refs": [],
  "limits": {"max_refs": 8, "max_chars": 50000},
  "truncated": false
}
```

The tool resolves attempt-scoped L0A object identifiers for occurrence groups,
context windows, candidate anchors, failure episodes, distributed incidents,
and progress/checkpoint/setup markers. It MUST read the immutable L0A bundle;
it MUST NOT rescan the log. Results are versioned and bounded by reference
count and serialized size. Missing, invalid, omitted, and truncated data are
reported explicitly.

### Tool-Call Observability

Every tool call MUST emit trace fields that make production/eval divergence
debuggable:

- `tool_call_id`;
- `analysis_profile_id` or profile fingerprint;
- phase: `terminal`, `progressive_start`, or `progressive_end`;
- model turn/request id that caused the call, when applicable;
- tool name;
- normalized argument hash plus a short redacted argument summary;
- start time, end time, and latency;
- input log offset or visible line range at call time;
- returned line count, returned character count, and new line ids returned;
- total matches and truncation status for search tools;
- configured cap values and any caps hit;
- timeout, provider/tool error, or retry status.

Rejected or unsupported tool requests MUST also be traced with the requested
tool name, argument summary when parseable, reason for rejection, model
turn/request id, phase, and analysis profile id.

The trace SHOULD aggregate tool-call count, latency, error rate, truncation
rate, unsupported-request rate, and new-line yield by profile, phase, and tool
name. These metrics are diagnostic only; they do not directly change policy
scores.

## Analysis Pipeline

### L0 Deterministic Pre-Pass

L0 is non-LLM analysis. It MUST produce the compact evidence bundle specified
in `EVIDENCE_BUNDLE.md`. The list below is a pipeline summary retained here so
the L1/tool contract can be read in sequence.

Required L0 outputs:

- file metadata: path, size, line count;
- rank extraction when rank prefixes are present, plus node/GPU extraction only
  when explicit fields or validated rank-to-GPU mapping are available;
- normalized occurrence groups for candidate error, progress, checkpoint,
  scheduler, and cascade lines;
- context windows selected from representative normalized occurrence groups with raw original
  log lines and line numbers;
- candidate anchors/event timeline entries, including high-signal lines without
  taxonomy matches, nearby observed progress context, rank relation for later
  progress observations when available, downstream registry/cascade links, and
  context-window coverage;
- application progress markers such as iteration, step, or epoch, including
  highest observed value and whether progress appears before or after candidate
  failures;
- checkpoint markers, including completed checkpoint saves, checkpoint step when
  parseable, last completed checkpoint line, and whether each candidate failure
  is before or after the last completed checkpoint;
- progress-centered failure episodes with recent compatible progress before the
  exception, terminal exception context, cancellation/downstream fallout, and
  first compatible progress after the episode when present;
- bounded explicit scheduler/kernel/runtime cause confirmations linked to a
  preceding terminal episode when no compatible progress intervenes;
- recovery markers after candidate faults, such as successful retry, successful
  path access after an earlier path error, resumed iteration, or other evidence
  that the training loop continued;
- workload recovery hints such as bad-token/bad-sample detection, token-window
  retry, token-window skip/quarantine, retry counter, and stable
  `data_position_fingerprint` when the log exposes one;
- scheduler or external-policy markers such as time limit, preemption, or
  cancellation;
- registry matches with line numbers, signatures, `registry_id`, `fine_class`,
  `policy_class`, role, fingerprint, and candidate `fault_outcome`;
- first-fault candidates sorted by line/time;
- progress segments that group nearby candidate faults and mark whether each
  segment is terminal, recovered, progressed-after, or unresolved;
- selection/lossiness metadata for filters, deduplication, sampling, context
  windows, and configured caps;
- progressive state metadata and persisted candidate summaries when running in
  progressive mode;
- cascade groups such as repeated CUDA/NCCL aborts after a prior rank failure;
- health-before-fault signal;
- root fingerprint candidates with volatile tokens stripped;
- cascade fingerprints for repeated downstream symptoms.

Same-rank evidence is not same-GPU evidence. L0 MUST keep rank, node, GPU, and
rank-to-GPU mapping as separate structured fields so the policy engine can
distinguish same-rank recurrence, validated same-GPU recurrence, same-node
recurrence, and cross-node recurrence.

Cascade groups MUST be first-class objects:

```json
{
  "fine_class": "nccl_cascade",
  "policy_class": "cascade",
  "cascade_fingerprint": "nccl_cascade:watchdog_timeout",
  "first_line": 1300,
  "last_line": 1880,
  "count": 200,
  "sample_lines": [1300, 1314, 1341],
  "rank_spread": ["0", "1", "31"],
  "node_spread": [],
  "reason": "all instances occur after line 1174 primary fault candidate"
}
```

L0 SHOULD implement every built-in registry row in `PATTERN_REGISTRY.md`.

L0 evidence alone MUST NOT select semantic `STOP`. The deterministic fallback
may still stop through an exhausted L3/L4 same-root retry budget.

Implementation note: the L0 evidence bundle is a thin structured view of the log,
not an LLM summary. It SHOULD preserve metadata, ordering, line references,
sample raw lines, grouped repeated symptoms, candidate roots, and cascade groups.
The purpose is to remove repetition and expose root-cause candidates without
discarding auditable evidence. The LLM MAY use tools to inspect raw windows
around bundle entries before returning structured evidence.

L0 SHOULD use deterministic regex/parser rules for application iteration,
checkpoint completion, and scheduler markers first. Ambiguous progress or
checkpoint text MAY be left for L1 inspection, but model-authored progress
claims MUST NOT directly drive policy. L3 uses only deterministic L0 progress
facts.
L0 SHOULD also emit compact `run_progress_summary` and `job_metadata` facts
when deterministic markers are available. Explicit `world_size` lines are
authoritative job metadata; rank-derived world size is only an
`inferred_world_size_lower_bound` and MUST be labeled as such.
An iteration explicitly attached to a terminal exception is exposed as
`latest_observed_failure_iteration` and checkpoint-load distance, not as a
completed progress marker.
Nearby progress observations in candidate anchors are ordering context, not
candidate outcome decisions. A later compatible, advancing workload marker may
establish job-level `progressed_after` in an interleaved multi-rank log, but it
does not prove recovery of the same rank, node, GPU, or component.

L0 MUST classify candidate fault outcome when evidence is available:

- `terminal`: no later recovery/progress evidence and the candidate is tied to
  job termination.
- `recovered`: later log evidence shows the same operation retried or succeeded.
- `progressed_after`: later application iteration/step/checkpoint progress is
  observed after the candidate.
- `unresolved`: no clear terminal or recovery/progress evidence.

Only `terminal` or explicitly terminal-linked `unresolved` candidates can become
the deterministic primary represented in `AttemptFailureFacts` and supplied to
L3/L4. Recovered/progressed-after faults remain useful context but are not
policy-active roots by themselves.

Failure episodes SHOULD be built before generic error-pattern classification
when possible. A simple implementation may use the first terminal-looking
traceback/exception after the latest compatible progress marker, then check for
later compatible progress before assigning `terminal` or `progressed_after`.
Progress segments may still group broader candidate fault line positions, but
the output contract is the status and cited original lines, not the particular
split algorithm.

For workload-detected bad-token or token-window failures, L0 SHOULD extract the
stable data/token/window identity separately as `data_position_fingerprint` when
available. It SHOULD also extract any logged retry, skip, or quarantine status
for observability and future policy work. The generic MVP retry policy does not
yet grant a workload-managed skip exception.

### L1 Read-Only Tool Loop

The MVP supports an optional L1 read-only tool loop for readable, non-empty logs
when an L1 profile is configured. Tool calling is a profile-declared
context-access mechanism, not a different decision scheme. L1 MUST start from
the deterministic L0 evidence bundle; it MUST NOT begin from an unstructured
large raw log. L1 may then use `overview`, `grep_log`, and `read_window` to
inspect additional current-log context before returning structured evidence.

Production and eval MUST use the same L1 profile when comparing effectiveness.
If the production profile enables tools, production-fidelity eval enables the
same advertised tools and budget policy. If the production profile disables
tools, eval disables them. A terminal static replay that uses a different tool
profile is a useful debug run, not a comparable production measurement.

Tool-call count, extra model turns, incremental token cost, duplicate raw-window
reads, and new evidence returned are first-class optimization signals. A tool
call that only re-reads lines already available in the L0 bundle should drive
bundle/prompt improvement before introducing narrower fault-specific registry
rules.

L0 may skip L1 only before semantic log analysis is possible, such as missing,
unreadable, or empty logs, or when no L1 profile is configured. For readable
logs with an L1 profile, L0 MUST NOT skip L1 because a registry row appears
decisive. L0 registry matches are candidate evidence for the bundle. The
deterministic fallback may still produce an interim recommendation from L0
facts and history while L1 is in flight.

If L1 is absent, times out, returns malformed output, or fails its structural
output contract, L0 may still provide a fallback `primary_failure` and grounded
history identity. L4 applies the general retry rule unless deterministic input
supports a narrower reviewed rule. Result provenance and L1 degradation remain
visible in the output and trace.

Implemented client guarantees:

- The profile limits total tool-loop model turns through `tools.max_rounds`;
  tools-disabled execution makes one model request.
- Every provider request is bounded by the smaller of the route request timeout
  and the remaining whole-analysis deadline.
- The provider output cap, retry count, retry backoff, requested tools, tool
  responses, unsupported requests, and failures are traced.
- Reaching the turn limit without valid structured evidence marks L1 degraded
  and retains the already-published deterministic fallback result.
- Tool errors or cap exhaustion MUST be recorded in the trace. If L1 fails but
  L0 has valid client-selected current-log evidence, L3/L4 MAY use those facts
  and MUST mark the result provenance to show L1 was attempted but not used.
  Partial tool context without valid structured evidence MUST NOT become policy
  input unless it is converted into source-grounded client evidence.
- L1 MUST analyze only the current log. It MUST NOT apply attempt history,
  compute recurrence, or choose the final action. Its confidence is semantic
  confidence in its current-attempt assessment, not an action score.

The prompt asks the model to use tools only for missing context, search for an
initiating mechanism rather than terminal tail noise, and cite retrieved
source lines. These are model instructions and measured behavioral-efficiency
signals, not client-enforced preconditions. No-new-evidence termination and a
forced final non-tool turn are possible future profile controls; they are not
implemented settings in `restart_agent_config.v1`.

If the provider does not support OpenAI-compatible tool calling, use a fallback
prompt containing L0 summary plus bounded head, anchor windows, and tail. The
trace MUST mark `context_mode: fallback_precurated`.

### Taxonomy Context For L1

The prompt MUST include the semantic vocabulary from `TAXONOMY.md`. It MAY
include additional reviewed examples from that vocabulary and the eval set.
Executable detector definitions remain in `PATTERN_REGISTRY.md`; L1 sees only
the observational registry matches already carried by the evidence view.

The model MUST treat registry matches as retrieval hints, not final answers.
Ordering, progress, and raw log windows decide how L1 structures current-log
evidence. History is applied later by the policy engine.

### L1 Structured Evidence

Before emitting final model JSON, the model MUST perform these logical steps
internally or in structured scratch fields:

1. List candidate failures.
2. For each traceback, identify its terminal exception and causal role.
3. Select the initiating primary failure; explain any choice after an earlier
   terminal exception.
4. Mark secondary, cascade, cleanup, and teardown failures.
5. Assess semantic mechanism and exactly two recovery claims: failure domain
   and retry outlook without workload change. Each claim has its own evidence
   status and confidence. Registry hints remain provisional retrieval aids
   rather than answers.
6. Propose diagnostic operation/mechanism/component/artifact fields; the client
   derives stable root and cascade fingerprints.
7. Cite evidence lines.
8. Emit concise current-log evidence and justification.

The final model JSON MUST NOT expose long chain-of-thought. It MUST expose
concise current-log evidence and justification. It MUST NOT include an action,
retry budget, history conclusion, user/not-user score, `decision_basis`, or
`evidence_coverage`.

L1 MUST validate required response structure and may make one bounded no-tool
contract-repair call when the first response is incomplete. L2 MUST then audit
evidence-line/quote grounding and causal-role credibility without judging the
semantic conclusion. The client MUST NOT synthesize missing model evidence or
justification and report the result as normal L1 output.

After L2, L3 MUST compute exact-job, exact-root history observations. L4 MUST
then select the versioned retry rule and allowed-retry count, evaluate whether
that budget is exhausted, choose `decision_basis`, and emit the final analyzer
output. The current boundary has no aggregate user/not-user score.

### Future Verification And Arbitration

Verifier and ensemble/arbitration behavior are post-MVP extensions recorded in
`STATUS.md`. They are not numbered pipeline stages and must receive dedicated
profiles, schemas, policy ownership, and eval qualification before promotion.

## Prompt And Client Contract

The LLM client targets an OpenAI-compatible chat endpoint:

```text
POST {base_url}/chat/completions
```

Requirements:

- When `tools_enabled=true`, advertise exactly the tools declared by the
  analysis profile and use `tool_choice=auto` when supported.
- Do not execute tool calls for tools outside the selected profile; reject and
  trace them instead.
- Apply provider-specific reasoning/thinking options from the analysis profile
  when supported, such as Qwen `enable_thinking`.
- Record both requested and resolved reasoning/thinking mode. If the provider
  does not support an explicit knob, record `unsupported` or `provider_default`
  instead of treating it as equivalent to enabled or disabled.
- Send the resolved profile's sampling settings. Qualification profiles should
  prefer `temperature=0` or the lowest provider-supported deterministic setting
  when reproducibility is the goal; the library fallback remains `0.2/0.7` for
  callers that do not supply a profile.
- Omit unsupported sampling parameters for providers that reject them.
- Set a request timeout. Default: 120 seconds.
- Enforce the configured absolute whole-analysis deadline. Default: 600
  seconds from analysis start. Clamp every provider request to the lesser of
  the route request timeout and remaining analysis budget.
- Retry only provider/transient failures that are safe to retry.
- Check the remaining analysis budget before every retry, model turn, tool
  call, and forced final response. Do not start work after the deadline.
- In `collect_all` mode, build L0A, Decision Evidence, and L0B once, query all
  declared model routes concurrently with bounded workers, and run L2-L4
  independently for each response. Return every route result without preference
  or arbitration. A route failure is isolated and produces a degraded per-route
  result while other routes continue. Publish the deterministic fallback before
  starting any route. At the whole-analysis deadline, return completed routes
  plus `deadline_exceeded` fallback results for unfinished routes without
  waiting for their worker threads.
- A future `priority_select` profile will query ordered model candidates
  concurrently within the configured decision window and select the
  highest-priority valid in-window response. Late, malformed, unsupported, or
  schema-invalid responses cannot win arbitration and remain traceable.
- Never log API keys.
- Save the analysis profile id, final prompt metadata, tool calls, tool
  results, model ID, and timing in the compact trace.
- Save a detailed interaction transcript artifact for the analyzer-visible
  LLM/tool exchange. It MUST include rendered chat messages, advertised tool
  schemas, request options, model id, provider route, timeout/deadline context,
  retry attempt number, raw visible model output, parsed tool requests, tool
  execution results, unsupported tool requests, provider errors, bad-gateway or
  throttling responses, token-limit/finish-reason details, JSON repair attempts,
  schema-validation failures, and arbitration outcomes. The transcript MUST NOT
  include API keys, auth headers, bearer tokens, or configured secrets.
- Do not patch final action policy into prompts. Prompt changes may improve
  evidence extraction, but any rule that affects final `STOP` / `RESTART`
  mapping MUST be represented in `POLICY.md`, `TAXONOMY.md`, or deterministic
  client code.
- Do not expose long chain-of-thought in final output or traces. Reasoning mode
  may change internal model behavior, but the analyzer stores only concise
  evidence, justification, provider options, and timing.

Determinism guarantee: L0 and the policy engine MUST be deterministic given the
same log, request, effective configuration, immutable `PriorAttemptView`, and
structured evidence. L1 tool-call
sequencing is not guaranteed to be reproducible across models, providers, or
model revisions, even at `temperature=0`; traces MUST preserve enough prompt,
tool, and model metadata to audit those differences.

The interaction transcript is the audit source for model behavior. The compact
trace summarizes it; it should not inline large prompts, raw tool windows, or
full model payloads unless the deployment intentionally chooses a single-file
trace mode. Any transcript truncation or missing payload MUST be recorded as an
observability anomaly.

The LLM client MUST parse final model JSON. It MUST repair malformed JSON once
by asking the same model to return only valid JSON.

Provider timeout, provider unavailability, token/output truncation, failed
forced final call, or failed repair call means raw L1 output is not trusted. If
valid L0/client evidence is available, the analyzer may still produce a normal
or degraded L0-driven result and MUST record that the model was attempted but
not used. If no valid L0/client evidence exists, the analyzer MUST return the
external output schema with `decision_basis=malformed_model_output`,
`result_quality=fallback_only`, and the corresponding trace anomaly.

### System Prompt Contract

The system prompt MUST remain a generic L1 contract. It MUST include these
clauses:

- Analyze one distributed training log and return the structured current-log
  evidence object defined by the supplied response schema.
- Separate observed mechanism, root-cause assessment, and operational recovery
  assessment.
- Identify the initiating failure using chronology and complete traceback
  context before classifying its domain or recovery semantics.
- Return exactly two recovery claims, `failure_domain` and
  `retry_outlook_without_workload_change`, each with its own value, evidence
  status, confidence, and claim-tagged citations; do not return a retry rule or
  action.
- Treat application, model/data/configuration, and workload-selected
  framework/library behavior as workload-domain when supported by the log.
- Do not infer ownership or persistence of mutable external resource state from
  the workload call stack alone.
- Evaluate recovery in the next NVRx cycle using the supplied restart context,
  including process recreation, normal restart delay, and possible changes to
  hardware allocation or mutable external-service state. A fixed resource
  request proves repeated selection, not persistence of conflicting mutable
  state; absent cleanup/release messages do not prove persistence.
- Assess whether the exact workload may recover through the declared restart
  transition without a workload change.
- Require affirmative evidence for persistent and transient conclusions.
  Repeated rendering or multi-rank fanout within one causal episode is not
  cross-attempt recurrence; when one log cannot distinguish persistence from
  retry recovery, preserve plausible or unknown semantics.
- Keep durable remediation and long-term preventive advice outside the
  current-attempt recovery assessment.
- Treat reporting components, call stacks, resource names, and diagnostic
  suggestions as context rather than proof of ownership, cause, transience, or
  persistence.
- Separate downstream cascades, wrappers, cleanup, and teardown from the
  initiating failure.
- Cite only line numbers and quotes retrieved from the supplied evidence bundle
  or log tools.
- Return final JSON only; do not expose long chain-of-thought.

Failure-family examples and provider-specific observations MUST NOT accumulate
in the static system prompt. They belong in typed L0 registries/taxonomy, eval
gold, or a separately versioned prompt experiment. Promotion of a specialized
prompt rule requires corpus evidence and an A/B result showing improvement
without regressions; a single analyzed log is insufficient.

The static prompt SHOULD use positive task and schema instructions instead of
enumerating absent inputs or prohibited output fields. The response schema,
client-side parsing, and L2 audit enforce the boundary. A concise negative
instruction may be restored only when corpus results show a recurring contract
violation that those mechanisms do not handle adequately.

### User Prompt Contract

The dynamic user prompt MUST carry only request-specific evidence, context, and
the structural response contract, while behavioral and policy semantics remain
canonical in the system prompt. It MUST include `response_schema`,
`decision_evidence`, `attempt_execution_context`,
`restart_environment_context`, and `evidence_bundle`.

The dynamic JSON MUST NOT duplicate the static task, behavioral instructions,
advertised tool names/schemas, or tool-loop limits. Provider tool definitions
are conveyed through the request's `tools` field. Tool-loop policy remains
trace/profile metadata enforced by the client.

The user prompt MUST NOT duplicate causal-analysis, persistence, recovery-path,
or output-behavior clauses from the system prompt. This
keeps policy wording single-sourced, reduces attention overhead on every tool
turn, and makes prompt revisions attributable in evaluation.

The user prompt MUST NOT include absolute paths, basenames, parent directory
names, eval case ids, observed label directories, or other path/run-name hints.

### Final Model Answer Contract

The final model message MUST be valid JSON matching the model evidence schema.
It MUST NOT contain Markdown fences, prose before/after JSON, multiple candidate
JSON objects, `decision_basis`, `decision`, or `evidence_coverage`.
The analyzer client MUST compute coverage, retry-policy state,
`decision_basis`, and `decision` using `POLICY.md` before returning final output.
