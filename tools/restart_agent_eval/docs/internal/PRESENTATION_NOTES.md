# Restart Log Analyzer Presentation Notes

Status: non-normative source for a future presentation. This is not a product or
eval contract. Examples and metric tables explain the design; canonical field
names, schemas, defaults, and requirements remain in the linked product and eval
specifications.

Review [STAGE_INPUT_OUTPUT.md](STAGE_INPUT_OUTPUT.md) for the explicit L0-L4
runtime handoffs before applying the next stage-contract revision to the
presentation deck.

## Purpose

Explain the Restart Log Analyzer approach, why it is promising, what the prototype has
demonstrated, and what work remains before production qualification.

The presentation covers two related but distinct systems:

1. **Restart Log Analyzer product:** the runtime layered log analyzer that produces one
   grounded `STOP`/`RESTART` result.
2. **Eval/tuning harness:** the offline system that assembles candidate product
   profiles, runs them over a corpus, measures every layer, and recommends a
   reproducible profile for external approval and promotion.

The presentation should leave the audience with one accurate conclusion:

> Deterministic log context plus structured model reasoning and deterministic
> policy is a viable architecture for restart decisions. Its larger advantage
> over a coupled black-box analyzer is that every stage can be measured and an
> eval harness can optimize versioned bundle, prompt, model/tool, and policy
> profiles over a corpus before production promotion.

## Intended Audience

Primary audience: NVRx, fault-tolerance, training-runtime, and inference/model
platform engineers.

Secondary audience: technical leads deciding whether to invest in productizing
and operating this approach.

Audience assumptions:

- understands distributed training and restart cost;
- does not need implementation-level schema detail;
- needs to see why this is more reliable than sending a raw log to an LLM;
- will care about latency, false `STOP`, model variability, observability, and
  operational ownership.

## Narrative

1. Restart decisions are difficult because the initiating failure is buried in
   repeated multi-rank output and downstream cascades.
2. Establish two systems explicitly: **A. Tune** is the offline eval/tuning
   harness; **B. Act** is the production Restart Log Analyzer.
3. Explain B, the runtime product: L0-L4 divide evidence assembly, model
   semantics, grounding, history enrichment, and policy into observable stages.
4. Explain A, the harness: it varies declared tunables, runs the same product
   path over human-approved cases, scores stage and final outputs, and recommends
   a profile.
5. This separation addresses the black-box problem: a regression can be assigned
   to bundle, prompt/model, grounding, history, policy, endpoint, or harness.
6. Exploratory runs validate the product architecture and show why layered
   tuning is useful, but automated search, corpus coverage, progressive latency,
   and promotion remain incomplete.

## Deck Structure V4: Tune In A, Act In B

1. **Title and premise**: a tunable pipeline for restart decisions.
2. **The restart-decision problem**: noisy multi-rank logs, cascades, recovery,
   and the NVRx deadline.
3. **Two systems, one controlled loop**: A tunes and qualifies a profile; B
   executes the promoted profile; traces and outcomes flow back to A.

### B. Act: Restart Log Analyzer

4. **Fallback before model enrichment**: L0A freezes shared Decision Evidence;
   deterministic L3/L4 starts immediately while L0B fans out to all enabled L1
   model routes in parallel. Each route continues independently through L2-L4,
   and current `collect_all` mode retains every route result without arbitration.
5. **Progressive L0A lifecycle**: cycle start enables tail/precompute,
   `progressive_end` starts the decision window, post-end drain overlaps L0A,
   and log convergence releases final L0A assembly.
6. **L0A latency opportunity**: current one-time L0A measurements beside median
   L1 latency, with the explicit caveat that terminal runs do not measure
   post-end convergence or progressive parity.
7. **Registries give L0 generic structure**: generic failure roles and
   progress/checkpoint detectors produce typed observations and normalized
   occurrence groups, not literal signature-to-action mappings.
8. **L0A output**: `L0Bundle` retains complete structured, loss-accounted
   collections; `DecisionEvidence` selects canonical policy-relevant facts and
   references back to those collections.
9. **L0B output**: bounded, prioritized model-visible Decision Evidence plus
   selected L0A support; optional tools can retrieve retained evidence.
10. **L1 recovery-assessment contract**: mechanism/RCA plus two independently
    qualified claims: failure domain and retry outlook without workload change;
    L1
    does not choose `STOP` or `RESTART`.
11. **Prompt contract and tuning**: static generic rules plus a dynamic L0B
    message/schema/tool profile, evaluated on quality, compliance, efficiency,
    and unrelated holdouts.
12. **One product walkthrough**: CUDA graph-capture log through L0A/L0B, L1,
    L2, L3, L4, and final action.
13. **History resolves one-cycle ambiguity**: stable observational identity,
    exact-job/root recurrence, marker-compatible progress relations, and retry
    grace. `RestartAgentRuntime` owns the default-enabled 3,000-attempt-per-job
    in-memory store; library/unit tests seed and inspect that store directly.
    The CLI may explicitly import/export editable manual-test fixtures but does
    not maintain history automatically, while MCP remains a thin adapter.

### A. Tune: Eval And Tuning Harness

14. **Why a separate harness is required**: one `restart_agent.json`
    configuration can declare N model routes, product `collect_all` builds L0
    once and runs those routes concurrently, and the harness scores every
    independent result as a controlled, attributable experiment.
15. **Fast plus heavyweight production routing**: launch a predictable fast
    route and a preferred deeper-reasoning route together. Use the preferred
    result when ready before the NVRx deadline; otherwise use the fast result.
    A valid late preferred assessment cannot revise the closed-cycle action but
    becomes the semantic assessment retained for later history.
16. **Harness architecture and loop**: corpus/gold, candidate profiles, product
    execution, layer scoring, holdouts, recommendation, and external promotion.
17. **Tunables and results at every layer**: registry/bundle, prompt/model/tools,
    audit, history, and policy with their own outputs and metrics.
18. **Anti-pattern and regression controls**: PR lint, runtime audit, human gold,
    ablations, and unrelated holdouts.
19. **Model-route qualification**: semantic quality, behavioral efficiency,
    and endpoint reliability are separate axes; route outcome is their combined
    operational result.
20. **Exploratory evidence**: agreement does not imply operational fitness;
    compare quality, turns/tools, latency, tokens, and endpoint behavior.
21. **Productization path**: feasibility, corpus tuning, profile qualification,
    progressive replay, and production authority.

### System-Of-Systems Visual

```text
                  EVAL / TUNING HARNESS

 human gold + production outcomes + candidate generator
                         |
                         v
       versioned candidate product profile
       - L0 pattern/progress registry, bundle strategy, and budgets
       - L1 prompt, model, thinking, and tools
       - L2 grounding and advisory-audit configuration
       - L3 history parameters
       - L4 policy parameters
                         |
                         v
           RESTART LOG ANALYZER PRODUCT
       log -> L0 -> L1 -> L2 -> L3 -> L4 -> result
               |     |     |     |     |
               +-----+-----+-----+-----+
                         |
              stage outputs, traces, metrics
                         |
                         v
        score -> diagnose -> reject/retain/iterate
                         |
              holdout/progressive qualification
                         |
                         v
          profile recommendation -> human approval
```

The harness “constructs the product” by assembling and evaluating a versioned
profile over declared product components and tunables. It does not generate
arbitrary runtime code or mutate the deployed analyzer online. A finding that
requires a new algorithm or interface still becomes an engineering change and
PR before it enters candidate search.

## Detailed Content Bank

The following sections preserve material for the current deck and backup
slides. Their headings are topics rather than final slide numbering.

### Content: The Restart Decision Problem

**Message**

A failed large-scale training cycle needs a fast `RESTART` or `STOP` decision,
but logs rarely state that decision directly.

**Content**

- One initiating failure can produce thousands of repeated rank messages.
- CUDA, NCCL, framework, scheduler, and teardown errors may be consequences.
- Some apparent faults recover while training continues.
- Some workload/data failures repeat unchanged and waste another cycle.
- NVRx owns a bounded post-failure decision window; a late answer may be unused.

**Visual**

A timeline showing normal progress, first fault, multi-rank cascade, cycle end,
and the NVRx decision deadline.

**Avoid**

Do not frame every logged error as a failed workload. Progress after a warning or
fault is first-class evidence.

### Content: From Coupled Analyzer To Tunable Pipeline

**Message**

The major design goal is to replace opaque end-to-end behavior with a pipeline
whose context, reasoning, grounding, and policy can be tuned independently and
evaluated together.

**Content**

- Large logs contain repeated templates and interleaved ranks.
- Full-log context increases tokens and may exceed context limits.
- Missing context causes extra model/tool turns and latency.
- Different models can map the same ambiguous symptom to opposite actions.
- A model explanation can contradict its numeric score or recommendation.
- Endpoint latency and reliability are separate from reasoning quality.
- In a coupled system, a better or worse result does not clearly identify which
  component should change.
- A layered pipeline exposes stage inputs, outputs, traces, and metrics so the
  eval harness can compare controlled profile variants.

**Visual**

Contrast two paths:

```text
raw log -> model -> action

raw log -> deterministic evidence -> model semantics -> grounding -> policy
              ^                 versioned profile                 |
              |---------------------------------------------------|
                         eval, score, tune, qualify
```

**LogSage Context**

LogSage is valuable as a battle-hardened source of progress, pattern, and
operational lessons. The limitation motivating this design is not that LogSage
has no useful internals; it is that context construction, model behavior, and
action behavior are operationally coupled enough that regressions and tuning
ownership can be difficult to isolate. The Restart Log Analyzer makes those boundaries
explicit.

### Content: The Restart Log Analyzer Architecture

**Message**

Each stage has one job, one traceable output, and its own quality signals.

**Content**

```text
log + optional compact history
              |
              v
L0A complete deterministic evidence bundle
              |
              v
       shared Decision Evidence
              |
       +------+---------------------------+
       |                                  |
       v                                  v
L3 history -> L4 policy       L0B attention-efficient view
       |                                  |
       |                                  v
       |                     L1 semantics -> L2 grounding + identity + audit
       |                                  |
       |                                  v
       |                     L3 history -> L4 policy
       |                                  |
       +----------------+-----------------+
                        v
       best ready candidate before deadline
                        |
                        v
       STOP / RESTART + provenance + trace
```

- L0A finds progress, failure episodes, candidate anchors, excerpts, patterns,
  cascades, checkpoint facts, and assembly lossiness. L0B selects and renders
  the versioned, bounded view initially supplied to L1.
- Decision Evidence is built once from L0A and is byte-identical at the branch
  point. The fallback branch does not rescan the log or wait for L1.
- The enriched branch adds model semantics, grounding, identity, and an advisory audit to the same canonical
  facts. Its result is preferred only when it is usable before the caller
  deadline; otherwise the deterministic fallback remains available.
- A distributed mechanism incident may have one observer; a distributed
  fanout incident requires at least two distinct ranks. An ordinary single-rank
  failure remains an episode without a distributed incident.
- L1 identifies mechanism, root cause and its certainty, then returns exactly
  two recovery claims: failure domain and retry outlook without workload
  change. Each claim has its own evidence status, confidence, and cited support.
- L0B supplies deterministic restart assumptions: unchanged workload,
  recreated process state, normal restart delay, and hardware allocation or
  mutable external-service state that may change.
- L2 checks whether evidence is present and credible without replacing the
  raw model opinion. It may record an unsupported claim or suggested
  interpretation, but the suggestion remains unapplied.
- L3 compares compatible prior cycles and emits recurrence, prior-outcome, and
  history-completeness facts without choosing an action.
- L4 applies deterministic policy to unchanged L1 semantics plus L3 history,
  selecting a retry rule, applying its retry budget, and emitting the final
  action.

The deck must show this handoff explicitly:

```text
L1 semantic inputs
  mechanism + RCA
  RCA status + missing evidence
  failure domain {value, status, confidence}
  retry outlook without workload change {value, status, confidence}
        |
        v
L2 grounded identity/advisory audit + L3 observational history facts
        |
        v
L4 deterministic retry rule + budget -> STOP / RESTART
```

L1 does not recommend an action. L3 reports history observations without
choosing a threshold or proving persistence. L4 consumes the unchanged L1
assessment and L3 observations. Per-claim confidence remains display-only;
the two claim values and statuses are policy inputs and their use must
be visible in the L4 trace.
An L4-qualified history rule may still support a later STOP.
Infrastructure is a RESTART prior, not an action lock: L4 may retain
infrastructure attribution while choosing STOP after a qualifying history
rule exhausts the selected budget.

The audience-facing L1 slide should frame this as a **recovery assessment**, not
as a user-failure score or a hidden action recommendation:

| L1 output group | Contents | Use |
| --- | --- | --- |
| Observed failure | mechanism, anchor, causal role, related failures, citations | L2 audits citations and derives stable client identity. |
| Root-cause assessment | summary, evidence status, plausible causes, missing evidence | The raw L1 RCA remains visible and unchanged. |
| Recovery assessment | failure domain and retry outlook without workload change, each with value and evidence status | L4 policy inputs remain the L1 values; L2 concerns are shown separately, and L3 contributes compatible recurrence/progress facts. |
| Calibration fields | independent confidence for each recovery claim | Display, compare, and calibrate in the harness; confidence is not used by L4. |

Only L4 maps the unchanged L1 assessment plus history to a retry rule, budget
state, and `STOP`/`RESTART`.

**Declared restart context is L1 input, not L1 output.** It is supplied by NVRx
and the analyzer and is traced with the request:

| Context fact | Meaning |
| --- | --- |
| Workload and configuration unchanged | The next attempt runs the same intended code, configuration, and input. |
| Process-local state recreated | In-memory and process-local state starts fresh. |
| Normal restart delay applies | Teardown, cleanup, or temporary contention may have time to clear. |
| Hardware allocation may change | Placement can change, but reallocation is not guaranteed. |
| Mutable external-service state may change | Storage, network, or service health may recover independently. |

`May change` is not evidence that state will change or that retry will recover.
L1 uses the context to assess the next unchanged attempt; L4 still owns the
action.

**The declared read-only tool profile names concrete operations:**

| Tool | Purpose | Default advertisement |
| --- | --- | --- |
| `overview()` | Return log structure, metadata, and progress summary. | on for tool-enabled routes |
| `grep_log(pattern)` | Find matching lines across the raw log. | on for tool-enabled routes |
| `read_window(start, end)` | Return a bounded original-log excerpt. | on for tool-enabled routes |
| `get_evidence_objects(ids)` | Fetch retained typed L0A evidence by reference. | off; experimental route-profile option |

The trace records tool name, arguments, returned lines or objects, latency,
tokens, and whether the call supplied new evidence. Tools fill specific L0B
gaps; they are not the primary analysis path.

L0B gives L1 a current-attempt execution summary derived from L0A: successful runtime,
iteration range, latest completed checkpoint, replay distance from that
checkpoint, and later-progress observations after earlier fault-like events in
the current log.
Later compatible progress proves job continuation, not recovery of the same
rank, storage path, network, or component. Replay distance explains restart
cost and how quickly the same data/phase may recur; neither fact is an action by
itself.

**Attention-Efficiency Ownership**

Attention efficiency is an objective of the L0-to-L1 interface, with distinct
stage responsibilities:

- **L0A objective:** produce a complete structured, traceable, loss-accounted
  representation of the observed log.
- **L0B objective:** maximize decision-relevant evidence per unit of model
  context. Produce a bounded, prioritized, deduplicated, progress-aware view
  that makes the initiating failure and its surrounding state easy to find.
- **L1 objective:** produce an accurate structured semantic assessment from that
  bundle with minimal additional interaction. Use tools when evidence is
  genuinely missing or ambiguous, not to rediscover supplied context.
- **End-to-end objective:** produce a grounded, policy-usable decision within the
  NVRx decision window.

Each L0 sub-stage has a quality scorecard and an operational scorecard. L0A
quality uses human gold to measure primary evidence coverage, selected-primary
accuracy, progress/checkpoint detection, typed-event detection, episode/incident
construction, later-progress facts, and lossiness correctness. L0A operations
measure assembly latency, scan volume/throughput, bundle size/shape, caps, and
deterministic replay.

L0 also owns the deterministic fallback root fingerprint. Its runtime KPI is
fingerprint availability/source/history-readiness; its gold/corpus KPIs are
fingerprint accuracy, false-merge rate, and false-split rate.

L0B quality uses human gold and controlled model comparisons to measure required
evidence retention, primary retention from L0A, support-context coverage,
compaction safety, first-turn completion, and tool-context yield. L0B operations
measure projection latency, model-view size, budget utilization,
selection/compaction counts, and projection integrity/hash. L1 has direct
measurements such as semantic quality, model turns, tool calls,
duplicate/no-new-context calls, model latency, tokens, and contract repairs.

Some L1 measurements are downstream diagnostics for L0, not automatic L0
failures:

- a tool call that finds new, relevant, decision-changing evidence suggests an
  L0 bundle gap;
- the same missing-context request across models strengthens that diagnosis;
- a tool call that rereads bundled evidence suggests model/profile/tool-use
  inefficiency;
- high decision latency without extra context requests may be model or endpoint
  behavior rather than bundle quality.

The eval harness should report these dimensions separately rather than collapse
them into one attention-efficiency score before deployment tradeoffs are known.

**Layer Metrics Reference**

This is speaker/backup-slide material. The main architecture slide should show
only two or three headline metrics per layer.

Availability labels:

- `current`: emitted by the current product/eval artifacts;
- `derived`: calculated from current artifacts;
- `gold`: requires human-approved labels and corpus aggregation;
- `progressive`: requires progressive replay or production lifecycle data;
- `experiment`: requires an ablation or controlled comparison.

**Measurement Maturity**

The metrics should be introduced in stages rather than treated as one immediate
implementation checklist.

1. **Mechanical metrics available now**
   - stage and model-call latency;
   - bundle/request size and shape;
   - total model turns, tool-driven turns, contract-repair turns, and tool calls;
   - tool-returned lines absent from or already present in the initial prompt;
   - duplicate/no-new-context calls and tool truncation/errors;
   - parsing, contract repair, retries, timeouts, token usage, and endpoint
     errors;
   - exact/rendered/nearby/unresolved citation grounding;
   - final primary/support citations that depend on tool-only lines;
   - raw L1 assessment, L3 history facts, L4 mapping reason, decision basis,
     and final action.
2. **Attribution proxies requiring interpretation**
   - new tool lines are a possible L0 gap, but are not necessarily relevant;
   - final citation of a tool-only line is stronger evidence of missing initial
     context, but does not prove the action changed;
   - the same missing-context request across models strengthens L0 ownership;
   - duplicate reads by one model strengthen model/profile ownership.
3. **Quality metrics requiring human gold**
   - progress/failure-episode correctness;
   - primary/RCA/domain/recurrence/persistence/precondition/action accuracy;
   - unsupported claims, false `STOP`, audit false positives/negatives, and
     confidence calibration.
4. **Causal metrics requiring controlled experiments**
   - whether a tool result changed RCA, policy assessment, or action;
   - whether a smaller/larger bundle caused a quality or latency change;
   - whether thinking, tools, or model choice caused the observed difference.
5. **Production metrics requiring progressive/longitudinal data**
   - post-end latency and decision-window hit rate;
   - recurrence precision, fingerprint false merges/splits, progress-relation
     accuracy, and shadow-mode false-`STOP` outcomes.

The initial KPI implementation should start with category 1 and preserve enough
artifacts to support categories 2-5 later. It should not manufacture one numeric
attention-efficiency score from incomplete proxies.

#### L0: Evidence Assembly And Projection

L0 is measured on speed, bundle quality, and whether it gives L1 sufficient
high-value context.

| Metric | What it measures | Availability |
| --- | --- | --- |
| L0A quality | Gold primary coverage/selection, progress/checkpoint detection, typed events, episodes/incidents, later-progress facts, lossiness correctness | eval gold |
| L0A operations | Assembly latency, scan throughput, source/bundle size, shape, caps, deterministic replay | current |
| L0B quality | Gold evidence retention, primary retention, supporting context, compaction safety, first-turn/tool-context yield | eval gold + corpus |
| L0B operations | Projection latency, view size, budget use, selection/compaction counts, projection integrity/hash | current |
| Scan throughput | Source bytes or lines divided by L0 build time | derived |
| Source size | Lines and bytes processed; denominator for scale comparisons | current |
| Bundle shape | Normalized occurrence groups, windows, anchors, episodes, and cascades produced | current |
| Fanout compaction | Exact aggregate event counts versus bounded stored representatives | current |
| Model-facing size | L0B characters/estimated evidence tokens and provider-reported first-turn input tokens | current |
| Replay consistency | Whether every panel target received the byte-identical L0A bundle and L0B view | current |
| Primary-in-bundle rate | Fraction of gold/model-selected primary lines represented anywhere in L0 | derived/gold |
| Primary-in-excerpt rate | Fraction of primary lines visible in the initial raw excerpts | derived/gold |
| Anchor excerpt coverage | Candidate anchors lacking a prompt-facing excerpt | current |
| Progress coverage | Whether progress-before/after-fault and checkpoint facts were available | current |
| Progress accuracy | Precision/recall of progress, checkpoint, recovery, and terminal episode state | gold |
| Recovered-noise promotion rate | Recovered or `progressed_after` episodes consuming top anchor/excerpt slots | derived/gold |
| Lossiness | Candidates/windows/patterns omitted or truncated by selection budgets | current/derived |
| Shared missing-context rate | Fraction of models requesting the same relevant evidence absent from the prompt | derived |
| Tool-only evidence dependency | Final primary/support lines available only through tools | current |
| Decision-changing missing evidence | Whether adding tool-only evidence changes RCA, policy assessment, or action | experiment |

Headline L0 metrics for the main slide: build latency, primary-in-excerpt rate,
and shared relevant missing-context rate.

Tool count by itself is not an L0 metric. It becomes an L0 diagnostic only after
the retrieved context is classified as new, relevant, and missing from the
initial bundle.

**Worked KPI example for the deck:** gold marks checkpoint decode line 12083 as
the initiating failure, iteration line 11800 as last progress, and cleanup line
12135 as teardown. L0A passes primary coverage if 12083 is a candidate and
passes selected-primary accuracy only if its singleton is 12083. L0B passes
required-evidence coverage when 11800/12000/12083/12135 are all represented in
the initial view; compacting 19 rank copies to one representative plus an exact
count passes compaction safety. Show operational values separately, for example
`9 -> 5` windows, `24.8k` estimated tokens, `78%` budget use, and `0.04 s`
projection latency, labeled illustrative.

#### L1: Semantic Model Analysis

L1/model-route qualification uses three independent axes plus their combination:

- **Semantic quality:** whether the delivered response correctly explains the
  primary, mechanism, recovery semantics, and uncertainty.
- **Behavioral efficiency:** how many turns, tools, repairs, and tokens the
  model needed after context delivery.
- **Endpoint reliability:** whether provider attempts succeeded without
  retries, timeouts, gateway failures, or other service errors.
- **Route outcome:** whether the complete route produced a usable enriched
  result within the measured decision window, or required fallback.

Semantic quality is conditional on a delivered response. An endpoint-only
failure is `not_observed`, not a wrong answer. End-to-end L1 latency belongs to
route outcome because it contains model behavior, local tools, transport,
queueing, and service time.

Provider-reported timing is optional L1 operational detail, not a separate
quality axis. When response headers provide it, show the proxy's downstream
LLM API span plus proxy pre-processing, post-processing, and message-copy
spans. The downstream span is not model compute and may contain backend
transport, queueing, prefill, decode, and response delivery. Omit this detail
when the provider does not report it; never derive missing components from
client wall time.

| Metric | What it measures | Availability |
| --- | --- | --- |
| Root-cause accuracy | Human score for initiating mechanism and RCA summary | gold |
| Primary-anchor accuracy | Exact or approved-nearby primary line/failure episode | gold |
| Domain accuracy | Correct `workload`, `infrastructure`, or `unknown` assessment | gold |
| Retry-outlook accuracy | Correct `cannot_recover`, `may_recover`, or `unknown` assessment under the declared restart transition | gold |
| Recovery-claim status accuracy | Correct `established_by_current_log`, `supported_but_unconfirmed`, `hypothesis_only`, or `unknown` status for each claim | gold |
| Cascade/role accuracy | Correct related failures and initiating/cascade/teardown roles | gold |
| Unsupported-claim rate | Model claims not supported by the log or permitted context | gold |
| Confidence calibration | Whether stated confidence matches empirical correctness | gold, corpus only |
| Structured-output usability | Parsed, required fields present, and usable by downstream stages | current |
| First-turn completion rate | Fraction producing usable evidence without tools or repair | derived |
| Model turns | Initial request plus all conversational continuations | current |
| Tool-driven turns | Continuations caused by executed tool requests | current |
| Contract-repair turns | Continuations requested for malformed required structure | current |
| Tool calls | Number and names of read-only tools requested | current |
| Duplicate/no-new-context calls | Tool calls that reread prompt evidence or add no new line context | current |
| New-context yield | New relevant lines returned per tool call | current plus review relevance |
| Final-answer tool dependency | Whether final primary or cited evidence depends on tool-only lines | current |
| L1 latency | Initial request through accepted structured result, including tools | current |
| Model/tool latency split | Time in provider calls versus local tool execution | current |
| Provider-reported L1 timing | Optional downstream-LLM-API and proxy processing spans from response headers; not model compute | current when provider reports it |
| Input/output/total tokens | Provider usage by call and accumulated across turns | current when provider reports it |
| Contract-repair rate | Runs requiring another response because output structure was malformed | current |
| Token-limit rate | Runs ending or degrading because a context/output limit was reached | current |
| Context-budget adjustment | Calls whose effective output cap was reduced to fit estimated input plus safety reserve | current |
| Unsupported-tool rate | Requests for tools outside the declared profile | current |

Headline L1 metrics for the main slide: semantic quality, behavioral efficiency,
endpoint reliability, and route outcome. The detailed metrics below explain
each headline without collapsing them into one score.

Retries, timeouts, gateways, and provider errors are shown with L1 runtime
health but attributed to endpoint reliability, not semantic quality or
behavioral efficiency.
Distinct conversational turns are reported separately from provider attempts:
a retry adds an attempt but does not create a new turn. Timeout, HTTP-error,
and provider-error counts can overlap on the same failed attempt and are not
summed into one endpoint-issue number.

**Worked KPI example for the deck:** for the same log, gold expects
`checkpoint_load / metadata_deserialization / UnicodeDecodeError`, a
`may_recover` or `unknown` retry outlook, and supported or unknown status for
each claim.
A model selecting 12083 with all mechanism fields passes semantic primary and
mechanism accuracy. `python_user_exception@12083` passes primary selection but
is mechanism-underspecified. Selecting cleanup line 12135 fails semantic
primary accuracy. Compare a one-turn/no-tool 8-second result with a four-turn
35-second result separately from correctness.

Use two audience-facing L1 KPI slides. The first defines the four independent
views: semantic quality, behavioral efficiency, endpoint reliability, and
their combined route outcome. The second applies those views to the checkpoint
example: compare a fast correct route, a slower equally correct route, and an
endpoint-only failure whose semantic quality is `not_observed` and whose route
falls back deterministically.

#### L2: Evidence Grounding, Identity, And Audit

L2 is measured on citation/claim credibility and audit correctness. It is not
measured on whether it agrees with the model's semantic conclusion.

| Metric | What it measures | Availability |
| --- | --- | --- |
| Audit completion/status | Whether the audit ran and returned clean/findings status | current |
| Primary/policy availability | Whether required L1 claims were present to audit | current |
| Supporting-evidence visibility | Cited support lines visible in prompt/tool context versus total cited | current |
| Exact raw citation count | Quotes matching original log text exactly | current |
| Exact rendered citation count | Quotes matching model-visible rendered/truncated text | current |
| Nearby-resolution count | Credible citations resolved within the permitted nearby range | current |
| Ungrounded citation count | Citations not resolvable to visible source evidence | current |
| Findings by reason | Missing line, mismatched quote, prohibited action field, or role inconsistency | current |
| Mechanical reference-repair count | Deterministic citation repairs applied without changing semantics | current |
| Policy audit observation count | Semantic concerns and suggestions emitted with `applied=false` | current |
| Operation/artifact comparison strength | Exact physical unit, same logical artifact with shard uncertainty, different artifact, or unknown | current |
| Audit false-positive rate | Credible L1 claims incorrectly flagged | gold, audit-focused cases |
| Audit false-negative rate | Demonstrably ungrounded claims accepted | gold, audit-focused cases |
| Identity completeness | Whether stable observed exception, callsite, stack, path, and position fields were derived | current |
| Enriched root fingerprint readiness | Whether L2 produced a policy-active key, its source, and whether L3 can consume it | current |
| L0 relationship | Whether the L2 key equals the L0 fallback key; diagnostic when primaries differ | current |
| Root fingerprint accuracy | Whether the L2 key matches reviewed gold | gold |
| Root fingerprint false merges/splits | Distinct failures merged or equivalent failures split by L2 identity | corpus gold |
| Cross-model identity agreement | L2 root-fingerprint agreement for models selecting the same primary | derived |
| L2 latency | Time required for local grounding and identity derivation | current |

Headline L2 metrics for the main slide: ungrounded citation rate, audit
false-positive rate, and L2 latency.

Provider timing is not an L2 KPI. L2 is local grounding work after the model
response and has only its own audit latency.

#### L3: History Enrichment

L3 is measured on whether compatible prior-cycle evidence is matched correctly.
It emits history facts and never chooses `STOP` or `RESTART`.

The state boundary is explicit: `RestartAgentRuntime` owns current-process
history, takes one immutable in-memory view before fallback/model branches run, and
upserts the selected current-attempt record afterward. History is enabled by
default with 3,000 entries per exact job. Product configuration may disable or
resize it. Library/unit tests use the in-memory seed/inspect/clear seam. The CLI
may explicitly round-trip editable JSON-array fixtures for manual scenarios;
MCP adds transport only.

**Presentation example: why history helps**

A single port-bind or checkpoint checksum/read failure is ambiguous. A port may
still be held briefly after a rapid restart; a checkpoint read may fail because
of transient I/O or persisted corruption. The first compatible occurrence
should therefore remain restartable unless the current cycle independently
proves persistence.

The history path is:

1. Ground the earliest actionable primary failure.
2. Consolidate duplicate serialized, inner-cause, and outer-wrapper exception
   renderings into one episode and choose its causal observed exception anchor.
3. Derive the policy-active `root_fingerprint` from registry-stable fields or
   that observed exception plus a normalized stable signal. Do not encode
   model-inferred ownership or an unconfirmed cause.
4. Remove volatile occurrence data such as rank, node, GPU, PID, timestamp,
   source line, retry counter, and memory address.
5. Compare only distinct earlier cycles with exact job and root-fingerprint
   equality.
6. Compare only like-for-like failure-iteration, completed-step, or checkpoint
   markers and emit `advanced`, `same`, `regressed`, or `unknown` plus delta.
7. Preserve each attempt's absolute progress summary: observed versus absent or
   unknown training/checkpoint progress, first/last iteration and checkpoint
   markers with span/count, and whether failure occurred before or after
   observed training progress.
8. Preserve exact failure/data/artifact matches and consecutive same-root
   no-observed-advance counts for L4. L3 does not choose the required count.

The two progress views are intentionally separate. Relative progress says
whether the current cycle advanced farther than a comparable prior cycle.
Absolute attempt progress says whether a cycle failed early or after doing
observable work. L4 receives both; the MVP policy uses relative advancement and
the no-observed-advance streak while absolute progress remains an explicit,
measurable input for a later versioned policy rule.

Examples:

- **Port bind:** cycle 1 reports `address already in use` and remains
  restartable. A later compatible cycle with the same grounded root and a
  comparable marker that does not advance is persistence evidence; repeated
  lines inside one traceback are not. Missing or incompatible markers produce
  `unknown`, not no-observed-advance evidence.
- **Checksum/read:** cycle 1 cannot distinguish transient I/O from corruption.
  Repetition of the same root with the same stable artifact/read position is
  stronger corruption evidence. A different artifact or failure position must
  remain distinct.

The product also emits experimental identities for corpus evaluation. The
family identity combines L1 operation and mechanism with the observed exception.
The concrete identity adds observed callsite/stack plus a grounded artifact path
and named failure position; structured fields are serialized as canonical
sorted-key JSON and SHA-256 hashed. `client_concrete` excludes L1 wording and is
derived only from the grounded primary and deterministic source context. These
identities remain `policy_active=false` until false-merge/false-split behavior is
qualified and an explicit policy/profile version activates them.

| Metric | What it measures | Availability |
| --- | --- | --- |
| History availability | Whether eligible prior attempts were available | current |
| Same-job/exact-root count | Distinct prior attempts matching the history identity | current |
| Progress relation accuracy | Correct `advanced`, `same`, `regressed`, or `unknown` relation and delta | L3 gold |
| Attempt-progress accuracy | Correct early/after-progress/unknown classification and marker/checkpoint summary | L3 gold |
| Exact-position accuracy | Correct failure-iteration, data-position, and artifact matches | L3 gold |
| No-observed-advance streak accuracy | Correct consecutive same-root streak; L3 applies no policy threshold | L3 gold |
| Family/concrete match precision | Whether recurrence matches the same underlying failure | gold, multi-cycle cases |
| Fingerprint false-merge/split rate | Different failures merged or same failures separated by identity | gold, corpus/multi-cycle |
| Late-result incorporation | Late prior-cycle analysis correctly updates next-cycle history | progressive/multi-cycle |
| History completeness | Whether missing or partial history is explicitly represented | current |
| Store behavior | Enabled state, configured bound, evictions, seed provenance, and idempotent replacement | runtime/state tests |
| Fixture round-trip | Exported manual-test records re-import identically in stable job/cycle order | CLI/state tests |
| L3 latency | Time for ordered history lookup and comparison | current |

Headline L3 metrics: root-match precision, progress-relation accuracy, exact-
position accuracy, history completeness, and L3 latency.

#### L4: Policy Decision

L4 is measured on deterministic policy correctness and final-result usability.

| Metric | What it measures | Availability |
| --- | --- | --- |
| Mapping determinism | Same L1/L2/L3 inputs always produce the same action | current/test |
| Mapping reason coverage | Every final action records an explicit deterministic reason | current |
| Action accuracy | Final `STOP`/`RESTART` membership in the approved action set | gold |
| Decision-basis accuracy | Correct reason for action, not merely the correct binary result | gold |
| `STOP` precision | Fraction of emitted `STOP` results that truly require intervention | gold/shadow outcomes |
| False-`STOP` rate | Restartable/recoverable cases incorrectly stopped | gold/shadow outcomes |
| Ambiguous-case acceptance | Final action remains within the approved action set | gold |
| Retry-grace correctness | Recovery-capable cases receive the intended retry allowance | gold, multi-cycle cases |
| Result quality | Normal, degraded, fallback, late, or unusable result classification | current |
| NVRx eligibility | Whether provenance/quality permits NVRx to consume the result | current |
| L4 latency | Time for deterministic policy mapping | current |

Headline L4 metrics: action accuracy, `STOP` precision, and deadline success.

#### End-To-End Product

These metrics evaluate the whole pipeline and should not be assigned to one
layer without diagnosis.

| Metric | What it measures | Availability |
| --- | --- | --- |
| Final RCA/action quality | Product result after L2/L3/L4 compared with human gold | gold |
| Terminal request-to-result latency | Offline/manual full-pipeline latency | current |
| Post-`progressive_end` p50/p90/p99 | Production decision-gate latency | progressive |
| Decision-window hit rate | Fraction ready before the NVRx-owned deadline | progressive |
| Analyzer fallback rate | Missing, late, malformed, or unusable results requiring NVRx fallback | progressive/current by mode |
| Endpoint reliability | Retry, timeout, gateway/provider error, and failed-call latency rates | current |
| Profile reproducibility | Same product/profile versions reproduce the declared prompt/tool/policy shape | future profile qualification |

Headline end-to-end metrics: final action quality, post-end p90/p99, and
decision-window hit rate.

**Visual**

Use the pipeline as the main deck diagram. Under each layer show three small
labels: function, trace, metrics.

### Content: Registry And Prompt Boundaries

**Message**

The registry structures evidence for L0; the prompt defines how L1 reasons
about that evidence. Neither owns the final action.

**Pattern/progress registry**

- is a versioned L0 input containing generic deterministic detectors;
- recognizes progress, checkpoints, observed exceptions, diagnostic advice,
  teardown, cancellation, and other line roles;
- supports pattern deduplication, candidate anchors, and failure episodes;
- must not accumulate case paths, ranks, timestamps, or literal
  error-to-action mappings;
- does not determine root cause or `STOP`/`RESTART`.

**Slide-ready registry example**

Start with four representative source lines:

```text
1000  [rank 7] iteration 418 completed
1012  [rank 7] RuntimeError: CUDA out of memory
1013  [rank 9] RuntimeError: CUDA out of memory
1030  [rank 7] destroy_process_group() called during shutdown
```

Generic detectors assign structural roles rather than actions:

| Detector | Generic shape | Structural output |
| --- | --- | --- |
| `iteration_progress` | `iteration <integer> completed` | progress marker: iteration 418 at line 1000 |
| `cuda_oom` | `CUDA out of memory` | observed-failure candidates at lines 1012 and 1013 |
| `process_group_teardown` | `destroy_process_group` | teardown/cascade marker at line 1030 |

L0 can then construct three complementary evidence views over the same source
lines:

| Structure | What it captures | Question answered |
| --- | --- | --- |
| `NormalizedOccurrenceGroup` | Repeated normalized text shape | How often and where did this observation appear? |
| `FailureEpisode` | Local chronology around a failure candidate | What happened before and after the candidate? |
| `DistributedIncident` | Distributed mechanism or same-attempt rank fanout | Is this one distributed event rendered by multiple observers? |

```text
NormalizedOccurrenceGroup
  shape: cuda_out_of_memory
  first_line: 1012
  count: 2
  observed_ranks: [7, 9]

FailureEpisode
  last_progress_before: iteration 418 at line 1000
  primary_candidate: observed failure at line 1012
  repeated_rank_copy: line 1013
  first_teardown_line: 1030
  later_compatible_progress: not observed

DistributedIncident
  incident_kind: distributed_fanout
  mechanism: cuda_out_of_memory
  primary_observed_line: 1012
  member_event_lines: [1012, 1013]
  observed_ranks: [7, 9]
```

The incident exists here because two distinct ranks rendered the same failure
within one attempt. A single-rank OOM would still create an occurrence group
and failure episode, but it would not create a `distributed_fanout` incident.
Conversely, an inherently distributed mechanism such as a collective timeout
may create a `distributed_mechanism` incident even when only one observer is
visible.

The registry has answered: "What kind of evidence is each line?" It has not
answered whether the OOM is transient, likely to repeat, or should produce
`STOP`. L1 supplies that semantic assessment, L2 audits its evidence, L3 adds
compatible history, and L4 owns the action.

**Prompt**

- is a versioned L1 semantic contract applied to the evidence bundle;
- requests mechanism, root cause status/missing evidence, domain, retry outlook
  without workload change, each claim's evidence status and confidence, and
  related failure roles;
- instructs the model to distinguish initiating evidence, cascades, teardown,
  and diagnostic advice;
- does not apply history or emit the final action;
- must remain generic rather than encode expected answers for individual log
  signatures.

The production request uses two messages plus optional provider tool schemas:

1. **Static system contract:** `SYSTEM_PROMPT` from `prompts.py`. It is stable
   across logs at one prompt version and defines task boundaries, causal rules,
   recovery concepts, evidence discipline, uncertainty, and interpretation of
   execution and restart facts. It stays generic and does not encode named
   failure cases or expected actions.
2. **Per-attempt user payload:** `_initial_user_message(...)` from `llm.py`. It
   is rebuilt for each log as a facts-only JSON object with exactly five
   top-level sections: `response_schema`, `decision_evidence`,
   `attempt_execution_context`, `restart_environment_context`, and
   `evidence_bundle`.
3. **Optional `tools` request field:** provider function schemas for the tools
   advertised by the selected route profile. Tool-loop limits and route settings
   remain client/profile and trace metadata; they are not repeated in the user
   evidence payload.

“Dynamic prompt” therefore means the per-attempt user evidence request; it is
not a second policy prompt. Stable semantic instructions belong in the system
contract, current-log facts belong in the user payload, and provider tool
capabilities travel in the separate `tools` field.

```text
log + registry + L0A configuration -> complete L0A evidence bundle
L0A bundle + L0B projection profile -> typed model-facing view
L0B view + prompt + model/tool profile -> L1 semantics
L1 semantics + L2 grounding/identity/audit + L3 history -> L4 policy action
```

The main deck uses a dedicated **Registries** slide, separate **L0A output** and
**L0B output** slides, and a dedicated **Prompt contract and tuning** slide.
The registry slide should not imply that registries and prompts are one
combined stage.

### Content: Prompt Contract And Tuning

**Message**

The prompt is a versioned L1 semantic contract and an offline-tuned product
input. It asks for observed failure, RCA, and current-attempt recovery
assessment; it is not an action policy or a case-specific answer key.

**Static system prompt**

- single-sources generic causal and recovery-assessment semantics;
- defines initiating evidence, cascades, teardown, diagnostics, uncertainty,
  root-cause status, domain, retry outlook without workload change, and the
  independent evidence status and confidence of each claim;
- explains how to interpret supplied execution and restart facts without
  treating progress as proof of transience or a possible restart transition as
  proof of recovery;
- remains generic: it does not name workload frameworks, error signatures, or
  expected actions for individual cases.

**Dynamic user message**

- supplies the exact response schema;
- carries `decision_evidence`, current-attempt execution facts, declared restart
  environment facts, and the bounded `evidence_bundle`;
- contains data only, with no duplicate behavioral instructions, tool-loop
  configuration, or route metadata.

The response schema has no final action fields. If a model emits unexpected
fields, the client preserves the raw response for observability and audits or
ignores the extras rather than relying on a negative prompt prohibition.

**Provider tool field**

- advertises the selected read-only function schemas separately from the user
  message;
- may be omitted entirely for a no-tools route;
- is versioned by the route profile so production and evaluation use the same
  capability shape.

**Harness tuning loop**

- compare baseline and candidate prompt versions over human-reviewed cases;
- score primary/RCA quality, recovery-assessment field accuracy and calibration,
  unsupported claims, contract compliance, first-turn completion, tool use,
  latency, and tokens;
- run affected-family tests and unrelated holdouts;
- promote only a reviewed, versioned prompt revision.

**Anti-patterns**

- case paths, exact ranks, test names, or expected actions;
- literal signature-to-answer instructions;
- repeating the full policy in both system and dynamic prompts;
- treating model consensus as gold.

### Presentation Scope: Analysis Profile

Keep the analysis-profile definition in `STAGE_INPUT_OUTPUT.md` as engineering
reference material, but do not add a standalone Analysis Profile slide to the
PowerPoint deck. The deck may say that the harness qualifies and promotes a
versioned profile, but it should not enumerate the full profile configuration
schema. The audience-facing story should remain focused on stage inputs,
outputs, tunables, and measurements.

### Content: Evidence Bundle And Progressive Operation

**Message**

Context quality and latency are addressed together by assembling evidence while
the training cycle is still running.

**Content**

- NVRx signals cycle start so L0A can tail and precompute.
- L0A builds an overview across the observed log: progress/checkpoint segments,
  repeated patterns, candidate faults, recoveries, cascades, and termination.
- Each candidate fault is connected to the nearest compatible progress before
  and after it; this is safer than treating the last error as the root cause.
- Faults followed by compatible progress or explicit recovery remain in the
  structured overview as `progressed_after` or `recovered`, but normally do not
  consume top candidate/excerpt budget.
- Raw excerpts are selected around the highest-value candidate anchors and
  failure episodes, not simply around the point where progress stopped.
- Excerpt boundaries are relaxed enough to preserve coherent surrounding
  context such as the complete traceback or related event sequence.
- A terminal candidate is typically recent progress followed by a coherent
  exception/failure episode, downstream termination or cancellation evidence,
  and no later compatible progress.
- Diagnostic advice is retained as context but not promoted as a primary fault.
- At cycle end, the analyzer continues ingesting the final shared-log tail.
- The production latency gate begins after `progressive_end`, not at cycle start.
- `progressive_end` signals workload-cycle completion, not shared-log
  completeness. At large scale, multi-rank output has been observed to continue
  arriving for roughly 10-20 seconds.
- The primary failure may arrive in the first few seconds of that drain, but the
  analyzer cannot assume it has. It must monitor file growth and update terminal
  candidates under a bounded post-end observation strategy.
- Evidence convergence means the available bytes required to finalize L0A have
  arrived. Only then does the analyzer freeze Decision Evidence, start the
  deterministic L3/L4 fallback, and launch the L0B/L1 enriched path.
- Progressive mode therefore shifts precomputable L0A work earlier; it does not
  remove log-convergence latency. Measure workload end, evidence convergence,
  and decision readiness as distinct times.

**Visual**

```text
cycle start       training         progressive_end  evidence converges  result
    |---------- L0A precompute ------------|<-- log-drain / L0A overlap -->|
                                           |<------ decision window ------->|
```

Add a small evidence-bundle inset containing progress summary, first failure
episode, bounded excerpt, cascades, and lossiness.

Also show the different treatment of noisy and terminal episodes:

```text
progress -> warning/fault -> later progress
             `progressed_after`: summarized background context

progress -> coherent fault -> teardown/cancel -> no later progress
             terminal candidate: prioritized anchor + bounded excerpt
```

**Avoid**

Do not describe this as finding the exact line where progress stopped. Rank
interleaving, buffering, and asynchronous reporting make that boundary
approximate; L0 uses compatible progress before/after and preserves the raw
line evidence.

**Evidence Size And Behavioral Efficiency**

The complete L0A bundle and L0B model-facing view are different typed artifacts.
L0A may retain many normalized occurrence groups and windows for trace/debug, fallback,
grounding, and optional tools. L0B serializes the bounded subset used in the
initial request; L1 consumes it without independently rebuilding context.

The primary concern is not token cost. It is whether the model can find and
reason about the causal evidence efficiently. An unnecessarily large bundle can
increase request/prefill latency, dilute attention with irrelevant candidates,
make primary/cascade selection less stable, and encourage additional tool
searches even when the required evidence was technically present.

Deduplication addresses repeated copies of the same normalized event. For
distributed rank fanout, L0 computes exact aggregate counts first, then retains
bounded first/last representatives for excerpts and downstream processing. It
does not solve a log containing many different warnings or failure-looking
lines. Progress-aware outcome classification and relevance-ranked
candidate/excerpt selection are what keep distinct but recovered or unrelated
noise from dominating the model-facing view.

Input-token use is approximately:

```text
tokens(model request) = tokenize(
    system prompt
  + instructions/output contract
  + L0B model-facing view
  + advertised tool schemas
  + prior assistant/tool messages on later turns
)
```

Source-log bytes and full-bundle bytes therefore do not map directly to input
tokens. The tokenizer is model/provider-specific, and structured JSON plus log
text may tokenize differently from prose.

One exploratory CUDA panel illustrates the reduction and remaining pressure:

- source log: about 5.4 MB and 50,296 lines;
- complete stored L0 bundle: about 1.7 MB;
- first request payload: about 149 KB;
- reported first-request input tokens: about 35K-52K across model routes.

Current prompt construction bounds repeated structures and raw context: it
includes at most 30 normalized occurrence groups, 20 registry groups, 20 cascades, 16
candidate anchors, 10 episode/summary records, and four prompt context windows.
Each prompt window is bounded to 240 lines, 50,000 characters, and 360
characters per line. These limits prevent linear growth with repeated log size,
but they are not a tokenizer-aware total prompt budget.

Noise can still create pressure when it produces many unique patterns, occupies
high-signal/anchor slots, or causes tool calls. Tool results are appended to the
conversation, and every continuation resends the accumulated conversation. In
the same panel, a one-turn model used about 46K total tokens, while a nine-turn
tool-heavy run accumulated about 536K tokens even though each individual call
fit its context window.

The product now supports a declared model context window and safety reserve. It
estimates the complete request before every model turn, including accumulated
assistant/tool messages, and lowers that turn's effective output cap when the
configured cap would not fit. The trace records the estimated input, configured
and effective output caps, and whether adjustment occurred. This prevents a
known invalid request shape; eval must still qualify estimator conservatism and
whether the remaining output budget is sufficient. Prompt size, adjustment
rate, and token-limit events remain explicit tuning signals.

The correct bundle size must be established empirically. Eval should compare
core, medium, and expanded bundle profiles using the same logs and model:

- semantic/root-cause and action quality;
- first-turn completion rate;
- tool calls and extra model turns;
- request and total decision latency;
- whether tools found genuinely missing evidence or reread bundled context.

A smaller bundle is better only while it preserves enough context to avoid
wrong answers or compensating tool calls.

**Exploratory Stage-Latency Evidence**

Three terminal/manual runs provide an initial estimate of work that
progressive execution could shift out of the post-end decision path:

| Log family | One-time L0A assembly | Median L1 across five routes | Median L2 | Recorded policy/history |
| --- | ---: | ---: | ---: | ---: |
| CUDA code | 4.6 sec | 31.4 sec | 0.19 sec | <0.01 sec |
| Bad token | 8.2 sec | 26.0 sec | 0.29 sec | <0.01 sec |
| Checkpoint | 23.5 sec | 24.9 sec | 0.53 sec | about 0.01 sec |

An early NCCL-timeout replay separately exposed rank-fanout scaling. The
89,630-line log originally took about 93 seconds to build a 13.5 MB bundle.
Exact-count/bounded-representative compaction and indexed episode processing
reduced the same local replay to about 14 seconds and roughly 0.5-0.7 MB while
preserving 12,288 aggregate registry events and one coherent timeout episode.
The rank reports are fanout of that one distributed operation, not independent
recurrence or persistence evidence. This single replay is a regression datum,
not a latency percentile.

These measurements demonstrate overlap opportunity, not progressive production
latency. The runs did not exercise `progressive_start`/`progressive_end`, log
convergence, or post-end parity. Progressive replay must measure how much L0 is
actually complete before cycle end and how much work remains after late rank
output arrives.

### Content: Model And Endpoint Selection

**Message**

Choose a model route using three independent axes, then evaluate their combined
route outcome; do not hide materially different failure modes inside one score.

**Semantic quality**

- RCA/mechanism and operational semantics versus human gold;
- action membership for unambiguous and approved-ambiguous cases;
- confidence calibration, grounding, unsupported claims, and contract
  compliance.

**Behavioral efficiency**

- first-turn completion;
- necessary versus unnecessary tools;
- duplicate/no-new-context calls;
- model turns and input/output token use;
- whether tool context changed primary, attribution, or policy assessment.

**Endpoint reliability**

- successful and failed provider attempts;
- timeout, retry, rate-limit, gateway, and provider-error rates;
- availability, empty service responses, and failed-call latency.

**Route outcome**

- model-enriched versus deterministic-fallback contribution;
- usable result and NVRx eligibility;
- p50/p90/p99 end-to-end latency and decision-deadline success.

Semantic quality and behavioral efficiency describe observed model behavior
under a declared profile. Endpoint reliability is a provider/service property.
A sound model behind an unreliable endpoint should not be labeled inaccurate.
Route outcome combines the three views for production gating without replacing
their raw measurements with one opaque blended score.

### Content: What The Exploratory Runs Taught Us

**Message**

The architecture improved through general lessons from different failure
families, not by encoding individual answers.

**Examples**

| Failure family | Observation | Design consequence |
| --- | --- | --- |
| Injected/device-side CUDA failure | The symptom alone supported both `STOP` and `RESTART`; injected context changed the answer | Preserve ambiguity and use recurrence/history when the current log cannot prove persistence |
| Numeric instability / bad token | Deep prior progress, the first exception, and no later progress marker observed were more useful than a late generic error | Make progress and failure episodes central to L0 |
| Checkpoint load failure | A decode/read exception could mean corrupted checkpoint data or a transient read failure | L0 must not make semantic short-circuit decisions; L1/history preserve ambiguity |
| CUDA graph capture code violation | Framework diagnostic boilerplate distracted models from the actual prohibited operation | Mark debugging advice as diagnostic context and ask L1 for domain plus the two recovery concepts |
| DataLoader bus error after long progress | The application log suggested shared memory, while external context implicated intermittent storage/network access | Preserve conditional diagnostics as hypotheses, report current-attempt progress and checkpoint replay distance, and require both workload and cannot-recover claims to be directly established before first-cycle STOP |

**Visual**

Four compact before/after examples, with the generalized design lesson more
prominent than the literal error text.

**Prompt And Semantic-Contract Tuning**

Prompt tuning is a first-class L1/profile activity. It is not model-weight
fine-tuning and should not encode the desired answer for one error string. The
prompt defines the semantic task and the boundaries between model reasoning and
deterministic policy.

The current system contract has five important responsibilities:

1. **Task boundary:** return the structured current-log evidence object defined
   by the supplied schema. Final action fields are absent from that schema and
   remain client-owned.
2. **Semantic decomposition:** separate observed mechanism/root cause from
   failure domain and retry outlook without workload change; qualify each
   claim independently.
3. **Causal discipline:** prefer the initiating terminal exception over repeated
   cascades, cleanup, finalizer, or teardown symptoms.
4. **Evidence discipline:** require affirmative log evidence for a transient
   alternative and distinguish diagnostic advice from an observed fault.
5. **Output contract:** emit one primary, bounded related failures, confidence,
   rationale, and grounded evidence in a stable schema.

The per-attempt user message is intentionally not another instruction layer. It
contains only:

1. `response_schema`: the exact structured response shape;
2. `decision_evidence`: canonical policy-relevant L0 facts and references;
3. `attempt_execution_context`: current-log progress, checkpoint, position, and
   comparable-operation facts;
4. `restart_environment_context`: declared facts about what restart may change;
5. `evidence_bundle`: the bounded model-visible evidence selected by L0B.

Optional tool schemas are sent in the provider request's separate `tools` field.
Tool-loop limits, route identity, and request budgets remain client/profile and
trace metadata.

The CUDA graph capture case illustrates why this matters. A generic CUDA error
can sound like infrastructure, and a framework stack can make ownership appear
ambiguous. The tuned contract instead asks:

- what operation actually failed;
- whether the behavior belongs to the application/model/data/config or its
  selected framework/library stack;
- whether the exact unchanged workload is likely to repeat on healthy
  infrastructure;
- whether the log contains affirmative retry/recovery evidence;
- whether nearby CUDA/PyTorch text is merely debugging advice.

That framing led models to identify a CPU transfer inside CUDA graph capture as
a workload-domain behavior that the unchanged workload could not recover from,
with both claims directly established by the current log. L4 then mapped that
assessment to `STOP`. The prompt did not need a special policy for the literal
CUDA error string.

Prompt changes require the same discipline as L0 changes:

- state a generic hypothesis;
- version the prompt/profile and use the same version in eval and production;
- compare raw L1 semantics before looking at L4 action;
- test affected families and unrelated holdouts;
- measure semantic quality and behavioral efficiency; report endpoint
  reliability and route outcome separately so prompt effects are not confused
  with service behavior;
- reject a change that fixes one case by overgeneralizing or suppressing valid
  ambiguity elsewhere.

Prompt tuning and L0 tuning interact. Better evidence may make an existing
prompt succeed; better instructions may let the model use an existing bundle
without tools. Eval should change one dimension at a time when causal attribution
matters.

**Anti-Patterns And Enforcement**

The model may violate explicit prompt instructions. Therefore prompt text is not
the only control plane:

- static lint and PR review catch case-specific registry/prompt content;
- L1 contract validation catches prohibited fields and malformed output;
- deterministic line roles prevent diagnostic advice from silently becoming
  trusted root evidence;
- L2 audits citations, causal roles, and credibility;
- L4 ignores model-authored actions and records deterministic mapping;
- corpus/holdout eval detects semantic overfitting that static checks cannot.

Examples from exploratory work include removing an injected-test-specific
pattern, replacing specific CUDA failure patterns with generic observed
exception extraction, and removing an overly broad generic Python-exception
shortcut that promoted downstream checkpoint symptoms. These were architecture
corrections, not evidence that the registry should accumulate every newly seen
error.

Use three complementary gates:

1. **PR gate:** Is the prompt/pattern generic, correctly layered, and supported
   by positive/negative tests?
2. **Runtime audit:** Did the model obey the contract and ground its evidence?
3. **Corpus gate:** Did the change improve the intended family without harming
   ambiguous and unrelated holdouts?

### Content: Evidence That The Premise Works

**Message**

With improved context and semantics, multiple model families can converge on the
same initiating failure and operational conclusion, while the trace exposes
their cost differences.

**Exploratory example**

For a deterministic CUDA graph capture violation:

- five model targets selected the same initiating source line;
- all five returned workload + likely repeat + change required;
- deterministic L4 mapped each assessment to `STOP`;
- CUDA/PyTorch debugging advice remained diagnostic context;
- the client-derived concrete identity converged across all five runs;
- model behavior still varied from one turn with no tools to many turns with
  repeated context and substantially higher token use.

This is feasibility evidence, not an accuracy benchmark. The case did not have
a human-approved corpus label when first reviewed.

**Visual**

A narrow comparison table:

```text
model | primary agreement | raw assessment | action | turns/tools | latency
```

Follow with a callout: semantic agreement does not imply equal operational
fitness.

### Content: What Is Proven And What Is Not

**Message**

The architecture is validated; production readiness is not.

**Established**

- deterministic compression of large repeated logs into inspectable evidence;
- shared byte-identical evidence for fair model comparison;
- useful structured RCA and operational assessments;
- explicit separation of raw model opinion, grounding, and policy;
- detailed traceability of prompts, tools, retries, tokens, and decisions;
- a practical path toward deterministic recurrence identity.

**Not established**

- representative root-cause or decision accuracy;
- false-`STOP` rate and confidence calibration;
- preferred model, endpoint, prompt, thinking mode, or tool profile;
- fingerprint false-merge and false-split behavior;
- post-`progressive_end` p50/p90/p99 latency;
- production endpoint capacity and service integration;
- safe arbitration among multiple models.

**Visual**

Two-column boundary: “Feasibility demonstrated” and “Qualification required.”

### Content: Harness Loop And Productization Path

**Message**

The next phase is an eval-driven optimization loop, not simply adding more
patterns or manually changing prompts one case at a time.

**Content**

1. Build a representative, human-approved corpus.
2. Score L0, L1, L2, L3, and L4 independently.
3. Assign every failure to bundle, model/profile, grounding, policy, endpoint,
   or harness ownership before changing code.
4. Prefer generic structural improvements and run holdout regressions.
5. Compare semantic quality, behavioral efficiency, endpoint reliability, and
   route outcome.
6. Implement versioned profiles and reproducible promotion/rollback.
7. Add progressive replay, service integration, and production deadline tests.
8. Use shadow mode to measure false `STOP` before granting action authority.

**Automated Offline Tuning Loop**

```text
human-approved corpus + production outcomes
                  |
                  v
candidate versioned profiles
  - L0 pattern/progress registry and bundle selection/budgets
  - L1 prompt/schema/few-shot version
  - model, thinking, tools, and endpoint controls
  - L3 history parameters
  - L4 policy parameters
                  |
                  v
run identical product path in eval
                  |
                  v
layer metrics + final quality + latency/reliability
                  |
                  v
reject / retain / generate next candidates
                  |
                  v
holdout and progressive qualification
                  |
                  v
profile recommendation -> human approval -> production promotion
```

“Self-tuning” means automated offline candidate search, scoring, regression, and
profile recommendation. It does not mean that the production analyzer rewrites
its prompt or policy online, learns from unreviewed model consensus, or promotes
itself without human-approved gold and deployment controls.

The optimizer should use constrained selection rather than one blended score:

1. satisfy semantic quality and false-`STOP` thresholds;
2. satisfy endpoint-reliability thresholds;
3. satisfy route-outcome gates for NVRx usability and progressive deadline;
4. among qualifying profiles, improve first-turn completion, tool efficiency,
   and resource use;
5. reject candidates that regress protected failure families or holdouts.

The current harness provides one-log review, shared-bundle model panels, traces,
and initial scoring. For a panel it invokes product `collect_all` once with one
`restart_agent.json`: product prepares L0A, Decision Evidence, and L0B once,
runs N enabled model routes concurrently, and returns every independent L1-L4
result without selecting a winner. Each route may independently configure its
model, endpoint, credential reference, tool advertisement, reasoning controls,
and request limits. The harness then creates per-model reviews and the panel
summary. Automated
candidate generation/search, complete aggregate ranking, holdout orchestration,
progressive replay, and profile promotion are intended next capabilities.

### Content: Fast Result, Preferred Result

**Message**

Parallel routes are a latency/quality strategy, not a voting scheme.

**Production example**

```text
shared L0 evidence
      |
      +--> fast profile ----------------> usable result at t=8s
      |     one turn, no tools                    |
      |                                           +--> deadline candidate
      |
      +--> preferred heavyweight ------> enriched result at t=42s
            deeper reasoning + tools               |
                                                   +--> replaces fast result
                                                        if before deadline

NVRx deadline at t=60s: choose the highest-priority usable result already ready.
```

If the heavyweight result arrives at `t=75s`, NVRx does not revise the closed
cycle. The history store preserves two facts separately:

- `action_result_used`: the fast route result that NVRx acted on at `t=60s`;
- `preferred_history_assessment`: the valid heavyweight assessment that arrived
  later and should inform the next cycle.

The late assessment may improve mechanism, RCA, and recovery semantics, but it
cannot change the deterministic client fingerprint used to match compatible
failures. A malformed, ungrounded, or degraded late result does not supersede a
usable fast assessment. Every readiness, selection, and history-supersession
event is traced.

**Visual**

A staged path:

```text
feasibility -> corpus tuning -> profile qualification -> progressive replay
            -> shadow mode -> production authority
```

## Evidence Bank

Retain these facts for later slide construction. Verify numbers against saved
panel artifacts before placing them in the final deck.

### CUDA Graph Capture Exploratory Panel

- Five model targets agreed on the primary line and `STOP` action.
- All five raw L1 assessments identified workload domain and an unchanged
  workload that could not recover; L4 did not manufacture the semantic
  agreement.
- The client concrete fingerprint was identical across all five targets even
  though model-authored fine classes and family identities differed.
- One model completed in one call without tools; another used nine model calls
  and eight tools, mostly rereading existing context.
- Terminal L1 latency and total token use differed substantially by model.
- The run was terminal/manual, so it did not measure the production
  post-`progressive_end` gate.

### Numeric Instability Exploratory Panel

- Better progress metadata and a coherent failure episode reduced tool use.
- The important context was substantial prior progress, the first exception,
  and lack of forward progress afterward.
- This motivated progress metadata such as iteration/checkpoint state and job
  scale, while keeping error signatures generic.

### Checkpoint Exploratory Cases

- Models initially selected generic downstream Python/file symptoms.
- Complete checkpoint context improved identification of the checkpoint
  operation and serialization/deserialization mechanism.
- A single read failure remained legitimately ambiguous between persistent data
  corruption and retryable I/O behavior.
- Stable read position, artifact identity, and recurrence may later help history
  distinguish those cases.

## Presentation Principles

- Lead with the operational problem and architecture, not model brands.
- Use one concrete walkthrough; keep other cases as summarized lessons.
- Separate observed evidence from claims of accuracy.
- Show raw L1 assessment, L3 history facts, and final L4 action separately.
- Treat tool calls as inspectable optimization opportunities, not automatic
  failures.
- Avoid implying that L0 is a rule-based replacement for semantic analysis.
- Avoid implying that any current model is selected for production.
- Keep implementation schemas, complete taxonomies, and API payloads in backup
  slides only.

### Self-Contained Stage Sections

Each numbered stage section must be understandable without a presenter filling
in missing definitions. A stage breaker is followed by enough slides to answer:

1. What enters the stage?
2. What does the stage do, and what does it explicitly not own?
3. What typed object or decision leaves the stage?
4. What does the behavior look like on a concrete log example?
5. Which quality and operational metrics judge that stage?

The main deck therefore uses this sequence after the L0 walkthrough:

- **L1:** input/work/output boundary, two recovery concepts, checkpoint decode
  example, prompt tuning, and semantic/behavior/endpoint KPIs.
- **L2:** evidence-audit boundary, raw L1 versus unapplied audit suggestion, and
  audit-quality KPIs.
- **L3:** recurrence examples, deterministic fingerprint construction, and
  history matching/availability KPIs.
- **L4:** policy boundary, workload-attribution-versus-current-action example,
  and action/basis/false-`STOP` KPIs.

Labels such as `recurrence`, `grounded`, `fingerprint`, and `policy basis` must
not appear without a nearby plain-language definition or worked value. The
slides should use prose to state the governing rule and examples to show why it
exists; terse contract tables remain supporting material, not the explanation.

## Stage Deep-Dive And Backup Slides

The first five topics are part of the main explanatory path; the remaining
topics may move to backup depending on meeting length.

1. L0 evidence bundle anatomy.
2. L1 structured output contract and worked recovery assessment.
3. L2 evidence-audit examples and forgiving nearby-line behavior.
4. L3 history recurrence, fingerprint construction, and compatibility.
5. L4 operational mapping with attribution/action separation.
6. Model-panel efficiency and endpoint metrics.
7. Eval gold-label and holdout workflow.
8. Progressive API lifecycle and observability.
9. Experimental client-derived failure identity.

## Open Decisions

- Audience level: engineering deep dive, leadership review, or mixed.
- Target duration and expected discussion time.
- Whether to name model endpoints or anonymize them as model A-E.
- Which single log walkthrough is most representative.
- Whether the deck seeks approval for continued investment, design alignment,
  or a production milestone.
- Which quantitative results are mature enough to show outside the immediate
  working group.
