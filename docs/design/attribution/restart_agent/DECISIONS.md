# Active Design Decisions

This file explains why consequential architectural choices were made. It is not
a normative contract; the focused specifications own current behavior.
Superseded decisions remain in git history.

## D1: Use A Layered Decision Pipeline

- **Context:** A single model answer hides whether errors came from evidence
  selection, semantic reasoning, grounding, history, or action policy.
- **Decision:** Separate deterministic evidence, model interpretation,
  grounding, history comparison, and policy into L0A/L0B, L1, L2, L3, and L4.
- **Rejected alternative:** One end-to-end model prompt with a deterministic
  postprocessor that exposes only the final action.
- **Consequence:** Each stage has a typed boundary, trace, and independent KPI.
- **Revisit when:** Corpus results show a stage adds material latency or
  complexity without improving safety, accuracy, or diagnosis.

## D2: Keep L2 Grounding Non-Overriding

- **Context:** Strict semantic validation rejected credible model assessments
  when citations were nearby, incomplete, or semantically expressed.
- **Decision:** L2 mechanically grounds evidence and derives client identity,
  while broader credibility findings remain advisory and preserve raw L1 output.
- **Rejected alternative:** Rewrite or reject the model's semantic assessment
  whenever an audit heuristic disagrees.
- **Consequence:** Policy can require grounded facts without hiding what the
  model actually concluded.
- **Revisit when:** A corpus-validated transformation has explicit semantics,
  measurable safety benefit, and its own observable contract.

## D3: Derive History Identity In The Client

- **Context:** Model-authored class names and fingerprints vary in wording and
  granularity across models and repeated runs.
- **Decision:** L0 derives deterministic identity and L2 derives the grounded
  enriched equivalent. Identity consists of a normalized root and optional
  exact affected entity.
- **Rejected alternative:** Compare free-form L1 classes, summaries, or
  model-generated fingerprints directly.
- **Consequence:** L3 can perform deterministic recurrence comparison while L1
  semantic identity remains visible for analysis.
- **Revisit when:** A stronger structured runtime identity becomes available or
  corpus evidence justifies extending the client identity contract.

## D4: Introduce Prior Attempts Only At L3

- **Context:** Showing history to L1 would mix current-log interpretation with
  recurrence and could make model behavior depend on record ordering.
- **Decision:** L1 assesses only the current attempt. L3 receives the selected
  current failure facts and an immutable exact-job `PriorAttemptView`.
- **Rejected alternative:** Include prior logs or attempt summaries in the L1
  prompt and ask the model whether the issue recurred.
- **Consequence:** Current-attempt semantics and history policy can be evaluated
  independently.
- **Revisit when:** Deterministic history proves insufficient and an explicit,
  separately evaluated history-reasoning stage is proposed.

## D5: Keep Stateful Runtime Outside The Analysis Core

- **Context:** History, deadlines, route fanout, and record generations are
  invocation concerns; L0-L4 should remain replayable from explicit inputs.
- **Decision:** `RestartAgentRuntime` owns current-process state and orchestration.
  The pipeline consumes injected immutable inputs and produces invocation-owned
  outputs.
- **Rejected alternative:** Store history or caller-visible last-run state
  inside stage implementations or a module-global singleton.
- **Consequence:** Library, CLI, tests, and future service adapters exercise the
  same core with replaceable state and infrastructure.
- **Revisit when:** A distributed runtime requires a different store
  implementation, while preserving the same injected interfaces.

## D6: Use Deterministic Prior Facts For MVP History

- **Context:** Each route may produce a different enriched identity, while MVP
  has no canonical route-selection policy.
- **Decision:** Store route-keyed enriched facts for observation, but compare
  prior attempts using their deterministic L0 facts.
- **Rejected alternative:** Let completion order or an arbitrary preferred
  route determine the history identity.
- **Consequence:** Model timing and route availability cannot change recurrence
  policy.
- **Revisit when:** A production route-arbitration contract defines which
  enriched result becomes canonical.

## D7: Integrate Attrsvc Through The Library Boundary

- **Context:** The first integration needs the existing NVRx HTTP lifecycle but
  does not need another transport or the legacy analysis controller.
- **Decision:** Attrsvc constructs and invokes the runtime in process behind
  `ANALYSIS_BACKEND=lib`.
- **Rejected alternative:** Require a Restart Agent MCP service for the first
  product integration.
- **Consequence:** Attrsvc remains the transport boundary while the Restart
  Agent library owns analysis and current-process history.
- **Revisit when:** Independent scaling, isolation, deployment ownership, or
  cross-language access justifies a thin MCP adapter.

## D8: Use Score-Free Deterministic Retry Policy

- **Context:** User/not-user scores conflated attribution confidence with action
  and produced contradictory explanations and thresholds.
- **Decision:** L1 emits typed domain and recovery claims; L3 emits recurrence
  and progress; L4 applies ordered retry rules and budgets.
- **Rejected alternative:** Convert one model probability or score threshold
  directly into `STOP` or `RESTART`.
- **Consequence:** Every action is traceable to explicit semantic facts, history,
  and configured policy.
- **Revisit when:** Calibration evidence demonstrates a specific probabilistic
  policy improves outcomes and can remain independently observable.

## D9: Declare Workload-Managed Recovery

- **Context:** Behavior such as retry-then-skip is a workload capability that
  cannot be reliably inferred from generic failure text.
- **Decision:** Trusted configuration declares recovery capabilities; L4 applies
  one only when grounded classifier, behavior, and affected entity match.
- **Rejected alternative:** Ask L1 to invent or infer retry/skip support.
- **Consequence:** Special retry budgets are explicit, testable, and cannot
  silently spread to unrelated failures.
- **Revisit when:** A structured workload/runtime signal can declare the same
  capability more directly.

## D10: Publish The Deterministic Recommendation Before L1

- **Context:** Model and endpoint latency may delay semantic enrichment, while
  callers may need an earlier recommendation.
- **Decision:** Run L3/L4 from deterministic facts and publish that candidate
  before starting model routes.
- **Rejected alternative:** Wait for the preferred model and construct a deterministic result
  only after timeout or failure.
- **Consequence:** A recommendation exists independently of L1. A caller may
  act on it without closing the continuing Restart Agent analysis.
- **Revisit when:** A different precomputed candidate demonstrably provides
  better safety or pre-end semantic work proves necessary.

## D11: Keep Parallel Model Results Independent

- **Context:** Multiple routes help compare speed, reliability, and semantic
  quality, but no validated production arbitration policy exists.
- **Decision:** Build L0 once, execute routes concurrently, and return every
  route result independently.
- **Rejected alternative:** Vote, merge semantic fields, or select a winner
  implicitly in MVP.
- **Consequence:** Route behavior remains measurable without manufacturing
  consensus or allowing one failure to fail the batch.
- **Revisit when:** A reviewed priority or arbitration policy defines validity,
  cutoff, winner selection, and canonical history behavior.

## D12: Keep Root And Entity Retry Ledgers Independent

- **Context:** Exact affected entities improve recurrence precision, but entities
  may be absent or alternate across attempts with the same root. Selecting only
  one history scope could let those changes reset the effective retry budget.
- **Decision:** L3 always reports independent same-root and
  same-root-plus-entity no-advance streaks. L4 always evaluates the general
  same-root ceiling and concurrently evaluates any narrower selected-rule
  budget. A narrower budget may stop earlier but cannot extend the ceiling.
- **Rejected alternative:** Let the selected rule choose one exclusive history
  scope and ignore the other count.
- **Consequence:** Exact entities support workload-managed recovery without
  allowing entity or rule changes to create unbounded retries.
- **Revisit when:** A validated policy requires a specialized rule to exceed
  the general ceiling and provides an explicit independent safety bound.

## D13: Keep Scheduler Time Policy Outside Restart Policy

- **Context:** An application log may record a Slurm wall-time termination, but
  the scheduler enforces allocation lifetime regardless of the Restart Agent
  recommendation.
- **Decision:** L0 may preserve an explicit time-limit event as observed
  evidence. L4 gives it no special rule or retry-budget exemption.
- **Rejected alternative:** Infer that any `time limit` or `wall-time` text is
  an expected restartable termination.
- **Consequence:** Restart Agent does not duplicate scheduler policy, and
  incidental time-related text cannot bypass normal recurrence accounting.
- **Revisit when:** A caller provides a structured scheduler outcome with a
  product requirement for a distinct recommendation.
