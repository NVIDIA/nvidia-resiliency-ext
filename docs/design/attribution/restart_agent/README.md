# Restart Agent

The NVRx Restart Agent analyzes one failed distributed-training attempt and
returns an auditable `RESTART` or `STOP` recommendation. Training logs are
large, interleaved, and full of downstream failures, so the design separates
deterministic evidence construction, model interpretation, grounding, history,
and action policy.

L0A reconstructs a logical failure narrative from overlapping per-rank
observations. It keeps the initiating primary strict and nullable; when that
cause is absent, it may separately retain one canonical terminal failure
surface. `L0A.md` defines both selection rules.

```text
current log
  -> L0A complete typed evidence
  -> DecisionEvidence
       -> AttemptRecord(progress + deterministic facts)
            +-> L3(compare deterministic) -> L4   deterministic recommendation
            +-> L0B context management
                 -> failure narrative + compact facts + selected support
                 -> Initial Model Evidence View -> L1 -> L2
                 -> add route-keyed primary and observation tracks
                 -> L3(compare all tracks independently)
                 -> L4 select primary, observation, or deterministic path
```

L0 records observed facts, preserves source references, and supplies the shared
progress and deterministic failure blocks of the current `AttemptRecord`. L1
describes the failure and its recovery outlook but does not choose the action.
L2 minimally grounds model-selected evidence and supplies independent
route-primary and route-observation blocks; its broader credibility findings
are advisory. Root identity and the weaker observation-only identity remain
separate. L3 compares deterministic, primary, and observation tracks
independently against like-kind exact-job history and reports shared progress
relations. L4 alone selects an evidence path, a versioned
retry rule and action. The deterministic recommendation is ready independently of
model or endpoint latency. Model enrichment is the product default, while the
deterministic recommendation remains available first and independently.
Configuration may explicitly select deterministic-only execution.

Observation-only identity says that the same visible failure surface was seen;
it does not claim the same initiating cause. Generic policy therefore uses it
only to preserve a root-independent retry recommendation and relies on same-job
progress rather than a fabricated root match.
The three-track composition contract is implemented. See [STATUS.md](STATUS.md)
for the remaining production-qualification gaps.

`RestartAgentRuntime` owns a bounded in-memory `AttemptRecordStore`
for its current process lifetime. The same record type represents an attempt
while it is current and later when it appears in an immutable
`PriorAttemptView`; no history-record conversion occurs. History is enabled by
default; `CONFIGURATION.md` owns its per-job and total bounds. Library/unit
tests can seed, inspect, and clear this state in memory. The
CLI may read a test-only JSON-array fixture and explicitly export the resulting
records for scenario construction. It does not maintain history automatically.
MCP remains a later thin transport adapter rather than the owner of history
semantics.

The terminal-first implementation establishes feasibility and observability.
Progressive pre-end polling is available as a disabled-by-default optimization.
Neither path yet establishes corpus-level accuracy, a preferred model-route
configuration, target-scale latency, or production readiness. See
[STATUS.md](STATUS.md) for the current proof boundary.

## Read In This Order

1. [REQUIREMENTS.md](REQUIREMENTS.md) - problem, production constraints, and acceptance criteria.
2. [DESIGN.md](DESIGN.md) - canonical architecture, stage boundaries, and global invariants.
3. [L0A.md](L0A.md) - complete deterministic evidence, Decision Evidence, progress, and deterministic identity.
4. [L0B.md](L0B.md) - deterministic failure narrative and bounded,
   attention-efficient initial model evidence.
5. [L1.md](L1.md) - semantic model task, prompts, route execution, and usability.
6. [L2.md](L2.md) - source grounding, enriched identity, and non-overriding audit.
7. [L3.md](L3.md) - same-job recurrence and progress comparison.
8. [L4.md](L4.md) - ordered retry rules, budgets, policy contexts, and action.
9. [RUNTIME.md](RUNTIME.md) - runtime composition, attempt-record ownership, injection, and replay.
10. [CONFIGURATION.md](CONFIGURATION.md) - complete product configuration field reference and resolution.
11. [SCHEMA.md](SCHEMA.md) - exact public and internal serialization reference.
12. [ATTRSVC_INTEGRATION.md](ATTRSVC_INTEGRATION.md) - NVRx/attrsvc composition and lifecycle.

## Focused Reference

| Question | Document |
| --- | --- |
| Which deterministic patterns are active, and what may they establish? | [PATTERN_REGISTRY.md](PATTERN_REGISTRY.md) |
| How do structural, semantic, history, and policy labels differ? | [TAXONOMY.md](TAXONOMY.md) |
| What can L1 inspect beyond its initial evidence? | [TOOLS.md](TOOLS.md) |
| Who owns configuration, runtime state, and history? | [RUNTIME.md](RUNTIME.md) |
| How does optional progressive execution work? | [PROGRESSIVE.md](PROGRESSIVE.md) |
| How does NVRx reach the Restart Agent through attrsvc? | [ATTRSVC_INTEGRATION.md](ATTRSVC_INTEGRATION.md) |
| Where is the product/eval ownership boundary? | [EVALUATION.md](EVALUATION.md) |
| What is implemented and what remains open? | [STATUS.md](STATUS.md) |
| Why were consequential alternatives accepted or rejected? | [DECISIONS.md](DECISIONS.md) |

## Code Map

```text
src/nvidia_resiliency_ext/attribution/restart_agent/
  pipeline.py             small public RestartAgent facade
  agent_runtime.py        stateful invocation and attempt-record owner
  preparation.py          shared L0 preparation and callback boundary
  single_run.py           one-route orchestration and deadline handling
  multi_route.py          parallel route fanout and collection
  decision_pipeline.py    L2/L3/L4 decision composition
  models.py               shared immutable stage/result contracts
  config.py               versioned routing and runtime configuration
  runtime.py              current injectable clock, sleeper, and executor boundaries
  l0/                     log assembly, DecisionEvidence, projection, registry
  l1/                     semantic contract, prompt, tools, provider adapter
  l2/                     source grounding, typed history identity, audit
  l3/                     cross-attempt history comparison
  l4/                     deterministic retry policy
  infrastructure/         log and artifact I/O
  observability/          per-stage trace rendering and envelope schemas
```

The companion eval harness source lives under `tools/restart_agent_eval/` in a
separate development change and does not ship in the NVRx package. It invokes
the product rather than implementing another analyzer.
