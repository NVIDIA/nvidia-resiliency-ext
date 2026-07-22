# Restart Agent

The NVRx Restart Agent analyzes one failed distributed-training attempt and
returns an auditable `RESTART` or `STOP` recommendation. Training logs are
large, interleaved, and full of downstream failures, so the design separates
deterministic evidence construction, model interpretation, grounding, history,
and action policy.

```text
current log
  -> L0A complete typed evidence
  -> DecisionEvidence
       -> AttemptRecord(progress + deterministic facts)
            +-> L3(PriorAttemptView) -> L4         deterministic fallback
            +-> L0B bounded model view -> L1 -> L2
                 -> add route-keyed enriched facts
                 -> L3(same PriorAttemptView) -> L4 enriched candidate
```

L0 records observed facts, preserves source references, and supplies the shared
progress and deterministic failure blocks of the current `AttemptRecord`. L1 describes the
failure and its recovery outlook but does not choose the action. L2 minimally
grounds model-selected evidence and supplies one route-keyed enriched failure
block; its broader credibility findings are advisory. L3 reads the selected
current block and compares it against exact-job,
same-root history and reports progress relations. L4 alone selects a versioned
retry rule and action. The deterministic fallback is ready independently of
model or endpoint latency.

The target `RestartAgentRuntime` owns a bounded in-memory `AttemptRecordStore`
for its current process lifetime. The same record type represents an attempt
while it is current and later when it appears in an immutable
`PriorAttemptView`; no history-record conversion occurs. History is enabled by
default with 10 attempts per exact `job_id` and 3000 records total;
configuration may disable it or change either bound. Library/unit tests can
seed, inspect, and clear this state in memory. The
CLI may read a test-only JSON-array fixture and explicitly export the resulting
records for scenario construction. It does not maintain history automatically.
MVP recurrence compares only deterministic blocks from prior records. Enriched
blocks remain observable for later policy work. MCP remains a later thin
transport adapter rather than the owner of history semantics.

The terminal implementation establishes feasibility and observability. It does
not yet establish corpus-level accuracy, a preferred model profile,
progressive-path latency, or production readiness. See [STATUS.md](STATUS.md) for the
current proof boundary.

## Read In This Order

1. [REQUIREMENTS.md](REQUIREMENTS.md) - problem, production constraints, and acceptance criteria.
2. [DESIGN.md](DESIGN.md) - canonical architecture, stage boundaries, and global invariants.
3. [EVIDENCE_BUNDLE.md](EVIDENCE_BUNDLE.md) - how raw text becomes L0A, `DecisionEvidence`, and L0B.
4. [POLICY.md](POLICY.md) - history observations, retry budgets, and STOP/RESTART rules.
5. [RUNTIME.md](RUNTIME.md) - runtime composition, attempt-record ownership, injection, and replay.
6. [SCHEMA.md](SCHEMA.md) - public and internal data contracts.

## Focused Reference

| Question | Document |
| --- | --- |
| Which deterministic patterns are recognized? | [PATTERN_REGISTRY.md](PATTERN_REGISTRY.md) |
| Which semantic classes and roles are canonical? | [TAXONOMY.md](TAXONOMY.md) |
| What can L1 inspect beyond its initial evidence? | [TOOLS.md](TOOLS.md) |
| Who owns configuration, runtime state, and history? | [RUNTIME.md](RUNTIME.md) |
| How are model routes and tunables reproduced? | [PROFILE.md](PROFILE.md) |
| What is the target progressive lifecycle? | [PROGRESSIVE.md](PROGRESSIVE.md) |
| Where is the product/eval ownership boundary? | [EVALUATION.md](EVALUATION.md) |
| What is implemented and what remains open? | [STATUS.md](STATUS.md) |
| Why were active architectural choices made? | [DECISIONS.md](DECISIONS.md) |

## Code Map

```text
src/nvidia_resiliency_ext/attribution/restart_agent/
  pipeline.py             small public RestartAgent facade
  preparation.py          shared L0 preparation and callback boundary
  single_run.py           one-route orchestration and deadline handling
  multi_route.py          parallel route fanout and collection
  decision_pipeline.py    L2/L3/L4 decision composition
  models.py               shared immutable stage/result contracts
  config.py               versioned route/profile wiring
  runtime.py              current injectable clock, sleeper, and executor boundaries
  l0/                     log assembly, DecisionEvidence, projection, registry
  l1/                     semantic contract, prompt, tools, provider adapter
  l2/                     source grounding, enriched failure identity, audit
  l3/                     cross-attempt history comparison
  l4/                     deterministic retry policy
  infrastructure/         log and artifact I/O
  observability/          per-stage trace rendering and envelope schemas
```

The companion eval harness source lives under `tools/restart_agent_eval/` in a
separate development change and does not ship in the NVRx package. It invokes
the product rather than implementing another analyzer.
