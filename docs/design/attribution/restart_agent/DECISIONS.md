# Active Design Decisions

This file records choices that still shape the implementation. Superseded
discussion remains available in git history; it is not part of the current
navigation or contract.

| Decision | Current choice | Consequence |
| --- | --- | --- |
| Analysis shape | Root-cause-first, layered pipeline | Deterministic evidence, semantic interpretation, grounding, history, and policy are independently observable and testable. |
| Default action | Bias to `RESTART` | `STOP` requires a qualified unrecoverable predicate or exhausted same-root/no-advance retry budget. |
| L0 authority | Evidence only | Registry matches and deterministic primary selection cannot directly establish semantic user/workload failure. |
| L1 authority | Semantic recovery assessment, not action | The model reports mechanism, domain, unchanged-retry outlook, per-claim evidence status/confidence, and citations. L3 owns cross-attempt persistence; L4 owns action. |
| L2 authority | Minimal grounding plus advisory credibility audit | L2 may omit demonstrably invalid references from its grounded projection and derive identity, but it preserves the raw L1 response and does not rewrite L1 semantic opinion. |
| Attempt-record lifecycle | One neutral record type for current and prior state | L0 supplies shared progress and deterministic facts; L2 supplies route-keyed enriched facts; the runtime assembler commits immutable replacements. L3/L4 consume but never mutate the record, and no current-to-history conversion occurs. |
| History boundary | Introduce prior attempts only at L3 | L3 receives the current `AttemptRecord`, a current-fact selector, and a separately selected immutable `PriorAttemptView`. |
| Runtime ownership | Stateful runtime outside the stateless analysis pipeline | `RestartAgentRuntime` owns request orchestration, record generations, deadlines, and current-process history; configuration parsing and dependency construction stay in the loader/composition root. |
| History persistence | Default-enabled bounded in-memory store | History retains 10 attempts per exact job and 3000 records total by default; configuration may disable it or override either bound. Restart-surviving persistence is outside MVP. |
| Policy-active prior facts | Deterministic record block for MVP | Each completed route may add an enriched block to the current record, but all prior-record comparisons use deterministic blocks. Route-selected enriched history is deferred. |
| History fixtures | Explicit manual-test import/export | Library/unit tests use `AttemptRecordControl` to seed, inspect, and clear the runtime store. The CLI may explicitly read and write deterministic JSON-array fixtures, but these are not automatic persistence or a production history-transfer contract. |
| MCP boundary | Thin runtime adapter later | MCP need not expose a product history API and must not implement separate history semantics. |
| History identity | Client-derived grounded root fingerprint | L0 derives the deterministic identity and L2 derives each enriched identity. Model-authored fingerprint text is retained for comparison but is not the comparison key. |
| Policy | Score-free versioned retry budgets | L4 uses typed semantics and L3 observations instead of user/not-user confidence thresholds. |
| Execution artifacts | Invocation-owned `run()` / `run_many()` envelopes | Results, traces, evidence, and candidates belong to one call; the core stores no caller-visible last-run state. |
| Deterministic fallback | Compute and publish before waiting for L1 | NVRx can use a history/policy candidate when model routes are late or unavailable. |
| Multi-model mode | Shared L0, parallel independent routes | The product returns every route; it does not vote or arbitrate in MVP. |
| Incremental publication | Canonical atomic artifacts plus lifecycle events | Shared L0, fallback, and each route become inspectable as they complete without duplicate payload trees. |
| Downstream roles | Preserve cascade and teardown in public evidence | Cleanup and fanout remain observable but cannot be mistaken for independent roots or policy inputs. |
| Progressive mode | Separate follow-up change chain, specified in `PROGRESSIVE.md` | Terminal architecture and fixtures stabilize first; production start/end state and post-end latency are not implied by terminal tests. |
