# Restart Agent Status

This document is descriptive. Normative behavior remains in `DESIGN.md` and the
canonical focused specifications that it indexes.

## Current Maturity

The terminal analyzer is a feasibility implementation with complete stage
tracing and deterministic replay support. It is not production-qualified.
Exploratory one-log runs show that structured evidence, semantic recovery
assessment, minimal grounding, history comparison, and deterministic policy can
produce useful decisions. The current gold corpus is intentionally small and
cannot establish fleet-wide accuracy or false-STOP safety.

## Implemented

- Invocation-owned `run()` and `run_many()` results with shared L0 replay.
- L0A typed evidence, DecisionEvidence, and bounded L0B model view.
- Deterministic fallback publication before model completion.
- Provider-neutral L1 contracts, OpenAI-compatible routes, optional read-only
  tools, retry/deadline handling, token accounting, and parallel route fanout.
- L2 source grounding, enriched failure identity, and advisory audit
  findings, including route-keyed `AttemptRecord.enriched` facts.
- Runtime-owned bounded attempt records with exact-job prior selection,
  same-cycle replacement, per-job/total eviction, generation closure, and
  explicit library/CLI seed-inspect-export controls.
- Shared `AttemptProgressSummary` construction and L3 same-root comparison over
  completed-step, checkpoint-step, and fallback failure-iteration dimensions.
- L4 versioned retry budgets and binary action.
- Per-stage timings, outputs, provenance, endpoint events, token/tool metrics,
  and atomic result/trace publication.
- Deterministic behavior tests derived from reviewed checkpoint, permission,
  port-conflict, world-size, CUDA-code, and IB-port-flap scenarios. The gold
  corpus itself belongs to the companion eval harness and is not shipped in the
  product package.

## Qualification Work

1. Expand and review a representative gold corpus before adding narrow
   signatures or action rules.
2. Measure L0A/L0B quality, semantic accuracy, model behavior, endpoint
   reliability, fingerprint false merges/splits, policy accuracy, and repeated
   decision stability.
3. Qualify model-route profiles and regulated inference routing.
4. Implement and measure the progressive start/end path, log-drain overlap,
   best-available candidate selection, and post-end latency.
5. Run shadow-mode STOP validation before production action authority.

## Explicit Follow-Ups

- Progressive service state and NVRx integration are a separate change chain;
  `PROGRESSIVE.md` records its target contract.
- Workload-managed bad-token/token-window retry-and-skip policy needs a robust
  generic signal before it can alter retry budgets.
- Client-derived concrete fingerprints remain experimental until corpus tests
  characterize false merges and false splits.
- L1 model-created classes, phases, and data-position claims remain observable
  but cannot become history identity without deterministic grounding.
- Optional verifier, route arbitration, MCP transport, structured runtime
  signals, attrsvc hydration, restart-surviving history, and isolation
  recommendations are post-MVP extensions.
- Persistent/distributed history, attrsvc hydration, and an MCP transport remain
  follow-ups; the implemented MVP store is current-process memory only.
- Complete analysis-profile identity/resolution is also target work. The current
  executable route configuration records `config_id`, `config_version`, and
  `config_fingerprint`; it does not yet emit a separate `profile_fingerprint`.
- Cross-invocation provider concurrency limiting, a stateful circuit breaker,
  and provider-capacity ownership classification remain production follow-ups;
  the current terminal path enforces per-analysis fanout, request limits,
  retries, and the absolute analysis deadline.

## Code Debt

Stage ownership is explicit in the package tree. Public orchestration, shared
preparation, single-route execution, multi-route execution, decision
composition, infrastructure, and per-stage trace rendering now have separate
owners. L0 detection/contextualization/assembly, DecisionEvidence selection,
L2 citation grounding, and L3/L4 result composition have typed internal seams.
Clocks, executors, credentials, HTTP, retry sleep, configuration environments,
log sources, and evidence tools are replaceable at their infrastructure or
composition boundaries. The detailed pattern algorithms, provider protocol
encoding, and advisory audit rules remain large but cohesive domain modules;
future physical moves should be driven by measured ownership changes rather
than line count alone.
