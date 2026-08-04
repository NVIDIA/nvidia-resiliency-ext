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

## Current Scope

- Terminal-first L0A-L4 analysis over one byte-chunk ingestion, incremental
  decoding, and observation-index path, with typed evidence, deterministic
  recommendation, bounded same-job history, and deterministic retry policy.
  Parallel model enrichment is implemented and enabled by default. Progressive
  pre-end L0A polling remains implemented and disabled by default.
- Library and CLI execution with history seed, inspection, and export controls.
- Direct attrsvc integration for progressive registration and L0A
  precomputation, terminal drain/finalization, background enrichment, and
  nonblocking result probes.
- File-backed finalized source access with compact line offsets, incremental
  terminal drain, one canonical final reduction, compact attrsvc retention, and
  a 240-second default enrichment timeout.
- Per-stage results, provenance, timing, model/tool activity, endpoint events,
  and token usage in result and trace artifacts.

## Production Qualification Gates

1. Expand and review a representative gold corpus before adding narrow
   signatures or action rules.
2. Measure L0A/L0B quality, semantic accuracy, model behavior, endpoint
   reliability, fingerprint false merges/splits, policy accuracy, and repeated
   decision stability.
3. Qualify model-route configurations and regulated inference routing.
4. Measure terminal log-drain/L0A and end-to-candidate latency at target scale.
   Qualify progressive precompute only if those measurements justify enabling
   it.
5. Run shadow-mode STOP validation before production action authority.

## Deferred Scope

- Restart-surviving or distributed history and attrsvc history hydration.
- Route arbitration, verifier models, and a Restart Agent MCP transport.
- Structured runtime signals, isolation recommendations, and provider-capacity
  control.
