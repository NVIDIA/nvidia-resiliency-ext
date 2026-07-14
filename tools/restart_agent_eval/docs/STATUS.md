# Evaluation Harness Status

The harness has one execution path and one corpus format. Manual review and
scored evaluation both invoke the product analyzer; source logs, durable
`gold.json`, and disposable runs use mirrored non-overlapping roots.

## Implemented

- One-log deterministic or N-route parallel model review.
- Incremental shared-L0, fallback, and per-route artifact visibility.
- Strict `restart_agent_eval.v1` human-review and source-hash validation.
- Per-stage product scoring, per-route reviews, panel summary, and exhaustive
  diagnostics.
- Semantic quality, behavioral efficiency, endpoint reliability, route
  outcome, latency, token, and tool-use reporting.
- Repeated-run stability cohorts and flip metrics.
- Deterministic behavior fixture capture/check across the human-gold corpus.
- Reusable package modules with thin command wrappers.

## Current Corpus

Six reviewed deterministic fixtures cover checkpoint load, permission, port
conflict, world-size/progress-log, CUDA code error, and IB port-flap cases. This is enough to
guard structural refactors; it is not enough to rank model profiles or qualify
STOP behavior.

## Next

1. Expand balanced human gold across failure families and holdout sets.
2. Add model-backed trace fixtures for L1 provider/tool-loop and L2 grounding/audit
   refactors.
3. Aggregate stage quality, model efficiency, endpoint reliability, stability,
   and policy accuracy by route profile.
4. Add prompt/registry/profile anti-pattern lint and candidate experiments.
5. Add progressive replay and post-end deadline measurement after the product
   progressive path exists.
6. Produce explicit profile promotion or rejection reports.

## Code Debt

Execution, product-trace adaptation, scoring, artifact I/O, per-route Markdown,
panel normalization, and panel Markdown now have separate module owners.
Review and corpus workflows have explicit application composition roots;
clocks, subprocess execution, polling sleep, environment snapshots, and local
artifact storage have replaceable boundaries. Review, panel, and tool-efficiency
normalization also accept already-loaded payloads. The flat panel-row adapter
and exhaustive diagnostic renderer remain long boundary serializers, but
contain no product policy. Split gold scoring from runtime KPI derivation only
when corpus fixtures show that it improves ownership or independent evolution.
