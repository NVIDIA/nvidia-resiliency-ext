# Decision Stability

## Purpose

Decision stability asks a narrow question: given the same model-visible input,
route profile, product implementation, and source log, does a route produce the
same usable semantic assessment and final action on repeated invocations?

Stability is not accuracy. A route can be consistently wrong, and an accurate
route can be operationally unreliable. Gold-scored correctness, semantic
stability, tool behavior, latency, tokens, and endpoint reliability remain
separate measurements.

The stability summarizer is a pure harness operation. It reads completed
one-log run directories and never invokes a model.

## Comparable Cohorts

The harness compares samples only when these identities agree:

- source-log SHA-256;
- product commit;
- complete analyzer configuration fingerprint, including routing and policy;
- runtime-input SHA-256;
- route-profile SHA-256, including model, endpoint, request, reasoning, tool,
  retry, and timeout settings;
- complete L0A bundle SHA-256;
- L0B model-view payload SHA-256;
- first model-request payload SHA-256, which covers the actual initial
  messages, schema, model parameters, and advertised tools.

A changed value creates a different cohort. The harness does not average those
runs together. Missing identities make comparability `incomplete`. A dirty
product checkout makes a matching cohort `provisional_dirty_product`, because
the commit alone does not identify uncommitted policy or prompt changes.

This protects the primary interpretation: observed variation inside a verified
cohort is output variation, while variation across cohorts may be caused by an
input, profile, evidence, prompt, or implementation change.

## Measurements

Each route/cohort reports:

- usable L1 response rate;
- final `STOP`/`RESTART` distribution, modal agreement, and sequential flips;
- exact agreement and per-field distributions for the action-driving scalar
  fields in `model_recovery_assessment`;
- root-cause status, model confidence, semantic primary class/line, and grounded
  root-fingerprint agreement;
- model/tool call variability, no-new-context calls, tool sequences, latency,
  and tokens;
- endpoint status, failed attempts, retries, and timeouts;
- L1 semantic and L4 action accuracy when human-approved gold is attached.

Confidence, rationale text, and cited-line ordering are retained in ordinary
run artifacts but excluded from the exact policy-input tuple. Natural-language
wording changes should not by themselves count as a policy-semantic flip.

Sequential flip rate compares adjacent applicable observations in run order.
Modal agreement measures the largest observed value count divided by the
applicable sample count. Both are shown because a 90/10 distribution and a
route that alternates every request have different operational behavior.

## Status

The default minimum is ten comparable samples:

- `insufficient_samples`: fewer than the configured minimum;
- `comparability_incomplete`: enough samples, but an exact input identity is
  unavailable;
- `observed_stable`: every sample has usable L1 semantics and both final action
  and exact policy-input tuple agree;
- `observed_unstable`: the minimum is met but usability, action, or policy-input
  agreement is below 100 percent.

These are descriptive statuses, not release gates or promotion verdicts.
Thresholds can be introduced only after the reviewed corpus establishes the
acceptable false-STOP and semantic-variance envelope.

## Usage

Generate repeated one-log runs with the normal review command. When enough
runs and gold cases exist, summarize all completed runs for that log:

```bash
./examples/summarize_decision_stability.sh \
  --runs-root "$RESTART_AGENT_EVAL_RUN_ROOT/checkpoint_logs/job.log" \
  --route qwen397b \
  --minimum-samples 10
```

The command discovers directories containing `panel_summary.json`. It writes:

```text
<per-log-runs-root>/stability/<timestamp>/stability_summary.json
<per-log-runs-root>/stability/<timestamp>/stability_summary.md
```

Explicit run directories are also accepted, which is useful for a curated
comparison that excludes exploratory runs:

```bash
./examples/summarize_decision_stability.sh \
  --route qwen397b \
  /path/to/run-01 /path/to/run-02 /path/to/run-03
```

`--latest N` limits discovery to the newest N completed run directories. Route
filters accept either the harness target name or exact model name. The
summarizer does not create the repeated model runs itself; this keeps expensive
execution explicit and allows the same analyzer to work over existing result
directories.
