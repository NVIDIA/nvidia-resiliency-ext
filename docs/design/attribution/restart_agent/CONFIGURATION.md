# Restart Agent Configuration

`restart_agent_config.v1` is the complete deployment configuration for one
Restart Agent runtime. There is no separate executable profile object.

This document is the human-readable field reference. `config.py` is the
executable authority for parsing, defaults, validation, resolution, and
fingerprint construction. `SCHEMA.md` owns request, response, and internal
stage serialization rather than duplicating this configuration contract.

The maintained example is
`examples/attribution/restart_agent.json`.

## Top-Level Fields

| Field | Type | Required / default | Meaning and constraints |
| --- | --- | --- | --- |
| `schema_version` | string | required | Must be exactly `restart_agent_config.v1`. |
| `config_id` | string | required | Non-empty human-managed name. It is reported but excluded from the behavior fingerprint. |
| `config_version` | integer | required | Human-managed revision, at least `1`. It is reported but excluded from the behavior fingerprint. |
| `enrichment` | object | `{}` | Enables or disables L1/L2 model enrichment. |
| `routing` | object | `{}` | Controls route fanout and the whole-analysis timeout. |
| `runtime` | object | `{}` | Controls in-process history and L0 source reading. |
| `retry_policy` | object | documented defaults | Configures L4 retry budgets. |
| `policy_contexts` | object | documented defaults | Configures trusted L4 policy contexts supplied outside log semantics. |
| `model_defaults` | object | `{}` | Shared defaults inherited by every model route. |
| `model_routes` | array | `[]` | Independent L1 routes. Must be non-empty when enrichment is enabled and empty when it is disabled. |

The configuration does not contain request data, attempt history, prompts,
evidence, model responses, evaluation labels, API keys, or resolved
credentials. Prompt, response-schema, detector, and stage-algorithm versions
belong to the product build. The immutable `ClusterExecutionContext` is also a
product contract rendered into the static L1 prompt; it is not a deployment
configuration field.

## Enrichment

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `enrichment.enabled` | boolean | `true` | Runs configured L1/L2 model routes when true. |

Deterministic-only execution is an explicit configuration:

```json
{
  "enrichment": {"enabled": false},
  "routing": {"mode": "collect_all", "max_parallel_models": 0},
  "model_routes": []
}
```

The deterministic L0 -> L3 -> L4 recommendation remains available when
enrichment is enabled; model routes produce additional independent candidates.

## Routing

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `routing.mode` | string | `collect_all` | Only `collect_all` is implemented. It returns every route independently and does not vote, merge, prioritize, or select a winner. |
| `routing.max_parallel_models` | integer | number of resolved routes | Maximum concurrent model routes. Minimum is `1` with enrichment and `0` without it. The resolved value cannot exceed the number of routes. |
| `routing.timeout_seconds` | number | `240` | Positive whole-analysis deadline shared by all routes. Provider calls, retries, and tool turns are bounded by its remaining time. This is not the external NVRx action deadline. |

L0 is constructed once before `collect_all` fans out. Every route receives the
same immutable L0A, Decision Evidence, L0B, request, prior-attempt view, and
analysis deadline.

## Runtime

### History

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `runtime.history.enabled` | boolean | `true` | Enables the runtime-owned, current-process attempt-record store. |
| `runtime.history.max_attempts_per_job` | integer | `10` | Positive per-job retained-attempt bound. |
| `runtime.history.max_total_records` | integer | `3000` | Positive total retained-record bound across jobs. |

The per-job limit evicts the smallest cycle id for that job. The total limit
evicts the oldest insertion. Disabling history supplies a null store; it does
not move history ownership into L0-L4.

### L0 Source

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `runtime.l0_source.read_mode` | string | `chunked` | `chunked` is the production path. `single_snapshot` is the parity/regression path. |
| `runtime.l0_source.chunk_size_bytes` | integer | `1048576` | Positive byte-delivery chunk size. It applies to chunked source reading. |

Both modes must produce equivalent canonical L0A evidence for the same captured
source bytes. These fields affect source delivery and operational performance,
not policy semantics.

## Retry Policy

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `retry_policy.concrete_confirmation_retry_allowed_retries` | integer | `1` | Non-negative budget for exact root-and-entity confirmation. Must not exceed the general budget. |
| `retry_policy.workload_confirmation_retry_allowed_retries` | integer | `1` | Non-negative root-only budget for a qualifying L1 workload failure without full identity. Must not exceed the general budget. |
| `retry_policy.general_retry_allowed_retries` | integer | `2` | Non-negative same-root safety ceiling for root identity; root-independent same-job no-progress budget for observation-only general retry. |
| `retry_policy.job_no_progress_allowed_retries` | integer | `3` | Root-independent same-job no-progress guard. |
| `retry_policy.job_unknown_progress_allowed_retries` | integer | `3` | Root-independent same-job unknown-progress guard. |

L4 owns the interpretation and ordered selection of these budgets. See
`L4.md`; configuring a number does not cause an earlier stage to select that
rule.

## Policy Contexts

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `policy_contexts.cuda_oom_no_retry.enabled` | boolean | `true` | Applies the zero-retry product policy when the selected terminal failure has the typed CUDA OOM signature. |
| `policy_contexts.port_bind_confirmation_retry.enabled` | boolean | `true` | Enables matching a selected terminal address-in-use bind failure independent of its L1 domain label. |
| `policy_contexts.port_bind_confirmation_retry.allowed_retries` | integer | `1` | Non-negative same-root confirmation budget for a matched port-bind conflict. |
| `policy_contexts.rejected_iteration_retry_then_skip.enabled` | boolean | `true` | Enables matching the typed rejected-nonfinite-iteration signature. |
| `policy_contexts.rejected_iteration_retry_then_skip.allowed_retries` | integer | `2` | Non-negative retry budget supplied when that current signature matches. |

A policy context does not change L0 identity or L1 semantics. L4 records the
ordinary base rule, then applies a matching context as the effective policy.
History absence or mismatch affects budget consumption, not whether a valid
current-attempt context is selected. See `L4.md` for the complete signature.

## Model Defaults And Routes

`model_defaults` and each `model_routes[]` item support the same route settings.
A route additionally requires a unique, non-empty `route_id`.

Resolution is shallow by field group:

1. start with built-in provider defaults;
2. apply top-level fields and nested groups from `model_defaults`;
3. apply that route's top-level fields and nested group members; and
4. validate and record the credential-free effective route.

A route may therefore override one request setting without repeating the rest
of `model_defaults`.

### Route Identity

| Field | Type | Required / default | Meaning and constraints |
| --- | --- | --- | --- |
| `route_id` | string | required per route | Unique, non-empty result and trace identity. Not used in `model_defaults`. |
| `model` | string | `nvidia/qwen/qwen3.5-35b-a3b` | Non-empty provider model identifier. |
| `base_url` | string | `https://inference-api.nvidia.com/v1` | Non-empty OpenAI-compatible endpoint base URL. |
| `credential_ref` | string | `LLM_API_KEY_FILE` | Name of an environment variable containing a readable API-key file path. |

The credential reference name may appear in effective configuration and trace.
The environment value, file path, and key contents may not.

### Request

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `request.timeout_seconds` | number | `120` | Positive per-provider-request timeout, clamped to the remaining whole-analysis deadline. |
| `request.max_output_tokens` | integer | `64000` | Positive provider output-token cap. |
| `request.context_window_tokens` | integer or omitted | model-specific when known | Positive total context-window cap. Known Qwen routes have built-in limits; otherwise omission means no configured cap. |
| `request.context_safety_tokens` | integer | `4096` | Non-negative reserve removed from the usable context budget. |
| `request.temperature` | number or omitted | `0.2` | Sampling temperature in `[0, 2]`. Some provider/model profiles omit sampling parameters. |
| `request.top_p` | number or omitted | `0.7` | Nucleus-sampling value in `[0, 1]`. Some provider/model profiles omit sampling parameters. |

Context budgeting accounts for the complete stateless conversation sent on
each model turn, including prior messages and tool results. Explicit route
configuration takes precedence over a built-in model limit. If neither exists,
Restart Agent records the estimated input but performs no client-side context
rejection or output-cap adjustment; the provider remains authoritative.

### Tools

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `tools.enabled` | boolean | `true` | Whether tool definitions are advertised and tool requests may execute. |
| `tools.advertisement.overview` | boolean | `false` | Advertises source/evidence orientation for clients that do not already supply the normal L0B view. |
| `tools.advertisement.grep_log` | boolean | `true` | Advertises source-log search. |
| `tools.advertisement.read_window` | boolean | `true` | Advertises bounded raw source reading. |
| `tools.advertisement.get_evidence_objects` | boolean | `true` | Advertises structured evidence-object retrieval for references emitted by L0B. |
| `tools.max_rounds` | integer | `3` | Non-negative maximum tool-request rounds. `0` disables tool advertisement and execution for the route. |

Tool implementation and response limits are defined in `TOOLS.md`. A tool is
model-visible only when both `tools.enabled` and its advertisement value are
true and `tools.max_rounds` is positive. With `max_rounds=0`, the route performs
one tools-disabled model turn and reports `effective_advertised=[]` without
altering its configured advertisement flags.

### Reasoning

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `reasoning.thinking_mode` | string | `auto` | One of `auto`, `disable`, or `allow`. `auto` applies provider/model-specific behavior. |
| `reasoning.reasoning_effort` | string or omitted | omitted | Optional provider reasoning-effort value. Unsupported providers may ignore or reject it. |

Thinking mode controls provider reasoning features; it does not change the L1
response contract or transfer L4 policy ownership to the model.

### Reliability

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `reliability.max_retries` | integer | `1` | Non-negative retry count for retryable provider failures. |
| `reliability.retry_backoff_seconds` | number | `0.5` | Non-negative delay between provider retries, bounded by the remaining deadline. |

Retries, provider failures, and timeout exhaustion remain visible in the route
trace and metrics. `L1.md` defines the closed provider-outcome classification
that determines which failures consume this retry budget.

## Credentials And Compliance

Configuration contains credential-reference names only. The composition layer
resolves each referenced environment variable to a readable key file while
constructing provider clients.

The operator owns workload classification and route authorization. For
export-controlled context, every route and network fallback must be approved
for the regulated inference environment. Model names and URLs do not prove
compliance, and the configuration fingerprint is not a compliance certificate.

## Validation And Effective Configuration

The loader:

1. validates required identity and supported top-level/runtime/policy fields;
2. validates enrichment and route-count consistency;
3. merges model defaults and route overrides;
4. validates numeric ranges, route-id uniqueness, tool names, and routing mode;
5. resolves credential references without retaining secret values; and
6. returns an immutable `RestartAgentConfig` with route specifications and
   credential-free `effective_config`.

Only fields documented here participate in resolution. Unknown fields in
closed configuration objects are errors; route/default keys outside the
documented groups are unsupported and must not be used.

Configuration parsing and dependency construction occur outside the stateful
runtime. Loading a JSON file and parsing an already-loaded mapping produce the
same resolved contract.

## Configuration Identity

Every resolved configuration records:

- `config_id`;
- `config_version`;
- `config_fingerprint`; and
- credential-free `effective_config`.

`config_fingerprint` is `sha256:` plus SHA-256 over canonical JSON for
`effective_config`, with sorted object keys, compact separators, and array
order preserved. It includes resolved routing, history, L0 source settings,
retry policy, policy contexts, and route settings. It excludes secrets,
the source file path, `config_id`, and `config_version`.

Different names or file locations with identical resolved behavior therefore
produce the same fingerprint. A route, timeout, tool, reasoning, reliability,
history, L0 source, retry-policy, or policy-context change produces a different
fingerprint.

The fingerprint does not identify product code, prompt text, response-schema
implementation, detector logic, or workload input.

## Production And Evaluation Parity

A reproducible comparison requires:

- the same product revision and emitted contract versions;
- the same `config_fingerprint`; and
- explicit reporting of intentional code or configuration differences.

The eval harness may recommend configuration changes, but production does not
tune routes, tools, reasoning controls, retry budgets, or provider mappings at
runtime. Model-panel experiments and qualification results belong to the eval
harness rather than this product contract.
