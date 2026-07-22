# Analysis Profile Spec

An analysis profile is the versioned deployment and comparison unit for the
restart agent. Production consumes a selected profile. The eval
harness qualifies profiles and may recommend profile changes, but runtime does
not discover model, tool, provider, or threshold behavior on its own.

## Scope

A profile binds deterministic analyzer behavior and model/client behavior:

- profile schema version;
- human-readable `profile_id` and monotonically managed `profile_version`;
- prompt, model-evidence schema, external-output schema, policy, taxonomy, and
  tool-interface versions;
- evidence-bundle and large-log context assembly config;
- retry-policy parameters, history bounds, and fallback behavior;
- tool-enabled mode, advertised read-only tools, and tool budget policy;
- model route candidates, routing mode, per-route credential references,
  priority when applicable, deadline budgets, and concurrency limits;
- requested reasoning/thinking mode and provider-specific options;
- declared model context window, request-time safety reserve, and requested
  maximum output tokens;
- local/eval decision-window defaults and caps;
- observability sinks and trace redaction policy.

A profile MUST NOT include request fields such as `log_path`, `job_id`, or
`cycle_id`; attempt records or prior-attempt views; eval labels; or production
outcome labels. It also MUST NOT contain API keys, auth headers, bearer tokens,
or other inference credentials.

## Regulated Inference Routing

For a workload subject to export-compliance controls, the effective profile and
deployment configuration MUST resolve every model candidate and network
fallback to an approved ECCN-compliant route on the Regulated Inference Hub.
The credential source remains external to the profile and is subject to the
hub's access-control requirements. An approved deterministic no-model fallback
may remain available; an unapproved inference endpoint may not.

The analyzer does not inspect workload content to choose this mode and does not
certify endpoint compliance. The workload owner or deployment operator selects
the regulated hub, model route, and authorized credential source. Model names
or URLs containing `eccn` are not sufficient proof of compliance.

Profiles and traces SHOULD retain non-secret route identity, hosting
classification, and operator-supplied compliance classification for audit and
reproducibility. These values describe the configured route; they do not replace
external access control or compliance approval.

## Multi-Model Routing Modes

`REQUIREMENTS.md` owns the behavioral requirements for these modes. This
section defines how routes and mode-specific configuration are represented.

`collect_all` is the implemented non-arbitrating mode. It runs N independently
configured model/endpoint routes concurrently after one shared L0A, Decision
Evidence, and L0B preparation. Every route receives the same immutable evidence
and `PriorAttemptView` and produces its own L1-L4 result. The batch also retains
the shared deterministic result. No route is preferred and no result is merged
or selected as the batch winner.

`priority_select` is the future production arbitration mode. It will reuse the
same route execution and per-route result schemas, then select a valid result
according to route priority and the caller-owned deadline.

Its intended first production profile is a latency/quality pair:

- `fast_candidate`: predictable low latency, normally one turn with no tools or
  a tightly bounded tool profile;
- `preferred_enriched`: higher route priority, deeper reasoning and optionally
  tools, with a less predictable latency envelope.

Both start concurrently. The preferred enriched result wins when usable before
the NVRx deadline. If it is still in flight, the best ready fast result is used;
if no model result is ready, the deterministic fallback is used. Selection of
enriched prior-record facts is not part of the MVP and requires a separate
`priority_select` contract. The target `collect_all` record contains the
required deterministic block plus completed route-keyed enriched blocks, while
MVP prior comparisons use only deterministic blocks. Unfinished route output is
abandoned at the deadline. The stable history join remains a deterministic client
fingerprint, never a model-authored class or explanation.

Every model route has a unique `route_id`, effective model, endpoint, and
`credential_ref`. In the executable config loader, `credential_ref` names an
environment variable whose value is a readable API-key file path. Library
composition may supply another `CredentialProvider`, but that provider is not
selected by `restart_agent_config.v1`. API keys and resolved secret values
never appear in the config, collect-all result, or trace.

Concrete deployment and eval configurations SHOULD state the inference endpoint on
each route, even when routes currently share one endpoint. This keeps the
model/endpoint/credential tuple visible and allows routes to move independently
without changing implicit defaults. Effective traces always record the resolved
non-secret endpoint.

The implemented canonical configuration uses `restart_agent_config.v1`. Its
current executable subset groups model-route behavior as:

- `request`: timeout, input/output context budgets, temperature, and top-p;
- `tools`: route-wide enabled state, per-tool advertisement booleans, and
  maximum rounds;
- `reasoning`: thinking mode and provider reasoning effort;
- `reliability`: retry count and retry backoff.

The conventional filename is `restart_agent.json`. Execution mode belongs in
`routing.mode`, not in the filename; `collect_all` is the only
implemented value today. Shared defaults should carry common settings, while a
route should override only verified model- or endpoint-specific behavior. The
example uses a shared 64K output cap and records the verified 200K Qwen 235B
context window on its two Qwen 235B routes. Unknown endpoint limits remain
unset rather than being guessed.

The checked-in example resolves sampling to `temperature=0` and `top_p=1` for
reproducible qualification. A direct library caller with no config receives the
adapter fallbacks `0.2/0.7`. These are distinct resolution paths, not competing
defaults; the effective traced profile is authoritative for a run.

Top-level `model_defaults` use the same groups. Each `model_routes` entry
overrides defaults by field, allowing the same Qwen model once without tools
for a fast candidate and once with tools/reasoning for an enriched candidate.
The config also owns `routing.mode`, `routing.max_parallel_models`, and
`routing.timeout_seconds`. The timeout is the absolute whole-analysis budget
from analysis start; its default is 600 seconds. Route-level
`request.timeout_seconds` remains the per-provider-request cap and is reduced
to the remaining whole-analysis budget when necessary.
Single-model CLI flags do not mutate a supplied config.

Future L0, evidence-selection, history, policy, and observability sections may
extend this same top-level config. They are not accepted as executable settings
until their schemas and loaders are implemented.

The implemented tool-advertisement map names all four supported tools. Its
default is `overview=true`, `grep_log=true`, `read_window=true`, and
`get_evidence_objects=false`. Setting the object-read tool to `true` opts that
route into advertising its schema; implementation availability alone does not
expose it.

## Config Identity

The canonical executable file has `config_id`, `config_version`, and a computed
`config_fingerprint`. The fingerprint covers canonical JSON for the resolved,
credential-free effective config, including routing and every model route. It
excludes secret values and the source file path. Collect-all results and traces
record this identity under `restart_agent_config`.

The current implementation exposes `config_fingerprint` separately from the
broader target `profile_fingerprint` contract below. When the complete L0-L4
profile becomes executable through `restart_agent.json`, the design should
decide whether those identities converge or remain separately useful.

## Analysis Profile Identity

The complete analysis-profile identity below is a target contract. The current
executable route configuration records `config_id`, `config_version`, and
`config_fingerprint`; it does not yet expose a second profile identity. When the
full L0-L4 profile becomes executable, every run MUST record
`profile_fingerprint` and SHOULD also record `profile_id` plus
`profile_version`.

`profile_fingerprint` is `sha256:` followed by the SHA-256 digest of canonical
UTF-8 JSON for the effective profile. Canonical JSON means sorted object keys,
no insignificant whitespace, normalized numeric and boolean values, and arrays
preserved in semantic order.

The fingerprint input MUST include every field that can affect analyzer output,
latency, fallback, or trace interpretation. It MUST exclude non-semantic fields
such as comments, author, creation time, file path, and display description.

Profile implementations MUST publish the canonicalized profile payload used for
fingerprinting or preserve a lossless reference to it in trace artifacts.

## Default Resolution

The target complete-profile selection precedence is:

1. explicit `analysis_profile_id` or profile payload supplied to the library or
   CLI entrypoint;
2. service or deployment configured default profile;
3. repository built-in default profile.

If a future entrypoint supplies `analysis_profile_id` and the service cannot resolve it,
the analyzer MUST use the deterministic fallback defined by `TOOLS.md`, return
a restart-biased analyzer output, and record the profile-resolution failure in
the trace. It MUST NOT silently substitute an unrelated profile without tracing
that substitution.

The unresolved-profile fallback MUST use the repository built-in deterministic
fallback profile. That fallback profile has its own stable fingerprint and
MUST be recorded as the trace/profile `profile_fingerprint`. The
unresolved requested id and error are trace anomalies/context, not a nullable
profile fingerprint.

## Runtime Override Precedence

Profiles may carry local/eval defaults for decision windows and deadline
budgets, but NVRx owns the production post-failure decision window.
Numeric default values, when needed, belong in NVRx/service configuration or in
a concrete local/eval profile. Canonical design docs should refer to the
configured decision window rather than repeating a numeric default.

Effective deadline/window precedence for verdict-producing analysis is:

1. caller-provided absolute deadline;
2. caller-provided remaining budget;
3. caller-provided window;
4. service-local terminal deadline/window derived from NVRx policy;
5. NVRx/service workload or cluster policy/config;
6. profile local/eval default.

The analyzer MUST record the effective value and source in the trace. When
multiple caller or service fields are present, the highest-precedence field
wins and the lower-precedence fields are trace context only. A profile default
MUST NOT override an NVRx-provided or service-configured production value.

## Profile Deltas

Eval and debug runs MAY declare a `profile_delta` relative to a baseline
profile. A delta is comparable to production only when the report names the
baseline fingerprint, the delta payload, and the resulting effective
fingerprint.

Production-vs-eval comparison requires either the same effective
`profile_fingerprint` or an explicitly declared profile delta. Silent profile
differences make the comparison invalid.

## Promotion

The eval harness may recommend a profile for production when it satisfies the
configured quality, latency, malformed-output, timeout, and observability gates.
Promotion MUST record:

- the profile payload and fingerprint;
- the eval run id and case-set id;
- model route hosting classification;
- p50/p90/p99 latency gates by relevant phase and route;
- selected fallback behavior;
- known unsupported provider/tool/reasoning states.

Production applies the promoted profile. Production does not tune thresholds,
discover provider capability mappings, or change tool availability at runtime.

## Experimental Qwen Tool Profile

Eval may use `qwen235b.experimental.one_tool_round.v1` to measure whether a
small tool escape hatch improves Qwen 235B without allowing an unbounded
context-growth loop. It enables the declared read-only tools for one
tool-enabled model round, then requires one tools-disabled final model turn.
Therefore its normal ceiling is one tool round and two model turns; provider
retries are recorded separately and do not increase the semantic turn budget.

This is an experimental eval profile, not a repository-wide product default.
The harness MUST record it in per-model review artifacts and may compare it
against a tools-disabled Qwen profile. Promotion requires corpus evidence that
the tool round improves answer quality enough to justify its added latency and
token cost.

Eval may also declare `qwen397b.tools_supported.v1` for
`nvidia/qwen/eccn-qwen3-5-397b-a17b`. The route uses the ECCN credential
reference, declares a 262,144-token context window, disables thinking, and
advertises the same read-only tools without a Qwen-specific round cap. It
inherits the product's general tool-loop safety policy. The route remains a
separate candidate rather than replacing Qwen 235B. Promotion requires corpus
evidence of a semantic-quality gain and route latency that fits the applicable
decision window; successful tool-call syntax alone is insufficient.
