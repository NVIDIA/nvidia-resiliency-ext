# Product And Evaluation Boundary

This document is canonical for the boundary between the installable Restart
Agent and its non-installable evaluation harness. It defines ownership,
packaging, secrets, and parity requirements. Harness workflow, scoring, and
artifact schemas remain under `tools/restart_agent_eval/docs/`.

## Ownership

| Concern | Owner |
| --- | --- |
| L0A, Decision Evidence, L0B, L1-L4, route execution, runtime results, traces, and metrics | Product |
| Log corpus, human gold, review artifacts, panel summaries, corpus aggregation, stability analysis, profile comparison, and promotion recommendations | Harness |
| Production route/profile choice, data classification, and caller deadline | NVRx deployment/operator |

The harness MUST invoke the product under test. It MUST NOT reimplement evidence
construction, prompts, tools, grounding, identity, history, or policy. Product
`collect_all` owns shared-L0 parallel route execution; the harness selects its
review routes and evaluates the independent product results afterward.

The dependency is one-way. The harness may invoke or import an explicitly
selected product checkout, but installable product code MUST NOT import the
harness or depend on harness artifacts to produce a valid result.

## Repository And Packaging

The harness may be versioned in the product repository so product contracts,
harness expectations, tests, and documentation can change together. It lives
outside `src/nvidia_resiliency_ext`, has no product package entry point, and is
excluded from product distributions.

Installing `nvidia-resiliency-ext` MUST NOT install the harness, corpus, gold,
generated runs, provider credentials, or harness-only dependencies. Harness
implementation documents remain beside the tool rather than being copied into
the product specifications.

## Production And Eval Parity

By default, the harness evaluates the NVRx checkout that contains it. An
explicit `NVRX_RESTART_AGENT_PRODUCT_REPO` override may select another checkout
for version-comparison experiments; the run manifest MUST record that product's
commit and dirty state.

A production-comparable run MUST use the same product version and effective
analysis profile, or record an explicit profile delta. That includes prompt,
schema, evidence-selection, model route, provider request, reasoning, tool,
retry, deadline, history, and policy settings.

Gold labels, expected actions, case names, source-directory labels, and review
notes MUST NOT enter product or model-visible inputs. The product trace is
authoritative for what L1 received and returned. The harness scores only after
canonical product artifacts are complete.

## Environment And Secrets

Model-backed harness runs use the product's credential-resolution contract:

- `LLM_API_KEY_FILE` identifies the primary readable API-key file;
- `LLM_API_KEY_OLD_FILE` identifies the secondary key file used by routes that
  cannot use the primary key;
- `NVRX_LLM_BASE_URL` selects the default inference endpoint; and
- `NVRX_LLM_MODEL` selects the default model for configured single-route use.

Product-supported reasoning, timeout, sampling, and tool controls retain their
normal profile semantics. For a route that declares the secondary credential
slot, the harness may map `LLM_API_KEY_OLD_FILE` to `LLM_API_KEY_FILE` only in
that child product process. It MUST NOT pass either key-file path as a command
argument.

The repository MUST NOT contain API keys, bearer tokens, copied key material,
credential-bearing configuration, or user-specific fallback key paths. Route
profiles may contain public model identifiers, endpoint identifiers,
credential-reference names, and non-secret controls, but never credential paths
or values. Commands, traces, manifests, and reports MUST exclude credentials,
authorization headers, key-file paths, and key-file contents.

## Compliance

The operator is responsible for selecting an endpoint authorized for the log
content. Export-controlled workloads are expected to use an approved Regulated
Inference Hub route and credentials. Neither product nor harness infers data
classification or compliance from log text or model names.

## Artifacts And Metrics

The product emits stage artifacts, per-route results and traces, invocation
health, timing, token, tool, and endpoint measurements. The harness consumes
those artifacts and adds gold comparisons, model/prompt/profile comparisons,
stability measurements, and promotion reports.

Product schemas are defined in `SCHEMA.md`. Harness artifact roots, gold schema,
KPI definitions, and panel layout are defined under
`tools/restart_agent_eval/docs/`; they are intentionally not duplicated here.
