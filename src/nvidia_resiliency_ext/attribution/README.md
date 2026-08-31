# `nvidia_resiliency_ext.attribution`

Python library for **failure attribution** on job logs: restart-agent/attrsvc,
optional **NCCL flight-recorder** analysis, **request coalescing**, and
best-effort observability hooks (direct dataflow HTTP posting and Slack).

The main wheel ships restart-agent/attrsvc, shared attribution types, request
coalescing, and FR analysis. `nvidia-resiliency-ext[attribution]` adds packaged
FR MCP support. Legacy LogSage, splitlog orchestration, and LogSage-backed MCP
tools live under `legacy_logsage/` and are source-checkout-only; run them with
MCP support plus `langchain-core`, `langchain-openai`, and `logsage` installed
manually.

**How it is structured (current packaged APIs, the `legacy_logsage/` source-only island, MCP, and pipeline modes):**

**[ARCHITECTURE.md](./ARCHITECTURE.md)**

The public API is re-exported from `nvidia_resiliency_ext.attribution` (see package `__init__.py`).

## Restart agent

The experimental restart agent lives in
`restart_agent/`. It builds deterministic log evidence, optionally asks
an LLM for structured current-log interpretation, and applies deterministic
policy to emit `STOP` or `RESTART` guidance. Its canonical engineering specs
start at:

```text
docs/design/attribution/restart_agent/DESIGN.md
```
