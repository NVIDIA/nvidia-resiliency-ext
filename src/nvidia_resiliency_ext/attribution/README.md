# `nvidia_resiliency_ext.attribution`

Python library for **failure attribution** and scheduler observations:
restart-agent/attrsvc, optional **NCCL flight-recorder** analysis,
**scheduler exclusions**, **request coalescing**, and
best-effort observability hooks (direct dataflow HTTP posting and Slack).

The main wheel ships restart-agent/attrsvc, shared attribution types, request
coalescing, scheduler exclusions, and FR analysis. `nvidia-resiliency-ext[attribution]` adds packaged
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

## Scheduler exclusions

`scheduler_exclusions/` observes Slurm node state and publishes per-array-task
exclusions using the deployed Segment Health artifact contract. It owns polling,
publication, and isolated process lifecycle management. It uses only the standard
library and does not depend on log analysis or the fault-tolerance launcher.

`ft_launcher` selects the process owner and supplies explicit job configuration;
`services/scheduler_exclusions` provides the standalone CLI adapter. Attrsvc
hosting is not implemented. Deployment and the reusable API are described in
[the publisher guide](../../../services/scheduler_exclusions/README.md).
