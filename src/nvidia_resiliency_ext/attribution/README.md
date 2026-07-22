# `nvidia_resiliency_ext.attribution`

Python library for **failure attribution** on job logs: LogSage (LLM over logs), optional **NCCL flight-recorder** analysis, optional **LLM merge** of log + trace, **request coalescing**, SLURM-oriented **splitlog** tracking, and best-effort observability hooks (direct dataflow HTTP posting and Slack).

Install the optional attribution dependency set with:

```bash
pip install 'nvidia-resiliency-ext[attribution]'
```

**How it is structured (subsystems, diagrams, `AttributionController`, `Analyzer` / `LogAnalyzerConfig`, MCP vs in-process LogSage, pipeline modes):**

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
