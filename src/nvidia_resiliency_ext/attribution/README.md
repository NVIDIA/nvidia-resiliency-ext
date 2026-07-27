# `nvidia_resiliency_ext.attribution`

Python library for **failure attribution** on job logs: LogSage (LLM over logs), optional **NCCL flight-recorder** analysis, optional **LLM merge** of log + trace, **request coalescing**, SLURM-oriented **splitlog** tracking, and best-effort observability hooks (direct dataflow HTTP posting and Slack).

Install NVIDIA Resiliency Extension with:

```bash
pip install nvidia-resiliency-ext
```

**How it is structured (subsystems, diagrams, `AttributionController`, `Analyzer` / `LogAnalyzerConfig`, MCP vs in-process LogSage, pipeline modes):**

**[ARCHITECTURE.md](./ARCHITECTURE.md)**

The public API is re-exported from `nvidia_resiliency_ext.attribution` (see package `__init__.py`).
