---
name: nvrx-attr
description: >
  Orchestration layer over nvidia_resiliency_ext attribution modules. Provides
  log-analysis, fr-analysis, and a Megatron-LM-oriented fault-injection feedback
  loop for benchmarking attribution quality on SLURM workloads.
compatibility: Requires Python 3.10+, nvidia-resiliency-ext installed, logsage, langchain-openai, and NVIDIA_API_KEY (env var, NVIDIA_API_KEY_FILE, or ~/.nvidia_api_key). The fault-injection loop has only been validated with Megatron-LM workloads.
metadata:
  author: nvidia
---

# Attribution Skills

High-level orchestration layer over the `nvidia_resiliency_ext.attribution` modules.
Each subdirectory is a self-contained skill with its own `SKILL.md` and helper scripts.

## Skills

| Directory | Purpose | Entry point |
|-----------|---------|------------|
| [`log-analysis/`](./log-analysis/SKILL.md) | Analyze SLURM job logs for failure root-cause and restart decisions | `NVRxLogAnalyzer` (`nvrx_logsage.py`) |
| [`fr-analysis/`](./fr-analysis/SKILL.md) | Analyze NCCL flight-recorder dumps for collective-hang root-cause | `CollectiveAnalyzer` (`fr_attribution.py`) |
| [`fault-injection-loop/`](./fault-injection-loop/SKILL.md) | Run a batched SLURM fault-injection feedback loop and score attribution accuracy | `prepare_node_alloc.sh` / `watch_and_analyze.sh` |

## How skills relate to the library

```
src/nvidia_resiliency_ext/
├── attribution/
│   ├── log_analyzer/nvrx_logsage.py      ← log-analysis implementation
│   ├── trace_analyzer/fr_attribution.py  ← fr-analysis implementation
│   ├── analyzer/engine.py                ← combined orchestration entry point
│   └── combined_log_fr/                  ← optional log + FR fusion
└── skills/
    └── nvrx-attr/                        ← this skill bundle
        ├── log-analysis/
        ├── fr-analysis/
        └── fault-injection-loop/
```

The `Analyzer` (`analyzer/engine.py`) is the recommended entry point when you need
request coalescing, result caching, or the combined `LOG_AND_TRACE` pipeline.
Use the individual skills when you want to run one analysis type directly without the
full coalescing stack.

## Common prerequisites

- `NVIDIA_API_KEY` environment variable, `NVIDIA_API_KEY_FILE`, or `~/.nvidia_api_key`
- `langchain-openai` installed
- `logsage` package installed (required by `log_analysis`)
- Package installed: `pip install nvidia-resiliency-ext` or `pip install -e .` from repo root
- The fault-injection loop has only been validated with Megatron-LM training scripts
