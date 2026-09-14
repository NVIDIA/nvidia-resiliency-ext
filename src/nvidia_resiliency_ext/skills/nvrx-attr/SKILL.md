---
name: nvrx-attr
description: >
  Orchestration layer over nvidia_resiliency_ext attribution modules. Provides
  log-analysis, fr-analysis, and a Megatron-LM-oriented fault-injection feedback
  loop for benchmarking attribution quality on SLURM workloads.
compatibility: Requires Python 3.10+, NVRx attribution support, and LLM_API_KEY (env var, LLM_API_KEY_FILE, or ~/.llm_api_key) for L1 restart-agent enrichment. The fault-injection loop has only been validated with Megatron-LM workloads.
metadata:
  author: nvidia
---

# Attribution Skills

High-level orchestration layer over the `nvidia_resiliency_ext.attribution` modules.
Each subdirectory is a self-contained skill with its own `SKILL.md` and helper scripts.

## Skills

| Directory | Purpose | Entry point |
|-----------|---------|------------|
| [`log-analysis/`](./log-analysis/SKILL.md) | Analyze SLURM job logs for failure root-cause and restart decisions | `RestartAgent` (`python -m nvidia_resiliency_ext.attribution.restart_agent.cli`) |
| [`fr-analysis/`](./fr-analysis/SKILL.md) | Analyze NCCL flight-recorder dumps for collective-hang root-cause | `CollectiveAnalyzer` (`fr_attribution.py`) |
| [`fault-injection-loop/`](./fault-injection-loop/SKILL.md) | Run a batched SLURM fault-injection feedback loop and score attribution accuracy | `prepare_node_alloc.sh` / `watch_and_analyze.sh` |

## How skills relate to the library

```
src/nvidia_resiliency_ext/
├── attribution/
│   ├── restart_agent/                    ← log-analysis implementation
│   ├── trace_analyzer/fr_attribution.py  ← fr-analysis implementation
│   └── mcp_integration/                  ← packaged restart-agent + FR MCP tools
└── skills/
    └── nvrx-attr/                        ← this skill bundle
        ├── log-analysis/
        ├── fr-analysis/
        └── fault-injection-loop/
```

Use the packaged `restart_agent` and `trace_analyzer` entry points for current
log and FR analysis. Legacy LogSage, SPLITLOG, and combined LogSage+FR tools live
under `attribution/legacy_logsage/` for source-checkout workflows only.

## Common prerequisites

- `LLM_API_KEY` environment variable, `LLM_API_KEY_FILE`, or `~/.llm_api_key`
- Source checkout or installed package with the attribution extra for restart-agent and FR analysis
- The fault-injection loop has only been validated with Megatron-LM training scripts

## Fault-Loop Local Setup

Before using `fault-injection-loop/`, create the local config file from the tracked
template and fill in your site-specific values:

```bash
cp scripts/user.env.example scripts/user.env
```

The feedback-loop scripts require `src/nvidia_resiliency_ext/skills/nvrx-attr/scripts/user.env`
to exist at runtime. Keep `user.env` local and untracked.
