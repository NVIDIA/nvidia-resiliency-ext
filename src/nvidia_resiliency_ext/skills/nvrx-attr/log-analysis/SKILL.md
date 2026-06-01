---
name: log-analysis
description: >
  Analyze a SLURM job log file for failure root-cause attribution and restart decisions using
  NVRxLogAnalyzer. Use when you have a SLURM training job log and need to determine why the
  job failed and whether it should be restarted. Performs per-cycle chunking, fast-path pattern
  matching, and LLM-based classification.
compatibility: Requires LLM_API_KEY, langchain-openai, and logsage packages installed. nvidia-resiliency-ext must be installed.
metadata:
  entry-point: NVRxLogAnalyzer
  script: scripts/nvrx_logsage.py
---

# Skill: log_analysis

Analyze a SLURM job log file for failure root-cause attribution and restart decisions using `NVRxLogAnalyzer`.

**Script:** [`scripts/nvrx_logsage.py`](./scripts/nvrx_logsage.py) → `attribution/log_analyzer/nvrx_logsage.py`

---

## What it does

1. Reads the log file (UTF-8, falls back to latin-1).
2. Splits into per-cycle chunks using `chunk_logs_strict` (scans for `profiling.py:.*Cycle:\s*N` markers). Falls back to a single chunk when no markers are found.
3. For each chunk, extracts application errors via `return_application_errors` (logsage).
4. Classifies each chunk with fast-path pattern matching (training done, SLURM cancelled, preemption, time limit) or calls the LLM via `get_proposed_solution_cat`.
5. Returns one result tuple per cycle.

---

## CLI

```bash
python scripts/nvrx_logsage.py \
    --log-path /path/to/job.log \
    [--model MODEL] \
    [--temperature 0.2] \
    [--top_p 0.7] \
    [--max_tokens 8192] \
    [--exclude_nvrx_logs] \
    [--is_per_cycle]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--log-path` | required | Path to the job log file |
| `--model` | `nvidia/nemotron-3-super-120b-a12b` | LLM model |
| `--temperature` | `0.2` | Sampling temperature |
| `--top_p` | `0.7` | Top-p nucleus sampling |
| `--max_tokens` | `8192` | Max output tokens |
| `--exclude_nvrx_logs` / `--no-exclude_nvrx_logs` | on | Strip `nvidia_resiliency_ext` / `[workload:]` lines before chunking (default on; use `--no-exclude_nvrx_logs` to disable) |
| `--is_per_cycle` | off | Skip chunking — treat the whole file as a single pre-split cycle |

---

## Programmatic API

```python
from nvidia_resiliency_ext.attribution.log_analyzer.nvrx_logsage import NVRxLogAnalyzer

analyzer = NVRxLogAnalyzer({
    "log_path": "/path/to/job.log",
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 8192,
    "exclude_nvrx_logs": False,
    "is_per_cycle": False,
})
results = analyzer.run_sync({"log_path": "/path/to/job.log"})
# results: tuple[list[RawAnalysisResultItem], AttributionState]
```

Run-time overrides take precedence over constructor config (see `base.effective_run_or_init_config`).

---

## Output

Each returned `RawAnalysisResultItem` keeps `raw_text` with five fields joined by `\n`,
but also carries the parsed fields directly so consumers do not reparse the text:

```
<restart_decision>      # "RESTART IMMEDIATE" | "STOP - DONT RESTART IMMEDIATE"
<error_explanation>     # short string or ""
<attribution_text>      # "Attribution: Primary issues: [...], Secondary issues: [...]"
<additional_detail>     # extended text or ""
<checkpoint_saved>      # "True" | "False"
```

The serialized cycle fields are `auto_resume`, `auto_resume_explanation`,
`attribution_text`, `checkpoint_saved_flag`, `primary_issues`, and
`secondary_issues`, plus the parsed cycle `action`. The overall client decision
is emitted separately as `recommendation.action` / `recommendation.source`. The runner's internal
`AttributionState.STOP` is set only when the parsed cycle action is `STOP`.

### Fast-path decisions (no LLM call)

| Detected condition | restart_decision | attribution_text |
|--------------------|-----------------|-----------------|
| Training complete | `STOP - DONT RESTART IMMEDIATE` | `TRAINING DONE` |
| SLURM preemption | `RESTART IMMEDIATE` | `SLURM CANCELLED DUE TO PREEMPTION` |
| SLURM step cancelled | `RESTART IMMEDIATE` | `SLURM STEP CANCELLED` |
| SLURM job requeue | `RESTART IMMEDIATE` | `SLURM STEP CANCELLED JOB REQUEUE` |
| Time-limit exceeded | `STOP - DONT RESTART IMMEDIATE` | status string |
| Empty log | — | `NO LOGS` |
| No errors found | — | `ERRORS NOT FOUND` |
| LLM failure | — | `LLM FAILURE` |

---

## Prerequisites

- `LLM_API_KEY` set (env var, `LLM_API_KEY_FILE`, or `~/.llm_api_key`)
- `langchain-openai` and `logsage` packages installed
