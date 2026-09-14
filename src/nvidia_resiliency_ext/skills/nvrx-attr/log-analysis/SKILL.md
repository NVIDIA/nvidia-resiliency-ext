---
name: log-analysis
description: >
  Analyze a SLURM job log file for failure root-cause attribution and restart decisions using
  the packaged NVRx restart-agent. Use when you have a SLURM training job log and need to
  determine why the job failed and whether it should be restarted. Performs deterministic
  evidence extraction plus optional LLM enrichment.
compatibility: Requires NVRx attribution support and LLM_API_KEY for L1 enrichment. Can run deterministic-only with --disable-l1.
metadata:
  entry-point: RestartAgent
  command: python -m nvidia_resiliency_ext.attribution.restart_agent.cli
---

# Skill: log_analysis

Analyze a SLURM job log file for failure root-cause attribution and restart decisions using `RestartAgent`.

**Command:** `python -m nvidia_resiliency_ext.attribution.restart_agent.cli`

---

## What it does

1. Reads the log file (UTF-8, falls back to latin-1).
2. Builds deterministic L0 evidence from training progress, failure lines, checkpoint state, and known restart-policy signals.
3. Optionally runs L1 model enrichment to explain the primary failure and related evidence.
4. Applies the restart-agent retry/recovery policy and returns a structured `STOP` or `RESTART` decision.

---

## CLI

```bash
python -m nvidia_resiliency_ext.attribution.restart_agent.cli \
    /path/to/job.log \
    [--job-id JOB_ID] \
    [--cycle-id CYCLE_ID] \
    [--config examples/attribution/restart_agent.json] \
    [--llm-model MODEL] \
    [--llm-base-url URL] \
    [--llm-api-key-file PATH] \
    [--disable-l1] \
    [--summary]
```

| Flag | Default | Description |
|------|---------|-------------|
| `log_path` | required | Path to the job log file |
| `--job-id` | unset | Optional job id for request metadata and retry history |
| `--cycle-id` | unset | Optional cycle id for retry history |
| `--config` | unset | Versioned restart-agent config for configured model routes and retry policy |
| `--llm-model` | environment/default config | L1 enrichment model when `--config` is not used |
| `--llm-base-url` | environment/default config | OpenAI-compatible base URL when `--config` is not used |
| `--llm-api-key-file` | environment/default config | File containing the L1 API key |
| `--disable-l1` | off | Run deterministic-only analysis |
| `--summary` | off | Emit a compact human-readable summary to stderr in addition to JSON stdout |

---

## Programmatic API

```python
from nvidia_resiliency_ext.attribution.restart_agent.l1 import LlmConfig, LlmEvidenceExtractor
from nvidia_resiliency_ext.attribution.restart_agent.models import RestartAgentRequest
from nvidia_resiliency_ext.attribution.restart_agent.pipeline import RestartAgent

agent = RestartAgent(evidence_extractor=LlmEvidenceExtractor(LlmConfig.from_env()))
run = agent.run(RestartAgentRequest(log_path="/path/to/job.log", job_id="12345"))
result = run.result.to_payload()
```

---

## Output

The CLI prints one JSON object to stdout. Important fields include:

| Field | Meaning |
|---|---|
| `decision` | `RESTART` or `STOP` |
| `decision_basis` | Policy reason for the decision |
| `retry_policy` | Retry budget and selected rule details |
| `primary_failure` | Primary failure evidence, when identified |
| `secondary_failures` | Related or cascading failure evidence |
| `evidence_coverage` | Which evidence sources were checked/found |
| `justification` | Human-readable explanation for the final decision |

---

## Prerequisites

- `LLM_API_KEY` set (env var, `LLM_API_KEY_FILE`, or `~/.llm_api_key`) for L1 enrichment
- Use `--disable-l1` when you want deterministic-only analysis without model access
