Run the full InJob validation pipeline end-to-end: enable → deploy → monitor → validate.

## Usage
```
/nvrx-run <original_sbatch> <cluster> [--profile <profile.yaml>] [--fault-schedule "cycle:rank,..."] [--thresholds <thresholds.yaml>] [--artifacts-dir <path>]
```

- `<original_sbatch>`: original researcher sbatch (without InJob)
- `<cluster>`: cluster name (e.g. `hsg`, `lyris`)
- `--profile`: cluster profile YAML (default: `tools/fault_tolerance/profiles/<cluster>.yaml`)
- `--fault-schedule`: `NVRX_INJECT_GPU_FAILURE` spec for hardware fault injection
- `--thresholds`: validation thresholds YAML
- `--artifacts-dir`: local artifacts directory (default `~/.cache/nvrx_artifacts/<cluster>/<job_id>/`)

If no arguments given, ask the user interactively.

## Your task

Run the four stages in sequence. Maintain all context in memory (no need to re-read state files between stages since this is one session).

### Stage 1 — Enable

Apply `/nvrx-create` logic:
- Use `--env mine` (this pipeline is for the validator's own test run)
- Use the provided profile
- Apply fault schedule if given
- Write output as `<original_name>_nvrx_mine.sh`
- Print diff summary

Ask the user: "Ready to deploy? (y/n)" before proceeding to Stage 2.

### Stage 2 — Deploy

Apply `/nvrx-submit` logic:
- Upload the `_nvrx_mine.sh` to the cluster
- Run `sbatch`, capture job_id
- Print job_id

### Stage 3 — Monitor

Apply `/nvrx-watch` logic:
- Poll until job completes or timeout
- Show live status lines
- Collect artifacts

### Stage 4 — Validate

Apply `/nvrx-validate` logic:
- Run all checks (A, B, C, D)
- Write report.md
- Print summary table

## Final output

```
=== InJob Validation Complete ===
Job: <job_id>  Cluster: <cluster>
Final state: COMPLETED

OVERALL: PASS / FAIL

Checks passed: N/M
Report: ~/.cache/nvrx_reports/<cluster>/<job_id>.md
```
