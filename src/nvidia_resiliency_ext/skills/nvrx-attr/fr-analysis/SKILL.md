---
name: fr-analysis
description: >
  Analyze PyTorch NCCL flight-recorder (FR) dumps to identify collective operation hangs and
  isolate the responsible ranks using CollectiveAnalyzer. Use when a distributed training job
  hangs due to an NCCL collective timeout and FR dump files are available. Detects the wavefront
  process group where collectives diverge and returns the root-cause suspect ranks.
compatibility: Requires PyTorch NCCL FR dumps (TORCH_NCCL_TRACE_BUFFER_SIZE > 0 must be set during training).
metadata:
  entry-point: CollectiveAnalyzer
  script: scripts/fr_attribution.py
---

# Skill: fr_analysis

Analyze PyTorch NCCL flight-recorder (FR) dumps to identify the collective operation hang
and isolate the ranks responsible, using `CollectiveAnalyzer`.

**Script:** [`scripts/fr_attribution.py`](./scripts/fr_attribution.py) → `attribution/trace_analyzer/fr_attribution.py`

---

## What it does

1. Loads all FR dump files matching a glob pattern under `--fr-path`.
2. Parses each dump into `Collective` records (op type, ranks, process group, timing, state).
3. Groups collectives by process group and sequence ID across ranks to detect mismatches.
4. Identifies the **wavefront** — the process group boundary where collectives diverge — and
   returns the missing ranks at that boundary as the root-cause suspects.

---

## CLI

```bash
python scripts/fr_attribution.py \
    --fr-path /path/to/fr_dumps/ \
    [-p "_dump_*"] \
    [--verbose] \
    [--health-check] \
    [--debug]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fr-path` | required | Path to a directory (or single file) containing FR dump files |
| `--pattern`, `-p` | `_dump_*` | Glob pattern for dump files within `--fr-path` |
| `--verbose`, `-v` | off | Print detailed per-rank collective tables |
| `--health-check`, `-c` | off | Include node health check results in output |
| `--debug` | off | Convert binary trace files to JSON for inspection |

---

## Programmatic API

```python
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer

analyzer = CollectiveAnalyzer({
    "fr_path": "/path/to/fr_dumps/",
    "pattern": "_dump_*",
    "verbose": False,
    "health_check": False,
})
results = analyzer.run_sync({
    "fr_path": "/path/to/fr_dumps/",
})
# results: tuple[FRAnalysisResult | str, AttributionState]
```

---

## Output

Returns `(result, AttributionState)` where `result` is the FR analysis table and describes:

- The selected wavefront/front process group
- **Missing ranks** at that process group (root-cause suspects)
- Per-rank collective status tables (when `--verbose`)
- Node health summary (when `--health-check`)

`AttributionState.STOP` indicates the hang is unrecoverable; `CONTINUE` indicates the job
may be restartable after isolating the identified ranks.

---

## Dump file formats

| Format | Notes |
|--------|-------|
| `_dump_*` files | PyTorch FR dump prefix pattern used by the feedback loop |
| Binary pickle / JSON payloads | Detected automatically; use `--debug` to convert binary traces to JSON |

FR dumps are typically written to the directory specified by `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`
or triggered automatically on NCCL timeout.

---

## Prerequisites

- FR dump files produced by PyTorch NCCL (set `TORCH_NCCL_TRACE_BUFFER_SIZE` > 0)
- `FR_DEBUG=1` env var enables verbose debug logging in the script
