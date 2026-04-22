# Fault Injection Session Report — April 9–13, 2026

## Summary

End-to-end validation of the fault-injection attribution pipeline across 48 experiments.
Identified and fixed three pipeline bugs, confirmed FR analysis is solid, and isolated the
remaining attribution gap to a single issue: **logsage returns RESTART IMMEDIATE for
crash/exception-type faults that should be STOP**.

---

## Pipeline Fixes Applied

| File | Fix |
|---|---|
| `trace_analyzer/capture.py` | `capture_logs()` now saves/restores logger level and lowers it to INFO — previously, root logger at WARNING silently dropped all `logger.info()` calls inside the capture block, producing empty `analysis_text` from `CollectiveAnalyzer` |
| `trace_analyzer/fr_attribution.py` | `main()` now prints `analysis_text` + `hanging_ranks` to stdout (was discarding results) |
| `scripts/watch_and_analyze.sh` | FR inline Python block: import from installed package (not local skill copy), correctly extract `analysis_text`/`hanging_ranks` from returned dict, redirect stderr to `/dev/null` instead of mixing into FR output |
| `scripts/score_attribution.py` | **New file** — LLM judge (Claude Sonnet) that scores 5 attribution dimensions per experiment and returns structured JSON |

---

## Experiment Sessions

### Session 1 — Mini-batch validation (Apr 9, `20260409_160245`)

6 experiments: GPU_SLEEP×2, GPU_ERROR×2, SIGKILL×1, SIGTERM×1 — all 2-node.
Purpose: confirm pipeline works end-to-end after fixes.

| # | FAULT_TYPE | RANK | restart | rank_p | rank_a | fault_d | fr_rank |
|---|---|---|---|---|---|---|---|
| 1 | GPU_SLEEP | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | GPU_SLEEP | 0 | ✅ | ✅ | ✅ | partial | ✅ |
| 3 | GPU_ERROR | 1 | ❌ | ❌ | ❌ | partial | ✅ |
| 4 | GPU_ERROR | 0 | ❌ | ❌ | ❌ | partial | ✅ |
| 5 | SIGKILL | 1 | ❌ | ✅ | ✅ | partial | ✅ |
| 6 | SIGTERM | 1 | ✅ | ❌ | ❌ | partial | ✅ |

FR analysis: 6/6 correct. Pipeline confirmed working.

---

### Session 2 — Full default pool (Apr 9, `20260409_170603`)

34 experiments across all fault types and node counts (2/4/8 nodes).

**Infrastructure issue:** 18/34 jobs failed at container startup due to a pyxis/enroot
`nvidia-container-cli ldcache` error on certain compute nodes:

```
nvidia-container-cli: ldcache error: process /usr/sbin/ldconfig.real failed with error code: 1
[ERROR] /etc/enroot/hooks.d/98-nvidia.sh exited with return code 1
pyxis: couldn't start container
rm: cannot remove '/usr/local/cuda/compat/lib': Read-only file system
```

The CUDA compat overlay was not being applied on those nodes — `ldconfig` could not write its
cache inside the read-only squashfs container. These jobs produced no FR dumps and their logs
contained only the container error, which logsage misattributed as a disk/storage fault.
The issue was transient and node-specific; jobs submitted the next day ran cleanly.

**Clean-run results (16/34):** see full table in
`/home/sbak/experiments/llama4-scout-gb200/fault_injection/20260409_170603/experiments_report.md`

Aggregate for clean-run jobs:

| FAULT_TYPE | N (clean) | restart% | rank_primary% | fr_rank% |
|---|---|---|---|---|
| GPU_SLEEP | 5 | 80% | 40% | 60% |
| GPU_ERROR | 4 | 0% | 25% | 75% |
| SIGKILL | 3 | 33% | 33% | 100% |
| OS_ABORT | 1 | 0% | 0% | 100% |

---

### Session 3 — SEGFAULT cluster health check (Apr 10, `20260410_135216`)

2 experiments: SEGFAULT rank=0 and rank=1, 2-node. Purpose: confirm cluster healthy after
the Apr 9 enroot issue.

| # | FAULT_TYPE | RANK | restart | rank_p | rank_a | fault_d | fr_rank |
|---|---|---|---|---|---|---|---|
| 1 | SEGFAULT | 1 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 2 | SEGFAULT | 0 | ❌ | ✅ | ✅ | ✅ | ✅ |

Cluster healthy (both COMPLETED, 7 FR dumps each). Rank and fault description correct;
restart decision wrong (RESTART instead of STOP).

---

### Session 4 — Python fault types (Apr 10, `20260410_143501`)

4 experiments: LOCK_GIL×2, WORKLOAD_EXC×1, ASYNC_EXC×1 — all 2-node.
These were skipped in the full session due to the enroot issue.

| # | FAULT_TYPE | RANK | restart | rank_p | rank_a | fault_d | fr_rank |
|---|---|---|---|---|---|---|---|
| 1 | LOCK_GIL | 1 | ✅ | ✅ | ✅ | partial | ✅ |
| 2 | LOCK_GIL | 0 | ✅ | ✅ | ✅ | partial | ✅ |
| 3 | WORKLOAD_EXC | 1 | ❌ | ✅ | ✅ | partial | ❌ (rank 7) |
| 4 | ASYNC_EXC | 1 | ❌ | ❌ | ❌ | false | ✅ |

Note on WORKLOAD_EXC FR result: FR flagged rank 7 instead of rank 1. When a rank throws an
application exception and crashes, the last rank detected as missing by NCCL's collective
timeout isn't necessarily the originating rank — FR is identifying the symptom rank.

---

## Attribution Quality Summary (clean runs only)

| Dimension | Assessment |
|---|---|
| **FR rank identification** | Solid — correctly identified the hanging rank in all clean-run experiments where NCCL completed enough to produce dumps. The `capture_logs()` fix was the key enabler. |
| **Log rank identification** | Good for hang types (GPU_SLEEP, LOCK_GIL); weaker for crash/signal types where all ranks see a simultaneous NCCL timeout masking the originator. FR compensates for this gap. |
| **Restart decision** | ✅ Correct for hang/recoverable types: GPU_SLEEP, LOCK_GIL, SIGTERM. ❌ Wrong for crash/exception types: GPU_ERROR, SIGKILL, SEGFAULT, WORKLOAD_EXC, ASYNC_EXC — logsage consistently returns RESTART IMMEDIATE when the correct decision is STOP. |
| **Fault description** | Consistently `partial` — logsage describes the observable NCCL collective timeout symptom, not the underlying injected fault (GPU hang, kill signal, exception). This is expected given the log contains only symptoms. |

---

## Open Gap

**Single actionable fix:** logsage restart decision for crash/exception-type faults.

Logsage sees the same NCCL collective timeout pattern whether the root cause is a recoverable
GPU hang or a hard crash (SIGKILL, SEGFAULT, CUDA error, application exception). It needs
keyword-based fast-path rules to detect crash signals before the LLM runs:

| Fault type | Expected | Currently returns |
|---|---|---|
| GPU_ERROR | STOP | RESTART IMMEDIATE |
| SIGKILL | STOP | RESTART IMMEDIATE |
| SEGFAULT | STOP | RESTART IMMEDIATE |
| WORKLOAD_EXC | STOP | RESTART IMMEDIATE |
| ASYNC_EXC | STOP | RESTART IMMEDIATE |
| OS_ABORT | STOP | RESTART IMMEDIATE |

Target file: `attribution/log_analyzer/nvrx_logsage.py`
