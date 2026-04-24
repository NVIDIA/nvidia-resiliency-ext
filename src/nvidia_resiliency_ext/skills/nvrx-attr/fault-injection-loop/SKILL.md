---
name: fault-injection-loop
description: >
  Closed-loop fault injection and attribution accuracy benchmark. Draws from a
  prioritized pool of (fault_type, rank, iter, nodes) experiments and submits them
  2 at a time via sbatch — waiting for each pair to finish before submitting the
  next — to bound filesystem load. GPU-related faults are front-loaded in the pool.
  After all jobs complete, runs /log-analysis and /fr-analysis on every experiment,
  scores attribution vs. ground truth, aggregates gaps, and iterates on attribution
  modules to close them.
compatibility: Requires SLURM cluster access, sbatch, NVIDIA_API_KEY, langchain-openai, logsage, and nvidia-resiliency-ext installed. This workflow has only been validated with Megatron-LM workloads.
metadata:
  author: nvidia
  sub-skills: [log-analysis, fr-analysis]
---

# Skill: fault-injection-loop

Iterative closed-loop skill that runs a prioritized fault-injection experiment pool
2 jobs at a time, analyzes every artifact, scores attribution accuracy, aggregates
gaps across the matrix, and proposes targeted improvements to attribution modules.

---

## Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│  0. POOL     → build ordered pool of (fault_type, rank, iter, nodes)  │
│               GPU faults first, then crash, Python-hang, signal       │
│                                                                        │
│  repeat until pool exhausted:                                          │
│  1. SUBMIT   → sbatch 2 jobs from pool head                            │
│  2. WAIT     → poll until both jobs leave RUNNING/PENDING              │
│                                                                        │
│  after all jobs done:                                                  │
│  3. ANALYZE  → watch_and_analyze.sh: /log-analysis + /fr-analysis     │
│               per completed job, streaming as jobs finish              │
│  4. SCORE    → compare attribution output vs injected ground truth     │
│  5. AGGREGATE→ build results table; identify systematic failure modes  │
│  6. IMPROVE  → patch log_analyzer/nvrx_logsage.py                     │
│  7. LOOP     → re-run same pool with updated attribution code          │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Step 0 — Fault Pool Design

The pool is defined as an ordered list of `FAULT_TYPE:RANK:ITER:NODES` entries
inside `scripts/prepare_node_alloc.sh`. Default pool (34 experiments, 17 batches):

```
# GPU hangs — highest priority; full rank sweep across all node counts
GPU_SLEEP:1:5:2   GPU_SLEEP:0:5:2      # 2-node: rank-1, rank-0
GPU_SLEEP:4:5:2   GPU_SLEEP:7:5:2      # 2-node: mid-rank, last-rank
GPU_SLEEP:1:5:4   GPU_SLEEP:0:5:4      # 4-node: rank-1, rank-0
GPU_SLEEP:8:5:4   GPU_SLEEP:15:5:4     # 4-node: mid, last
GPU_SLEEP:1:5:8   GPU_SLEEP:0:5:8      # 8-node: rank-1, rank-0
GPU_SLEEP:16:5:8  GPU_SLEEP:31:5:8     # 8-node: mid, last

# GPU errors — high priority; rank-0 and rank-1 across all node counts
GPU_ERROR:1:5:2   GPU_ERROR:0:5:2
GPU_ERROR:1:5:4   GPU_ERROR:0:5:4
GPU_ERROR:1:5:8   GPU_ERROR:0:5:8

# Crash faults
SIGKILL:1:5:2     SIGKILL:0:5:2
SIGKILL:1:5:4     SIGKILL:1:5:8
SEGFAULT:1:5:2    SEGFAULT:0:5:2
SEGFAULT:1:5:4    OS_ABORT:1:5:2

# Python-level hangs
LOCK_GIL:1:5:2    LOCK_GIL:0:5:2
WORKLOAD_EXC:1:5:2  ASYNC_EXC:1:5:2

# Signals
SIGTERM:1:5:2     SIGINT:1:5:2
SIGSTOP:1:5:2     SIGNAL_EXC:1:5:2
```

Rank coverage per node count (4 GPUs/node):

| Nodes | Total ranks | rank-0 | rank-1 | mid | last |
|-------|-------------|--------|--------|-----|------|
| 2     | 8           | 0      | 1      | 4   | 7    |
| 4     | 16          | 0      | 1      | 8   | 15   |
| 8     | 32          | 0      | 1      | 16  | 31   |

To run a custom subset, override `POOL` before calling the script:
```bash
POOL="GPU_SLEEP:0:5:2 GPU_SLEEP:1:5:2" bash scripts/prepare_node_alloc.sh
```

## Local User Config

Start from the tracked template:

```bash
cp scripts/user.env.example scripts/user.env
```

Then edit `scripts/user.env` with cluster-specific settings. This file is
sourced by `run_session.sh`, `prepare_node_alloc.sh`, `watch_and_analyze.sh`, and
`l4_gb200_reduced.sh`, and it is intended to stay local and untracked.

Recommended contents:

```bash
PARTITION=gb-nvl-134-135
BASE_EXPERIMENTS_DIR="${HOME}/nvrx-attr-experiments"
MEGATRON_REPO_HOST_PATH="${HOME}/megatron-lm-main"
SHARED_TMP_BASE_DIR="${HOME}/tmp"
WORKSPACE_HOST_PATH="${HOME}/tmp"
CONTAINER_IMAGE="nvcr.io/nvidia/nemo:26.04"
NVIDIA_API_KEY_FILE="${HOME}/.nvidia_api_key"
JUDGE_API_KEY_FILE="${HOME}/.nvidia_api_key"
NVRX_LLM_MODEL="nvidia/nemotron-3-super-120b-a12b"
NVRX_LLM_BASE_URL="https://integrate.api.nvidia.com/v1"
JUDGE_MODEL="qwen/qwen3.5-397b-a17b"
JUDGE_BASE_URL="https://integrate.api.nvidia.com/v1"
FR_RACK_SIZE=32
```

Use `user.env` for stable site defaults such as partition, container image, and
host paths, plus local LLM credentials and endpoint settings for log-analysis
and the judge. Use per-run environment overrides for experiment-specific
controls such as `POOL`, `WORKLOAD`, `BATCH_SIZE`, `FAULT_TYPE`,
`FAULT_AT_ITER`, or `FAULT_DELAY`.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `WORKLOAD` | `llama4_scout` | Select a registered workload by name (see `scripts/workloads.conf`) |
| `ACCOUNT` | _(cluster default or `scripts/slurm.conf`)_ | SLURM account |
| `PARTITION` | _(cluster default or `scripts/slurm.conf`)_ | SLURM partition |
| `GPUS_PER_NODE` | `4` | GPUs per node |
| `TIME` | `00:30:00` | Per-job wall-clock limit |
| `BATCH_SIZE` | `2` | Jobs submitted per round |
| `POLL_INTERVAL` | `30` | Seconds between queue polls |
| `BASE_EXPERIMENTS_DIR` | `${HOME}/nvrx-attr-experiments` | Root for all output |
| `MEGATRON_REPO_HOST_PATH` | `${HOME}/megatron-lm-main` | Host path to the Megatron checkout mounted into the container |
| `SHARED_TMP_BASE_DIR` | `${HOME}/tmp` | Shared filesystem path used for cross-step coordination |
| `WORKSPACE_HOST_PATH` | `${HOME}/tmp` | Host path mounted at `/workspace` inside the container |
| `CONTAINER_IMAGE` | `nvcr.io/nvidia/nemo:26.04` | Container image used by the workload script |
| `NVIDIA_API_KEY_FILE` | _unset_ | File containing the log-analysis API key |
| `JUDGE_API_KEY_FILE` | _unset_ | File containing the judge API key |
| `NVRX_LLM_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Model for log-analysis |
| `NVRX_LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Base URL for log-analysis |
| `JUDGE_MODEL` | `qwen/qwen3.5-397b-a17b` | Model for judge scoring |
| `JUDGE_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Base URL for judge scoring |
| `FR_RACK_SIZE` | `32` | Ranks per rack for coarse FR scoring |
| `SBATCH_SCRIPT` | `scripts/l4_gb200_reduced.sh` | Job script to submit |
| `POOL` | _(default pool above)_ | Space-separated experiment triplets |

### Registered workloads (`scripts/workloads.conf`)

| Name | Script | Base dir | Description |
|---|---|---|---|
| `llama4_scout` | `l4_gb200_reduced.sh` | `${HOME}/nvrx-attr-experiments` | Llama4-Scout (reduced layers) on GB200; minimum supported size is 2 nodes |
| `n3_super` | `n3_super_gb200_fi.sh` | `${HOME}/nvrx-attr-experiments` | Nemotron3-Super on GB200; minimum supported size is 8 nodes |

Workload note:
- `llama4_scout` requires at least 2 nodes.
- `n3_super` requires at least 8 nodes. Its default registered pool contains only 8-node experiments.

```bash
# Run the full pool against the validated example workload
bash scripts/prepare_node_alloc.sh

# Run a custom subset against llama4_scout
POOL="GPU_SLEEP:1:5:2 SIGKILL:1:5:2" WORKLOAD=llama4_scout bash scripts/prepare_node_alloc.sh
```

---

## Step 1 & 2 — Batched Submission + Wait (automated)

```bash
bash scripts/prepare_node_alloc.sh
```

The script loops: submit 2 jobs → poll `squeue` every 30 s until both finish →
submit next 2. Progress is printed inline:

```
>>> Batch 1: experiments 1–2 of 34
  submitted: GPU_SLEEP rank=1  iter=5 nodes=2 -> job=1850
  submitted: GPU_SLEEP rank=0  iter=5 nodes=2 -> job=1851
  waiting for GPU_SLEEP:1:5:2 GPU_SLEEP:0:5:2 (1850,1851) ... 30s 60s done.
>>> Batch 2: experiments 3–4 of 34
  ...
```

A session directory and TSV tracking file are created at launch time:
```
${BASE_EXPERIMENTS_DIR}/fault_injection/<YYYYMMDD_HHMMSS>/
  experiments.tsv                              ← tracking file (all job IDs + paths)
  n<N>_<FAULT>_r<R>_i<I>/                     ← one subdir per experiment
    logs/slurm/<JOB_ID>.launch.out
    logs/slurm/<JOB_ID>.*.1.main_workload.log  ← log-analysis input
    checkpoints/                               ← fr-analysis input (FR dumps)
    tensorboard/
  experiments_report.md                        ← generated by watch_and_analyze.sh
```

Tracking file columns: `JOB_ID  FAULT_TYPE  RANK  ITER  NODES  EXPERIMENT_DIR`

---

## Step 3 — Analyze All Experiments

Run the watcher/analyzer — it reads the tracking file and processes each experiment
as its job state leaves RUNNING/PENDING (works whether jobs are still running or
already done):

```bash
bash scripts/watch_and_analyze.sh \
    ${BASE_EXPERIMENTS_DIR}/fault_injection/<YYYYMMDD_HHMMSS>/experiments.tsv
```

The watcher:
1. Reads each row from the tracking TSV
2. Calls `nvrx_logsage.py --exclude_nvrx_logs` and parses the text output to get
   `restart_decision` and `attribution_text`
3. Calls FR analysis as `python -m nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution --fr-path "${EXPERIMENT_DIR}/checkpoints" -p "_dump_*"` and passes the raw table output to the judge
4. Scores 7 dimensions (restart correctness, rank primary, rank any, category, type, FR rank)
5. Appends a scored row to `<session>_report.md` as a markdown table row
6. Repeats until all experiments are analyzed

To also run the sub-skills interactively for a single experiment:
```bash
/log-analysis --log-path "${EXPERIMENT_DIR}/logs/slurm/${JOB_ID}.*.1.main_workload.log"
/fr-analysis  --fr-path "${EXPERIMENT_DIR}/checkpoints" -p "_dump_*"
```

---

## Step 4 — Score Each Experiment

Scoring is performed by `scripts/score_attribution.py`, an LLM judge that
receives the ground truth, the filtered raw log, the logsage attribution output, and the FR
analysis output, then returns structured JSON scores with a reasoning note.

| Column | Values | Meaning |
|---|---|---|
| **restart_correct** | `true` / `false` / `N/A` | Restart decision matches expected for this fault type |
| **rank_primary** | `true` / `false` / `partial` | Injected rank is the primary root-cause in attribution |
| **rank_any** | `true` / `false` | Injected rank mentioned anywhere in attribution |
| **fault_described** | `true` / `false` / `partial` | Fault nature (hang/crash/signal/exception) correctly described |
| **fr_rank_correct** | `rank` / `node` / `rack` / `false` / `no_dumps` | FR analysis narrows correctly to the injected rank, node, rack, or fails to narrow usefully |
| **judge_notes** | string | One-sentence summary of the main gap or confirmation |

The judge is given:
1. Ground truth: `fault_type`, `rank`, `iter`, `nodes`
2. Expected restart decision + rationale (derived from `score_attribution.py:_RESTART_TABLE`)
3. Filtered raw log (last 400 lines, same `exclude_nvrx_logs` filtering as logsage)
4. Raw logsage stdout (5-field text format)
5. Raw FR analysis table output from `fr_attribution.py --fr-path ... -p "_dump_*"`
6. `GPUS_PER_NODE` and `FR_RACK_SIZE` to map the injected rank to node and rack scopes for FR scoring

Default judge model: `qwen/qwen3.5-397b-a17b`. Override with `--model` in `score_attribution.py`.
Default rack size for FR scope scoring: `32` ranks. Override with `FR_RACK_SIZE`.

---

## Step 5 — Aggregate Results

The canonical output of the loop is the markdown table in `<session>_report.md`.
When summarizing results for users, prefer linking to that file and reproducing the
same table shape rather than flattening the results into plain prose.

The report markdown table from `watch_and_analyze.sh` gives a matrix view. Look for
patterns across rows:

```
| FAULT_TYPE | NODES | RANK | restart_correct | rank_primary | rank_any | fault_described | fr_rank_correct | judge_notes |
|------------|-------|------|-----------------|--------------|----------|-----------------|-----------------|-------------|
| GPU_SLEEP  |   2   |  0   |      true       |    false     |   true   |      true       |      true       | rank-0 identified only in secondary issues |
| GPU_SLEEP  |   2   |  1   |      true       |     true     |   true   |      true       |      true       | correct on all dimensions |
| GPU_ERROR  |   2   |  1   |      false      |    false     |  false   |     partial     |      true       | LLM issued RESTART; rank not mentioned |
| SIGKILL    |   2   |  0   |      true       |    false     |  false   |     false       |      true       | attribution describes timeout not kill signal |
```

Common failure mode patterns and their meaning:

| Pattern | Interpretation |
|---|---|
| `rank_primary=false`, `rank_any=true` | Rank detected but treated as collateral; logsage putting it in secondary issues |
| `rank_any=false` for rank-0 | Rank-0 hang silences watchdog on other ranks; logsage lacks rank-0 signal |
| `fault_described=partial` for crash types | Crash keywords present but fault type not specifically named |
| `restart_correct=false` for GPU_ERROR | LLM conflating hardware error with recoverable hang |
| `fr_rank_correct=no_dumps` | NCCL watchdog did not fire before job ended — adjust `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` |
| `fr_rank_correct=node` | FR isolated the correct node but not the exact rank |
| `fr_rank_correct=rack` | FR isolated the correct rack-sized rank group but not the exact node/rank |

---

## Step 6 — Identify and Apply Improvements

### FR analysis
Deterministic graph algorithm — **do not modify automatically**.
Note misidentifications and escalate to the team.

### Log analysis (safe to modify)

| Observation | Target location | Suggested fix |
|---|---|---|
| Wrong restart for hang | `nvrx_logsage.py` fast-path | Strengthen NCCL timeout → `RESTART IMMEDIATE` mapping |
| Missing rank in attr text | `nvrx_logsage.py` prompt | Extract rank from NCCL watchdog lines; add regex |
| Crash misclassified as hang | `nvrx_logsage.py` | Add SIGKILL/SEGFAULT/GPU_ERROR keyword patterns |
| `ERRORS NOT FOUND` when errors exist | `return_application_errors` config | Loosen error extraction filter |
| rank-0 not detected | prompt or fast-path | Add explicit rank-0 hang heuristic (other ranks silent) |
| attr off by many iters | prompt context | Increase weight of iteration-stamped log lines |
| LLM wrong on GPU_ERROR | prompt | Distinguish `cudaError` → crash from NCCL timeout → hang |

Editable file: `attribution/log_analyzer/nvrx_logsage.py`

After each patch, re-run the same pool subset that previously failed:
```bash
POOL="GPU_SLEEP:0:5:2 GPU_ERROR:1:5:2" bash scripts/prepare_node_alloc.sh
```

---

## Step 7 — Loop

Increment experiment counter. Suggested sweep order across code-change iterations:

1. **Iteration 1**: full default pool (34 experiments)
2. **Iteration 2**: targeted re-run of all failing cells from iteration 1
3. **Iteration 3**: expand iter dimension (FAULT_AT_ITER=2 and 10) for remaining gaps
4. **Iteration 4**: add SEGFAULT and LOCK_GIL 4-node/8-node coverage

Stop condition: all cells pass all four scoring dimensions for two consecutive
code-change iterations.

---

## Adapting A SLURM Script For The Feedback Loop

The feedback loop is not tied to `l4_gb200_reduced.sh`, but your sbatch script must
match a small contract so the loop can submit, analyze, and score each run.

Required changes for a custom workload script:

1. Accept these exported variables from `prepare_node_alloc.sh`:
   `FAULT_TYPE`, `FAULT_RANK`, `FAULT_AT_ITER`, `EXPERIMENT_DIR`, `BASE_EXPERIMENTS_DIR`,
   and `GPUS_PER_NODE`.
2. Write the main training log to:
   `${EXPERIMENT_DIR}/logs/slurm/${SLURM_JOB_ID}.*.1.main_workload.log`
   so `watch_and_analyze.sh` can find it.
3. Write NCCL flight-recorder dumps under `${EXPERIMENT_DIR}/checkpoints/`.
4. Emit a fault-injection marker when the fault is injected.
   `watch_and_analyze.sh` uses this to decide whether the run reached the injection point.
5. Preserve the per-experiment directory layout:
   `logs/slurm/`, `checkpoints/`, and `tensorboard/`.

This has only been validated with Megatron-LM because the current run-valid check and
fault markers depend on Megatron's `debug_fault_injection.py` behavior. If you adapt the
loop to another framework, update both the sbatch script and `watch_and_analyze.sh`.

## Appendix A: SBATCH_SCRIPT fault parameters

The example `SBATCH_SCRIPT` reads these env vars from `prepare_node_alloc.sh` via `--export`:

| Variable | Default | Description |
|---|---|---|
| `FAULT_AT_ITER` | `5` | Training iteration at which to inject |
| `FAULT_DELAY` | `15` | Delay in seconds before fault injection after the iteration anchor |
| `FAULT_RANK` | `1` | Global rank to inject `[0, total_ranks)` |
| `FAULT_TYPE` | `GPU_SLEEP` | Megatron fault type enum name |
| `GPUS_PER_NODE` | `4` | GPUs per node (used to compute `TOTAL_TASKS`) |
| `EXPERIMENT_DIR` | `${BASE_EXPERIMENTS_DIR}/fault_injection/n${SLURM_NNODES}_${FAULT_TYPE}_r${FAULT_RANK}_i${FAULT_AT_ITER}` | Per-experiment output root |
| `BASE_EXPERIMENTS_DIR` | `${HOME}/nvrx-attr-experiments` | Shared root (datacache, triton/inductor caches) |

Valid `FAULT_TYPE` values:
`GPU_ERROR`, `GPU_SLEEP`, `WORKLOAD_EXC`, `ASYNC_EXC`, `SIGNAL_EXC`, `OS_ABORT`,
`LOCK_GIL`, `SEGFAULT`, `SIGINT`, `SIGKILL`, `SIGTERM`, `SIGSTOP`

---

## Appendix B: Single-experiment manual run

```bash
# Manual runs land under fault_injection/manual/ by default (no session dir needed)
EXPERIMENT_DIR=${HOME}/nvrx-attr-experiments/fault_injection/manual/n2_GPU_SLEEP_r1_i5
mkdir -p ${EXPERIMENT_DIR}/logs/slurm ${EXPERIMENT_DIR}/checkpoints ${EXPERIMENT_DIR}/tensorboard

sbatch \
    --nodes=2 \
    --output=${EXPERIMENT_DIR}/logs/slurm/%j.launch.out \
    --error=${EXPERIMENT_DIR}/logs/slurm/%j.launch.err \
    --export=ALL,FAULT_TYPE=GPU_SLEEP,FAULT_RANK=1,FAULT_AT_ITER=5,FAULT_DELAY=15,GPUS_PER_NODE=4,EXPERIMENT_DIR=${EXPERIMENT_DIR} \
    scripts/l4_gb200_reduced.sh
```

Optional site-specific cleanup:

```bash
export CONTAINER_CLEANUP_CMD='
ENROOT_DIR="/var/lib/enroot/data/$(id -u)"
rm -rf "${ENROOT_DIR:?}"/* 2>/dev/null || true
echo "$(hostname): / $(df -h / | tail -1 | awk "{print \$3\" used, \"\$4\" avail\"}")"
'
```
