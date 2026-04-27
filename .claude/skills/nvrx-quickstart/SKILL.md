---
name: nvrx-quickstart
description: Try InJob on a small 1–2 node example using the built-in train_ddp_sections_api.py. Injects a fault to demonstrate a live restart. Use when the user is new to InJob, wants to see a restart without a production script, or asks for a quick demo.
argument-hint: [--nodes <N>] [--nproc-per-node <N>] [--device cuda|cpu] [--image <path>] [--run-dir <path>] [--ssh-config <yaml>]
allowed-tools: [Read, Write, Bash, Glob]
---

Try InJob on a small 1–2 node example using the built-in train_ddp_sections_api.py. Injects one fault so you can watch a restart happen.

## Usage
```
/nvrx-quickstart [--nodes <N>] [--nproc-per-node <N>] [--device cuda|cpu]
                  [--image <path>] [--run-dir <path>] [--ssh-config <yaml>]
```

Defaults: `--nodes 1`, `--nproc-per-node 4`, `--device cuda`.

All args optional — ask interactively for any missing required values
(`ssh_host`, `ssh_user`, `image`, `run_dir`).

## SSH config YAML (optional)
```yaml
ssh_host: "my-cluster-login.example.com"
ssh_user: "myuser"
```
If not provided, prompt interactively.

## Your task

### Step 1 — Gather config

Resolve `ssh_host`, `ssh_user`, `image`, `run_dir` from args, YAML, or interactive prompts.
Open SSH ControlMaster (reuse if alive):

```bash
SSH_USER="<ssh_user>"
SSH_HOST="<ssh_host>"
CTRL_SOCK="/tmp/nvrx_ssh_${SSH_USER}_${SSH_HOST}.ctrl"
SSH_OPTS="-o ControlMaster=no -o ControlPath=${CTRL_SOCK} -o StrictHostKeyChecking=no"

if ! ssh -O check -o ControlPath=${CTRL_SOCK} ${SSH_USER}@${SSH_HOST} 2>/dev/null; then
    ssh -nNf -o ControlMaster=yes -o ControlPath=${CTRL_SOCK} \
        -o ControlPersist=yes -o ServerAliveInterval=30 \
        -o StrictHostKeyChecking=no ${SSH_USER}@${SSH_HOST}
    for i in $(seq 1 15); do [ -S "${CTRL_SOCK}" ] && break; sleep 1; done
fi
```

### Step 2 — Stage example scripts

Create remote dirs and rsync the four example files:

```bash
ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} \
    "mkdir -p <run_dir>/quickstart/checkpoints <run_dir>/quickstart/nvrx/cycle_infos"

rsync -az -e "ssh ${SSH_OPTS}" \
    examples/fault_tolerance/train_ddp_sections_api.py \
    examples/fault_tolerance/dist_utils.py \
    examples/fault_tolerance/log_utils.py \
    examples/fault_tolerance/fault_tol_cfg_sections.yaml \
    ${SSH_USER}@${SSH_HOST}:<run_dir>/quickstart/
```

### Step 3 — Generate and submit sbatch

Generate this sbatch inline and write it to a temp file, then upload and submit.

**Parameter guidance for a clean single-restart demo:**

The fault fires `delay` seconds after cycle start (once per process lifetime). For cycle 1 to
complete before the second fault fires, training-time-per-cycle must satisfy:
`fault_delay < total_training_time < 2 × fault_delay`

Recommended defaults (tuned for ~60–90s/cycle on a modern GPU):
- `--simulated_fault rank_killed,60` (fault fires at ~60–64s)
- `--train_dataset_size 5000 --batch 8 --epochs 5 --save_interval 50`
- Estimated ~90–120s total training → cycle 0 runs ~60s then restarts; cycle 1 runs ~30–60s and completes

For CPU (`--device cpu`), use smaller model: add `--hidden 128 --train_dataset_size 500`.

**Generated sbatch:**

```bash
#!/bin/bash
#SBATCH --job-name=nvrx_quickstart
#SBATCH --nodes=<nodes>
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=<run_dir>/quickstart/nvrx_quickstart_%j.log

RUN_DIR="<run_dir>"
IMAGE="<image>"

mkdir -p ${RUN_DIR}/quickstart/checkpoints ${RUN_DIR}/quickstart/nvrx/cycle_infos

RDZV_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)

LAUNCHER_ARGS=" \
    --nnodes <nodes> \
    --nproc-per-node <nproc_per_node> \
    --rdzv-id ${SLURM_JOB_ID} \
    --rdzv-endpoint ${RDZV_HOST}:29400 \
    --max-restarts 3 \
    --ft-rank-section-timeouts setup:120,step:90,checkpoint:120 \
    --ft-rank-out-of-section-timeout 90 \
    --ft-cycle-info-dir ${RUN_DIR}/quickstart/nvrx/cycle_infos \
    --ft-per-cycle-applog-prefix ${RUN_DIR}/quickstart/nvrx_quickstart_${SLURM_JOB_ID} \
"

TRAIN_ARGS=" \
    --device <device> \
    --output_dir ${RUN_DIR}/quickstart/checkpoints \
    --train_dataset_size 5000 \
    --batch 8 \
    --epochs 5 \
    --save_interval 50 \
    --simulated_fault rank_killed,60 \
"
# For CPU: append: --hidden 128 --train_dataset_size 500

srun \
    --ntasks=${SLURM_JOB_NUM_NODES} \
    --ntasks-per-node=1 \
    --container-image=${IMAGE} \
    --container-mounts="${RUN_DIR}:${RUN_DIR}" \
    sh -c "cd ${RUN_DIR}/quickstart && ft_launcher ${LAUNCHER_ARGS} train_ddp_sections_api.py ${TRAIN_ARGS}"
```

Upload sbatch to `/tmp/nvrx_quickstart_<timestamp>.sh` on the cluster, submit via `sbatch`,
capture job_id. Write `~/.cache/nvrx_artifacts/<cluster>/<job_id>/run_result.json`:

```json
{
  "job_id": "<job_id>",
  "cluster": "<cluster>",
  "ssh_host": "<ssh_host>",
  "ssh_user": "<ssh_user>",
  "remote_run_dir": "<run_dir>/quickstart",
  "remote_nvrx_dir": "<run_dir>/quickstart/nvrx",
  "remote_logs_dir": "<run_dir>/quickstart",
  "artifacts_dir": "~/.cache/nvrx_artifacts/<cluster>/<job_id>",
  "final_state": "pending",
  "completed_at": null
}
```

Print: `Submitted: job_id=<job_id>  Watching...`

### Step 4 — Watch and narrate

Poll squeue every 15s. Also count cycle_infos on the remote:

```bash
STATE=$(ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} \
    "squeue -j <job_id> -h -o '%T' 2>/dev/null || echo 'DONE'")
CYCLES=$(ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} \
    "ls <run_dir>/quickstart/nvrx/cycle_infos/cycle_info.<job_id>.* 2>/dev/null | wc -l")
echo "[$(date '+%H:%M:%S')] state=${STATE:-DONE} cycles=${CYCLES}"
```

Apply same cold-spare-aware completion check as `nvrx-watch.md`:
- Empty/DONE → COMPLETED
- FAILED/CANCELLED/TIMEOUT/NODE_FAIL → terminal
- No RUNNING task but at least one COMPLETED → COMPLETED

On completion, collect artifacts (targeted rsync):

```bash
mkdir -p ~/.cache/nvrx_artifacts/<cluster>/<job_id>/logs ~/.cache/nvrx_artifacts/<cluster>/<job_id>/nvrx/cycle_infos

rsync -az \
    --include="cycle_infos/" --include="cycle_infos/cycle_info.<job_id>.*" --exclude="*" \
    -e "ssh ${SSH_OPTS}" \
    ${SSH_USER}@${SSH_HOST}:<run_dir>/quickstart/nvrx/ \
    ~/.cache/nvrx_artifacts/<cluster>/<job_id>/nvrx/

rsync -az \
    --include="*<job_id>*" --exclude="*" \
    -e "ssh ${SSH_OPTS}" \
    ${SSH_USER}@${SSH_HOST}:<run_dir>/quickstart/ \
    ~/.cache/nvrx_artifacts/<cluster>/<job_id>/logs/
```

Update `run_result.json` with actual `final_state` and `completed_at`.

### Step 5 — Narrate results

Read artifacts and produce a plain-language narrative (not a full report.md):

```bash
# Get cycle info
python tools/fault_tolerance/cycle_info_reader.py \
    ~/.cache/nvrx_artifacts/<job_id> --job-id <job_id>

# Get stage timings
LOG_PREFIX=$(ls ~/.cache/nvrx_artifacts/<cluster>/<job_id>/logs/*cycle0* 2>/dev/null | \
    head -1 | sed 's/_cycle0.*//')
python tools/fault_tolerance/parse_restart_metrics.py ${LOG_PREFIX} --json

# Find fault-fired line
grep -rh "Simulating fault\|rank_killed\|rank_hung" \
    ~/.cache/nvrx_artifacts/<cluster>/<job_id>/logs/ 2>/dev/null | head -5

# Find first failure_detected per cycle (UTC)
grep -rh "Event: failure_detected" ~/.cache/nvrx_artifacts/<cluster>/<job_id>/logs/ 2>/dev/null | \
    grep -oP "Cycle: \K\d+|Time: \K\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+" | \
    paste - - | sort -t$'\t' -k1,1n -k2,2 | awk '!seen[$1]++'
```

Print narrative to terminal:

```
=== InJob Quickstart Complete ===
Job <job_id> on <ssh_host>  Final state: <state>

Cycle 0  Training started on <active_nodes>
         Fault injected: rank_killed fired at ~<fire_time_utc> (rank <N>)
         → failure_detected: <utc>  (latency: ~<N>s)
         → terminate: <N>s  health_check: <N>s  rendezvous: <N>s  → restarted

Cycle 1  Training resumed from checkpoint
         → Completed normally  (no fault in this cycle)

InJob restart: SUCCESS ✓

What happened:
  ft_launcher detected the rank failure, terminated all workers across all
  nodes, ran a health check, re-ran rendezvous, and restarted training —
  all within the SAME SLURM allocation. No node reallocation. No requeue.

Restart overhead: ~<total_restart_time>s  (failure_detected → workers running)

Artifacts: ~/.cache/nvrx_artifacts/<cluster>/<job_id>/
To dig deeper:
  /nvrx-validate ~/.cache/nvrx_artifacts/<cluster>/<job_id>/   — full validation report
  /nvrx-run <your_script.sh> <cluster>             — test your own training script
```

If the job failed or more than one restart occurred (fault fired multiple times), explain
what happened and suggest adjusting the fault delay or dataset size.
