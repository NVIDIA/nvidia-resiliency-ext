---
name: nvrx-create
description: Transform an original sbatch script into an InJob-enabled version. Use when the user has an sbatch script and wants to add ft_launcher / fault tolerance, or asks how to make a training script restartable.
argument-hint: <original_sbatch> [--profile <profile.yaml>] [--hot-spare <N>] [--fault-schedule "cycle:rank,..."]
allowed-tools: [Read, Write]
---

Transform an original sbatch script into an InJob-enabled version.

## Usage
```
/nvrx-create <original_sbatch> [--profile <profile.yaml>] [--hot-spare <N>] [--fault-schedule "cycle:rank,..."]
```

- `<original_sbatch>`: the original single-job sbatch (without InJob)
- `--profile`: optional YAML to override paths and settings. Without this, all original paths are preserved as-is.
- `--hot-spare <N>`: number of hot spare racks. Default 0 = single-job InJob. N≥1 = convert to job-array with N hot spare rack(s). Overrides `slurm.hot_spare` in profile if both present.
- `--fault-schedule`: set `NVRX_INJECT_GPU_FAILURE` spec, e.g. `"2:33,4:33,6:33"`. Only relevant when `--profile` provides a different `run_dir`.

Output: `<name>_nvrx.sh` — always one file.

If no arguments are given, ask the user for the original sbatch path.

## Profile YAML format

The profile has four sections. All sections and all fields are optional — only specified fields are overridden.

```yaml
slurm:
  account: "coreai_dlalgo_llm"
  partition: "batch_block1,batch_block2"
  qos: "batch_large_long"    # optional; overrides original --qos if present
  nodes: 16          # nodes per array task (only relevant when --hot-spare > 0)

model:
  run_dir: "/lustre/fsw/.../hexinw/run"
  image: "/lustre/fs1/.../nvrx.sqsh"
  megatron_dir: "/lustre/fsw/.../megatron-lm"
  # Any Megatron OPTIONS arg can be overridden here using exact flag names (without --).
  # If present, these override the corresponding args in the original script's OPTIONS.
  num-layers: 27
  global-batch-size: 384
  train-samples: 3051757
  lr-warmup-samples: 24414
  lr-decay-samples: 3048706
  lr-wsd-decay-samples: 610351

ft_launcher:
  # Exact ft_launcher flag names (without --). All are optional with built-in defaults.
  nnodes-min: 32
  max-restarts: 10
  ft-segment: 16
  ft-rank-section-timeouts: "setup:1200,step:90,checkpointing:300"
  ft-rank-out-of-section-timeout: 120
  rdzv-conf: "join_timeout=1200,store_connect_wait_seconds=600"
  ft-node-health-check-endpoint: "/var/run/nvhcd/nvhcd.sock"
  ft-max-no-progress-cycles: 10

```

> **Note:** SSH/cluster config is not part of this profile. To submit the generated script to a remote cluster, use `/nvrx-submit <script> --ssh-config <ssh_config.yaml>`.

## Your task

1. **Read** the original sbatch script at the given path.
2. **Read** the profile YAML if provided.
3. **Generate** the output script by applying the transformations below.
4. **Write** the output file.
5. **Print a diff summary** of every change made (grouped by section) so the researcher can review.

## Transformation rules

Apply ALL of the following. Rules marked "(with profile)" only apply when `--profile` is provided.

### Mode selection: single-job vs job-array

**Determine mode first**, before applying any other rules:

- **Single-job mode** (default): `--hot-spare` is absent or 0. Keep the original `--nodes=<N>`. Do NOT add `--array`. Simpler coordination (no cross-task sync needed).
- **Job-array mode**: `--hot-spare N` where N≥1. Convert to job array. Compute sizing from original `--nodes=<N>`:

```
nodes_per_task   = slurm.nodes if present, else 16   # NVL72: 1 rack = 16 nodes
active_tasks     = N / nodes_per_task                 # one array task per rack (training tasks)
concurrency      = active_tasks + slurm.hot_spare     # running tasks including hot spare(s)
cold_spare       = 1                                  # always one queued cold spare
total_tasks      = concurrency + cold_spare
array_spec       = "0-{total_tasks-1}%{concurrency}"
ACTIVE_SLURM_ARRAY_TASKS = concurrency
```

Example: `--nodes=768`, `hot_spare=1` → 48 training + 1 hot spare → `--array=0-49%49`, `ACTIVE_SLURM_ARRAY_TASKS=49`

### SBATCH header changes

General rule: preserve all original `#SBATCH` parameters as-is. The profile `slurm` section may override any of them by name; if an original parameter is absent and the profile doesn't add it, do not add it. If a profile value is `null` (YAML `~`), remove that header from the output entirely.

Exceptions and additions on top of that rule:
- `--ntasks-per-node`: always change to 1 (ft_launcher runs as one task per node).
- `--comment` APS JSON: add `"nvrx": "enabled"` field. Preserve all other fields.
- (with profile) Also add `"environment": "dev"` to the APS JSON comment.
- **(job-array mode only)**: change `--nodes` to `nodes_per_task` (16); add `--array=<array_spec>`.

### Environment setup block

After the existing env var exports, add:
```bash
export NVRX_GPUS_PER_NUMA=2
export NVRX_LOG_DEBUG=true   # with profile only
```

### Path variable changes

Keep `BASE_DIR` unchanged (researcher's base). Keep `LOGS_DIR`, `CHECKPOINT_DIR`, `TENSORBOARD_DIR` unchanged (derived from researcher's RUN_DIR).

Add after existing path vars:
```bash
NVRX_DIR="${RUN_DIR}/nvrx/"
mkdir -p ${NVRX_DIR}
```

- (with profile) Override `RUN_DIR` to `model.run_dir` from profile.
- (with profile) Override `IMAGE` to `model.image` from profile.
- (with profile) Override `MEGATRON_LM_DIR` to `model.megatron_dir` from profile.
- (no profile) Append `+nvrx` to the container image tag/filename (before `.sqsh` extension or at end).

In the `if [ -n "${SLURM_JOB_ID:-}" ]` block, keep `SLURM_JOB_ID` in the condition and `SCRIPT_PATH`. Change `ENV_LOG_FILENAME` to use `SLURM_ARRAY_JOB_ID` in job-array mode, or keep `SLURM_JOB_ID` in single-job mode:
```bash
# job-array mode
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_ARRAY_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

# single-job mode — keep original as-is (SLURM_JOB_ID throughout)
```

### DATETIME coordination block

- **(job-array mode)** Replace simple assignment with array-task-0-writes pattern:
```bash
DATETIME_FILE=${NVRX_DIR}/datetime_${SLURM_ARRAY_JOB_ID}.txt
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    echo "$DATETIME" > "$DATETIME_FILE"
fi
while [[ ! -f "$DATETIME_FILE" ]]; do
    sleep 1
done
DATETIME=$(cat "$DATETIME_FILE")
```
- **(single-job mode)** Keep the original `DATETIME=$(date ...)` unchanged.

### Env logging block

- **(job-array mode)** Replace tee-based logging with flock-guarded block (only task 0 writes PATHS+GIT; all tasks write ENV):
```bash
ENV_LOG_LOCK=${NVRX_DIR}/env_log_${SLURM_ARRAY_JOB_ID}.lock
ENV_LOG_INITIALIZED=${NVRX_DIR}/env_log_initialized_${SLURM_ARRAY_JOB_ID}

(
    flock -x 200

    if [[ ! -f "$ENV_LOG_INITIALIZED" ]]; then
        # <existing PATHS and GIT logging lines here>
        touch "$ENV_LOG_INITIALIZED"
    fi

    # <existing ENV logging lines here>

) 200>"$ENV_LOG_LOCK"
```
- **(single-job mode)** Keep the original tee-based logging unchanged.

### RDZV coordination block

- **(job-array mode)** Add before the LAUNCHER_ARGS block:
```bash
# Array element 0 publishes rendezvous host
RDZV_FILE=${NVRX_DIR}/rdzv_host_${SLURM_ARRAY_JOB_ID}
RDZV_CLOSED=${NVRX_DIR}/rdzv_closed_${SLURM_ARRAY_JOB_ID}

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    RDZV_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
    echo "$RDZV_HOST" > "$RDZV_FILE"
    rm -f "$RDZV_CLOSED"
    trap "echo 'Task 0 exiting, creating RDZV_CLOSED file'; touch '$RDZV_CLOSED'" EXIT
elif [[ -f "$RDZV_CLOSED" ]]; then
    echo "RDZV_CLOSED file detected. Task 0 has exited. Exiting task ${SLURM_ARRAY_TASK_ID} with failure."
    exit 1
fi

while [[ ! -f "$RDZV_FILE" ]]; do
    sleep 1
done
export RDZV_HOST=$(cat "$RDZV_FILE")
export RDZV_PORT=29400
```
- **(single-job mode)** No RDZV coordination block. Set RDZV_HOST directly:
```bash
export RDZV_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export RDZV_PORT=29400
```

### NVRX_INJECT_GPU_FAILURE (with --fault-schedule only)

Add before the LAUNCHER_CMD line:
```bash
export NVRX_INJECT_GPU_FAILURE="<value from --fault-schedule arg>"
```
If `--fault-schedule` is not provided, omit this line (or add a commented placeholder).

### Megatron OPTIONS changes

Prepend to the OPTIONS variable value:
```
--enable-ft-package \
--ft-num-warmup-iters 5 \
```

(with profile) Also add fault injector args. Use values from profile `ft_launcher` block if present for these keys, otherwise use these defaults:
```
--fault-injector-ranks 11 \
--fault-injector-fault-delay 900 \
--fault-injector-fault-types async_exc,gpu_error,lock_gil \
```

Override any Megatron OPTIONS args present under `model` section (keys other than `run_dir`, `image`, `megatron_dir`). Use exact flag names — e.g. `num-layers: 27` in profile overrides `--num-layers <old>` in OPTIONS. If an arg is not present in the original OPTIONS, append it.

### LAUNCHER_ARGS block

Replace `RUN_CMD="python -u ..."` with a ft_launcher invocation. Add before the srun block.

**(job-array mode)**:
```bash
LAUNCHER_CMD="ft_launcher"
ACTIVE_SLURM_ARRAY_TASKS=<concurrency: active_tasks + hot_spare>
NVRX_PER_CYCLE_APPLOG_PREFIX=${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${DATETIME}.log

LAUNCHER_ARGS=" \
    --nnodes $(( SLURM_JOB_NUM_NODES * (ACTIVE_SLURM_ARRAY_TASKS - 1) )):$(( SLURM_JOB_NUM_NODES * ACTIVE_SLURM_ARRAY_TASKS )) \
    --nproc-per-node <ntasks_per_node_from_original> \
    --ft-segment=<ft_launcher.ft-segment or default 16> \
    --rdzv-id ${SLURM_ARRAY_JOB_ID} \
    --rdzv-endpoint ${RDZV_HOST}:${RDZV_PORT} \
    --rdzv-conf <ft_launcher.rdzv-conf or default join_timeout=1200,store_connect_wait_seconds=600> \
    --ft-per-cycle-applog-prefix=${NVRX_PER_CYCLE_APPLOG_PREFIX} \
    --ft-nvrx-logfile=${NVRX_PER_CYCLE_APPLOG_PREFIX} \
    --ft-cycle-info-dir=${NVRX_DIR}/cycle_infos \
    --ft-checkpoint-iteration-file=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt \
    --ft-node-health-check-endpoint=<ft_launcher.ft-node-health-check-endpoint or default /var/run/nvhcd/nvhcd.sock> \
    --max-restarts <ft_launcher.max-restarts or default 10> \
    --ft-rank-section-timeouts <ft_launcher.ft-rank-section-timeouts or default setup:1200,step:90,checkpointing:300> \
    --ft-rank-out-of-section-timeout <ft_launcher.ft-rank-out-of-section-timeout or default 120> \
"

RUN_CMD="${LAUNCHER_CMD} ${LAUNCHER_ARGS} <original_training_script> ${OPTIONS}"
```

**(single-job mode)**:
```bash
LAUNCHER_CMD="ft_launcher"
NVRX_PER_CYCLE_APPLOG_PREFIX=${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${DATETIME}.log

LAUNCHER_ARGS=" \
    --nnodes <ft_launcher.nnodes-min if present, else ${SLURM_JOB_NUM_NODES}> \
    --nproc-per-node <ntasks_per_node_from_original> \
    --ft-segment=<ft_launcher.ft-segment or default 16> \
    --rdzv-id ${SLURM_JOB_ID} \
    --rdzv-endpoint ${RDZV_HOST}:${RDZV_PORT} \
    --rdzv-conf <ft_launcher.rdzv-conf or default join_timeout=1200,store_connect_wait_seconds=600> \
    --ft-per-cycle-applog-prefix=${NVRX_PER_CYCLE_APPLOG_PREFIX} \
    --ft-nvrx-logfile=${NVRX_PER_CYCLE_APPLOG_PREFIX} \
    --ft-cycle-info-dir=${NVRX_DIR}/cycle_infos \
    --ft-checkpoint-iteration-file=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt \
    --ft-node-health-check-endpoint=<ft_launcher.ft-node-health-check-endpoint or default /var/run/nvhcd/nvhcd.sock> \
    --max-restarts <ft_launcher.max-restarts or default 10> \
    --ft-rank-section-timeouts <ft_launcher.ft-rank-section-timeouts or default setup:1200,step:90,checkpointing:300> \
    --ft-rank-out-of-section-timeout <ft_launcher.ft-rank-out-of-section-timeout or default 120> \
"

RUN_CMD="${LAUNCHER_CMD} ${LAUNCHER_ARGS} <original_training_script> ${OPTIONS}"
```

### srun invocation changes

Replace the original srun with:
```bash
srun \
     --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --mpi=none \
     --container-image=${IMAGE} \
     --container-mounts="/lustre:/lustre" \
     --container-env=<preserve original --container-env list> \
     sh -c "${RUN_CMD}"
```
Remove `srun -l` output labeling. Remove `--output` flag from srun (per-cycle log prefix handles log routing). Keep `--container-mounts` and `--container-env` from original.

## Output

Write each output file, then print a diff summary like:

```
=== Changes made to produce <output_file> ===

[SBATCH headers]
  + --array=0-2%2
  ~ --nodes: 768 → 16
  ~ --ntasks-per-node: 4 → 1
  ~ --comment: added "nvrx": "enabled"

[Paths]
  + NVRX_DIR="${RUN_DIR}/nvrx/"
  ~ IMAGE: added +nvrx suffix

[New sections added]
  + DATETIME_FILE coordination block
  + flock-based env logging
  + RDZV_FILE/RDZV_CLOSED coordination block
  + LAUNCHER_ARGS block

[OPTIONS]
  + --enable-ft-package
  + --ft-num-warmup-iters 5
  ~ --num-layers: 96 → 27
  ~ --global-batch-size: 1024 → 384

[srun]
  ~ --ntasks-per-node: 4 → 1
  ~ command: python → ft_launcher wrapper
  - removed: -l flag
  - removed: --output flag
```
