---
name: nvrx-watch
description: Poll a running InJob SLURM job and collect artifacts when it completes. Reads the state file written by nvrx-submit. Use after submitting a job with nvrx-submit, or when the user wants to monitor a job and gather logs and cycle infos.
argument-hint: [<artifacts_dir>] [--ssh-config <yaml>]
allowed-tools: [Read, Write, Bash]
---

Poll a running InJob SLURM job and collect artifacts when it completes.

## Usage
```
/nvrx-watch <state_file> [--poll-interval <seconds>] [--timeout <seconds>] [--artifacts-dir <path>]
```

- `<state_file>`: JSON state file written by `/nvrx-submit`
- `--poll-interval`: seconds between squeue polls (default 30)
- `--timeout`: max seconds to wait before giving up (default 14400)
- `--artifacts-dir`: local directory to collect artifacts into (default `~/.cache/nvrx_artifacts/<cluster>/<job_id>/`)

If no state file given, ask the user for job_id and cluster.

## Your task

1. Load the state file.
2. Reuse or open SSH ControlMaster (same fixed socket as nvrx-submit).
3. Poll job status until completion or timeout.
4. Collect artifacts via rsync.
5. Write `run_result.json`.

## SSH ControlMaster

Use the same fixed socket path as nvrx-submit — reuse if alive, open if not. **Never close it.**

```bash
SSH_USER="<ssh_user>"
SSH_HOST="<ssh_host>"
CTRL_SOCK="/tmp/nvrx_ssh_${SSH_USER}_${SSH_HOST}.ctrl"
SSH_OPTS="-o ControlMaster=no -o ControlPath=${CTRL_SOCK} -o StrictHostKeyChecking=no"

if ! ssh -O check -o ControlPath=${CTRL_SOCK} ${SSH_USER}@${SSH_HOST} 2>/dev/null; then
    ssh -nNf \
        -o ControlMaster=yes \
        -o ControlPath=${CTRL_SOCK} \
        -o ControlPersist=yes \
        -o ServerAliveInterval=30 \
        -o StrictHostKeyChecking=no \
        ${SSH_USER}@${SSH_HOST}
    for i in $(seq 1 15); do [ -S "${CTRL_SOCK}" ] && break; sleep 1; done
fi
```

## Poll loop

```bash
START_TIME=$(date +%s)
while true; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ ${ELAPSED} -gt <timeout> ]; then
        echo "Timeout after ${ELAPSED}s"
        FINAL_STATE="TIMEOUT"
        break
    fi

    # Get job state
    STATE=$(ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} \
        "squeue -j <job_id> -h -o '%T' 2>/dev/null || echo 'DONE'")

    # Count cycles
    CYCLES=$(ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} \
        "ls <remote_nvrx_dir>/cycle_infos/cycle_info.<job_id>.* 2>/dev/null | wc -l")

    TIMESTAMP=$(date '+%H:%M:%S')
    echo "[${TIMESTAMP}] elapsed=${ELAPSED}s state=${STATE:-DONE} cycles=${CYCLES}"

    if [[ -z "${STATE}" || "${STATE}" == "DONE" ]]; then
        FINAL_STATE="COMPLETED"
        break
    fi

    case "${STATE}" in
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
            FINAL_STATE="${STATE}"
            break
            ;;
    esac

    # Job arrays with hot spares: cold spare tasks stay PENDING (JobArrayTaskLimit) even
    # after the training task completes. If no task is RUNNING but at least one COMPLETED,
    # the training is done — remaining PENDING are cold spares that never ran.
    if ! echo "${STATE}" | grep -qE '(^|/)RUNNING($|/)' && \
         echo "${STATE}" | grep -qE '(^|/)COMPLETED($|/)'; then
        FINAL_STATE="COMPLETED"
        break
    fi

    sleep <poll_interval>
done
```

## Artifact collection

Default `<artifacts_dir>`: `~/.cache/nvrx_artifacts/<cluster>/<job_id>/` (never in any git repo; persists across reboots).

Always collect, regardless of final state. Pull only job-specific files that nvrx-validate needs.

```bash
mkdir -p <artifacts_dir>/nvrx/cycle_infos <artifacts_dir>/logs <artifacts_dir>/checkpoints

# cycle_info files for this job only (named cycle_info.<job_id>.*)
rsync -az \
    --include="cycle_infos/" \
    --include="cycle_infos/cycle_info.<job_id>.*" \
    --exclude="*" \
    -e "ssh ${SSH_OPTS}" \
    ${SSH_USER}@${SSH_HOST}:<remote_nvrx_dir>/ \
    <artifacts_dir>/nvrx/

# Per-cycle training logs for this job (filenames contain job id)
rsync -az \
    --include="*<job_id>*" \
    --exclude="*" \
    -e "ssh ${SSH_OPTS}" \
    ${SSH_USER}@${SSH_HOST}:<remote_logs_dir>/ \
    <artifacts_dir>/logs/

# Only the checkpoint iteration pointer (not full checkpoints)
rsync -az -e "ssh ${SSH_OPTS}" \
    ${SSH_USER}@${SSH_HOST}:<remote_run_dir>/checkpoints/phase1/latest_checkpointed_iteration.txt \
    <artifacts_dir>/checkpoints/ 2>/dev/null || true
```

## Write run_result.json

Write `<artifacts_dir>/run_result.json`:
```json
{
  "job_id": "<job_id>",
  "array_job_id": "<job_id>",
  "cluster": "<cluster>",
  "final_state": "<FINAL_STATE>",
  "remote_run_dir": "<remote_run_dir>",
  "remote_nvrx_dir": "<remote_nvrx_dir>",
  "remote_logs_dir": "<remote_logs_dir>",
  "artifacts_dir": "<artifacts_dir>",
  "completed_at": "<ISO timestamp>"
}
```

## Output

```
Job <job_id> finished: <FINAL_STATE>
Artifacts collected to: <artifacts_dir>/
run_result.json written.

Next: /nvrx-validate <artifacts_dir>
```

If final state is not COMPLETED, warn that validation will run on partial data.
