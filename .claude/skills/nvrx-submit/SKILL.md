---
name: nvrx-submit
description: Upload an InJob-enabled sbatch script to a remote cluster and submit it via SSH. Manages SSH ControlMaster and writes a state file for nvrx-watch. Use after nvrx-create, or when the user wants to deploy a script to a cluster.
argument-hint: <nvrx_sbatch> [--ssh-config <yaml>]
allowed-tools: [Read, Write, Bash]
---

Upload an InJob-enabled sbatch script to a cluster and submit it via SSH.

## Usage
```
/nvrx-submit <sbatch_script> [--ssh-config <ssh_config.yaml>]
```

- `<sbatch_script>`: local path to the InJob sbatch (output of `/nvrx-create`)
- `--ssh-config <yaml>`: path to a cluster SSH config YAML. If omitted or fields missing, you will be prompted interactively.

## SSH config YAML format

```yaml
# e.g. ~/workspace/nvidia-experiment/slurm/hsg/ssh_config.yaml
ssh_host: "oci-hsg-cs-001-login-01.nvidia.com"
ssh_user: "hexinw"
ssh_identity_file: "~/.ssh/id_rsa"   # optional
```

## Resolving SSH config

Use the first source that provides all required fields (`ssh_host`, `ssh_user`):

1. **`--ssh-config` YAML** — if provided and file exists
2. **Interactive prompt** — ask the user for any missing fields:
   ```
   SSH host (e.g. hsg-login.nvidia.com): _
   SSH user: _
   ```
3. **Local mode** — if the user responds "local" or "skip", run `sbatch` directly on the local machine (useful when already on the cluster login node).

## Your task

1. Resolve SSH config from `--ssh-config` or interactive prompt.
2. Open a persistent SSH ControlMaster connection.
3. Create necessary remote directories.
4. Upload the sbatch script via rsync.
5. Submit via `sbatch` and capture the job ID.
6. Write a state file for use by `/nvrx-watch`.
7. Print the job ID and next step hint.

## SSH ControlMaster setup

Use a fixed socket path per host so it persists across nvrx-submit / nvrx-watch / nvrx-validate calls within the same session. Check if already alive before opening a new one. **Never close it** — it lives until the Claude session exits or SSH times out.

```bash
SSH_USER="<ssh_user>"
SSH_HOST="<ssh_host>"
CTRL_SOCK="/tmp/nvrx_ssh_${SSH_USER}_${SSH_HOST}.ctrl"
SSH_OPTS="-o ControlMaster=no -o ControlPath=${CTRL_SOCK} -o StrictHostKeyChecking=no"

# Reuse if alive, otherwise open new master
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

All subsequent commands use `ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} "<cmd>"`.

## Steps

### 1. Determine remote paths

- `REMOTE_SCRIPT="/tmp/ft_$(date +%s)_$(basename <sbatch_script>)"` — temp path on login node; SLURM reads and stores the script at submission time so this file does not need to persist after `sbatch` returns.
- Parse `LOGS_DIR` and `NVRX_DIR` from the sbatch script variables.

### 2. Create remote directories

```bash
ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} "mkdir -p ${REMOTE_LOGS_DIR} ${REMOTE_NVRX_DIR}"
```

### 3. Upload script

```bash
rsync -az -e "ssh ${SSH_OPTS}" <sbatch_script> ${SSH_USER}@${SSH_HOST}:${REMOTE_SCRIPT}
```

### 4. Submit

```bash
SBATCH_OUT=$(ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST} "sbatch ${REMOTE_SCRIPT}")
echo "${SBATCH_OUT}"
JOB_ID=$(echo "${SBATCH_OUT}" | grep -oP '(?<=Submitted batch job )\d+')
```

### 5. Write state file

Write `<sbatch_script>.state.json` locally:
```json
{
  "job_id": "<JOB_ID>",
  "ssh_host": "<ssh_host>",
  "ssh_user": "<ssh_user>",
  "ctrl_sock": "<CTRL_SOCK>",
  "remote_nvrx_dir": "<remote_nvrx_dir>",
  "remote_logs_dir": "<remote_logs_dir>",
  "submitted_at": "<ISO timestamp>"
}
```

## Output

```
Submitted: job_id=<JOB_ID> host=<ssh_host>
State file: <sbatch_script>.state.json

Next: /nvrx-watch <sbatch_script>.state.json
```
