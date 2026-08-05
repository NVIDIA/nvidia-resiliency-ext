#!/bin/bash
# Enqueue a singleton chain of NVRx job-array generations.
#
# Computes --array/--nodes/--job-name from the same env vars nvrx_singleton_array.sbatch
# reads, so the array throttle and --nnodes max cannot drift apart, then submits
# NVRX_CHAIN_DEPTH jobs. --dependency=singleton (set in the sbatch) runs them one at a
# time; each is a generation that resumes from the last checkpoint of the previous one.
#
#   ./submit_chain.sh                              # defaults, 4 generations
#   NVRX_MODEL_PROFILE=8b NVRX_TRAIN_TASKS=2 ./submit_chain.sh
#   NVRX_DRY_RUN=1 ./submit_chain.sh               # print the sbatch commands only
#
# Any SBATCH_* env var slurm understands (SBATCH_ACCOUNT, SBATCH_PARTITION, ...) is
# honoured, as are extra flags passed straight through: ./submit_chain.sh -q myqos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/nvrx_singleton_array.sbatch"

# Shape. Defaults form the smallest run that shows a hand-off: 1 training node + 1 hot
# spare + 1 cold spare. Scale up for real training (see the README).
export NVRX_TRAIN_TASKS="${NVRX_TRAIN_TASKS:-1}"     # tasks the model is sized for
export NVRX_HOT_SPARES="${NVRX_HOT_SPARES:-1}"       # running, standby, absorb a failure
NVRX_COLD_SPARES="${NVRX_COLD_SPARES:-3}"            # queued; several so the pending-task
                                                    # cancellation (cancel_chain /
                                                    # cancel_pending_spares) is visible
NODES_PER_TASK="${NODES_PER_TASK:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"   # GB200 tray default; 8 for HGX/DGX

# Chain. Default is deep enough that the no-restart demo has several queued generations
# to cancel when it exits 93 (1 runs, the rest are cancelled by cancel_chain).
NVRX_CHAIN_DEPTH="${NVRX_CHAIN_DEPTH:-4}"            # generations queued up front
NVRX_JOB_NAME="${NVRX_JOB_NAME:-nvrx_singleton}"
NVRX_TIME_LIMIT="${NVRX_TIME_LIMIT:-00:25:00}"

# Workload -- passed through to the sbatch
export NVRX_WORK_DIR="${NVRX_WORK_DIR:-${PWD}/nvrx-run}"
export MEGATRON_PATH="${MEGATRON_PATH:-/workspace/megatron-lm}"
export NVRX_MODEL_PROFILE="${NVRX_MODEL_PROFILE:-small}"
export NVRX_CONTAINER_IMAGE="${NVRX_CONTAINER_IMAGE:-}"
# ${VAR-default}, not ${VAR:-default}: an explicit NVRX_CONTAINER_MOUNTS= means "mount
# nothing". This export is what the sbatch actually sees, so the colon here would defeat
# the matching fix in nvrx_singleton_array.sbatch no matter what that line says.
export NVRX_CONTAINER_MOUNTS="${NVRX_CONTAINER_MOUNTS-/lustre:/lustre}"

ACTIVE_TASKS=$(( NVRX_TRAIN_TASKS + NVRX_HOT_SPARES ))
TOTAL_TASKS=$(( ACTIVE_TASKS + NVRX_COLD_SPARES ))
export NVRX_ACTIVE_TASKS="${ACTIVE_TASKS}"

if (( NVRX_TRAIN_TASKS < 1 )); then
    echo "NVRX_TRAIN_TASKS must be >= 1" >&2; exit 1
fi

# A cold spare only helps if SLURM will admit it; past MaxArraySize the array is
# rejected outright, which is a confusing way to learn the pool was too deep.
MAX_ARRAY_SIZE=$(scontrol show config 2>/dev/null \
                 | awk -F'= *' '/^MaxArraySize/ {print $2}' | tr -d ' ' || true)
if [[ -n "${MAX_ARRAY_SIZE:-}" && "$MAX_ARRAY_SIZE" =~ ^[0-9]+$ ]] && (( TOTAL_TASKS > MAX_ARRAY_SIZE )); then
    echo "Pool of ${TOTAL_TASKS} tasks exceeds MaxArraySize=${MAX_ARRAY_SIZE}." >&2
    echo "Lower NVRX_COLD_SPARES, or submit a deeper chain instead of a deeper pool." >&2
    exit 1
fi

mkdir -p "${NVRX_WORK_DIR}"

# The two settings with no safe default: everything else here is a tuning knob, but a
# work dir the nodes cannot all see, or a Megatron that is not where we think it is,
# fails minutes into the run instead of now.
case "$(readlink -f "${NVRX_WORK_DIR}")" in
    /tmp/*|/var/tmp/*|/dev/shm/*)
        echo "NVRX_WORK_DIR=${NVRX_WORK_DIR} looks node-local. The rendezvous host is" >&2
        echo "published through this directory, so every node must see the same one." >&2
        exit 1 ;;
esac
if [[ -z "$NVRX_CONTAINER_IMAGE" && ! -f "${MEGATRON_PATH}/pretrain_gpt.py" ]]; then
    # Only checkable without a container: with one, MEGATRON_PATH is a path inside it.
    echo "MEGATRON_PATH=${MEGATRON_PATH} has no pretrain_gpt.py." >&2
    echo "Clone https://github.com/NVIDIA/Megatron-LM and point MEGATRON_PATH at it." >&2
    exit 1
fi

echo "Chain:        ${NVRX_CHAIN_DEPTH} generations of '${NVRX_JOB_NAME}' (singleton)"
echo "Per generation: ${NVRX_TRAIN_TASKS} training + ${NVRX_HOT_SPARES} hot spare(s)"
echo "                + ${NVRX_COLD_SPARES} cold spare(s) = array 0-$(( TOTAL_TASKS - 1 ))%${ACTIVE_TASKS}"
echo "World size:   $(( NVRX_TRAIN_TASKS * NODES_PER_TASK * GPUS_PER_NODE )) GPUs"
echo "              (${NODES_PER_TASK} node(s)/task x ${GPUS_PER_NODE} GPU(s)/node)"
echo "Profile:      ${NVRX_MODEL_PROFILE}    Work dir: ${NVRX_WORK_DIR}"

# Materialise a self-contained sbatch in the work dir with the concrete NVRX_WORK_DIR
# baked in (the template otherwise reads it from the exported env). This makes the run's
# --ft-cycle-info-dir / --ft-checkpoint-iteration-file recoverable by anyone who can read
# the file -- nvrx-watch resolves them straight from this script, the same way it reads a
# production InJob sbatch. Submitting this copy also means the job's Command= points at it.
RUN_SBATCH="${NVRX_WORK_DIR}/nvrx_singleton_array.sbatch"
if [[ "${NVRX_DRY_RUN:-0}" != "1" ]]; then
    sed "s#^NVRX_WORK_DIR=.*#NVRX_WORK_DIR=\"${NVRX_WORK_DIR}\"#" "${SBATCH_SCRIPT}" > "${RUN_SBATCH}"
    chmod +x "${RUN_SBATCH}"
fi

SBATCH_ARGS=(
    --job-name="${NVRX_JOB_NAME}"
    --nodes="${NODES_PER_TASK}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --array="0-$(( TOTAL_TASKS - 1 ))%${ACTIVE_TASKS}"
    --time="${NVRX_TIME_LIMIT}"
    --export=ALL
    "$@"
)

for (( i = 1; i <= NVRX_CHAIN_DEPTH; i++ )); do
    if [[ "${NVRX_DRY_RUN:-0}" == "1" ]]; then
        echo "[dry-run] sbatch ${SBATCH_ARGS[*]} ${RUN_SBATCH}"
    else
        # Not --parsable in a loop by choice: the JobId line per generation is what you
        # read back when reconciling the chain against squeue.
        sbatch "${SBATCH_ARGS[@]}" "${RUN_SBATCH}"
    fi
done

# nvrx-watch only reports an exhausted chain when this marker exists, so an account
# with no chain running stays quiet.
if [[ "${NVRX_DRY_RUN:-0}" != "1" ]]; then
    printf '%s\n' "${NVRX_JOB_NAME}" > "${HOME}/.nvrx_watch_expect_chain"
    echo "Marked ${HOME}/.nvrx_watch_expect_chain; nvrx-watch will now page on chain exhaustion."
fi
