#!/bin/bash

# Validated only with Megatron-LM as the feedback-loop example workload.
# Direct sbatch usage:
#   sbatch --account=<account> --partition=<partition> scripts/l4_gb200_reduced.sh
# If your cluster has defaults for those, the extra flags are not required.

#SBATCH --time=00:30:00

#SBATCH --job-name=llama4-scout-gb200
#SBATCH --output=/tmp/slurm-%j.launch.out
#SBATCH --error=/tmp/slurm-%j.launch.err

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ENV_FILE="${SCRIPT_DIR}/user.env"
NVRX_SRC_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
NVRX_REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
if [[ -f "${USER_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${USER_ENV_FILE}"
fi

log_msg() {
    local msg="$1"
    UNIX_DATETIME=$(date +%s)
    HUMAN_DATETIME=$(date -d "@$UNIX_DATETIME" '+%Y-%m-%d %H:%M:%S.%3N')
    echo ">>> ${msg} ${UNIX_DATETIME} (${HUMAN_DATETIME})"
}

log_msg "START SBATCH"
echo "Running on nodes: ${SLURM_NODELIST}"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export PYXIS_LOG_LEVEL=debug
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export ONE_LOGGER_JOB_CATEGORY=test
export LOGLEVEL=DEBUG
export TORCHINDUCTOR_WORKER_START=fork
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_RETHROW_CUDA_ERRORS=0
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=20
export TORCH_DIST_INIT_BARRIER=0
export TORCH_INCLUDE_STACK_TRACE=0
export TORCH_INCLUDE_ONLY_ACTIVE=1
export TORCH_NCCL_EXTRA_DUMP_ON_EXEC=1

# Fault injection parameters (overridable via sbatch --export or environment)
# Current Megatron behavior:
# - FAULT_AT_ITER anchors the fault-delay timer after iteration N completes
# - FAULT_DELAY is the delay in seconds from that anchor (or from training start if unset)
export FAULT_AT_ITER="${FAULT_AT_ITER:-5}"
export FAULT_DELAY="${FAULT_DELAY:-15}"
export FAULT_RANK="${FAULT_RANK:-1}"
export FAULT_TYPE="${FAULT_TYPE:-GPU_SLEEP}"
export ENABLE_FAULT_INJECTION="${ENABLE_FAULT_INJECTION:-1}"

# Checkpoint settings (overridable via sbatch --export)
export NVRX_CKPT_USE_CPU_SHM="${NVRX_CKPT_USE_CPU_SHM:-0}"
# Enable GPU-IPC cached-data-structure path without cpu-shm (for comparison baseline)
export NVRX_CKPT_USE_CACHED_STRUCTURE="${NVRX_CKPT_USE_CACHED_STRUCTURE:-0}"
export DIST_TIMEOUT_AFTER_INIT="${DIST_TIMEOUT_AFTER_INIT:-1}"
export ENABLE_NFS_CACHE_STAGING="${ENABLE_NFS_CACHE_STAGING:-0}"
export NFS_TRITON_CACHE="${NFS_TRITON_CACHE:-}"
export NFS_INDUCTOR_CACHE="${NFS_INDUCTOR_CACHE:-}"
# USE_ASYNC_CKPT=1: enable async checkpointing every CKPT_SAVE_INTERVAL iters
export USE_ASYNC_CKPT="${USE_ASYNC_CKPT:-0}"
export CKPT_SAVE_INTERVAL="${CKPT_SAVE_INTERVAL:-100}"
export CONTAINER_CLEANUP_CMD="${CONTAINER_CLEANUP_CMD:-}"

# Node / task geometry (SLURM_NNODES is set by SLURM from --nodes override)
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TOTAL_TASKS=$((SLURM_NNODES * GPUS_PER_NODE))

# Per-experiment output directory (overridable via sbatch --export)
export BASE_EXPERIMENTS_DIR="${BASE_EXPERIMENTS_DIR:-${HOME}/nvrx-attr-experiments}"
FAULT_LABEL="i${FAULT_AT_ITER}"
if [[ -n "${FAULT_DELAY}" ]]; then
    FAULT_LABEL="d${FAULT_DELAY}"
fi
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${BASE_EXPERIMENTS_DIR}/fault_injection/manual/n${SLURM_NNODES}_${FAULT_TYPE}_r${FAULT_RANK}_${FAULT_LABEL}}"
export NVRX_REPO_ROOT="${NVRX_REPO_ROOT:-${NVRX_REPO_ROOT_DEFAULT}}"
export NVRX_SRC_ROOT="${NVRX_SRC_ROOT:-${NVRX_SRC_ROOT_DEFAULT}}"
export NVRX_CONTAINER_REPO_PATH="${NVRX_CONTAINER_REPO_PATH:-${HOME}/nvidia-resiliency-ext}"
export NVRX_CONTAINER_SRC_PATH="${NVRX_CONTAINER_SRC_PATH:-${NVRX_CONTAINER_REPO_PATH}/src}"
export SHARED_TMP_BASE_DIR="${SHARED_TMP_BASE_DIR:-${HOME}/tmp}"
export MEGATRON_REPO_HOST_PATH="${MEGATRON_REPO_HOST_PATH:-${HOME}/megatron-lm}"
export WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-${HOME}/tmp}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.04}"
export CONTAINER_NAME="${CONTAINER_NAME:-}"
export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/}"

mkdir -p ${BASE_EXPERIMENTS_DIR}/datacache
mkdir -p ${EXPERIMENT_DIR}/tensorboard

: "${SLURM_RESTART_COUNT:=0}"

LOG_DIR=${EXPERIMENT_DIR}/logs
mkdir -p ${LOG_DIR}
echo "Writing logs to ${LOG_DIR}"
LOG_FILE_BASE="${LOG_DIR}/slurm/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT}"

# ── Shared-tmp directory (NFS, for cross-srun-step communication) ─────────────
# Mounted to /shared_tmp (NOT /tmp) so the container keeps its native fast /tmp.
SHARED_TMP_HOST=${SHARED_TMP_BASE_DIR}/${SLURM_JOB_ID}
mkdir -p ${SHARED_TMP_HOST}

# ── Pre-populate .myenv with all variables that must reach the container ───────
# Pyxis env forwarding is unreliable for vars set via sbatch --export; writing
# them into .myenv guarantees the inner bash picks them up via `source`.
MYENV_FILE=${SHARED_TMP_HOST}/.myenv_${SLURM_JOB_ID}.sh
cat > ${MYENV_FILE} << MYENVEOF
# Auto-generated by l4_gb200_reduced.sh — do not edit by hand.
export NVRX_CKPT_USE_CPU_SHM=${NVRX_CKPT_USE_CPU_SHM}
export NVRX_CKPT_USE_CACHED_STRUCTURE=${NVRX_CKPT_USE_CACHED_STRUCTURE}
export DIST_TIMEOUT_AFTER_INIT=${DIST_TIMEOUT_AFTER_INIT}
export USE_ASYNC_CKPT=${USE_ASYNC_CKPT}
export CKPT_SAVE_INTERVAL=${CKPT_SAVE_INTERVAL}
export FAULT_AT_ITER=${FAULT_AT_ITER}
export FAULT_DELAY=${FAULT_DELAY}
export FAULT_RANK=${FAULT_RANK}
export FAULT_TYPE=${FAULT_TYPE}
export ENABLE_FAULT_INJECTION=${ENABLE_FAULT_INJECTION}
export ENABLE_NFS_CACHE_STAGING=${ENABLE_NFS_CACHE_STAGING}
export NFS_TRITON_CACHE=${NFS_TRITON_CACHE}
export NFS_INDUCTOR_CACHE=${NFS_INDUCTOR_CACHE}
# Prepend local nvrx checkout so container picks up our changes without a pip install step.
export NVRX_REPO_ROOT=${NVRX_CONTAINER_REPO_PATH}
export NVRX_SRC_ROOT=${NVRX_CONTAINER_SRC_PATH}
export PYTHONPATH=\${NVRX_REPO_ROOT}:\${NVRX_SRC_ROOT}:\${PYTHONPATH}
MYENVEOF

# Mounts
LUSTRE=/home:/home
SHARED_TMP=${SHARED_TMP_HOST}:/shared_tmp
LOGS=${EXPERIMENT_DIR}/logs:/logs
MEGATRON_REPO=${MEGATRON_REPO_HOST_PATH}:/megatron-lm_repo
DATACACHE=${BASE_EXPERIMENTS_DIR}/datacache:/datacache
TENSORBOARD=${EXPERIMENT_DIR}/tensorboard:/tensorboard
WORKSPACE=${WORKSPACE_HOST_PATH}:/workspace
CHECKPOINTS=${EXPERIMENT_DIR}/checkpoints:/checkpoints
mkdir -p ${EXPERIMENT_DIR}/checkpoints
CONTAINER_MOUNTS=$LUSTRE,$SHARED_TMP,$LOGS,$MEGATRON_REPO,$DATACACHE,$TENSORBOARD,$WORKSPACE,$CHECKPOINTS
CONTAINER_ARGS=(
    --container-mounts "${CONTAINER_MOUNTS}"
    --container-image "${CONTAINER_IMAGE}"
    --container-workdir "${CONTAINER_WORKDIR}"
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    CONTAINER_ARGS+=(--container-name "${CONTAINER_NAME}")
fi

# ── Optional site-specific container cleanup hook ──────────────────────────────
if [[ -n "${CONTAINER_CLEANUP_CMD}" ]]; then
    log_msg "START disk_cleanup"
    srun \
        --label \
        --ntasks-per-node=1 \
        --ntasks=${SLURM_NNODES} \
        --kill-on-bad-exit=0 \
        --mpi=none \
        bash -lc "${CONTAINER_CLEANUP_CMD}"
    log_msg "END disk_cleanup"
else
    log_msg "SKIP disk_cleanup"
fi

# all node setup
#--------------------------------
log_msg "START all_node_setup"
srun \
    --label \
    "${CONTAINER_ARGS[@]}" \
    --exclusive \
    --error=${LOG_FILE_BASE}.0.all_node_setup.log \
    --output=${LOG_FILE_BASE}.0.all_node_setup.log \
    --ntasks-per-node=1 \
    --ntasks=${SLURM_NNODES} \
    --kill-on-bad-exit=0 \
    --mpi=none \
    bash -c '
        # Use a per-node NFS path so all ranks on each node find the right clone.
        MEGATRON_PATH=/shared_tmp/megatron_$(hostname)_${SLURM_JOB_ID}
        mkdir -p ${MEGATRON_PATH}
        pushd $MEGATRON_PATH
        CURRENT_BRANCH=$(git -C /megatron-lm_repo branch --show-current)
        echo "Cloning Megatron branch $CURRENT_BRANCH to ${MEGATRON_PATH}"
        git clone --single-branch --branch $CURRENT_BRANCH /megatron-lm_repo .
        rm -rf ${MEGATRON_PATH}/nvidia_resiliency_ext
        rsync -a ${NVRX_CONTAINER_SRC_PATH}/nvidia_resiliency_ext/ ${MEGATRON_PATH}/nvidia_resiliency_ext/
        popd
    '
log_msg "END all_node_setup"

# main workload
#--------------------------------
log_msg "START main_workload"
srun \
    --label \
    "${CONTAINER_ARGS[@]}" \
    --error=${LOG_FILE_BASE}.1.main_workload.log \
    --output=${LOG_FILE_BASE}.1.main_workload.log \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --ntasks=${TOTAL_TASKS} \
    --kill-on-bad-exit=0 \
    --mpi=none \
    bash -c '
        source /shared_tmp/.myenv_${SLURM_JOB_ID}.sh
        MEGATRON_PATH=/shared_tmp/megatron_$(hostname)_${SLURM_JOB_ID}
        export PYTHONPATH=${MEGATRON_PATH}:${NVRX_REPO_ROOT}:${NVRX_SRC_ROOT}:${PYTHONPATH}
        echo "NVRX_REPO_ROOT=${NVRX_REPO_ROOT}"
        echo "NVRX_SRC_ROOT=${NVRX_SRC_ROOT}"
        echo "PYTHONPATH=${PYTHONPATH}"
        python3 - <<'"'"'PY'"'"'
import sys
print(f"sys.path[:8]={sys.path[:8]}")
import nvidia_resiliency_ext
from nvidia_resiliency_ext.shared_utils.inject_fault import Fault
print(f"nvidia_resiliency_ext={nvidia_resiliency_ext.__file__}")
print(f"fault_enum={Fault}")
PY

        # Per-rank Triton/inductor cache on the container native /tmp (local fast storage).
        export TRITON_CACHE_DIR=/tmp/triton_${SLURM_LOCALID}
        export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_LOCALID}
        mkdir -p ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}

        # Optional pre/post-stage between a shared cache and the node-local /tmp cache.
        if [[ "${ENABLE_NFS_CACHE_STAGING}" == "1" && "${SLURM_LOCALID}" == "0" ]]; then
            if [[ -n "${NFS_TRITON_CACHE}" && -d "${NFS_TRITON_CACHE}" ]]; then
                echo "Pre-staging triton cache from ${NFS_TRITON_CACHE}..."
                rsync -a --ignore-existing "${NFS_TRITON_CACHE}/" "${TRITON_CACHE_DIR}/" 2>/dev/null || true
            fi
            if [[ -n "${NFS_INDUCTOR_CACHE}" && -d "${NFS_INDUCTOR_CACHE}" ]]; then
                echo "Pre-staging inductor cache from ${NFS_INDUCTOR_CACHE}..."
                rsync -a --ignore-existing "${NFS_INDUCTOR_CACHE}/" "${TORCHINDUCTOR_CACHE_DIR}/" 2>/dev/null || true
            fi
        fi

        # Post-stage: write back to NFS on exit (one rank per node)
        _stage_back() {
            if [[ "${ENABLE_NFS_CACHE_STAGING}" == "1" && "${SLURM_LOCALID}" == "0" ]]; then
                if [[ -n "${NFS_TRITON_CACHE}" ]]; then
                    mkdir -p "${NFS_TRITON_CACHE}"
                    rsync -a --ignore-existing "${TRITON_CACHE_DIR}/" "${NFS_TRITON_CACHE}/" 2>/dev/null || true
                fi
                if [[ -n "${NFS_INDUCTOR_CACHE}" ]]; then
                    mkdir -p "${NFS_INDUCTOR_CACHE}"
                    rsync -a --ignore-existing "${TORCHINDUCTOR_CACHE_DIR}/" "${NFS_INDUCTOR_CACHE}/" 2>/dev/null || true
                fi
            fi
        }
        trap _stage_back EXIT

        # Checkpoint directory — NFS path mounted to /checkpoints inside the container.
        # /dev/shm is reserved for IPC shm tensors and the DataLoader.
        # Note: --log-progress is NOT set. Megatron will not write/read progress.txt
        # (which would be per-node and invisible across nodes).
        CKPT_DIR=/checkpoints
        mkdir -p ${CKPT_DIR}

        # Build checkpoint args (controlled by USE_ASYNC_CKPT from .myenv).
        # No --load: we only want to test save here.
        CKPT_SAVE_ARGS=""
        if [[ "${USE_ASYNC_CKPT}" == "1" ]]; then
            CKPT_SAVE_ARGS="--save ${CKPT_DIR} --save-interval ${CKPT_SAVE_INTERVAL} --async-save --use-persistent-ckpt-worker --use-dist-ckpt --ckpt-fully-parallel-save --ckpt-assume-constant-structure"
        fi

        pushd $MEGATRON_PATH
        LAUNCHER_CMD="python3"
        LAUNCHER_ARGS=" \
        "
        WORKLOAD_CMD=${MEGATRON_PATH}/pretrain_gpt.py
        FAULT_INJECTOR_ARGS=""
        if [[ "${ENABLE_FAULT_INJECTION}" == "1" ]]; then
            FAULT_INJECTOR_ARGS=" \
                --fault-injector-ranks ${FAULT_RANK} \
                --fault-injector-fault-types ${FAULT_TYPE} \
            "
            if [[ -n "${FAULT_DELAY}" ]]; then
                FAULT_INJECTOR_ARGS="${FAULT_INJECTOR_ARGS} --fault-injector-fault-delay ${FAULT_DELAY}"
                if [[ -n "${FAULT_AT_ITER}" ]]; then
                    FAULT_INJECTOR_ARGS="${FAULT_INJECTOR_ARGS} --fault-injector-delay-start-iteration ${FAULT_AT_ITER}"
                fi
            elif [[ -n "${FAULT_AT_ITER}" ]]; then
                FAULT_INJECTOR_ARGS="${FAULT_INJECTOR_ARGS} --fault-injector-fault-delay 0 --fault-injector-delay-start-iteration ${FAULT_AT_ITER}"
            fi
        fi
        WORKLOAD_ARGS=" \
            --exit-duration-in-mins 5750 \
            --distributed-timeout-minutes 10 \
            --disable-gloo-process-groups \
            --mock-data \
            --data-cache-path /datacache \
            --no-create-attention-mask-in-dataloader \
            --no-mmap-bin-files \
            --tokenizer-type NullTokenizer \
            --tiktoken-pattern v2 \
            --micro-batch-size 1 \
            --global-batch-size 64 \
            --train-samples 10240000 \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --adam-eps 1e-05 \
            --lr-decay-style cosine \
            --lr-warmup-samples 1024000 \
            --lr-decay-samples 20480000 \
            --lr 0.0003 \
            --min-lr 2.9999999999999997e-05 \
            --weight-decay 0.1 \
            --clip-grad 1.0 \
            --loss-scale 1.0 \
            --use-mcore-models \
            --untie-embeddings-and-output-weights \
            --disable-bias-linear \
            --attention-backend flash \
            --transformer-impl transformer_engine \
            --position-embedding-type rope \
            --rotary-base 500000 \
            --rotary-interleaved \
            --use-rope-scaling \
            --rope-scaling-factor 8.0 \
            --no-rope-fusion \
            --no-rope-freq 4 \
            --use-flash-attn \
            --cross-entropy-fusion-impl te \
            --cross-entropy-loss-fusion \
            --seq-length 8192 \
            --max-position-embeddings 8192 \
            --num-layers 12 \
            --swiglu \
            --hidden-size 5120 \
            --num-attention-heads 40 \
            --group-query-attention \
            --num-query-groups 8 \
            --ffn-hidden-size 16384 \
            --kv-channels 128 \
            --normalization RMSNorm \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --grad-reduce-in-bf16 \
            --qk-l2-norm \
            --num-experts 16 \
            --moe-layer-freq 1 \
            --moe-ffn-hidden-size 8192 \
            --moe-shared-expert-intermediate-size 8192 \
            --moe-router-topk 1 \
            --moe-router-score-function sigmoid \
            --moe-token-dispatcher-type alltoall \
            --moe-grouped-gemm \
            --moe-shared-expert-overlap \
            --moe-router-bias-update-rate 0.001 \
            --moe-router-load-balancing-type aux_loss \
            --moe-aux-loss-coeff 0.01 \
            --moe-router-enable-expert-bias \
            --moe-apply-probs-on-input \
            --moe-router-force-load-balancing \
            --bf16 \
            --fp8-format hybrid \
            --fp8-recipe delayed \
            --fp8-param-gather \
            --fp8-amax-history-len 1024 \
            --fp8-amax-compute-algo max \
            --fp8-margin 0 \
            --te-rng-tracker \
            --sequence-parallel \
            --use-distributed-optimizer \
            --overlap-grad-reduce \
            --overlap-param-gather \
            --ddp-num-buckets 5 \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 1 \
            --expert-model-parallel-size 8 \
            --expert-tensor-parallel-size 1 \
            --ddp-average-in-collective \
            --log-interval 1 \
            --timing-log-option minmax \
            --log-params-norm \
            --log-num-zeros-in-grad \
            --log-throughput \
            --check-weight-hash-across-dp-replicas-interval 20000 \
            --tensorboard-dir /tensorboard \
            --logging-level 10 \
            --eval-iters 14 \
            --eval-interval 2000 \
            --manual-gc \
            --manual-gc-interval 100 \
            --num-workers 1 \
            --rerun-mode validate_results \
            --log-straggler \
            --disable-straggler-on-startup \
            --straggler-minmax-count 16 \
            --local-rank ${SLURM_LOCALID} \
            --context-parallel-size 1 \
            --vocab-size 238600 \
            ${FAULT_INJECTOR_ARGS} \
            --distributed-timeout-seconds-after-init ${DIST_TIMEOUT_AFTER_INIT} \
            --flight-recorder-dump-path ${CKPT_DIR} \
        "
        PYTHONPATH=${MEGATRON_PATH}:${NVRX_REPO_ROOT}:${NVRX_SRC_ROOT}:${PYTHONPATH} \
            $LAUNCHER_CMD $LAUNCHER_ARGS $WORKLOAD_CMD $WORKLOAD_ARGS $CKPT_SAVE_ARGS
    '
log_msg "END main_workload"

log_msg "END SBATCH"

set +x
