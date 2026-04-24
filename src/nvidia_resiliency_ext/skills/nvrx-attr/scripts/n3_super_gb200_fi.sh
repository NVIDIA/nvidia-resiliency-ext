#!/bin/bash
# n3_super_gb200_fi.sh — fault-injection job script for the n3_super_gb200 workload.
# Production model args are kept aligned with the previously working nemotron config.
# Only path/container plumbing is adapted for the nvrx-attr feedback-loop workflow.

#SBATCH --time=00:30:00

#SBATCH --job-name=n3-super-gb200-fi
#SBATCH --output=/tmp/slurm-%j.launch.out
#SBATCH --error=/tmp/slurm-%j.launch.err

#SBATCH --nodes=8
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

# ── Platform / NCCL ───────────────────────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export PYXIS_LOG_LEVEL=debug
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN

# ── PyTorch / TE / inductor (from n3_super_gb200.sh ENV_VARS) ─────────────────
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork
export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_MNNVL=1

# ── DeepEP (hybridep MoE routing) — set USE_DEEPEP=0 to use alltoall instead ──
export USE_DEEPEP="${USE_DEEPEP:-1}"
if [[ "${USE_DEEPEP}" == "1" ]]; then
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32
fi

# ── Logging / debugging ───────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export ONE_LOGGER_JOB_CATEGORY=test
export LOGLEVEL=DEBUG
export TORCH_CPP_LOG_LEVEL=WARNING
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_RETHROW_CUDA_ERRORS=0
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30
export TORCH_DIST_INIT_BARRIER=0
export TORCH_INCLUDE_STACK_TRACE=0
export TORCH_INCLUDE_ONLY_ACTIVE=1
export TORCH_NCCL_EXTRA_DUMP_ON_EXEC=1

# ── Fault injection parameters (overridable via sbatch --export) ──────────────
export FAULT_AT_ITER="${FAULT_AT_ITER:-5}"
export FAULT_DELAY="${FAULT_DELAY:-15}"
export FAULT_RANK="${FAULT_RANK:-1}"
export FAULT_TYPE="${FAULT_TYPE:-GPU_SLEEP}"
export ENABLE_FAULT_INJECTION="${ENABLE_FAULT_INJECTION:-1}"

# ── CUDA graph (set ENABLE_CUDA_GRAPH=0 to disable) ───────────────────────────
export ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-1}"

# ── Node / task geometry (SLURM_NNODES is set by SLURM from --nodes override) ─
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TOTAL_TASKS=$((SLURM_NNODES * GPUS_PER_NODE))

# ── Per-experiment output directory (overridable via sbatch --export) ─────────
export BASE_EXPERIMENTS_DIR="${BASE_EXPERIMENTS_DIR:-${HOME}/nvrx-attr-experiments}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${BASE_EXPERIMENTS_DIR}/fault_injection/manual/n${SLURM_NNODES}_${FAULT_TYPE}_r${FAULT_RANK}_i${FAULT_AT_ITER}}"
export NVRX_REPO_ROOT="${NVRX_REPO_ROOT:-${NVRX_REPO_ROOT_DEFAULT}}"
export NVRX_SRC_ROOT="${NVRX_SRC_ROOT:-${NVRX_SRC_ROOT_DEFAULT}}"
export NVRX_CONTAINER_REPO_PATH="${NVRX_CONTAINER_REPO_PATH:-${HOME}/nvidia-resiliency-ext}"
export NVRX_CONTAINER_SRC_PATH="${NVRX_CONTAINER_SRC_PATH:-${NVRX_CONTAINER_REPO_PATH}/src}"
export SHARED_TMP_BASE_DIR="${SHARED_TMP_BASE_DIR:-${HOME}/tmp}"
export MEGATRON_REPO_HOST_PATH="${MEGATRON_REPO_HOST_PATH:-${HOME}/megatron-lm-main}"
export WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-${HOME}/tmp}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.04}"
export CONTAINER_NAME="${CONTAINER_NAME:-}"
export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/}"
export CONTAINER_CLEANUP_CMD="${CONTAINER_CLEANUP_CMD:-}"
export ENABLE_NFS_CACHE_STAGING="${ENABLE_NFS_CACHE_STAGING:-0}"
export NFS_TRITON_CACHE="${NFS_TRITON_CACHE:-}"
export NFS_INDUCTOR_CACHE="${NFS_INDUCTOR_CACHE:-}"

mkdir -p ${BASE_EXPERIMENTS_DIR}/datacache
mkdir -p ${EXPERIMENT_DIR}/checkpoints
mkdir -p ${EXPERIMENT_DIR}/tensorboard

: "${SLURM_RESTART_COUNT:=0}"

LOG_DIR=${EXPERIMENT_DIR}/logs
mkdir -p ${LOG_DIR}
echo "Writing logs to ${LOG_DIR}"
LOG_FILE_BASE="${LOG_DIR}/slurm/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT}"

# ── Container mounts ──────────────────────────────────────────────────────────
LUSTRE=/home:/home
SHARED_TMP_HOST=${SHARED_TMP_BASE_DIR}/${SLURM_JOB_ID}
mkdir -p ${SHARED_TMP_HOST}
SHARED_TMP=${SHARED_TMP_HOST}:/shared_tmp
LOGS=${EXPERIMENT_DIR}/logs:/logs
MEGATRON_REPO=${MEGATRON_REPO_HOST_PATH}:/megatron-lm_repo
DATACACHE=${BASE_EXPERIMENTS_DIR}/datacache:/datacache
CHECKPOINT_LOAD=${EXPERIMENT_DIR}/checkpoints:/checkpoint-load
CHECKPOINT_SAVE=${EXPERIMENT_DIR}/checkpoints:/checkpoint-save
TENSORBOARD=${EXPERIMENT_DIR}/tensorboard:/tensorboard
WORKSPACE=${WORKSPACE_HOST_PATH}:/workspace
CONTAINER_MOUNTS=$LUSTRE,$SHARED_TMP,$LOGS,$MEGATRON_REPO,$DATACACHE,$CHECKPOINT_LOAD,$CHECKPOINT_SAVE,$TENSORBOARD,$WORKSPACE
CONTAINER_ARGS=(
    --container-mounts "${CONTAINER_MOUNTS}"
    --container-image "${CONTAINER_IMAGE}"
    --container-workdir "${CONTAINER_WORKDIR}"
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    CONTAINER_ARGS+=(--container-name "${CONTAINER_NAME}")
fi

MYENV_FILE=${SHARED_TMP_HOST}/.myenv_${SLURM_JOB_ID}.sh
cat > ${MYENV_FILE} << MYENVEOF
export FAULT_AT_ITER=${FAULT_AT_ITER}
export FAULT_DELAY=${FAULT_DELAY}
export FAULT_RANK=${FAULT_RANK}
export FAULT_TYPE=${FAULT_TYPE}
export ENABLE_FAULT_INJECTION=${ENABLE_FAULT_INJECTION}
export ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH}
export USE_DEEPEP=${USE_DEEPEP}
export ENABLE_NFS_CACHE_STAGING=${ENABLE_NFS_CACHE_STAGING}
export NFS_TRITON_CACHE=${NFS_TRITON_CACHE}
export NFS_INDUCTOR_CACHE=${NFS_INDUCTOR_CACHE}
export NVRX_REPO_ROOT=${NVRX_CONTAINER_REPO_PATH}
export NVRX_SRC_ROOT=${NVRX_CONTAINER_SRC_PATH}
export PYTHONPATH=\${NVRX_REPO_ROOT}:\${NVRX_SRC_ROOT}:\${PYTHONPATH}
MYENVEOF

# ── Optional site-specific cleanup hook ───────────────────────────────────────
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

# ── All-node setup: clone Megatron into a per-job tmpdir ─────────────────────
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
        MEGATRON_PATH=/shared_tmp/megatron_${SLURM_NODEID}
        rm -rf "${MEGATRON_PATH}"
        mkdir -p "${MEGATRON_PATH}"
        pushd $MEGATRON_PATH
        CURRENT_BRANCH=$(git -C /megatron-lm_repo branch --show-current)
        echo "Cloning Megatron branch $CURRENT_BRANCH into $MEGATRON_PATH"
        git clone --single-branch --branch $CURRENT_BRANCH /megatron-lm_repo .
        rm -rf "${MEGATRON_PATH}/nvidia_resiliency_ext"
        if command -v rsync >/dev/null 2>&1; then
            rsync -a "${NVRX_CONTAINER_SRC_PATH}/nvidia_resiliency_ext/" "${MEGATRON_PATH}/nvidia_resiliency_ext/"
        else
            cp -a "${NVRX_CONTAINER_SRC_PATH}/nvidia_resiliency_ext" "${MEGATRON_PATH}/"
        fi
        popd
    '
log_msg "END all_node_setup"

# ── Main workload ─────────────────────────────────────────────────────────────
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
        MEGATRON_PATH=/shared_tmp/megatron_${SLURM_NODEID}
        export PYTHONPATH=${MEGATRON_PATH}:${NVRX_REPO_ROOT}:${NVRX_SRC_ROOT}:${PYTHONPATH}

        # Triton/inductor cache strategy:
        #   - /tmp inside the container is the node-local in-memory tmpfs (not NFS-backed)
        #   - Optional pre-stage from a persistent cache to each local rank /tmp dir
        #   - Barrier via marker file in /tmp ensures other ranks wait before Python starts
        #   - On exit: global rank 0 stages back to NFS only on cold start
        TRITON_READY=/tmp/.triton_ready_${SLURM_JOB_ID}

        export TRITON_CACHE_DIR=/tmp/triton_${SLURM_LOCALID}
        export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_LOCALID}

        if [[ "${ENABLE_NFS_CACHE_STAGING}" == "1" && "${SLURM_LOCALID}" == "0" ]]; then
            if [[ -d "${NFS_TRITON_CACHE}" ]] && [[ -n "$(ls -A ${NFS_TRITON_CACHE} 2>/dev/null)" ]]; then
                TRITON_CACHE_WAS_WARM=1
            else
                TRITON_CACHE_WAS_WARM=0
            fi
            for r in 0 1 2 3; do
                mkdir -p /tmp/triton_${r} /tmp/inductor_${r}
                [[ -d "${NFS_TRITON_CACHE}" ]] && rsync -a --ignore-existing "${NFS_TRITON_CACHE}/" "/tmp/triton_${r}/" 2>/dev/null || true
                [[ -d "${NFS_INDUCTOR_CACHE}" ]] && rsync -a --ignore-existing "${NFS_INDUCTOR_CACHE}/" "/tmp/inductor_${r}/" 2>/dev/null || true
            done
            touch "${TRITON_READY}"
            echo "Pre-staged triton/inductor cache for all local ranks (was_warm=${TRITON_CACHE_WAS_WARM})."
        elif [[ "${SLURM_LOCALID}" != "0" ]]; then
            until [[ -f "${TRITON_READY}" ]]; do sleep 1; done
        fi

        mkdir -p ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}

        _stage_back() {
            if [[ "${ENABLE_NFS_CACHE_STAGING}" == "1" && "${SLURM_LOCALID}" == "0" && "${SLURM_NODEID}" == "0" && "${TRITON_CACHE_WAS_WARM:-0}" == "0" ]]; then
                echo "Staging triton cache back to NFS (cold start)..."
                mkdir -p "${NFS_TRITON_CACHE}" "${NFS_INDUCTOR_CACHE}"
                rsync -a --ignore-existing "${TRITON_CACHE_DIR}/" "${NFS_TRITON_CACHE}/" 2>/dev/null || true
                rsync -a --ignore-existing "${TORCHINDUCTOR_CACHE_DIR}/" "${NFS_INDUCTOR_CACHE}/" 2>/dev/null || true
                echo "Cache staged back."
            fi
        }
        trap _stage_back EXIT

        if [[ "${ENABLE_CUDA_GRAPH}" == "1" ]]; then
            CUDA_GRAPH_ARGS="--enable-cuda-graph --cuda-graph-scope mamba attn"
        else
            CUDA_GRAPH_ARGS=""
        fi

        if [[ "${USE_DEEPEP:-1}" == "1" ]]; then
            MOE_DISPATCHER_ARGS="--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --moe-hybridep-num-sms 32"
        else
            MOE_DISPATCHER_ARGS="--moe-token-dispatcher-type alltoall"
        fi

        pushd $MEGATRON_PATH
        LAUNCHER_CMD="python3"
        LAUNCHER_ARGS=" \
        "
        WORKLOAD_CMD=${MEGATRON_PATH}/pretrain_mamba.py
        FAULT_INJECTOR_ARGS=""
        if [[ "${ENABLE_FAULT_INJECTION}" == "1" ]]; then
            FAULT_INJECTOR_ARGS=" \
                --fault-injector-ranks ${FAULT_RANK} \
                --fault-injector-fault-types ${FAULT_TYPE} \
            "
            if [[ -n "${FAULT_DELAY:-}" ]]; then
                FAULT_INJECTOR_ARGS="${FAULT_INJECTOR_ARGS} --fault-injector-fault-delay ${FAULT_DELAY}"
                if [[ -n "${FAULT_AT_ITER:-}" ]]; then
                    FAULT_INJECTOR_ARGS="${FAULT_INJECTOR_ARGS} --fault-injector-delay-start-iteration ${FAULT_AT_ITER}"
                fi
            elif [[ -n "${FAULT_AT_ITER:-}" ]]; then
                FAULT_INJECTOR_ARGS="${FAULT_INJECTOR_ARGS} --fault-injector-fault-delay 0 --fault-injector-delay-start-iteration ${FAULT_AT_ITER}"
            fi
        fi
        WORKLOAD_ARGS=" \
            --exit-duration-in-mins 5750 \
            --exit-interval 100 \
            --distributed-timeout-minutes 10 \
            --disable-gloo-process-groups \
            --mock-data \
            --data-cache-path /datacache \
            --no-create-attention-mask-in-dataloader \
            --no-mmap-bin-files \
            --tokenizer-type NullTokenizer \
            --tiktoken-pattern v2 \
            --vocab-size 128000 \
            --micro-batch-size 1 \
            --global-batch-size 32 \
            --train-samples 12207031 \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --lr 4.5e-4 \
            --min-lr 4.5e-6 \
            --lr-decay-style WSD \
            --lr-warmup-samples 24414063 \
            --lr-decay-samples 3048706055 \
            --lr-wsd-decay-style minus_sqrt \
            --lr-wsd-decay-samples 610351563 \
            --weight-decay 0.1 \
            --clip-grad 1.0 \
            --override-opt_param-scheduler \
            --use-mcore-models \
            --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
            --is-hybrid-model \
            --mamba-num-heads 128 \
            --num-layers 88 \
            --hidden-size 4096 \
            --ffn-hidden-size 2688 \
            --num-attention-heads 32 \
            --group-query-attention \
            --num-query-groups 2 \
            --kv-channels 128 \
            --hybrid-override-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME \
            --position-embedding-type none \
            --normalization RMSNorm \
            --untie-embeddings-and-output-weights \
            --init-method-std 0.014 \
            --disable-bias-linear \
            --squared-relu \
            --use-fused-weighted-squared-relu \
            --seq-length 8192 \
            --max-position-embeddings 8192 \
            --num-experts 512 \
            --moe-router-topk 22 \
            --moe-router-topk-scaling-factor 5.0 \
            --moe-router-score-function sigmoid \
            --moe-router-enable-expert-bias \
            --moe-router-dtype fp32 \
            --moe-router-load-balancing-type seq_aux_loss \
            --moe-aux-loss-coeff 1e-4 \
            ${MOE_DISPATCHER_ARGS} \
            --moe-grouped-gemm \
            --moe-permute-fusion \
            --moe-latent-size 1024 \
            --moe-shared-expert-intermediate-size 5376 \
            --calculate-per-token-loss \
            --bf16 \
            --first-last-layers-bf16 \
            --num-layers-at-start-in-bf16 0 \
            --num-layers-at-end-in-bf16 14 \
            --fp4-format e2m1 \
            --fp4-recipe nvfp4 \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --sequence-parallel \
            --use-distributed-optimizer \
            --overlap-grad-reduce \
            --overlap-param-gather \
            --ddp-num-buckets 10 \
            --ddp-pad-buckets-for-high-nccl-busbw \
            --high-priority-stream-groups ep \
            --tensor-model-parallel-size 4 \
            --pipeline-model-parallel-size 1 \
            --expert-model-parallel-size 32 \
            --expert-tensor-parallel-size 1 \
            --cross-entropy-loss-fusion \
            --cross-entropy-fusion-impl native \
            --attention-backend flash \
            ${CUDA_GRAPH_ARGS} \
            --te-rng-tracker \
            --manual-gc \
            --manual-gc-interval 10 \
            --num-workers 1 \
            --eval-interval 1000 \
            --eval-iters 14 \
            --log-interval 1 \
            --log-params-norm \
            --log-num-zeros-in-grad \
            --log-timers-to-tensorboard \
            --log-memory-to-tensorboard \
            --log-throughput \
            --log-progress \
            --log-energy \
            --log-memory-interval 500 \
            --logging-level 20 \
            --timing-log-option minmax \
            --check-weight-hash-across-dp-replicas-interval 20000 \
            --tensorboard-dir /tensorboard \
            --local-rank ${SLURM_LOCALID} \
            --distributed-timeout-seconds-after-init 1 \
            --flight-recorder-dump-path /checkpoint-save \
        "
        WORKLOAD_ARGS="${WORKLOAD_ARGS} ${FAULT_INJECTOR_ARGS}"
        $LAUNCHER_CMD $LAUNCHER_ARGS $WORKLOAD_CMD $WORKLOAD_ARGS
    '
log_msg "END main_workload"

log_msg "END SBATCH"

set +x
