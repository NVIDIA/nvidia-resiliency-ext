#!/bin/bash
# n3_super_gb200_shm_test.sh — one-time validation: Nemotron Super 8N with async cpu-shm ckpt.
# Model/infra config mirrors n3_super_gb200_fi.sh. No fault injection.
# Checkpoints to node-local /tmp (discardable — not cross-node accessible).

#SBATCH --account=root
#SBATCH --partition=gb-nvl-134-135
#SBATCH --time=00:45:00

#SBATCH --job-name=n3-super-shm-test
#SBATCH --output=/tmp/slurm-%j.launch.out
#SBATCH --error=/tmp/slurm-%j.launch.err

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0

log_msg() {
    local msg="$1"
    UNIX_DATETIME=$(date +%s)
    HUMAN_DATETIME=$(date -d "@$UNIX_DATETIME" '+%Y-%m-%d %H:%M:%S.%3N')
    echo ">>> ${msg} ${UNIX_DATETIME} (${HUMAN_DATETIME})"
}

log_msg "START SBATCH"
echo "Running on nodes: ${SLURM_NODELIST}"

# ── Platform / NCCL / RITS ────────────────────────────────────────────────────
export RITS_PLATFORM_TYPE=gb200
export RITS_GPUS_PER_NODE=4
export RITS_NVL_DOMAIN_SIZE=72
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export RITS_CLUSTER_NAME=nvl72
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
USE_DEEPEP="${USE_DEEPEP:-1}"
if [[ "${USE_DEEPEP}" == "1" ]]; then
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32
fi

# ── Logging / debugging ───────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export ONE_LOGGER_JOB_CATEGORY=test
export LOGLEVEL=DEBUG
export TORCH_CPP_LOG_LEVEL=INFO
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

# ── CUDA graph ────────────────────────────────────────────────────────────────
export ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-1}"

# ── Quantization mode: set USE_FP8=1 to use FP8, USE_FP4=1 for FP4 (default) ─
# Only one may be active at a time.
export USE_FP4="${USE_FP4:-0}"
export USE_FP8="${USE_FP8:-1}"

# ── Async checkpoint shm mode (default on) ────────────────────────────────────
export USE_CPU_SHM="${USE_CPU_SHM:-1}"

# ── Overlap comm (default off) ────────────────────────────────────────────────
export USE_OVERLAP_COMM="${USE_OVERLAP_COMM:-0}"

# ── Node / task geometry ─────────────────────────────────────────────────────
export GPUS_PER_NODE=4
TOTAL_TASKS=$((SLURM_NNODES * GPUS_PER_NODE))

# ── Per-experiment output directory ───────────────────────────────────────────
export BASE_EXPERIMENTS_DIR="${BASE_EXPERIMENTS_DIR:-/home/sbak/experiments/n3-super-gb200}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${BASE_EXPERIMENTS_DIR}/shm_test}"

mkdir -p ${BASE_EXPERIMENTS_DIR}/datacache
mkdir -p ${EXPERIMENT_DIR}/tensorboard

: "${SLURM_RESTART_COUNT:=0}"

LOG_DIR=${EXPERIMENT_DIR}/logs
mkdir -p ${LOG_DIR}
echo "Writing logs to ${LOG_DIR}"
LOG_FILE_BASE="${LOG_DIR}/slurm/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT}"

# ── Container mounts ──────────────────────────────────────────────────────────
LUSTRE=/home:/home
SHARED_TMP=/home/sbak/tmp/${SLURM_JOB_ID}:/shared_tmp
LOGS=${EXPERIMENT_DIR}/logs:/logs
MEGATRON_REPO=/home/sbak/megatron-lm-main:/megatron-lm_repo
DATACACHE=${BASE_EXPERIMENTS_DIR}/datacache:/datacache
TENSORBOARD=${EXPERIMENT_DIR}/tensorboard:/tensorboard
WORKSPACE=/home/sbak/tmp:/workspace
# No /checkpoints mount — saves go to node-local /tmp inside the container.
CONTAINER_MOUNTS=$LUSTRE,$SHARED_TMP,$LOGS,$MEGATRON_REPO,$DATACACHE,$TENSORBOARD,$WORKSPACE
mkdir -p /home/sbak/tmp/${SLURM_JOB_ID}

# ── Disk cleanup: remove stale enroot containers from prior jobs ──────────────
log_msg "START disk_cleanup"
srun \
    --label \
    --ntasks-per-node=1 \
    --ntasks=${SLURM_NNODES} \
    --kill-on-bad-exit=0 \
    --mpi=none \
    bash -c '
        ENROOT_DIR="/var/lib/enroot/data/$(id -u)"
        rm -rf "${ENROOT_DIR:?}"/* 2>/dev/null || true
        echo "$(hostname): / $(df -h / | tail -1 | awk "{print \$3\" used, \"\$4\" avail\"}")"
    '
log_msg "END disk_cleanup"

# ── All-node setup: clone Megatron into a per-node tmpdir ─────────────────────
log_msg "START all_node_setup"
srun \
    --label \
    --container-mounts ${CONTAINER_MOUNTS} \
    --container-image /home/sbak/mcore_ci_040825.sqsh \
    --container-name ${SLURM_JOB_ID} \
    --container-workdir / \
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
        popd

        # Install local nvidia-resiliency-ext so container picks up src changes.
        uv pip install -e /home/sbak/nvidia-resiliency-ext
    '
log_msg "END all_node_setup"

# ── Main workload ─────────────────────────────────────────────────────────────
log_msg "START main_workload"
srun \
    --label \
    --container-mounts ${CONTAINER_MOUNTS} \
    --container-image /home/sbak/mcore_ci_040825.sqsh \
    --container-name ${SLURM_JOB_ID} \
    --container-workdir / \
    --error=${LOG_FILE_BASE}.1.main_workload.log \
    --output=${LOG_FILE_BASE}.1.main_workload.log \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --ntasks=${TOTAL_TASKS} \
    --kill-on-bad-exit=0 \
    --mpi=none \
    bash -c '
        MEGATRON_PATH=/shared_tmp/megatron_${SLURM_NODEID}

        NFS_TRITON_CACHE=/home/sbak/experiments/n3-super-gb200/triton_cache
        NFS_INDUCTOR_CACHE=/home/sbak/experiments/n3-super-gb200/inductor_cache
        TRITON_READY=/tmp/.triton_ready_${SLURM_JOB_ID}

        export TRITON_CACHE_DIR=/tmp/triton_${SLURM_LOCALID}
        export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_LOCALID}

        if [[ "${SLURM_LOCALID}" == "0" ]]; then
            if [[ -d "${NFS_TRITON_CACHE}" ]] && [[ -n "$(ls -A ${NFS_TRITON_CACHE} 2>/dev/null)" ]]; then
                TRITON_CACHE_WAS_WARM=1
            else
                TRITON_CACHE_WAS_WARM=0
            fi
            for r in $(seq 0 $((GPUS_PER_NODE - 1))); do
                mkdir -p /tmp/triton_${r} /tmp/inductor_${r}
                [[ -d "${NFS_TRITON_CACHE}" ]]   && rsync -a --ignore-existing "${NFS_TRITON_CACHE}/"   "/tmp/triton_${r}/"   2>/dev/null || true
                [[ -d "${NFS_INDUCTOR_CACHE}" ]] && rsync -a --ignore-existing "${NFS_INDUCTOR_CACHE}/" "/tmp/inductor_${r}/" 2>/dev/null || true
            done
            touch "${TRITON_READY}"
            echo "Pre-staged triton/inductor cache for all local ranks (was_warm=${TRITON_CACHE_WAS_WARM})."
        else
            until [[ -f "${TRITON_READY}" ]]; do sleep 1; done
        fi

        mkdir -p ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}

        _stage_back() {
            if [[ "${SLURM_LOCALID}" == "0" && "${SLURM_NODEID}" == "0" && "${TRITON_CACHE_WAS_WARM}" == "0" ]]; then
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

        if [[ "${USE_FP8:-0}" == "1" ]]; then
            QUANT_ARGS="--fp8-param-gather \
            --reuse-grad-buf-for-mxfp8-param-ag \
            --fp8-recipe mxfp8 \
            --fp8-format hybrid \
            --fp8-amax-history-len 1024 \
            --fp8-amax-compute-algo max"
        elif [[ "${USE_FP4:-1}" == "1" ]]; then
            QUANT_ARGS="--first-last-layers-bf16 \
            --num-layers-at-start-in-bf16 0 \
            --num-layers-at-end-in-bf16 14 \
            --fp4-format e2m1 \
            --fp4-recipe nvfp4"
        else
            QUANT_ARGS=""
        fi

        # Checkpoint directory — node-local /tmp inside the container.
        # Shards are not cross-node accessible; intentional for one-time shm validation.
        CKPT_DIR=/tmp/ckpt_${SLURM_JOB_ID}
        mkdir -p ${CKPT_DIR}

        pushd $MEGATRON_PATH
        LAUNCHER_CMD="python3"
        WORKLOAD_CMD=${MEGATRON_PATH}/pretrain_mamba.py
        WORKLOAD_ARGS=" \
            --exit-duration-in-mins 40 \
            --exit-interval 100 \
            --distributed-timeout-minutes 30 \
            --distributed-timeout-seconds-after-init 1800 \
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
            ${QUANT_ARGS} \
            --attention-dropout 0.0 \
            --hidden-dropout 0.0 \
            --sequence-parallel \
            --use-distributed-optimizer \
            $([[ "${USE_OVERLAP_COMM}" == "1" ]] && echo "--overlap-grad-reduce --overlap-param-gather") \
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
            --log-energy \
            --log-memory-interval 500 \
            --logging-level 10 \
            --timing-log-option minmax \
            --check-weight-hash-across-dp-replicas-interval 20000 \
            --tensorboard-dir /tensorboard \
            --local-rank ${SLURM_LOCALID} \
            --save ${CKPT_DIR} \
            --save-interval 10 \
            --ckpt-format torch_dist \
            --ckpt-fully-parallel-save \
            --ckpt-assume-constant-structure \
            --async-save \
            --use-persistent-ckpt-worker \
            $([[ "${USE_CPU_SHM}" == "1" ]] && echo "--async-ckpt-use-cpu-shm") \
        "
        $LAUNCHER_CMD $WORKLOAD_CMD $WORKLOAD_ARGS
    '
log_msg "END main_workload"

log_msg "END SBATCH"

set +x
