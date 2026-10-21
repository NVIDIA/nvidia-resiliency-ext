#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Small GPT recovery example. See docs/source/fault_tolerance/examples/megatron_gpt.rst.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MEGATRON_PATH:?Set MEGATRON_PATH to an installed Megatron-LM source checkout}"
: "${NVRX_WORK_DIR:?Set NVRX_WORK_DIR to an absolute directory visible on all nodes}"
[[ "$NVRX_WORK_DIR" == /* ]] || { echo 'NVRX_WORK_DIR must be absolute' >&2; exit 2; }
[[ -f "$MEGATRON_PATH/pretrain_gpt.py" ]] || { echo 'pretrain_gpt.py not found' >&2; exit 2; }

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
TP="${TP:-1}"
PP="${PP:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
NVRX_DEMO_FAULT="${NVRX_DEMO_FAULT:-kill}"
case "$NVRX_DEMO_FAULT" in
    none|kill|hang) ;;
    *) echo 'NVRX_DEMO_FAULT must be none, kill, or hang' >&2; exit 2 ;;
esac
for value in "$NNODES" "$GPUS_PER_NODE" "$TP" "$PP" "$TRAIN_ITERS" "$SAVE_INTERVAL"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Shape and iteration settings must be positive integers' >&2; exit 2; }
done
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
(( WORLD_SIZE % (TP * PP) == 0 && 4 % PP == 0 && 8 % TP == 0 )) || {
    echo 'World size must be divisible by TP*PP; PP must divide four layers and TP eight heads' >&2; exit 2;
}
if (( NNODES > 1 )); then
    : "${RDZV_ENDPOINT:?Set a common reachable host:port for all nodes}"
    : "${NVRX_RUN_ID:=${SLURM_JOB_ID:-}}"
    : "${NVRX_RUN_ID:?Set the same unique run ID on all nodes}"
else
    RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:0}"
    NVRX_RUN_ID="${NVRX_RUN_ID:-$(date -u +%Y%m%dT%H%M%S)-$$}"
fi
[[ "$NVRX_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Use letters, digits, underscores or hyphens in NVRX_RUN_ID' >&2; exit 2; }
RUN_DIR="${NVRX_WORK_DIR}/runs/${NVRX_RUN_ID}"
CHECKPOINT_DIR="${NVRX_CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
[[ "$CHECKPOINT_DIR" == /* ]] || { echo 'NVRX_CHECKPOINT_DIR must be absolute' >&2; exit 2; }
if [[ -e "$RUN_DIR/fault_injected.json" ]]; then
    echo 'This run ID already has an injection marker; use a new NVRX_RUN_ID' >&2
    exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/cycle_infos" "$CHECKPOINT_DIR"
if [[ -e "$CHECKPOINT_DIR/ft_state.json" ]]; then
    echo 'This fixed-timeout example requires a checkpoint directory without calculated ft_state.json' >&2
    exit 2
fi

launcher=(
    ft_launcher --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE"
    --rdzv-backend=c10d --rdzv-endpoint="$RDZV_ENDPOINT" --rdzv-id="$NVRX_RUN_ID"
    --max-restarts=3
    --ft-rank-section-timeouts=setup:900,step:120,checkpointing:600
    --ft-rank-out-of-section-timeout=180
    --ft-checkpoint-iteration-file="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
    --ft-cycle-info-dir="$RUN_DIR/cycle_infos"
    --ft-per-cycle-applog-prefix="$RUN_DIR/logs/train.log"
    --ft-nvrx-logfile="$RUN_DIR/logs/launcher.log"
)
workload=(
    "$SCRIPT_DIR/megatron_gpt_worker.py"
    --megatron-path "$MEGATRON_PATH" --checkpoint-dir "$CHECKPOINT_DIR"
    --fault-marker "$RUN_DIR/fault_injected.json" --fault "$NVRX_DEMO_FAULT"
    --fault-rank "${NVRX_FAULT_RANK:-1}" --
    --enable-ft-package --ft-num-warmup-iters=0
    --mock-data --tokenizer-type=NullTokenizer --vocab-size=1024
    --num-layers=4 --hidden-size=512 --num-attention-heads=8
    --seq-length=256 --max-position-embeddings=256
    --tensor-model-parallel-size="$TP" --pipeline-model-parallel-size="$PP"
    --micro-batch-size=1 --global-batch-size="$WORLD_SIZE"
    --train-iters="$TRAIN_ITERS" --lr=0.0001 --lr-decay-style=constant
    --bf16 --use-distributed-optimizer
    --load="$CHECKPOINT_DIR" --save="$CHECKPOINT_DIR" --save-interval="$SAVE_INTERVAL"
    --ckpt-format=torch_dist --async-save --use-persistent-ckpt-worker
    --eval-iters=0 --log-interval=1
)

printf 'Run artifacts: %s\nCheckpoint directory: %s\n' "$RUN_DIR" "$CHECKPOINT_DIR"
if [[ "${NVRX_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${launcher[@]}" "${workload[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${launcher[@]}" "${workload[@]}" "$@"
