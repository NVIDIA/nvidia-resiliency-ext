#!/bin/bash
# NVRx colocated in-job restart, deployed as a singleton job-array chain.
#
# Submit with submit_chain.sh, which computes --array/--nodes/--job-name from the
# same env vars this script reads and enqueues the chain:
#
#     ./submit_chain.sh                       # default: NO-RESTART (exit 93) demo --
#                                             # exit 93 cancels the whole queued chain
#     NVRX_NO_RESTART_DEMO=0 ./submit_chain.sh # hand-off demo instead (see below/README)
#
# Direct `sbatch nvrx_singleton_array.sbatch` also works; the #SBATCH defaults below
# then apply and only one generation is queued.
#
# SINGLETON RESTART MODEL
#   Array task 0 is both a training rank and the rendezvous host, so its liveness IS
#   the generation's liveness. Losing it ends the whole array; --dependency=singleton
#   means the next queued job takes over from the last checkpoint. --no-requeue turns
#   requeue off for every cause, NODE_FAIL included, so task 0 can never reappear under
#   its old index. That is what lets the other tasks treat the RDZV_CLOSED flag as final
#   and exit on the spot, instead of waiting to see whether task 0 is coming back. The
#   price is that a dead node is no longer repaired in place -- it costs a generation.
#
# See ../README.md for the model, README.md for the knobs.

#SBATCH --job-name=nvrx_singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
# 2 active (1 training + 1 hot spare), the rest cold spares. Deeper pool = longer
# generation and a more visible pending-task cancellation. Check MaxArraySize first.
#SBATCH --array=0-4%2
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --time=00:25:00

set -x

# ---------------------------------------------------------------------------------
# Configuration -- every value below is overridable from the environment.
# ---------------------------------------------------------------------------------
# Shape. TRAIN_TASKS * nodes-per-task is the world the model is sized for; hot spares
# are extra tasks that rendezvous but stay standby with 0 workers until one is needed.
NVRX_TRAIN_TASKS="${NVRX_TRAIN_TASKS:-1}"
NVRX_HOT_SPARES="${NVRX_HOT_SPARES:-1}"
# 4 is the GB200 tray default; set 8 for HGX/DGX nodes.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# Where everything is written. Must be on a filesystem shared by every node: the
# rendezvous host is published through it.
NVRX_WORK_DIR="${NVRX_WORK_DIR:-${PWD}/nvrx-run}"

# Public Megatron-LM checkout (https://github.com/NVIDIA/Megatron-LM). Inside the
# container if NVRX_CONTAINER_IMAGE is set.
MEGATRON_PATH="${MEGATRON_PATH:-/workspace/megatron-lm}"

# Container is optional. Leave NVRX_CONTAINER_IMAGE empty to run in the bare
# environment (module-loaded venv, conda, whatever srun inherits). The mount default
# covers the common case where the shared filesystem is /lustre; override for other
# layouts (ignored when no image is set). Empty means "mount nothing" -- a site with no
# /lustre must be able to clear this, or pyxis refuses to start the container on a bind
# source that does not exist -- so the expansion is ${VAR-default}, not ${VAR:-default}.
# submit_chain.sh exports this one, so its default must stay in sync with this line.
NVRX_CONTAINER_IMAGE="${NVRX_CONTAINER_IMAGE:-}"
NVRX_CONTAINER_MOUNTS="${NVRX_CONTAINER_MOUNTS-/lustre:/lustre}"

# small (default, runs anywhere) or 8b (8B dense; ~2 nodes of GB200 is enough).
NVRX_MODEL_PROFILE="${NVRX_MODEL_PROFILE:-small}"

# Optional. Endpoint of the node health check daemon; omitted when empty.
NVRX_HEALTH_CHECK_ENDPOINT="${NVRX_HEALTH_CHECK_ENDPOINT:-}"
# Minimum nodes per NVLink domain, identified by GPU ClusterUUID -- a restart cycle then
# only ever selects complete segments. 1 suits GB200 NVL72 and other NVLink-domain
# systems (DGX H200, HGX B200). Set empty on systems with no ClusterUUID (e.g. H100),
# which is NVRx's simple hot spare mode -- so, again, ${VAR-default} and not
# ${VAR:-default}. Passing --ft-segment 1 where there is no ClusterUUID is not a degraded
# mode: the rendezvous raises RuntimeError on every node of every cycle and the whole
# chain burns. NVRX_SEGMENT=0 is not the escape hatch either -- it is still -n, so it
# emits --ft-segment 0 rather than omitting the flag.
NVRX_SEGMENT="${NVRX_SEGMENT-1}"
# ---------------------------------------------------------------------------------
# ***  DEMO MODE -- ON BY DEFAULT  ***
# Out of the box this runs the NO-RESTART (exit 93) demo: NVRx declares "no progress",
# exits 93, and task 0's trap cancels the whole queued chain (NVRX_NO_RESTART_DEMO below;
# raise NVRX_CHAIN_DEPTH so there are several generations to cancel). Two other modes:
#   Hand-off demo:   NVRX_NO_RESTART_DEMO=0 ./submit_chain.sh
#                    (workload fault -> in-job restart, then a GPU failure on task 0 ->
#                     singleton hand-off resuming from checkpoint)
#   Real training:   NVRX_NO_RESTART_DEMO=0 NVRX_FAULT_INJECT=0 \
#                    NVRX_INJECT_GPU_FAILURE= ./submit_chain.sh
# ---------------------------------------------------------------------------------
# Workload fault: inject into rank N after M seconds via megatron.core.fault_injector
# (public Megatron-LM). Drives an in-job restart on a training rank; it does NOT
# exercise the rank-0/host-failure path -- NVRX_INJECT_GPU_FAILURE below does that.
NVRX_FAULT_INJECT="${NVRX_FAULT_INJECT:-1}"
NVRX_FAULT_RANKS="${NVRX_FAULT_RANKS:-2}"
NVRX_FAULT_DELAY="${NVRX_FAULT_DELAY:-60}"
NVRX_FAULT_TYPES="${NVRX_FAULT_TYPES:-async_exc}"
# Node HW failure -- the headline path this deployment exists for. NVRx's health-check
# injector (testing_utils/health_check_injector.py) fails a physical node before
# rendezvous, keyed by "cycle:infra_rank[,cycle:infra_rank...]" where infra_rank is the
# physical node rank. infra_rank 0 is array task 0 -- the rendezvous host -- so "1:0"
# fails it at cycle 1, ending the generation and forcing the singleton chain to start
# the next one from the last checkpoint. The default "1:0,2:0" fires after the workload
# fault has advanced the cycle and written a checkpoint. Forwarded into the container
# below.
#
# The expansion is ${VAR-default}, NOT ${VAR:-default}: only the former distinguishes
# "unspecified" (unset -> take the demo default) from "the operator turned it off"
# (NVRX_INJECT_GPU_FAILURE= -> stay empty). With the colon, empty substitutes the default
# too, and the real-training recipe above silently gets task 0's node failed at cycles 1
# and 2. Do not put the colon back.
NVRX_INJECT_GPU_FAILURE="${NVRX_INJECT_GPU_FAILURE-1:0,2:0}"
# NO-RESTART (exit 93) demo. Points the progress tracker at a checkpoint-iteration file
# that never advances (seeded below). The tracker seeds its baseline from that file at
# startup, so every cycle -- including the first -- reads the same number and scores as
# "no progress"; after NVRX_MAX_NO_PROGRESS_CYCLES such cycles (default 1, so the first
# fault) NVRx exits 93, and task 0's trap runs cancel_chain -- cancelling the whole
# queued chain instead of starting the next generation. Off by default; turn it on with
# NVRX_NO_RESTART_DEMO=1 (it needs the workload fault for the restart and turns the
# host-failure injector off, since that would hand off instead).
NVRX_NO_RESTART_DEMO="${NVRX_NO_RESTART_DEMO:-1}"
NVRX_MAX_NO_PROGRESS_CYCLES="${NVRX_MAX_NO_PROGRESS_CYCLES:-}"
if [[ "$NVRX_NO_RESTART_DEMO" == "1" ]]; then
    NVRX_FAULT_INJECT=1
    NVRX_INJECT_GPU_FAILURE=""
    NVRX_MAX_NO_PROGRESS_CYCLES="${NVRX_MAX_NO_PROGRESS_CYCLES:-1}"
fi
export NVRX_INJECT_GPU_FAILURE

RDZV_PORT="${RDZV_PORT:-29400}"
export NVRX_LOG_DEBUG="${NVRX_LOG_DEBUG:-true}"
# GPUs per NUMA node, used for rank-to-NUMA binding. 2 matches a 4-GPU GB200 tray.
export NVRX_GPUS_PER_NUMA="${NVRX_GPUS_PER_NUMA:-2}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
# One MNNVL clique per array task: tasks are independent allocations and must not be
# fused into a single NVLink domain. Harmless on non-MNNVL systems.
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
export NCCL_MNNVL_CLIQUE_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ---------------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------------
NODES_PER_TASK="${SLURM_JOB_NUM_NODES:-1}"

# The active width is the %N throttle on --array, and --nnodes max must equal it: a
# larger max would have the rendezvous wait for tasks SLURM will never start while one
# is running. Read it from the controller rather than trusting an env var to track the
# throttle by hand -- that duplication is a standing footgun. Env is the fallback.
NVRX_ACTIVE_TASKS="${NVRX_ACTIVE_TASKS:-$(( NVRX_TRAIN_TASKS + NVRX_HOT_SPARES ))}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    THROTTLE=$(scontrol show job "${SLURM_ARRAY_JOB_ID}" 2>/dev/null \
               | grep -o 'ArrayTaskThrottle=[0-9]*' | head -n1 | cut -d= -f2)
    if [[ -n "$THROTTLE" && "$THROTTLE" -gt 0 ]]; then
        if [[ "$THROTTLE" != "$NVRX_ACTIVE_TASKS" ]]; then
            echo "Array throttle is ${THROTTLE}, not ${NVRX_ACTIVE_TASKS}; using the throttle."
        fi
        NVRX_ACTIVE_TASKS="$THROTTLE"
    fi
fi

MIN_NODES=$(( NVRX_TRAIN_TASKS * NODES_PER_TASK ))
MAX_NODES=$(( NVRX_ACTIVE_TASKS * NODES_PER_TASK ))
WORLD_SIZE=$(( MIN_NODES * GPUS_PER_NODE ))

LOGS_DIR="${NVRX_WORK_DIR}/logs"
CHECKPOINT_DIR="${NVRX_WORK_DIR}/checkpoints"
TENSORBOARD_DIR="${NVRX_WORK_DIR}/tensorboard"
NVRX_DIR="${NVRX_WORK_DIR}/nvrx"
mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${NVRX_DIR}"

NVRX_JOB_DIR="${NVRX_DIR}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p "${NVRX_JOB_DIR}"/{control,cycle_infos,logs}

# Control files for this generation. The directory is keyed by SLURM_ARRAY_JOB_ID, so
# each generation gets a fresh one -- load-bearing, not tidiness: task 0 publishes
# rdzv_host and would-be-stale RDZV_CLOSED are not written atomically with respect to
# each other, and a peer that saw a new host next to a previous generation's flag would
# abandon a generation that had just come up healthy. Do not consolidate this directory
# across generations.
RDZV_FILE=${NVRX_JOB_DIR}/control/rdzv_host
RDZV_CLOSED=${NVRX_JOB_DIR}/control/rdzv_closed

# The no-restart verdict as a file, because $? on task 0 cannot carry it wherever the site
# sets KillOnBadExit=1 (srun's default source; slurm.conf defaults it to 0, and this script
# does not override either way). There, at NODES_PER_TASK>1, the first peer to exit
# NO_RESTART_EXIT_CODE tears the step down while the store host is by design the last one
# alive -- it sleeps ~3s after closing the rendezvous so peers can read the final TCPStore
# state -- so it is signalled mid-wait and srun hands this script 143. task0_exit would then
# read the ordinary branch and the chain would restart a job NVRx said must not restart.
# Measured, not deduced: srun returns 143 with the peer's 93 already on disk.
#
# Race-free because SLURM's task is the `sh` below, not ft_launcher: the step teardown
# cannot begin until an `sh` exits, and the touch precedes that exit. Whichever node fires
# the kill has already written the flag.
#
# Generation-scoped on purpose. A chain-scoped twin would make cancel_chain durable --
# level-triggered, so a successor that escaped the scancel would stop itself -- but only if
# something reliably clears it when a new chain starts, and the sbatch is submittable
# without submit_chain.sh. A stale flag would then refuse every later run into the same
# NVRX_WORK_DIR, which is a worse failure than the one it prevents. cancel_chain stays the
# mechanism; nvrx-watch is the out-of-band backstop for a scancel that failed.
NO_RESTART_FLAG=${NVRX_JOB_DIR}/control/no_restart

# The file the progress tracker watches. Normally Megatron's live iteration file, which
# advances every checkpoint. In the no-restart demo it is a static file seeded once by
# task 0 and never touched again, so every restart cycle reads the same number -> no
# progress -> exit 93.
FT_CKPT_ITER_FILE="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
if [[ "$NVRX_NO_RESTART_DEMO" == "1" ]]; then
    FT_CKPT_ITER_FILE="${NVRX_JOB_DIR}/control/static_iter.txt"
    if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" && ! -f "$FT_CKPT_ITER_FILE" ]]; then
        echo 1 > "$FT_CKPT_ITER_FILE"   # non-zero so tracking is active; never advances
    fi
fi

# One timestamp for the whole generation, so every task's logs share a suffix.
DATETIME_FILE=${NVRX_JOB_DIR}/control/datetime.txt
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    date +'date_%y-%m-%d_time_%H-%M-%S' > "${DATETIME_FILE}.tmp" \
        && mv -f "${DATETIME_FILE}.tmp" "$DATETIME_FILE"
fi
# Unbounded BY CHOICE, and deliberately not configurable. Backfill can schedule task 0
# well after its peers, so a timeout here races the scheduler rather than any real
# event, and firing it would kill a generation that was about to work. The cost is that
# peers hold their nodes until task 0 arrives or the walltime expires. The rdzv_host
# wait further down is bounded instead, and safely: by the time a peer waits on that,
# task 0 has demonstrably already started.
#
# RDZV_CLOSED is still checked here, so a task 0 that started and then exited releases
# its peers immediately rather than leaving them here until the walltime.
DT_WAIT=0
while [[ ! -s "$DATETIME_FILE" ]]; do
    if [[ -f "$RDZV_CLOSED" ]]; then
        echo "RDZV_CLOSED present before task 0 published: generation is over. Exiting."
        exit 1
    fi
    if (( DT_WAIT % 60 == 0 )); then
        echo "task ${SLURM_ARRAY_TASK_ID} waiting for task 0 to start (${DT_WAIT}s)."
    fi
    sleep 1
    (( DT_WAIT += 1 ))
done
DATETIME=$(cat "$DATETIME_FILE")

# ---------------------------------------------------------------------------------
# Workload arguments -- public Megatron-LM, mock data
# ---------------------------------------------------------------------------------
# NullTokenizer + --mock-data: no corpus, no vocab file, nothing to download. The
# tensor shapes and the step time are representative; the loss is not.
DATA_ARGS=" \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    --no-create-attention-mask-in-dataloader \
    --num-workers 1 "

if [[ "$NVRX_MODEL_PROFILE" == "8b" ]]; then
    # 8B dense. GLOBAL_BATCH_SIZE must be divisible by the data-parallel size, which is
    # WORLD_SIZE/(TP*PP) -- computed from MIN_NODES, since hot spares stay standby and
    # do not enlarge the world.
    MODEL_ARGS=" \
        --num-layers 32 \
        --hidden-size 4096 \
        --ffn-hidden-size 21504 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 8 \
        --kv-channels 128 \
        --squared-relu \
        --init-method-std 0.014 \
        --rotary-base 1000000 \
        --seq-length 8192 \
        --max-position-embeddings 8192 \
        --recompute-granularity selective "
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( WORLD_SIZE * 2 ))}"
    TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
    PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
else
    # Small enough to reach a step and a checkpoint on a couple of GPUs, which is all
    # this example needs to demonstrate the restart behaviour.
    MODEL_ARGS=" \
        --num-layers 8 \
        --hidden-size 1024 \
        --ffn-hidden-size 4096 \
        --num-attention-heads 16 \
        --swiglu \
        --init-method-std 0.02 \
        --seq-length 1024 \
        --max-position-embeddings 1024 "
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${WORLD_SIZE}}"
    TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
    PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
fi

FAULT_ARGS=""
if [[ "$NVRX_FAULT_INJECT" == "1" ]]; then
    FAULT_ARGS=" \
        --fault-injector-ranks ${NVRX_FAULT_RANKS} \
        --fault-injector-fault-delay ${NVRX_FAULT_DELAY} \
        --fault-injector-fault-types ${NVRX_FAULT_TYPES} "
fi

# --enable-ft-package is what makes Megatron call NVRx's section API, so the
# --ft-rank-section-timeouts below actually bound setup, step and checkpointing.
options=" \
    --enable-ft-package \
    ${DATA_ARGS} \
    ${MODEL_ARGS} \
    ${FAULT_ARGS} \
    --use-mcore-models \
    --distributed-timeout-minutes 10 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size ${TENSOR_PARALLEL} \
    --pipeline-model-parallel-size ${PIPELINE_PARALLEL} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-samples ${TRAIN_SAMPLES:-10485760} \
    --lr-decay-samples ${LR_DECAY_SAMPLES:-1038336} \
    --lr-warmup-samples 10240 \
    --lr 6e-4 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --manual-gc \
    --exit-duration-in-mins ${EXIT_DURATION_IN_MINS:-110} \
    --eval-iters 14 \
    --eval-interval 2000 \
    --log-interval 20 \
    --log-progress \
    --log-throughput \
    --log-energy \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --timing-log-option minmax \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval ${SAVE_INTERVAL:-20} \
    --save-retain-interval ${SAVE_RETAIN_INTERVAL:-10000} \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --ckpt-assume-constant-structure \
    --async-save \
    --use-persistent-ckpt-worker \
    --tensorboard-dir ${TENSORBOARD_DIR} "

# ---------------------------------------------------------------------------------
# Generation teardown
# ---------------------------------------------------------------------------------
# End this generation outright -- every task, any state, scoped to this array by
# $SLURM_ARRAY_JOB_ID. Deliberately not --state=PENDING: that filter is evaluated once, so
# it cannot touch a cold spare that already left PENDING, and such a spare then reads a
# published rdzv_host, probes a store host that is gone, and waits out
# store_connect_wait_seconds (300s) holding a node. --dependency=singleton holds the
# successor generation until this array fully exits, so that drain is on the chain's
# critical path. Cancelling the array job itself is also level-triggered: no remaining task
# of it can start afterwards, which no state filter can promise.
#
# The cost is that RUNNING peers lose the tail of their NVRx teardown. Bounded, not abrupt:
# SLURM sends SIGTERM first and waits KillWait before SIGKILL, and ft_launcher handles
# SIGTERM by stopping its ranks (see --ft-* termination intervals). Confirm KillWait
# exceeds those intervals for the site, or the tail is cut anyway.
#
# Includes task 0, i.e. this script: nothing may follow the call in the EXIT trap.
cancel_generation() {
    echo "Cancelling all remaining tasks of array ${SLURM_ARRAY_JOB_ID}."
    if ! scancel "${SLURM_ARRAY_JOB_ID}"; then
        # Not fatal and not silent: nvrx-watch corrects this state from outside, on a
        # retrying loop. Swallowing the status here is what made it undiagnosable.
        echo "WARNING: scancel of this generation failed; nvrx-watch should reconcile."
    fi
}

# ft_launcher exits with this when NVRx has decided the job must not restart --
# attribution returned STOP, or the progress tracker saw no checkpoint progress.
# Matches DEFAULT_NO_RESTART_EXIT_CODE in fault_tolerance/utils.py.
NO_RESTART_EXIT_CODE="${NVRX_NO_RESTART_EXIT_CODE:-93}"

# On that verdict the queued successor generations have to go too, or the singleton
# chain simply starts the next one and reproduces the same failure. Matched by job
# name, so this also covers the current array's own pending spares. Blast radius is
# every PENDING job of ours under this name -- which under singleton is exactly the
# chain, since the dependency already forbids two of them running at once. The flip
# side: a second run sharing this job name would have its queued generations cancelled
# too, so give each concurrent run a distinct NVRX_JOB_NAME.
cancel_chain() {
    if [[ -z "${SLURM_JOB_NAME:-}" ]]; then
        echo "SLURM_JOB_NAME unset; refusing to cancel by name."
        return 0
    fi
    echo "Exit ${1}: NVRx requested no restart. Cancelling queued ${SLURM_JOB_NAME} jobs."
    scancel --state=PENDING --name="${SLURM_JOB_NAME}" \
            --user="${SLURM_JOB_USER:-$USER}" \
        || echo "WARNING: chain cancel failed; the successor generation may still start."
}

# $? must be read first -- any command here would overwrite it. Bash saves the pending
# exit status before running an EXIT trap and restores it after, so nothing below can
# change the code SLURM records.
#
# The flag is tested before rc and not merely alongside it: rc is the code that survived
# ft_launcher -> sh -c -> srun, and under kill-on-bad-exit task 0's own launcher is the one
# whose code the teardown rewrites (see NO_RESTART_FLAG). A flag written by any node that
# did exit cleanly is the more trustworthy witness.
task0_exit() {
    local rc=$?
    echo "Task 0 exiting (rc=${rc}), creating RDZV_CLOSED file"
    touch "$RDZV_CLOSED"
    if [[ -f "$NO_RESTART_FLAG" ]] || (( rc == NO_RESTART_EXIT_CODE )); then
        cancel_chain "$rc"
    fi
    # Last: this cancels task 0 too, so nothing after it is guaranteed to run.
    cancel_generation
}

# ---------------------------------------------------------------------------------
# Rendezvous host publication and gate
# ---------------------------------------------------------------------------------
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    # Armed FIRST, before anything that can fail. Peers wait on rdzv_host, so an exit
    # leaving no RDZV_CLOSED would hang them until RH_WAIT expires -- exactly what a
    # naive "validate, then exit 1" placed above this line would cause. Primary path:
    # covers every orderly exit, but not a hard node death.
    trap task0_exit EXIT
    # The batch script's own node, which SLURM guarantees is the first node of the
    # allocation -- the same value scontrol yields, but needing no SLURM, no env var
    # and no PATH.
    RDZV_HOST=$(scontrol show hostnames "$SLURM_NODELIST" 2>/dev/null | head -n1)
    [[ -z "$RDZV_HOST" ]] && RDZV_HOST=$(hostname)
    if [[ -z "$RDZV_HOST" ]]; then
        echo "FATAL: cannot determine rendezvous host (SLURM_NODELIST='${SLURM_NODELIST}')" >&2
        exit 1
    fi
    # Publish by rename. 'echo > file' creates then writes, so a reader landing between
    # those two syscalls sees an existing but EMPTY file, passes its gate, and launches
    # at ':29400'. A rename is atomic: readers see nothing, or the whole value.
    printf '%s\n' "$RDZV_HOST" > "${RDZV_FILE}.tmp" && mv -f "${RDZV_FILE}.tmp" "$RDZV_FILE"
fi

# One loop, covering both cases: the flag already set when we arrive (task 0 published,
# then exited), and the flag appearing while we wait. Tested before the host, because
# when both are present task 0 is gone and the host it left is stale. Under --no-requeue
# the flag is always final, so exit is the right answer in both cases.
#
# Deliberately no scancel here. Task 0's trap already released the pool, and having
# every other task repeat it aims a burst of up to N scancel RPCs at slurmctld at the
# one moment it is also processing their mass exit -- N-1 of them no-ops, since the
# first cancel takes the whole pending set. nvrx-watch covers the case where task 0's
# own scancel failed, from outside the job and on a retrying loop.
#
# RH_WAIT backstops SIGKILL or a dead node, where neither file will ever appear. Its
# realistic trigger is a shared-filesystem stall delaying task 0's write.
RH_WAIT=0
while true; do
    if [[ -f "$RDZV_CLOSED" ]]; then
        echo "RDZV_CLOSED present on task ${SLURM_ARRAY_TASK_ID}: task 0 exited. Exiting."
        exit 1
    fi
    [[ -s "$RDZV_FILE" ]] && break   # -s, not -f: an empty file is not a host
    if (( RH_WAIT >= ${NVRX_RDZV_HOST_WAIT:-300} )); then
        echo "rdzv_host not published after ${RH_WAIT}s. Exiting task ${SLURM_ARRAY_TASK_ID}."
        exit 1
    fi
    sleep 1
    (( RH_WAIT += 1 ))
done
export RDZV_HOST=$(cat "$RDZV_FILE")

# ---------------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------------
NVRX_PER_CYCLE_APPLOG_PREFIX=${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${DATETIME}.log
NVRX_LOGFILE=${NVRX_JOB_DIR}/logs/nvrx_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${DATETIME}.log

OPTIONAL_ARGS=""
[[ -n "$NVRX_HEALTH_CHECK_ENDPOINT" ]] && \
    OPTIONAL_ARGS+=" --ft-node-health-check-endpoint=${NVRX_HEALTH_CHECK_ENDPOINT}"
[[ -n "$NVRX_SEGMENT" ]] && OPTIONAL_ARGS+=" --ft-segment ${NVRX_SEGMENT}"
[[ -n "$NVRX_MAX_NO_PROGRESS_CYCLES" ]] && \
    OPTIONAL_ARGS+=" --ft-max-no-progress-cycles ${NVRX_MAX_NO_PROGRESS_CYCLES}"

# store_connect_wait_seconds bounds the dead-generation drain: a cold spare that reaches
# CONFIGURING/RUNNING after task 0's srun exits but before its EXIT trap writes
# RDZV_CLOSED cannot be stopped by the flag (not written yet). It reads a published
# rdzv_host, breaks the gate above on its first iteration, and probes a host that is
# gone. Nothing upstream closes that race -- a task-0 liveness check does not either,
# since task 0 is still RUNNING while its own trap executes.
#
# cancel_generation is what normally collects such a spare, seconds later and whatever
# state it is in; this bound is the backstop for the case where that scancel failed, so
# it stays sized for a spare nobody cancels.
#
# 300 matches NVRX_RDZV_HOST_WAIT, so both escape paths cost the same: a spare that
# starts before publication waits out the gate, one that starts after waits out this.
# The drain is on the critical path -- the array holds its nodes until every task
# exits, and --dependency=singleton holds the successor generation until the array
# does. Raise it where task 0's startup can exceed it (container pull, cold shared FS):
# too tight and no generation ever forms, which burns the chain in minutes.
LAUNCHER_ARGS=" \
    --nnodes ${MIN_NODES}:${MAX_NODES} \
    --nproc-per-node ${GPUS_PER_NODE} \
    --rdzv-id ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} \
    --rdzv-endpoint ${RDZV_HOST}:${RDZV_PORT} \
    --rdzv-conf join_timeout=${NVRX_JOIN_TIMEOUT:-1200},store_connect_wait_seconds=${NVRX_STORE_CONNECT_WAIT:-300} \
    --max-restarts ${NVRX_MAX_RESTARTS:-7} \
    --ft-cycle-info-dir=${NVRX_JOB_DIR}/cycle_infos \
    --ft-per-cycle-applog-prefix=${NVRX_PER_CYCLE_APPLOG_PREFIX} \
    --ft-nvrx-logfile=${NVRX_LOGFILE} \
    --ft-checkpoint-iteration-file=${FT_CKPT_ITER_FILE} \
    --ft-rank-section-timeouts setup:900,step:60,checkpointing:300 \
    --ft-rank-out-of-section-timeout 90 \
    ${OPTIONAL_ARGS} "

# The trailing shell is what carries the no-restart verdict out of the step teardown: the
# launcher's own exit code does not survive it on the store host (see NO_RESTART_FLAG).
# Every \$ is escaped so the outer bash leaves it for the `sh -c` that srun runs.
run_cmd="ft_launcher ${LAUNCHER_ARGS} ${MEGATRON_PATH}/pretrain_gpt.py ${options}
rc=\$?
if [ \"\$rc\" -eq ${NO_RESTART_EXIT_CODE} ]; then touch '${NO_RESTART_FLAG}'; fi
exit \$rc"

# Names of host env vars to forward into the container. NVRX_GPUS_PER_NUMA (rank-to-NUMA
# binding) and NVRX_INJECT_GPU_FAILURE (the HW-failure injector) must reach the ranks;
# the latter only when set, so it stays inert by default.
CONTAINER_ENV="NCCL_MNNVL_ENABLE,NCCL_MNNVL_CLIQUE_ID,NVRX_LOG_DEBUG,NVRX_GPUS_PER_NUMA"
[[ -n "$NVRX_INJECT_GPU_FAILURE" ]] && CONTAINER_ENV+=",NVRX_INJECT_GPU_FAILURE"

SRUN_ARGS=(
    -l
    --ntasks="${SLURM_JOB_NUM_NODES}"
    --ntasks-per-node=1
    --gpus-per-node="${GPUS_PER_NODE}"
)
if [[ -n "$NVRX_CONTAINER_IMAGE" ]]; then
    SRUN_ARGS+=(
        --container-image "$NVRX_CONTAINER_IMAGE"
        --container-name "nvrx_node${SLURM_NODEID}"
        --no-container-mount-home
        --container-env="$CONTAINER_ENV"
    )
    [[ -n "$NVRX_CONTAINER_MOUNTS" ]] && SRUN_ARGS+=(--container-mounts "$NVRX_CONTAINER_MOUNTS")
fi

srun "${SRUN_ARGS[@]}" sh -c "${run_cmd}"
