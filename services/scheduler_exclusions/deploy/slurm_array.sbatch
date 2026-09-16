#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reference lifecycle for one Scheduler Exclusion service per Slurm job array.
# Supply the allocation directives through sbatch or adapt this file locally.

#SBATCH --job-name=nvrx-scheduler-exclusion
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SERVICE="${NVRX_SCHEDULER_EXCLUSION_RUN_SERVICE:-${SCRIPT_DIR}/run_service.sh}"
SCHEDULER_EXCLUSION_DIR="${NVRX_SCHEDULER_EXCLUSION_DIR:?Set NVRX_SCHEDULER_EXCLUSION_DIR to a shared absolute path}"
SERVICE_HOST="${NVRX_SCHEDULER_EXCLUSION_HOST:-127.0.0.1}"
SERVICE_PORT="${NVRX_SCHEDULER_EXCLUSION_PORT:-18080}"
SERVICE_PYTHON="${NVRX_SCHEDULER_EXCLUSION_PYTHON:-python3}"
SERVICE_START_TIMEOUT_SECONDS="${NVRX_SCHEDULER_EXCLUSION_START_TIMEOUT_SECONDS:-60}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
SERVICE_LOG="${SCHEDULER_EXCLUSION_DIR}/scheduler_exclusion_service.${JOB_ID}.log"

if [[ $# -eq 0 ]]; then
    echo "Pass ft_launcher arguments and its workload after the batch file." >&2
    exit 2
fi
if [[ "${SCHEDULER_EXCLUSION_DIR}" != /* ]]; then
    echo "NVRX_SCHEDULER_EXCLUSION_DIR must be absolute: ${SCHEDULER_EXCLUSION_DIR}" >&2
    exit 2
fi

mkdir -p "${SCHEDULER_EXCLUSION_DIR}"

supervisor_pid=""

stop_supervisor() {
    if [[ -n "${supervisor_pid}" ]]; then
        kill "${supervisor_pid}" 2>/dev/null || true
        wait "${supervisor_pid}" 2>/dev/null || true
        supervisor_pid=""
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_supervisor
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

supervise_service() {
    local child_pid=""
    local delay=1
    local rc=0

    stop_child() {
        if [[ -n "${child_pid}" ]]; then
            kill "${child_pid}" 2>/dev/null || true
            wait "${child_pid}" 2>/dev/null || true
            child_pid=""
        fi
    }
    trap 'stop_child; exit 0' INT TERM

    while true; do
        printf '%s starting Scheduler Exclusion service\n' \
            "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"${SERVICE_LOG}"
        "${RUN_SERVICE}" \
            --host "${SERVICE_HOST}" \
            --port "${SERVICE_PORT}" \
            --output-dir "${SCHEDULER_EXCLUSION_DIR}" \
            >>"${SERVICE_LOG}" 2>&1 &
        child_pid=$!

        set +e
        wait "${child_pid}"
        rc=$?
        set -e
        child_pid=""

        printf '%s service exited rc=%s; restarting in %ss\n' \
            "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${rc}" "${delay}" >>"${SERVICE_LOG}"
        sleep "${delay}" &
        child_pid=$!
        wait "${child_pid}"
        child_pid=""
        if (( delay < 8 )); then
            delay=$((delay * 2))
        fi
    done
}

service_is_ready() {
    "${SERVICE_PYTHON}" - "http://${SERVICE_HOST}:${SERVICE_PORT}/healthz" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_service() {
    local deadline=$((SECONDS + SERVICE_START_TIMEOUT_SECONDS))
    until service_is_ready; do
        if ! kill -0 "${supervisor_pid}" 2>/dev/null; then
            echo "Scheduler Exclusion supervisor exited during startup" >&2
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Scheduler Exclusion service did not become ready" >&2
            tail -100 "${SERVICE_LOG}" >&2 || true
            return 1
        fi
        sleep 1
    done
}

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
if [[ "${ARRAY_TASK_ID}" == "${ARRAY_TASK_MIN}" ]]; then
    supervise_service &
    supervisor_pid=$!
    wait_for_service
fi

srun --kill-on-bad-exit=1 \
    ft_launcher \
    --ft-scheduler-exclusion-dir "${SCHEDULER_EXCLUSION_DIR}" \
    "$@"
