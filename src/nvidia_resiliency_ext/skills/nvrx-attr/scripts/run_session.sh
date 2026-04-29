#!/bin/bash
# run_session.sh
# End-to-end fault-injection session: submit all experiments from the pool
# (2 at a time, waiting for each pair), then analyze every completed job and
# produce a scored report.  Designed to be run unattended via nohup.
#
# Usage:
#   nohup bash scripts/run_session.sh > /path/to/session.log 2>&1 &
#   EXPERIMENT_MATRIX="GPU_SLEEP:1:5:2 SIGKILL:1:5:4" nohup bash scripts/run_session.sh ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_ENV_FILE="${SCRIPT_DIR}/user.env"
if [[ ! -f "${USER_ENV_FILE}" ]]; then
    echo "ERROR: required local config not found: ${USER_ENV_FILE}" >&2
    echo "Create it from ${SCRIPT_DIR}/user.env.example and fill in your local settings." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${USER_ENV_FILE}"
WORKLOAD="${WORKLOAD:-llama4_scout}"

# ---- Phase 1: submit and wait for all experiments ----
echo "========================================"
echo "PHASE 1: Fault injection"
echo "========================================"
WORKLOAD="${WORKLOAD}" bash "${SCRIPT_DIR}/prepare_node_alloc.sh"

# prepare_node_alloc.sh prints the tracking file path; re-derive it the same way
# (SESSION_TAG is the timestamp when prepare_node_alloc ran, which is a few seconds
# before this line — find the newest session dir instead of recomputing the tag)
BASE_EXPERIMENTS_DIR="${BASE_EXPERIMENTS_DIR:-${HOME}/nvrx-attr-experiments}"
TRACKING_FILE=$(ls -td "${BASE_EXPERIMENTS_DIR}/fault_injection"/[0-9]* 2>/dev/null \
    | head -1)/experiments.tsv

if [[ ! -f "${TRACKING_FILE}" ]]; then
    echo "ERROR: could not locate experiments.tsv in latest session dir" >&2
    exit 1
fi

echo ""
echo "========================================"
echo "PHASE 2: Analysis"
echo "Tracking: ${TRACKING_FILE}"
echo "========================================"
bash "${SCRIPT_DIR}/watch_and_analyze.sh" "${TRACKING_FILE}"
