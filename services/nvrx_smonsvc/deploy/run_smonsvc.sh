#!/bin/bash
# Run the SLURM Monitor Service with logging
#
# Usage:
#   ./run_smonsvc.sh [output_dir]
#
# Required:
#   Attribution service must be running and accessible
#
# Optional environment variables:
#   NVRX_ATTRSVC_URL        - Attribution service URL (default: http://localhost:8000)
#   NVRX_SMONSVC_PORT       - Listen port (default: 8100)
#   NVRX_SMONSVC_PARTITIONS - SLURM partitions (default: "batch batch_long")
#
# Output files (in output_dir):
#   smonsvc.log   - Service stdout/stderr
#   smonsvc.pid   - Process ID
#
# Example:
#   export NVRX_SMONSVC_PARTITIONS="gpu gpu_long"
#   ./run_smonsvc.sh ~/nvrx_logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Configuration
OUTPUT_DIR="${1:-${NVRX_LOGS_DIR:-${HOME}/nvrx_logs}}"
ATTRSVC_URL="${NVRX_ATTRSVC_URL:-http://localhost:8000}"
PORT="${NVRX_SMONSVC_PORT:-8100}"

# Use timestamp prefix if set (from run_services.sh), otherwise generate one
PREFIX="${NVRX_RUN_PREFIX:-$(date +"%Y%m%d_%H%M%S")}"
LOG_FILE="${OUTPUT_DIR}/${PREFIX}_smonsvc.log"
PID_FILE="${OUTPUT_DIR}/${PREFIX}_smonsvc.pid"

# Create output directory
ensure_directory "${OUTPUT_DIR}" "logs directory" || exit 1

# Check if attrsvc is reachable (warning only, doesn't exit)
check_attrsvc_reachable "${ATTRSVC_URL}" || true

# Start service
echo "Starting SLURM Monitor..."
echo "  Port:     ${PORT}"
echo "  Attrsvc:  ${ATTRSVC_URL}"
echo "  Log:      ${LOG_FILE}"

export NVRX_ATTRSVC_URL="${ATTRSVC_URL}"
export NVRX_SMONSVC_PORT="${PORT}"
nvrx-smonsvc >> "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"
echo "  PID:      ${PID}"

# Wait for service to be ready
wait_for_background_service "SLURM Monitor" "$PID" "$PORT" "$LOG_FILE" "$PID_FILE" || exit 1
