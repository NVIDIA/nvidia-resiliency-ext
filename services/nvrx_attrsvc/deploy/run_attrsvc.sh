#!/bin/bash
# Run the Attribution Service with logging
#
# Usage:
#   ./run_attrsvc.sh [output_dir]
#
# Required environment variables:
#   NVRX_ATTRSVC_ALLOWED_ROOT - Root path for log files to analyze
#   NVIDIA_API_KEY            - API key for LLM (or NVIDIA_API_KEY_FILE)
#
# Optional environment variables:
#   NVRX_ATTRSVC_PORT         - Listen port (default: 8000)
#
# Output files (in output_dir):
#   attrsvc.log   - Service stdout/stderr
#   attrsvc.pid   - Process ID
#
# Example:
#   export NVRX_ATTRSVC_ALLOWED_ROOT=/lustre/logs
#   export NVIDIA_API_KEY=nvapi-...
#   ./run_attrsvc.sh ~/nvrx_logs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Configuration
OUTPUT_DIR="${1:-${NVRX_LOGS_DIR:-${HOME}/nvrx_logs}}"
PORT="${NVRX_ATTRSVC_PORT:-8000}"

# Use timestamp prefix if set (from run_services.sh), otherwise generate one
PREFIX="${NVRX_RUN_PREFIX:-$(date +"%Y%m%d_%H%M%S")}"
LOG_FILE="${OUTPUT_DIR}/${PREFIX}_attrsvc.log"
PID_FILE="${OUTPUT_DIR}/${PREFIX}_attrsvc.pid"

# Validate required environment
validate_attrsvc_allowed_root || exit 1

# Setup API key
setup_nvidia_api_key || exit 1

# Create output directory
ensure_directory "${OUTPUT_DIR}" "logs directory" || exit 1

# Start service
echo "Starting Attribution Service..."
echo "  Port: ${PORT}"
echo "  Log:  ${LOG_FILE}"

export NVRX_ATTRSVC_PORT="${PORT}"
nvrx-attrsvc >> "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"
echo "  PID:  ${PID}"

# Wait for service to be ready
wait_for_background_service "Attribution Service" "$PID" "$PORT" "$LOG_FILE" "$PID_FILE" || exit 1
