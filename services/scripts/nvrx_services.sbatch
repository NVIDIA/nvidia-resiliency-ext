#!/bin/bash
#SBATCH --job-name=nvrx-services
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=nvrx-services-%j.out
#SBATCH --error=nvrx-services-%j.err
#
# Run both NVRX Attribution Service and SLURM Job Monitor together
#
# This is the recommended deployment for most use cases. For running
# services separately, see:
#   - nvrx_attrsvc/deploy/slurm.sbatch (attribution only)
#   - nvrx_smonsvc/deploy/slurm.sbatch (monitor only, requires external attrsvc)
#
# Submit with (account is required):
#   sbatch --account=myaccount nvrx_services.sbatch
#
# Required environment variables:
#   NVRX_ATTRSVC_ALLOWED_ROOT - Root path for log files to analyze
#
# Optional environment variables:
#   NVRX_LOGS_DIR - Directory for service logs (default: current directory)
#
# API Key Options (in priority order):
#   1. NVIDIA_API_KEY env var (direct key)
#   2. NVIDIA_API_KEY_FILE env var (path to file containing key)
#   3. Default: ~/.nvidia_api_key
#
# Example with custom output directory:
#   export NVRX_LOGS_DIR=/my/logs
#   NVRX_ATTRSVC_ALLOWED_ROOT=/lustre/path \
#     sbatch --account=myaccount \
#     --time=7-00:00:00 \
#     -o ${NVRX_LOGS_DIR}/nvrx-%j.out \
#     -e ${NVRX_LOGS_DIR}/nvrx-%j.err \
#     nvrx_services.sbatch
#
# The services will be accessible at:
#   http://<hostname>:8000  (Attribution Service)
#   http://<hostname>:8100  (Slurm Job Monitor Service)
#
# Check which node it's running on:
#   squeue -j <jobid> -o "%N"

set -e

# Source common setup functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Output directory for service logs (can be overridden via env var)
export NVRX_LOGS_DIR="${NVRX_LOGS_DIR:-.}"
ensure_directory "${NVRX_LOGS_DIR}" "logs directory" || exit 1
echo "Logs directory: ${NVRX_LOGS_DIR}"

# Configuration - Attribution Service (can be overridden via env vars)
export NVRX_ATTRSVC_PORT="${NVRX_ATTRSVC_PORT:-8000}"
export NVRX_ATTRSVC_URL="http://localhost:${NVRX_ATTRSVC_PORT}"

# Validate required environment
validate_attrsvc_allowed_root || exit 1

# nvdataflow configuration
export NVRX_ATTRSVC_NVDATAFLOW_PROJECT="${NVRX_ATTRSVC_NVDATAFLOW_PROJECT:-}"
export NVRX_ATTRSVC_CLUSTER_NAME="${NVRX_ATTRSVC_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-unknown}}"

# Setup API key
setup_nvidia_api_key || exit 1

# Configuration - SLURM Monitor Service (can be overridden via env vars)
export NVRX_SMONSVC_PORT="${NVRX_SMONSVC_PORT:-8100}"
export NVRX_SMONSVC_PARTITIONS="${NVRX_SMONSVC_PARTITIONS:-batch batch_long}"
export NVRX_SMONSVC_INTERVAL="${NVRX_SMONSVC_INTERVAL:-180}"
export NVRX_SMONSVC_VERBOSE="${NVRX_SMONSVC_VERBOSE:-false}"

# Install packages (both services)
install_nvrx_packages "both"

# Validate required commands exist
validate_commands "nvrx-attrsvc" "nvrx-smonsvc" "curl" "hostname" || exit 1

# Get hostname for status messages
HOSTNAME=$(hostname)

echo "============================================================"
echo "Attribution Services - SLURM Job ${SLURM_JOB_ID}"
echo "============================================================"
echo "  Hostname:         ${HOSTNAME}"
echo "  Attribution port: ${NVRX_ATTRSVC_PORT}"
echo "  Monitor port:     ${NVRX_SMONSVC_PORT}"
echo "  Partitions:       ${NVRX_SMONSVC_PARTITIONS}"
echo ""
echo "  Attribution URL:  http://${HOSTNAME}:${NVRX_ATTRSVC_PORT}"
echo "  Monitor URL:      http://${HOSTNAME}:${NVRX_SMONSVC_PORT}"
echo "============================================================"

# Initialize PIDs (in case cleanup runs before services start)
ATTRSVC_PID=""
MONITOR_PID=""

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "Shutting down services..."
    [[ -n "$ATTRSVC_PID" ]] && kill "$ATTRSVC_PID" 2>/dev/null || true
    [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true
    wait 2>/dev/null
    echo "Services stopped."
}
# Trap signals: EXIT (normal), SIGTERM (scancel/timeout), SIGINT (Ctrl+C), SIGUSR1/2 (SLURM preemption)
trap cleanup EXIT SIGTERM SIGINT SIGUSR1 SIGUSR2

# Start Attribution Service
echo ""
echo "Starting Attribution Service..."
nvrx-attrsvc &
ATTRSVC_PID=$!
echo "  PID: ${ATTRSVC_PID}"

# Wait for service to be ready
wait_for_service "Attribution service" "$ATTRSVC_PID" "$NVRX_ATTRSVC_PORT" || exit 1

# Start SLURM Job Monitor
echo ""
echo "Starting SLURM Job Monitor..."
nvrx-smonsvc &
MONITOR_PID=$!
echo "  PID: ${MONITOR_PID}"

# Wait for service to be ready
wait_for_service "SLURM Monitor" "$MONITOR_PID" "$NVRX_SMONSVC_PORT" || exit 1

echo ""
echo "============================================================"
echo "Services running!"
echo "============================================================"
echo ""
echo "Check status:"
echo "  curl http://${HOSTNAME}:${NVRX_ATTRSVC_PORT}/stats"
echo "  curl http://${HOSTNAME}:${NVRX_SMONSVC_PORT}/stats"
echo ""
echo "Stop with: scancel ${SLURM_JOB_ID}"
echo ""

# Monitor services - exit if either dies or stops responding
while true; do
    # Check if processes are alive
    if ! kill -0 "$ATTRSVC_PID" 2>/dev/null; then
        echo "ERROR: Attribution service process died"
        exit 1
    fi
    if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "ERROR: SLURM Job Monitor process died"
        exit 1
    fi
    # Check if services are responding (lightweight health check)
    if ! curl -sf "http://localhost:${NVRX_ATTRSVC_PORT}/healthz" >/dev/null 2>&1; then
        echo "WARNING: Attribution service not responding on port ${NVRX_ATTRSVC_PORT}"
    fi
    if ! curl -sf "http://localhost:${NVRX_SMONSVC_PORT}/healthz" >/dev/null 2>&1; then
        echo "WARNING: SLURM Job Monitor not responding on port ${NVRX_SMONSVC_PORT}"
    fi
    sleep 60
done
