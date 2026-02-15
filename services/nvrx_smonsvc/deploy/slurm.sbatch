#!/bin/bash
#SBATCH --job-name=nvrx-smonsvc
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=nvrx-smonsvc-%j.out
#SBATCH --error=nvrx-smonsvc-%j.err
#
# Run the SLURM Job Monitor Service on a compute node
#
# Submit with (account is required):
#   sbatch --account=myaccount services/nvrx_smonsvc/deploy/slurm.sbatch
#
# Required:
#   Attribution service must be running and accessible
#
# Configuration via environment variables:
#   NVRX_ATTRSVC_URL - Attribution service URL (default: http://localhost:8000)
#   NVRX_SMONSVC_PARTITIONS - SLURM partitions to monitor (default: "batch batch_long")
#   NVRX_SMONSVC_PORT - HTTP status server port (default: 8100)
#   NVRX_SMONSVC_INTERVAL - Poll interval in seconds (default: 180)
#
# Example:
#   NVRX_ATTRSVC_URL=http://cpu-0011:8000 \
#   NVRX_SMONSVC_PARTITIONS="gpu gpu_long" \
#   sbatch --account=myaccount slurm.sbatch
#
# The service will be accessible at:
#   http://<hostname>:8100
#
# Check which node it's running on:
#   squeue -j <jobid> -o "%N"

set -e

# Source common setup functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Configuration (can be overridden via env vars)
export NVRX_ATTRSVC_URL="${NVRX_ATTRSVC_URL:-http://localhost:8000}"
export NVRX_SMONSVC_PORT="${NVRX_SMONSVC_PORT:-8100}"
export NVRX_SMONSVC_PARTITIONS="${NVRX_SMONSVC_PARTITIONS:-batch batch_long}"
export NVRX_SMONSVC_INTERVAL="${NVRX_SMONSVC_INTERVAL:-180}"
export NVRX_SMONSVC_VERBOSE="${NVRX_SMONSVC_VERBOSE:-false}"

# Install packages
install_nvrx_packages "smonsvc"

# Validate command exists
validate_commands "nvrx-smonsvc" "curl" || exit 1

HOSTNAME=$(hostname)

echo "============================================================"
echo "SLURM Monitor Service - SLURM Job ${SLURM_JOB_ID}"
echo "============================================================"
echo "  Hostname:    ${HOSTNAME}"
echo "  Port:        ${NVRX_SMONSVC_PORT}"
echo "  URL:         http://${HOSTNAME}:${NVRX_SMONSVC_PORT}"
echo "  Attrsvc URL: ${NVRX_ATTRSVC_URL}"
echo "  Partitions:  ${NVRX_SMONSVC_PARTITIONS}"
echo "  Interval:    ${NVRX_SMONSVC_INTERVAL}s"
echo "============================================================"

# Check if attribution service is reachable (warning only)
check_attrsvc_reachable "${NVRX_ATTRSVC_URL}" || true

# Cleanup on exit
cleanup() {
    echo "Shutting down..."
}
trap cleanup EXIT SIGTERM SIGINT SIGUSR1 SIGUSR2

# Run service (foreground)
nvrx-smonsvc
