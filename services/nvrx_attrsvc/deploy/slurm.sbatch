#!/bin/bash
#SBATCH --job-name=nvrx-attrsvc
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=nvrx-attrsvc-%j.out
#SBATCH --error=nvrx-attrsvc-%j.err
#
# Run the Attribution Service on a compute node
#
# Submit with (account is required):
#   sbatch --account=myaccount services/nvrx_attrsvc/deploy/slurm.sbatch
#
# Required environment variables:
#   NVRX_ATTRSVC_ALLOWED_ROOT - Root path for log files to analyze
#
# API Key Options (in priority order):
#   1. NVIDIA_API_KEY env var (direct key)
#   2. NVIDIA_API_KEY_FILE env var (path to file containing key)
#   3. Default: ~/.nvidia_api_key
#
# Example:
#   NVRX_ATTRSVC_ALLOWED_ROOT=/lustre/logs sbatch --account=myaccount slurm.sbatch
#
# The service will be accessible at:
#   http://<hostname>:8000
#
# Check which node it's running on:
#   squeue -j <jobid> -o "%N"

set -e

# Source common setup functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Configuration (can be overridden via env vars)
export NVRX_ATTRSVC_PORT="${NVRX_ATTRSVC_PORT:-8000}"

# Validate required environment
validate_attrsvc_allowed_root || exit 1

# nvdataflow configuration
export NVRX_ATTRSVC_NVDATAFLOW_PROJECT="${NVRX_ATTRSVC_NVDATAFLOW_PROJECT:-}"
export NVRX_ATTRSVC_CLUSTER_NAME="${NVRX_ATTRSVC_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-unknown}}"

# Setup API key
setup_nvidia_api_key || exit 1

# Install packages
install_nvrx_packages "attrsvc"

# Validate command exists
validate_commands "nvrx-attrsvc" || exit 1

HOSTNAME=$(hostname)

echo "============================================================"
echo "Attribution Service - SLURM Job ${SLURM_JOB_ID}"
echo "============================================================"
echo "  Hostname: ${HOSTNAME}"
echo "  Port:     ${NVRX_ATTRSVC_PORT}"
echo "  URL:      http://${HOSTNAME}:${NVRX_ATTRSVC_PORT}"
echo "============================================================"

# Cleanup on exit
cleanup() {
    echo "Shutting down..."
}
trap cleanup EXIT SIGTERM SIGINT SIGUSR1 SIGUSR2

# Run service (foreground)
nvrx-attrsvc
