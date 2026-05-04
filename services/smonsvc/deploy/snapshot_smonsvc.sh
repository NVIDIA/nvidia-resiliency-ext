#!/bin/bash
# Periodically snapshot nvrx-smonsvc endpoints
#
# Usage:
#   ./snapshot_smonsvc.sh [host] [port]
#
# Environment variables:
#   NVRX_SMONSVC_HOST - Service host (default: localhost)
#   NVRX_SMONSVC_PORT - Service port (default: 8100)
#   SNAPSHOT_INTERVAL - Interval in seconds (default: 600)
#   SNAPSHOT_OUTPUT_DIR - Output directory (default: ~/nvrx_snapshots)
#
# Output:
#   - smonsvc.log: Snapshots of /healthz, /stats

set -e

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Configuration
HOST="${1:-${NVRX_SMONSVC_HOST:-localhost}}"
PORT="${2:-${NVRX_SMONSVC_PORT:-8100}}"
INTERVAL="${SNAPSHOT_INTERVAL:-600}"
OUTPUT_DIR="${SNAPSHOT_OUTPUT_DIR:-${HOME}/nvrx_snapshots}"
OUTPUT_FILE="${OUTPUT_DIR}/smonsvc.log"

mkdir -p "${OUTPUT_DIR}"

echo "=== NVRX SLURM Monitor Snapshot ==="
echo "Target:   http://${HOST}:${PORT}"
echo "Interval: ${INTERVAL}s"
echo "Output:   ${OUTPUT_FILE}"
echo "Press Ctrl+C to stop"
echo ""

count=0

while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    count=$((count + 1))
    
    echo "[${timestamp}] Snapshot #${count}"
    snapshot_smonsvc "${HOST}" "${PORT}" "${OUTPUT_FILE}" "${count}" "${timestamp}"
    
    sleep "${INTERVAL}"
done
