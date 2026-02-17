#!/bin/bash
# Periodically snapshot nvrx-attrsvc endpoints
#
# Usage:
#   ./snapshot_attrsvc.sh [host] [port]
#
# Environment variables:
#   NVRX_ATTRSVC_HOST - Service host (default: localhost)
#   NVRX_ATTRSVC_PORT - Service port (default: 8000)
#   SNAPSHOT_INTERVAL - Interval in seconds (default: 600)
#   SNAPSHOT_OUTPUT_DIR - Output directory (default: ~/nvrx_snapshots)
#
# Output:
#   - attrsvc.log: Snapshots of /healthz, /stats, /jobs

set -e

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Configuration
HOST="${1:-${NVRX_ATTRSVC_HOST:-localhost}}"
PORT="${2:-${NVRX_ATTRSVC_PORT:-8000}}"
INTERVAL="${SNAPSHOT_INTERVAL:-600}"
OUTPUT_DIR="${SNAPSHOT_OUTPUT_DIR:-${HOME}/nvrx_snapshots}"
OUTPUT_FILE="${OUTPUT_DIR}/attrsvc.log"

mkdir -p "${OUTPUT_DIR}"

echo "=== NVRX Attribution Service Snapshot ==="
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
    snapshot_attrsvc "${HOST}" "${PORT}" "${OUTPUT_FILE}" "${count}" "${timestamp}"
    
    sleep "${INTERVAL}"
done
