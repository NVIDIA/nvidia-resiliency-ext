#!/bin/bash
# Periodically snapshot both nvrx-attrsvc and nvrx-smonsvc endpoints
#
# Usage:
#   ./snapshot_services.sh [host]
#
# Environment variables:
#   NVRX_HOST - Common host for both services (default: localhost)
#   NVRX_ATTRSVC_PORT - Attribution service port (default: 8000)
#   NVRX_SMONSVC_PORT - Monitor service port (default: 8100)
#   SNAPSHOT_INTERVAL - Interval in seconds (default: 600)
#   SNAPSHOT_OUTPUT_DIR - Output directory (default: ~/nvrx_snapshots)
#
# Output:
#   - <timestamp>_snapshot_attrsvc.log: Attribution service snapshots
#   - <timestamp>_snapshot_smonsvc.log: Monitor service snapshots

set -e

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Configuration
HOST="${1:-${NVRX_HOST:-localhost}}"
ATTRSVC_PORT="${NVRX_ATTRSVC_PORT:-8000}"
SMONSVC_PORT="${NVRX_SMONSVC_PORT:-8100}"
INTERVAL="${SNAPSHOT_INTERVAL:-600}"
OUTPUT_DIR="${SNAPSHOT_OUTPUT_DIR:-${HOME}/nvrx_snapshots}"

# Use timestamp for log files (consistent with run_services.sh)
SNAPSHOT_TIMESTAMP="${SNAPSHOT_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ATTRSVC_LOG="${OUTPUT_DIR}/${SNAPSHOT_TIMESTAMP}_snapshot_attrsvc.log"
SMONSVC_LOG="${OUTPUT_DIR}/${SNAPSHOT_TIMESTAMP}_snapshot_smonsvc.log"

mkdir -p "${OUTPUT_DIR}"

echo "=== NVRX Services Snapshot ==="
echo "attrsvc: http://${HOST}:${ATTRSVC_PORT} -> ${ATTRSVC_LOG}"
echo "smonsvc: http://${HOST}:${SMONSVC_PORT} -> ${SMONSVC_LOG}"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

count=0

while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    count=$((count + 1))
    
    echo "[${timestamp}] Snapshot #${count}"
    snapshot_attrsvc "${HOST}" "${ATTRSVC_PORT}" "${ATTRSVC_LOG}" "${count}" "${timestamp}"
    snapshot_smonsvc "${HOST}" "${SMONSVC_PORT}" "${SMONSVC_LOG}" "${count}" "${timestamp}"
    
    sleep "${INTERVAL}"
done
