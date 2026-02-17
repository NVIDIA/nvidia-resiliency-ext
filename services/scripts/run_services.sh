#!/bin/bash
# Manage NVRX Attribution Services (background processes, no systemd)
#
# Usage:
#   ./run_services.sh <command>
#
# Commands:
#   install   Install/update packages only
#   start     Start services in background
#   stop      Stop running services
#   restart   Restart services
#   status    Check service status
#   logs      Tail log files
#   run       Run in foreground (original behavior, Ctrl+C to stop)
#
# Required environment variables:
#   NVRX_ATTRSVC_ALLOWED_ROOT - Root path for log files to analyze
#   NVIDIA_API_KEY            - API key for LLM (or NVIDIA_API_KEY_FILE)
#
# Optional environment variables:
#   NVRX_LOGS_DIR             - Output directory for logs (default: ~/nvrx_logs)
#   NVRX_SMONSVC_PARTITIONS   - SLURM partitions to monitor (default: "batch batch_long")
#   NVRX_ATTRSVC_PORT         - Attribution service port (default: 8000)
#   NVRX_SMONSVC_PORT         - Monitor service port (default: 8100)
#   NVRX_ATTRSVC_CLUSTER_NAME - Cluster name for dataflow (auto-detected from SLURM)
#   SNAPSHOT_INTERVAL         - Snapshot interval in seconds (default: 600)
#
# Example:
#   export NVRX_ATTRSVC_ALLOWED_ROOT=/lustre/logs
#   export NVIDIA_API_KEY=nvapi-...
#   ./run_services.sh install
#   ./run_services.sh start
#   ./run_services.sh status
#   ./run_services.sh stop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
LIB_ROOT="${REPO_DIR}/.."  # Project root (one level up from services/)
source "${SCRIPT_DIR}/common.sh"

# Ensure user-installed scripts (nvrx-attrsvc, nvrx-smonsvc) are on PATH (pip may install to ~/.local/bin)
export PATH="${HOME}/.local/bin:${PATH}"

# Configuration
export NVRX_LOGS_DIR="${NVRX_LOGS_DIR:-${HOME}/nvrx_logs}"
ATTRSVC_PORT="${NVRX_ATTRSVC_PORT:-8000}"
SMONSVC_PORT="${NVRX_SMONSVC_PORT:-8100}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-600}"
# Cluster name for dataflow (auto-detect from SLURM if not set)
if [[ -z "${NVRX_ATTRSVC_CLUSTER_NAME:-}" ]]; then
    # Try SLURM env var, then scontrol, then hostname
    NVRX_ATTRSVC_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-}"
    if [[ -z "${NVRX_ATTRSVC_CLUSTER_NAME}" ]]; then
        NVRX_ATTRSVC_CLUSTER_NAME=$(scontrol show config 2>/dev/null | grep -oP 'ClusterName\s*=\s*\K\S+' || echo "")
    fi
    if [[ -z "${NVRX_ATTRSVC_CLUSTER_NAME}" ]]; then
        NVRX_ATTRSVC_CLUSTER_NAME=$(hostname -s 2>/dev/null || echo "unknown")
    fi
fi
export NVRX_ATTRSVC_CLUSTER_NAME

# PID files (fixed names for background mode)
PID_DIR="${NVRX_LOGS_DIR}"
ATTRSVC_PID_FILE="${PID_DIR}/attrsvc.pid"
SMONSVC_PID_FILE="${PID_DIR}/smonsvc.pid"
SNAPSHOT_PID_FILE="${PID_DIR}/snapshot.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ─── Helper functions ───

usage() {
    head -35 "$0" | grep "^#" | sed 's/^# //'
    exit 0
}

is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# ─── Commands ───

cmd_install() {
    install_nvrx_packages "both" "${LIB_ROOT}"
    echo -e "${GREEN}Installation complete${NC}"
}

cmd_start() {
    # Validate environment
    validate_attrsvc_allowed_root || exit 1
    ensure_directory "${NVRX_LOGS_DIR}" "logs directory" || exit 1
    
    # Load API key from file if not already set
    if [[ -z "$NVIDIA_API_KEY" ]]; then
        # Check NVIDIA_API_KEY_FILE first
        if [[ -n "$NVIDIA_API_KEY_FILE" && -f "$NVIDIA_API_KEY_FILE" ]]; then
            export NVIDIA_API_KEY=$(cat "$NVIDIA_API_KEY_FILE")
            echo "Loaded API key from: $NVIDIA_API_KEY_FILE"
        # Check common locations
        elif [[ -f "${HOME}/.nvidia_api_key" ]]; then
            export NVIDIA_API_KEY=$(cat "${HOME}/.nvidia_api_key")
            echo "Loaded API key from: ~/.nvidia_api_key"
        elif [[ -f "${HOME}/.config/nvrx/nvidia_api_key" ]]; then
            export NVIDIA_API_KEY=$(cat "${HOME}/.config/nvrx/nvidia_api_key")
            echo "Loaded API key from: ~/.config/nvrx/nvidia_api_key"
        else
            echo -e "${RED}Error: NVIDIA_API_KEY not set and no key file found${NC}"
            echo "Either:"
            echo "  export NVIDIA_API_KEY=nvapi-xxx"
            echo "  export NVIDIA_API_KEY_FILE=/path/to/keyfile"
            echo "  Or create ~/.nvidia_api_key"
            exit 1
        fi
    fi
    
    # Check if already running
    if is_running "${ATTRSVC_PID_FILE}"; then
        echo -e "${YELLOW}Attribution service already running (PID: $(cat ${ATTRSVC_PID_FILE}))${NC}"
        echo "Use './run_services.sh stop' first, or './run_services.sh restart'"
        exit 1
    fi
    
    # Install/update packages
    install_nvrx_packages "both" "${LIB_ROOT}"
    echo ""
    
    # Timestamp for this run's log files
    RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    ATTRSVC_LOG="${NVRX_LOGS_DIR}/${RUN_TIMESTAMP}_attrsvc.log"
    SMONSVC_LOG="${NVRX_LOGS_DIR}/${RUN_TIMESTAMP}_smonsvc.log"
    SNAPSHOT_LOG="${NVRX_LOGS_DIR}/${RUN_TIMESTAMP}_snapshot.log"
    
    echo "=== Starting NVRX Services ==="
    echo "Logs directory:    ${NVRX_LOGS_DIR}"
    echo "Allowed root:      ${NVRX_ATTRSVC_ALLOWED_ROOT}"
    echo "Attrsvc port:      ${ATTRSVC_PORT}"
    echo "Smonsvc port:      ${SMONSVC_PORT}"
    echo ""
    
    # Start Attribution Service
    echo "Starting Attribution Service..."
    nohup nvrx-attrsvc > "${ATTRSVC_LOG}" 2>&1 &
    echo $! > "${ATTRSVC_PID_FILE}"
    echo -e "  ${GREEN}Started${NC} (PID: $(cat ${ATTRSVC_PID_FILE}), log: ${ATTRSVC_LOG})"
    
    # Wait for attrsvc to be ready
    sleep 2
    
    # Start SLURM Monitor
    echo "Starting SLURM Monitor..."
    export NVRX_ATTRSVC_URL="http://localhost:${ATTRSVC_PORT}"
    export NVRX_SMONSVC_PORT="${SMONSVC_PORT}"
    nohup nvrx-smonsvc > "${SMONSVC_LOG}" 2>&1 &
    echo $! > "${SMONSVC_PID_FILE}"
    echo -e "  ${GREEN}Started${NC} (PID: $(cat ${SMONSVC_PID_FILE}), log: ${SMONSVC_LOG})"
    
    # Start snapshot process
    echo "Starting Snapshot process..."
    export SNAPSHOT_OUTPUT_DIR="${NVRX_LOGS_DIR}"
    export SNAPSHOT_TIMESTAMP="${RUN_TIMESTAMP}"
    nohup "${SCRIPT_DIR}/snapshot_services.sh" > "${SNAPSHOT_LOG}" 2>&1 &
    echo $! > "${SNAPSHOT_PID_FILE}"
    echo -e "  ${GREEN}Started${NC} (PID: $(cat ${SNAPSHOT_PID_FILE}), log: ${SNAPSHOT_LOG})"
    
    echo ""
    echo -e "${GREEN}All services started${NC}"
    echo ""
    echo "Check status:  ./run_services.sh status"
    echo "View logs:     ./run_services.sh logs"
    echo "Stop:          ./run_services.sh stop"
}

cmd_stop() {
    echo "Stopping NVRX services..."
    local stopped=0
    
    # Stop snapshot
    if [[ -f "${SNAPSHOT_PID_FILE}" ]]; then
        local pid=$(cat "${SNAPSHOT_PID_FILE}")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "  Stopped snapshot (PID: $pid)"
            stopped=1
        fi
        rm -f "${SNAPSHOT_PID_FILE}"
    fi
    
    # Stop smonsvc
    if [[ -f "${SMONSVC_PID_FILE}" ]]; then
        local pid=$(cat "${SMONSVC_PID_FILE}")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "  Stopped smonsvc (PID: $pid)"
            stopped=1
        fi
        rm -f "${SMONSVC_PID_FILE}"
    fi
    
    # Stop attrsvc
    if [[ -f "${ATTRSVC_PID_FILE}" ]]; then
        local pid=$(cat "${ATTRSVC_PID_FILE}")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "  Stopped attrsvc (PID: $pid)"
            stopped=1
        fi
        rm -f "${ATTRSVC_PID_FILE}"
    fi
    
    if [[ $stopped -eq 0 ]]; then
        echo "  No services were running"
    else
        echo -e "${GREEN}Services stopped${NC}"
    fi
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

cmd_status() {
    echo "=== NVRX Service Status ==="
    echo ""
    
    # Attribution service
    if is_running "${ATTRSVC_PID_FILE}"; then
        local pid=$(cat "${ATTRSVC_PID_FILE}")
        echo -e "Attribution Service: ${GREEN}running${NC} (PID: $pid)"
        # Try to get health
        local health=$(curl -s "http://localhost:${ATTRSVC_PORT}/healthz" 2>/dev/null || echo "unreachable")
        echo "  Health: $health"
    else
        echo -e "Attribution Service: ${RED}stopped${NC}"
    fi
    
    # SLURM Monitor
    if is_running "${SMONSVC_PID_FILE}"; then
        local pid=$(cat "${SMONSVC_PID_FILE}")
        echo -e "SLURM Monitor:       ${GREEN}running${NC} (PID: $pid)"
        local health=$(curl -s "http://localhost:${SMONSVC_PORT}/healthz" 2>/dev/null || echo "unreachable")
        echo "  Health: $health"
    else
        echo -e "SLURM Monitor:       ${RED}stopped${NC}"
    fi
    
    # Snapshot
    if is_running "${SNAPSHOT_PID_FILE}"; then
        local pid=$(cat "${SNAPSHOT_PID_FILE}")
        echo -e "Snapshot:            ${GREEN}running${NC} (PID: $pid)"
    else
        echo -e "Snapshot:            ${RED}stopped${NC}"
    fi
    
    echo ""
    echo "Logs directory: ${NVRX_LOGS_DIR}"
}

cmd_logs() {
    echo "Tailing logs from ${NVRX_LOGS_DIR}..."
    echo "Press Ctrl+C to stop"
    echo ""
    tail -f "${NVRX_LOGS_DIR}"/*.log 2>/dev/null || echo "No log files found"
}

cmd_run() {
    # Original foreground behavior
    validate_attrsvc_allowed_root || exit 1
    ensure_directory "${NVRX_LOGS_DIR}" "logs directory" || exit 1
    
    # Load API key from file if not already set
    if [[ -z "$NVIDIA_API_KEY" ]]; then
        if [[ -n "$NVIDIA_API_KEY_FILE" && -f "$NVIDIA_API_KEY_FILE" ]]; then
            export NVIDIA_API_KEY=$(cat "$NVIDIA_API_KEY_FILE")
            echo "Loaded API key from: $NVIDIA_API_KEY_FILE"
        elif [[ -f "${HOME}/.nvidia_api_key" ]]; then
            export NVIDIA_API_KEY=$(cat "${HOME}/.nvidia_api_key")
            echo "Loaded API key from: ~/.nvidia_api_key"
        elif [[ -f "${HOME}/.config/nvrx/nvidia_api_key" ]]; then
            export NVIDIA_API_KEY=$(cat "${HOME}/.config/nvrx/nvidia_api_key")
            echo "Loaded API key from: ~/.config/nvrx/nvidia_api_key"
        else
            echo -e "${RED}Error: NVIDIA_API_KEY not set and no key file found${NC}"
            echo "Either:"
            echo "  export NVIDIA_API_KEY=nvapi-xxx"
            echo "  export NVIDIA_API_KEY_FILE=/path/to/keyfile"
            echo "  Or create ~/.nvidia_api_key"
            exit 1
        fi
    fi
    
    install_nvrx_packages "both" "${LIB_ROOT}"
    echo ""
    
    # Timestamp prefix for this run
    RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    export NVRX_RUN_PREFIX="${RUN_TIMESTAMP}"
    
    ATTRSVC_LOG="${NVRX_LOGS_DIR}/${RUN_TIMESTAMP}_attrsvc.log"
    SMONSVC_LOG="${NVRX_LOGS_DIR}/${RUN_TIMESTAMP}_smonsvc.log"
    ATTRSVC_SNAP="${NVRX_LOGS_DIR}/${RUN_TIMESTAMP}_attrsvc.snap"
    SMONSVC_SNAP="${NVRX_LOGS_DIR}/${RUN_TIMESTAMP}_smonsvc.snap"
    
    # Track PIDs for cleanup
    ATTRSVC_PID=""
    SMONSVC_PID=""
    SNAPSHOT_ATTRSVC_PID=""
    SNAPSHOT_SMONSVC_PID=""
    
    cleanup() {
        echo ""
        echo "Shutting down services..."
        [[ -n "$SNAPSHOT_ATTRSVC_PID" ]] && kill "$SNAPSHOT_ATTRSVC_PID" 2>/dev/null || true
        [[ -n "$SNAPSHOT_SMONSVC_PID" ]] && kill "$SNAPSHOT_SMONSVC_PID" 2>/dev/null || true
        [[ -n "$SMONSVC_PID" ]] && kill "$SMONSVC_PID" 2>/dev/null || true
        [[ -n "$ATTRSVC_PID" ]] && kill "$ATTRSVC_PID" 2>/dev/null || true
        wait 2>/dev/null || true
        echo "Shutdown complete."
    }
    trap cleanup EXIT SIGTERM SIGINT
    
    echo "=== NVRX Attribution Services (Foreground) ==="
    echo "Logs directory:    ${NVRX_LOGS_DIR}"
    echo "Allowed root:      ${NVRX_ATTRSVC_ALLOWED_ROOT}"
    echo "Attrsvc port:      ${ATTRSVC_PORT}"
    echo "Smonsvc port:      ${SMONSVC_PORT}"
    echo ""
    
    # Start Attribution Service
    echo "Starting Attribution Service..."
    nvrx-attrsvc > "${ATTRSVC_LOG}" 2>&1 &
    ATTRSVC_PID=$!
    echo "  PID: ${ATTRSVC_PID}, Log: ${ATTRSVC_LOG}"
    
    sleep 2
    
    # Start SLURM Monitor
    echo "Starting SLURM Monitor..."
    export NVRX_ATTRSVC_URL="http://localhost:${ATTRSVC_PORT}"
    export NVRX_SMONSVC_PORT="${SMONSVC_PORT}"
    nvrx-smonsvc > "${SMONSVC_LOG}" 2>&1 &
    SMONSVC_PID=$!
    echo "  PID: ${SMONSVC_PID}, Log: ${SMONSVC_LOG}"
    
    # Start snapshot processes
    echo "Starting periodic snapshots (every ${SNAPSHOT_INTERVAL}s)..."
    (
        while true; do
            sleep "${SNAPSHOT_INTERVAL}"
            ts=$(date +"%Y-%m-%d %H:%M:%S")
            echo "=== ${ts} ===" >> "${ATTRSVC_SNAP}"
            curl -s "http://localhost:${ATTRSVC_PORT}/stats?pretty=true" >> "${ATTRSVC_SNAP}" 2>/dev/null || echo "FAILED" >> "${ATTRSVC_SNAP}"
            echo "" >> "${ATTRSVC_SNAP}"
        done
    ) &
    SNAPSHOT_ATTRSVC_PID=$!
    
    (
        while true; do
            sleep "${SNAPSHOT_INTERVAL}"
            ts=$(date +"%Y-%m-%d %H:%M:%S")
            echo "=== ${ts} ===" >> "${SMONSVC_SNAP}"
            curl -s "http://localhost:${SMONSVC_PORT}/stats?pretty=true" >> "${SMONSVC_SNAP}" 2>/dev/null || echo "FAILED" >> "${SMONSVC_SNAP}"
            echo "" >> "${SMONSVC_SNAP}"
        done
    ) &
    SNAPSHOT_SMONSVC_PID=$!
    
    echo ""
    echo "=== Running (Ctrl+C to stop) ==="
    echo ""
    echo "Logs:     tail -f ${ATTRSVC_LOG} ${SMONSVC_LOG}"
    echo "Status:   curl http://localhost:${ATTRSVC_PORT}/stats"
    echo ""
    
    # Wait for services
    while true; do
        if ! kill -0 "$ATTRSVC_PID" 2>/dev/null; then
            echo "ERROR: Attribution service died"
            exit 1
        fi
        if ! kill -0 "$SMONSVC_PID" 2>/dev/null; then
            echo "ERROR: SLURM Monitor died"
            exit 1
        fi
        sleep 30
    done
}

# ─── Command dispatcher ───

COMMAND="${1:-}"

case "$COMMAND" in
    install)
        cmd_install
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs
        ;;
    run)
        cmd_run
        ;;
    -h|--help|help)
        usage
        ;;
    "")
        echo "Usage: $0 <command>"
        echo "Commands: install, start, stop, restart, status, logs, run"
        echo "Run '$0 --help' for details"
        exit 1
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run '$0 --help' for available commands"
        exit 1
        ;;
esac
