#!/bin/bash
# Manage NVRX Attribution Services (systemd deployment)
#
# Usage:
#   sudo ./setup_systemd.sh <command>    # System-wide (as root)
#   ./setup_systemd.sh <command>         # User-local (as regular user)
#
# Commands:
#   install   Install packages, create config, install service files
#   start     Start both services
#   stop      Stop both services
#   restart   Restart both services
#   status    Show service status
#   logs      Tail log files
#   snapshot  Run periodic endpoint snapshots
#
# Mode is auto-detected:
#   - Running as root → system-wide (/opt/nvrx, /etc/nvrx, /var/log/nvrx)
#   - Running as user → user-local (~/.local/share/nvrx, ~/.config/nvrx)
#
# Requirements:
#   - Python 3.11+ (for install)
#   - Internet access (for install)

set -e

# Auto-detect mode based on effective user ID
if [[ $EUID -eq 0 ]]; then
    USER_MODE=false
else
    USER_MODE=true
    # Ensure XDG_RUNTIME_DIR is set for systemctl --user (required in SSH sessions)
    if [[ -z "$XDG_RUNTIME_DIR" ]]; then
        export XDG_RUNTIME_DIR="/run/user/$(id -u)"
    fi
fi

# Configuration based on mode
PYTHON_CMD="${PYTHON_CMD:-python3}"
SERVICES="nvrx-attrsvc nvrx-smonsvc"

if [[ "$USER_MODE" == true ]]; then
    # User-local paths
    INSTALL_DIR="${HOME}/.local/share/nvrx"
    CONFIG_DIR="${HOME}/.config/nvrx"
    LOG_DIR="${HOME}/.local/share/nvrx/logs"
    SNAPSHOT_PID_FILE="${HOME}/.local/share/nvrx/snapshot.pid"
    SYSTEMD_DIR="${HOME}/.config/systemd/user"
    SYSTEMCTL_CMD="systemctl --user"
    NVRX_USER="$(whoami)"
    NVRX_GROUP="$(id -gn)"
else
    # System-wide paths (requires root)
    INSTALL_DIR="/opt/nvrx"
    CONFIG_DIR="/etc/nvrx"
    LOG_DIR="/var/log/nvrx"
    SNAPSHOT_PID_FILE="/var/run/nvrx-snapshot.pid"
    SYSTEMD_DIR="/etc/systemd/system"
    SYSTEMCTL_CMD="systemctl"
    # Default to the user who invoked sudo, or current user if not using sudo
    if [[ -n "$SUDO_USER" ]]; then
        NVRX_USER="$SUDO_USER"
        NVRX_GROUP="$(id -gn "$SUDO_USER")"
    else
        NVRX_USER="$(whoami)"
        NVRX_GROUP="$(id -gn)"
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
ATTRSVC_DIR="${REPO_DIR}/nvrx_attrsvc"
SMONSVC_DIR="${REPO_DIR}/nvrx_smonsvc"

# ─── Helper functions ───

usage() {
    head -25 "$0" | grep "^#" | sed 's/^# //'
    exit 0
}

require_root() {
    if [[ "$USER_MODE" == true ]]; then
        return 0  # No root needed for user mode
    fi
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}Error: This command must be run as root (use sudo)${NC}"
        echo "Or use --user flag for user-local installation"
        exit 1
    fi
}

cmd_start() {
    require_root
    
    # Enable lingering for user mode (keeps services running after logout)
    if [[ "$USER_MODE" == true ]]; then
        echo "Enabling lingering for $NVRX_USER..."
        loginctl enable-linger "$NVRX_USER"
    fi
    
    echo "Starting NVRX services..."
    $SYSTEMCTL_CMD start $SERVICES
    
    # Start snapshot in background
    echo "Starting snapshot process..."
    export SNAPSHOT_OUTPUT_DIR="${LOG_DIR}"
    nohup "${SCRIPT_DIR}/snapshot_services.sh" > /dev/null 2>&1 &
    echo $! > "${SNAPSHOT_PID_FILE}"
    echo "Snapshot PID: $(cat ${SNAPSHOT_PID_FILE})"
    
    $SYSTEMCTL_CMD status $SERVICES --no-pager
}

cmd_stop() {
    require_root
    echo "Stopping NVRX services..."
    $SYSTEMCTL_CMD stop $SERVICES
    
    # Stop snapshot process
    if [[ -f "${SNAPSHOT_PID_FILE}" ]]; then
        local pid=$(cat "${SNAPSHOT_PID_FILE}")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping snapshot process (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "${SNAPSHOT_PID_FILE}"
    fi
    
    # Disable lingering for user mode
    if [[ "$USER_MODE" == true ]]; then
        echo "Disabling lingering for $NVRX_USER..."
        loginctl disable-linger "$NVRX_USER"
    fi
    
    echo -e "${GREEN}Services stopped${NC}"
}

cmd_restart() {
    require_root
    cmd_stop
    cmd_start
}

cmd_status() {
    $SYSTEMCTL_CMD status $SERVICES --no-pager || true
    
    # Show snapshot status
    echo ""
    if [[ -f "${SNAPSHOT_PID_FILE}" ]]; then
        local pid=$(cat "${SNAPSHOT_PID_FILE}")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}Snapshot: running (PID: $pid)${NC}"
        else
            echo -e "${YELLOW}Snapshot: not running (stale PID file)${NC}"
        fi
    else
        echo "Snapshot: not running"
    fi
}

cmd_logs() {
    echo "Tailing logs from ${LOG_DIR}..."
    echo "Press Ctrl+C to stop"
    echo ""
    tail -f "${LOG_DIR}"/*.log
}

cmd_snapshot() {
    echo "Starting periodic snapshots..."
    export SNAPSHOT_OUTPUT_DIR="${LOG_DIR}"
    exec "${SCRIPT_DIR}/snapshot_services.sh"
}

cmd_install() {
    require_root

    # Check Python version
    echo "=== Checking Python ==="
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo -e "${RED}Error: Python not found. Set PYTHON_CMD environment variable.${NC}"
        exit 1
    fi

    PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Python: $PYTHON_CMD (version $PYTHON_VERSION)"

    # Check for required files
    echo ""
    echo "=== Checking source files ==="
    if [[ ! -f "${REPO_DIR}/pyproject.toml" ]]; then
        echo -e "${RED}Error: pyproject.toml not found at ${REPO_DIR}${NC}"
        echo "Run this script from services/scripts/"
        exit 1
    fi
    echo "Source directory: ${REPO_DIR}"

    # Verify user exists (only for system mode)
    echo ""
    echo "=== Service user ==="
    if [[ "$USER_MODE" == false ]]; then
        if ! getent passwd "$NVRX_USER" > /dev/null 2>&1; then
            echo -e "${RED}Error: User '$NVRX_USER' does not exist${NC}"
            exit 1
        fi
    fi
    echo "Services will run as: $NVRX_USER (group: $NVRX_GROUP)"

    # Create directories
    echo ""
    echo "=== Creating directories ==="

    mkdir -p "$INSTALL_DIR"
    if [[ "$USER_MODE" == false ]]; then
        chown "$NVRX_USER:$NVRX_GROUP" "$INSTALL_DIR"
    fi
    chmod 755 "$INSTALL_DIR"
    echo "Created: $INSTALL_DIR"

    mkdir -p "$CONFIG_DIR"
    if [[ "$USER_MODE" == false ]]; then
        chmod 750 "$CONFIG_DIR"
        chown root:"$NVRX_GROUP" "$CONFIG_DIR"
    else
        chmod 700 "$CONFIG_DIR"
    fi
    echo "Created: $CONFIG_DIR"

    mkdir -p "$LOG_DIR"
    if [[ "$USER_MODE" == false ]]; then
        chown "$NVRX_USER:$NVRX_GROUP" "$LOG_DIR"
    fi
    chmod 755 "$LOG_DIR"
    echo "Created: $LOG_DIR"

    # Create systemd user directory (for user mode)
    if [[ "$USER_MODE" == true ]]; then
        mkdir -p "$SYSTEMD_DIR"
        echo "Created: $SYSTEMD_DIR"
    fi

    # Create virtual environment
    echo ""
    echo "=== Creating virtual environment ==="
    VENV_DIR="${INSTALL_DIR}/venv"

    if [[ -d "$VENV_DIR" ]]; then
        echo -e "${YELLOW}Virtual environment already exists, skipping creation${NC}"
    else
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        if [[ "$USER_MODE" == false ]]; then
            chown -R "$NVRX_USER:$NVRX_GROUP" "$VENV_DIR"
        fi
        echo "Created: $VENV_DIR"
    fi

    # Upgrade pip and install packages using common.sh
    echo ""
    echo "=== Installing packages ==="
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet

    # Source common.sh and use venv's pip
    source "${SCRIPT_DIR}/common.sh"
    export PATH="${VENV_DIR}/bin:${PATH}"
    LIB_ROOT="${REPO_DIR}/.."  # Project root (one level up from services/)
    install_nvrx_packages "both" "${LIB_ROOT}"

    # Verify installation
    echo ""
    echo "=== Verifying installation ==="
    if [[ -f "${VENV_DIR}/bin/nvrx-attrsvc" ]]; then
        echo -e "${GREEN}✓ nvrx-attrsvc installed${NC}"
    else
        echo -e "${RED}✗ nvrx-attrsvc not found${NC}"
        exit 1
    fi

    if [[ -f "${VENV_DIR}/bin/nvrx-smonsvc" ]]; then
        echo -e "${GREEN}✓ nvrx-smonsvc installed${NC}"
    else
        echo -e "${RED}✗ nvrx-smonsvc not found${NC}"
        exit 1
    fi

    # Create config file (required - systemd loads all env from this file)
    echo ""
    echo "=== Creating config file ==="

    if [[ ! -f "${CONFIG_DIR}/nvrx.env" ]]; then
        cat > "${CONFIG_DIR}/nvrx.env" << EOF
# NVRX Services Configuration
# Shared by nvrx-attrsvc and nvrx-smonsvc

# ─── Common ───
NVIDIA_API_KEY_FILE=${CONFIG_DIR}/nvidia_api_key
NVRX_LOGS_DIR=${LOG_DIR}

# ─── Attribution Service (nvrx-attrsvc) ───
# Required: Root directory for log files to analyze
NVRX_ATTRSVC_ALLOWED_ROOT=/
NVRX_ATTRSVC_PORT=8000
NVRX_ATTRSVC_HOST=0.0.0.0
NVRX_ATTRSVC_LOG_LEVEL_NAME=DEBUG
# NVRX_ATTRSVC_DATAFLOW_INDEX=your-index
# NVRX_ATTRSVC_CLUSTER_NAME=your-cluster

# ─── SLURM Monitor (nvrx-smonsvc) ───
NVRX_ATTRSVC_URL=http://localhost:8000
NVRX_SMONSVC_PORT=8100
NVRX_SMONSVC_PARTITIONS=batch batch_long
NVRX_SMONSVC_INTERVAL=180
NVRX_SMONSVC_LOG_LEVEL=DEBUG
EOF
        if [[ "$USER_MODE" == false ]]; then
            chmod 640 "${CONFIG_DIR}/nvrx.env"
            chown root:"$NVRX_GROUP" "${CONFIG_DIR}/nvrx.env"
        else
            chmod 600 "${CONFIG_DIR}/nvrx.env"
        fi
        echo "Created: ${CONFIG_DIR}/nvrx.env"
    else
        echo "Already exists: ${CONFIG_DIR}/nvrx.env"
    fi

    # Install systemd service files
    echo ""
    echo "=== Installing systemd service files ==="

    ATTRSVC_SERVICE="${ATTRSVC_DIR}/deploy/nvrx-attrsvc.service"
    SMONSVC_SERVICE="${SMONSVC_DIR}/deploy/nvrx-smonsvc.service"

    if [[ -f "$ATTRSVC_SERVICE" ]]; then
        if [[ "$USER_MODE" == true ]]; then
            # User mode: remove User/Group lines, update paths
            sed -e '/^User=/d' \
                -e '/^Group=/d' \
                -e "s|/opt/nvrx|${INSTALL_DIR}|g" \
                -e "s|/etc/nvrx|${CONFIG_DIR}|g" \
                -e "s|/var/log/nvrx|${LOG_DIR}|g" \
                "$ATTRSVC_SERVICE" > "${SYSTEMD_DIR}/nvrx-attrsvc.service"
            echo "Installed: ${SYSTEMD_DIR}/nvrx-attrsvc.service"
        else
            # System mode: update User/Group
            sed -e "s/^User=.*/User=${NVRX_USER}/" \
                -e "s/^Group=.*/Group=${NVRX_GROUP}/" \
                "$ATTRSVC_SERVICE" > /etc/systemd/system/nvrx-attrsvc.service
            echo "Installed: /etc/systemd/system/nvrx-attrsvc.service (User=${NVRX_USER}, Group=${NVRX_GROUP})"
        fi
    else
        echo -e "${YELLOW}Warning: ${ATTRSVC_SERVICE} not found${NC}"
    fi

    if [[ -f "$SMONSVC_SERVICE" ]]; then
        if [[ "$USER_MODE" == true ]]; then
            # User mode: remove User/Group lines, update paths
            sed -e '/^User=/d' \
                -e '/^Group=/d' \
                -e "s|/opt/nvrx|${INSTALL_DIR}|g" \
                -e "s|/etc/nvrx|${CONFIG_DIR}|g" \
                -e "s|/var/log/nvrx|${LOG_DIR}|g" \
                "$SMONSVC_SERVICE" > "${SYSTEMD_DIR}/nvrx-smonsvc.service"
            echo "Installed: ${SYSTEMD_DIR}/nvrx-smonsvc.service"
        else
            # System mode: update User/Group
            sed -e "s/^User=.*/User=${NVRX_USER}/" \
                -e "s/^Group=.*/Group=${NVRX_GROUP}/" \
                "$SMONSVC_SERVICE" > /etc/systemd/system/nvrx-smonsvc.service
            echo "Installed: /etc/systemd/system/nvrx-smonsvc.service (User=${NVRX_USER}, Group=${NVRX_GROUP})"
        fi
    else
        echo -e "${YELLOW}Warning: ${SMONSVC_SERVICE} not found${NC}"
    fi

    # Reload systemd
    $SYSTEMCTL_CMD daemon-reload
    echo "Reloaded systemd daemon"

    # Print summary
    echo ""
    echo "=============================================="
    echo -e "${GREEN}Setup complete!${NC}"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo ""

    if [[ "$USER_MODE" == true ]]; then
        echo "1. Create NVIDIA API key file:"
        echo "   echo 'nvapi-xxx' > ${CONFIG_DIR}/nvidia_api_key"
        echo "   chmod 600 ${CONFIG_DIR}/nvidia_api_key"
        echo ""
        echo "2. Edit configuration (set NVRX_ATTRSVC_ALLOWED_ROOT):"
        echo "   vim ${CONFIG_DIR}/nvrx.env"
        echo ""
        echo "3. Enable and start services:"
        echo "   systemctl --user enable nvrx-attrsvc nvrx-smonsvc"
        echo "   $0 --user start"
        echo ""
        echo "4. Check status:"
        echo "   $0 --user status"
        echo ""
        echo "5. View logs:"
        echo "   $0 --user logs"
    else
        echo "1. Create NVIDIA API key file:"
        echo "   echo 'nvapi-xxx' | sudo tee ${CONFIG_DIR}/nvidia_api_key"
        echo "   sudo chmod 640 ${CONFIG_DIR}/nvidia_api_key"
        echo "   sudo chown root:${NVRX_GROUP} ${CONFIG_DIR}/nvidia_api_key"
        echo ""
        echo "2. Edit configuration (set NVRX_ATTRSVC_ALLOWED_ROOT):"
        echo "   sudo vim ${CONFIG_DIR}/nvrx.env"
        echo ""
        echo "3. Enable and start services:"
        echo "   sudo systemctl enable nvrx-attrsvc nvrx-smonsvc"
        echo "   sudo $0 start"
        echo ""
        echo "4. Check status:"
        echo "   sudo $0 status"
        echo ""
        echo "5. View logs:"
        echo "   sudo $0 logs"
    fi

    echo ""
    echo "Configuration:"
    echo "   ${CONFIG_DIR}/nvrx.env        - All settings for both services"
    echo "   ${CONFIG_DIR}/nvidia_api_key  - API key (create this)"
    echo ""
    echo "Installed:"
    echo "   ${INSTALL_DIR}/venv/bin/nvrx-attrsvc"
    echo "   ${INSTALL_DIR}/venv/bin/nvrx-smonsvc"
    echo "   ${LOG_DIR}/  (log files)"
    echo ""
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
    snapshot)
        cmd_snapshot
        ;;
    -h|--help|help)
        usage
        ;;
    "")
        echo "Usage: $0 <command>"
        echo "Commands: install, start, stop, restart, status, logs, snapshot"
        echo "Run '$0 --help' for details"
        exit 1
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run '$0 --help' for available commands"
        exit 1
        ;;
esac
