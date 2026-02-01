#!/bin/bash
# Common setup functions for NVRX Attribution services
# Source this file: source "$(dirname "$0")/common.sh"

# Get the directory where this script lives
COMMON_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup NVIDIA API key from environment, file, or default locations
# Checks in order:
#   1. NVIDIA_API_KEY environment variable
#   2. NVIDIA_API_KEY_FILE environment variable (path to file)
#   3. ~/.nvidia_api_key file
#   4. ~/.config/nvrx/nvidia_api_key file
# Sets NVIDIA_API_KEY environment variable
setup_nvidia_api_key() {
    if [[ -n "${NVIDIA_API_KEY}" ]]; then
        echo "Using NVIDIA_API_KEY from environment"
        return 0
    fi
    
    # Check NVIDIA_API_KEY_FILE
    if [[ -n "${NVIDIA_API_KEY_FILE}" ]]; then
        if [[ ! -f "${NVIDIA_API_KEY_FILE}" ]]; then
            echo "ERROR: NVIDIA_API_KEY_FILE specified but not found: ${NVIDIA_API_KEY_FILE}"
            return 1
        fi
        export NVIDIA_API_KEY=$(cat "${NVIDIA_API_KEY_FILE}" | tr -d '[:space:]')
        if [[ -z "${NVIDIA_API_KEY}" ]]; then
            echo "ERROR: NVIDIA_API_KEY_FILE is empty: ${NVIDIA_API_KEY_FILE}"
            return 1
        fi
        echo "Using API key from: ${NVIDIA_API_KEY_FILE}"
        return 0
    fi
    
    # Check default locations
    local KEY_LOCATIONS=(
        "${HOME}/.nvidia_api_key"
        "${HOME}/.config/nvrx/nvidia_api_key"
    )
    for key_file in "${KEY_LOCATIONS[@]}"; do
        if [[ -f "${key_file}" ]]; then
            export NVIDIA_API_KEY=$(cat "${key_file}" | tr -d '[:space:]')
            if [[ -n "${NVIDIA_API_KEY}" ]]; then
                echo "Using API key from: ${key_file}"
                return 0
            fi
        fi
    done
    
    echo "WARNING: NVIDIA_API_KEY not found - LLM analysis may fail"
    echo "  Set NVIDIA_API_KEY, NVIDIA_API_KEY_FILE, or create ~/.nvidia_api_key"
    return 1
}

# Install NVRX packages from local repo (editable mode)
# Args:
#   $1 - "attrsvc" or "smonsvc" or "both" (default: both)
#   $2 - repo root directory (default: NVRX_REPO_DIR or ~/nvidia-resiliency-ext)
# Environment:
#   PIP_EXTRA_INDEX_URL - if set, installs dataflow dependencies for attrsvc
install_nvrx_packages() {
    local mode="${1:-both}"
    local repo_dir="${2:-${NVRX_REPO_DIR:-${HOME}/nvidia-resiliency-ext}}"
    
    echo "Installing NVRX packages from ${repo_dir}..."
    
    # Install main library (skip CUPTI build for CPU-only environments)
    if [[ "$mode" == "attrsvc" || "$mode" == "both" || "$mode" == "smonsvc" ]]; then
        # Uninstall any existing copy so the editable install from repo is the one used.
        # Otherwise a pre-installed wheel/site-packages copy can take precedence.
        if pip show nvidia-resiliency-ext &>/dev/null; then
            echo "  Uninstalling existing nvidia-resiliency-ext..."
            pip uninstall nvidia-resiliency-ext -y --quiet 2>/dev/null || true
        fi
        echo "  Installing nvidia-resiliency-ext (editable from repo)..."
        STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install --no-cache-dir --no-deps -e "${repo_dir}" --quiet
    fi
    
    # Install attribution service
    if [[ "$mode" == "attrsvc" || "$mode" == "both" ]]; then
        if [[ -n "${PIP_EXTRA_INDEX_URL}" ]]; then
            echo "  Installing nvrx-attrsvc with dataflow..."
            pip install --no-cache-dir \
                --extra-index-url "${PIP_EXTRA_INDEX_URL}" \
                -e "${repo_dir}/services[dataflow]" --quiet
        else
            echo "  Installing nvrx-attrsvc..."
            pip install --no-cache-dir -e "${repo_dir}/services" --quiet
        fi
    elif [[ "$mode" == "smonsvc" ]]; then
        echo "  Installing nvrx-attrsvc (for smonsvc)..."
        pip install --no-cache-dir -e "${repo_dir}/services" --quiet
    fi

    # Always re-apply editable install of the library so it wins over any reinstall from nvrx-attrsvc deps (nvidia-resiliency-ext>=0.5.0).
    echo "  Ensuring nvidia-resiliency-ext is editable from repo..."
    STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install --no-cache-dir --no-deps -e "${repo_dir}" --force-reinstall --quiet

    echo "  Done."
    
    # Ensure pip script directory is in PATH
    export PATH="${HOME}/.local/bin:${PATH}"
}

# Validate that required commands exist
# Args: list of command names
validate_commands() {
    for cmd in "$@"; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: Required command '$cmd' not found in PATH"
            return 1
        fi
    done
}

# Wait for a service to be ready on a given port
# Args:
#   $1 - service name (for messages)
#   $2 - PID of the service process
#   $3 - port number
#   $4 - max attempts (default: 10)
wait_for_service() {
    local name="$1"
    local pid="$2"
    local port="$3"
    local max_attempts="${4:-10}"
    
    echo "  Waiting for ${name} to be ready..."
    for i in $(seq 1 $max_attempts); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: ${name} process died"
            return 1
        fi
        if curl -sf "http://localhost:${port}/healthz" >/dev/null 2>&1; then
            echo "  ${name} is responding on port ${port}"
            return 0
        fi
        if [[ $i -eq $max_attempts ]]; then
            echo "ERROR: ${name} not responding on port ${port}"
            return 1
        fi
        sleep 1
    done
}

# =============================================================================
# Snapshot/Monitoring Functions
# =============================================================================

# Snapshot a single endpoint and append to log file
# Args:
#   $1 - service name (for display)
#   $2 - host
#   $3 - port
#   $4 - endpoint path (e.g., "/healthz")
#   $5 - output file
#   $6 - timestamp
snapshot_endpoint() {
    local service="$1"
    local host="$2"
    local port="$3"
    local endpoint="$4"
    local outfile="$5"
    local timestamp="$6"
    
    echo "--- ${endpoint} @ ${timestamp} ---" >> "${outfile}"
    if curl -s --max-time 10 "http://${host}:${port}${endpoint}" >> "${outfile}" 2>/dev/null; then
        echo "" >> "${outfile}"  # newline after JSON
        echo "  ${service} ${endpoint}: OK"
    else
        echo "FAILED" >> "${outfile}"
        echo "  ${service} ${endpoint}: FAILED"
    fi
}

# Write snapshot header to log file
# Args:
#   $1 - output file
#   $2 - dump count
#   $3 - timestamp
snapshot_header() {
    local outfile="$1"
    local count="$2"
    local timestamp="$3"
    
    echo "" >> "${outfile}"
    echo "========== SNAPSHOT #${count} @ ${timestamp} ==========" >> "${outfile}"
}

# Snapshot all attrsvc endpoints
# Args:
#   $1 - host
#   $2 - port
#   $3 - output file
#   $4 - dump count
#   $5 - timestamp
snapshot_attrsvc() {
    local host="$1"
    local port="$2"
    local outfile="$3"
    local count="$4"
    local timestamp="$5"
    
    snapshot_header "${outfile}" "${count}" "${timestamp}"
    snapshot_endpoint "attrsvc" "${host}" "${port}" "/healthz" "${outfile}" "${timestamp}"
    snapshot_endpoint "attrsvc" "${host}" "${port}" "/stats" "${outfile}" "${timestamp}"
    snapshot_endpoint "attrsvc" "${host}" "${port}" "/jobs" "${outfile}" "${timestamp}"
}

# Snapshot all smonsvc endpoints
# Args:
#   $1 - host
#   $2 - port
#   $3 - output file
#   $4 - dump count
#   $5 - timestamp
snapshot_smonsvc() {
    local host="$1"
    local port="$2"
    local outfile="$3"
    local count="$4"
    local timestamp="$5"
    
    snapshot_header "${outfile}" "${count}" "${timestamp}"
    snapshot_endpoint "smonsvc" "${host}" "${port}" "/healthz" "${outfile}" "${timestamp}"
    snapshot_endpoint "smonsvc" "${host}" "${port}" "/stats" "${outfile}" "${timestamp}"
}

# =============================================================================
# Validation Functions
# =============================================================================

# Validate NVRX_ATTRSVC_ALLOWED_ROOT is set
# Returns 1 if not set (caller should exit)
validate_attrsvc_allowed_root() {
    if [[ -z "${NVRX_ATTRSVC_ALLOWED_ROOT}" ]]; then
        echo "ERROR: NVRX_ATTRSVC_ALLOWED_ROOT is required."
        echo "  Set it before running: NVRX_ATTRSVC_ALLOWED_ROOT=/path/to/logs ..."
        return 1
    fi
}

# Check if attribution service is reachable
# Args:
#   $1 - attrsvc URL (default: http://localhost:8000)
# Returns 0 if reachable, 1 if not (with warning)
check_attrsvc_reachable() {
    local url="${1:-http://localhost:8000}"
    echo "Checking attribution service at ${url}..."
    if curl -sf "${url}/healthz" >/dev/null 2>&1; then
        echo "  Attribution service is reachable"
        return 0
    else
        echo "WARNING: Attribution service not reachable at ${url}"
        echo "  Monitor will retry on each poll cycle"
        return 1
    fi
}

# Wait for a background service to be ready (with PID file)
# Args:
#   $1 - service name (for messages)
#   $2 - PID
#   $3 - port number
#   $4 - log file path
#   $5 - PID file path
#   $6 - max attempts (default: 30)
# Returns 0 on success, 1 on failure (cleans up PID file)
wait_for_background_service() {
    local name="$1"
    local pid="$2"
    local port="$3"
    local log_file="$4"
    local pid_file="$5"
    local max_attempts="${6:-30}"
    
    echo "  Waiting for service..."
    for i in $(seq 1 $max_attempts); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: ${name} died. Check ${log_file}"
            rm -f "${pid_file}"
            return 1
        fi
        if curl -sf "http://localhost:${port}/healthz" >/dev/null 2>&1; then
            echo "  Ready: http://localhost:${port}"
            return 0
        fi
        sleep 1
    done
    
    echo "ERROR: ${name} not responding on port ${port}. Check ${log_file}"
    return 1
}

# Ensure a directory exists, create if needed
# Args:
#   $1 - directory path
#   $2 - description (for messages, default: "directory")
ensure_directory() {
    local dir="$1"
    local desc="${2:-directory}"
    
    if [[ ! -d "${dir}" ]]; then
        echo "Creating ${desc}: ${dir}"
        mkdir -p "${dir}"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: Failed to create ${desc}: ${dir}"
            return 1
        fi
    fi
}
