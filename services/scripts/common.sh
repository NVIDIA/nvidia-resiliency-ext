#!/bin/bash
# Common setup functions for NVRX Attribution services
# Source this file: source "$(dirname "$0")/common.sh"

# Get the directory where this script lives
COMMON_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup LLM API key for processes that read nvidia_resiliency_ext.attribution.api_keys.load_llm_api_key
# Checks in order:
#   1. LLM_API_KEY environment variable
#   2. LLM_API_KEY_FILE (path to file; must exist)
#   3. ~/.llm_api_key — sets export LLM_API_KEY_FILE to that path
#   4. ~/.config/nvrx/llm_api_key
setup_llm_api_key() {
    if [[ -n "${LLM_API_KEY}" ]]; then
        echo "Using LLM_API_KEY from environment"
        return 0
    fi

    if [[ -n "${LLM_API_KEY_FILE}" ]]; then
        if [[ ! -f "${LLM_API_KEY_FILE}" ]]; then
            echo "ERROR: LLM_API_KEY_FILE specified but not found: ${LLM_API_KEY_FILE}"
            return 1
        fi
        echo "Using LLM_API_KEY_FILE=${LLM_API_KEY_FILE}"
        return 0
    fi

    local KEY_LOCATIONS=(
        "${HOME}/.llm_api_key"
        "${HOME}/.config/nvrx/llm_api_key"
    )
    for key_file in "${KEY_LOCATIONS[@]}"; do
        if [[ -f "${key_file}" ]]; then
            export LLM_API_KEY_FILE="${key_file}"
            echo "Using API key from: ${key_file}"
            return 0
        fi
    done

    echo "WARNING: LLM API key not found - LLM analysis may fail"
    echo "  Set LLM_API_KEY, LLM_API_KEY_FILE, or create ~/.llm_api_key"
    return 1
}

# Install NVRX packages from local repo (editable mode)
# Args:
#   $1 - "attrsvc" or "smonsvc" or "both" (default: both)
#   $2 - repo root directory (default: NVRX_REPO_DIR or ~/nvidia-resiliency-ext)
# Environment:
#   PIP_EXTRA_INDEX_URL - optional extra index for package resolution
install_nvrx_packages() {
    local mode="${1:-both}"
    local repo_dir="${2:-${NVRX_REPO_DIR:-${HOME}/nvidia-resiliency-ext}}"
    local extras=""
    local -a pip_extra_args=()
    
    echo "Installing NVRX packages from ${repo_dir}..."

    # Remove older split-service installs so the root package owns both commands.
    if pip show nvidia-resiliency-ext &>/dev/null || pip show nvrx-attrsvc &>/dev/null; then
        echo "  Uninstalling existing NVRX packages..."
        pip uninstall nvidia-resiliency-ext nvrx-attrsvc -y --quiet 2>/dev/null || true
    fi

    if [[ "$mode" == "attrsvc" || "$mode" == "both" ]]; then
        extras="[attribution]"
        if [[ -n "${PIP_EXTRA_INDEX_URL}" ]]; then
            pip_extra_args=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")
        fi
    fi

    echo "  Installing nvidia-resiliency-ext${extras} (editable from repo)..."
    STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install \
        --no-cache-dir \
        "${pip_extra_args[@]}" \
        -e "${repo_dir}${extras}" \
        --quiet

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

# Wait for a service to be ready on a port or endpoint
# Args:
#   $1 - service name (for messages)
#   $2 - PID of the service process
#   $3 - port number or endpoint
#   $4 - max attempts (default: 10)
wait_for_service() {
    local name="$1"
    local pid="$2"
    local target="$3"
    local max_attempts="${4:-10}"
    local -a curl_args
    if [[ "${target}" == unix://* ]]; then
        curl_args=(--unix-socket "${target#unix://}" "http://localhost/healthz")
    elif [[ "${target}" == /* ]]; then
        curl_args=(--unix-socket "${target}" "http://localhost/healthz")
    elif [[ "${target}" == http* ]]; then
        curl_args=("${target%/}/healthz")
    else
        curl_args=("http://localhost:${target}/healthz")
    fi
    
    echo "  Waiting for ${name} to be ready..."
    for i in $(seq 1 $max_attempts); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: ${name} process died"
            return 1
        fi
        if curl -sf "${curl_args[@]}" >/dev/null 2>&1; then
            echo "  ${name} is responding at ${target}"
            return 0
        fi
        if [[ $i -eq $max_attempts ]]; then
            echo "ERROR: ${name} not responding at ${target}"
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
    local -a curl_args
    local attrsvc_endpoint="${NVRX_ATTRSVC_ENDPOINT:-}"
    local attrsvc_uds_path=""
    if [[ "${attrsvc_endpoint}" == unix://* ]]; then
        attrsvc_uds_path="${attrsvc_endpoint#unix://}"
    elif [[ "${attrsvc_endpoint}" == /* ]]; then
        attrsvc_uds_path="${attrsvc_endpoint}"
    fi
    
    echo "--- ${endpoint} @ ${timestamp} ---" >> "${outfile}"
    if [[ "${service}" == "attrsvc" && -n "${attrsvc_uds_path}" ]]; then
        curl_args=(--unix-socket "${attrsvc_uds_path}" "http://localhost${endpoint}")
    elif [[ "${service}" == "attrsvc" && "${attrsvc_endpoint}" == http* ]]; then
        curl_args=("${attrsvc_endpoint%/}${endpoint}")
    else
        curl_args=("http://${host}:${port}${endpoint}")
    fi
    if curl -s --max-time 10 "${curl_args[@]}" >> "${outfile}" 2>/dev/null; then
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
#   $1 - attrsvc endpoint (default: http://localhost:8000)
# Returns 0 if reachable, 1 if not (with warning)
check_attrsvc_reachable() {
    local endpoint="${1:-http://localhost:8000}"
    local -a curl_args
    if [[ "${endpoint}" == unix://* ]]; then
        curl_args=(--unix-socket "${endpoint#unix://}" "http://localhost/healthz")
    elif [[ "${endpoint}" == /* ]]; then
        curl_args=(--unix-socket "${endpoint}" "http://localhost/healthz")
    else
        curl_args=("${endpoint%/}/healthz")
    fi
    echo "Checking attribution service at ${endpoint}..."
    if curl -sf "${curl_args[@]}" >/dev/null 2>&1; then
        echo "  Attribution service is reachable"
        return 0
    else
        echo "WARNING: Attribution service not reachable at ${endpoint}"
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
