#!/bin/bash
# Build an enroot squash image for NVRX Attribution Services
# (includes both nvrx-attrsvc and nvrx-smonsvc)
#
# Usage:
#   ./build_enroot_image.sh [output_path]
#
# Prerequisites:
#   - enroot installed
#   - Run from repo root: ./services/scripts/build_enroot_image.sh
#
# The resulting .sqsh file can be used with Slurm + pyxis:
#
#   # Run attribution service
#   srun --container-image=/path/to/nvrx-services.sqsh \
#        --container-env=NVRX_ATTRSVC_ALLOWED_ROOT=/data \
#        --container-env=NVIDIA_API_KEY=${NVIDIA_API_KEY} \
#        --container-mounts=/path/to/logs:/data:ro \
#        nvrx-attrsvc
#
#   # Run SLURM monitor (requires attrsvc running elsewhere)
#   srun --container-image=/path/to/nvrx-services.sqsh \
#        --container-env=NVRX_ATTRSVC_URL=http://attrsvc-host:8000 \
#        --container-env=NVRX_SMONSVC_PARTITIONS="batch gpu" \
#        nvrx-smonsvc

set -euo pipefail

# Configuration
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
CONTAINER_NAME="${CONTAINER_NAME:-nvrx-services-build}"
OUTPUT_PATH="${1:-nvrx-services.sqsh}"

echo "=== NVRX Attribution Services - Enroot Image Builder ==="
echo "Base image: ${BASE_IMAGE}"
echo "Output: ${OUTPUT_PATH}"
echo ""

# Validate enroot is installed
if ! command -v enroot &>/dev/null; then
    echo "ERROR: 'enroot' command not found in PATH"
    echo "Install enroot: https://github.com/NVIDIA/enroot"
    exit 1
fi

# Check we're in repo root
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: Run this script from the repo root directory"
    echo "  cd /path/to/nvidia-resiliency-ext"
    echo "  ./services/build_enroot_image.sh"
    exit 1
fi

# Cleanup function for failed builds
BUILD_STARTED=false
cleanup_on_failure() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]] && [[ "$BUILD_STARTED" == "true" ]]; then
        echo ""
        echo "=== Build failed, cleaning up ==="
        enroot remove -f "${CONTAINER_NAME}" 2>/dev/null || true
        rm -f "${CONTAINER_NAME}.sqsh" 2>/dev/null || true
        echo "Cleanup complete."
    fi
    exit $exit_code
}
trap cleanup_on_failure EXIT

# Clean up any existing container with same name
enroot remove -f "${CONTAINER_NAME}" 2>/dev/null || true
BUILD_STARTED=true

# Step 1: Import base image
echo "=== Step 1: Importing base image ==="
enroot import -o "${CONTAINER_NAME}.sqsh" "docker://${BASE_IMAGE}"

# Step 2: Create container from image
echo "=== Step 2: Creating container ==="
enroot create --name "${CONTAINER_NAME}" "${CONTAINER_NAME}.sqsh"
rm -f "${CONTAINER_NAME}.sqsh"  # Clean up intermediate file

# Step 3: Install system dependencies
echo "=== Step 3: Installing system dependencies ==="
enroot start --rw --root "${CONTAINER_NAME}" bash -c '
    set -e
    apt-get update
    apt-get install -y --no-install-recommends build-essential git curl
    rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir --upgrade pip
'

# Step 4: Copy source code and install packages
# Note: This mirrors install_nvrx_packages() from common.sh but runs inside container
echo "=== Step 4: Installing nvidia_resiliency_ext and nvrx services ==="
REPO_ROOT="$(pwd)"

enroot start --rw --root --mount "${REPO_ROOT}:/tmp/repo" "${CONTAINER_NAME}" bash -c '
    set -e
    cd /tmp/repo
    
    # Install main library without CUDA extensions (skip CUPTI build)
    echo "  Installing nvidia-resiliency-ext..."
    STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install --no-cache-dir --no-deps .
    
    # Install attribution service (installs fastapi, uvicorn, mcp, logsage, etc.)
    echo "  Installing nvrx-attrsvc..."
    pip install --no-cache-dir ./services
    
    # Verify commands are available
    echo "  Verifying installed commands..."
    command -v nvrx-attrsvc || { echo "ERROR: nvrx-attrsvc not found"; exit 1; }
    command -v nvrx-smonsvc || { echo "ERROR: nvrx-smonsvc not found"; exit 1; }
    echo "  nvrx-attrsvc: $(command -v nvrx-attrsvc)"
    echo "  nvrx-smonsvc: $(command -v nvrx-smonsvc)"
'

# Step 5: Set default environment variables
# These can be overridden at runtime via --container-env
echo "=== Step 5: Setting default environment variables ==="
enroot start --rw --root "${CONTAINER_NAME}" bash -c '
    cat >> /etc/environment << EOF
# Attribution Service defaults
NVRX_ATTRSVC_HOST=0.0.0.0
NVRX_ATTRSVC_PORT=8000
NVRX_ATTRSVC_LOG_LEVEL_NAME=INFO
NVRX_ATTRSVC_ALLOWED_ROOT=/data

# SLURM Monitor defaults
NVRX_SMONSVC_PORT=8100
NVRX_SMONSVC_INTERVAL=180
NVRX_SMONSVC_VERBOSE=false
NVRX_ATTRSVC_URL=http://localhost:8000
EOF
'

# Step 6: Export as squash file
echo "=== Step 6: Exporting squash image ==="
enroot export -o "${OUTPUT_PATH}" "${CONTAINER_NAME}"

# Cleanup
echo "=== Cleanup ==="
enroot remove -f "${CONTAINER_NAME}"

echo ""
echo "=== Build complete ==="
echo "Squash image: ${OUTPUT_PATH}"
echo ""
echo "Usage with Slurm + pyxis:"
echo ""
echo "  # Attribution service"
echo "  srun --container-image=${OUTPUT_PATH} \\"
echo "       --container-env=NVRX_ATTRSVC_ALLOWED_ROOT=/data \\"
echo "       --container-env=NVIDIA_API_KEY=\${NVIDIA_API_KEY} \\"
echo "       --container-mounts=/path/to/logs:/data:ro \\"
echo "       nvrx-attrsvc"
echo ""
echo "  # SLURM monitor"
echo "  srun --container-image=${OUTPUT_PATH} \\"
echo "       --container-env=NVRX_ATTRSVC_URL=http://attrsvc-host:8000 \\"
echo "       --container-env=NVRX_SMONSVC_PARTITIONS=\"batch gpu\" \\"
echo "       nvrx-smonsvc"
echo ""
echo "Or use the combined SLURM script:"
echo "  sbatch services/scripts/nvrx_services.sbatch"
