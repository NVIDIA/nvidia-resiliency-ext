#!/bin/bash
# Build an enroot squash image for NVRX Attribution Service
#
# Usage:
#   ./build_enroot_image.sh [output_path]
#
# Prerequisites:
#   - enroot installed
#   - Run from repo root: ./examples/attribution/build_enroot_image.sh
#
# The resulting .sqsh file can be used with Slurm + pyxis:
#   srun --container-image=/path/to/nvrx-attrsvc.sqsh \
#        --container-env=NVRX_ATTRSVC_ALLOWED_ROOT=/data \
#        --container-mounts=/path/to/logs:/data:ro \
#        nvrx-attrsvc

set -euo pipefail

# Configuration
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
CONTAINER_NAME="${CONTAINER_NAME:-nvrx-attrsvc-build}"
OUTPUT_PATH="${1:-nvrx-attrsvc.sqsh}"

echo "=== NVRX Attribution Service - Enroot Image Builder ==="
echo "Base image: ${BASE_IMAGE}"
echo "Output: ${OUTPUT_PATH}"
echo ""

# Check we're in repo root
if [[ ! -f "pyproject.toml" ]]; then
    echo "ERROR: Run this script from the repo root directory"
    echo "  cd /path/to/nvidia-resiliency-ext"
    echo "  ./examples/attribution/build_enroot_image.sh"
    exit 1
fi

# Clean up any existing container with same name
enroot remove -f "${CONTAINER_NAME}" 2>/dev/null || true

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
    apt-get install -y --no-install-recommends build-essential git
    rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir --upgrade pip
'

# Step 4: Copy source code and install packages
echo "=== Step 4: Installing nvidia_resiliency_ext (no-deps) and nvrx-attrsvc ==="
REPO_ROOT="$(pwd)"

# Echo the command for debugging
echo "Running: enroot start --rw --root --mount ${REPO_ROOT}:/tmp/repo ${CONTAINER_NAME} bash -c '...'"

enroot start --rw --root --mount "${REPO_ROOT}:/tmp/repo" "${CONTAINER_NAME}" bash -c '
    set -e
    cd /tmp/repo
    
    # Install main library without CUDA extensions (skip CUPTI build)
    STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install --no-cache-dir --no-deps .
    
    # Install attribution service (installs fastapi, uvicorn, mcp, logsage, etc.)
    pip install --no-cache-dir ./examples/attribution
'

# Step 5: Set environment variables
echo "=== Step 5: Setting environment variables ==="
enroot start --rw --root "${CONTAINER_NAME}" bash -c '
    cat >> /etc/environment << EOF
NVRX_ATTRSVC_HOST=0.0.0.0
NVRX_ATTRSVC_PORT=8000
NVRX_ATTRSVC_LOG_LEVEL_NAME=INFO
NVRX_ATTRSVC_ALLOWED_ROOT=/data
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
echo "  srun --container-image=${OUTPUT_PATH} \\"
echo "       --container-env=NVRX_ATTRSVC_ALLOWED_ROOT=/data \\"
echo "       --container-mounts=/path/to/logs:/data:ro \\"
echo "       nvrx-attrsvc"
echo ""
echo "Or submit as a batch job:"
echo "  sbatch --container-image=${OUTPUT_PATH} job_script.sh"
