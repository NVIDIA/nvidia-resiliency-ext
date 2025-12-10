#!/bin/bash
set -e

# Extract git commit hash
NVRX_COMMIT=$(git rev-parse --short HEAD)

# Extract base version
base_version=$(grep -Po '(?<=^version = ")[^"]+' pyproject.toml)

# Compose local version segment
local_version="${NVRX_COMMIT}"


# Function to revert version
cleanup() {
  sed -i "s/^version = .*/version = \"${base_version}\"/" pyproject.toml
}

# Set trap to run cleanup on exit
trap cleanup EXIT

# Construct final version string
new_version="${base_version}+${local_version}"

echo "You are installing NVRx: ${new_version}"
sed -i "s/^version = .*/version = \"${new_version}\"/" pyproject.toml

# Proceed with installation
pip install .
