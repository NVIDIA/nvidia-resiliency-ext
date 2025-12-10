#!/bin/bash
# Extract git commit hash
NVRX_COMMIT=$(git rev-parse HEAD)

# Extract base version
base_version=$(grep -Po '(?<=^version = ")[^"]+' pyproject.toml)

# Compose local version segment
local_version="git.${NVRX_COMMIT:0:8}"

# Extract cuda version from nvcc if available
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep -Po 'V[0-9]+\.[0-9]+' | head -1 | tr -d 'V')
    if [[ -n "$cuda_version" ]]; then
        if [[ -n "$local_version" ]]; then
            local_version="cu${cuda_version//./}-${local_version}"
        else
            local_version="cu${cuda_version//./}"
        fi
    fi
fi

# Construct final version string
if [[ -n "$local_version" ]]; then
  new_version="${base_version}+${local_version}"
else
  new_version="${base_version}"
fi

echo "You are installing NVRx: ${new_version}"
sed -i "s/^version = .*/version = \"${new_version}\"/" pyproject.toml

# Proceed with installation
pip install .

# Revert version change after installation
sed -i "s/^version = .*/version = \"${base_version}\"/" pyproject.toml
