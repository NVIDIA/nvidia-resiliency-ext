#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}

PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/private/tmp/nvrx-restart-agent-eval-pycache} \
  "$python_bin" "$script_dir/../src/summarize_decision_stability.py" "$@"
