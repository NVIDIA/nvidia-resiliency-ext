#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /absolute/path/to/input.log [target ...]" >&2
  echo "set RESTART_AGENT_EVAL_LOG_ROOT, RESTART_AGENT_EVAL_GOLD_ROOT, and RESTART_AGENT_EVAL_RUN_ROOT" >&2
  exit 2
fi

log_path=$1
shift
if [[ $# -eq 0 ]]; then
  set -- deterministic
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
product_repo=${NVRX_RESTART_AGENT_PRODUCT_REPO:-$(cd "$script_dir/../../.." && pwd)}

PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/private/tmp/nvrx-restart-agent-eval-pycache} \
  "$python_bin" "$script_dir/../src/review_log.py" \
  --log "$log_path" \
  --product-repo "$product_repo" \
  "$@"
