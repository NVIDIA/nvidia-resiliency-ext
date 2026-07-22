#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /absolute/path/to/log-directory [target ...]" >&2
  exit 2
fi

log_dir=$1
shift
if [[ $# -eq 0 ]]; then
  set -- deterministic
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
product_repo=${NVRX_RESTART_AGENT_PRODUCT_REPO:-$(cd "$script_dir/../../.." && pwd)}
log_root=${RESTART_AGENT_EVAL_LOG_ROOT:-$log_dir}
gold_root=${RESTART_AGENT_EVAL_GOLD_ROOT:?set RESTART_AGENT_EVAL_GOLD_ROOT}
run_root=${RESTART_AGENT_EVAL_RUN_ROOT:?set RESTART_AGENT_EVAL_RUN_ROOT}

shopt -s nullglob
logs=("$log_dir"/*.log)
if [[ ${#logs[@]} -eq 0 ]]; then
  echo "no .log files found under $log_dir" >&2
  exit 2
fi

for log_path in "${logs[@]}"; do
  echo "reviewing: $log_path"
  PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/private/tmp/nvrx-restart-agent-eval-pycache} \
    "$python_bin" "$script_dir/../src/review_log.py" \
    --log "$log_path" \
    --log-root "$log_root" \
    --gold-root "$gold_root" \
    --run-root "$run_root" \
    --product-repo "$product_repo" \
    "$@"
done

echo "all reviews: $run_root"
