#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/path/to/input.log [output-directory]" >&2
  exit 2
fi

log_path=$1
if [[ ! -f "$log_path" ]]; then
  echo "log file does not exist: $log_path" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tool_root=$(cd "$script_dir/.." && pwd)
product_repo=${NVRX_RESTART_AGENT_PRODUCT_REPO:-$(cd "$tool_root/../../.." && pwd)}
python_bin=${NVRX_RESTART_AGENT_PRODUCT_PYTHON:-"$product_repo/.venv/bin/python"}

if [[ ! -x "$python_bin" ]]; then
  echo "product Python is not executable: $python_bin" >&2
  exit 2
fi

if [[ $# -eq 2 ]]; then
  run_dir=$2
else
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  run_dir="${RESTART_AGENT_EVAL_RUN_ROOT:-/private/tmp}/deterministic_history/$timestamp"
fi

mkdir -p "$run_dir"
history_path="$run_dir/attempt_records.json"
l0_bundle_path="$run_dir/l0_bundle.json"
job_id="deterministic-history-smoke"

echo "source log: $log_path"
echo "run directory: $run_dir"
echo

for cycle_id in 1 2 3 4 5; do
  result_path="$run_dir/cycle_${cycle_id}.result.json"
  trace_path="$run_dir/cycle_${cycle_id}.trace.json"
  stdout_path="$run_dir/cycle_${cycle_id}.stdout.json"

  args=(
    "$log_path"
    --job-id "$job_id"
    --cycle-id "$cycle_id"
    --attempt-records-json-out "$history_path"
    --result-json "$result_path"
    --trace-json "$trace_path"
    --summary
  )
  if [[ $cycle_id -eq 1 ]]; then
    args+=(--l0-bundle-json-out "$l0_bundle_path")
  else
    args+=(
      --attempt-records-json-in "$history_path"
      --l0-bundle-json-in "$l0_bundle_path"
    )
  fi

  PYTHONPATH="$product_repo/src${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/private/tmp/nvrx-restart-agent-pycache} \
    "$python_bin" \
    -m nvidia_resiliency_ext.attribution.restart_agent.cli \
    "${args[@]}" >"$stdout_path"

  "$python_bin" -c '
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
policy = payload["retry_policy"]
root = policy["general_root_ceiling"]
selected = policy.get("selected_rule_budget")
decision = payload["decision"]
decision_basis = payload["decision_basis"]
rule = policy["rule"]
root_matching = root["matching_prior_failures"]
root_allowed = root["allowed_retries"]
exhausted_by = policy["exhausted_by"]
selected_text = "-"
if selected:
    selected_rule = selected["rule"]
    selected_matching = selected["matching_prior_failures"]
    selected_allowed = selected["allowed_retries"]
    selected_text = (
        f"{selected_rule}:prior_matches={selected_matching} "
        f"matching_occurrences={selected_matching + 1} "
        f"allowed_retries={selected_allowed}"
    )
print(
    f"cycle={sys.argv[2]} decision={decision} "
    f"basis={decision_basis} rule={rule} "
    f"root_prior_matches={root_matching} "
    f"root_matching_occurrences={root_matching + 1} "
    f"root_allowed_retries={root_allowed} "
    f"selected={selected_text} exhausted_by={exhausted_by}"
)
' "$result_path" "$cycle_id"
done

"$python_bin" -c '
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
actual = [
    json.loads((run_dir / f"cycle_{cycle}.result.json").read_text())["decision"]
    for cycle in range(1, 6)
]
expected = ["RESTART", "STOP", "STOP", "STOP", "STOP"]
if actual != expected:
    raise SystemExit(f"unexpected decisions: actual={actual}, expected={expected}")

first_stop = json.loads((run_dir / "cycle_2.result.json").read_text())
policy = first_stop["retry_policy"]
selected = policy.get("selected_rule_budget") or {}
if policy["exhausted_by"] != ["selected_rule_budget"]:
    raise SystemExit(
        "cycle 2 did not exhaust the confirmation retry budget: "
        f"{policy['exhausted_by']}"
    )
if selected.get("rule") != "confirmation_retry":
    raise SystemExit(f"unexpected selected rule budget: {selected}")
final = json.loads((run_dir / "cycle_5.result.json").read_text())
final_selected = final["retry_policy"].get("selected_rule_budget") or {}
if final_selected.get("matching_prior_failures") != 4:
    raise SystemExit(
        "cycle 5 did not observe four prior no-progress matches: "
        f"{final_selected}"
    )
history_path = run_dir / "attempt_records.json"
print()
print(f"PASS decisions={actual}")
print("history_count=5 no_progress_comparisons=4 first_stop_cycle=2")
print(f"history={history_path}")
print(f"artifacts={run_dir}")
' "$run_dir"
