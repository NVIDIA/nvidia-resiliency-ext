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
  run_dir="${RESTART_AGENT_EVAL_RUN_ROOT:-/private/tmp}/rejected_iteration_history/$timestamp"
fi

mkdir -p "$run_dir"
history_path="$run_dir/attempt_records.json"
l0_bundle_path="$run_dir/l0_bundle.json"
job_id="rejected-iteration-history-replay"

echo "source log: $log_path"
echo "run directory: $run_dir"
echo

for cycle_id in 0 1 2; do
  result_path="$run_dir/cycle_${cycle_id}.result.json"
  trace_path="$run_dir/cycle_${cycle_id}.trace.json"
  stdout_path="$run_dir/cycle_${cycle_id}.stdout.json"

  args=(
    "$log_path"
    --job-id "$job_id"
    --cycle-id "$cycle_id"
    --disable-l1
    --attempt-records-json-out "$history_path"
    --result-json "$result_path"
    --trace-json "$trace_path"
  )
  if [[ $cycle_id -eq 0 ]]; then
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
effective = policy.get("effective_policy") or {}
context = policy.get("applied_policy_context") or {}
ledger = policy.get("selected_policy_ledger") or {}
decision = payload["decision"]
decision_basis = payload["decision_basis"]
rule = effective.get("rule")
matched = context.get("matched")
prior_matches = ledger.get("matching_prior_attempts", 0)
allowed_retries = ledger.get("allowed_retries")
exhausted = ledger.get("exhausted", False)
print(
    f"cycle={sys.argv[2]} decision={decision} "
    f"basis={decision_basis} policy={rule} matched={matched} "
    f"prior_matches={prior_matches} allowed_retries={allowed_retries} "
    f"exhausted={exhausted}"
)
' "$result_path" "$cycle_id"
done

"$python_bin" -c '
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
expected_decisions = ["RESTART", "RESTART", "STOP"]
results = [
    json.loads((run_dir / f"cycle_{cycle}.result.json").read_text(encoding="utf-8"))
    for cycle in range(3)
]
actual_decisions = [result["decision"] for result in results]
if actual_decisions != expected_decisions:
    raise SystemExit(
        f"unexpected decisions: actual={actual_decisions}, expected={expected_decisions}"
    )

expected_matches = [0, 1, 2]
for cycle, (result, expected_match_count) in enumerate(zip(results, expected_matches)):
    policy = result["retry_policy"]
    effective = policy.get("effective_policy") or {}
    context = policy.get("applied_policy_context") or {}
    ledger = policy.get("selected_policy_ledger") or {}
    if effective.get("rule") != "rejected_iteration_retry_then_skip":
        raise SystemExit(f"cycle {cycle} did not select the policy: {effective}")
    if context.get("matched") is not True:
        raise SystemExit(f"cycle {cycle} did not record a policy-context match: {context}")
    actual_match_count = ledger.get("matching_prior_attempts", 0)
    if actual_match_count != expected_match_count:
        raise SystemExit(
            f"cycle {cycle} prior-match count is incorrect: "
            f"actual={actual_match_count}, expected={expected_match_count}"
        )

final_exhausted_by = results[-1]["retry_policy"].get("exhausted_by") or []
if "selected_policy_ledger" not in final_exhausted_by:
    raise SystemExit(
        "cycle 2 did not include selected_policy_ledger exhaustion: "
        f"{final_exhausted_by}"
    )

print()
print(f"PASS decisions={actual_decisions}")
print("policy=rejected_iteration_retry_then_skip allowed_retries=2 first_stop_cycle=2")
history_path = run_dir / "attempt_records.json"
print(f"history={history_path}")
print(f"artifacts={run_dir}")
' "$run_dir"
