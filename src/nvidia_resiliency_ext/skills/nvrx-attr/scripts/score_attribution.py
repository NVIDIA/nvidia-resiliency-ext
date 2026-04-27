#!/usr/bin/env python3
"""LLM-judge scorer for fault-injection attribution experiments.

Uses the same ChatOpenAI / NVIDIA inference API setup as nvrx_logsage.py.
Reads ground-truth fault parameters and the raw text outputs of nvrx_logsage
and CollectiveAnalyzer, then asks a judge model to score each attribution
dimension and return structured JSON.

Usage (called by watch_and_analyze.sh):
    python3 score_attribution.py \
        --fault-type GPU_SLEEP --rank 0 --iter 5 --nodes 2 \
        --log-output "$LOG_OUT" \
        --fr-output  "$FR_OUT" \
        [--model qwen/qwen3.5-397b-a17b] \
        [--base-url https://inference.api.nvidia.com/v1]

Stdout: one line of JSON with keys:
    restart_correct, rank_primary, rank_any, fault_described, fr_rank_correct, notes
"""

import argparse
import json
import logging
import os
import re
import sys

from langchain_openai import ChatOpenAI

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4]))
from nvidia_resiliency_ext.attribution.api_keys import load_nvidia_api_key
from nvidia_resiliency_ext.attribution.svc.config import DEFAULT_LLM_BASE_URL

logger = logging.getLogger(__name__)

INJECTION_MARKERS = (
    "FAULT INJECTION",
    "nvidia_resiliency_ext.shared_utils.inject_fault",
)

# Default judge model — override with --model
DEFAULT_JUDGE_MODEL = "qwen/qwen3.5-397b-a17b"
DEFAULT_GPUS_PER_NODE = int(os.getenv("GPUS_PER_NODE", "4"))
DEFAULT_FR_SEGMENT_SIZE = int(os.getenv("FR_SEGMENT_SIZE", os.getenv("FR_RACK_SIZE", "32")))

# Expected restart decision and rationale per fault type
_RESTART_TABLE = {
    "GPU_SLEEP": ("RESTART IMMEDIATE", "transient GPU hang, recoverable"),
    "LOCK_GIL": ("RESTART IMMEDIATE", "transient Python GIL hang, recoverable"),
    "SIGTERM": ("RESTART IMMEDIATE", "external termination signal, recoverable"),
    "SIGINT": ("RESTART IMMEDIATE", "external interrupt signal, recoverable"),
    "SIGSTOP": ("RESTART IMMEDIATE", "external stop signal, recoverable"),
    "SIGNAL_EXC": ("RESTART IMMEDIATE", "signal-based exception, typically recoverable"),
    "GPU_ERROR": ("STOP - DONT RESTART IMMEDIATE", "hardware GPU error, may be persistent"),
    "SIGKILL": ("STOP - DONT RESTART IMMEDIATE", "hard kill, possible external pressure or OOM"),
    "SEGFAULT": (
        "STOP - DONT RESTART IMMEDIATE",
        "segmentation fault, likely code or memory corruption",
    ),
    "OS_ABORT": (
        "STOP - DONT RESTART IMMEDIATE",
        "OS abort, likely severe system or hardware fault",
    ),
    "WORKLOAD_EXC": ("STOP - DONT RESTART IMMEDIATE", "application exception, likely a code bug"),
    "ASYNC_EXC": (
        "STOP - DONT RESTART IMMEDIATE",
        "async exception in workload, likely a code bug",
    ),
}


def load_log_excerpt(log_path, max_lines=400):
    """Return up to max_lines from the log, keeping the tail (where errors appear).

    Applies the same exclude_nvrx_logs filtering as nvrx_logsage.py:analyze_logs().
    """
    if not log_path:
        return "(log file not provided)"
    try:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(log_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        # Mirrors nvrx_logsage.py exclude_nvrx_logs logic exactly
        lines = [line for line in lines if "nvidia_resiliency_ext" not in line]
        lines = [line for line in lines if "[workload:" not in line or 'Cycle:' in line]
        # Strip fault-injection markers — the judge must not see which rank/fault was
        # injected in the raw log; it knows the ground truth from the structured args.
        lines = [line for line in lines if not any(marker in line for marker in INJECTION_MARKERS)]
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return "".join(lines).strip()
    except Exception as exc:
        return f"(could not read log file: {exc})"


def parse_fr_missing_ranks(fr_output: str) -> set[int]:
    if not fr_output or fr_output.strip() in ("", "no_dumps", "no results", "run_invalid"):
        return set()

    ranks: set[int] = set()
    for line in fr_output.splitlines():
        if "|" not in line or "Missing Ranks" in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 6:
            continue
        last_col = parts[-1]
        for match in re.finditer(r"\d+", last_col):
            ranks.add(int(match.group(0)))
    return ranks


def normalize_fr_rank_correct(
    raw_label: str,
    fr_output: str,
    rank: int,
    total_ranks: int,
    gpus_per_node: int,
    segment_size: int,
) -> str:
    label = (raw_label or "").strip().lower()
    if label in {"n/a", ""}:
        return raw_label
    if label == "rack":
        label = "segment"
    if label == "no_dumps":
        return "no_dumps"

    fr_ranks = parse_fr_missing_ranks(fr_output)
    if not fr_ranks:
        return "no_dumps"

    node_start = (rank // gpus_per_node) * gpus_per_node
    node_end = min(node_start + gpus_per_node - 1, total_ranks - 1)
    segment_start = (rank // segment_size) * segment_size
    segment_end = min(segment_start + segment_size - 1, total_ranks - 1)

    in_node = all(node_start <= fr_rank <= node_end for fr_rank in fr_ranks)
    in_segment = all(segment_start <= fr_rank <= segment_end for fr_rank in fr_ranks)

    if label == "node":
        if in_node:
            return "node"
        if in_segment:
            return "segment"
        return "false"

    if label == "segment":
        return "segment" if in_segment else "false"

    return label


def build_judge_prompt(
    fault_type,
    rank,
    iter_,
    nodes,
    run_valid,
    log_output,
    fr_output,
    log_excerpt,
    gpus_per_node,
    segment_size,
):
    total_ranks = nodes * gpus_per_node
    node_index = rank // gpus_per_node
    segment_start = (rank // segment_size) * segment_size
    segment_end = min(segment_start + segment_size - 1, total_ranks - 1)
    expected_restart, restart_rationale = _RESTART_TABLE.get(
        fault_type, ("unknown", "unknown fault type")
    )

    if not run_valid:
        # Return early dict — caller will skip LLM call
        return {
            "restart_correct": "N/A",
            "rank_primary": "N/A",
            "rank_any": "N/A",
            "fault_described": "N/A",
            "fr_rank_correct": "N/A",
            "notes": "run_invalid: training did not reach the fault injection point; scores not meaningful",
        }

    fr_section = (
        fr_output
        if fr_output and fr_output.strip() not in ("no_dumps", "no results", "run_invalid", "")
        else "(no flight-recorder dumps available for this experiment)"
    )

    log_section = log_excerpt.strip() if log_excerpt.strip() else "(not provided)"

    return f"""You are evaluating the accuracy of an AI-based fault attribution system for \
distributed ML training.

## Ground truth (injected fault)
- Fault type : {fault_type}
- Injected rank : {rank}  (global rank index, 0-based; total ranks = {total_ranks})
- Injected node : {node_index}  (using {gpus_per_node} GPUs per node)
- Injected segment : ranks {segment_start}-{segment_end}  (using segment size {segment_size})
- Injected at iteration : {iter_}
- Cluster : {nodes} nodes × {gpus_per_node} GPUs = {total_ranks} total ranks

## Expected correct behavior
- restart_decision should be : {expected_restart}
  Rationale: {restart_rationale}
- Rank {rank} should appear in Primary issues as the root cause
- FR scope scoring:
  - "rank" if FR points directly to rank {rank}
  - "node" if FR does not isolate rank {rank} but correctly narrows to node {node_index}
  - "segment" if FR does not isolate rank {rank} or node {node_index} but correctly narrows to segment ranks {segment_start}-{segment_end}
  - "false" if FR points elsewhere or is not useful
  - "no_dumps" if there is no actionable FR output

## Raw job log (filtered, last 400 lines)
{log_section}

## Log attribution output (from nvrx_logsage)
{log_output if log_output.strip() else "(no log output — analyzer produced no output)"}

## FR (flight recorder) analysis output (from CollectiveAnalyzer)
{fr_section}

## Scoring instructions
Score each dimension below. Use only the values listed for each.

1. **restart_correct** — Is the restart decision in the log output correct for {fault_type}?
   Values: "true" | "false" | "N/A" (if log output is empty or unparseable)

2. **rank_primary** — Is rank {rank} identified as the PRIMARY root cause (in Primary issues)?
   Values: "true" | "false" | "partial" (rank mentioned but only as secondary/collateral)

3. **rank_any** — Is rank {rank} mentioned anywhere in the log attribution output?
   Values: "true" | "false"

4. **fault_described** — Does the log output correctly describe the nature of the fault
   (e.g., GPU hang, segfault, signal kill) appropriate for {fault_type}?
   Values: "true" | "false" | "partial" (category right but specifics wrong)

5. **fr_rank_correct** — How precise is the FR analysis output?
   Values: "rank" | "node" | "segment" | "false" | "no_dumps"
   Use "rank" only if rank {rank} is explicitly implicated.
   Use "node" only if the FR output narrows correctly to node {node_index} but not the exact rank.
   Use "segment" only if the FR output narrows correctly to segment ranks {segment_start}-{segment_end} but not the exact node or rank.
   Use "false" if the FR output points somewhere else, is misleading, or does not narrow correctly.
   Use "no_dumps" if there is no actionable FR output.

6. **notes** — One concise sentence summarizing the main gap or confirming correctness.

Respond ONLY with a JSON object — no markdown, no explanation outside the JSON:
{{
  "restart_correct": "...",
  "rank_primary": "...",
  "rank_any": "...",
  "fault_described": "...",
  "fr_rank_correct": "...",
  "notes": "..."
}}"""


def score(args):
    args.run_valid = args.run_valid.lower() == "true"
    api_key = os.getenv("JUDGE_API_KEY", "").strip()
    if not api_key:
        judge_key_file = os.getenv("JUDGE_API_KEY_FILE", "").strip()
        if judge_key_file:
            try:
                with open(judge_key_file, encoding="utf-8") as f:
                    api_key = f.read().strip()
            except OSError:
                api_key = ""
    if not api_key:
        api_key = load_nvidia_api_key()
    if not api_key:
        raise ValueError(
            "Judge API key not found. Set JUDGE_API_KEY/JUDGE_API_KEY_FILE, "
            "or NVIDIA_API_KEY/NVIDIA_API_KEY_FILE, or create ~/.nvidia_api_key"
        )

    base_url = os.getenv("JUDGE_BASE_URL", "").strip() or args.base_url

    llm = ChatOpenAI(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_completion_tokens=512,
    )

    log_excerpt = load_log_excerpt(args.log_path) if args.log_path else ""

    prompt_or_result = build_judge_prompt(
        fault_type=args.fault_type,
        rank=args.rank,
        iter_=args.iter,
        nodes=args.nodes,
        run_valid=args.run_valid,
        log_output=args.log_output,
        fr_output=args.fr_output,
        log_excerpt=log_excerpt,
        gpus_per_node=args.gpus_per_node,
        segment_size=args.segment_size,
    )

    # build_judge_prompt returns a dict directly for invalid runs (no LLM call needed)
    if isinstance(prompt_or_result, dict):
        return prompt_or_result

    response = llm.invoke(prompt_or_result)
    text = response.content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()

    result = json.loads(text)
    total_ranks = args.nodes * args.gpus_per_node
    result["fr_rank_correct"] = normalize_fr_rank_correct(
        raw_label=result.get("fr_rank_correct", ""),
        fr_output=args.fr_output,
        rank=args.rank,
        total_ranks=total_ranks,
        gpus_per_node=args.gpus_per_node,
        segment_size=args.segment_size,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="LLM judge for fault attribution scoring")
    parser.add_argument("--fault-type", required=True, help="Injected fault type")
    parser.add_argument("--rank", type=int, required=True, help="Injected global rank")
    parser.add_argument("--iter", type=int, required=True, help="Injected iteration")
    parser.add_argument("--nodes", type=int, required=True, help="Node count")
    parser.add_argument(
        "--run-valid",
        default="true",
        help="'true' if training reached the fault injection point, 'false' otherwise",
    )
    parser.add_argument("--log-path", default="", help="Path to the raw job log file")
    parser.add_argument("--log-output", default="", help="Raw stdout from nvrx_logsage")
    parser.add_argument("--fr-output", default="no_dumps", help="Raw text from CollectiveAnalyzer")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge LLM model")
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL, help="API base URL")
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=DEFAULT_GPUS_PER_NODE,
        help="GPUs per node for rank-to-node mapping",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=DEFAULT_FR_SEGMENT_SIZE,
        help="Ranks per segment for coarse FR scope scoring",
    )
    parser.add_argument(
        "--rack-size",
        type=int,
        default=None,
        help="Deprecated alias for --segment-size",
    )
    args = parser.parse_args()
    if args.rack_size is not None:
        args.segment_size = args.rack_size

    try:
        result = score(args)
        print(json.dumps(result))
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        print(json.dumps({"notes": f"judge_failed: {exc}"}))
        sys.exit(0)  # non-fatal — caller handles missing keys gracefully


if __name__ == "__main__":
    if not logging.root.handlers:
        logging.basicConfig(level=logging.WARNING)
    main()
