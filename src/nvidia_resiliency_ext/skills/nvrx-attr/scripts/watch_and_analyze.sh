#!/bin/bash
# watch_and_analyze.sh
# Poll SLURM for job completions from a fault-injection session tracking file,
# run log-analysis and fr-analysis on each completed job, then call the LLM judge
# (score_attribution.py) to score each attribution dimension.
#
# Usage:
#   bash scripts/watch_and_analyze.sh <TRACKING_FILE>

set -euo pipefail

TRACKING_FILE="${1:?Usage: $0 <tracking_file.tsv>}"
POLL_INTERVAL=30

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "${SCRIPT_DIR}")"
NVRX_SRC_DIR="$(cd "${SKILL_DIR}/../../.." && pwd)"

LOGSAGE_PY="${SKILL_DIR}/log-analysis/scripts/nvrx_logsage.py"
SCORE_PY="${SCRIPT_DIR}/score_attribution.py"

# Ensure nvidia_resiliency_ext is importable from source tree
export PYTHONPATH="${NVRX_SRC_DIR}${PYTHONPATH:+:$PYTHONPATH}"

strip_injection_markers() {
    local input_log="$1"
    local output_log="$2"
    grep -v -E 'FAULT INJECTION|nvidia_resiliency_ext\.shared_utils\.inject_fault' \
        "${input_log}" > "${output_log}" 2>/dev/null || true
}

REPORT_FILE="${TRACKING_FILE%.tsv}_report.md"
DONE_JOBS_FILE="${TRACKING_FILE%.tsv}_done.txt"

touch "${DONE_JOBS_FILE}"

cat > "${REPORT_FILE}" <<'EOF'
# Fault Injection Experiment Report

| # | FAULT_TYPE | NODES | RANK | ITER | JOB_ID | STATE | run_valid | restart_correct | rank_primary | rank_any | fault_described | fr_rank_correct | judge_notes |
|---|------------|-------|------|------|--------|-------|-----------|-----------------|--------------|----------|-----------------|-----------------|-------------|
EOF

echo ">>> Watching tracking file: ${TRACKING_FILE}"
echo ">>> Report: ${REPORT_FILE}"
echo ">>> Polling every ${POLL_INTERVAL}s ..."

TOTAL=$(tail -n +2 "${TRACKING_FILE}" | wc -l)
EXP_NUM=0

while true; do
    PENDING=0

    while IFS=$'\t' read -r JOB_ID FAULT_TYPE RANK ITER NODES EXPERIMENT_DIR; do
        # Skip already-analyzed jobs
        if grep -q "^${JOB_ID}$" "${DONE_JOBS_FILE}" 2>/dev/null; then
            continue
        fi

        # Check job state
        STATE=$(scontrol show job "${JOB_ID}" 2>/dev/null \
            | grep -oP 'JobState=\K\S+' || echo "UNKNOWN")

        case "${STATE}" in
            RUNNING|PENDING|COMPLETING)
                PENDING=$((PENDING + 1))
                continue
                ;;
            COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL)
                ;;
            *)
                # Job left the queue — treat as done
                ;;
        esac

        EXP_NUM=$((EXP_NUM + 1))
        echo ""
        echo ">>> [${EXP_NUM}/${TOTAL}] Analyzing: ${FAULT_TYPE} n=${NODES} rank=${RANK} iter=${ITER} job=${JOB_ID} state=${STATE}"

        # ---- Log analysis ----
        LOG_GLOB="${EXPERIMENT_DIR}/logs/slurm/${JOB_ID}.*.1.main_workload.log"
        LOG_FILE=$(ls ${LOG_GLOB} 2>/dev/null | head -1 || true)
        LOG_OUT=""

        # ---- Check run validity: did the fault actually arm/fire? ----
        # The fault injector prints:
        #   [timestamp] FAULT INJECTION: Rank R will inject fault TYPE at timestamp
        RUN_VALID="false"
        STRIPPED_LOG=""
        if [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then
            echo "    log: ${LOG_FILE}"
            if grep -q "FAULT INJECTION" "${LOG_FILE}" 2>/dev/null; then
                RUN_VALID="true"
            fi
            echo "    run_valid: ${RUN_VALID}"

            # Strip fault-injection markers so neither nvrx_logsage nor the judge
            # can see which rank/fault was injected — evaluation must be fair.
            # This removes:
            # - scheduler lines from megatron.core.fault_injector ("FAULT INJECTION")
            # - direct fault-tool log lines from nvidia_resiliency_ext.shared_utils.inject_fault
            STRIPPED_LOG=$(mktemp /tmp/fi_log_stripped.XXXXXX)
            strip_injection_markers "${LOG_FILE}" "${STRIPPED_LOG}"

            # nvrx_logsage.py prints 5 newline-joined fields to stdout:
            #   line 1: restart_decision
            #   line 2: error_explanation  (often empty)
            #   line 3+: attribution_text  (multi-line, starts with "Attribution:")
            #   then: additional_detail    (often empty)
            #   last line: checkpoint_saved ("True" / "False")
            LOG_OUT=$(python3 "${LOGSAGE_PY}" \
                --log-path "${STRIPPED_LOG}" \
                --exclude_nvrx_logs 2>/dev/null || echo "")
            LOG_RESTART=$(echo "${LOG_OUT}" | head -1)
            echo "    restart_decision: ${LOG_RESTART:-<empty>}"
        else
            echo "    WARN: no log file at ${LOG_GLOB}"
            echo "    run_valid: false (no log)"
        fi

        # ---- FR analysis (only when run is valid) ----
        FR_DIR="${EXPERIMENT_DIR}/checkpoints"
        FR_OUT="no_dumps"

        if [[ "${RUN_VALID}" == "true" ]] && ls "${FR_DIR}"/_dump_* 2>/dev/null | grep -q .; then
            echo "    FR dumps: $(ls "${FR_DIR}"/_dump_* 2>/dev/null | wc -l) files"
            FR_OUT=$(python3 -c "
import sys, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, '${NVRX_SRC_DIR}')
from nvidia_resiliency_ext.attribution.trace_analyzer.fr_attribution import CollectiveAnalyzer
try:
    ca = CollectiveAnalyzer({'fr_path': '${FR_DIR}'})
    results = ca.run_sync({'fr_path': '${FR_DIR}'})
    if results:
        result_data = results[0]
        if isinstance(result_data, dict):
            text = result_data.get('analysis_text', '')
            ranks = result_data.get('hanging_ranks', '')
            if text:
                print(text)
            if ranks:
                print(ranks)
        else:
            print(str(result_data))
    else:
        print('no results')
except Exception as e:
    print('error: ' + str(e), file=sys.stderr)
    print('no_dumps')
" 2>/dev/null || echo "no_dumps")
        elif [[ "${RUN_VALID}" == "false" ]]; then
            FR_OUT="run_invalid"
            echo "    FR analysis skipped (run did not reach fault injection point)"
        fi

        # ---- LLM judge scoring ----
        echo "    scoring with judge..."
        SCORE_JSON=$(python3 "${SCORE_PY}" \
            --fault-type "${FAULT_TYPE}" \
            --rank "${RANK}" \
            --iter "${ITER}" \
            --nodes "${NODES}" \
            --run-valid "${RUN_VALID}" \
            --log-path "${STRIPPED_LOG:-}" \
            --log-output "${LOG_OUT}" \
            --fr-output "${FR_OUT}" 2>/dev/null || echo '{"notes":"judge_failed"}')

        # Clean up temp stripped log
        [[ -n "${STRIPPED_LOG}" && -f "${STRIPPED_LOG}" ]] && rm -f "${STRIPPED_LOG}"

        _get() { echo "${SCORE_JSON}" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('$1','N/A'))" 2>/dev/null || echo "N/A"; }

        RESTART_CORRECT=$(_get restart_correct)
        RANK_PRIMARY=$(_get rank_primary)
        RANK_ANY=$(_get rank_any)
        FAULT_DESC=$(_get fault_described)
        FR_RANK=$(_get fr_rank_correct)
        JUDGE_NOTES=$(_get notes)

        echo "    run_valid=${RUN_VALID}  restart_correct=${RESTART_CORRECT}  rank_primary=${RANK_PRIMARY}  rank_any=${RANK_ANY}  fault_described=${FAULT_DESC}  fr_rank=${FR_RANK}"
        echo "    judge: ${JUDGE_NOTES}"

        # Append to report
        printf "| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
            "${EXP_NUM}" "${FAULT_TYPE}" "${NODES}" "${RANK}" "${ITER}" \
            "${JOB_ID}" "${STATE}" "${RUN_VALID}" \
            "${RESTART_CORRECT}" "${RANK_PRIMARY}" "${RANK_ANY}" \
            "${FAULT_DESC}" "${FR_RANK}" \
            "${JUDGE_NOTES}" >> "${REPORT_FILE}"

        echo "${JOB_ID}" >> "${DONE_JOBS_FILE}"

    done < <(tail -n +2 "${TRACKING_FILE}")

    DONE_COUNT=$(wc -l < "${DONE_JOBS_FILE}")
    echo "$(date '+%H:%M:%S') >>> ${DONE_COUNT}/${TOTAL} done, ${PENDING} still running"

    if [[ ${DONE_COUNT} -ge ${TOTAL} ]]; then
        break
    fi

    sleep "${POLL_INTERVAL}"
done

echo ""
echo ">>> All ${TOTAL} experiments analyzed."
echo ">>> Report: ${REPORT_FILE}"
echo ""
cat "${REPORT_FILE}"
