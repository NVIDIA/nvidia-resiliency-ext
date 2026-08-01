#!/bin/bash
# Drain-node PRODUCER for InJob shared-rendezvous job-arrays. Backgrounded on ARRAY TASK 0's node0
# (batch step, OUTSIDE the container, where the SLURM client lives). ONE producer for the whole array
# polls SLURM for DRAIN-flagged nodes and publishes one file per array task under a shared Lustre dir.
# Pure bash + coreutils + slurm CLI (no python, no container). Cheap at scale: 1 sinfo + 1 squeue per
# interval (unchanged at 12K/192); `scontrol show hostnames` is a LOCAL expansion (no RPC); `grep
# -Fxf` intersects each task's nodes with the drained set in memory (no per-task scratch).
#
# Output -- ${STATE_DIR}/task_<K>.drained, one per running task, segment-scoped (K's 16-node NVLink domain):
#   NON-EMPTY => a node in task K's segment is drained
#   EMPTY     => producer ran, segment clean
#   MISSING   => not polled yet
#
# CONSUMER (separate integration, NOT this script): each task's FIRST node reads its OWN file
# in-container, folded into ft_launcher's nvhcd node-health-check AT RENDEZVOUS -- a single read, not a
# poll. NON-EMPTY => that node reports UNHEALTHY => ft_launcher exits failure => SLURM terminates the
# array task => cold spare swaps in (MISSING fails open). The exit must be non-zero but NOT the
# deployment's no-restart code (e.g. restart_matrix job_array.sh uses 93 to cancel the singleton
# chain); task 0's exit is handled by SINGLETON restart, not requeue. First-node-only => no Lustre read
# storm (SLURM tears down the other segment nodes). An out-of-container consumer poller was rejected:
# can't set a controlled exit code, and would fight ft_launcher's restart logic.
#
# Ships with nvidia-resiliency-ext (examples/fault_tolerance/deployment/slurm). A deployment's array
# coordinator backgrounds it on task 0; e.g. from a batch script:
#   STATE_DIR="<shared-lustre-dir>/drained" ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}" \
#     bash "$(dirname "$0")/drain_poller.sh" &
#   DRAIN_POLLER_PID=$!; trap 'kill "${DRAIN_POLLER_PID}" 2>/dev/null' EXIT
#
# Env:
#   ARRAY_JOB_ID   (req) SLURM_ARRAY_JOB_ID of the array to watch
#   STATE_DIR      (req) shared Lustre dir for the output files (created if missing)
#   POLL_INTERVAL  (opt, default 120) seconds between polls; drains aren't latency-critical
#   DRAIN_STATES   (opt, default "drain") sinfo -t filter; the DRAIN flag catches drng + drained
#   ONESHOT        (opt) run exactly one poll pass and exit (testing)
#   SIMULATE_DRAINED_FILE (opt, TEST ONLY) nodenames unioned into the drained set, to exercise a hit
#                  without draining a real node
set -u

: "${ARRAY_JOB_ID:?drain_poller: set ARRAY_JOB_ID (SLURM_ARRAY_JOB_ID)}"
: "${STATE_DIR:?drain_poller: set STATE_DIR (shared Lustre control dir)}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
DRAIN_STATES="${DRAIN_STATES:-drain}"

mkdir -p "${STATE_DIR}"
WORK="${STATE_DIR}/.work"          # scratch on the SAME fs as STATE_DIR so mv is an atomic rename
mkdir -p "${WORK}"

log() { printf '[drain_poller %s] %s\n' "$(date +'%H:%M:%S')" "$*" >&2; }

# Publish $2 to $1 only if content changed (avoids Lustre churn); returns 0 iff it changed, so the
# caller can log the transition.
publish_if_changed() {
    local dst="$1" tmp="$2"
    if [[ -f "$dst" ]] && cmp -s "$tmp" "$dst"; then rm -f "$tmp"; return 1; fi
    mv -f "$tmp" "$dst"       # atomic rename within STATE_DIR's filesystem
    return 0
}

poll_once() {
    # 1) Cluster-wide DRAIN-flagged nodes, one per line (small: only matching nodes are returned).
    #    -N one-line-per-node; %N nodename; sort -u collapses the per-partition duplicate rows.
    local drained="${WORK}/drained.all"
    sinfo -t "${DRAIN_STATES}" -N -h -o '%N' 2>/dev/null | sort -u > "${drained}"
    # TEST hook: union in simulated drained nodes so intersection can be exercised without a real drain.
    if [[ -n "${SIMULATE_DRAINED_FILE:-}" && -f "${SIMULATE_DRAINED_FILE}" ]]; then
        sort -u "${SIMULATE_DRAINED_FILE}" "${drained}" -o "${drained}"
    fi
    local ndrained; ndrained=$(wc -l < "${drained}")

    # 2) node->task map for THIS array: one row per RUNNING task element (%K index, %N its nodelist).
    #    For each task, expand its compact nodelist (slurm parses the ranges) and keep only the nodes
    #    that are in the drained set -- `grep -Fxf` hashes the tiny drained pattern file and matches
    #    exact whole lines (-Fx), so the intersect is in-memory: no per-task scratch, no comm/sort of
    #    the task side. The result is written straight to the publish tmp.
    #    Loop is null-safe: if no task is running yet, the while body simply never executes.
    local ntasks=0 k nl tmp dst had
    while IFS='|' read -r k nl; do
        [[ -z "${k}" || "${k}" == "N/A" ]] && continue
        ntasks=$((ntasks+1))
        dst="${STATE_DIR}/task_${k}.drained"; tmp="${WORK}/task_${k}.drained.tmp"
        had=0; [[ -s "${dst}" ]] && had=1
        scontrol show hostnames "${nl}" 2>/dev/null | grep -Fxf "${drained}" > "${tmp}" || true
        # Log only real transitions (drain appears / clears); steady state is silent. The initial
        # missing->empty creation is not a "clear" (had=0), so startup stays quiet too.
        if publish_if_changed "${dst}" "${tmp}"; then
            if [[ -s "${dst}" ]]; then log "task ${k} DRAINED: $(tr '\n' ' ' < "${dst}")"
            elif (( had )); then log "task ${k} cleared"; fi
        fi
    done < <(squeue -j "${ARRAY_JOB_ID}" -r -h -t R -o '%K|%N' 2>/dev/null)

    # One-time liveness line after the first successful poll; silent thereafter.
    if [[ -n "${FIRST:-}" ]]; then log "watching tasks=${ntasks} drained_cluster=${ndrained}"; FIRST=; fi
}

log "start array=${ARRAY_JOB_ID} state_dir=${STATE_DIR} interval=${POLL_INTERVAL}s states=${DRAIN_STATES}"
FIRST=1
while :; do
    poll_once
    [[ -n "${ONESHOT:-}" ]] && { log "oneshot done"; break; }
    sleep "${POLL_INTERVAL}"
done
