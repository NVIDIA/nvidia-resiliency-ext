#!/bin/bash
# Keep a Slurm-hosted attrsvc allocation alive across transient process exits.

set -u

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

BACKOFF_SPEC="${NVRX_ATTRSVC_SUPERVISOR_BACKOFF_SECONDS:-1 2 5 10 20}"
STABLE_SECONDS="${NVRX_ATTRSVC_SUPERVISOR_STABLE_SECONDS:-300}"
read -r -a BACKOFF_SECONDS <<< "${BACKOFF_SPEC}"

if [[ ${#BACKOFF_SECONDS[@]} -eq 0 ]]; then
    echo "ERROR: NVRX_ATTRSVC_SUPERVISOR_BACKOFF_SECONDS must not be empty" >&2
    exit 2
fi
for index in "${!BACKOFF_SECONDS[@]}"; do
    delay=${BACKOFF_SECONDS[$index]}
    if [[ ! "${delay}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid attrsvc supervisor backoff value: ${delay}" >&2
        exit 2
    fi
    BACKOFF_SECONDS[$index]=$((10#${delay}))
done
if [[ ! "${STABLE_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NVRX_ATTRSVC_SUPERVISOR_STABLE_SECONDS must be a positive integer" >&2
    exit 2
fi
STABLE_SECONDS=$((10#${STABLE_SECONDS}))
if (( STABLE_SECONDS == 0 )); then
    echo "ERROR: NVRX_ATTRSVC_SUPERVISOR_STABLE_SECONDS must be greater than zero" >&2
    exit 2
fi

child_pid=""
backoff_pid=""
shutdown_requested=0

supervisor_log() {
    printf '%s attrsvc-supervisor: %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

request_shutdown() {
    local signal_name="$1"
    shutdown_requested=1
    supervisor_log "received ${signal_name}; stopping attrsvc"
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
    fi
    if [[ -n "${backoff_pid}" ]] && kill -0 "${backoff_pid}" 2>/dev/null; then
        kill -TERM "${backoff_pid}" 2>/dev/null || true
    fi
}

stop_and_reap_child() {
    if [[ -z "${child_pid}" ]]; then
        return
    fi
    while kill -0 "${child_pid}" 2>/dev/null; do
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    done
    wait "${child_pid}" 2>/dev/null || true
    child_pid=""
}

stop_and_reap_backoff() {
    if [[ -z "${backoff_pid}" ]]; then
        return
    fi
    while kill -0 "${backoff_pid}" 2>/dev/null; do
        kill -TERM "${backoff_pid}" 2>/dev/null || true
        wait "${backoff_pid}" 2>/dev/null || true
    done
    wait "${backoff_pid}" 2>/dev/null || true
    backoff_pid=""
}

exit_if_shutdown_requested() {
    if [[ ${shutdown_requested} -ne 1 ]]; then
        return
    fi
    stop_and_reap_child
    stop_and_reap_backoff
    supervisor_log "attrsvc stopped for supervisor shutdown"
    exit 0
}

trap 'request_shutdown SIGTERM' SIGTERM
trap 'request_shutdown SIGINT' SIGINT
trap 'request_shutdown SIGUSR1' SIGUSR1
trap 'request_shutdown SIGUSR2' SIGUSR2

supervisor_log \
    "starting command=$1 restart_backoff_seconds=${BACKOFF_SPEC// /,} stable_reset_seconds=${STABLE_SECONDS}"

restart_index=0
while true; do
    exit_if_shutdown_requested
    started_at=$(date +%s)
    # A trap is deferred while command substitution runs. Recheck before launch.
    exit_if_shutdown_requested
    "$@" &
    child_pid=$!
    # A signal can also arrive after launch but before child_pid is recorded.
    exit_if_shutdown_requested
    supervisor_log "attrsvc started pid=${child_pid}"

    wait "${child_pid}"
    child_status=$?
    exit_if_shutdown_requested
    child_pid=""

    ended_at=$(date +%s)
    # A trap is deferred while command substitution runs. Do not schedule a restart
    # after shutdown was requested while collecting the process end time.
    exit_if_shutdown_requested
    runtime_seconds=$((ended_at - started_at))
    if (( runtime_seconds >= STABLE_SECONDS )); then
        supervisor_log \
            "attrsvc ran for ${runtime_seconds}s; resetting consecutive restart count"
        restart_index=0
    fi

    if (( restart_index >= ${#BACKOFF_SECONDS[@]} )); then
        supervisor_log \
            "restart limit reached after ${#BACKOFF_SECONDS[@]} restart attempts; "\
            "last_status=${child_status}"
        if [[ ${child_status} -eq 0 ]]; then
            exit 1
        fi
        exit "${child_status}"
    fi

    delay=${BACKOFF_SECONDS[$restart_index]}
    attempt=$((restart_index + 1))
    restart_index=$((restart_index + 1))
    supervisor_log \
        "attrsvc exited unexpectedly status=${child_status} runtime_seconds=${runtime_seconds}; "\
        "restart_attempt=${attempt}/${#BACKOFF_SECONDS[@]} delay_seconds=${delay}"

    exit_if_shutdown_requested
    sleep "${delay}" &
    backoff_pid=$!
    # A signal can arrive after launch but before backoff_pid is recorded.
    exit_if_shutdown_requested
    wait "${backoff_pid}" 2>/dev/null || true
    exit_if_shutdown_requested
    backoff_pid=""
done
