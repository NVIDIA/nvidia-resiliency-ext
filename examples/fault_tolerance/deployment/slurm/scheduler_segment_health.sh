#!/bin/bash
# Publish scheduler-unavailable Slurm array tasks for SegmentHealthCheck.
#
# Source this file from array task 0's sbatch control flow, outside the workload
# environment, call scheduler_segment_health_configure once, call
# scheduler_segment_health_poll_once periodically, and call
# scheduler_segment_health_cleanup on exit. The caller owns cadence; direct
# execution performs the complete lifecycle for testing and manual validation.
#
# The producer publishes one per-task control file and a centralized best-effort
# JSONL history:
#
#   segment_health_check.<array_job_id>.<task_id>
#   segment_health_check.<array_job_id>.<task_id>.inactive
#   segment_health_check_history.<array_job_id>.log
#
# A non-empty current file excludes that task. A missing or zero-byte current
# file does not. Its canonical CSV content is diagnostic context; consumers use
# only file size.
#
# Environment:
#   SLURM_ARRAY_JOB_ID           Required numeric Slurm array parent job ID.
#   NVRX_SEGMENT_HEALTH_CHECK_DIR Required absolute shared artifact directory.
#   SLURM_JOB_PARTITION          Required partition containing the array job.
#   NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS
#                                Clear exclusions after this long without a
#                                complete poll. Default: 1800 seconds.
#   NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT
#                                sinfo/squeue timeout. Default: 30 seconds.

# Hostlist expansion is local, repeated per task, and should finish quickly.
_SCHEDULER_SEGMENT_HEALTH_HOSTLIST_EXPANSION_TIMEOUT=5

_scheduler_segment_health_validate_positive_integer() {
    local name="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'scheduler_segment_health: %s must be a positive integer: %s\n' \
            "${name}" "${value}" >&2
        return 2
    fi
}

_scheduler_segment_health_log() {
    printf '[scheduler_segment_health %s] %s\n' "$(date +'%H:%M:%S')" "$*" >&2
}

_scheduler_segment_health_trim_whitespace() {
    local value="$1"

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

_scheduler_segment_health_timestamp_utc() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

scheduler_segment_health_configure() {
    if [[ -n "${_SCHEDULER_SEGMENT_HEALTH_CONFIGURED:-}" ]]; then
        return 0
    fi
    if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
        printf 'scheduler_segment_health: set SLURM_ARRAY_JOB_ID\n' >&2
        return 2
    fi
    if [[ -z "${NVRX_SEGMENT_HEALTH_CHECK_DIR:-}" ]]; then
        printf 'scheduler_segment_health: set NVRX_SEGMENT_HEALTH_CHECK_DIR\n' >&2
        return 2
    fi
    if [[ -z "${SLURM_JOB_PARTITION:-}" ]]; then
        printf 'scheduler_segment_health: set SLURM_JOB_PARTITION\n' >&2
        return 2
    fi
    if [[ "${NVRX_SEGMENT_HEALTH_CHECK_DIR}" != /* ]]; then
        printf 'scheduler_segment_health: NVRX_SEGMENT_HEALTH_CHECK_DIR must be absolute: %s\n' \
            "${NVRX_SEGMENT_HEALTH_CHECK_DIR}" >&2
        return 2
    fi
    if [[ ! "${SLURM_ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
        printf 'scheduler_segment_health: SLURM_ARRAY_JOB_ID must be numeric: %s\n' \
            "${SLURM_ARRAY_JOB_ID}" >&2
        return 2
    fi

    NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS="${NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS:-1800}"
    NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT="${NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT:-30}"
    _scheduler_segment_health_validate_positive_integer \
        NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS "${NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS}" || return $?
    _scheduler_segment_health_validate_positive_integer \
        NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT "${NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT}" || return $?

    mkdir -p "${NVRX_SEGMENT_HEALTH_CHECK_DIR}" || return 1
    _SCHEDULER_SEGMENT_HEALTH_WORK=$(mktemp -d \
        "${TMPDIR:-/tmp}/scheduler_segment_health.${SLURM_ARRAY_JOB_ID}.XXXXXX") || return 1
    _SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE="${_SCHEDULER_SEGMENT_HEALTH_WORK}/published-task-state.map"
    _SEGMENT_HEALTH_UNAVAILABLE_NODES_FILE="${_SCHEDULER_SEGMENT_HEALTH_WORK}/unavailable.nodes"
    _SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE="${_SCHEDULER_SEGMENT_HEALTH_WORK}/desired-task-state.map"
    _SCHEDULER_SEGMENT_HEALTH_UNAVAILABLE_STATES="drain,down,fail,no_respond"
    _SCHEDULER_SEGMENT_HEALTH_LEDGER_PATH="${NVRX_SEGMENT_HEALTH_CHECK_DIR}/segment_health_check_history.${SLURM_ARRAY_JOB_ID}.log"
    _SCHEDULER_SEGMENT_HEALTH_LAST_COMPLETE_POLL_AT="${SECONDS}"
    _SCHEDULER_SEGMENT_HEALTH_MISSING_TASK_IDS="|"
    _scheduler_segment_health_log \
        "configured array=${SLURM_ARRAY_JOB_ID} partition=${SLURM_JOB_PARTITION} output_dir=${NVRX_SEGMENT_HEALTH_CHECK_DIR}"
    _scheduler_segment_health_log \
        "stale_decision=${NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS}s states=${_SCHEDULER_SEGMENT_HEALTH_UNAVAILABLE_STATES}"
    _SCHEDULER_SEGMENT_HEALTH_CONFIGURED=1
}

scheduler_segment_health_cleanup() {
    local work="${_SCHEDULER_SEGMENT_HEALTH_WORK:-}"

    # Best-effort process-lifetime cleanup. An unexpected exit may leave this
    # one small directory for the node's normal /tmp cleanup policy.
    [[ -n "${work}" ]] || return 0
    if ! rm -rf -- "${work}"; then
        _scheduler_segment_health_log "could not remove temporary workspace ${work}"
        return 1
    fi
    unset _SCHEDULER_SEGMENT_HEALTH_WORK
    unset _SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE
    unset _SEGMENT_HEALTH_UNAVAILABLE_NODES_FILE
    unset _SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE
    unset _SCHEDULER_SEGMENT_HEALTH_CONFIGURED
}

_scheduler_segment_health_read_file_exact() {
    local path="$1"
    local captured

    # Command substitution strips trailing newlines; the sentinel preserves
    # the exact file value, including an empty marker.
    captured=$(
        cat -- "${path}"
        status=$?
        printf '\036'
        exit "${status}"
    ) || return 1
    _SCHEDULER_SEGMENT_HEALTH_FILE_VALUE="${captured%$'\036'}"
}

_scheduler_segment_health_has_complete_lines() {
    local path="$1"
    local last_byte

    [[ -s "${path}" ]] || return 0
    # Command substitution strips a final newline, yielding an empty value.
    last_byte=$(tail -c 1 "${path}") || return 1
    [[ -z "${last_byte}" ]]
}

_scheduler_segment_health_current_path() {
    printf '%s/segment_health_check.%s.%s' \
        "${NVRX_SEGMENT_HEALTH_CHECK_DIR}" "${SLURM_ARRAY_JOB_ID}" "$1"
}

_scheduler_segment_health_inactive_path() {
    printf '%s/segment_health_check.%s.%s.inactive' \
        "${NVRX_SEGMENT_HEALTH_CHECK_DIR}" "${SLURM_ARRAY_JOB_ID}" "$1"
}

_scheduler_segment_health_state_lookup() {
    local source="$1"
    local task_id="$2"
    local line
    local rest

    # Read task|status|value and expose the status and value to the caller.
    # K means known canonical state; U means unreadable or noncanonical state.
    line=$(grep -m 1 "^${task_id}|" "${source}" 2>/dev/null) || return 1
    rest="${line#*|}"
    _SCHEDULER_SEGMENT_HEALTH_STATE_STATUS="${rest%%|*}"
    _SCHEDULER_SEGMENT_HEALTH_STATE_VALUE="${rest#*|}"
    return 0
}

_scheduler_segment_health_snapshot_published_state() {
    local path
    local name
    local task_id
    local value

    # Per-task files are the source of truth. Read each once into a temporary
    # task|status|value snapshot that remains stable for this poll. K marks a
    # readable canonical value; U marks state that must be reconciled.
    : >"${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}" || return 1
    for path in "${NVRX_SEGMENT_HEALTH_CHECK_DIR}/segment_health_check.${SLURM_ARRAY_JOB_ID}."*; do
        [[ -e "${path}" || -L "${path}" ]] || continue
        name="${path##*/}"
        task_id="${name#segment_health_check.${SLURM_ARRAY_JOB_ID}.}"
        if [[ ! "${task_id}" =~ ^[0-9]+$ ]]; then
            continue
        fi
        if _scheduler_segment_health_read_file_exact "${path}"; then
            value="${_SCHEDULER_SEGMENT_HEALTH_FILE_VALUE}"
            if [[ -z "${value}" || "${value}" =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]]; then
                printf '%s|K|%s\n' "${task_id}" "${value}" >>"${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}" || return 1
            else
                _scheduler_segment_health_log "current artifact has a noncanonical value task=${task_id}; marking it unknown"
                printf '%s|U|\n' "${task_id}" >>"${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}" || return 1
            fi
        else
            _scheduler_segment_health_log "could not read current artifact ${path}; marking it unknown"
            printf '%s|U|\n' "${task_id}" >>"${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}" || return 1
        fi
    done
    LC_ALL=C sort -t '|' -k1,1n -o "${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}" \
        "${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}"
}

_scheduler_segment_health_append_audit() {
    local event="$1"
    local task_id="$2"
    local nodes="$3"
    local reason="$4"
    local observed_at="$5"
    local observed_at_json="null"
    local detail=""

    if [[ -z "${observed_at}" ]]; then
        _scheduler_segment_health_log \
            "history timestamp unavailable task=${task_id} event=${event}"
    else
        printf -v observed_at_json '"%s"' "${observed_at}"
    fi

    if [[ -n "${nodes}" ]]; then
        printf -v detail ',"nodes":"%s"' "${nodes}"
    elif [[ -n "${reason}" ]]; then
        printf -v detail ',"reason":"%s"' "${reason}"
    fi

    if ! printf '{"event":"%s","task_id":%s%s,"observed_at":%s}\n' \
        "${event}" "${task_id}" "${detail}" "${observed_at_json}" \
        >>"${_SCHEDULER_SEGMENT_HEALTH_LEDGER_PATH}"; then
        _scheduler_segment_health_log "could not append history event task=${task_id} event=${event}"
    fi
    return 0
}

_scheduler_segment_health_set_current_value() (
    local task_id="$1"
    local value="$2"
    local path

    path=$(_scheduler_segment_health_current_path "${task_id}") || return 1
    if [[ -z "${value}" ]]; then
        : >"${path}"
        return $?
    fi

    # FD 9 is an arbitrary non-standard descriptor scoped to this subshell.
    # Keep the file open without truncating so readers never observe an empty
    # marker during replacement.
    if [[ -e "${path}" ]]; then
        exec 9<>"${path}" || return 1
    else
        exec 9>"${path}" || return 1
    fi
    if ! printf '%s' "${value}" >&9; then
        exec 9>&-
        return 1
    fi
    # Remove any suffix left by a longer prior value.
    if ! truncate -s "${#value}" "${path}"; then
        exec 9>&-
        return 1
    fi
    exec 9>&-
    return 0
)

_scheduler_segment_health_reconcile_after_write_failure() {
    local task_id="$1"
    local desired="$2"
    local path
    local actual

    path=$(_scheduler_segment_health_current_path "${task_id}") || return 1
    if [[ -f "${path}" ]] && _scheduler_segment_health_read_file_exact "${path}"; then
        actual="${_SCHEDULER_SEGMENT_HEALTH_FILE_VALUE}"
        [[ "${actual}" == "${desired}" ]]
        return $?
    fi
    return 1
}

_scheduler_segment_health_publish_current_value() {
    local task_id="$1"
    local desired="$2"

    if _scheduler_segment_health_set_current_value "${task_id}" "${desired}"; then
        return 0
    fi

    _scheduler_segment_health_log "could not publish current artifact task=${task_id}; reconciling its actual value"
    _scheduler_segment_health_reconcile_after_write_failure "${task_id}" "${desired}"
}

_scheduler_segment_health_move_task_inactive() {
    local task_id="$1"
    local source
    local destination

    source=$(_scheduler_segment_health_current_path "${task_id}") || return 1
    destination=$(_scheduler_segment_health_inactive_path "${task_id}") || return 1
    if [[ ! -e "${source}" && ! -L "${source}" ]]; then
        return 0
    fi
    if [[ -e "${destination}" ]]; then
        _scheduler_segment_health_log "replacing prior inactive artifact task=${task_id}"
    fi
    # Leave the active control namespace while retaining the last observation.
    if mv -f "${source}" "${destination}"; then
        return 0
    fi

    _scheduler_segment_health_log "could not move task artifact inactive task=${task_id}"
    return 1
}

_scheduler_segment_health_derive_desired_state() {
    # Build an all-or-nothing decision snapshot without modifying published
    # task artifacts. First query unavailable nodes in the job partition. If
    # unavailable nodes or prior exclusions exist, query running array tasks,
    # expand each task's hostlist, and intersect it with the unavailable set.
    # On success, stage two deterministic poll outputs:
    #   unavailable.nodes       one scheduler-unavailable node per line
    #   desired-task-state.map  task_id|comma-separated-unavailable-nodes;
    #                           an empty value means the running task is healthy
    # This function does not modify persistent per-task artifacts; the
    # reconciliation phase consumes these snapshots to set, clear, or retire them.
    # Any incomplete query or malformed row aborts the derivation so the caller
    # can preserve the prior published decision.
    local unavailable_raw="${_SCHEDULER_SEGMENT_HEALTH_WORK}/unavailable.raw"
    local unavailable_staged="${_SCHEDULER_SEGMENT_HEALTH_WORK}/unavailable.staged"
    local taskmap="${_SCHEDULER_SEGMENT_HEALTH_WORK}/taskmap.raw"
    local expanded_raw="${_SCHEDULER_SEGMENT_HEALTH_WORK}/expanded.raw"
    local expanded="${_SCHEDULER_SEGMENT_HEALTH_WORK}/expanded.nodes"
    local matched="${_SCHEDULER_SEGMENT_HEALTH_WORK}/matched.nodes"
    local node
    local row
    local task_id
    local hostlist
    local csv

    : >"${_SEGMENT_HEALTH_UNAVAILABLE_NODES_FILE}"
    : >"${_SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE}"

    # First collect the partition-wide unavailable-node set.
    if ! timeout --kill-after=5s "${NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT}" \
        sinfo --partition="${SLURM_JOB_PARTITION}" \
        --states="${_SCHEDULER_SEGMENT_HEALTH_UNAVAILABLE_STATES}" --Node --noheader --format='%N' \
        >"${unavailable_raw}"; then
        _scheduler_segment_health_log "sinfo failed or timed out; preserving the previous decision"
        return 1
    fi
    if ! _scheduler_segment_health_has_complete_lines "${unavailable_raw}"; then
        _scheduler_segment_health_log "sinfo returned an incomplete row; preserving the previous decision"
        return 1
    fi

    : >"${unavailable_staged}"
    while IFS= read -r node; do
        node=$(_scheduler_segment_health_trim_whitespace "${node}")
        [[ -n "${node}" ]] || continue
        if [[ ! "${node}" =~ ^[A-Za-z0-9._-]+$ ]]; then
            _scheduler_segment_health_log "invalid sinfo node '${node}'; preserving the previous decision"
            return 1
        fi
        printf '%s\n' "${node}" >>"${unavailable_staged}" || return 1
    done <"${unavailable_raw}"
    LC_ALL=C sort -u "${unavailable_staged}" >"${_SEGMENT_HEALTH_UNAVAILABLE_NODES_FILE}" || return 1

    # No unavailable nodes and no prior decisions means no task-map query.
    if [[ ! -s "${_SEGMENT_HEALTH_UNAVAILABLE_NODES_FILE}" && ! -s "${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}" ]]; then
        return 0
    fi

    # Map running array tasks to their nodes only when a decision may change.
    if ! timeout --kill-after=5s "${NVRX_SEGMENT_HEALTH_SLURM_CMD_TIMEOUT}" \
        squeue --jobs="${SLURM_ARRAY_JOB_ID}" --states=RUNNING --array --noheader \
        --format='%K|%N' >"${taskmap}"; then
        _scheduler_segment_health_log "squeue failed or timed out; preserving the previous decision"
        return 1
    fi
    if ! _scheduler_segment_health_has_complete_lines "${taskmap}"; then
        _scheduler_segment_health_log "squeue returned an incomplete row; preserving the previous decision"
        return 1
    fi

    while IFS= read -r row; do
        # Validate and split the squeue task-to-hostlist mapping. Any malformed
        # or duplicate task makes the poll incomplete, so preserve prior state.
        row=$(_scheduler_segment_health_trim_whitespace "${row}")
        [[ -n "${row}" ]] || continue
        if [[ "${row}" != *"|"* || "${row#*|}" == *"|"* ]]; then
            _scheduler_segment_health_log "invalid squeue task row '${row}'; preserving the previous decision"
            return 1
        fi
        task_id=$(_scheduler_segment_health_trim_whitespace "${row%%|*}")
        hostlist=$(_scheduler_segment_health_trim_whitespace "${row#*|}")
        if [[ ! "${task_id}" =~ ^[0-9]+$ || -z "${hostlist}" || "${hostlist}" == *"{"* ]]; then
            _scheduler_segment_health_log "invalid squeue task row '${row}'; preserving the previous decision"
            return 1
        fi
        if grep -q "^${task_id}|" "${_SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE}"; then
            _scheduler_segment_health_log "duplicate squeue task '${task_id}'; preserving the previous decision"
            return 1
        fi

        # Expand Slurm's compressed hostlist for this task, then normalize it
        # into a sorted node set suitable for comm(1).
        if ! timeout --kill-after=5s "${_SCHEDULER_SEGMENT_HEALTH_HOSTLIST_EXPANSION_TIMEOUT}" \
            scontrol show hostnames "${hostlist}" >"${expanded_raw}"; then
            _scheduler_segment_health_log "hostlist expansion failed or timed out for task=${task_id}; preserving prior state"
            return 1
        fi
        if ! _scheduler_segment_health_has_complete_lines "${expanded_raw}"; then
            _scheduler_segment_health_log "hostlist expansion returned an incomplete row for task=${task_id}; preserving prior state"
            return 1
        fi
        : >"${expanded}"
        while IFS= read -r node; do
            node=$(_scheduler_segment_health_trim_whitespace "${node}")
            [[ -n "${node}" ]] || continue
            if [[ ! "${node}" =~ ^[A-Za-z0-9._-]+$ ]]; then
                _scheduler_segment_health_log "invalid expanded node '${node}'; preserving the previous decision"
                return 1
            fi
            printf '%s\n' "${node}" >>"${expanded}" || return 1
        done <"${expanded_raw}"
        if [[ ! -s "${expanded}" ]]; then
            _scheduler_segment_health_log "empty hostlist expansion for task=${task_id}; preserving the previous decision"
            return 1
        fi
        LC_ALL=C sort -u -o "${expanded}" "${expanded}" || return 1

        # The intersection is both the task's exclusion decision and its
        # diagnostic node payload. An empty value records a healthy task.
        csv=""
        if [[ -s "${_SEGMENT_HEALTH_UNAVAILABLE_NODES_FILE}" ]]; then
            LC_ALL=C comm -12 "${_SEGMENT_HEALTH_UNAVAILABLE_NODES_FILE}" "${expanded}" >"${matched}" || return 1
            if [[ -s "${matched}" ]]; then
                csv=$(paste -s -d ',' "${matched}") || return 1
            fi
        fi
        printf '%s|%s\n' "${task_id}" "${csv}" >>"${_SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE}" || return 1
    done <"${taskmap}"

    # Produce a deterministic snapshot for publication and reconciliation.
    LC_ALL=C sort -t '|' -k1,1n -o "${_SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE}" "${_SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE}" || return 1
    return 0
}

_scheduler_segment_health_reconcile_published_state() {
    local observed_at="$1"
    local task_id
    local desired
    local prior_status
    local prior_value
    local event
    # The process-global set records first misses from the prior complete poll.
    # Reaching the absent-task branch below proves the task is missing now: a
    # prior-set match is therefore its second consecutive miss. First misses
    # accumulate here and replace the prior set after every complete poll.
    local missing_next="|"
    local publication_failed=0

    # Reconcile every running task. Healthy tasks without an existing artifact
    # are skipped so files are created only after a task is first excluded.
    while IFS='|' read -r task_id desired; do
        [[ -n "${task_id}" ]] || continue
        event=excluded
        if _scheduler_segment_health_state_lookup "${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}" "${task_id}"; then
            prior_status="${_SCHEDULER_SEGMENT_HEALTH_STATE_STATUS}"
            prior_value="${_SCHEDULER_SEGMENT_HEALTH_STATE_VALUE}"
            if [[ "${prior_status}" == K && "${prior_value}" == "${desired}" ]]; then
                continue
            fi
            if [[ -z "${desired}" ]]; then
                if _scheduler_segment_health_publish_current_value "${task_id}" ""; then
                    if [[ "${prior_status}" == K && -n "${prior_value}" ]]; then
                        _scheduler_segment_health_append_audit cleared "${task_id}" "" "" "${observed_at}"
                    fi
                else
                    publication_failed=1
                fi
                continue
            fi
            if [[ "${prior_status}" != K || -n "${prior_value}" ]]; then
                event=updated
            fi
        elif [[ -z "${desired}" ]]; then
            continue
        fi
        if _scheduler_segment_health_publish_current_value "${task_id}" "${desired}"; then
            _scheduler_segment_health_append_audit "${event}" "${task_id}" "${desired}" "" "${observed_at}"
        else
            publication_failed=1
        fi
    done <"${_SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE}"

    # Reconcile prior control files absent from the desired-state snapshot. Move
    # one inactive only after it is missing from two complete task maps.
    while IFS='|' read -r task_id prior_status prior_value; do
        [[ -n "${task_id}" ]] || continue
        if grep -q "^${task_id}|" "${_SEGMENT_HEALTH_DESIRED_TASK_STATE_FILE}"; then
            continue
        fi
        if [[ "${_SCHEDULER_SEGMENT_HEALTH_MISSING_TASK_IDS:-|}" == *"|${task_id}|"* ]]; then
            if _scheduler_segment_health_move_task_inactive "${task_id}"; then
                if [[ "${prior_status}" == K && -n "${prior_value}" ]]; then
                    _scheduler_segment_health_append_audit \
                        cleared "${task_id}" "" task_inactive "${observed_at}"
                fi
            else
                publication_failed=1
                missing_next+="${task_id}|"
            fi
        else
            missing_next+="${task_id}|"
            _scheduler_segment_health_log \
                "task absent from one complete squeue result; deferring inactive transition task=${task_id}"
        fi
    done <"${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}"

    _SCHEDULER_SEGMENT_HEALTH_MISSING_TASK_IDS="${missing_next}"
    return "${publication_failed}"
}

_scheduler_segment_health_clear_stale_decisions() {
    local observed_at="$1"
    local task_id
    local status
    local value

    # Fail open when Slurm has not produced a complete desired-state snapshot
    # within the TTL. Reuse this poll's validated published-state snapshot and
    # clear every non-empty or untrusted active task artifact to zero bytes;
    # healthy empty artifacts need no write. A later successful poll can
    # publish fresh exclusions again.
    while IFS='|' read -r task_id status value; do
        [[ -n "${task_id}" ]] || continue
        if [[ "${status}" == K && -z "${value}" ]]; then
            continue
        fi
        if _scheduler_segment_health_publish_current_value "${task_id}" ""; then
            _scheduler_segment_health_append_audit cleared "${task_id}" "" stale_decision_expired "${observed_at}"
        fi
    done <"${_SEGMENT_HEALTH_PUBLISHED_TASK_STATE_FILE}"
}

scheduler_segment_health_poll_once() {
    local last_status=0
    local observed_at
    local stale_seconds

    scheduler_segment_health_configure || return $?
    if [[ ! -d "${NVRX_SEGMENT_HEALTH_CHECK_DIR}" ]]; then
        _scheduler_segment_health_log \
            "artifact directory is unavailable: ${NVRX_SEGMENT_HEALTH_CHECK_DIR}"
        return 1
    fi
    if [[ ! -d "${_SCHEDULER_SEGMENT_HEALTH_WORK}" ]]; then
        _scheduler_segment_health_log \
            "temporary workspace is unavailable: ${_SCHEDULER_SEGMENT_HEALTH_WORK}"
        return 1
    fi

    # Each poll follows three phases:
    #   1. Read the currently published per-task state from the filesystem.
    #   2. Query Slurm and derive the desired state for every running task.
    #   3. Reconcile current with desired: set, update, clear, or retire files.
    # A failure before reconciliation preserves the previously published state.
    if _scheduler_segment_health_snapshot_published_state; then
        if _scheduler_segment_health_derive_desired_state; then
            _SCHEDULER_SEGMENT_HEALTH_LAST_COMPLETE_POLL_AT="${SECONDS}"
            observed_at=$(_scheduler_segment_health_timestamp_utc 2>/dev/null) || observed_at=""
            if _scheduler_segment_health_reconcile_published_state "${observed_at}"; then
                last_status=0
            else
                last_status=1
                _scheduler_segment_health_log "one or more artifact operations failed; continuing with other tasks"
            fi
        else
            last_status=1
            stale_seconds=$((SECONDS - _SCHEDULER_SEGMENT_HEALTH_LAST_COMPLETE_POLL_AT))
            _scheduler_segment_health_log "scheduler poll failed stale_seconds=${stale_seconds}/${NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS}"
            if (( stale_seconds >= NVRX_SEGMENT_HEALTH_STALE_DECISION_SECONDS )); then
                observed_at=$(_scheduler_segment_health_timestamp_utc 2>/dev/null) || observed_at=""
                _scheduler_segment_health_clear_stale_decisions "${observed_at}" || true
            fi
        fi
    else
        last_status=1
        _scheduler_segment_health_log "could not snapshot current artifacts; preserving the previous decision"
    fi

    return "${last_status}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -uo pipefail
    scheduler_segment_health_poll_once
    status=$?
    scheduler_segment_health_cleanup || true
    exit "${status}"
fi
