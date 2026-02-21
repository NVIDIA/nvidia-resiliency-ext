# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# This is a test of the FT launcher --restart-policy=any-failed
# It verifies that a workload is not restarted until the number of runnign nodes falls below the minimum
# To run the script: ./tests/fault_tolerance/func/run_launcher_any_failed_mode_test.sh
# Expected result: TEST PASSED is printed and the exit code is 0

set -e

cleanup() {
    echo "Cleaning up before exit..."
    ( kill $(pgrep -f ft_launcher) > /dev/null 2>&1 || true )
    wait
}
trap cleanup EXIT

RUNS_COUNT_FILE="/tmp/_run_cnt.txt"
WORLD_SIZE_FILE="/tmp/_world_size.txt"
THIS_SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WORKER_SCRIPT="${THIS_SCRIPT_DIR}/_launcher_mode_test_worker.py"

function is_running {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

function get_num_running {
    local pids=("$@")
    local res=0
    for pid in "${pids[@]}"; do
        if is_running "$pid" ; then 
            res=$((res + 1))
        fi
    done
    echo $res
}

function assert_eq {
    local v1=$1
    local v2=$2
    if [[ "$v1" != "$v2" ]] ; then
        echo "VALUES ARE NOT EQUAL: $v1 vs $v2"
        exit 1
    fi
}

function get_runs_count {
    if [[ -f "$RUNS_COUNT_FILE" ]]; then
        cat "$RUNS_COUNT_FILE"
    else
        echo 0
    fi
}

function get_world_size {
    if [[ -f "$WORLD_SIZE_FILE" ]]; then
        cat "$WORLD_SIZE_FILE"
    else
        echo 0
    fi
}

function get_still_alive {
    local alive_pids=()
    for pid in "$@"; do
        if is_running "$pid"; then
            alive_pids+=("$pid")
        fi
    done
    echo "${alive_pids[@]}"
}

log_title() {
    local title="$1"
    local formatted_title="### $title ###"
    local length=${#formatted_title}
    local border
    border=$(printf '#%.0s' $(seq 1 "$length"))
    echo "$border"
    echo "$formatted_title"
    echo "$border"
}


( kill -s SIGKILL $(pgrep -f ft_launcher) > /dev/null 2>&1 || true ) 

export LOGLEVEL='DEBUG'
COMMON_FT_ARGS="--ft-log-level=DEBUG"
COMMON_LAUNCHER_ARGS="--nproc-per-node=2 --nnodes=2:3 --rdzv-backend=c10d --rdzv_endpoint=localhost:12345 --max-restarts=5 --ft-restart-policy=any-failed"
COMMON_TEST_SCRIPT_ARGS="--max-time=60 --run-cnt-file=${RUNS_COUNT_FILE} --world-size-file=${WORLD_SIZE_FILE}"
rm -f "${RUNS_COUNT_FILE}"
rm -f "${WORLD_SIZE_FILE}"

assert_eq "$(get_runs_count)" 0
assert_eq "$(get_world_size)" 0

log_title "Starting 4 initial agents..."

ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=1' ${WORKER_SCRIPT} ${COMMON_TEST_SCRIPT_ARGS} &
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} ${COMMON_TEST_SCRIPT_ARGS} &
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} ${COMMON_TEST_SCRIPT_ARGS} &
agent_pid0=$!

sleep 10 # ensure that the following agent will join as a spare

ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} ${COMMON_TEST_SCRIPT_ARGS} &
agent_pid1=$!

sleep 10
launcher_pids=$(jobs -p)

assert_eq "$(get_num_running $launcher_pids)" 4
assert_eq "$(get_runs_count)" 1
assert_eq "$(get_world_size)" 6

log_title "Terminate a running node, workload should be restarted using the spare, running=3 spares=0"
kill ${agent_pid0}
sleep 30
assert_eq "$(get_num_running $launcher_pids)" 3
assert_eq "$(get_runs_count)" 2
assert_eq "$(get_world_size)" 6

log_title "Terminate another running node, workload should not be restarted, running=2 spares=0"
kill ${agent_pid1}
sleep 30
assert_eq "$(get_num_running $launcher_pids)" 2
assert_eq "$(get_runs_count)" 2
assert_eq "$(get_world_size)" 4

log_title "Introduce one more spare node, training should be interrupted, running=3 spares=0"
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} ${COMMON_TEST_SCRIPT_ARGS} &
agent_pid2=$!
launcher_pids="${launcher_pids} ${agent_pid2}"
sleep 30

assert_eq "$(get_num_running $launcher_pids)" 3
assert_eq "$(get_runs_count)" 3
assert_eq "$(get_world_size)" 6

log_title "Introduce one more spare node, training should not be interrupted, running=3 spares=1"
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} ${COMMON_TEST_SCRIPT_ARGS} &
agent_pid3=$!
launcher_pids="${launcher_pids} ${agent_pid3}"
sleep 10
assert_eq "$(get_num_running $launcher_pids)" 4
assert_eq "$(get_runs_count)" 3
assert_eq "$(get_world_size)" 6

log_title "Terminate the spare, training should not be affected"
kill ${agent_pid3}
sleep 10
assert_eq "$(get_num_running $launcher_pids)" 3
assert_eq "$(get_runs_count)" 3
assert_eq "$(get_world_size)" 6

log_title "Wait for clean exit due to test script timeout"
alive_pids=$(get_still_alive $launcher_pids)
if ! wait $alive_pids ; then
    echo "TEST FAILED: Some launcher(s) failed"
    exit 1
fi

echo "TEST PASSED"
exit 0