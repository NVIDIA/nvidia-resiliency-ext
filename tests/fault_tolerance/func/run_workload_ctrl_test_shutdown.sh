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

# This is a test of the FT workload shutdown request
# It runs a few launchers(agents) each of them is running just 1 worker(rank)
# Ranks are terminated randomly a few times, to initiate the restart and rendezvous
# Finally, WorkloadAction.ShutdownWorkload control request is sent, that should terminate the workload
# To run the script: ./tests/fault_tolerance/func/run_workload_ctrl_test_shutdown.sh
# Expected result: TEST PASSED is printed and the exit code is 0

set -e

cleanup() {
    echo "Cleaning up before exit..."
    ( kill $(pgrep -f ft_launcher) > /dev/null 2>&1 || true )
    wait
}
trap cleanup EXIT

RANK_CONTROL_FILE="/tmp/_rank_ctl.txt"
RUNS_COUNT_FILE="/tmp/_run_cnt.txt"
THIS_SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WORKER_SCRIPT="${THIS_SCRIPT_DIR}/_workload_ctrl_test_worker.py"

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

function issue_workload_command {
    local cmd=$1
    local timeout=10
    echo "$cmd" > "${RANK_CONTROL_FILE}"
    
    for ((t=1; t<=timeout; t++)); do
        if [[ ! -e "$RANK_CONTROL_FILE" ]]; then
            return
        fi
        sleep 1
    done    
    
    if [[ -e "$RANK_CONTROL_FILE" ]]; then
        echo "ERROR: File $RANK_CONTROL_FILE still exists! Should have been consumed by a worker."
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

function get_still_alive {
    local alive_pids=()
    for pid in "$@"; do
        if is_running "$pid"; then
            alive_pids+=("$pid")
        fi
    done
    echo "${alive_pids[@]}"
}

( kill -s SIGKILL $(pgrep -f ft_launcher) > /dev/null 2>&1 || true ) 

NODES=4
RANK_CONTROL_FILE="/tmp/_rank_ctl.txt"
THIS_SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WORKER_SCRIPT="${THIS_SCRIPT_DIR}/_workload_ctrl_test_worker.py"

export LOGLEVEL='DEBUG'
COMMON_FT_ARGS="--ft-param-log_level=DEBUG --ft-param-rank_heartbeat_timeout=5 --ft-param-initial_rank_heartbeat_timeout=5"
COMMON_LAUNCHER_ARGS="--nproc-per-node=1 --nnodes=${NODES} --rdzv-backend=c10d --rdzv_endpoint=localhost:12345 --max-restarts=100"

rm -f "${RANK_CONTROL_FILE}"
rm -f "${RUNS_COUNT_FILE}"

assert_eq "$(get_runs_count)" 0

ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=1' ${WORKER_SCRIPT} --max-time=90 --ctl-file=${RANK_CONTROL_FILE} --run-cnt-file=${RUNS_COUNT_FILE} --is-agent-rdzv-host=1 &
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} --max-time=90 --ctl-file=${RANK_CONTROL_FILE} --run-cnt-file=${RUNS_COUNT_FILE} --is-agent-rdzv-host=0 &
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} --max-time=90 --ctl-file=${RANK_CONTROL_FILE} --run-cnt-file=${RUNS_COUNT_FILE} --is-agent-rdzv-host=0 &
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' ${WORKER_SCRIPT} --max-time=90 --ctl-file=${RANK_CONTROL_FILE} --run-cnt-file=${RUNS_COUNT_FILE} --is-agent-rdzv-host=0 &

sleep 15
launcher_pids=$(jobs -p)
assert_eq "$(get_num_running $launcher_pids)" $NODES
assert_eq "$(get_runs_count)" 1

for i in {1..4} ; do
    issue_workload_command "terminate_rand_rank"
    sleep 20
    assert_eq "$(get_num_running $launcher_pids)" 4
done

issue_workload_command "shutdown_workload"
wait $launcher_pids || true # launchers should fail due to closed rendezvous

assert_eq "$(get_runs_count)" 5

echo "TEST PASSED"
exit 0
