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
#
# Fault Tolerance functional test, using NeMo 2.0 and FT callback.
# This script should be run on a node with 8 GPUs and with docker available.
# It runs 4 containers that "simulate" 4 nodes, each with 2 GPUs.
#
# Should be run from the root of the repository:
# ./tests/ptl_resiliency/func/nemo20/ft_test_sim_nodes.sh
#
# Expected result is "All tests passed." printed at the end.
#

set -x # show commands as they are executed
set -o pipefail # pipelined commands exit code is 1 if any command in a pipe fails

REPO_ROOT=$(git rev-parse --show-toplevel)
FT_CONT_OUT_DIR="/mnt/ft_test_storage"
TOKENIZER_PATH="/mnt/nvdl/datasets/ft/models/llama/tokenizer.model"
CONTAINER_MOUNTS="-v ft_test_storage:${FT_CONT_OUT_DIR} -v ${TOKENIZER_PATH}:${TOKENIZER_PATH}:ro"

FT_TEST_BASE_IMG="${FT_TEST_BASE_IMG:-'<not set>'}"
TEST_IMG="ft_test_nemo_img"
CONTAINER_COMMON_ARGS="--rm --net testnet --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ${CONTAINER_MOUNTS}"

RDZV_PORT=2323
COMMON_LAUNCHER_ARGS="--rdzv_backend=c10d --rdzv_endpoint=node0:${RDZV_PORT} --nnodes=4 --nproc-per-node=2"

LAUNCHERS_WAIT_EXIT_CODE=0

function run_on_simulated_nodes {
    # Run the same command on 4 simulated nodes, each with 2 GPUs
    # NOTE: node0 is the rendezvous host node, hence it needs to expose the port.
    cmd=$1
    LAUNCHERS_WAIT_EXIT_CODE=0
    rm -f ./node*log.txt
    { docker run -h node0 -p ${RDZV_PORT}:${RDZV_PORT} --gpus='"device=0,1"' ${CONTAINER_COMMON_ARGS} ${TEST_IMG} bash -c "${cmd}" 2>&1 | tee "./node0log.txt" ; } &
    { docker run -h node1 --gpus='"device=2,3"' ${CONTAINER_COMMON_ARGS} ${TEST_IMG} bash -c "${cmd}" 2>&1 | tee "./node1log.txt" ; } &
    { docker run -h node2 --gpus='"device=4,5"' ${CONTAINER_COMMON_ARGS} ${TEST_IMG} bash -c "${cmd}" 2>&1 | tee "./node2log.txt" ; } &
    { docker run -h node3 --gpus='"device=6,7"' ${CONTAINER_COMMON_ARGS} ${TEST_IMG} bash -c "${cmd}" 2>&1 | tee "./node3log.txt" ; } &
    wait `jobs -p`
    LAUNCHERS_WAIT_EXIT_CODE=$?
}

function assert_log_contains {
    expected_str="$1"
    if ! grep -q "${expected_str}" ./node*log.txt  ; then
        echo "Expected string not found in logs from nodes: ${expected_str}"
        exit 1
    fi
}

function assert_not_in_log {
    not_expected_str="$1"
    if grep -q "${not_expected_str}" ./node*log.txt  ; then
        echo "Not expected string found in logs from nodes: ${not_expected_str}"
        exit 1
    fi
}

function assert_checkpoint_saved {
    if [ -d "${FT_CONT_OUT_DIR}/default/checkpoints/step*-last" ] ; then
        echo "Expected last checkpoint to be saved, but not found in ${FT_CONT_OUT_DIR}/default/checkpoints/"
        exit 1
    fi
}

function assert_number_of_runs {
    expected_num=$1
    actual_num=$(grep -c "All distributed processes registered." ./node0log.txt)
    if [ $expected_num -ne $actual_num ] ; then
        echo "Expected number of runs: ${expected_num}, but got ${actual_num}"
        exit 1
    fi
}

function assert_all_launchers_succeded {
    if [ $LAUNCHERS_WAIT_EXIT_CODE -ne 0 ] ; then
        echo "Not all launchers succeeded. LAUNCHERS_WAIT_EXIT_CODE=${LAUNCHERS_WAIT_EXIT_CODE}"
        exit 1
    fi
}

function assert_launchers_failed {
    if [ $LAUNCHERS_WAIT_EXIT_CODE -eq 0 ] ; then
        echo "Some launchers expected to fail, but LAUNCHERS_WAIT_EXIT_CODE=${LAUNCHERS_WAIT_EXIT_CODE}"
        exit 1
    fi
}

####### PREPARE TEST ENVIRONMENT #####

set -e # exit on error during initialization

# Prepare output dir
docker volume rm -f ft_test_storage
docker volume create ft_test_storage

# Build the test container with current sources
docker build --build-arg BASE_IMG="${FT_TEST_BASE_IMG}" -f ${REPO_ROOT}/tests/ptl_resiliency/func/nemo20/Dockerfile.ft_test -t ${TEST_IMG} ${REPO_ROOT}

# Network for the containers
docker network create testnet || echo "Network 'testnet' already exists"

set +e # some errors are expected in the tests

######## TEST STAGE 1 #########
# Simulated fault after 5min.

run_on_simulated_nodes \
    "ft_launcher --ft-param-initial_rank_heartbeat_timeout=60 --ft-param-rank_heartbeat_timeout=60 ${COMMON_LAUNCHER_ARGS} \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=4 \
            --num-gpus=2  \
            --sim-fault-desc='rank_hung,300'"

# Fault should be detected due to the configured FT timeouts.
# There should be a checkpoint saved before the fault happens.
assert_log_contains "Simulating fault"
assert_log_contains "Did not get subsequent heartbeat. Waited 60.00 seconds"
assert_checkpoint_saved
assert_launchers_failed

######## TEST STAGE 2 #########
# Resume run, train until 7min limit

run_on_simulated_nodes \
    "ft_launcher --ignore-missing-fault-tol-cfg ${COMMON_LAUNCHER_ARGS} \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=4 \
            --num-gpus=2  \
            --max-runtime=420"

# Expect updated FT timeouts at the end, as the run was successful and there was a checkpoint loading
assert_log_contains "Time limit reached."
assert_log_contains "Updated FT timeouts."
assert_all_launchers_succeded

######## TEST STAGE 3 #########
# Load computed timeouts, simulated fault after 1min, 2 in-job restarts

run_on_simulated_nodes \
  "ft_launcher --ignore-missing-fault-tol-cfg  --max-restarts=2 ${COMMON_LAUNCHER_ARGS}  \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=4 \
            --num-gpus=2  \
            --sim-fault-desc='rank_hung,60'"

# 3 runs expected
# no timeouts update is expected when a run fails
assert_number_of_runs 3
assert_not_in_log "Updated FT timeouts."
assert_launchers_failed

###### Stage 4 #########
# Final sucessful run until the training time limit is reached

run_on_simulated_nodes \
  "ft_launcher --ignore-missing-fault-tol-cfg ${COMMON_LAUNCHER_ARGS}  \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=4 \
            --num-gpus=2  \
            --max-runtime=900"

assert_log_contains "Updated FT timeouts."
assert_log_contains "Time limit reached."
assert_all_launchers_succeded

echo "All tests passed."
