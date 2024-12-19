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
# Local checkpointing functional test, using NeMo 2.0 and FT callback.
# Based on `ft_test.sh`.
# This script should be run on a node with 8 GPUs and with docker available.
# It runs 4 containers that "simulate" 4 nodes, each with 2 GPUs.
#
# Should be run from the root of the repository:
# ./tests/ptl_resiliency/func/nemo20/local_ckpt_test.sh
#
# Expected result is "All tests passed." printed at the end.
#

# TODO: consider merging with `ft_test.sh`

set -x # show commands as they are executed
set -o pipefail # pipelined commands exit code is 1 if any command in a pipe fails

REPO_ROOT=$(git rev-parse --show-toplevel)
FT_CONT_OUT_DIR="/mnt/ft_test_storage"
TOKENIZER_PATH="/mnt/nvdl/datasets/ft/models/llama/tokenizer.model"
echo EXTRA_CONTAINER_MOUNTS=${EXTRA_CONTAINER_MOUNTS}
CONTAINER_MOUNTS="-v ./ft_test_storage:${FT_CONT_OUT_DIR} -v ${TOKENIZER_PATH}:${TOKENIZER_PATH}:ro ${EXTRA_CONTAINER_MOUNTS}"

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

####### PREPARE TEST ENVIRONMENT #####

set -e # exit on error during initialization

# Build the test container with current sources
docker build --build-arg BASE_IMG="${FT_TEST_BASE_IMG}" -f ${REPO_ROOT}/tests/ptl_resiliency/func/nemo20/Dockerfile.ft_test -t ${TEST_IMG} ${REPO_ROOT}

# Network for the containers
docker network create testnet || echo "Network 'testnet' already exists"

set +e # some errors are expected in the tests

######## TEST STAGE 1: LOCAL CKPT SAVE #########

mkdir -p ft_test_storage
docker run ${CONTAINER_COMMON_ARGS} ${TEST_IMG} bash -c "rm -rf ${FT_CONT_OUT_DIR}/default ${FT_CONT_OUT_DIR}/lightning_logs ${FT_CONT_OUT_DIR}/local_ckpt"

run_on_simulated_nodes \
    "MEGATRON_LOGGING_LEVEL=10 ft_launcher --ft-param-initial_rank_heartbeat_timeout=600 --ft-param-rank_heartbeat_timeout=600 ${COMMON_LAUNCHER_ARGS} \
        ./tests/ptl_resiliency/func/nemo20/test_local_ckpt_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=4 \
            --num-gpus=2 \
            --max-steps=100 \
            --local-checkpoint-interval 20"

rm -rf ${FT_CONT_OUT_DIR}/local_ckpt/node1

######## TEST STAGE 2: LOCAL CKPT LOAD #########

run_on_simulated_nodes \
    "MEGATRON_LOGGING_LEVEL=10 ft_launcher --ft-param-initial_rank_heartbeat_timeout=600 --ft-param-rank_heartbeat_timeout=600 ${COMMON_LAUNCHER_ARGS} \
        ./tests/ptl_resiliency/func/nemo20/test_local_ckpt_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=4 \
            --num-gpus=2 \
            --max-steps=200 \
            --local-checkpoint-interval 20"

echo "LOADING DONE"
assert_log_contains "Resuming from a local checkpoint"
echo "All tests passed"