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
# This script is designed to be executed within a SLURM job environment with --nodes=${NUM_NODES} --ntasks-per-node=1
#
# Stages are supposed to be run one after another for complete testing scenario
#
# To run the first stage:
# ./tests/ptl_resiliency/func/nemo20/ft_test_launchers.sh 1
#
# To run the second stage:
# ./tests/ptl_resiliency/func/nemo20/ft_test_launchers.sh 2
#
# Output from each stage can be verified with `ft_test_asserts.sh`

set -x
set -o pipefail

: "${NUM_NODES:?Error: NUM_NODES is not set or empty}"
: "${TOKENIZER_PATH:?Error: TOKENIZER_PATH is not set or empty}"
: "${FT_CONT_OUT_DIR:?Error: FT_CONT_OUT_DIR is not set or empty}"

if [[ "$FT_CALLBACK_TYPE" == "heartbeats" ]]; then
    FT_ARGS="--ft-param-initial_rank_heartbeat_timeout=60 --ft-param-rank_heartbeat_timeout=60 --ft-param-log_level=DEBUG"
elif [[ "$FT_CALLBACK_TYPE" == "sections" ]]; then
    FT_ARGS="--ft-param-rank_section_timeouts=setup:60,step:60,checkpointing:60 --ft-param-rank_out_of_section_timeout=60 --ft-param-log_level=DEBUG"
else
    echo "\$FT_CALLBACK_TYPE should be 'heartbeats' or 'sections'"
    exit 1
fi


# Parse input arg
STAGE=$1

if [[ -z "$STAGE" ]]; then
    echo "Usage: $0 <stage_number>"
    echo "Example: $0 1"
    exit 1
fi

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

REPO_ROOT=$(git rev-parse --show-toplevel)
NUM_GPUS_PER_NODE=8
COMMON_LAUNCHER_ARGS="--rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --nnodes=${NUM_NODES} --nproc-per-node=${NUM_GPUS_PER_NODE}"


set +e # some errors are expected in the tests

# Run the specific stage
case "$STAGE" in
    1)
        echo "######## TEST STAGE 1 #########"
        echo "Simulated fault after 3min."
        ft_launcher ${FT_ARGS} ${COMMON_LAUNCHER_ARGS} \
            ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
                --tokenizer-path=${TOKENIZER_PATH} \
                --log-dir=${FT_CONT_OUT_DIR} \
                --num-nodes=${NUM_NODES} \
                --num-gpus=${NUM_GPUS_PER_NODE} \
                --sim-fault-desc="rank_hung,180" \
                --cb-type ${FT_CALLBACK_TYPE}
        ;;
    2)
        echo "######## TEST STAGE 2 ###########"
        echo "Resume run, train until 5min limit."
        ft_launcher ${FT_ARGS} ${COMMON_LAUNCHER_ARGS} \
            ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
                --tokenizer-path=${TOKENIZER_PATH} \
                --log-dir=${FT_CONT_OUT_DIR} \
                --num-nodes=${NUM_NODES} \
                --num-gpus=${NUM_GPUS_PER_NODE} \
                --max-runtime=300 \
                --cb-type ${FT_CALLBACK_TYPE}
        ;;
    3)
        echo "######## TEST STAGE 3 #########"
        echo "Load computed timeouts, simulated fault after 1min, 2 in-job restarts."
        ft_launcher ${FT_ARGS} --max-restarts=2 ${COMMON_LAUNCHER_ARGS} \
            ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
                --tokenizer-path=${TOKENIZER_PATH} \
                --log-dir=${FT_CONT_OUT_DIR} \
                --num-nodes=${NUM_NODES} \
                --num-gpus=${NUM_GPUS_PER_NODE} \
                --sim-fault-desc="rank_hung,60" \
                --cb-type ${FT_CALLBACK_TYPE}
        ;;
    4)
        echo "###### TEST STAGE 4 #########"
        echo "Final successful run until the 12min training time limit is reached."
        ft_launcher ${FT_ARGS} ${COMMON_LAUNCHER_ARGS} \
            ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
                --tokenizer-path=${TOKENIZER_PATH} \
                --log-dir=${FT_CONT_OUT_DIR} \
                --num-nodes=${NUM_NODES} \
                --num-gpus=${NUM_GPUS_PER_NODE} \
                --max-runtime=720 \
                --cb-type ${FT_CALLBACK_TYPE}
        ;;
    *)
        echo "Invalid stage: $STAGE"
        echo "Usage: $0 <stage_number>"
        echo "Valid stages: 1, 2, 3, 4"
        exit 1
        ;;
esac
