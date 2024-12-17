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
# This script should be run on a SLURM cluster login node
#
# Should be run from the root of the NVRx repository:
# TEST_IMG="nemo20_container" \
# SLURM_ACCOUNT="slurm_account" \
# SLURM_PARTITION="slurm_partition" \
# NUM_NODES=2 \
# TOKENIZER_PATH="llama3 tokenizer path" \
# ./tests/ptl_resiliency/func/nemo20/ft_test_slurm.sh"
#
# Expected result is "All tests passed." printed at the end.
#

set -x # show commands as they are executed
set -o pipefail # pipelined commands exit code is 1 if any command in a pipe fails

: "${SLURM_ACCOUNT:?Error: SLURM_ACCOUNT is not set or empty}"
: "${SLURM_PARTITION:?Error: SLURM_PARTITION is not set or empty}"
: "${NUM_NODES:?Error: NUM_NODES is not set or empty}"
: "${TEST_IMG:?Error: TEST_IMG is not set or empty}"
: "${TOKENIZER_PATH:?Error: TOKENIZER_PATH is not set or empty}"

REPO_ROOT=$(git rev-parse --show-toplevel)
FT_CONT_OUT_DIR="${REPO_ROOT}/ft_test_storage"
CONTAINER_MOUNTS="${REPO_ROOT}:${REPO_ROOT},${FT_CONT_OUT_DIR}:${FT_CONT_OUT_DIR},${TOKENIZER_PATH}:${TOKENIZER_PATH}"

NUM_GPUS_PER_NODE=8
COMMON_LAUNCHER_ARGS="--rdzv_backend=c10d --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} --nnodes=${NUM_NODES} --nproc-per-node=${NUM_GPUS_PER_NODE}"

# copy repo to the container and install, to avoid issues when multiple containers try to install from the same dir
INIT_CONT="hostname && nvidia-smi && cp -r ${REPO_ROOT} /tmp/repo && pip install /tmp/repo && cd ${REPO_ROOT}"

LAUNCHERS_WAIT_EXIT_CODE=0

function run_on_nodes {
    cmd=$1
    srun --nodes=${NUM_NODES} --ntasks-per-node=1 --cpus-per-task=64 --account=${SLURM_ACCOUNT} --partition=${SLURM_PARTITION} --job-name="${SLURM_ACCOUNT}-nvrx.ci" --time='00:30:00' \
	   --exclusive --container-image=${TEST_IMG} --container-mounts=${CONTAINER_MOUNTS} bash -c "${INIT_CONT} && ${cmd}" 2>&1 | tee "${FT_CONT_OUT_DIR}/node0log.txt"
    LAUNCHERS_WAIT_EXIT_CODE=$?
}

function assert_log_contains {
    expected_str="$1"
    if ! grep -q "${expected_str}" ${FT_CONT_OUT_DIR}/node*log.txt  ; then
        echo "Expected string not found in logs from nodes: ${expected_str}"
        exit 1
    fi
}

function assert_not_in_log {
    not_expected_str="$1"
    if grep -q "${not_expected_str}" ${FT_CONT_OUT_DIR}/node*log.txt  ; then
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
    actual_num=$(grep -c "All distributed processes registered." ${FT_CONT_OUT_DIR}/node0log.txt)
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

echo "####### PREPARE TEST ENVIRONMENT #####"

set -e # exit on error during initialization

# Prepare output dir
if [ -d "${FT_CONT_OUT_DIR}" ] ; then
  [[ "${FT_CONT_OUT_DIR}" == *ft_test* ]] || exit 1
  rm -rf "${FT_CONT_OUT_DIR}"
fi
mkdir -p "${FT_CONT_OUT_DIR}"

set +e # some errors are expected in the tests

echo "######## TEST STAGE 1 #########"
echo "Simulated fault after 5min."

run_on_nodes \
    "ft_launcher --ft-param-initial_rank_heartbeat_timeout=60 --ft-param-rank_heartbeat_timeout=60 ${COMMON_LAUNCHER_ARGS} \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=${NUM_NODES} \
            --num-gpus=${NUM_GPUS_PER_NODE}  \
            --sim-fault-desc='rank_hung,300'"

# Fault should be detected due to the configured FT timeouts.
# There should be a checkpoint saved before the fault happens.
assert_log_contains "Simulating fault"
assert_log_contains "Did not get subsequent heartbeat. Waited 60.00 seconds"
assert_checkpoint_saved
assert_launchers_failed

echo "######### TEST STAGE 2 ###########"
echo "Resume run, train until 7min limit"

run_on_nodes \
    "ft_launcher --ignore-missing-fault-tol-cfg ${COMMON_LAUNCHER_ARGS} \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=${NUM_NODES} \
            --num-gpus=${NUM_GPUS_PER_NODE}  \
            --max-runtime=420"

# Expect updated FT timeouts at the end, as the run was successful and there was a checkpoint loading
assert_log_contains "Time limit reached."
assert_log_contains "Updated FT timeouts."
assert_all_launchers_succeded

echo "######## TEST STAGE 3 #########"
echo "Load computed timeouts, simulated fault after 1min, 2 in-job restarts"

run_on_nodes \
  "ft_launcher --ignore-missing-fault-tol-cfg  --max-restarts=2 ${COMMON_LAUNCHER_ARGS}  \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=${NUM_NODES} \
            --num-gpus=${NUM_GPUS_PER_NODE}  \
            --sim-fault-desc='rank_hung,60'"

# 3 runs expected
# no timeouts update is expected when a run fails
assert_number_of_runs 3
assert_not_in_log "Updated FT timeouts."
assert_launchers_failed

echo "###### Stage 4 #########"
echo "Final sucessful run until the training time limit is reached"

run_on_nodes \
  "ft_launcher --ignore-missing-fault-tol-cfg ${COMMON_LAUNCHER_ARGS}  \
        ./tests/ptl_resiliency/func/nemo20/ft_test_llama3.py \
            --tokenizer-path=${TOKENIZER_PATH} \
            --log-dir=${FT_CONT_OUT_DIR} \
            --num-nodes=${NUM_NODES} \
            --num-gpus=${NUM_GPUS_PER_NODE}  \
            --max-runtime=900"

assert_log_contains "Updated FT timeouts."
assert_log_contains "Time limit reached."
assert_all_launchers_succeded

echo "All tests passed."
