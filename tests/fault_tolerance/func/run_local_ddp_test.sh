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

NUM_TESTS=4
RES_DIR="./local_ddp_test_results"
WORLD_SIZE=8
TORCH_PORT=12321
WORKLOAD_MONITOR_PORT=11223
MAX_RESTARTS=32


mkdir -p ${RES_DIR}
rm -rf ${RES_DIR}/output_*
rm -rf ${RES_DIR}/_ft_scratch_dir


function assert_log_contains {
   log_file="$1"
   expected_str="$2"
   if ! grep -q "${expected_str}" ${log_file}  ; then
      echo "Expected string not found in logs from nodes: ${expected_str}"
      exit 1
   fi
}

function assert_not_in_log {
   log_file="$1"
   not_expected_str="$2"
    if grep -q "${not_expected_str}" ${log_file}  ; then
        echo "Not expected string found in logs from nodes: ${not_expected_str}"
        exit 1
    fi
}

for i in `seq ${NUM_TESTS}`
do
   echo "### Starting TEST ${i}/${NUM_TESTS} ###"
   mkdir ${RES_DIR}/output_$i

   echo "TEST ${i}, run ${training_parts_num}"

   # launch the training
   TRAIN_CMD="examples/fault_tolerance/train_ddp.py "
   TRAIN_CMD+=" --device cuda "
   TRAIN_CMD+=" --init_distributed_method tcp "
   TRAIN_CMD+=" --output_dir ${RES_DIR}/output_$i "
   TRAIN_CMD+=" --logging_interval=100 "
   
   # 1st run should be full, uninterrupted run,
   # other runs will be interrupted with simulated faults.
   if [ "${i}" -ne "1" ]
   then
      TRAIN_CMD+=" --simulated_fault=random,8 "
   fi

   # training will be interrupted with some simulated faults
   # ft_launcher should respawn the training after each simulated fault
   # FIXME: we simulate hangs with SIGTSTP sent to a rank, which sometimes
   # causes rank monitor to hang on IPC reading. when this happens we rely
   # on ft_launcher to terminate the rank; but ft_launcher has large default
   # timeout between SIGTERM and SIGKILL so the test takes long time. 
   # as a workaround, we reduce the term timeout 
   echo "Launching ${WORLD_SIZE} x \"${TRAIN_CMD}\" with ft launcher..."

   ft_launcher --nproc-per-node=${WORLD_SIZE} --max-restarts=${MAX_RESTARTS} \
      --fault-tol-cfg-path="examples/fault_tolerance/fault_tol_cfg.yaml" --term-timeout=15 \
      ${TRAIN_CMD} 2>&1 | tee -a ${RES_DIR}/output_$i/log.log

   if [ $? -ne 0 ]
   then
      echo "FAILED: Training ${i} exit code is non-zero!"
      exit 1
   fi

   # 1st run should be full, uninterrupted run,
   # other runs will be interrupted with simulated faults.
   if [ "${i}" -eq "1" ]
   then
      assert_not_in_log "${RES_DIR}/output_$i/log.log" "Simulating fault"
   else
      assert_log_contains "${RES_DIR}/output_$i/log.log" "Simulating fault"
   fi

   # Timeouts should be updated during each run
   assert_log_contains "${RES_DIR}/output_$i/log.log" "Updated timeouts"
   # Training should be completed after each run
   assert_log_contains "${RES_DIR}/output_$i/log.log" "Leaving main"

done

# Check if results number is OK, 
# We only test hang detection and restarting, so no need to look into the logs.
results_num=$(find ${RES_DIR}/ -name "log.log" |wc -l)
if [ "$results_num" -ne "${NUM_TESTS}" ] 
then
   echo "FAILED: Results number ${results_num} does not match requested tests number ${NUM_TESTS}"
   exit 1  
fi

echo "SUCCESS"