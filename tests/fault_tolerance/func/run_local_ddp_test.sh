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
WORLD_SIZE=2 # must be 2 for deterministic all reduce
TORCH_PORT=12321
WORKLOAD_MONITOR_PORT=11223
MAX_RESTARTS=32


mkdir -p ${RES_DIR}
rm -rf ${RES_DIR}/output_*
rm -rf ${RES_DIR}/_ft_scratch_dir

for i in `seq ${NUM_TESTS}`
do
   echo "### Starting TEST ${i}/${NUM_TESTS} ###"
   mkdir ${RES_DIR}/output_$i

   echo "TEST ${i}, run ${training_parts_num}"

   # launch the training
   TRAIN_CMD="examples/train_ddp.py "
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
      --fault-tol-cfg-path="examples/fault_tol_cfg.yaml" --term-timeout=15 \
      ${TRAIN_CMD} >> ${RES_DIR}/output_$i/log.log

   if [ $? -ne 0 ]
   then
      echo "FAILED: Training ${i} exit code is non-zero!"
      exit 1
   fi

   # Training is finished, we have new results in "${RES_DIR}/output_$i"
   # extract special lines used for verification into separate file
   # remove duplicated lines
   grep "CHECK" ${RES_DIR}/output_$i/log.log | \
      awk '{if ($0==lst) {lst=""} else {print $0; lst=$0}}' > ${RES_DIR}/output_$i/short_log.log

   # Check if results not empty
   if [ ! -s ${RES_DIR}/output_$i/short_log.log ]
   then
      echo "FAILED: Results from training ${i} are empty!"
      exit 1    
   fi

   # Check if current results match
   if ( ! diff -yq ${RES_DIR}/output_1/short_log.log ${RES_DIR}/output_$i/short_log.log )
   then
      echo "FAILED: Results from trainings 1 and ${i} are not the same"
      exit 1    
   fi
done

# Check if results number is OK
results_num=$(find ${RES_DIR}/ -name "short_log.log" |wc -l)
if [ "$results_num" -ne "${NUM_TESTS}" ] 
then
   echo "FAILED: Results number ${results_num} does not match requested tests number ${NUM_TESTS}"
   exit 1  
fi

# Double check all results
if ( find ${RES_DIR}/ -name "short_log.log" |xargs -L 1 diff -yq ${RES_DIR}/output_1/short_log.log )
then
   echo "SUCCESS: all trainings results match"
else
   echo "FAILED: Results from trainings are not the same"
   exit 1    
fi
