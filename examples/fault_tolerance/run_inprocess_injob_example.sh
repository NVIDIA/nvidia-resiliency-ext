#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script demonstrates the use of in-process and fault-tolerance (FT) packages within 
# a single PyTorch-based workload. The in-process package detects rank exceptions and crashes 
# and restarts the training using the remaining worker processes with minimal overhead.
# The FT package is used to detect hangs and can restart training processes if needed.
# NOTE: The in-process package has monitoring capabilities that can also be used. 
# Please refer to the in-process package documentation for details.
#
# When using `ft_launcher --restart-policy=min-healthy --nnodes=2:3`, fault tolerance 
# restarts the training when the number of healthy agents (agents with all ranks alive) 
# falls below 2.
#
# Simulated sequence:
# - The minimum number of nodes (launcher instances, agents) is set to 2, and the maximum to 3.
# - Training starts with 3 agents, each having 2 ranks; hence, the world size is 6.
# - A simulated exception in rank 5 at iteration 300 is handled with an in-process restart, 
#   reusing all training processes to resume training.
# - Rank 5 is terminated with SIGKILL at iteration 500. The in-process mechanism handles the failure, 
#   but since the rank process is lost, training resumes with the remaining 5 processes.
# - Rank 4 is terminated with SIGKILL, further reducing the world size to 4 processes.
# - Rank 3 is terminated, reducing the world size to 3. At this point, the FT package 
#   detects that training has failed because only one healthy (fully utilized) node remains.
# - The FT package restarts all 6 training processes, and training is resumed.
# - A simulated hang is detected by the fault-tolerance package, triggering another full restart.
# - Training continues and successfully completes when the iteration limit is reached.
#
# Usage: ./examples/fault_tolerance/run_inprocess_injob_example.sh
# Expected result: All iterations are completed, and agent exit codes are 0.


THIS_SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WORKER_SCRIPT="${THIS_SCRIPT_DIR}/in_job_and_in_process_example.py"

COMMON_LAUNCHER_ARGS="--nnodes=2:3 --max-restarts=2 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500"
COMMON_FT_ARGS="--ft-param-log_level=DEBUG --ft-param-rank_heartbeat_timeout=12 --ft-param-initial_rank_heartbeat_timeout=12 --restart-policy=min-healthy"

# WAR: this example currently does not work with the NIC monitor
COMMON_FT_ARGS="${COMMON_FT_ARGS} --ft_param_enable_nic_monitor=False"

# Avoid c10d log clutter with "(...) TCPStore.cpp:122] [c10d] sendBytes failed (...)" 
# when terminating a hung workload and the TCP Store hosting rank is killed.
export TORCH_CPP_LOG_LEVEL=ERROR

pkill -f ft_launcher
rm -f /tmp/_in_process_example_checkpoint.pt

SIM_FAULTS_DESC='5:300:exc,5:500:sigkill,4:700:sigkill,3:900:sigkill,5:1800:sleep'

ft_launcher $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=1' $COMMON_FT_ARGS \
    $WORKER_SCRIPT --fault-iters=$SIM_FAULTS_DESC --total-iterations=2000 --log-interval=10 --chkpt-interval=10 &

ft_launcher $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' $COMMON_FT_ARGS \
    $WORKER_SCRIPT --fault-iters=$SIM_FAULTS_DESC --total-iterations=2000 --log-interval=10 --chkpt-interval=10 &

ft_launcher $COMMON_LAUNCHER_ARGS --rdzv-conf='is_host=0' $COMMON_FT_ARGS \
    $WORKER_SCRIPT --fault-iters=$SIM_FAULTS_DESC --total-iterations=2000 --log-interval=10 --chkpt-interval=10 &

wait $(jobs -p)