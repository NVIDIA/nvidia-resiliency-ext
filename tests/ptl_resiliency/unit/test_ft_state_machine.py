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


from nvidia_resiliency_ext.ptl_resiliency.fault_tolerance_callback import _TrainingStateMachine


class TestFaultTolerance:
    def test_training_ended_ok(self):
        # Training ended if there were no training iterations nor error
        sm = _TrainingStateMachine()
        assert sm.is_training_completed() is False
        sm.on_validation_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_validation_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_teardown()
        assert sm.is_training_completed() is True

    def test_training_ended_false_00(self):
        # Training is not completed if there was an error
        sm = _TrainingStateMachine()
        assert sm.is_training_completed() is False
        sm.on_exception()
        sm.on_teardown()
        assert sm.is_training_completed() is False

    def test_training_ended_false_01(self):
        # Training is not completed if there were some training iterations
        # (we detect last "empty run" to determine that the training is completed)
        sm = _TrainingStateMachine()
        assert sm.is_training_completed() is False
        sm.on_train_batch_end()
        sm.on_train_batch_end()
        sm.on_teardown()
        assert sm.is_training_completed() is False

    def test_training_can_upd_timeouts(self):
        sm = _TrainingStateMachine()
        assert sm.can_update_timeouts is False
        sm.on_load_checkpoint()
        sm.on_save_checkpoint()
        assert sm.can_update_timeouts is False
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        # cant save, as mid-epoch checkpoint saving not seen
        assert sm.can_update_timeouts is False
        sm.on_save_checkpoint()
        # now checkpointing was done, but need following heartbeat
        assert sm.can_update_timeouts is False
        sm.on_train_batch_end()
        sm.on_validation_batch_end()
        # expects "sm.on_ft_heartbeat_sent()" not on_*_batch_end
        assert sm.can_update_timeouts is False
        sm.on_ft_heartbeat_sent()
        # finally, post checkpointing hb was observed
        assert sm.can_update_timeouts is True
        sm.on_ft_timeouts_updated()
        # on_timeouts_updated() resets the flag
        # should not back to True, only one update per run is allowed
        assert sm.can_update_timeouts is False
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_save_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        assert sm.can_update_timeouts is False

    def test_training_cant_upd_when_exception(self):
        # sanity check
        sm = _TrainingStateMachine()
        assert sm.can_update_timeouts is False
        sm.on_load_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_save_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        assert sm.can_update_timeouts is True
        # exiting due to an unexpected exc
        sm = _TrainingStateMachine()
        assert sm.can_update_timeouts is False
        sm.on_load_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_save_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_exception(exc=ValueError())
        assert sm.can_update_timeouts is False
        # exiting due to an exit with code != 0
        sm = _TrainingStateMachine()
        assert sm.can_update_timeouts is False
        sm.on_load_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_save_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_exception(exc=SystemExit(1))
        assert sm.can_update_timeouts is False
        # exiting with exit code=0 is OK
        sm = _TrainingStateMachine()
        assert sm.can_update_timeouts is False
        sm.on_load_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_save_checkpoint()
        sm.on_train_batch_end()
        sm.on_ft_heartbeat_sent()
        sm.on_exception(exc=SystemExit(0))
        assert sm.can_update_timeouts is True
