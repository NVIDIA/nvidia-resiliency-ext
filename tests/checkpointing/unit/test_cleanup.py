# spdx-filecopyrighttext: copyright (c) 2024 nvidia corporation & affiliates. all rights reserved.
# spdx-license-identifier: apache-2.0
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
# http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import logging
import re
import time
from pathlib import Path

import pytest

from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)

from . import TempNamedDir
from .test_utilities import SimpleTensorAwareStateDict, Utils


class TestLocalCheckpointing:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        pass

    def _async_save(self, async_save_request, async_save):
        if async_save:
            async_save_request.execute_sync()
        else:
            assert async_save_request is None

    @pytest.mark.parametrize(('use_ramdisk'), [False, True])
    @pytest.mark.parametrize(('async_save'), [True])
    def test_basic_save_load_scenarios(self, tmp_path_dist_ckpt, use_ramdisk, async_save, caplog):
        if use_ramdisk:
            tmp_path_dist_ckpt = Path("/dev/shm")
        with (
            TempNamedDir(tmp_path_dist_ckpt / "test_save_load") as local_ckpt_dir,
            caplog.at_level(logging.DEBUG),
        ):
            local_ckpt_dir = local_ckpt_dir / "subdir"  # Test handling of non-existent directories

            # Test performance on SSD only to save compute time.
            tensor_num = 10 if use_ramdisk else 16384

            checkpoint_manager = LocalCheckpointManager(local_ckpt_dir)
            # "Multiple saves"
            intermediete_state_dict = SimpleTensorAwareStateDict(iteration=1, tensor_num=tensor_num)
            # SAVE
            async_save_request = checkpoint_manager.save(intermediete_state_dict, 1, async_save)
            self._async_save(async_save_request, async_save)
            ckpt_id = checkpoint_manager._ckpt_id(1)
            first_ckpt_path = checkpoint_manager._local_ckpt_path_from_id(ckpt_id)
            assert first_ckpt_path.exists()
            intermediete_state_dict = SimpleTensorAwareStateDict(iteration=2, tensor_num=tensor_num)
            # SAVE
            async_save_request = checkpoint_manager.save(intermediete_state_dict, 2, async_save)
            self._async_save(async_save_request, async_save)
            ckpt_id = checkpoint_manager._ckpt_id(2)
            second_ckpt_path = checkpoint_manager._local_ckpt_path_from_id(ckpt_id)
            assert second_ckpt_path.exists()
            time.sleep(0.8)
            assert not first_ckpt_path.exists()

            def extract_finalize_time_from_log(caplog):
                pattern = r"finalize_fn took ([\d.]+)s"
                matches = re.findall(pattern, caplog.text)
                if matches:
                    return float(matches[-1])  # Return the last match as a float
                return None

            time_to_finalize = extract_finalize_time_from_log(caplog)
            # Async cleanup based on Processes: ~0.04s, sync: >0.1s
            assert time_to_finalize < 0.03
