# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

from nvidia_resiliency_ext.shared_utils.health_check import NicHealthCheck, PciMixin, PynvmlMixin


class TestPynvmlMixin(unittest.TestCase):

    @patch("nvidia_resiliency_ext.shared_utils.health_check.pynvml", create=True)
    def test_get_gpu_pci_mapping(self, mock_pynvml):
        """Test retrieving GPU PCI mapping."""
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle_{i}"
        mock_pynvml.nvmlDeviceGetPciInfo.side_effect = lambda handle: MagicMock(
            busId=f"0000:{handle[-1]}9:00.0".encode()
        )

        mixin = PynvmlMixin()
        mixin.pynvml = mock_pynvml  # Manually set after calling check_pynvml_availability()
        gpu_pci_map = mixin.get_gpu_pci_mapping()

        self.assertEqual(gpu_pci_map, {0: "0000:09:00.0", 1: "0000:19:00.0"})

    @patch("nvidia_resiliency_ext.shared_utils.health_check.pynvml", create=True)
    def test_get_gpu_pci_mapping_lowercase_conversion(self, mock_pynvml):
        """Test that upper case values are correctly converted to lowercase."""
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = MagicMock(busId=b"0000:AB:CD.0")

        mixin = PynvmlMixin()
        mixin.pynvml = mock_pynvml
        gpu_pci_map = mixin.get_gpu_pci_mapping()

        self.assertEqual(gpu_pci_map, {0: "0000:ab:cd.0"})

    @patch("nvidia_resiliency_ext.shared_utils.health_check.pynvml", create=True)
    def test_get_gpu_pci_mapping_with_non_ascii_bytes(self, mock_pynvml):
        """Test that non-ASCII characters in busId (bytes format) are correctly decoded and converted."""
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"

        # Simulate a busId with a non-ASCII character (é in UTF-8: \xc3\xa9)
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = MagicMock(busId=b"0000:\xc3\xa9:00.0")

        mixin = PynvmlMixin()
        mixin.pynvml = mock_pynvml
        gpu_pci_map = mixin.get_gpu_pci_mapping()

        # Ensure non-ASCII character is correctly decoded and lowercased
        self.assertEqual(gpu_pci_map, {0: "0000:é:00.0"})  # Expect correct decoding

    @patch("nvidia_resiliency_ext.shared_utils.health_check.pynvml", create=True)
    def test_get_gpu_pci_mapping_with_non_ascii_string(self, mock_pynvml):
        """Test handling of non-ASCII characters in busId (string format)."""
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"

        # Simulate non-ASCII character in busId (directly as a string)
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = MagicMock(busId="0000:ÅB:00.0")

        mixin = PynvmlMixin()
        mixin.pynvml = mock_pynvml

        gpu_pci_map = mixin.get_gpu_pci_mapping()

        # Expect ÅB to remain but be lowercased
        self.assertEqual(gpu_pci_map, {0: "0000:åb:00.0"})

    @patch("nvidia_resiliency_ext.shared_utils.health_check.pynvml", create=True)
    def test_get_gpu_pci_mapping_correct_lower_after_decode(self, mock_pynvml):
        """Test that decode() first, then lower() correctly handles non-ASCII characters."""
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"

        # Simulate a busId containing a non-ASCII character (É in UTF-8: \xc3\x89)
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = MagicMock(busId=b"0000:\xc3\x89:00.0")

        mixin = PynvmlMixin()
        mixin.pynvml = mock_pynvml
        gpu_pci_map = mixin.get_gpu_pci_mapping()

        # Ensure correct decoding and lowercase conversion
        self.assertEqual(gpu_pci_map, {0: "0000:é:00.0"})

    @patch("nvidia_resiliency_ext.shared_utils.health_check.pynvml", create=True)
    def test_get_gpu_pci_mapping_with_none_bus_id(self, mock_pynvml):
        """Test handling of None busId values from NVML."""
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"

        # Simulate None busId
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = MagicMock(busId=None)

        mixin = PynvmlMixin()
        mixin.pynvml = mock_pynvml

        with self.assertRaises(TypeError):  # Since None does not support .lower()
            mixin.get_gpu_pci_mapping()

    @patch("nvidia_resiliency_ext.shared_utils.health_check.pynvml", create=True)
    def test_get_gpu_pci_mapping_with_empty_bus_id(self, mock_pynvml):
        """Test behavior when NVML returns an empty busId."""
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"

        # Empty string busId
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = MagicMock(busId="")

        mixin = PynvmlMixin()
        mixin.pynvml = mock_pynvml
        gpu_pci_map = mixin.get_gpu_pci_mapping()

        self.assertEqual(gpu_pci_map, {0: ""})  # Should store an empty string


class TestPciMixin(unittest.TestCase):

    @patch("os.path.exists", side_effect=lambda path: path != "/sys/bus/pci/devices/0000:20:00.0")
    def test_get_pci_ancestor(self, mock_exists):
        """Test walking up the PCI hierarchy."""
        mixin = PciMixin()

        def realpath_side_effect(path):
            if path == "/sys/bus/pci/devices/0000:19:00.0/..":
                # First call mock
                return "/sys/bus/pci/devices/pci0000:16/0000:16:01.0/0000:17:00.0/0000:18:00.0"
            # Subsequent call
            return path.rsplit("/", 2)[0]

        with patch("os.path.realpath", side_effect=realpath_side_effect):
            ancestors = mixin.get_pci_ancestor("0000:19:00.0")
        self.assertEqual(
            ancestors, ["0000:19:00.0", "0000:18:00.0", "0000:17:00.0", "0000:16:01.0"]
        )

    @patch("os.listdir", return_value=["mlx5_0", "mlx5_1"])
    def test_get_ib_pci_mapping(self, mock_listdir):
        """Test retrieving IB device to PCI bus mapping."""
        mixin = PciMixin()

        def realpath_side_effect(path):
            if path.endswith("mlx5_0"):
                return "/sys/devices/pci0000:16/0000:16:01.0/0000:17:00.0/0000:18:01.0/0000:1a:00.0/infiniband/mlx5_0"
            # Subsequent call
            return "/sys/devices/pci0000:16/0000:16:01.0/0000:17:00.0/0000:18:02.0/0000:1b:00.0/infiniband/mlx5_1"

        with patch("os.path.realpath", side_effect=realpath_side_effect):
            ib_map = mixin.get_ib_pci_mapping()
        self.assertEqual(ib_map, {"mlx5_0": "0000:1a:00.0", "mlx5_1": "0000:1b:00.0"})


class TestNicHealthCheck(unittest.TestCase):

    xml_content = """<system version="1">
  <cpu numaid="0" affinity="0000ffff,0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="23" modelid="49">
    <pci busid="ffff:ff:01.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="0003:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0103:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0004:00:00.0" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="0104:00:00.0" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
  </cpu>
</system>"""

    @patch("nvidia_resiliency_ext.shared_utils.health_check.NicHealthCheck.get_pci_ancestor")
    def test_get_gpu_ib_assignments_from_system(self, mock_get_pci_ancestor):
        """Test GPU-IB assignment based on PCI system hierarchy."""
        mock_get_pci_ancestor.side_effect = lambda pci_id: [pci_id, "root"]
        checker = NicHealthCheck()

        gpu_pci_map = {0: "0000:19:00.0"}
        ib_pci_map = {"mlx5_0": "0000:18:00.0"}

        assignments = checker._get_gpu_ib_assignments_from_system(gpu_pci_map, ib_pci_map)

        self.assertEqual(assignments, [(0, "mlx5_0", 2)])

    def test_get_gpu_ib_assignments_from_topo(self):
        """Test GPU-IB assignment based on PCI system hierarchy."""
        tmp_file = tempfile.mktemp()
        with open(tmp_file, "w") as ff:
            ff.write(self.xml_content)
            ff.flush()

        checker = NicHealthCheck(pci_topo_file=tmp_file)

        gpu_pci_map = {0: "0003:00:00.0"}
        ib_pci_map = {"mlx5_0": "0103:00:00.0"}

        assignments = checker._get_gpu_ib_assignments_from_topo(gpu_pci_map, ib_pci_map)
        os.unlink(tmp_file)

        self.assertEqual(assignments, [(0, "mlx5_0", 0)])

    def test_set_nic_device_success(self):
        """Test setting NIC device for a given GPU local rank."""
        checker = NicHealthCheck()
        checker._gpu_ib_map = {0: "mlx5_0"}

        checker.set_nic_device(0)

        self.assertEqual(checker.nic_name, "mlx5_0")
        self.assertEqual(checker._local_rank, 0)

    def test_set_nic_device_failure(self):
        """Test behavior when a GPU is missing in gpu_ib_map."""
        checker = NicHealthCheck()
        checker._gpu_ib_map = {1: "mlx5_1"}  # No entry for rank 0

        checker.set_nic_device(0)

        self.assertIsNone(checker.nic_name)

    @patch("os.path.exists", return_value=False)
    def test_health_check_no_link_downed_file(self, mock_exists):
        checker = NicHealthCheck()
        checker.nic_name = "mlx5_0"
        result = checker._perform_health_check()
        self.assertTrue(result)  # Should return True if file doesn't exist

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="5")
    def test_health_check_link_downed_incremented(self, mock_open, mock_exists):
        checker = NicHealthCheck()
        checker.nic_name = "mlx5_0"
        checker._prev_link_downed = 3
        result = checker._perform_health_check()
        self.assertFalse(result)  # Should return False as link_downed increased

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="3")
    def test_health_check_no_increment(self, mock_open, mock_exists):
        checker = NicHealthCheck()
        checker.nic_name = "mlx5_0"
        checker._prev_link_downed = 3
        result = checker._perform_health_check()
        self.assertTrue(result)  # No increment, should return True

    def test_sync_call_healthy(self):
        checker = NicHealthCheck()
        checker.nic_name = "mlx5_0"
        with patch.object(checker, "_perform_health_check", return_value=True):
            result = checker()
        self.assertTrue(result)

    def test_sync_call_unhealthy(self):
        checker = NicHealthCheck()
        checker.nic_name = "mlx5_0"
        with patch.object(checker, "_perform_health_check", return_value=False):
            result = checker()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
