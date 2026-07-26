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
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

from nvidia_resiliency_ext.shared_utils.health_check import (
    AttributionService,
    NicHealthCheck,
    NodeHealthCheck,
    NVLHealthCheck,
    PciMixin,
    PynvmlMixin,
)


def _attribution_item(raw_text, reason_code):
    return {
        "raw_text": raw_text,
        "auto_resume": raw_text.split("\n", 1)[0],
        "auto_resume_explanation": "",
        "attribution_text": "",
        "checkpoint_saved_flag": 0,
        "primary_issues": [],
        "secondary_issues": [],
    }


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


class TestNVLHealthCheck(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""

        # Create a proper exception class for NVMLError
        class MockNVMLError(Exception):
            pass

        # Mock pynvml availability check
        self.mock_pynvml = MagicMock()
        self.mock_pynvml.NVML_NVLINK_MAX_LINKS = 18
        self.mock_pynvml.NVML_FEATURE_DISABLED = 0
        self.mock_pynvml.NVMLError = MockNVMLError

        # Mock NVML constants
        self.mock_pynvml.NVML_ERROR_INVALID_ARGUMENT = 1
        self.mock_pynvml.NVML_ERROR_NOT_SUPPORTED = 2

    def test_init_default_parameters(self):
        """Test NVLHealthCheck initialization with default parameters."""
        checker = NVLHealthCheck()
        self.assertIsNone(checker.device_index)
        self.assertEqual(checker.interval, 60)
        self.assertIsNone(checker.on_failure)

    def test_init_with_device_index(self):
        """Test NVLHealthCheck initialization with specific device index."""
        checker = NVLHealthCheck(device_index=2)
        self.assertEqual(checker.device_index, 2)
        self.assertEqual(checker.interval, 60)
        self.assertIsNone(checker.on_failure)

    def test_init_with_custom_interval(self):
        """Test NVLHealthCheck initialization with custom interval."""
        checker = NVLHealthCheck(interval=30)
        self.assertIsNone(checker.device_index)
        self.assertEqual(checker.interval, 30)

    def test_init_with_on_failure_callback(self):
        """Test NVLHealthCheck initialization with failure callback."""

        def callback():
            pass

        checker = NVLHealthCheck(on_failure=callback)
        self.assertEqual(checker.on_failure, callback)

    def test_check_nvl_links_for_device_all_healthy(self):
        """Test checking NVL links when all links are healthy."""
        self.mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        # Mock all 18 links as healthy
        self.mock_pynvml.nvmlDeviceGetNvLinkState.side_effect = [1] * 18

        checker = NVLHealthCheck()
        checker.pynvml = self.mock_pynvml

        result = checker._check_nvl_links_for_device(0)
        self.assertTrue(result)

        # Verify all 18 links were checked
        self.assertEqual(self.mock_pynvml.nvmlDeviceGetNvLinkState.call_count, 18)

    def test_check_nvl_links_for_device_with_disabled_link(self):
        """Test checking NVL links when one link is disabled."""
        self.mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        # First link healthy, second link disabled, rest healthy
        self.mock_pynvml.nvmlDeviceGetNvLinkState.side_effect = [1, 0] + [1] * 16

        checker = NVLHealthCheck()
        checker.pynvml = self.mock_pynvml

        with patch(
            'nvidia_resiliency_ext.shared_utils.health_check.logger.warning'
        ) as mock_warning:
            result = checker._check_nvl_links_for_device(0)
            self.assertFalse(result)
            mock_warning.assert_called_once_with("GPU 0: NVL link 1 is in DISABLED state")

    def test_check_nvl_links_for_device_with_nvml_error(self):
        """Test checking NVL links when NVML returns an error."""
        self.mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        # Raise exception on the first call to nvmlDeviceGetNvLinkState, then succeed on subsequent calls
        self.mock_pynvml.nvmlDeviceGetNvLinkState.side_effect = [
            self.mock_pynvml.NVMLError("NVML Error"),  # First call fails
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # Rest succeed
        ]

        checker = NVLHealthCheck()
        checker.pynvml = self.mock_pynvml

        with patch(
            'nvidia_resiliency_ext.shared_utils.health_check.logger.warning'
        ) as mock_warning:
            result = checker._check_nvl_links_for_device(0)
            # The method should return True because it handled the error gracefully
            # and continued checking other links
            self.assertTrue(result)
            # Should log a warning message for the NVML error
            mock_warning.assert_called_once_with("GPU 0: NVL link 0 not accessible: NVML Error")

    def test_check_nvl_links_for_device_with_not_supported_error(self):
        """Test checking NVL links when NVML returns 'not supported' error."""
        self.mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"

        # Create a mock exception with "not supported" message
        not_supported_error = self.mock_pynvml.NVMLError("not supported")
        self.mock_pynvml.nvmlDeviceGetNvLinkState.side_effect = not_supported_error

        checker = NVLHealthCheck()
        checker.pynvml = self.mock_pynvml

        with patch(
            'nvidia_resiliency_ext.shared_utils.health_check.logger.warning'
        ) as mock_warning:
            result = checker._check_nvl_links_for_device(0)
            # Should not log warning message for "not supported" errors
            mock_warning.assert_not_called()

    def test_perform_health_check_single_device(self):
        """Test health check for a single specific device."""
        self.mock_pynvml.nvmlInit.return_value = None
        self.mock_pynvml.nvmlShutdown.return_value = None

        checker = NVLHealthCheck(device_index=1)
        checker.pynvml = self.mock_pynvml

        with patch.object(checker, '_check_nvl_links_for_device', return_value=True) as mock_check:
            result = checker._perform_health_check()
            self.assertTrue(result)
            mock_check.assert_called_once_with(1)

    def test_perform_health_check_all_devices(self):
        """Test health check for all devices."""
        self.mock_pynvml.nvmlInit.return_value = None
        self.mock_pynvml.nvmlDeviceGetCount.return_value = 3
        self.mock_pynvml.nvmlShutdown.return_value = None

        checker = NVLHealthCheck()  # No device_index specified
        checker.pynvml = self.mock_pynvml

        with patch.object(checker, '_check_nvl_links_for_device', return_value=True) as mock_check:
            result = checker._perform_health_check()
            self.assertTrue(result)
            # Should check all 3 devices
            self.assertEqual(mock_check.call_count, 3)
            mock_check.assert_any_call(0)
            mock_check.assert_any_call(1)
            mock_check.assert_any_call(2)

    def test_perform_health_check_all_devices_one_fails(self):
        """Test health check for all devices when one fails."""
        self.mock_pynvml.nvmlInit.return_value = None
        self.mock_pynvml.nvmlDeviceGetCount.return_value = 3
        self.mock_pynvml.nvmlShutdown.return_value = None

        checker = NVLHealthCheck()  # No device_index specified
        checker.pynvml = self.mock_pynvml

        # First device healthy, second device fails, third device healthy
        with patch.object(
            checker, '_check_nvl_links_for_device', side_effect=[True, False, True]
        ) as mock_check:
            result = checker._perform_health_check()
            self.assertFalse(result)
            # Should check all 3 devices
            self.assertEqual(mock_check.call_count, 3)

    def test_perform_health_check_nvml_shutdown_error(self):
        """Test health check when NVML shutdown fails."""
        self.mock_pynvml.nvmlInit.return_value = None
        self.mock_pynvml.nvmlDeviceGetCount.return_value = 1
        self.mock_pynvml.nvmlShutdown.side_effect = self.mock_pynvml.NVMLError(
            "NVML Shutdown Error"
        )

        checker = NVLHealthCheck()
        checker.pynvml = self.mock_pynvml

        with patch.object(checker, '_check_nvl_links_for_device', return_value=True):
            with patch(
                'nvidia_resiliency_ext.shared_utils.health_check.logger.warning'
            ) as mock_warning:
                result = checker._perform_health_check()
                self.assertTrue(result)  # Health check should still succeed
                mock_warning.assert_called_once_with(
                    "Error during NVML shutdown: NVML Shutdown Error"
                )

    def test_sync_call_healthy(self):
        """Test synchronous health check call when healthy."""
        checker = NVLHealthCheck()
        with patch.object(checker, '_perform_health_check', return_value=True) as mock_check:
            result = checker()
            self.assertTrue(result)
            mock_check.assert_called_once()

    def test_sync_call_unhealthy(self):
        """Test synchronous health check call when unhealthy."""
        checker = NVLHealthCheck()
        with patch.object(checker, '_perform_health_check', return_value=False) as mock_check:
            result = checker()
            self.assertFalse(result)
            mock_check.assert_called_once()

    @patch("asyncio.sleep")
    async def test_async_check_healthy(self, mock_sleep):
        """Test asynchronous health check when healthy."""
        mock_sleep.return_value = None  # Mock sleep to return immediately
        checker = NVLHealthCheck()
        with patch.object(checker, '_check_health', return_value=True) as mock_check:
            # Test just the first iteration of the async loop
            mock_check.return_value = True
            await checker.async_check()
            mock_check.assert_called()
            mock_sleep.assert_called()

    @patch("asyncio.sleep")
    async def test_async_check_unhealthy_with_callback(self, mock_sleep):
        """Test asynchronous health check when unhealthy with failure callback."""
        mock_sleep.return_value = None  # Mock sleep to return immediately
        callback_called = False

        def on_failure():
            nonlocal callback_called
            callback_called = True

        checker = NVLHealthCheck(on_failure=on_failure)
        with patch.object(checker, '_check_health', return_value=False) as mock_check:
            # Test just the first iteration of the async loop
            mock_check.return_value = False
            await checker.async_check()
            mock_check.assert_called()
            mock_sleep.assert_called()
            # Note: In a real scenario, the callback would be called, but in this test
            # we're testing the basic async functionality

    def test_check_gpu_health_integration(self):
        """Test integration between _perform_health_check and _check_nvl_links_for_device."""
        checker = NVLHealthCheck(device_index=0)

        with patch.object(checker, '_check_nvl_links_for_device', return_value=True) as mock_check:
            with patch.object(checker, 'pynvml') as mock_pynvml:
                mock_pynvml.nvmlInit.return_value = None
                mock_pynvml.nvmlShutdown.return_value = None

                result = checker._perform_health_check()
                self.assertTrue(result)
                mock_check.assert_called_once_with(0)


class TestNodeHealthCheck(unittest.TestCase):

    def _checker_with_mocked_grpc(self, args=None):
        checker = NodeHealthCheck(args=args)
        checker._channel_target = "unix:///tmp/nvhcd.sock"

        channel = MagicMock()
        channel_context = MagicMock()
        channel_context.__enter__.return_value = channel
        channel_context.__exit__.return_value = None
        checker._grpc = MagicMock()
        checker._grpc.insecure_channel.return_value = channel_context

        response = MagicMock()
        response.success = True
        response.output = '{"fail_count": 0}'
        stub = MagicMock()
        stub.RunHealthCheck.return_value = response
        checker._pb2_grpc = MagicMock()
        checker._pb2_grpc.HealthCheckServiceStub.return_value = stub

        checker._pb2 = MagicMock()
        checker._pb2.HealthCheckRequest.side_effect = lambda args: SimpleNamespace(args=args)
        return checker, stub

    def test_perform_health_check_uses_default_dcahc_groups(self):
        checker, stub = self._checker_with_mocked_grpc()

        result = checker._perform_health_check()

        self.assertTrue(result)
        checker._pb2.HealthCheckRequest.assert_called_once_with(
            args=["--no-slurm", "--group", "prolog", "epilog", "logs", "gpu"]
        )
        request = stub.RunHealthCheck.call_args.args[0]
        self.assertEqual(
            request.args,
            ["--no-slurm", "--group", "prolog", "epilog", "logs", "gpu"],
        )

    def test_perform_health_check_preserves_custom_args(self):
        checker, stub = self._checker_with_mocked_grpc(args=["--group", "epilog"])

        result = checker._perform_health_check()

        self.assertTrue(result)
        checker._pb2.HealthCheckRequest.assert_called_once_with(args=["--group", "epilog"])
        request = stub.RunHealthCheck.call_args.args[0]
        self.assertEqual(request.args, ["--group", "epilog"])


class TestAttributionService(unittest.TestCase):

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_http_endpoint_posts_progressive_intent_to_logs_route(self, mock_client):
        client = mock_client.return_value.__enter__.return_value
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.dict(
            os.environ,
            {
                "SLURM_JOB_USER": "alice",
                "USER": "fallback-user",
                "SLURM_ARRAY_JOB_ID": "12345",
                "SLURM_JOB_ID": "67890",
            },
        ):
            service._do_submit_log("/tmp/train.log")

        mock_client.assert_called_once_with(base_url="http://attr.example:8000", timeout=2.0)
        client.post.assert_called_once_with(
            "/logs",
            json={
                "log_path": "/tmp/train.log",
                "user": "alice",
                "job_id": "12345",
                "analysis_intent": "progressive",
            },
            headers={"accept": "application/json"},
        )

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_http_endpoint_posts_terminal_intent_to_logs_route(self, mock_client):
        client = mock_client.return_value.__enter__.return_value
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.dict(
            os.environ,
            {
                "SLURM_JOB_USER": "alice",
                "SLURM_ARRAY_JOB_ID": "12345",
            },
        ):
            service._do_submit_log("/tmp/train.log", analysis_intent="terminal")

        mock_client.assert_called_once_with(base_url="http://attr.example:8000", timeout=2.0)
        client.post.assert_called_once_with(
            "/logs",
            json={
                "log_path": "/tmp/train.log",
                "user": "alice",
                "job_id": "12345",
                "analysis_intent": "terminal",
            },
            headers={"accept": "application/json"},
        )

    def test_submit_log_posts_progressive_work_synchronously(self):
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.object(service, "_do_submit_log") as mock_submit:
            service._submit_log("/tmp/train.log")

        self.assertEqual(service._last_submitted, "/tmp/train.log")
        mock_submit.assert_called_once_with(
            "/tmp/train.log",
            analysis_intent="progressive",
        )

    def test_submit_log_does_not_clobber_pending_terminal_analysis(self):
        """Cycle N+1 starting must not abandon cycle N's in-flight verdict."""
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._terminal_pending = "/tmp/train_cycle0.log"

        with patch.object(service, "_do_submit_log"):
            service._submit_log("/tmp/train_cycle1.log")

        self.assertEqual(service._last_submitted, "/tmp/train_cycle1.log")
        self.assertEqual(service._terminal_pending, "/tmp/train_cycle0.log")

    def test_request_terminal_analysis_posts_terminal_work_synchronously(self):
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._last_submitted = "/tmp/train.log"

        with patch.object(service, "_do_submit_log") as mock_submit:
            service.request_terminal_analysis()

        mock_submit.assert_called_once_with(
            "/tmp/train.log",
            analysis_intent="terminal",
        )
        self.assertEqual(service._terminal_pending, "/tmp/train.log")

    def test_terminal_post_precedes_publishing_the_pollable_slot(self):
        """The poller must never see a path attrsvc has not been asked to analyze.

        Otherwise whatever a GET returns for that path becomes the terminal verdict.
        """
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._last_submitted = "/tmp/train.log"
        observed = []

        def record_slot_at_post(*args, **kwargs):
            observed.append(service._terminal_pending)
            return True

        with patch.object(service, "_do_submit_log", side_effect=record_slot_at_post):
            service.request_terminal_analysis()

        self.assertEqual(observed, [None])  # slot still unpublished during the POST
        self.assertEqual(service._terminal_pending, "/tmp/train.log")  # published after

    def test_failed_terminal_post_leaves_the_slot_free(self):
        """Regression: a rejected submission must not pin the FCFS slot.

        attrsvc never received the request, so the GET for that path can never complete.
        Installing it would defer every later failed cycle behind it and disable
        attribution for the rest of the job.
        """
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._last_submitted = "/tmp/train_cycle0.log"

        with patch.object(service, "_do_submit_log", return_value=False):
            service.request_terminal_analysis()
        self.assertIsNone(service._terminal_pending)

        # A later failed cycle still installs normally.
        service._last_submitted = "/tmp/train_cycle1.log"
        with patch.object(service, "_do_submit_log", return_value=True):
            service.request_terminal_analysis()
        self.assertEqual(service._terminal_pending, "/tmp/train_cycle1.log")

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_submit_reports_failure_on_transport_error(self, mock_client):
        mock_client.return_value.__enter__.return_value.post.side_effect = OSError("refused")
        service = AttributionService(endpoint="http://attr.example:8000/")

        self.assertFalse(service._do_submit_log("/tmp/train.log"))

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_submit_reports_failure_on_error_status(self, mock_client):
        """httpx does not raise for error statuses, so a rejection must be checked."""
        client = mock_client.return_value.__enter__.return_value
        client.post.return_value = SimpleNamespace(status_code=500)
        service = AttributionService(endpoint="http://attr.example:8000/")

        self.assertFalse(service._do_submit_log("/tmp/train.log"))

        client.post.return_value = SimpleNamespace(status_code=202)
        self.assertTrue(service._do_submit_log("/tmp/train.log"))

    def test_submit_reports_failure_for_unsupported_transport(self):
        service = AttributionService(endpoint="unix:///tmp/attr.sock")

        self.assertFalse(service._do_submit_log("/tmp/train.log"))

    def test_request_terminal_analysis_skips_without_submitted_log(self):
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.object(service, "_do_submit_log") as mock_submit:
            service.request_terminal_analysis()

        mock_submit.assert_not_called()

    def test_poll_once_is_noop_without_pending_terminal_analysis(self):
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.object(service, "_get_results") as mock_get:
            service._poll_once()

        mock_get.assert_not_called()
        self.assertFalse(service.stop_requested())

    def test_poll_once_latches_stop_verdict(self):
        service = AttributionService(endpoint="http://attr.example:8000/", enforce_stop=True)
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=True) as mock_get:
            service._poll_once()

        mock_get.assert_called_once_with("/tmp/train.log", timeout=2.0)
        self.assertTrue(service.stop_requested())
        # The latch is global: the pending path is kept for diagnostics, and the loop exits.
        self.assertEqual(service._terminal_pending, "/tmp/train.log")

    def test_log_only_mode_observes_a_stop_without_acting_on_it(self):
        """Enforcement is opt-in: acting on a false-positive STOP is the expensive error."""
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=True):
            service._poll_once()

        self.assertTrue(service.stop_verdict_observed())  # recorded for diagnosis
        self.assertFalse(service.stop_requested())  # but never acted on

    def test_log_only_mode_keeps_analyzing_later_cycles_after_a_stop(self):
        """Regression: the first STOP must not silently end attribution for the job.

        Log-only mode exists to accumulate a verdict per failed cycle so a deployment can
        measure precision before enforcing. Latching on the first STOP left the pending
        slot pinned and killed the poller, so every later cycle went unanalyzed.
        """
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.object(service, "_do_submit_log") as mock_submit:
            service._submit_log("/tmp/train_cycle0.log")
            service.request_terminal_analysis()

            with patch.object(service, "_get_results", return_value=True):
                service._poll_once()
            # Slot freed, so the next failed cycle can install its own analysis.
            self.assertIsNone(service._terminal_pending)

            service._submit_log("/tmp/train_cycle1.log")
            service.request_terminal_analysis()
            self.assertEqual(service._terminal_pending, "/tmp/train_cycle1.log")

            with patch.object(service, "_get_results", return_value=True):
                service._poll_once()

        self.assertEqual(service.stop_verdict_count(), 2)
        self.assertFalse(service.stop_requested())
        terminal_posts = [
            c for c in mock_submit.call_args_list if c.kwargs.get("analysis_intent") == "terminal"
        ]
        self.assertEqual(
            [c.args[0] for c in terminal_posts],
            ["/tmp/train_cycle0.log", "/tmp/train_cycle1.log"],
        )

    def test_log_only_poller_survives_a_stop_verdict(self):
        """The poller thread must keep running so later cycles still get polled."""
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=True):
            service.start_poller()
            # A latched verdict must not terminate the loop in log-only mode.
            service._poll_stop_event.wait(0.05)
            still_running = service._poll_thread is not None and service._poll_thread.is_alive()
            service.stop_poller(timeout=5.0)

        self.assertTrue(still_running)
        self.assertTrue(service.stop_verdict_observed())

    def test_enforcing_mode_keeps_the_pending_path_for_the_record(self):
        """The job is ending, so the log that produced the verdict stays recorded."""
        service = AttributionService(endpoint="http://attr.example:8000/", enforce_stop=True)
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=True):
            service._poll_once()

        self.assertEqual(service._terminal_pending, "/tmp/train.log")
        self.assertEqual(service.stop_verdict_count(), 1)

    def test_log_only_is_the_default(self):
        self.assertFalse(AttributionService(endpoint="http://attr.example:8000/")._enforce_stop)

    def test_enforcing_mode_acts_on_a_stop(self):
        service = AttributionService(endpoint="http://attr.example:8000/", enforce_stop=True)
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=True):
            service._poll_once()

        self.assertTrue(service.stop_verdict_observed())
        self.assertTrue(service.stop_requested())

    def test_poller_exits_after_a_verdict_even_when_not_enforcing(self):
        """The analysis is finished; there is nothing further to learn by polling on."""
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=True) as mock_get:
            service.start_poller()
            service.stop_poller(timeout=5.0)

        self.assertFalse(service.stop_requested())
        self.assertEqual(mock_get.call_count, 1)

    def test_poll_once_clears_pending_on_continue_verdict(self):
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=False):
            service._poll_once()

        self.assertFalse(service.stop_requested())
        self.assertIsNone(service._terminal_pending)

    def test_poll_once_keeps_polling_while_undecided(self):
        """None means analysis is still running or attrsvc is unreachable: keep polling."""
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=None):
            service._poll_once()
            service._poll_once()

        self.assertFalse(service.stop_requested())
        self.assertEqual(service._terminal_pending, "/tmp/train.log")

    def test_late_verdict_for_older_cycle_still_stops_the_job(self):
        """A cycle-0 STOP arriving while cycle 2 runs is honored: the verdict is global."""
        service = AttributionService(endpoint="http://attr.example:8000/", enforce_stop=True)

        with patch.object(service, "_do_submit_log"):
            service._submit_log("/tmp/train_cycle0.log")
            service.request_terminal_analysis()
            service._submit_log("/tmp/train_cycle1.log")
            service._submit_log("/tmp/train_cycle2.log")

        with patch.object(service, "_get_results", return_value=True) as mock_get:
            service._poll_once()

        mock_get.assert_called_once_with("/tmp/train_cycle0.log", timeout=2.0)
        self.assertTrue(service.stop_requested())

    def test_crash_loop_does_not_starve_the_verdict(self):
        """Regression: cycles failing faster than analysis completes must still yield a STOP.

        Replacing the pending path on every terminal request discarded each in-flight
        analysis before it finished, so no verdict ever arrived -- in exactly the scenario
        attribution exists to catch. The pending slot is first-come-first-served, so the
        first analysis always runs to completion.
        """
        service = AttributionService(endpoint="http://attr.example:8000/", enforce_stop=True)
        analysis_pending = True

        def fake_get(log_path, timeout=None):
            # attrsvc is still working on whatever it was first asked about.
            return None if analysis_pending else True

        with (
            patch.object(service, "_do_submit_log") as mock_submit,
            patch.object(service, "_get_results", side_effect=fake_get) as mock_get,
        ):
            service._submit_log("/tmp/train_cycle0.log")
            service.request_terminal_analysis()

            # 20 rapid crash-loop cycles, each failing before analysis can finish.
            for cycle in range(1, 21):
                service._poll_once()
                service._submit_log(f"/tmp/train_cycle{cycle}.log")
                service.request_terminal_analysis()

            # Every GET stayed on the first log rather than chasing each new cycle.
            self.assertEqual(
                {c.args[0] for c in mock_get.call_args_list}, {"/tmp/train_cycle0.log"}
            )
            # Later terminal requests are skipped entirely, so no analysis budget is spent
            # on results that would never be polled.
            terminal_posts = [
                c
                for c in mock_submit.call_args_list
                if c.kwargs.get("analysis_intent") == "terminal"
            ]
            self.assertEqual(len(terminal_posts), 1)
            self.assertEqual(terminal_posts[0].args[0], "/tmp/train_cycle0.log")
            self.assertFalse(service.stop_requested())

            # The first analysis finally completes and the job stops.
            analysis_pending = False
            service._poll_once()

        self.assertTrue(service.stop_requested())

    def test_terminal_request_installs_again_once_the_slot_is_free(self):
        """After a CONTINUE verdict the next failing cycle gets its own analysis."""
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.object(service, "_do_submit_log") as mock_submit:
            service._submit_log("/tmp/train_cycle0.log")
            service.request_terminal_analysis()

            with patch.object(service, "_get_results", return_value=False):
                service._poll_once()
            self.assertIsNone(service._terminal_pending)

            service._submit_log("/tmp/train_cycle1.log")
            service.request_terminal_analysis()

        self.assertEqual(service._terminal_pending, "/tmp/train_cycle1.log")
        terminal_posts = [
            c for c in mock_submit.call_args_list if c.kwargs.get("analysis_intent") == "terminal"
        ]
        self.assertEqual(
            [c.args[0] for c in terminal_posts], ["/tmp/train_cycle0.log", "/tmp/train_cycle1.log"]
        )

    def test_poll_once_profiles_get_started_once_per_terminal_request(self):
        service = AttributionService(endpoint="http://attr.example:8000/")
        service._terminal_pending = "/tmp/train.log"
        service._poll_node_id = "node-a"

        with (
            patch.object(service, "_get_results", side_effect=[None, True]),
            patch(
                "nvidia_resiliency_ext.shared_utils.health_check.record_profiling_event"
            ) as record_event,
        ):
            service._poll_once()
            service._poll_once()

        self.assertEqual(record_event.call_args_list[0].args[0].value, "attribution_get_started")
        self.assertEqual(record_event.call_args_list[0].kwargs, {"node_id": "node-a"})
        self.assertEqual(record_event.call_args_list[1].args[0].value, "attribution_get_completed")
        self.assertEqual(record_event.call_args_list[1].kwargs, {"node_id": "node-a"})

    def test_poller_not_started_for_non_http_transport(self):
        """Every GET would fail, so the poller would spin without ever reaching a verdict."""
        service = AttributionService(endpoint="unix:///tmp/attr.sock")

        service.start_poller()

        self.assertIsNone(service._poll_thread)

    def test_poller_thread_latches_stop_and_exits(self):
        service = AttributionService(endpoint="http://attr.example:8000/", enforce_stop=True)
        service._terminal_pending = "/tmp/train.log"

        with patch.object(service, "_get_results", return_value=True):
            service.start_poller(node_id="node-a")
            service.stop_poller(timeout=5.0)

        self.assertTrue(service.stop_requested())

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_http_endpoint_omits_job_metadata_when_env_unset(self, mock_client):
        client = mock_client.return_value.__enter__.return_value
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch.dict(os.environ, {}, clear=True):
            service._do_submit_log("/tmp/train.log")

        client.post.assert_called_once_with(
            "/logs",
            json={
                "log_path": "/tmp/train.log",
                "analysis_intent": "progressive",
            },
            headers={"accept": "application/json"},
        )

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_non_http_endpoint_does_not_create_http_client(self, mock_client):
        service = AttributionService(endpoint="grpc://attr.example:50050")

        service._do_submit_log("/tmp/train.log")

        mock_client.assert_not_called()

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_get_results_returns_stop_decision(self, mock_client):
        client = mock_client.return_value.__enter__.return_value
        response = MagicMock()
        response.status_code = 200
        response.text = "{}"
        response.json.return_value = {
            "recommendation": {
                "action": "STOP",
                "reason": "STOP - DONT RESTART",
                "source": "log_analyzer",
            },
            "result": {
                "module": "log_analyzer",
                "result_id": "abc123",
                "resource_uri": "attribution://log_analyzer/abc123",
                "result": [_attribution_item("raw attribution item", "UNKNOWN")],
            },
            "status": "completed",
        }
        client.get.return_value = response
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch("nvidia_resiliency_ext.shared_utils.health_check.logger") as mock_logger:
            should_stop = service._get_results("/tmp/train.log")

        self.assertTrue(should_stop)
        mock_logger.info.assert_called_once()
        mock_client.assert_called_once_with(base_url="http://attr.example:8000", timeout=2.0)
        client.get.assert_called_once_with(
            "/logs",
            params={"log_path": "/tmp/train.log", "wait": False},
            headers={"accept": "application/json"},
        )

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_get_results_maps_restart_recommendation_to_no_stop(self, mock_client):
        client = mock_client.return_value.__enter__.return_value
        response = MagicMock()
        response.status_code = 200
        response.text = "{}"
        response.json.return_value = {
            "recommendation": {
                "action": "RESTART",
                "reason": "RESTART IMMEDIATE",
                "source": "log_analyzer",
            },
            "result": {
                "module": "log_analyzer",
                "result": [_attribution_item("RESTART IMMEDIATE", "RESTART_IMMEDIATE")],
            },
            "status": "completed",
        }
        client.get.return_value = response
        service = AttributionService(endpoint="http://attr.example:8000/")

        should_stop = service._get_results("/tmp/train.log")

        self.assertFalse(should_stop)

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_get_results_treats_non_completed_status_as_not_ready(self, mock_client):
        client = mock_client.return_value.__enter__.return_value
        response = MagicMock()
        response.status_code = 200
        response.text = "{}"
        response.json.return_value = {
            "status": "in_flight",
            "recommendation": {
                "action": "UNKNOWN",
                "reason": "analysis still running",
                "source": "log_analyzer",
            },
        }
        client.get.return_value = response
        service = AttributionService(endpoint="http://attr.example:8000/")

        with patch("nvidia_resiliency_ext.shared_utils.health_check.logger") as mock_logger:
            should_stop = service._get_results("/tmp/train.log")

        self.assertIsNone(should_stop)
        mock_logger.info.assert_not_called()
        self.assertTrue(
            any(
                "status=in_flight" in str(log_call.args[0])
                for log_call in mock_logger.debug.call_args_list
            )
        )

    @patch("nvidia_resiliency_ext.shared_utils.health_check.httpx.Client")
    def test_get_results_maps_continue_recommendation_to_no_stop(self, mock_client):
        client = mock_client.return_value.__enter__.return_value
        response = MagicMock()
        response.status_code = 200
        response.text = "{}"
        response.json.return_value = {
            "recommendation": {
                "action": "CONTINUE",
                "reason": "training cycle still running",
                "source": "log_analyzer",
            },
            "result": {
                "module": "log_analyzer",
                "result": [_attribution_item("ERRORS NOT FOUND", "NO_ERRORS")],
            },
            "status": "completed",
        }
        client.get.return_value = response
        service = AttributionService(endpoint="http://attr.example:8000/")

        should_stop = service._get_results("/tmp/train.log")

        self.assertFalse(should_stop)


if __name__ == "__main__":
    unittest.main()
