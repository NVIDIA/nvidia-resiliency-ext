import asyncio
import logging
import os
import socket
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.multiprocessing

from nvidia_resiliency_ext.fault_tolerance.config import FaultToleranceConfig
from nvidia_resiliency_ext.fault_tolerance.data import (
    AuthkeyMsg,
    HeartbeatMsg,
    InitMsg,
    OkMsg,
    RankInfo,
    SectionAction,
    SectionMsg,
)
from nvidia_resiliency_ext.fault_tolerance.rank_monitor_server import RankMonitorServer
from nvidia_resiliency_ext.fault_tolerance.utils import (
    read_obj_from_ipc_stream,
    write_obj_to_ipc_stream,
)


class TestRankMonitorServer(unittest.TestCase):
    def setUp(self):
        # Create temporary socket paths
        self.worker_socket_path = os.path.join(
            tempfile.gettempdir(), "test_rank_monitor_worker.socket"
        )

        # Create a basic config with skip_section_response=False for testing
        # (tests expect to receive responses)
        # Also add test_section to rank_section_timeouts to avoid KeyError
        self.config = FaultToleranceConfig(
            skip_section_response=False, rank_section_timeouts={"test_section": 60.0}
        )

        # Create a mock logger
        self.logger = MagicMock()

        # Start the server in a subprocess
        self.server_process = RankMonitorServer.run_in_subprocess(
            cfg=self.config,
            ipc_socket_path=self.worker_socket_path,
            is_restarter_logger=False,
            mp_ctx=torch.multiprocessing,
        )

        # Wait for the server to start and create its launcher socket
        # The launcher socket path is constructed as: f"{tempfile.gettempdir()}/_ft_launcher{server_pid}_to_rmon.socket"
        server_pid = self.server_process.pid
        expected_socket_name = f"_ft_launcher{server_pid}_to_rmon.socket"
        self.launcher_socket_path = os.path.join(tempfile.gettempdir(), expected_socket_name)

        max_wait = 5  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if os.path.exists(self.launcher_socket_path):
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"Could not find launcher socket file {expected_socket_name} after waiting {max_wait} seconds"
        )

    def tearDown(self):
        # Clean up the server process
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.join(timeout=180)

        # Clean up the socket files
        for socket_path in [self.worker_socket_path, self.launcher_socket_path]:
            if os.path.exists(socket_path):
                os.unlink(socket_path)

    async def _setup_worker_connection(self):
        # Create a test connection to worker socket
        reader, writer = await asyncio.open_unix_connection(self.worker_socket_path)
        return reader, writer

    async def _setup_launcher_connection(self):
        # Create a test connection to launcher socket
        reader, writer = await asyncio.open_unix_connection(self.launcher_socket_path)
        return reader, writer

    async def _send_authkey_msg(self, writer):
        # Send authkey message
        authkey_msg = AuthkeyMsg()
        await write_obj_to_ipc_stream(authkey_msg, writer)

    async def _send_init_msg(self, writer, rank_info):
        # Send init message
        init_msg = InitMsg()
        init_msg.rank_info = rank_info
        await write_obj_to_ipc_stream(init_msg, writer)

    async def _send_heartbeat_msg(self, writer):
        # Send heartbeat message
        heartbeat_msg = HeartbeatMsg(rank=0)
        await write_obj_to_ipc_stream(heartbeat_msg, writer)

    async def _send_section_msg(self, writer, section="test_section", action=SectionAction.OPEN):
        # Send section message
        section_msg = SectionMsg(rank=0, section=section, action=action)
        await write_obj_to_ipc_stream(section_msg, writer)

    def test_initial_state(self):
        """Test the initial state of the server"""

        async def run_test():
            reader, writer = await self._setup_worker_connection()

            # Send authkey first
            await self._send_authkey_msg(writer)
            await reader.readexactly(4)  # Skip authkey response

            # Send init message
            rank_info = RankInfo(
                global_rank=0, local_rank=0, host=socket.gethostname(), pid=os.getpid()
            )
            await self._send_init_msg(writer, rank_info)
            await reader.readexactly(4)  # Skip init response

            writer.close()
            await writer.wait_closed()

        asyncio.run(run_test())

    def test_authkey_handling(self):
        """Test handling of authkey message"""

        async def run_test():
            reader, writer = await self._setup_worker_connection()
            await self._send_authkey_msg(writer)

            # Read and verify response
            response = await read_obj_from_ipc_stream(reader)
            self.assertIsInstance(response, OkMsg)

            writer.close()
            await writer.wait_closed()

        asyncio.run(run_test())

    def test_init_handling(self):
        """Test handling of init message"""

        async def run_test():
            reader, writer = await self._setup_worker_connection()

            # Send authkey first
            await self._send_authkey_msg(writer)
            await read_obj_from_ipc_stream(reader)  # Skip authkey response

            # Send init message
            rank_info = RankInfo(
                global_rank=0, local_rank=0, host=socket.gethostname(), pid=os.getpid()
            )
            await self._send_init_msg(writer, rank_info)

            # Read and verify response
            response = await read_obj_from_ipc_stream(reader)
            self.assertIsInstance(response, OkMsg)
            self.assertIsNotNone(response.cfg)

            writer.close()
            await writer.wait_closed()

        asyncio.run(run_test())

    def test_heartbeat_handling(self):
        """Test handling of heartbeat message"""

        async def run_test():
            reader, writer = await self._setup_worker_connection()

            # Send authkey first
            await self._send_authkey_msg(writer)
            await read_obj_from_ipc_stream(reader)  # Skip authkey response

            # Send init message
            rank_info = RankInfo(
                global_rank=0, local_rank=0, host=socket.gethostname(), pid=os.getpid()
            )
            await self._send_init_msg(writer, rank_info)
            await read_obj_from_ipc_stream(reader)  # Skip init response

            # Send heartbeat
            await self._send_heartbeat_msg(writer)

            # Read and verify response
            response = await read_obj_from_ipc_stream(reader)
            self.assertIsInstance(response, OkMsg)

            writer.close()
            await writer.wait_closed()

        asyncio.run(run_test())

    def test_section_handling(self):
        """Test handling of section messages"""

        async def run_test():
            reader, writer = await self._setup_worker_connection()

            # Send authkey first
            await self._send_authkey_msg(writer)
            await read_obj_from_ipc_stream(reader)  # Skip authkey response

            # Send init message
            rank_info = RankInfo(
                global_rank=0, local_rank=0, host=socket.gethostname(), pid=os.getpid()
            )
            await self._send_init_msg(writer, rank_info)
            await read_obj_from_ipc_stream(reader)  # Skip init response

            # Send section start
            await self._send_section_msg(writer, "test_section", SectionAction.OPEN)

            # Read and verify response
            response = await read_obj_from_ipc_stream(reader)
            self.assertIsInstance(response, OkMsg)

            # Send section end
            await self._send_section_msg(writer, "test_section", SectionAction.CLOSE)

            # Read and verify response
            response = await read_obj_from_ipc_stream(reader)
            self.assertIsInstance(response, OkMsg)

            writer.close()
            await writer.wait_closed()

        asyncio.run(run_test())

    def test_launcher_ipc_connection(self):
        """Test handling of launcher IPC connection"""

        async def run_test():
            # First establish a worker connection
            worker_reader, worker_writer = await self._setup_worker_connection()

            # Send authkey first
            await self._send_authkey_msg(worker_writer)
            await read_obj_from_ipc_stream(worker_reader)  # Skip authkey response

            # Send init message
            rank_info = RankInfo(
                global_rank=0, local_rank=0, host=socket.gethostname(), pid=os.getpid()
            )
            await self._send_init_msg(worker_writer, rank_info)
            await read_obj_from_ipc_stream(worker_reader)  # Skip init response

            # Send a heartbeat to ensure server is in INITIALIZE state
            await self._send_heartbeat_msg(worker_writer)
            await read_obj_from_ipc_stream(worker_reader)  # Skip heartbeat response

            # Now establish launcher connection and send close message
            launcher_reader, launcher_writer = await self._setup_launcher_connection()
            try:
                await write_obj_to_ipc_stream("close_worker_ipc_connection", launcher_writer)
            finally:
                # We need to close the launcher connection ourselves
                launcher_writer.close()
                await launcher_writer.wait_closed()

            # Verify server state after connection is closed
            # Try to establish a new worker connection - should succeed because server released the lock
            new_worker_reader, new_worker_writer = await self._setup_worker_connection()
            try:
                # Send authkey to verify server accepts new connection
                await self._send_authkey_msg(new_worker_writer)
                await read_obj_from_ipc_stream(new_worker_reader)  # Skip authkey response
            finally:
                new_worker_writer.close()
                await new_worker_writer.wait_closed()

        asyncio.run(run_test())

    def test_connection_lost_handling(self):
        """Test handling of connection loss"""

        async def run_test():
            reader, writer = await self._setup_worker_connection()

            # Send authkey first
            await self._send_authkey_msg(writer)
            await read_obj_from_ipc_stream(reader)  # Skip authkey response

            # Send init message
            rank_info = RankInfo(
                global_rank=0, local_rank=0, host=socket.gethostname(), pid=os.getpid()
            )
            await self._send_init_msg(writer, rank_info)
            await read_obj_from_ipc_stream(reader)  # Skip init response

            # Close connection abruptly
            writer.close()
            await writer.wait_closed()

        asyncio.run(run_test())

    def test_skip_section_response_enabled(self):
        """Test unidirectional communication when skip_section_response=True"""
        # Create a new server with skip_section_response=True (default behavior)
        config_unidirectional = FaultToleranceConfig(
            skip_section_response=True, rank_section_timeouts={"test_section": 60.0}
        )

        worker_socket_path_unidirectional = os.path.join(
            tempfile.gettempdir(), "test_rank_monitor_worker_unidirectional.socket"
        )

        server_process = RankMonitorServer.run_in_subprocess(
            cfg=config_unidirectional,
            ipc_socket_path=worker_socket_path_unidirectional,
            is_restarter_logger=False,
            mp_ctx=torch.multiprocessing,
        )

        try:
            # Wait for server to start
            time.sleep(0.5)

            async def run_test():
                reader, writer = await asyncio.open_unix_connection(
                    worker_socket_path_unidirectional
                )

                # Send authkey first
                await self._send_authkey_msg(writer)
                await read_obj_from_ipc_stream(reader)  # Authkey still gets response

                # Send init message
                rank_info = RankInfo(
                    global_rank=0, local_rank=0, host=socket.gethostname(), pid=os.getpid()
                )
                await self._send_init_msg(writer, rank_info)
                response = await read_obj_from_ipc_stream(reader)  # Init still gets response
                self.assertIsInstance(response, OkMsg)

                # Send heartbeat - should NOT receive response when skip_section_response=True
                await self._send_heartbeat_msg(writer)

                # Try to read with a timeout - should timeout since no response is sent
                try:
                    response = await asyncio.wait_for(read_obj_from_ipc_stream(reader), timeout=0.5)
                    # If we get here, the test should fail because we shouldn't receive a response
                    self.fail("Expected no response for heartbeat when skip_section_response=True")
                except asyncio.TimeoutError:
                    # This is expected - no response sent
                    pass

                # Send section message - should also NOT receive response
                await self._send_section_msg(writer, "test_section", SectionAction.OPEN)

                try:
                    response = await asyncio.wait_for(read_obj_from_ipc_stream(reader), timeout=0.5)
                    self.fail(
                        "Expected no response for section message when skip_section_response=True"
                    )
                except asyncio.TimeoutError:
                    # This is expected - no response sent
                    pass

                # Send another heartbeat to verify server is still processing messages
                await self._send_heartbeat_msg(writer)

                # Close section
                await self._send_section_msg(writer, "test_section", SectionAction.CLOSE)

                # Clean up
                writer.close()
                await writer.wait_closed()

            asyncio.run(run_test())

        finally:
            # Clean up
            server_process.terminate()
            server_process.join(timeout=5)
            if os.path.exists(worker_socket_path_unidirectional):
                os.unlink(worker_socket_path_unidirectional)


class TestRankMonitorServerWarmupAndStepSections(unittest.TestCase):
    """Unit tests for warmup/step section logic: _step_section_seen_this_cycle, iteration tracking, and warmup timeout."""

    def _create_server(
        self, rank_section_timeouts, num_warmup_iterations=3, rank_out_of_section_timeout=10.0
    ):
        with patch(
            "nvidia_resiliency_ext.fault_tolerance.rank_monitor_server.GPUHealthCheck",
            MagicMock(),
        ):
            config = FaultToleranceConfig(
                skip_section_response=True,
                rank_section_timeouts=rank_section_timeouts,
                num_warmup_iterations=num_warmup_iterations,
                rank_out_of_section_timeout=rank_out_of_section_timeout,
            )
            event = torch.multiprocessing.get_context().Event()
            logger = MagicMock()
            logger.level = logging.INFO
            server = RankMonitorServer(
                cfg=config,
                ipc_socket_path=tempfile.mktemp(suffix=".socket"),
                rank_monitor_ready_event=event,
                logger=logger,
                is_restarter_logger=False,
            )
            return server

    def test_out_of_section_skip_when_warmup_not_in_config_and_step_not_seen(self):
        """Out-of-section timeout is skipped when warmup not in config and first step not seen."""
        server = self._create_server(
            rank_section_timeouts={"setup": 60.0, "step": 30.0},
            rank_out_of_section_timeout=10.0,
        )
        server.open_sections = {}
        server.out_of_section_time = time.monotonic() - 1000
        server._step_section_seen_this_cycle = False
        curr_time = time.monotonic()
        self.assertFalse(server._is_section_timeout_elapsed(curr_time))

    def test_out_of_section_checked_when_warmup_not_in_config_but_step_seen(self):
        """Out-of-section timeout is enforced when warmup not in config but step was already seen."""
        server = self._create_server(
            rank_section_timeouts={"setup": 60.0, "step": 30.0},
            rank_out_of_section_timeout=10.0,
        )
        server.open_sections = {}
        server.out_of_section_time = time.monotonic() - 1000
        server._step_section_seen_this_cycle = True
        curr_time = time.monotonic()
        self.assertTrue(server._is_section_timeout_elapsed(curr_time))

    def test_out_of_section_checked_when_warmup_in_config(self):
        """Out-of-section timeout is enforced when warmup is in config (no skip)."""
        server = self._create_server(
            rank_section_timeouts={"setup": 60.0, "step": 30.0, "warmup": 120.0},
            rank_out_of_section_timeout=10.0,
        )
        server.open_sections = {}
        server.out_of_section_time = time.monotonic() - 1000
        server._step_section_seen_this_cycle = False
        curr_time = time.monotonic()
        self.assertTrue(server._is_section_timeout_elapsed(curr_time))

    def test_step_section_uses_warmup_timeout_when_iteration_in_warmup_range(self):
        """When step is open and iteration is in warmup range, warmup section timeout is used."""
        server = self._create_server(
            rank_section_timeouts={"step": 5.0, "warmup": 100.0},
            num_warmup_iterations=3,
        )
        curr_time = time.monotonic()
        server.open_sections = {"step": curr_time - 50}  # 50s elapsed
        server._initial_iteration = 0
        server._current_iteration = 1  # within warmup (0, 1, 2)
        # 50 < 100 (warmup) so not elapsed
        self.assertFalse(server._is_section_timeout_elapsed(curr_time))
        # 150s elapsed > 100 (warmup)
        server.open_sections = {"step": curr_time - 150}
        self.assertTrue(server._is_section_timeout_elapsed(curr_time))

    def test_step_section_uses_step_timeout_when_iteration_past_warmup(self):
        """When step is open and iteration is past warmup range, step section timeout is used."""
        server = self._create_server(
            rank_section_timeouts={"step": 5.0, "warmup": 100.0},
            num_warmup_iterations=3,
        )
        curr_time = time.monotonic()
        server.open_sections = {"step": curr_time - 10}  # 10s elapsed
        server._initial_iteration = 0
        server._current_iteration = 5  # past warmup
        # 10 > 5 (step timeout) so elapsed
        self.assertTrue(server._is_section_timeout_elapsed(curr_time))

    def test_iteration_not_updated_from_setup_section(self):
        """Iteration is only updated from 'step' section messages, not 'setup'."""
        server = self._create_server(
            rank_section_timeouts={"setup": 60.0, "step": 30.0},
        )

        async def run_test():
            writer = MagicMock()
            msg = SectionMsg(rank=0, section="setup", action=SectionAction.OPEN, iteration=0)
            await server._handle_section_msg(msg, writer)
            self.assertIsNone(server._initial_iteration)
            self.assertIsNone(server._current_iteration)

        asyncio.run(run_test())

    def test_iteration_and_step_seen_updated_on_step_open(self):
        """Opening 'step' section with iteration sets iteration fields and _step_section_seen_this_cycle."""
        server = self._create_server(
            rank_section_timeouts={"setup": 60.0, "step": 30.0},
        )

        async def run_test():
            writer = MagicMock()
            msg = SectionMsg(rank=0, section="step", action=SectionAction.OPEN, iteration=5)
            await server._handle_section_msg(msg, writer)
            self.assertEqual(server._initial_iteration, 5)
            self.assertEqual(server._current_iteration, 5)
            self.assertTrue(server._step_section_seen_this_cycle)

        asyncio.run(run_test())

    def test_connection_lost_resets_step_seen_and_iteration(self):
        """_handle_ipc_connection_lost resets _step_section_seen_this_cycle and iteration fields."""
        server = self._create_server(
            rank_section_timeouts={"step": 30.0},
        )
        server._initial_iteration = 0
        server._current_iteration = 10
        server._step_section_seen_this_cycle = True
        server._handle_ipc_connection_lost()
        self.assertIsNone(server._initial_iteration)
        self.assertIsNone(server._current_iteration)
        self.assertFalse(server._step_section_seen_this_cycle)


if __name__ == '__main__':
    unittest.main()
