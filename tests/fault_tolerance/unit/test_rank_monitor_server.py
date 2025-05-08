import asyncio
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
from nvidia_resiliency_ext.fault_tolerance.utils import write_obj_to_ipc_stream, read_obj_from_ipc_stream


class TestRankMonitorServer(unittest.TestCase):
    def setUp(self):
        # Create temporary socket paths
        self.worker_socket_path = os.path.join(
            tempfile.gettempdir(), "test_rank_monitor_worker.socket"
        )

        # Create a basic config
        self.config = FaultToleranceConfig()

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
        # The launcher socket path is constructed as: f"{tempfile.gettempdir()}/_ft_launcher{pid}_to_rmon.socket"
        max_wait = 5  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            # Try to find the launcher socket file
            for file in os.listdir(tempfile.gettempdir()):
                if file.startswith("_ft_launcher") and file.endswith("_to_rmon.socket"):
                    self.launcher_socket_path = os.path.join(tempfile.gettempdir(), file)
                    return
            time.sleep(0.1)
        raise RuntimeError("Could not find launcher socket file after waiting")

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


if __name__ == '__main__':
    unittest.main()
