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

import contextlib
import os
import socket
import threading
from queue import Queue
from typing import Any, List

from .utils import read_obj_from_ipc_socket, write_object_to_ipc_socket


class _WorkerThreadStopReq:
    pass


class IpcConnector:
    """
    IpcConnector allows messages to be passed unidirectionally between two processes
    via a Unix domain socket. One process (the "receiver") should start listening using
    `.start_receiving`, while the other process (the "sender") can use `.send` to transmit
    messages. All messages are received in a background thread and stored.
    The receiver can call `.fetch_received` to retrieve and remove all messages received so far,
    or `.peek_received` to retrieve them without removal.
    A "message" can be any picklable Python object.
    """

    THREAD_FINISH_TIMEOUT = 60

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.received_messages = Queue()
        self.accepting_thread = None

    def start_receiving(self) -> None:
        """Starts the receiver thread."""
        if self.accepting_thread is not None:
            raise RuntimeError("start_receiving called while running")

        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)

        def _accepting_thread(server):
            stop_flag = False
            with server:
                while not stop_flag:
                    conn, _ = server.accept()
                    with conn:
                        data = read_obj_from_ipc_socket(conn, raise_exc=True)
                        if isinstance(data, _WorkerThreadStopReq):
                            stop_flag = True
                        else:
                            self.received_messages.put(data)
                        write_object_to_ipc_socket(b'ok', conn)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(1)

        self.accepting_thread = threading.Thread(
            target=_accepting_thread, args=(server,), daemon=True
        )
        self.accepting_thread.start()

    def send(self, message: Any) -> None:
        """Sends a message to the receiver

        Args:
            message (Any): The object that will be transmitted - it should be picklable.
        """
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(self.socket_path)
            write_object_to_ipc_socket(message, client)
            resp = read_obj_from_ipc_socket(client, raise_exc=True)
            if resp != b'ok':
                raise RuntimeError(f"Unexpected response. Should be 'ok' got {resp}")

    def fetch_received(self) -> List[Any]:
        """Fetch and remove all messages from the internal queue.

        Returns:
            List of messages retrieved from the background thread.
            Messages are removed from the queue as they are fetched.
        """
        messages = []
        while not self.received_messages.empty():
            messages.append(self.received_messages.get())
        return messages

    def peek_received(self) -> List[Any]:
        """Retrieve all messages received so far without removing them from the queue.

        Returns:
            List of messages retrieved from the background thread.
        """
        return list(self.received_messages.queue)

    def clear(self) -> None:
        """Clears received messages"""
        while not self.received_messages.empty():
            self.received_messages.get()

    def stop_receiving(self) -> None:
        """Stops the receiving thread"""
        if self.accepting_thread is not None:
            self.send(_WorkerThreadStopReq())
            self.accepting_thread.join(timeout=self.THREAD_FINISH_TIMEOUT)
            self.accepting_thread = None
            self.clear()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)
