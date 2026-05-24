# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
from typing import Any

DEFAULT_RPC_TIMEOUT_S = 2.0
DEFAULT_MAX_RPC_BYTES = 16 * 1024 * 1024


def default_socket_path() -> str:
    return os.path.join(tempfile.gettempdir(), f"nvrx-fact-agent-{os.getuid()}.sock")


def json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def send_frame(conn: socket.socket, payload: dict[str, Any]) -> None:
    body = json_dumps(payload)
    conn.sendall(struct.pack("!I", len(body)) + body)


def recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(conn: socket.socket, *, max_bytes: int) -> dict[str, Any]:
    raw_size = recv_exact(conn, 4)
    size = struct.unpack("!I", raw_size)[0]
    if size > max_bytes:
        raise ValueError(f"UDS request is too large: {size} bytes > {max_bytes} bytes")
    body = recv_exact(conn, size)
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("UDS request must be a JSON object")
    return payload


def notify_fact_agent(
    *,
    socket_path: str,
    payload: dict[str, Any],
    timeout_s: float = DEFAULT_RPC_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_RPC_BYTES,
) -> dict[str, Any]:
    """Send one framed JSON RPC to the local FACT agent and return its ACK."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(socket_path)
        send_frame(sock, payload)
        return recv_frame(sock, max_bytes=max_bytes)
