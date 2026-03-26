# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify exceptions from MCP stdio transport / subprocess I/O for reconnect policy."""

from __future__ import annotations

import asyncio
import errno

try:
    from anyio import BrokenResourceError, ClosedResourceError

    _ANYIO_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
        ClosedResourceError,
        BrokenResourceError,
    )
except ImportError:  # pragma: no cover
    _ANYIO_TRANSPORT_ERRORS = ()


def is_mcp_connection_error(exc: BaseException) -> bool:
    """True if ``exc`` suggests the MCP stdio session or subprocess I/O is dead.

    Used by :meth:`nvidia_resiliency_ext.attribution.mcp_integration.mcp_client.NVRxMCPClient.run_module_resilient`
    to decide when to :meth:`~nvidia_resiliency_ext.attribution.mcp_integration.mcp_client.NVRxMCPClient.reconnect`.
    """
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (errno.EPIPE, errno.ECONNRESET):
        return True
    if isinstance(exc, asyncio.IncompleteReadError):
        return True
    if _ANYIO_TRANSPORT_ERRORS and isinstance(exc, _ANYIO_TRANSPORT_ERRORS):
        return True
    return False
