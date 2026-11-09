# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort unary gRPC calls that never block the caller.

Kept free of torch so the standalone gRPC log servers can import it.
"""

import logging
import threading
import time
from typing import Any, Optional

import grpc


class AsyncUnarySender:
    """Starts unary gRPC calls without waiting for them to complete.

    `send` starts the call (``rpc.future``) and returns; the channel's threads complete
    it. A failed call is logged and its request dropped. At most `max_pending` calls are
    in flight: while the receiver is slow or unreachable, further requests are dropped
    rather than queued. The caller owns the channel; `close` only waits for pending calls.
    """

    def __init__(
        self,
        logger: logging.Logger,
        rpc_timeout: float = 5.0,
        max_pending: int = 64,
        close_timeout: float = 1.0,
    ):
        self._logger = logger
        self._rpc_timeout = rpc_timeout
        self._max_pending = max_pending
        self._close_timeout = close_timeout
        self._lock = threading.Lock()
        self._pending = set()
        self._closed = False

    def send(self, rpc: Any, request: Any, what: str) -> bool:
        """Start `rpc(request)`; `what` names the request in log messages.

        Returns:
            True if the call was started, False if the request was dropped.
        """
        try:
            with self._lock:
                if self._closed:
                    self._logger.debug(f"Sender closed, dropped the {what}")
                    return False
                if len(self._pending) >= self._max_pending:
                    self._logger.warning(
                        f"{len(self._pending)} calls still in flight, dropped the {what}"
                    )
                    return False
                future = rpc.future(request, timeout=self._rpc_timeout)
                self._pending.add(future)
            # Outside the lock: runs immediately if the call has already completed.
            future.add_done_callback(lambda f: self._on_done(f, what))
            return True
        except Exception as e:
            self._logger.warning(f"Failed to send the {what}: {e!r}")
            return False

    def close(self, timeout: Optional[float] = None) -> None:
        """Wait up to `timeout` (default `close_timeout`) for pending calls, then cancel them."""
        with self._lock:
            self._closed = True
            pending = list(self._pending)
        timeout = self._close_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        for future in pending:
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except grpc.FutureTimeoutError:
                future.cancel()
            except (grpc.RpcError, grpc.FutureCancelledError):
                pass  # logged by _on_done

    def _on_done(self, future: Any, what: str) -> None:
        with self._lock:
            self._pending.discard(future)
        if future.cancelled():
            self._logger.warning(f"Cancelled sending the {what}")
            return
        error = future.exception()
        if error is None:
            return
        if isinstance(error, grpc.RpcError):
            error = f"{error.code().name}: {error.details()}"
        self._logger.warning(f"Failed to send the {what}: {error}")
