# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort timeout captures and a node-local, single-writer collector.

Capture output is staged locally; only the collector publishes it to shared
storage. Debugger command files may be read from shared storage by the tools.
Completed captures are the queue; no socket protocol or metadata sidecar is required.
"""

from .capture_manager import TimeoutDumpManager
from .collector import DiagnosticCollectorProcess
from .config import TimeoutDumpConfig, diagnostics_directory
from .ptrace import authorize_monitor_ptrace

__all__ = [
    "DiagnosticCollectorProcess",
    "TimeoutDumpConfig",
    "TimeoutDumpManager",
    "authorize_monitor_ptrace",
    "diagnostics_directory",
]
