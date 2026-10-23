# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeout dump state for one rank monitor."""

import time

from .capture import staging_dir
from .config import create_capture


class TimeoutDumpMonitor:
    """Captures a timed-out worker at most once per incarnation, one tool at a time.

    The last diagnostic child is kept until its process exits, so a debugger that
    outlives its capture budget blocks later attachments instead of overlapping them.
    """

    def __init__(self, cfg, logger):
        self.cfg = cfg  # TimeoutDumpConfig
        self.logger = logger
        self.attempted = False  # this worker incarnation has already been captured
        self.child = None  # last capture; kept until its process exits
        self._incarnation = None

    def on_worker_init(self, cycle, pid):
        """A worker connected; a new (cycle, PID) incarnation gets its own attempt."""
        if (cycle, pid) != self._incarnation:
            self.attempted = False
            self._incarnation = (cycle, pid)

    def reasons(self, timed_out_sections):
        """The timed-out sections that are selected for capture."""
        return set(timed_out_sections) & set(self.cfg.sections)

    def reap_finished_child(self):
        if self.child is not None and not self.child.is_running():
            self.child = None

    async def capture(self, rank_info, still_current, captures):
        """Run each tool in turn within the capture budget, appending to `captures`.

        Appending as it goes keeps partial results if the monitor is cancelled.
        Stops early once `still_current()` is false (a new worker connected).
        """
        if self.attempted:
            return
        self.attempted = True
        if self.child is not None and self.child.is_running():
            self.logger.warning(
                "Skipping timeout capture: previous diagnostic child is still alive"
            )
            return
        tools = self.cfg.tools
        deadline = time.monotonic() + self.cfg.capture_timeout
        for index, tool_cfg in enumerate(tools):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not still_current():
                return
            capture = create_capture(self.cfg, rank_info.pid, tool_cfg)
            captures.append(capture)
            self.child = capture
            await capture.run(timeout=remaining / (len(tools) - index))
            if capture.is_running():
                self.logger.warning(
                    "Skipping later timeout tools: %s child is still alive", capture.name
                )
                return

    def stage(self, captures, rank_info, cycle, reasons):
        """Write captures for the collector; called after termination was requested."""
        for capture in captures:
            try:
                capture.publish_local(rank_info, cycle, ",".join(sorted(reasons)))
            except OSError as exc:
                self.logger.warning("Could not stage timeout capture: %s", exc)
            finally:
                capture.output.clear()
        if captures:
            self.logger.info(
                "Timeout captures: cycle=%s global_rank=%s local_rank=%s pid=%s %s dir=%s",
                cycle,
                rank_info.global_rank,
                rank_info.local_rank,
                rank_info.pid,
                " ".join(f"{capture.name}={capture.status_text()}" for capture in captures),
                staging_dir(self.cfg) / f"cycle{cycle}",
            )
