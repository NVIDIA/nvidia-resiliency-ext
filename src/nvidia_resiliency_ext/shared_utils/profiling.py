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

# This file adds time profiling capabilities using nv one logger

import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from nv_one_logger.api.config import LoggerConfig, OneLoggerConfig
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import Event
from nv_one_logger.core.span import StandardSpanName
from nv_one_logger.exporter.file_exporter import FileExporter
from nv_one_logger.training_telemetry.api.training_telemetry_provider import (
    TrainingTelemetryProvider,
)

# WandB exporter is an optional dependency
try:
    from nv_one_logger.wandb.exporter.wandb_exporter import Config as WandBConfig
    from nv_one_logger.wandb.exporter.wandb_exporter import WandBExporterAsync

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..shared_utils.log_manager import LogConfig


class ProfilingEvent(Enum):
    """Enumeration of profiling events for fault tolerance metrics."""

    FAILURE_DETECTED = "failure_detected"
    WORKER_TERMINATED = "worker_terminated"
    RENDEZVOUS_STARTED = "rendezvous_started"
    RENDEZVOUS_COMPLETED = "rendezvous_completed"
    WORKER_START_STARTED = "worker_start_started"
    WORKER_START_COMPLETED = "worker_start_completed"


class FaultToleranceProfiler:
    """Profiler for measuring fault tolerance timing metrics using nv one logger."""

    def __init__(self):
        self._current_cycle = 0
        # Initialize logger as a member to avoid module-level logger issues
        self._logger = logging.getLogger(LogConfig.name)
        self._app_span = None  # Will hold the application span for attaching events

        # Initialize TrainingTelemetryProvider singleton if not already configured
        if not TrainingTelemetryProvider.instance().one_logger_ready:

            # Configure with minimal required settings for fault tolerance profiling
            # Use Slurm JobID as session tag for grouping related jobs (e.g., across restarts)
            base_config = OneLoggerConfig(
                application_name="nvrx",
                world_size_or_fn=1,
                session_tag_or_fn=lambda: os.environ.get("SLURM_JOB_ID", "local_run"),
                custom_metadata=(
                    {
                        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
                        "slurm_cluster": os.environ.get("SLURM_CLUSTER_NAME"),
                    }
                    if os.environ.get("SLURM_JOB_ID")
                    else None
                ),
                # Disable OneLogger's internal log files by redirecting to /dev/null
                logger_config=LoggerConfig(
                    log_file_path_for_info="/dev/null",
                    log_file_path_for_err="/dev/null",
                ),
            )

            # Configure exporters based on environment variable
            # Options: None (default, no exporter), "file", or "wandb"
            exporter_type = os.environ.get("NVRX_TELEMETRY_EXPORTER", "").lower()

            provider = TrainingTelemetryProvider.instance().with_base_config(base_config)

            # Add FileExporter if requested
            if exporter_type == "file":
                file_path = os.environ.get("NVRX_TELEMETRY_FILE", "nvrx_telemetry.json")
                file_exporter = FileExporter(file_path=Path(file_path))
                provider.with_exporter(file_exporter)

            # Add WandB exporter if requested and available
            elif exporter_type == "wandb":
                if not WANDB_AVAILABLE:
                    self._logger.warning(
                        "WandB exporter requested but nv_one_logger.wandb package not available. "
                        "Install with: pip install nv-one-logger-wandb"
                    )
                else:
                    # WandB entity is required; if not set, skip WandB exporter
                    wandb_entity = os.environ.get("WANDB_ENTITY")
                    if wandb_entity:
                        wandb_config = WandBConfig(
                            entity=wandb_entity,
                            project=os.environ.get("WANDB_PROJECT", "nvrx-telemetry"),
                            run_name=os.environ.get(
                                "WANDB_RUN_NAME", f"nvrx_{os.environ.get('SLURM_JOB_ID', 'local')}"
                            ),
                            api_key=os.environ.get(
                                "WANDB_API_KEY", ""
                            ),  # Falls back to .netrc or WANDB_API_KEY env
                            tags=["nvrx", "fault_tolerance"],
                        )
                        wandb_exporter = WandBExporterAsync(config=wandb_config)
                        provider.with_exporter(wandb_exporter)
                    else:
                        self._logger.warning("WandB exporter requested but WANDB_ENTITY not set")

            elif exporter_type:
                self._logger.warning(
                    f"Unknown exporter type: '{exporter_type}'. Valid options: 'file', 'wandb'"
                )

            provider.configure_provider()

        # Create application span for attaching events (if provider is ready)
        if TrainingTelemetryProvider.instance().one_logger_ready:
            # Add session and metadata attributes to the application span
            span_attributes = Attributes()

            # Add session tag (e.g., Slurm Job ID)
            config = TrainingTelemetryProvider.instance().config
            span_attributes.add("session_tag", config.session_tag)
            span_attributes.add("application_name", config.application_name)

            # Add custom metadata (Slurm job info)
            if config.custom_metadata:
                for key, value in config.custom_metadata.items():
                    span_attributes.add(key, value)

            self._app_span = TrainingTelemetryProvider.instance().recorder.start(
                span_name=StandardSpanName.APPLICATION, span_attributes=span_attributes
            )

    def _timestamp_to_utc_datetime(self, timestamp: float) -> str:
        """Convert timestamp to UTC datetime string."""
        utc_datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Remove last 3 digits for milliseconds

    def set_cycle(self, cycle: int) -> None:
        """Set the current cycle number.

        Called by the rendezvous handler when a newly joining node syncs its cycle number
        from the global_cycle_key in the store. This ensures newly joining nodes (e.g.,
        replacement array tasks) continue with the correct cycle number instead of starting from 0.

        Args:
            cycle: The cycle number to set. Only sets if >= current cycle to prevent backward jumps.
        """
        if cycle >= self._current_cycle:
            self._current_cycle = cycle
        else:
            self._logger.warning(
                f"Attempted to set profiler cycle to {cycle}, which is less than "
                f"current cycle {self._current_cycle}. Ignoring to prevent backward cycle jumps."
            )

    def _publish_metrics(
        self, event: ProfilingEvent, timestamp: float, node_id: Optional[Any], rank: Optional[int]
    ) -> None:
        """Publish metrics using TrainingTelemetryProvider.

        Note: This method is defensive and will not raise exceptions, only log warnings.
        We don't want telemetry failures to disrupt fault tolerance operations.
        """
        # Check if training telemetry is configured and ready
        if not TrainingTelemetryProvider.instance().one_logger_ready:
            return

        try:
            # Create attributes for the event
            attributes = Attributes()
            attributes.add("event_type", event.value)
            attributes.add("timestamp_ms", int(timestamp * 1000))
            attributes.add("cycle", self._current_cycle)
            if node_id:
                # Convert node_id to string to ensure JSON serializability
                attributes.add("node_id", str(node_id))
            if rank is not None:
                attributes.add("rank", rank)

            # Create and record the event (synchronous operation)
            event_obj = Event.create(f"ft.{event.value}", attributes)
            TrainingTelemetryProvider.instance().recorder.event(self._app_span, event_obj)
        except Exception as e:
            # If nv one logger fails, just log a warning and continue
            self._logger.warning(f"Failed to publish metrics to nv one logger: {e}")

    def record_event(
        self,
        event: ProfilingEvent,
        node_id: Optional[Any] = None,
        rank: Optional[int] = None,
    ) -> str:
        """Record a profiling event and return a unique event ID."""
        timestamp = time.time()
        # Convert node_id to string for event ID and logging
        node_id_str = str(node_id) if node_id is not None else 'unknown'
        event_id = f"{event.value}_{timestamp}_{node_id_str}_{rank or 'unknown'}"

        # Increment cycle count for failure detection events
        if event == ProfilingEvent.FAILURE_DETECTED:
            self._current_cycle += 1

        # Publish metrics using nv one logger
        self._publish_metrics(event, timestamp, node_id, rank)

        # Format log message with cycle count and UTC time
        utc_time = self._timestamp_to_utc_datetime(timestamp)
        self._logger.info(
            f"  - Cycle: {self._current_cycle} Event: {event.value} Node: {node_id_str} Rank: {rank} "
            f"Time: {utc_time} UTC"
        )
        return event_id


# Global profiler instance (lazy-initialized to avoid stdout output at import time)
_global_profiler: Optional[FaultToleranceProfiler] = None
_global_profiler_lock = threading.Lock()


def _get_global_profiler() -> FaultToleranceProfiler:
    """Get or create the global profiler instance (thread-safe)."""
    global _global_profiler
    if _global_profiler is None:
        with _global_profiler_lock:
            # Double-check pattern to avoid race conditions
            if _global_profiler is None:
                _global_profiler = FaultToleranceProfiler()
    return _global_profiler


def record_profiling_event(
    event: ProfilingEvent,
    node_id: Optional[Any] = None,
    rank: Optional[int] = None,
) -> str:
    """Convenience function to record a profiling event.

    Args:
        event: The profiling event to record
        node_id: Node identifier (can be any type, will be converted to string)
        rank: Rank identifier

    Returns:
        Event ID string
    """
    return _get_global_profiler().record_event(event, node_id, rank)


def set_profiling_cycle(cycle: int) -> None:
    """Set the current cycle number in the global profiler.

    Called by the rendezvous handler when a newly joining node syncs its cycle number
    from the global_cycle_key in the store. This ensures newly joining nodes (e.g.,
    replacement array tasks) continue with the correct cycle number instead of starting from 0.

    Args:
        cycle: The cycle number to set. Only sets if >= current cycle to prevent backward jumps.
    """
    _get_global_profiler().set_cycle(cycle)
