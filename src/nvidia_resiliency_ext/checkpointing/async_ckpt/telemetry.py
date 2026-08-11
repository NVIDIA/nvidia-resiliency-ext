# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Telemetry for the persistent async checkpoint worker.

The worker is spawned, so it inherits nothing from the trainer's memory: no config, no
TracerProvider. Telemetry has to be bootstrapped fresh from data passed explicitly
through the `ctx.Process` args, which is what `bootstrap` is - a plain dict, because it
crosses a spawn boundary by pickling and so must not hold a class instance or anything
else with an import-path dependency.

`shared_utils/telemetry.py` owns the nemo-lens seam; nothing here imports nemo directly.
"""

import logging
from typing import Any, ContextManager, Dict, Optional

from ...shared_utils import telemetry as backend
from ...shared_utils.log_manager import LogConfig

logger = logging.getLogger(LogConfig.name)

#: Span group these spans belong to, for nemo-lens group filtering.
SPAN_GROUP = 'checkpoint'


class CheckpointWorkerTelemetry:
    """Spans emitted by the checkpoint worker, plus the provider it owns.

    Always usable: when telemetry is disabled the spans are no-op context managers and
    there is no provider to shut down, so the worker loop needs no conditionals.
    """

    def __init__(self, handle: Optional[Any] = None):
        self._handle = handle

    @classmethod
    def from_bootstrap(
        cls, bootstrap: Optional[Dict[str, Any]], rank: int
    ) -> "CheckpointWorkerTelemetry":
        """Stand up this process's own telemetry from the trainer-supplied bootstrap.

        `bootstrap` of None means the caller did not configure telemetry at all. Any other
        value also triggers a config read from the environment, since `enabled` in the dict
        is only the CLI override: a run configured purely through the environment has to be
        picked up here too.
        """
        if bootstrap is None or not backend.HAS_NEMO_LENS:
            return cls()
        try:
            config = backend.config_from_env()
            if bootstrap.get('enabled'):
                config.enabled = True
            if bootstrap.get('service_name'):
                config.service_name = bootstrap['service_name']
            if bootstrap.get('span_groups'):
                config.span_groups = bootstrap['span_groups']
            handle = backend.setup_telemetry(
                config,
                rank=bootstrap.get('rank', rank),
                world_size=bootstrap.get('world_size', 1),
                resource_attributes=bootstrap.get('resource_attrs'),
            )
            _apply_resolved_span_groups(bootstrap.get('resolved_span_groups'))
            return cls(handle)
        except Exception as e:  # a broken exporter must not take down checkpointing
            logger.debug("nvrx telemetry: checkpoint worker telemetry disabled: %s", e)
            return cls()

    def request_span(self, call_idx: int) -> ContextManager:
        """Everything the worker does to handle one request, including D2H preload staging
        and queue/GC overhead. Deliberately separate from `write_span`: "time this request
        occupied the worker" and "actual write duration" answer different questions."""
        return backend.span(
            SPAN_GROUP,
            'nvrx.checkpoint.save.request',
            is_goodput_span=True,
            **{'nvrx.call_idx': call_idx},
        )

    def write_span(self) -> ContextManager:
        """Just the write call. The background write is overlapped - training runs straight
        through it - so it costs zero goodput and must stay out of the goodput view, where
        it would double-count against the exposed save.finalize. It stays visible for
        resiliency analysis either way."""
        return backend.span(SPAN_GROUP, 'nvrx.checkpoint.save.write', is_goodput_span=False)

    def shutdown(self) -> None:
        """Flush before the process exits. BatchSpanProcessor only flushes on a timer or on
        shutdown, never automatically at exit, so anything still queued - including the
        spans of the very last request this worker handled - would otherwise be dropped."""
        if self._handle is None:
            return
        try:
            self._handle.shutdown()
        except Exception as e:
            logger.debug("nvrx telemetry: checkpoint worker flush failed: %s", e)


def _apply_resolved_span_groups(groups) -> None:
    """Force the enabled span groups to the caller's fully-resolved set.

    setup_telemetry() resolves this process's span_groups with the base SpanGroup class,
    which cannot resolve consumer-specific group names; the caller resolves them on its
    side and passes the result here, which is what lets consumer-only groups reach the
    worker. Absent (older caller), the groups already set stand.
    """
    if not groups:
        return
    try:
        from nemo.lens.state import set_enabled_span_groups

        set_enabled_span_groups(frozenset(groups))
    except ImportError:
        pass
