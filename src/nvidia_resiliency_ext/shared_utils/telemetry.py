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

import logging
from contextlib import contextmanager
from typing import ClassVar, Final

from torch.distributed.elastic.multiprocessing.errors import SignalException

logger = logging.getLogger(__name__)

try:
    from nemo.lens import NemoLensConfig as _NemoLensConfig
    from nemo.lens import SpanGroup as _SpanGroup
    from nemo.lens import managed_span as _real_managed_span
    from nemo.lens import setup_telemetry as _setup_telemetry

    class _NVRxSpanGroup(_SpanGroup):
        FT = "nvrx.ft"
        CKPT = "nvrx.ckpt"
        ALL_GROUPS: Final[frozenset] = _SpanGroup.ALL_GROUPS | frozenset([FT, CKPT])
        _PRESETS: ClassVar[dict] = {
            **_SpanGroup._PRESETS,
            "default": _SpanGroup._PRESETS["default"] | frozenset([FT, CKPT]),
            "nvrx": frozenset([FT, CKPT]),
            "all": _SpanGroup.ALL_GROUPS | frozenset([FT, CKPT]),
        }

    _NEMO_LENS_AVAILABLE = True

except ModuleNotFoundError:
    _NEMO_LENS_AVAILABLE = False
    _real_managed_span = None

except Exception:
    logger.warning("nemo-lens import failed, continuing without telemetry", exc_info=True)
    _NEMO_LENS_AVAILABLE = False
    _real_managed_span = None


@contextmanager
def managed_span(group, name, tracer=None, **attributes):
    if _real_managed_span is None:
        yield None
        return
    try:
        cm = _real_managed_span(group, name, tracer=tracer, **attributes)
        span = cm.__enter__()
    except Exception:
        logger.debug("managed_span entry suppressed", exc_info=True)
        yield None
        return
    try:
        yield span
    except BaseException as exc:
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.debug("managed_span exit suppressed", exc_info=True)
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.debug("managed_span exit suppressed", exc_info=True)


class _NoOpHandle:
    def shutdown(self, timeout_ms: int = 5000):
        pass


def setup_telemetry(rank: int, world_size: int):
    if not _NEMO_LENS_AVAILABLE:
        return _NoOpHandle()
    try:
        return _setup_telemetry(
            _NemoLensConfig.from_env(span_group_cls=_NVRxSpanGroup),
            rank,
            world_size,
        )
    except Exception:
        logger.warning("nemo-lens init failed, continuing without telemetry", exc_info=True)
        return _NoOpHandle()


def force_flush(timeout_ms: int = 1500) -> None:
    """Flush pending spans without shutting down providers. Safe to call mid-run."""
    try:
        from opentelemetry import trace

        trace.get_tracer_provider().force_flush(timeout_millis=timeout_ms)
    except (SignalException, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        pass


def record_event(name: str, attributes: dict | None = None) -> None:
    """Add a timestamped event to the current active span."""
    try:
        from opentelemetry import trace

        trace.get_current_span().add_event(name, attributes or {})
    except (SignalException, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        pass
