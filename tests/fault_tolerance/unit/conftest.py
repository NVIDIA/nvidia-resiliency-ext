# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for fault_tolerance unit tests (e.g. barrier rendezvous).

Some tests run rendezvous participants in child processes, but the pytest worker
process still creates ``torch.distributed.TCPStore`` objects (master store,
class-scoped fixtures, etc.). Each TCPStore uses c10d non-daemon background
threads inside that process; they often do not exit cleanly on normal interpreter
shutdown, so pytest can hang after the session. On successful sessions we call
``os._exit(0)`` to skip that shutdown (see ``pytest_sessionfinish``).
"""

import logging
import os

import pytest


@pytest.fixture
def capture_nvrx_logs(caplog):
    """Capture 'nvrx' logger records via caplog, independent of log propagation.

    The shared LogConfig handler sets the 'nvrx' logger ``propagate = False``
    (log_manager.py) once NVRx logging is initialized (by any earlier test), so its
    records otherwise never reach caplog's root handler — making log-content
    assertions order-dependent (pass in isolation, fail in the full/sharded suite).

    Attach caplog's handler directly to the 'nvrx' logger so capture works no matter
    what ``propagate`` is, and pin ``propagate = False`` for the duration so the
    record is captured exactly once (never also via the root handler).
    """
    from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig

    nvrx_logger = logging.getLogger(LogConfig.name)
    prev_propagate = nvrx_logger.propagate
    nvrx_logger.addHandler(caplog.handler)
    nvrx_logger.propagate = False
    try:
        yield
    finally:
        nvrx_logger.removeHandler(caplog.handler)
        nvrx_logger.propagate = prev_propagate


def pytest_configure():
    logging.basicConfig(
        level=os.getenv('FT_UNIT_TEST_LOGLEVEL', 'DEBUG'),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def pytest_sessionfinish(session, exitstatus):
    """On success, exit immediately so c10d TCPStore threads in this process cannot block shutdown.

    Those threads live in the pytest worker, independent of subprocess-based test
    participants. We only use ``os._exit`` when ``exitstatus == 0`` so failures
    still get a normal teardown and full tracebacks (``os._exit`` would truncate
    output under ``-x`` / ``-sv``); that path can still hang if TCPStore was used.
    """
    # Success: skip interpreter shutdown (avoids TCPStore thread join hang).
    # Failure: normal exit for complete pytest output; may hang until CI timeout.
    if os.environ.get('PYTEST_FORCE_EXIT', '1') == '1' and exitstatus == 0:
        os._exit(0)
