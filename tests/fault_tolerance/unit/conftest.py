# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for barrier rendezvous tests.

TCPStore creates non-daemon background threads that don't terminate properly,
causing pytest to hang after all tests complete. This hook ensures clean exit.
"""

import os


def pytest_sessionfinish(session, exitstatus):
    """Force exit after test session to avoid TCPStore thread hang."""
    # Force exit with the test result status to bypass TCPStore background threads
    if os.environ.get('PYTEST_FORCE_EXIT', '1') == '1':
        os._exit(exitstatus)
