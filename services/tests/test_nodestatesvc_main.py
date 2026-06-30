# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvidia_resiliency_ext.services.nodestatesvc.__main__ import _default_float, _default_int


def test_default_int_reports_invalid_env_value(monkeypatch):
    monkeypatch.setenv("NVRX_TEST_INT", "not-an-int")

    with pytest.raises(SystemExit, match="expected an integer"):
        _default_int("NVRX_TEST_INT", 1)


def test_default_float_reports_invalid_env_value(monkeypatch):
    monkeypatch.setenv("NVRX_TEST_FLOAT", "not-a-float")

    with pytest.raises(SystemExit, match="expected a float"):
        _default_float("NVRX_TEST_FLOAT", 1.0)
