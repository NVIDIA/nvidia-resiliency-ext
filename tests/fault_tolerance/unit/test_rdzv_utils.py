# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nvidia_resiliency_ext.fault_tolerance.rdzv_utils import (
    rdzv_config_get_as_bool,
    rdzv_config_get_as_float,
    rdzv_config_get_as_int,
)


@pytest.mark.parametrize("value", [True, 1, "1", "true", "t", "yes", "y"])
def test_rdzv_config_get_as_bool_accepts_true_values(value):
    assert rdzv_config_get_as_bool({"is_host": value}, "is_host") is True


@pytest.mark.parametrize("value", [False, 0, "0", "false", "f", "no", "n"])
def test_rdzv_config_get_as_bool_accepts_false_values(value):
    assert rdzv_config_get_as_bool({"is_host": value}, "is_host") is False


def test_rdzv_config_get_as_bool_uses_default_for_absent_key():
    assert rdzv_config_get_as_bool({}, "is_host") is None
    assert rdzv_config_get_as_bool({}, "is_host", default=True) is True


def test_rdzv_config_get_as_bool_rejects_invalid_values():
    with pytest.raises(ValueError, match="valid boolean"):
        rdzv_config_get_as_bool({"is_host": "maybe"}, "is_host")


def test_rdzv_config_get_as_int_casts_values_and_uses_default():
    assert rdzv_config_get_as_int({"read_timeout": "60"}, "read_timeout") == 60
    assert rdzv_config_get_as_int({}, "read_timeout", default=30) == 30
    assert rdzv_config_get_as_int({}, "read_timeout") is None


def test_rdzv_config_get_as_int_rejects_invalid_values():
    with pytest.raises(ValueError, match="valid integer"):
        rdzv_config_get_as_int({"read_timeout": "slow"}, "read_timeout")


def test_rdzv_config_get_as_float_casts_values_and_uses_default():
    assert rdzv_config_get_as_float({"join_timeout": "12.5"}, "join_timeout") == 12.5
    assert rdzv_config_get_as_float({}, "join_timeout", default=5.0) == 5.0
    assert rdzv_config_get_as_float({}, "join_timeout") is None


def test_rdzv_config_get_as_float_rejects_invalid_values():
    with pytest.raises(ValueError, match="valid float"):
        rdzv_config_get_as_float({"join_timeout": "slow"}, "join_timeout")
