# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from argparse import Namespace

import torch

from nvidia_resiliency_ext.fault_tolerance import dict_utils as ft_utils

torch.set_default_device("cuda")


def test_merge_state_dicts():
    d1 = {}
    d2 = {"a": 123}
    ft_utils.merge_state_dicts_(d1, d2)
    assert set(d1.keys()) == set(["a"])
    assert d1["a"] == 123

    d1 = {"a": 0}
    d2 = {"a": 123}
    ft_utils.merge_state_dicts_(d1, d2)
    assert set(d1.keys()) == set(["a"])
    assert d1["a"] == 123

    d1 = {"a": 0}
    d2 = {"b": 123}
    ft_utils.merge_state_dicts_(d1, d2)
    assert set(d1.keys()) == set(["a", "b"])
    assert d1["a"] == 0
    assert d1["b"] == 123

    d1 = {"a": dict()}
    d2 = {"a": 123}
    ft_utils.merge_state_dicts_(d1, d2)
    assert set(d1.keys()) == set(["a"])
    assert d1["a"] == 123

    d1 = {"a": {"aa": 1}}
    d2 = {"a": 123}
    ft_utils.merge_state_dicts_(d1, d2)
    assert set(d1.keys()) == set(["a"])
    assert d1["a"] == 123

    d1 = {"a": {"aa": 1}}
    d2 = {"a": {"aa": 123, "bb": 321, "cc": {"aaa": 111}}}
    ft_utils.merge_state_dicts_(d1, d2)
    assert set(d1.keys()) == set(["a"])
    assert set(d1["a"].keys()) == set(["aa", "bb", "cc"])
    assert d1["a"]["aa"] == 123
    assert d1["a"]["bb"] == 321
    assert d1["a"]["cc"]["aaa"] == 111


def test_merge_ns():
    d1 = {"k0": Namespace(a=1, b=-1)}
    d2 = {"k0": Namespace(b=2)}
    ft_utils.merge_state_dicts_(d1, d2)
    assert vars(d1["k0"]).keys() == set(["a", "b"])
    assert d1["k0"].a == 1
    assert d1["k0"].b == 2

    d1 = {"k0": Namespace(a=1)}
    d2 = {"k0": Namespace(b=2)}
    ft_utils.merge_state_dicts_(d1, d2)
    assert vars(d1["k0"]).keys() == set(["a", "b"])
    assert d1["k0"].a == 1
    assert d1["k0"].b == 2


def test_compare_state_dicts_and_get_new_values_basic():
    d1 = {}
    d2 = {"a": 123}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["a"])
    assert d3["a"] == 123

    d1 = {"a": 0}
    d2 = {"a": 123}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["a"])
    assert d3["a"] == 123

    d1 = {"a": 0, "b": 1}
    d2 = {"a": {"aa": 123}, "c": 3}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["a", "c"])
    assert set(d3["a"].keys()) == set(["aa"])
    assert d3["a"]["aa"] == 123
    assert d3["c"] == 3

    d1 = {"a": {"aa": 123}, "c": 0}
    d2 = {"a": {"aa": 456, "bb": 999}, "b": 0}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["a", "b"])
    assert set(d3["a"].keys()) == set(["aa", "bb"])
    assert d3["a"]["aa"] == 456
    assert d3["a"]["bb"] == 999
    assert d3["b"] == 0


def test_compare_state_dicts_and_get_new_values_ns():
    d1 = {"k0": Namespace(a=1, b=2)}
    d2 = {"k0": Namespace(a=1, b=2)}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert len(d3) == 0

    d1 = {"k0": Namespace(a=1, b=2)}
    d2 = {"k0": Namespace(a=1, b=3)}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["k0"])
    assert vars(d3["k0"]).keys() == set(["b"])
    assert d3["k0"].b == 3

    d1 = {"k0": Namespace(a=1)}
    d2 = {"k0": Namespace(b=2)}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["k0"])
    assert vars(d3["k0"]).keys() == set(["b"])
    assert d3["k0"].b == 2


def test_compare_state_dicts_and_get_new_values_with_tensors():
    TEST_TENSOR0 = torch.ones((4,))
    TEST_TENSOR1 = torch.ones((4,))

    d1 = {"a": TEST_TENSOR0}
    d2 = {"a": TEST_TENSOR0}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert len(d3.keys()) == 0

    d1 = {"a": TEST_TENSOR0}
    d2 = {"a": TEST_TENSOR1}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["a"])
    assert d3["a"] is TEST_TENSOR1

    d1 = {"a": {"aa": TEST_TENSOR0}, "b": None}
    d2 = {"a": {"aa": TEST_TENSOR0}, "c": TEST_TENSOR1}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["c"])

    d1 = {"a": {"aa": TEST_TENSOR1}}
    d2 = {"a": {"aa": TEST_TENSOR1}}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert len(d3.keys()) == 0

    d1 = {"a": [[TEST_TENSOR1]]}
    d2 = {"a": [[TEST_TENSOR1]]}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert len(d3.keys()) == 0

    d1 = {"a": [[TEST_TENSOR1]]}
    d2 = {"a": [[TEST_TENSOR0, TEST_TENSOR1]]}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["a"])
    assert len(d3["a"][0]) == 2

    d1 = {"a": [0, 1, TEST_TENSOR0]}
    d2 = {"a": [0, 1, TEST_TENSOR1]}
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["a"])
    assert d3["a"][0] == 0
    assert d3["a"][1] == 1
    assert d3["a"][2] is TEST_TENSOR1


def test_compare_state_dicts_and_get_new_values_nested():
    TEST_TENSOR0 = torch.ones((4,))
    TEST_TENSOR1 = torch.ones((4,))
    TEST_TENSOR2 = torch.ones((4,))

    d1 = {
        'l0k0': 'test_value',
        'l0k1': TEST_TENSOR0,
        'l0k2': [1, 2, TEST_TENSOR1],
        'l0k3': {'l1k0': 0, 'l1k1': TEST_TENSOR2},
    }
    d2 = d1.copy()
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert len(d3) == 0

    d1 = {
        'l0k0': 'test_value',
        'l0k1': TEST_TENSOR0,
        'l0k2': [1, 2, TEST_TENSOR1],
        'l0k3': {'l1k0': 0, 'l1k1': TEST_TENSOR2},
    }
    d1 = {
        'l0k0': 'test_value',
        'l0k1': TEST_TENSOR0,
        'l0k2': [1, 2, TEST_TENSOR1],
        'l0k3': {'l1k0': 0, 'l1k1': TEST_TENSOR0},  #!updated tensor
    }
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["l0k3"])
    assert set(d3["l0k3"].keys()) == set(["l1k1"])

    d1 = {
        'l0k0': 'test_value',
        'l0k1': TEST_TENSOR0,
        'l0k2': [1, 2, TEST_TENSOR1],
        'l0k3': {'l1k0': 0, 'l1k1': TEST_TENSOR2},
    }
    d2 = {
        'l0k0': 'test_value',
        'l0k1': TEST_TENSOR0,
        'l0k2': [1, 2, TEST_TENSOR0],  #!updated tensor
        'l0k3': {'l1k0': 0, 'l1k1': TEST_TENSOR2},
    }
    d3 = ft_utils.compare_state_dicts_and_get_new_values(d1, d2)
    assert set(d3.keys()) == set(["l0k2"])
    assert d3["l0k2"][0:2] == [1, 2]
    assert d3["l0k2"][2] is TEST_TENSOR0
