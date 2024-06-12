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


def zip_strict(*args):
    """
    Alternative to Python's builtin zip(..., strict=True) (available in 3.10+).
    Apart from providing functionality in earlier versions of Python is also more verbose.
    (Python's zip does not print lengths, only which iterable has finished earlier)
    """
    args = [list(a) for a in args]
    lens = [len(a) for a in args]
    assert len(set(lens)) <= 1, f"Tried to zip iterables of unequal lengths: {lens}!"
    return zip(*args)
