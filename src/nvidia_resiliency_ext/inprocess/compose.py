# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar('T')


def find_common_ancestor(*instances):
    common_mro = set(type(instances[0]).mro())

    for instance in instances[1:]:
        common_mro &= set(type(instance).mro())

    if common_mro:
        mro_list = type(instances[0]).mro()
        common_ancestor = [cls for cls in mro_list if cls in common_mro]
        return common_ancestor[0]
    else:
        return None


class Compose:
    r'''
    Performs functional composition (chaining) of multiple callable class
    instances.

    Output of the previous callable is passed as input to the next callable,
    and the output of the last callable is returned as the final output of a
    :py:class:`Compose` instance.

    Constructed :py:class:`Compose` object is an instance of the lowest common
    ancestor in `method resolution order
    <https://docs.python.org/3/glossary.html#term-method-resolution-order>`_ of
    all input callable class instances.

    Example:

    .. code-block:: python

        composed = Compose(a, b, c)
        ret = composed(arg)  # is equivalent to ret = a(b(c(arg)))
    '''

    def __new__(cls, *instances: Callable[[T], T]):

        common_ancestor = find_common_ancestor(*instances)
        DynamicCompose = type(
            'DynamicCompose',
            (Compose, common_ancestor),
            {
                'instances': instances,
                '__new__': object.__new__,
            },
        )
        return DynamicCompose()

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args: Any):
        for instance in reversed(self.instances):
            ret = instance(*args or ())
            if ret is None and args and args != (None,):
                msg = (
                    f'{type(self).__name__} didn\'t chain arguments after '
                    f'calling {instance=} with {args=}'
                )
                warnings.warn(msg)
            if not isinstance(ret, tuple) and len(inspect.signature(instance).parameters) > 0:
                args = (ret,)
            else:
                args = ret
        return ret
