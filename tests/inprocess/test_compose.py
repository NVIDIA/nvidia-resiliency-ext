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

import unittest

import nvidia_resiliency_ext.inprocess as inprocess


class TestCompose(unittest.TestCase):
    def test_empty(self):
        counter = 0

        class Fn:
            def __call__(self):
                nonlocal counter
                counter += 1

        composed = inprocess.Compose(Fn(), Fn(), Fn(), Fn())
        composed()
        self.assertEqual(counter, 4)

    def test_none(self):
        counter = 0

        class Fn:
            def __call__(self, x):
                nonlocal counter
                counter += 1
                return x

        composed = inprocess.Compose(Fn(), Fn(), Fn(), Fn())
        ret = composed(None)
        self.assertIs(ret, None)
        self.assertEqual(counter, 4)

    def test_return(self):
        class Fn:
            def __call__(self):
                return 1

        composed = inprocess.Compose(Fn())
        ret = composed()
        self.assertEqual(ret, 1)

    def test_no_return_warns(self):
        class Fn:
            def __call__(self, x):
                pass

        composed = inprocess.Compose(Fn(), Fn())
        with self.assertWarns(UserWarning):
            ret = composed(1)
        self.assertEqual(ret, None)

    def test_propagate(self):
        class Fn:
            def __call__(self, counter):
                return counter + 1

        composed = inprocess.Compose(Fn(), Fn(), Fn(), Fn())
        ret = composed(0)
        self.assertEqual(ret, 4)

    def test_tuple(self):
        class Fn:
            def __call__(self, a, b, c):
                return a + 1, b + 1, c + 1

        composed = inprocess.Compose(Fn(), Fn(), Fn(), Fn())
        ret = composed(0, 1, 2)
        self.assertEqual(ret, (4, 5, 6))

    def test_basic_subclass(self):
        class Base:
            pass

        class Foo(Base):
            def __call__(self):
                pass

        class Bar(Base):
            def __call__(self):
                pass

        composed = inprocess.Compose(Foo(), Bar())
        self.assertIsInstance(composed, Base)
        self.assertNotIsInstance(composed, Foo)
        self.assertNotIsInstance(composed, Bar)

    def test_nested_subclass(self):
        class Base:
            pass

        class Foo(Base):
            def __call__(self):
                pass

        class Bar(Foo):
            def __call__(self):
                pass

        composed = inprocess.Compose(Foo(), Bar())
        self.assertIsInstance(composed, Base)
        self.assertIsInstance(composed, Foo)
        self.assertNotIsInstance(composed, Bar)
