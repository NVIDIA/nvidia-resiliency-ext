# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest bootstrap; unittest modules invoke the same setup explicitly."""

from _bootstrap import configure_test_imports

configure_test_imports()
