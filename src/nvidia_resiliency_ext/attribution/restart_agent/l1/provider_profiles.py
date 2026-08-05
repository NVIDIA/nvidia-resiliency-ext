# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit deployment profiles for provider-specific CLI defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderProfile:
    base_url: str
    model: str


NVIDIA_INFERENCE_HUB = ProviderProfile(
    base_url="https://inference-api.nvidia.com/v1",
    model="nvidia/qwen/qwen3.5-35b-a3b",
)
