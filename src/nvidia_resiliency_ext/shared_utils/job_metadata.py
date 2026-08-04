# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers for job metadata derived from launcher environment variables."""

import os
from typing import Mapping


def _nonempty_env_value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    value = values.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def job_user_from_env() -> str | None:
    """Read job user from SLURM_JOB_USER or USER env."""
    return os.environ.get("SLURM_JOB_USER") or os.environ.get("USER") or None


def job_id_from_env() -> str | None:
    """Read job id from SLURM_ARRAY_JOB_ID or SLURM_JOB_ID env."""
    return os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or None


def slurm_array_job_id_from_env(env: Mapping[str, str] | None = None) -> str | None:
    """Read the Slurm array parent job ID, if present."""
    return _nonempty_env_value("SLURM_ARRAY_JOB_ID", env)


def slurm_array_task_id_from_env(env: Mapping[str, str] | None = None) -> str | None:
    """Read the current Slurm array task ID, if present."""
    return _nonempty_env_value("SLURM_ARRAY_TASK_ID", env)


def slurm_restart_count_from_env(env: Mapping[str, str] | None = None) -> int:
    """Read and strictly validate the current Slurm allocation generation."""
    values = os.environ if env is None else env
    value = values.get("SLURM_RESTART_COUNT", "0").strip()
    try:
        restart_count = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid SLURM_RESTART_COUNT value: {value!r}") from exc
    if restart_count < 0:
        raise ValueError(f"SLURM_RESTART_COUNT must be non-negative: {restart_count}")
    return restart_count
