# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""nvrx-watch: out-of-job watcher for NVRx deployments.

Stdlib only, by design: it runs on a login node outside the training container, where
nvidia_resiliency_ext need not be installed. Copy the directory and run it.

See DESIGN.md for the architecture and the detector catalog.
"""

from .types import CRITICAL, INFO, WARNING, Finding, Snapshot  # noqa: F401

__all__ = ["Finding", "Snapshot", "INFO", "WARNING", "CRITICAL"]
__version__ = "0.1.0"
