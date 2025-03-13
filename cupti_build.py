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

import glob
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext


def build(setup_kwargs):
    cpp_extension = Pybind11Extension(
        "nvidia_resiliency_ext.straggler.cupti_module",
        # Sort .cpp files for reproducibility
        sorted(glob.glob("src/nvidia_resiliency_ext/straggler/cupti_src/*.cpp")),
        include_dirs=["/usr/local/cuda/include"],
        library_dirs=["/usr/local/cuda/lib64"],
        # prefer static CUPTI if available
        libraries=(
            ["cupti_static"]
            if os.path.exists("/usr/local/cuda/lib64/libcupti_static.a")
            else ["cupti"]
        ),
        extra_compile_args=["-O3"],
        language="c++",
        cxx_std=17,
    )
    ext_modules = [
        cpp_extension,
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": build_ext},
            "zip_safe": False,
        }
    )
