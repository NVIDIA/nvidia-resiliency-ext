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
import re
import shutil
import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext


def find_file_in_dir(cuda_path, sfile):
    """
    Looks for file under the directory specified by the cuda_path argument.
    If file is not found, returns None

    Args:
        cuda_path (str): Directory where to look for the files
        sfile (str): The file to look for (e.g., 'libcupti.so').

    Returns:
        tuple: (directory_of_file1, directory_of_file2) or (None, None) if either file is not found.
    """

    for root, _, files in os.walk(cuda_path):
        if sfile in files:
            return root
    return None


def _skip_ext_build():
    ans = os.environ.get('STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD', '0')
    return ans.lower() in ['1', 'on', 'yes', 'true']


def get_cuda_path():
    """
    Determines the path to the CUDA installation.

    Find the CUDA root directory under stanadard paths or using nvcc
    as it's typically done in build systems like CMake.

    1. Check if $CUDA_PATH is set or /usr/local/cuda exists; return it if so.
    2. If not, check if nvcc is in PATH. If yes, run "nvcc -v test.cu" and parse output for CUDA root.
    3. If neither method works, raise FileNotFoundError.

    Returns:
        str: The path to the CUDA installation directory.

    Raises:
        FileNotFoundError: If the CUDA installation cannot be found.
    """
    cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    if os.path.isdir(cuda_path):
        return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            # try to extract CUDA root from nvcc output
            result = subprocess.run(
                [nvcc_path, "-v", "test.cu"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                universal_newlines=True,
            )
            # match "#$ TOP=..." in output
            match = re.search(r'#\$ TOP=([^\r\n]*)', result.stdout)
            if match and os.path.isdir(match.group(1)):
                return match.group(1)
            else:
                # fallback: get directory where nvcc is located
                return os.path.dirname(os.path.dirname(nvcc_path))
        except Exception:
            pass

    raise FileNotFoundError(
        "CUDA installation not found in /usr/local/cuda or $CUDA_PATH, "
        "and could not determine CUDA path from nvcc"
    )


def _compile_protos(proto_dir, proto_filenames):
    """
    Compile multiple .proto files in a single protoc invocation.

    Args:
        proto_dir: Directory containing the .proto files and where output will be generated
        proto_filenames: List of .proto file names (e.g., ['nvhcd.proto', 'log_aggregation.proto'])

    Returns:
        list: Names of proto files that were successfully compiled

    Raises:
        subprocess.CalledProcessError: If protoc compilation fails
    """
    # Filter to only existing proto files
    existing_protos = []
    for proto_filename in proto_filenames:
        proto_file = os.path.join(proto_dir, proto_filename)
        if os.path.exists(proto_file):
            existing_protos.append(proto_filename)
        else:
            print(f"WARNING: Proto file not found: {proto_file}")

    if not existing_protos:
        return []

    print(f"Compiling {len(existing_protos)} proto file(s): {', '.join(existing_protos)}")

    # The key is to use -I (proto_path) pointing to src directory
    # and specify proto files as paths relative to src
    # This makes protoc generate imports with the full package path
    src_dir = os.path.join(os.path.dirname(proto_dir), "..", "..")  # Go up to src/
    src_dir = os.path.abspath(src_dir)

    # Proto files relative to src directory
    # e.g., nvidia_resiliency_ext/shared_utils/proto/nvhcd.proto
    proto_paths_rel_to_src = [
        os.path.join("nvidia_resiliency_ext", "shared_utils", "proto", f) for f in existing_protos
    ]

    # Compile all proto files in a single protoc invocation
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{src_dir}",  # Use src as the import path root
        f"--python_out={src_dir}",
        f"--pyi_out={src_dir}",
        f"--grpc_python_out={src_dir}",
        *proto_paths_rel_to_src,  # Pass relative paths
    ]

    subprocess.run(cmd, check=True)

    # Report generated files
    print(f"✓ Successfully compiled {len(existing_protos)} proto file(s)")
    for proto_filename in existing_protos:
        base_name = proto_filename.replace('.proto', '')
        generated_files = [
            f"{base_name}_pb2.py",
            f"{base_name}_pb2.pyi",
            f"{base_name}_pb2_grpc.py",
        ]
        for gen_file in generated_files:
            print(f"  Generated: {os.path.join(proto_dir, gen_file)}")

    return existing_protos


def build(setup_kwargs):
    # Generate gRPC Python files from .proto files
    proto_dir = os.path.join("src", "nvidia_resiliency_ext", "shared_utils", "proto")
    proto_files = [
        "nvhcd.proto",  # Health check service
        "log_aggregation.proto",  # gRPC logging service
        "nvrx_interface.proto",  # NVRx cycle info
    ]

    try:
        # Compile all proto files in one go (more efficient than one-by-one)
        compiled = _compile_protos(proto_dir, proto_files)

        if not compiled:
            print("WARNING: No proto files found to compile")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to compile protobuf files: {e}")
        print("Note: gRPC services require grpcio-tools:")
        print("  pip install grpcio-tools")
        raise
    except Exception as e:
        print(f"\nERROR: Unexpected error during proto compilation: {e}")
        raise

    # Optionally build the CUPTI extension
    if _skip_ext_build():
        print(
            "WARNING! CUPTI extension wont be build due to STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD flag."
        )
        return

    include_dirs = None
    library_dirs = None

    cuda_path = get_cuda_path()

    cupti_h = "cupti.h"
    libcupti_so = "libcupti.so"
    idir = find_file_in_dir(cuda_path, cupti_h)
    ldir = find_file_in_dir(cuda_path, libcupti_so)
    if idir and ldir:
        include_dirs = [idir]
        library_dirs = [ldir]
    else:
        raise FileNotFoundError(f"required files {libcupti_so} and {cupti_h} not found")

    cpp_extension = Pybind11Extension(
        'nvrx_cupti_module',
        # Sort .cpp files for reproducibility
        sorted(glob.glob('src/nvidia_resiliency_ext/attribution/straggler/cupti_src/*.cpp')),
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['cupti'],
        extra_compile_args=['-O3'],
        language='c++',
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
