#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

# SPDX-License-Identifier: BSD-3-Clause
# Modifications made by NVIDIA
# All occurences of 'torch.distributed.elastic' were replaced with 'nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat'

from nvidia_resiliency_ext.fault_tolerance._torch_elastic_compat.multiprocessing.errors.error_handler import ErrorHandler

__all__ = ['get_error_handler']

def get_error_handler():
    return ErrorHandler()
