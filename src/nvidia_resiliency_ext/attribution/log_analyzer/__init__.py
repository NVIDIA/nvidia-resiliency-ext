# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core log analysis engine.

The service-layer wiring (analysis pipeline, job tracking, splitlog, parsers, config, etc.)
lives in :mod:`nvidia_resiliency_ext.attribution.svc`.

Example:
    from nvidia_resiliency_ext.attribution.analyzer import Analyzer

    analyzer = Analyzer(allowed_root="/logs", use_lib_log_analysis=False)
    result = await analyzer.analyze("/logs/slurm-12345.out")
"""

from .nvrx_logsage import NVRxLogAnalyzer

__all__ = [
    "NVRxLogAnalyzer",
]
