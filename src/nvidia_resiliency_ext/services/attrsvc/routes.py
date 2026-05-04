# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP API route and parameter names for nvidia_resiliency_ext.services.attrsvc.

Result/mode constants (RESP_*, STATE_*, mode values) live in the library:
  nvidia_resiliency_ext.attribution (JobMode, RESP_*, STATE_TIMEOUT)
"""

# Logs endpoint path
ROUTE_LOGS = "/logs"

# POST /logs body and GET /logs query parameter names
PARAM_LOG_PATH = "log_path"
PARAM_USER = "user"
PARAM_JOB_ID = "job_id"

# GET /logs optional query parameters (splitlog mode)
PARAM_FILE = "file"
PARAM_WL_RESTART = "wl_restart"
