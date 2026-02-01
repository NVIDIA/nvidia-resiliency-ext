#  Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

"""HTTP API route and parameter names for nvrx_attrsvc.

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
