# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared attrsvc HTTP route names and client-side request helpers."""

from __future__ import annotations

from typing import Any

# Logs endpoint path
ROUTE_LOGS = "/logs"
ROUTE_HEALTHZ = "/healthz"

# POST /logs body and GET /logs query parameter names
PARAM_LOG_PATH = "log_path"
PARAM_USER = "user"
PARAM_JOB_ID = "job_id"

# GET /logs optional query parameters (splitlog mode)
PARAM_FILE = "file"
PARAM_WL_RESTART = "wl_restart"

ACCEPT_JSON_HEADERS = {"accept": "application/json"}


def log_submit_payload(
    log_path: str,
    *,
    user: str | None = None,
    job_id: str | None = None,
) -> dict[str, str]:
    """Build the JSON body for ``POST /logs``."""
    payload = {PARAM_LOG_PATH: log_path}
    if user is not None:
        payload[PARAM_USER] = user
    if job_id is not None:
        payload[PARAM_JOB_ID] = job_id
    return payload


def log_result_params(
    log_path: str,
    *,
    file: str | None = None,
    wl_restart: int | None = None,
) -> dict[str, Any]:
    """Build query params for ``GET /logs``."""
    params: dict[str, Any] = {PARAM_LOG_PATH: log_path}
    if file is not None:
        params[PARAM_FILE] = file
    if wl_restart is not None:
        params[PARAM_WL_RESTART] = wl_restart
    return params


def post_log(
    client: Any,
    log_path: str,
    *,
    user: str | None = None,
    job_id: str | None = None,
) -> Any:
    """Submit a log to attrsvc with an existing HTTP client."""
    return client.post(
        ROUTE_LOGS,
        json=log_submit_payload(log_path, user=user, job_id=job_id),
        headers=ACCEPT_JSON_HEADERS,
    )


def get_log_response(
    client: Any,
    log_path: str,
    *,
    file: str | None = None,
    wl_restart: int | None = None,
) -> Any:
    """Fetch a log-analysis response from attrsvc with an existing HTTP client."""
    return client.get(
        ROUTE_LOGS,
        params=log_result_params(log_path, file=file, wl_restart=wl_restart),
        headers=ACCEPT_JSON_HEADERS,
    )
