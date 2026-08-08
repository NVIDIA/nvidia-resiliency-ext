#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the standalone NVRx Scheduler Exclusion Service in the foreground. The batch
# script owns backgrounding, restart policy, log redirection, and shutdown.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARTIFACT="${NVRX_SCHEDULER_EXCLUSION_ARTIFACT:-${REPO_ROOT}/dist/nvrx-scheduler-exclusion-service.pyz}"
PYTHON="${NVRX_SCHEDULER_EXCLUSION_PYTHON:-python3}"

if [[ ! -r "${ARTIFACT}" ]]; then
    echo "Scheduler Exclusion artifact is not readable: ${ARTIFACT}" >&2
    exit 1
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Scheduler Exclusion Python interpreter not found: ${PYTHON}" >&2
    exit 1
fi

exec "${PYTHON}" "${ARTIFACT}" "$@"
