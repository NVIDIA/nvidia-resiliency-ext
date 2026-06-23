# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia_resiliency_ext.attribution.orchestration.log_path_metadata import extract_job_metadata


def test_extract_job_metadata_extracts_ft_cycle_id_case_insensitively():
    assert (
        extract_job_metadata("/mnt/logs/train_cycle3.log", warn_on_missing_job_id=False).cycle_id
        == 3
    )
    assert (
        extract_job_metadata("/mnt/logs/train_CYCLE4.log", warn_on_missing_job_id=False).cycle_id
        == 4
    )


def test_extract_job_metadata_extracts_job_id_and_cycle_id():
    metadata = extract_job_metadata(
        "/mnt/logs/prefix_12345678_date_26-06-23_time_10-11-12_cycle7.log"
    )

    assert metadata.job_id == "12345678"
    assert metadata.cycle_id == 7
