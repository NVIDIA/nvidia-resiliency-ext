# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvidia_resiliency_ext.attribution.fact import client as fact_client
from nvidia_resiliency_ext.attribution.fact.client import (
    collect_recent_dmesg_text,
    dmesg_text_to_raw_loki_streams,
    is_fact_relevant_dmesg_message,
    normalize_fact_attribution_url,
)


def test_dmesg_text_to_raw_loki_streams_uses_prefixed_hostname():
    streams, nodes = dmesg_text_to_raw_loki_streams(
        "gb-nvl-134-compute01: [1247249.751385] NVRM: Xid (PCI:0000:e4:00): 95\n",
        timestamp_start_ns=1777601387000000000,
    )

    assert nodes == ["gb-nvl-134-compute01"]
    assert streams[0]["stream"]["hostname"] == "gb-nvl-134-compute01"
    timestamp, body = streams[0]["values"][0]
    payload = json.loads(body)
    assert timestamp == "1777601387000000000"
    assert payload["attributes"]["hostname"] == "gb-nvl-134-compute01"
    assert payload["attributes"]["appname"] == "kernel"
    assert payload["severity"] == "err"
    assert "Xid" in payload["body"]


def test_dmesg_text_to_raw_loki_streams_uses_default_hostname_without_prefix():
    streams, nodes = dmesg_text_to_raw_loki_streams(
        "[1247249.751385] plain kernel line\n",
        default_hostname="default-node",
        timestamp_start_ns=1777601387000000000,
    )

    assert nodes == ["default-node"]
    payload = json.loads(streams[0]["values"][0][1])
    assert payload["attributes"]["hostname"] == "default-node"
    assert payload["body"] == "[1247249.751385] plain kernel line"


def test_dmesg_text_to_raw_loki_streams_preserves_zero_timestamp():
    streams, _ = dmesg_text_to_raw_loki_streams(
        "[1247249.751385] plain kernel line\n",
        default_hostname="default-node",
        timestamp_start_ns=0,
    )

    assert streams[0]["values"][0][0] == "0"


def test_dmesg_text_to_raw_loki_streams_normalizes_grpc_node_id():
    streams, nodes = dmesg_text_to_raw_loki_streams(
        "gb-nvl-134-compute03_2549019: [1247249.751385] NVRM: Xid 95\n",
        timestamp_start_ns=1777601387000000000,
    )

    assert nodes == ["gb-nvl-134-compute03"]
    assert streams[0]["stream"]["hostname"] == "gb-nvl-134-compute03"
    payload = json.loads(streams[0]["values"][0][1])
    assert payload["attributes"]["hostname"] == "gb-nvl-134-compute03"


def test_dmesg_text_to_raw_loki_streams_prefilters_fact_patterns():
    text = "\n".join(
        [
            "gb-nvl-134-compute01: [1.0] plain kernel line",
            "gb-nvl-134-compute01: [2.0] NVRM: Xid (PCI:0000:e4:00): 95",
            "gb-nvl-134-compute02: [3.0] SXid (PCI:0000:e5:00): 11012",
            "gb-nvl-134-compute03: [4.0] mlx5_core 0000:01:00.0: port 1 link down",
        ]
    )

    streams, nodes = dmesg_text_to_raw_loki_streams(
        text,
        timestamp_start_ns=1777601387000000000,
        prefilter=True,
    )

    assert nodes == ["gb-nvl-134-compute01", "gb-nvl-134-compute02"]
    bodies = [json.loads(value[1])["body"] for stream in streams for value in stream["values"]]
    assert any("NVRM: Xid" in body for body in bodies)
    assert any("SXid" in body for body in bodies)
    assert all("plain kernel line" not in body for body in bodies)
    assert all("mlx5_core" not in body for body in bodies)


def test_fact_relevant_dmesg_patterns_document_mlx5_gap():
    assert is_fact_relevant_dmesg_message("[1.0] NVRM: Xid (PCI:0000:e4:00): 95")
    assert not is_fact_relevant_dmesg_message("[4.0] mlx5_core 0000:01:00.0: port 1 link down")


def test_collect_recent_dmesg_uses_subprocess_timeout():
    with patch.object(fact_client.subprocess, "run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        collect_recent_dmesg_text(window_s=12.0, hostname="node-a")

    assert run.call_args.kwargs["timeout"] == fact_client._DMESG_COMMAND_TIMEOUT_S


def test_normalize_fact_attribution_url_accepts_service_root_or_api_root():
    assert normalize_fact_attribution_url("http://fact.example:8001") == (
        "http://fact.example:8001/latest"
    )
    assert normalize_fact_attribution_url("https://fact.example:8001/latest/") == (
        "https://fact.example:8001/latest"
    )
    assert normalize_fact_attribution_url("https://proxy.example/fact/latest") == (
        "https://proxy.example/fact/latest"
    )


@pytest.mark.parametrize(
    "url",
    [
        "",
        "fact.example:8001",
        "http://fact.example:8001/latest?debug=true",
        "http://fact.example:8001/latest#fragment",
    ],
)
def test_normalize_fact_attribution_url_rejects_invalid_base_url(url):
    with pytest.raises(ValueError):
        normalize_fact_attribution_url(url)
