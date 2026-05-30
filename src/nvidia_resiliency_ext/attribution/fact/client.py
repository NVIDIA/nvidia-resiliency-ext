#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import socket
import subprocess  # nosec B404
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Iterable, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx


_FACT_API_PREFIX = "/latest"
_DMESG_COMMAND_TIMEOUT_S = 10.0
_NVRX_GRPC_NODE_ID_RE = re.compile(r"^(.+)_\d+$")
_FACT_DMESG_REGEXES = (
    re.compile(r"\bNVRM:\s+Xid\b"),
    re.compile(r"\bXid\s+\(PCI:"),
    re.compile(r"\bSXid\b"),
    re.compile(r"\bSXid\s+\(PCI:"),
    re.compile(r"\bNV_ERR_[A-Z0-9_]*\b"),
    re.compile(r"\bNV_WARN_[A-Z0-9_]*\b"),
)
_FACT_DMESG_SUBSTRINGS = (
    "NVRM: rpcRmApiAlloc_GSP: GspRmAlloc failed",
    "NVRM: nvAssertFailedNoLog: Assertion failed:",
    "CTRL-EVENT-EAP-FAILURE EAP authentication failed",
    "CTRL-EVENT-EAP-SUCCESS EAP authentication completed successfully",
    "System is powering down",
    "Out of memory: Killed process",
    "NMI watchdog: Watchdog detected hard LOCKUP",
    "general protection fault",
    "kernel stack frame pointer at",
    "LustreError",
    "connection2:0: ping timeout",
    "detected conn error",
    "Abrupt nvidia-imex daemon shutdown detected, robust channel recovery invoked!",
    "Failed to collect nvlink status info!",
    "not responding, still trying",
    "Failed to update Rx Detect Link mask!",
    "warthog-fake: INFO APS/WARTHOG Induced fatal error",
    "Stopping nvidia-imex.service",
    "Connection lost to node",
    "Lost connection to GPU",
    "Unable to handle kernel",
)
_LOKI_TIMESTAMP_END_OFFSET_S = 1.0


@dataclass
class FactAttributionResult:
    attributor_id: str
    observation_ids: list[Any]
    faulty_nodes: list[str]
    attribution: dict[str, Any]


def _severity_from_dmesg(message: str) -> str:
    lowered = message.lower()
    if "xid" in lowered or "error" in lowered or "failed" in lowered:
        return "err"
    if "warn" in lowered:
        return "warning"
    return "info"


def _split_prefixed_dmesg_line(line: str, default_hostname: str) -> tuple[str, str]:
    stripped = line.rstrip("\n")
    if ": " not in stripped:
        return default_hostname, stripped

    maybe_host, message = stripped.split(": ", 1)
    if maybe_host and " " not in maybe_host and "/" not in maybe_host:
        return _normalize_node_id(maybe_host), message
    return default_hostname, stripped


def _normalize_node_id(node: str) -> str:
    """Map NVRx gRPC log node IDs like ``host_pid`` back to the host name."""
    match = _NVRX_GRPC_NODE_ID_RE.match(node)
    if match:
        return match.group(1)
    return node


def _fact_attributor_node_list(nodes: Iterable[str]) -> list[str]:
    """Return explicit FACT attributor node names without gRPC log-id normalization."""
    return sorted({node for raw_node in nodes if (node := str(raw_node))})


def normalize_fact_attribution_url(url: str) -> str:
    """Return the FACT API base URL, accepting either service root or API root."""
    normalized = url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("FACT attribution URL must include http(s) scheme and host")
    if parsed.query or parsed.fragment:
        raise ValueError("FACT attribution URL must not include query parameters or fragments")
    if parsed.path in ("", "/"):
        return f"{normalized}{_FACT_API_PREFIX}"
    return normalized


def is_fact_relevant_dmesg_message(message: str) -> bool:
    """Return whether a dmesg message matches FACT's current syslog extractors."""
    if any(pattern.search(message) for pattern in _FACT_DMESG_REGEXES):
        return True
    return any(term in message for term in _FACT_DMESG_SUBSTRINGS)


def _prefix_dmesg_text(text: str, hostname: Optional[str]) -> str:
    if not hostname:
        return text
    prefixed = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\n"):
            prefixed.append(f"{hostname}: {line}")
        else:
            prefixed.append(f"{hostname}: {line}\n")
    return "".join(prefixed)


def collect_recent_dmesg_text(
    *,
    window_s: float,
    hostname: Optional[str] = None,
) -> str:
    """Collect a bounded recent dmesg window and optionally prefix every line."""
    since = datetime.now() - timedelta(seconds=max(0.0, float(window_s)))
    proc = subprocess.run(  # nosec B603
        ["dmesg", "--since", since.strftime("%Y-%m-%d %H:%M:%S.%f")],
        capture_output=True,
        text=True,
        check=False,
        timeout=_DMESG_COMMAND_TIMEOUT_S,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"dmesg --since failed with rc={proc.returncode}: {err}")
    return _prefix_dmesg_text(proc.stdout or "", hostname)


def dmesg_lines_to_raw_loki_streams(
    lines: Iterable[str],
    *,
    default_hostname: Optional[str] = None,
    timestamp_start_ns: Optional[int] = None,
    prefilter: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert dmesg lines into FACT raw_loki_streams JSON."""
    hostname = default_hostname or socket.gethostname()
    base_ns = timestamp_start_ns if timestamp_start_ns is not None else time.time_ns()
    streams: "OrderedDict[str, list[list[str]]]" = OrderedDict()
    accepted_offset = 0

    for line in lines:
        if not line.strip():
            continue
        node, message = _split_prefixed_dmesg_line(line, hostname)
        if prefilter and not is_fact_relevant_dmesg_message(message):
            continue
        payload = {
            "body": message,
            "severity": _severity_from_dmesg(message),
            "attributes": {
                "hostname": node,
                "appname": "kernel",
                "facility": 0,
            },
            "resources": {},
        }
        streams.setdefault(node, []).append(
            [
                str(base_ns + accepted_offset),
                json.dumps(payload, separators=(",", ":")),
            ]
        )
        accepted_offset += 1

    raw_streams = [
        {
            "stream": {
                "job": "nvrx",
                "app": "dmesg",
                "hostname": node,
            },
            "values": values,
        }
        for node, values in streams.items()
    ]
    return raw_streams, list(streams.keys())


def dmesg_text_to_raw_loki_streams(
    text: str,
    *,
    default_hostname: Optional[str] = None,
    timestamp_start_ns: Optional[int] = None,
    prefilter: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert dmesg text into FACT raw_loki_streams JSON."""
    return dmesg_lines_to_raw_loki_streams(
        text.splitlines(),
        default_hostname=default_hostname,
        timestamp_start_ns=timestamp_start_ns,
        prefilter=prefilter,
    )


def _raw_loki_timestamp_anchor_ns(start_time: datetime, end_time: datetime) -> int:
    """Anchor synthetic dmesg Loki timestamps inside the observation interval.

    Plain ``dmesg`` output carries monotonic kernel timestamps, not wall-clock
    timestamps. Put the synthetic Loki times near collection end so short NVRx
    cycles do not place evidence before the FACT workload start time.
    """
    span_s = max(0.0, (end_time - start_time).total_seconds())
    if span_s <= 0.0:
        anchor = end_time
    else:
        anchor = end_time - timedelta(seconds=min(_LOKI_TIMESTAMP_END_OFFSET_S, span_s / 2.0))
    return int(anchor.timestamp() * 1_000_000_000)


class FactAttributionService:
    """Small client for FACT's attributor/observation/attribution API."""

    def __init__(
        self,
        *,
        url: str,
        timeout_s: float = 60.0,
    ) -> None:
        self.timeout_s = timeout_s
        self.base_url = normalize_fact_attribution_url(url)

    def create_failure_attributor(
        self,
        *,
        job_id: str,
        cycle_index: int,
        nodes: Iterable[str],
        ranks_per_node: int,
        nranks: int,
        start_time: datetime,
        end_time: datetime,
        username: Optional[str] = None,
        cluster: Optional[str] = None,
    ) -> str:
        import httpx

        observation_nodes = _fact_attributor_node_list(nodes)
        if not observation_nodes:
            observation_nodes = [socket.gethostname()]
        nranks = max(nranks, ranks_per_node * len(observation_nodes))
        attributor_info = self._build_attributor_info(
            job_id=job_id,
            cycle_index=cycle_index,
            nodes=observation_nodes,
            ranks_per_node=ranks_per_node,
            nranks=nranks,
            start_time=start_time,
            end_time=end_time,
            username=username,
            cluster=cluster,
        )
        with httpx.Client(timeout=self.timeout_s) as client:
            return self._create_attributor(client, attributor_info)

    def submit_dmesg_text_observation(
        self,
        *,
        attributor_id: str,
        dmesg_text: str,
        start_time: datetime,
        end_time: datetime,
        default_hostname: Optional[str] = None,
    ) -> Optional[Any]:
        import httpx

        raw_streams, _ = dmesg_text_to_raw_loki_streams(
            dmesg_text,
            default_hostname=default_hostname,
            timestamp_start_ns=_raw_loki_timestamp_anchor_ns(start_time, end_time),
            prefilter=True,
        )
        if not raw_streams:
            return None

        with httpx.Client(timeout=self.timeout_s) as client:
            return self._post_observation(
                client,
                attributor_id=attributor_id,
                source="syslog",
                format_="raw_loki_streams",
                body=json.dumps(raw_streams, separators=(",", ":")),
                start_time=start_time,
                end_time=end_time,
            )

    def get_attribution_result(
        self,
        *,
        attributor_id: str,
        observation_ids: Optional[list[Any]] = None,
    ) -> FactAttributionResult:
        import httpx

        with httpx.Client(timeout=self.timeout_s) as client:
            attribution = self._get_attribution(client, attributor_id)
        return FactAttributionResult(
            attributor_id=attributor_id,
            observation_ids=observation_ids or [],
            faulty_nodes=self._extract_faulty_nodes(attribution),
            attribution=attribution,
        )

    def _build_attributor_info(
        self,
        *,
        job_id: str,
        cycle_index: int,
        nodes: list[str],
        ranks_per_node: int,
        nranks: int,
        start_time: datetime,
        end_time: datetime,
        username: Optional[str] = None,
        cluster: Optional[str] = None,
    ) -> dict[str, Any]:
        username = (
            username
            or os.environ.get("SLURM_JOB_USER")
            or os.environ.get("USER")
            or os.environ.get("LOGNAME")
            or "unknown"
        )
        cluster = (
            cluster
            or os.environ.get("SLURM_CLUSTER_NAME")
            or os.environ.get("NVRX_CLUSTER_NAME")
            or "unknown"
        )
        tenant = os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("NVRX_TENANT") or "unknown"
        job_name = os.environ.get("SLURM_JOB_NAME") or "unknown"
        return {
            "workload": {
                "id": f"{job_id}:cycle{cycle_index}:{time.time_ns()}",
                "type": "slurm" if os.environ.get("SLURM_JOB_ID") else "",
                "srun_cmd": "unknown",
                "job_start_time": start_time.isoformat(),
                "job_end_time": end_time.isoformat(),
                "status": "FAILED",
                "nranks": nranks,
                "ranks_per_node": ranks_per_node,
                "nodes": nodes,
                "name": job_name,
                "username": username,
                "framework": os.environ.get("NVRX_FRAMEWORK", "unknown"),
                "exit_code_signal": 1,
            },
            "metadata": {
                "cluster": cluster,
                "agent": "nvrx-ft-launcher",
                "tenant": tenant,
                "ruleset": "default",
            },
        }

    def _create_attributor(self, client: httpx.Client, info: dict[str, Any]) -> str:
        response = client.post(
            f"{self.base_url}/attributor",
            json=info,
            headers={"accept": "application/json"},
        )
        response.raise_for_status()
        return str(response.json()["attributor_id"])

    def _post_observation(
        self,
        client: httpx.Client,
        *,
        attributor_id: str,
        source: str,
        format_: str,
        body: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Any:
        payload = {
            "context": {
                "time_interval": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "resources": "AllJobResources",
                "source": source,
                "format": format_,
            },
            "body": body,
        }
        response = client.post(
            f"{self.base_url}/attributor/{attributor_id}/observation",
            json=payload,
            headers={"accept": "application/json"},
        )
        response.raise_for_status()
        return response.json().get("observation_id")

    def _get_attribution(self, client: httpx.Client, attributor_id: str) -> dict[str, Any]:
        response = client.get(
            f"{self.base_url}/attributor/{attributor_id}/attribution",
            headers={"accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_faulty_nodes(attribution: dict[str, Any]) -> list[str]:
        nodes = []
        for item in attribution.get("attributions", []):
            if item.get("type") != "NodeAttribution":
                continue
            if item.get("attributions"):
                node = item.get("node")
                if node:
                    nodes.append(str(node))
        return sorted(set(nodes))
