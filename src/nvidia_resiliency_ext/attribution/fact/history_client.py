# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote

from nvidia_resiliency_ext.attribution.fact.models import FactHistoryRecord


def parse_duration(value: str | float | int | None, *, default: timedelta) -> timedelta:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    text = str(value).strip().lower()
    if not text:
        return default
    unit = text[-1]
    number_text = text[:-1] if unit.isalpha() else text
    number = float(number_text)
    if unit == "d":
        return timedelta(days=number)
    if unit == "h":
        return timedelta(hours=number)
    if unit == "m":
        return timedelta(minutes=number)
    if unit == "s" or not unit.isalpha():
        return timedelta(seconds=number)
    raise ValueError(f"unsupported duration suffix in {value!r}")


class FactHistoryClient:
    """Generic Elasticsearch-style FACT node-history client."""

    def __init__(
        self,
        *,
        es_url: str,
        auth_file: str,
        index: Optional[str] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.es_url = es_url.rstrip("/")
        self.auth_file = auth_file
        self.index = str(index).strip() if index else None
        self.timeout_s = float(timeout_s)

    def query_node_history(
        self,
        *,
        cluster: str,
        nodes: Iterable[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[FactHistoryRecord]:
        import httpx

        node_list = sorted({str(node) for node in nodes if str(node)})
        if not node_list:
            return []
        payload = self._build_query(
            cluster=cluster,
            nodes=node_list,
            start_time=start_time,
            end_time=end_time,
        )
        with httpx.Client(timeout=self.timeout_s, headers=self._auth_headers()) as client:
            response = client.post(self._search_url(), json=payload)
            response.raise_for_status()
            return self._parse_response(response.json())

    def _search_url(self) -> str:
        if self.es_url.endswith("/_search"):
            return self.es_url
        if self.index:
            return f"{self.es_url}/{quote(self.index, safe='*,')}/_search"
        return f"{self.es_url}/_search"

    @staticmethod
    def _build_query(
        *,
        cluster: str,
        nodes: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, Any]:
        return {
            "size": 10_000,
            "_source": ["cluster", "node", "episode_id", "event_time"],
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"cluster": cluster}},
                        {"terms": {"node": nodes}},
                        {
                            "range": {
                                "event_time": {
                                    "gte": _isoformat(start_time),
                                    "lt": _isoformat(end_time),
                                }
                            }
                        },
                    ]
                }
            },
        }

    def _auth_headers(self) -> dict[str, str]:
        text = Path(self.auth_file).read_text(encoding="utf-8").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"Authorization": _authorization_value(text)}
        if not isinstance(parsed, dict):
            return {}
        headers = parsed.get("headers")
        if isinstance(headers, dict):
            return {str(key): str(value) for key, value in headers.items()}
        if parsed.get("authorization"):
            return {"Authorization": str(parsed["authorization"])}
        if parsed.get("bearer_token"):
            return {"Authorization": f"Bearer {parsed['bearer_token']}"}
        if parsed.get("api_key"):
            return {"Authorization": f"ApiKey {parsed['api_key']}"}
        return {}

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> list[FactHistoryRecord]:
        hits = payload.get("hits", {}).get("hits", [])
        if not isinstance(hits, list):
            return []
        records = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            source = hit.get("_source") if isinstance(hit.get("_source"), dict) else {}
            fields = hit.get("fields") if isinstance(hit.get("fields"), dict) else {}
            cluster = _field_value(source, fields, "cluster")
            node = _field_value(source, fields, "node")
            episode_id = _field_value(source, fields, "episode_id")
            event_time = _parse_datetime(_field_value(source, fields, "event_time"))
            if not cluster or not node or not episode_id or event_time is None:
                continue
            records.append(
                FactHistoryRecord(
                    cluster=str(cluster),
                    node=str(node),
                    episode_id=str(episode_id),
                    event_time=event_time,
                )
            )
        return records


def _field_value(source: dict[str, Any], fields: dict[str, Any], name: str) -> Any:
    if name in source:
        value = source[name]
    else:
        value = fields.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _ensure_aware(datetime.fromisoformat(text))
    except ValueError:
        return None


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _isoformat(value: datetime) -> str:
    return _ensure_aware(value).isoformat()


def _authorization_value(text: str) -> str:
    for prefix in ("Bearer ", "ApiKey ", "Basic "):
        if text.startswith(prefix):
            return text
    return f"Bearer {text}"
