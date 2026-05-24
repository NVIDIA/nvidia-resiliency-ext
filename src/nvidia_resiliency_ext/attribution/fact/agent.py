# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import random
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from torch.distributed import TCPStore
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint

from nvidia_resiliency_ext.attribution.fact.client import (
    FactAttributionResult,
    FactAttributionService,
    collect_recent_dmesg_text,
)
from nvidia_resiliency_ext.attribution.fact.history_client import (
    FactHistoryClient,
    parse_duration,
)
from nvidia_resiliency_ext.attribution.fact.hot_cache import FactHotCache
from nvidia_resiliency_ext.attribution.fact.models import AvoidDecision
from nvidia_resiliency_ext.attribution.fact.repeat_offender_policy import (
    compute_repeat_offender_decision,
)
from nvidia_resiliency_ext.attribution.fact.rpc import (
    DEFAULT_MAX_RPC_BYTES,
    default_socket_path,
    recv_frame,
    send_frame,
)
from nvidia_resiliency_ext.shared_utils.log_manager import LogConfig, setup_logger
from nvidia_resiliency_ext.shared_utils.log_paths import get_source_cycle_log_file

logger = logging.getLogger(LogConfig.name)

DEFAULT_DMESG_WINDOW_S = 12.0 * 60.0
DEFAULT_OBSERVATION_DEADLINE_S = 30.0
DEFAULT_STORE_TIMEOUT_S = 60.0
DEFAULT_FACT_HISTORY_LOOKBACK = "14d"
DEFAULT_FACT_HISTORY_MAX_CANDIDATE_NODES = 16
DEFAULT_FACT_HISTORY_QUERY_TIMEOUT_S = 30.0
DEFAULT_FACT_MIN_REPEAT_COUNT_FOR_AVOID = 2
DEFAULT_FACT_MAX_ATTRIBUTION_AVOIDS_PER_CYCLE = 1
_ATTRIBUTOR_FAILURE_PREFIX = "__nvrx_fact_attributor_failed__:"
_POST_RETRY_INITIAL_DELAY_S = 0.25
_POST_RETRY_MAX_DELAY_S = 2.0
_POST_RETRY_MIN_REMAINING_S = 0.5
_GRPC_RESULT_DRAIN_TIMEOUT_S = 4.0


@dataclass(frozen=True)
class FactAgentRequest:
    run_id: str
    cycle: int
    rdzv_endpoint: str
    local_node: str
    is_store_host: bool = False
    store_timeout_s: float = DEFAULT_STORE_TIMEOUT_S
    job_id: Optional[str] = None
    expected_nodes: tuple[str, ...] = ()
    ranks_per_node: int = 1
    cycle_start_time: Optional[datetime] = None
    cycle_end_time: Optional[datetime] = None
    dmesg_path: Optional[str] = None
    result_path: Optional[str] = None
    grpc_server_address: Optional[str] = None
    grpc_node_id: Optional[str] = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        run_id: Optional[str] = None,
        rdzv_endpoint: Optional[str] = None,
        local_node: Optional[str] = None,
        is_store_host: bool = False,
        store_timeout_s: float = DEFAULT_STORE_TIMEOUT_S,
        job_id: Optional[str] = None,
        ranks_per_node: int = 1,
        cycle_start_time: Optional[datetime] = None,
        cycle_end_time: Optional[datetime] = None,
        dmesg_path: Optional[str] = None,
        result_path: Optional[str] = None,
        grpc_server_address: Optional[str] = None,
        grpc_node_id: Optional[str] = None,
    ) -> "FactAgentRequest":
        if payload.get("event") != "cycle_failed":
            raise ValueError("unsupported FACT agent event")
        resolved_run_id = str(run_id or "").strip()
        resolved_rdzv_endpoint = str(rdzv_endpoint or "").strip()
        resolved_local_node = str(local_node or socket.getfqdn(socket.gethostname()))
        if not resolved_run_id:
            raise ValueError("cycle_failed requires run_id")
        if not resolved_rdzv_endpoint:
            raise ValueError("cycle_failed requires rdzv_endpoint")
        raw_cycle = payload.get("cycle", payload.get("cycle_id"))
        if raw_cycle is None:
            raise ValueError("cycle_failed requires cycle")
        cycle = int(raw_cycle)
        expected_nodes_raw = payload.get("expected_nodes") or []
        if not isinstance(expected_nodes_raw, list):
            raise ValueError("expected_nodes must be a list when provided")
        expected_nodes = tuple(str(node) for node in expected_nodes_raw if str(node))
        return cls(
            run_id=resolved_run_id,
            cycle=cycle,
            rdzv_endpoint=resolved_rdzv_endpoint,
            local_node=resolved_local_node,
            is_store_host=bool(is_store_host),
            store_timeout_s=float(store_timeout_s),
            job_id=str(job_id or resolved_run_id),
            expected_nodes=expected_nodes,
            ranks_per_node=max(1, int(ranks_per_node)),
            cycle_start_time=cls._parse_datetime(
                payload.get("cycle_start_time"),
                fallback=cycle_start_time,
            ),
            cycle_end_time=cls._parse_datetime(
                payload.get("cycle_end_time"),
                fallback=cycle_end_time,
            ),
            dmesg_path=dmesg_path,
            result_path=result_path,
            grpc_server_address=grpc_server_address,
            grpc_node_id=grpc_node_id,
        )

    @staticmethod
    def _parse_datetime(value: Any, *, fallback: Optional[datetime] = None) -> Optional[datetime]:
        if value is None or value == "":
            return fallback
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        raise ValueError("cycle_start_time must be an ISO-8601 datetime when provided")


class FactAgentKeys:
    def __init__(self, run_id: str, cycle: int) -> None:
        self.prefix = f"fact_agent:{run_id}:cycle{cycle}"

    @property
    def attributor_id(self) -> str:
        return f"{self.prefix}:attributor_id"

    @property
    def done_count(self) -> str:
        return f"{self.prefix}:done_count"


StoreFactory = Callable[[FactAgentRequest], Any]
FactClientFactory = Callable[[], FactAttributionService]
FactHistoryClientFactory = Callable[[], FactHistoryClient]
DmesgCollector = Callable[[float, str], str]
GrpcWriterFactory = Callable[[queue.Queue, str, str, logging.Logger], threading.Thread]


class FactAgent:
    def __init__(
        self,
        *,
        fact_url: str,
        socket_path: Optional[str] = None,
        dmesg_window_s: float = DEFAULT_DMESG_WINDOW_S,
        observation_deadline_s: float = DEFAULT_OBSERVATION_DEADLINE_S,
        fact_timeout_s: float = 60.0,
        run_id: Optional[str] = None,
        rdzv_endpoint: Optional[str] = None,
        store_timeout_s: float = DEFAULT_STORE_TIMEOUT_S,
        local_node: Optional[str] = None,
        is_store_host: bool = False,
        job_id: Optional[str] = None,
        ranks_per_node: int = 1,
        username: Optional[str] = None,
        cluster: Optional[str] = None,
        health_log_prefix: Optional[str] = None,
        dmesg_artifact_enabled: bool = False,
        result_artifact_enabled: bool = False,
        grpc_server_address: Optional[str] = None,
        grpc_node_id: Optional[str] = None,
        fact_history_es_url: Optional[str] = None,
        fact_history_es_auth_file: Optional[str] = None,
        fact_history_lookback: str = DEFAULT_FACT_HISTORY_LOOKBACK,
        fact_history_index: Optional[str] = None,
        fact_history_max_candidate_nodes: int = DEFAULT_FACT_HISTORY_MAX_CANDIDATE_NODES,
        fact_history_query_timeout_s: float = DEFAULT_FACT_HISTORY_QUERY_TIMEOUT_S,
        fact_min_repeat_count_for_avoid: int = DEFAULT_FACT_MIN_REPEAT_COUNT_FOR_AVOID,
        fact_max_attribution_avoids_per_cycle: int = (
            DEFAULT_FACT_MAX_ATTRIBUTION_AVOIDS_PER_CYCLE
        ),
        store_factory: Optional[StoreFactory] = None,
        fact_client_factory: Optional[FactClientFactory] = None,
        fact_history_client_factory: Optional[FactHistoryClientFactory] = None,
        dmesg_collector: Optional[DmesgCollector] = None,
        grpc_writer_factory: Optional[GrpcWriterFactory] = None,
    ) -> None:
        self.fact_url = fact_url
        self.socket_path = socket_path or default_socket_path()
        self.dmesg_window_s = dmesg_window_s
        self.observation_deadline_s = observation_deadline_s
        self.fact_timeout_s = fact_timeout_s
        self.run_id = str(run_id).strip() if run_id else None
        self.rdzv_endpoint = str(rdzv_endpoint).strip() if rdzv_endpoint else None
        self.store_timeout_s = float(store_timeout_s)
        self.local_node = local_node or socket.getfqdn(socket.gethostname())
        self.is_store_host = bool(is_store_host)
        self.job_id = str(job_id).strip() if job_id else None
        self.ranks_per_node = max(1, int(ranks_per_node))
        self.username = str(username).strip() if username else None
        self.cluster = str(cluster).strip() if cluster else None
        self.health_log_prefix = str(health_log_prefix).strip() if health_log_prefix else None
        self.dmesg_artifact_enabled = bool(dmesg_artifact_enabled)
        self.result_artifact_enabled = bool(result_artifact_enabled)
        self.grpc_server_address = str(grpc_server_address).strip() if grpc_server_address else None
        self.grpc_node_id = str(grpc_node_id).strip() if grpc_node_id else None
        self.fact_history_es_url = str(fact_history_es_url).strip() if fact_history_es_url else None
        self.fact_history_es_auth_file = (
            str(fact_history_es_auth_file).strip() if fact_history_es_auth_file else None
        )
        self.fact_history_lookback = fact_history_lookback
        self.fact_history_index = str(fact_history_index).strip() if fact_history_index else None
        self.fact_history_max_candidate_nodes = max(1, int(fact_history_max_candidate_nodes))
        self.fact_history_query_timeout_s = max(0.1, float(fact_history_query_timeout_s))
        self.fact_min_repeat_count_for_avoid = max(1, int(fact_min_repeat_count_for_avoid))
        self.fact_max_attribution_avoids_per_cycle = max(
            0,
            int(fact_max_attribution_avoids_per_cycle),
        )
        self._store_factory = store_factory or self._connect_tcp_store
        self._fact_client_factory = fact_client_factory or self._new_fact_client
        self._fact_history_client_factory = (
            fact_history_client_factory or self._new_fact_history_client
        )
        self._dmesg_collector = dmesg_collector or self._collect_dmesg
        self._grpc_writer_factory = grpc_writer_factory or self._new_grpc_writer
        self._grpc_writers: dict[tuple[str, str], tuple[queue.Queue, threading.Thread]] = {}
        self._grpc_writers_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="nvrx-fact-agent")
        self._stop_event = threading.Event()
        self._hot_cache = FactHotCache()
        self._avoid_decisions: dict[int, AvoidDecision] = {}
        self._avoid_decisions_lock = threading.Lock()

    def _new_fact_client(self) -> FactAttributionService:
        return FactAttributionService(
            url=self.fact_url,
            timeout_s=self.fact_timeout_s,
        )

    def _new_fact_history_client(self) -> FactHistoryClient:
        if not self.fact_history_es_url or not self.fact_history_es_auth_file:
            raise RuntimeError("FACT history is not configured")
        return FactHistoryClient(
            es_url=self.fact_history_es_url,
            auth_file=self.fact_history_es_auth_file,
            index=self.fact_history_index,
            timeout_s=self.fact_history_query_timeout_s,
        )

    def _connect_tcp_store(self, request: FactAgentRequest) -> TCPStore:
        host, port = parse_rendezvous_endpoint(request.rdzv_endpoint, default_port=-1)
        if not host or port == -1:
            raise ValueError(f"invalid rendezvous endpoint: {request.rdzv_endpoint!r}")
        return TCPStore(
            host,
            port,
            is_master=False,
            timeout=timedelta(seconds=max(1.0, request.store_timeout_s)),
            multi_tenant=True,
        )

    @staticmethod
    def _collect_dmesg(window_s: float, local_node: str) -> str:
        return collect_recent_dmesg_text(window_s=window_s, hostname=local_node)

    @staticmethod
    def _new_grpc_writer(
        write_queue: queue.Queue,
        grpc_server_address: str,
        node_id: str,
        logger: logging.Logger,
    ) -> threading.Thread:
        from nvidia_resiliency_ext.fault_tolerance.per_cycle_logs import GrpcWriterThread

        return GrpcWriterThread(
            write_queue=write_queue,
            grpc_server_address=grpc_server_address,
            node_id=node_id,
            logger=logger,
        )

    def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("event") == "ping":
            return {"accepted": True}
        if payload.get("event") == "shutdown":
            self.request_stop()
            return {"accepted": True}
        if payload.get("event") == "get_avoid_nodes":
            return self._handle_get_avoid_nodes(payload)
        request = self._request_from_payload(payload)
        self._executor.submit(self.process_cycle_failed, request)
        return {"accepted": True}

    def _handle_get_avoid_nodes(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_store_host:
            return {"status": "skipped", "avoid_nodes": []}
        raw_cycle = payload.get("cycle", payload.get("cycle_id"))
        if raw_cycle is None:
            return {"status": "skipped", "avoid_nodes": []}
        cycle = int(raw_cycle)
        with self._avoid_decisions_lock:
            decision = self._avoid_decisions.get(cycle)
        if decision is None:
            return {"cycle_id": str(cycle), "status": "pending", "avoid_nodes": []}
        return {
            "cycle_id": str(cycle),
            "status": decision.status,
            "avoid_nodes": list(decision.avoid_nodes),
        }

    def _request_from_payload(self, payload: dict[str, Any]) -> FactAgentRequest:
        request = FactAgentRequest.from_payload(
            payload,
            run_id=self.run_id,
            rdzv_endpoint=self.rdzv_endpoint,
            local_node=self.local_node,
            is_store_host=self.is_store_host,
            store_timeout_s=self.store_timeout_s,
            job_id=self.job_id,
            ranks_per_node=self.ranks_per_node,
            grpc_server_address=self.grpc_server_address,
            grpc_node_id=self.grpc_node_id,
        )
        dmesg_path = request.dmesg_path
        result_path = request.result_path
        if self.health_log_prefix:
            if self.dmesg_artifact_enabled and request.grpc_server_address and not dmesg_path:
                dmesg_path = get_source_cycle_log_file(
                    self.health_log_prefix,
                    "dmesg",
                    request.cycle,
                )
            if self.result_artifact_enabled and request.grpc_server_address and not result_path:
                result_path = get_source_cycle_log_file(
                    self.health_log_prefix,
                    "fact",
                    request.cycle,
                )
        if dmesg_path != request.dmesg_path or result_path != request.result_path:
            return replace(request, dmesg_path=dmesg_path, result_path=result_path)
        return request

    def process_cycle_failed(self, request: FactAgentRequest) -> None:
        try:
            store = self._store_factory(request)
        except Exception as exc:
            logger.warning("FACT agent failed to connect to TCPStore: %s", exc)
            return
        keys = FactAgentKeys(request.run_id, request.cycle)
        if request.is_store_host:
            self._process_store_host(request, store, keys)
        else:
            self._submit_local_evidence(request, store, keys)

    def _process_store_host(
        self, request: FactAgentRequest, store: Any, keys: FactAgentKeys
    ) -> None:
        expected_nodes = list(dict.fromkeys(request.expected_nodes or (request.local_node,)))
        nranks = request.ranks_per_node * max(1, len(expected_nodes))
        end_time = datetime.now(timezone.utc)
        workload_start_time = request.cycle_start_time or end_time - timedelta(
            seconds=max(0.0, self.dmesg_window_s)
        )
        try:
            service = self._fact_client_factory()
            attributor_id = service.create_failure_attributor(
                job_id=request.job_id or request.run_id,
                cycle_index=request.cycle,
                nodes=expected_nodes,
                ranks_per_node=request.ranks_per_node,
                nranks=nranks,
                start_time=workload_start_time,
                end_time=end_time,
                username=self.username,
                cluster=self.cluster,
            )
            store.set(keys.attributor_id, str(attributor_id).encode("utf-8"))
        except Exception as exc:
            logger.warning("FACT agent failed to create FACT attributor: %s", exc)
            self._store_avoid_decision(AvoidDecision(cycle_id=request.cycle, status="skipped"))
            self._publish_attributor_failure(store, keys, exc)
            self._write_result_record(
                request,
                {
                    "record_type": "fact_result",
                    "status": "failed",
                    "phase": "create_attributor",
                    "run_id": request.run_id,
                    "cycle": request.cycle,
                    "job_id": request.job_id or request.run_id,
                    "error": str(exc),
                },
            )
            return

        self._submit_local_evidence(request, store, keys)
        completed_count = self._wait_for_completion_count(store, keys, len(expected_nodes))
        try:
            result = service.get_attribution_result(
                attributor_id=str(attributor_id),
                observation_ids=[],
            )
        except Exception as exc:
            logger.warning("FACT agent attribution GET failed: %s", exc)
            self._store_avoid_decision(AvoidDecision(cycle_id=request.cycle, status="skipped"))
            self._write_result_record(
                request,
                {
                    "record_type": "fact_result",
                    "status": "failed",
                    "phase": "get_attribution",
                    "run_id": request.run_id,
                    "cycle": request.cycle,
                    "job_id": request.job_id or request.run_id,
                    "attributor_id": str(attributor_id),
                    "completed_node_count": completed_count,
                    "expected_node_count": len(expected_nodes),
                    "error": str(exc),
                },
            )
            return
        avoid_decision = self._compute_avoid_decision(request, result)
        self._write_result_record(
            request,
            {
                "record_type": "fact_result",
                "status": "complete",
                "run_id": request.run_id,
                "cycle": request.cycle,
                "job_id": request.job_id or request.run_id,
                "expected_node_count": len(expected_nodes),
                "completed_node_count": completed_count,
                "avoid_nodes": list(avoid_decision.avoid_nodes),
                **self._result_payload(result),
            },
        )
        logger.info(
            "FACT attribution completed for run_id=%s cycle=%s "
            "completed_nodes=%s expected_nodes=%s faulty_nodes=%s",
            request.run_id,
            request.cycle,
            completed_count,
            len(expected_nodes),
            result.faulty_nodes,
        )

    def _compute_avoid_decision(
        self,
        request: FactAgentRequest,
        result: FactAttributionResult,
    ) -> AvoidDecision:
        cluster = self.cluster or "unknown"
        job_id = request.job_id or request.run_id
        current_suspects = sorted({str(node) for node in result.faulty_nodes if str(node)})
        cycle_end_time = request.cycle_end_time or datetime.now(timezone.utc)

        if not current_suspects:
            decision = AvoidDecision(cycle_id=request.cycle, status="skipped")
            self._store_avoid_decision(decision)
            return decision

        if len(current_suspects) > self.fact_history_max_candidate_nodes:
            decision = AvoidDecision(cycle_id=request.cycle, status="skipped")
            self._store_avoid_decision(decision)
            return decision

        history_records = []
        history_end_time = request.cycle_start_time or cycle_end_time
        if self.fact_history_es_url and self.fact_history_es_auth_file:
            try:
                lookback = parse_duration(
                    self.fact_history_lookback,
                    default=timedelta(days=14),
                )
                history_records = self._fact_history_client_factory().query_node_history(
                    cluster=cluster,
                    nodes=current_suspects,
                    start_time=history_end_time - lookback,
                    end_time=history_end_time,
                )
            except Exception as exc:
                logger.warning(
                    "FACT history query failed; no avoid_nodes for cycle %s: %s",
                    request.cycle,
                    exc,
                )
                decision = AvoidDecision(cycle_id=request.cycle, status="skipped")
                self._store_avoid_decision(decision)
                self._hot_cache.add_current_cycle(
                    cluster=cluster,
                    nodes=current_suspects,
                    job_id=job_id,
                    cycle_id=request.cycle,
                    event_time=cycle_end_time,
                )
                return decision

        hot_records = self._hot_cache.records_for(
            cluster=cluster,
            nodes=current_suspects,
            before=cycle_end_time,
        )
        decision = compute_repeat_offender_decision(
            cycle_id=request.cycle,
            current_suspect_nodes=current_suspects,
            history_records=history_records,
            hot_cache_records=hot_records,
            max_candidate_nodes=self.fact_history_max_candidate_nodes,
            min_repeat_count_for_avoid=self.fact_min_repeat_count_for_avoid,
            max_avoids_per_cycle=self.fact_max_attribution_avoids_per_cycle,
        )
        self._store_avoid_decision(decision)
        self._hot_cache.add_current_cycle(
            cluster=cluster,
            nodes=current_suspects,
            job_id=job_id,
            cycle_id=request.cycle,
            event_time=cycle_end_time,
        )
        logger.info(
            "FACT avoid decision for run_id=%s cycle=%s status=%s avoid_nodes=%s",
            request.run_id,
            request.cycle,
            decision.status,
            decision.avoid_nodes,
        )
        return decision

    def _store_avoid_decision(self, decision: AvoidDecision) -> None:
        with self._avoid_decisions_lock:
            self._avoid_decisions[decision.cycle_id] = decision

    def _submit_local_evidence(
        self, request: FactAgentRequest, store: Any, keys: FactAgentKeys
    ) -> None:
        operation_deadline = time.monotonic() + max(0.0, self.observation_deadline_s)
        node = request.local_node
        status: dict[str, Any] = {
            "record_type": "fact_observation",
            "run_id": request.run_id,
            "cycle": request.cycle,
            "job_id": request.job_id or request.run_id,
            "node": node,
            "source": "dmesg",
            "status": "skipped",
            "attributor_id": None,
            "observation_id": None,
            "lines_collected": 0,
            "bytes_collected": 0,
            "dmesg_path": "",
            "dmesg_write_error": "",
            "error": "",
        }

        try:
            dmesg_text = self._dmesg_collector(self.dmesg_window_s, node)
            collection_end_time = datetime.now(timezone.utc)
            status["lines_collected"] = len(dmesg_text.splitlines())
            status["bytes_collected"] = len(dmesg_text.encode("utf-8", errors="replace"))
        except Exception as exc:
            logger.warning("FACT agent failed to collect dmesg on %s: %s", node, exc)
            status.update(status="collect_failed", error=str(exc))
            self._write_result_record(request, status)
            self._write_terminal_completion(store, keys, node)
            return

        if request.dmesg_path and dmesg_text:
            try:
                self._write_dmesg_artifact(request, request.dmesg_path, dmesg_text)
                status["dmesg_path"] = request.dmesg_path
            except Exception as exc:
                logger.warning(
                    "FACT agent failed to write dmesg evidence %s: %s",
                    request.dmesg_path,
                    exc,
                )
                status["dmesg_write_error"] = str(exc)

        try:
            attributor_wait_s = request.store_timeout_s
            if self.observation_deadline_s > 0:
                remaining_s = max(0.001, operation_deadline - time.monotonic())
                attributor_wait_s = min(attributor_wait_s, remaining_s)
            raw_attributor_id = self._store_get_bytes_with_deadline(
                store, keys.attributor_id, attributor_wait_s
            )
            if not raw_attributor_id:
                raise RuntimeError("timed out waiting for attributor_id")
            attributor_id = raw_attributor_id.decode("utf-8")
            if attributor_id.startswith(_ATTRIBUTOR_FAILURE_PREFIX):
                error = attributor_id[len(_ATTRIBUTOR_FAILURE_PREFIX) :]
                raise RuntimeError(f"FACT attributor creation failed on store host: {error}")
            status["attributor_id"] = attributor_id
        except Exception as exc:
            logger.warning("FACT agent could not read attributor_id on %s: %s", node, exc)
            status.update(status="attributor_failed", error=str(exc))
            self._write_result_record(request, status)
            self._write_terminal_completion(store, keys, node)
            return

        end_time = collection_end_time
        start_time = end_time - timedelta(seconds=max(0.0, self.dmesg_window_s))
        try:
            observation_id = self._submit_dmesg_observation_with_retries(
                attributor_id=attributor_id,
                dmesg_text=dmesg_text,
                start_time=start_time,
                end_time=end_time,
                default_hostname=node,
                deadline_s=operation_deadline,
            )
            if observation_id is None:
                status["status"] = "empty"
            else:
                status.update(status="submitted", observation_id=observation_id)
        except Exception as exc:
            logger.warning("FACT agent failed to submit dmesg observation for %s: %s", node, exc)
            status.update(status="post_failed", error=str(exc))
        self._write_result_record(request, status)
        self._write_terminal_completion(store, keys, node)

    def _submit_dmesg_observation_with_retries(
        self,
        *,
        attributor_id: str,
        dmesg_text: str,
        start_time: datetime,
        end_time: datetime,
        default_hostname: str,
        deadline_s: float,
    ) -> Any:
        delay_s = _POST_RETRY_INITIAL_DELAY_S
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._fact_client_factory().submit_dmesg_text_observation(
                    attributor_id=attributor_id,
                    dmesg_text=dmesg_text,
                    start_time=start_time,
                    end_time=end_time,
                    default_hostname=default_hostname,
                )
            except Exception:
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= _POST_RETRY_MIN_REMAINING_S:
                    raise
                sleep_s = min(
                    delay_s * random.uniform(0.5, 1.5),
                    max(0.0, remaining_s - _POST_RETRY_MIN_REMAINING_S),
                )
                if sleep_s <= 0:
                    raise
                logger.info(
                    "FACT agent observation POST attempt %s failed for %s; " "retrying in %.2fs",
                    attempt,
                    default_hostname,
                    sleep_s,
                    exc_info=True,
                )
                time.sleep(sleep_s)
                delay_s = min(_POST_RETRY_MAX_DELAY_S, delay_s * 2.0)

    def _write_terminal_completion(
        self,
        store: Any,
        keys: FactAgentKeys,
        node: str,
    ) -> None:
        try:
            store.add(keys.done_count, 1)
        except Exception as exc:
            logger.warning("FACT agent failed to publish completion count for %s: %s", node, exc)

    def _publish_attributor_failure(
        self,
        store: Any,
        keys: FactAgentKeys,
        exc: Exception,
    ) -> None:
        try:
            store.set(keys.attributor_id, f"{_ATTRIBUTOR_FAILURE_PREFIX}{exc}".encode("utf-8"))
        except Exception as store_exc:
            logger.warning(
                "FACT agent failed to publish attributor failure sentinel: %s", store_exc
            )

    def _wait_for_completion_count(
        self,
        store: Any,
        keys: FactAgentKeys,
        expected_node_count: int,
    ) -> int:
        deadline = time.monotonic() + max(0.0, self.observation_deadline_s)
        completed_count = 0
        while time.monotonic() < deadline and completed_count < expected_node_count:
            try:
                completed_count = int(store.add(keys.done_count, 0))
            except Exception:
                pass
            if completed_count >= expected_node_count:
                break
            time.sleep(0.1)
        return min(completed_count, expected_node_count)

    @staticmethod
    def _store_get_bytes_with_deadline(store: Any, key: str, timeout_s: float) -> Optional[bytes]:
        timeout_s = max(0.0, timeout_s)
        wait_fn = getattr(store, "wait", None)
        if callable(wait_fn):
            try:
                wait_fn([key], timedelta(seconds=max(0.001, timeout_s)))
                return store.get(key)
            except TypeError:
                pass
            except Exception:
                return None

        deadline = time.monotonic() + max(0.0, timeout_s)
        sleep_s = random.uniform(0.05, 0.15)
        while True:
            try:
                if store.check([key]):
                    return store.get(key)
            except Exception:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(sleep_s, remaining))
            sleep_s = min(1.0, sleep_s * random.uniform(1.25, 1.75))

    def _write_dmesg_artifact(self, request: FactAgentRequest, path: str, payload: str) -> None:
        self._append_text_artifact(request, path, payload)

    def _append_text_artifact(self, request: FactAgentRequest, path: str, payload: str) -> None:
        if not request.grpc_server_address:
            raise RuntimeError("FACT artifact requires gRPC log aggregation")
        self._enqueue_grpc_artifact(
            request.grpc_server_address,
            request.grpc_node_id or request.local_node,
            path,
            self._ensure_trailing_newline(payload),
        )

    def _write_result_record(self, request: FactAgentRequest, payload: dict[str, Any]) -> None:
        if not request.result_path:
            return
        if not request.grpc_server_address:
            logger.warning(
                "FACT artifact requires gRPC log aggregation; skipping %s",
                request.result_path,
            )
            return
        try:
            text = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
            self._append_text_artifact(request, request.result_path, text)
        except Exception as exc:
            logger.warning(
                "FACT agent failed to write result artifact %s: %s", request.result_path, exc
            )

    def _enqueue_grpc_artifact(
        self,
        grpc_server_address: str,
        node_id: str,
        path: str,
        payload: str,
    ) -> None:
        writer_queue, writer = self._get_grpc_writer(grpc_server_address, node_id)
        writer_queue.put((path, payload))
        self._wait_for_grpc_writer_queue(writer_queue, writer)

    def _get_grpc_writer(
        self,
        grpc_server_address: str,
        node_id: str,
    ) -> tuple[queue.Queue, threading.Thread]:
        key = (grpc_server_address, node_id)
        with self._grpc_writers_lock:
            existing = self._grpc_writers.get(key)
            if existing is not None:
                return existing

            write_queue: queue.Queue = queue.Queue()
            writer = self._grpc_writer_factory(write_queue, grpc_server_address, node_id, logger)
            writer.start()
            self._grpc_writers[key] = (write_queue, writer)
            return write_queue, writer

    @staticmethod
    def _ensure_trailing_newline(payload: str) -> str:
        if payload and not payload.endswith("\n"):
            return payload + "\n"
        return payload

    @staticmethod
    def _result_payload(result: FactAttributionResult) -> dict[str, Any]:
        return {"fact_attribution_result": asdict(result)}

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._grpc_writers_lock:
            writers = list(self._grpc_writers.values())
            self._grpc_writers.clear()
        self._wait_for_grpc_writer_queues(writers)
        for _, writer in writers:
            shutdown = getattr(writer, "shutdown", None)
            if callable(shutdown):
                with contextlib.suppress(Exception):
                    shutdown()
        for _, writer in writers:
            with contextlib.suppress(Exception):
                writer.join(timeout=5.0)

    @staticmethod
    def _wait_for_grpc_writer_queues(
        writers: list[tuple[queue.Queue, threading.Thread]],
    ) -> None:
        deadline = time.monotonic() + _GRPC_RESULT_DRAIN_TIMEOUT_S
        for write_queue, writer in writers:
            FactAgent._wait_for_grpc_writer_queue(write_queue, writer, deadline=deadline)

    @staticmethod
    def _wait_for_grpc_writer_queue(
        write_queue: queue.Queue,
        writer: threading.Thread,
        *,
        deadline: Optional[float] = None,
    ) -> None:
        is_alive = getattr(writer, "is_alive", None)
        if not callable(is_alive):
            return
        deadline = deadline or (time.monotonic() + _GRPC_RESULT_DRAIN_TIMEOUT_S)
        while is_alive() and not write_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        if is_alive() and not write_queue.empty():
            logger.warning(
                "FACT result artifact gRPC queue still has %s records pending",
                write_queue.qsize(),
            )

    def serve_forever(self, *, max_rpc_bytes: int = DEFAULT_MAX_RPC_BYTES) -> None:
        socket_path = Path(self.socket_path)
        if socket_path.exists():
            socket_path.unlink()
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            try:
                server.bind(str(socket_path))
                os.chmod(socket_path, 0o600)
                server.listen(128)
                server.settimeout(1.0)
                logger.info("nvrx-fact-agent listening on %s", socket_path)
                while not self._stop_event.is_set():
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn, max_rpc_bytes),
                        daemon=True,
                    ).start()
            finally:
                self.stop()
                with contextlib.suppress(FileNotFoundError):
                    socket_path.unlink()

    def _handle_connection(self, conn: socket.socket, max_rpc_bytes: int) -> None:
        with conn:
            try:
                payload = recv_frame(conn, max_bytes=max_rpc_bytes)
                ack = self.handle_payload(payload)
                send_frame(conn, ack)
            except Exception as exc:
                logger.warning("nvrx-fact-agent rejected RPC: %s", exc)
                with contextlib.suppress(Exception):
                    send_frame(conn, {"accepted": False, "error": str(exc)})


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nvrx-fact-agent")
    parser.add_argument(
        "--fact-url", required=True, help="FACT API URL, e.g. http://host:8001/latest"
    )
    parser.add_argument("--socket-path", default=default_socket_path())
    parser.add_argument("--dmesg-window", type=float, default=DEFAULT_DMESG_WINDOW_S)
    parser.add_argument(
        "--observation-deadline", type=float, default=DEFAULT_OBSERVATION_DEADLINE_S
    )
    parser.add_argument("--fact-timeout", type=float, default=60.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--rdzv-endpoint", default=None)
    parser.add_argument("--store-timeout", type=float, default=DEFAULT_STORE_TIMEOUT_S)
    parser.add_argument("--local-node", default=None)
    parser.add_argument("--is-store-host", action="store_true")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--ranks-per-node", type=int, default=1)
    parser.add_argument("--username", default=None)
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--health-log-prefix", default=None)
    parser.add_argument("--dmesg-artifact-enabled", action="store_true")
    parser.add_argument("--result-artifact-enabled", action="store_true")
    parser.add_argument("--grpc-server-address", default=None)
    parser.add_argument("--grpc-node-id", default=None)
    parser.add_argument("--fact-history-es-url", default=None)
    parser.add_argument("--fact-history-es-auth-file", default=None)
    parser.add_argument("--fact-history-lookback", default=DEFAULT_FACT_HISTORY_LOOKBACK)
    parser.add_argument("--fact-history-index", default=None)
    parser.add_argument(
        "--fact-history-max-candidate-nodes",
        type=int,
        default=DEFAULT_FACT_HISTORY_MAX_CANDIDATE_NODES,
    )
    parser.add_argument(
        "--fact-history-query-timeout",
        type=float,
        default=DEFAULT_FACT_HISTORY_QUERY_TIMEOUT_S,
    )
    parser.add_argument(
        "--fact-min-repeat-count-for-avoid",
        type=int,
        default=DEFAULT_FACT_MIN_REPEAT_COUNT_FOR_AVOID,
    )
    parser.add_argument(
        "--fact-max-attribution-avoids-per-cycle",
        type=int,
        default=DEFAULT_FACT_MAX_ATTRIBUTION_AVOIDS_PER_CYCLE,
    )
    parser.add_argument("--max-rpc-bytes", type=int, default=DEFAULT_MAX_RPC_BYTES)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = get_arg_parser().parse_args(argv)
    setup_logger(node_local_tmp_prefix="nvrxfactagent")
    service = FactAgent(
        fact_url=args.fact_url,
        socket_path=args.socket_path,
        dmesg_window_s=args.dmesg_window,
        observation_deadline_s=args.observation_deadline,
        fact_timeout_s=args.fact_timeout,
        run_id=args.run_id,
        rdzv_endpoint=args.rdzv_endpoint,
        store_timeout_s=args.store_timeout,
        local_node=args.local_node,
        is_store_host=args.is_store_host,
        job_id=args.job_id,
        ranks_per_node=args.ranks_per_node,
        username=args.username,
        cluster=args.cluster,
        health_log_prefix=args.health_log_prefix,
        dmesg_artifact_enabled=args.dmesg_artifact_enabled,
        result_artifact_enabled=args.result_artifact_enabled,
        grpc_server_address=args.grpc_server_address,
        grpc_node_id=args.grpc_node_id,
        fact_history_es_url=args.fact_history_es_url,
        fact_history_es_auth_file=args.fact_history_es_auth_file,
        fact_history_lookback=args.fact_history_lookback,
        fact_history_index=args.fact_history_index,
        fact_history_max_candidate_nodes=args.fact_history_max_candidate_nodes,
        fact_history_query_timeout_s=args.fact_history_query_timeout,
        fact_min_repeat_count_for_avoid=args.fact_min_repeat_count_for_avoid,
        fact_max_attribution_avoids_per_cycle=args.fact_max_attribution_avoids_per_cycle,
    )

    def _handle_stop_signal(signum: int, _frame: Any) -> None:
        logger.info("nvrx-fact-agent received signal %s; requesting shutdown", signum)
        service.request_stop()

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    service.serve_forever(max_rpc_bytes=args.max_rpc_bytes)


if __name__ == "__main__":
    main()
